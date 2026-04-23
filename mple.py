"""Fit the latent-only conditional MPLE model for synthetic and real-data experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from scipy.optimize import OptimizeResult, minimize

from pymanopt import Problem, function
from pymanopt.manifolds import Euclidean, FixedRankEmbedded, Product
from pymanopt.optimizers import ConjugateGradient

from model_utils import (
    ModelArtifacts,
    OPTIMIZER_MODE_ALTERNATIVE_LOW_RANK,
    OPTIMIZER_MODE_MANIFOLD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    build_fit_model_artifacts,
    compose_field_matrix_from_theta,
    compose_interaction_matrix,
    compose_latent_field_matrix,
    free_scalar_parameter_names,
    interaction_effect,
    latent_field_bound_norm,
    load_model_artifacts,
    pack_theta,
    parameter_names,
    save_field_artifacts,
    scalar_parameter_names,
    summarize_theta_for_logging,
    unpack_theta,
    uses_full_matrix_parameterization,
    validate_fixed_scalar_params,
    with_theta_field,
)
from posterior_predictive_utils import _io_path, save_estimated_parameter_bundle


def load_yaml_config(path: str | Path):
    return OmegaConf.load(Path(path))


def first_existing_path(*paths: str | Path) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of the expected paths: "
        + ", ".join(str(Path(path)) for path in paths)
    )


def load_gamma_matrix(data_folder: str | Path):
    data_path = Path(data_folder)
    gamma_sparse = data_path / "gamma_matrix_sparse.npz"
    gamma_dense = data_path / "gamma_matrix.npy"
    if gamma_sparse.exists():
        return sparse.load_npz(gamma_sparse).tocsr()
    if gamma_dense.exists():
        return np.load(gamma_dense, allow_pickle=False)
    raise FileNotFoundError(f"Missing gamma matrix artifact in {data_path}.")


def _canonicalize_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    return pack_theta(
        theta_parts,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_panel_artifact(panel_path: str | Path):
    panel_path = Path(panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Could not find panel data artifact at {panel_path}.")
    return np.load(panel_path)


def pseudo_nll(
    x: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    if x.shape[0] != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )

    prev_x = np.vstack([x_0, x[:-1, :]])
    beta_feature = np.asarray(z, dtype=float)
    field_matrix = compose_field_matrix_from_theta(theta_parts, artifacts)
    h_x = (
        field_matrix
        + theta_parts["beta"] * beta_feature
        + theta_parts["eta"] * prev_x
        + theta_parts["xi"] * interaction_effect_x
    )
    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x
    gradient_denominator = x.size
    total_loss = loss_x.sum() / gradient_denominator

    if uses_full_matrix_parameterization(artifacts):
        field_grad = res_x.reshape(-1)
    else:
        node_grad = res_x.T @ theta_parts["time_factors"]
        time_grad = res_x @ theta_parts["node_factors"]
        field_grad = np.concatenate([node_grad.reshape(-1), time_grad.reshape(-1)])

    scalar_grad_lookup = {
        "beta": float((res_x * beta_feature).sum()) / gradient_denominator,
        "xi": float((res_x * interaction_effect_x).sum()) / gradient_denominator,
        "eta": float((res_x * prev_x).sum()) / gradient_denominator,
    }
    grad_parts = [field_grad / gradient_denominator]
    for name in free_scalar_parameter_names(fixed_scalar_params):
        grad_parts.append(np.array([scalar_grad_lookup[name]], dtype=float))
    return float(total_loss), np.concatenate(grad_parts)


def _pseudo_loss(
    x: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None,
) -> float:
    loss, _ = pseudo_nll(
        x,
        z,
        theta,
        x_0,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fixed_scalar_params=fixed_scalar_params,
    )
    return float(loss)


def _nuclear_norm(field_matrix: np.ndarray) -> float:
    if field_matrix.size == 0:
        return 0.0
    return float(
        np.linalg.svd(np.asarray(field_matrix, dtype=float), compute_uv=False).sum()
    )


def _nuclear_norm_normalizer(artifacts: ModelArtifacts) -> float:
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    size = int(artifacts.t_steps) * n_nodes
    if size <= 0:
        return 1.0
    return float(np.sqrt(size))


def _frobenius_norm_normalizer(artifacts: ModelArtifacts) -> float:
    return _nuclear_norm_normalizer(artifacts)


def _singular_value_threshold(field_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if field_matrix.size == 0 or threshold <= 0.0:
        return np.asarray(field_matrix, dtype=float)
    u, singular_values, vt = np.linalg.svd(
        np.asarray(field_matrix, dtype=float), full_matrices=False
    )
    shrunk = np.maximum(singular_values - float(threshold), 0.0)
    if not np.any(shrunk):
        return np.zeros_like(field_matrix, dtype=float)
    return (u * shrunk) @ vt


def _prox_nuclear_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None,
    threshold: float,
) -> np.ndarray:
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    theta_parts["field_matrix"] = _singular_value_threshold(
        np.asarray(theta_parts["field_matrix"], dtype=float),
        threshold,
    )
    return _canonicalize_theta(
        pack_theta(
            theta_parts,
            artifacts,
            fixed_scalar_params=fixed_scalar_params,
        ),
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )


def _penalized_nuclear_objective(
    theta: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None,
    lambda_nuclear: float,
) -> tuple[float, float, float, float]:
    smooth_loss = _pseudo_loss(
        x,
        z,
        theta,
        x_0,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fixed_scalar_params=fixed_scalar_params,
    )
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    nuclear_norm = _nuclear_norm(np.asarray(theta_parts["field_matrix"], dtype=float))
    normalized_nuclear_norm = nuclear_norm / _nuclear_norm_normalizer(artifacts)
    return (
        float(smooth_loss + float(lambda_nuclear) * normalized_nuclear_norm),
        float(smooth_loss),
        float(nuclear_norm),
        float(normalized_nuclear_norm),
    )


def _fit_mple_nuclear_norm(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    param_names: list[str],
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    seed: int,
    verbose_every: int,
    tol: float,
    logger,
    theta_init,
    fixed_scalar_params: dict[str, float] | None,
    lambda_nuclear: float,
    proximal_lr: float = 1.0,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    if lambda_nuclear < 0.0:
        raise ValueError("lambda_nuclear must be nonnegative.")
    if proximal_lr <= 0.0:
        raise ValueError("proximal_lr must be positive.")

    rng = np.random.default_rng(seed)
    raw_init = (
        rng.normal(0, 0.1, size=len(param_names))
        if theta_init is None
        else np.asarray(theta_init, dtype=float)
    )
    theta = _canonicalize_theta(
        raw_init,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    y = theta.copy()
    momentum = 1.0
    history: list[float] = []
    penalized_history: list[float] = []
    converged = False
    previous_objective = np.inf
    nuclear_normalizer = _nuclear_norm_normalizer(artifacts)
    initial_penalized_obj, initial_smooth_loss, _, _ = _penalized_nuclear_objective(
        theta,
        x,
        z,
        x_0,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fixed_scalar_params=fixed_scalar_params,
        lambda_nuclear=lambda_nuclear,
    )

    for iteration in range(max(1, int(steps))):
        loss_y, grad_y = pseudo_nll(
            x,
            z,
            y,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed_scalar_params,
        )
        stepped = y - float(proximal_lr) * grad_y
        candidate = _prox_nuclear_theta(
            stepped,
            artifacts,
            fixed_scalar_params,
            threshold=float(proximal_lr) * float(lambda_nuclear) / nuclear_normalizer,
        )
        (
            penalized_obj,
            smooth_loss,
            nuclear_norm,
            normalized_nuclear_norm,
        ) = _penalized_nuclear_objective(
            candidate,
            x,
            z,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed_scalar_params,
            lambda_nuclear=lambda_nuclear,
        )
        history.append(smooth_loss)
        penalized_history.append(penalized_obj)
        if verbose_every and iteration % verbose_every == 0:
            message = summarize_theta_for_logging(param_names, candidate)
            if logger is None:
                print(
                    f"Nuclear prox iter {iteration} | Loss: {smooth_loss:.6f} "
                    f"| Penalized: {penalized_obj:.6f}"
                )
                print(message)
            else:
                logger.info(
                    "Nuclear prox iter %s | Loss: %.6f | Penalized: %.6f | nuclear_norm: %.6f | normalized_nuclear_norm: %.6f",
                    iteration,
                    smooth_loss,
                    penalized_obj,
                    nuclear_norm,
                    normalized_nuclear_norm,
                )
                logger.info(message)

        if np.isfinite(previous_objective):
            improvement = abs(previous_objective - penalized_obj)
            if improvement <= float(tol) * max(1.0, abs(previous_objective)):
                converged = True
                theta = candidate
                break
        previous_objective = penalized_obj

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - theta)
        y = _canonicalize_theta(
            y,
            artifacts,
            fixed_scalar_params=fixed_scalar_params,
        )
        theta = candidate
        momentum = next_momentum

    theta = _canonicalize_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    (
        penalized_obj,
        smooth_loss,
        nuclear_norm,
        normalized_nuclear_norm,
    ) = _penalized_nuclear_objective(
        theta,
        x,
        z,
        x_0,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fixed_scalar_params=fixed_scalar_params,
        lambda_nuclear=lambda_nuclear,
    )
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    field_matrix = np.asarray(theta_parts["field_matrix"], dtype=float)
    result = OptimizeResult(
        x=theta,
        success=bool(converged),
        message=(
            "CONVERGED: proximal objective tolerance reached"
            if converged
            else "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        ),
        nit=len(history),
        nfev=len(history),
        iterations=int(len(history)),
        cost_evaluations=int(len(history)),
        optimizer_mode=OPTIMIZER_MODE_NUCLEAR_NORM,
        optimizer="nuclear_proximal_gradient",
        lambda_nuclear=float(lambda_nuclear),
        final_penalized_objective=float(penalized_obj),
        final_mple_loss=float(smooth_loss),
        nuclear_norm=float(nuclear_norm),
        normalized_nuclear_norm=float(normalized_nuclear_norm),
        nuclear_norm_normalizer=float(nuclear_normalizer),
        effective_rank=float(np.linalg.matrix_rank(field_matrix)),
        proximal_iterations=int(len(history)),
        mple_history=list(history),
        penalized_history=list(penalized_history),
        best_start=0,
        n_starts=1,
        start_summaries=[
            {
                "start_index": 0,
                "seed": int(seed),
                "initialization_kind": "random",
                "initial_mple_loss": float(initial_smooth_loss),
                "initial_penalized_objective": float(initial_penalized_obj),
                "final_mple_loss": float(smooth_loss),
                "final_penalized_objective": float(penalized_obj),
                "iterations": int(len(history)),
                "cost_evaluations": int(len(history)),
                "success": bool(converged),
                "message": (
                    "CONVERGED: proximal objective tolerance reached"
                    if converged
                    else "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
                ),
            }
        ],
    )
    if not history:
        history.append(float(smooth_loss))
    return theta, history, result


def _fixed_rank_field_matrix(
    u: np.ndarray, singular_values: np.ndarray, vt: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(u, dtype=float) * np.asarray(singular_values, dtype=float)
    ) @ np.asarray(vt, dtype=float)


def _fixed_rank_point_from_field(field_matrix: np.ndarray, rank: int, point_type):
    u, singular_values, vt = np.linalg.svd(
        np.asarray(field_matrix, dtype=float), full_matrices=False
    )
    rank = int(rank)
    padded = np.zeros(rank, dtype=float)
    padded[: min(rank, singular_values.size)] = singular_values[:rank]
    scale = max(1.0, float(np.max(np.abs(padded))) if padded.size else 1.0)
    jitter = np.finfo(float).eps * scale * np.arange(rank, 0, -1, dtype=float)
    singular_values = np.maximum(padded, 0.0) + jitter
    return point_type(u[:, :rank], singular_values, vt[:rank, :])


def _random_fixed_rank_point(
    rng: np.random.Generator,
    t_steps: int,
    n_nodes: int,
    rank: int,
    point_type,
):
    u, _ = np.linalg.qr(rng.normal(size=(int(t_steps), int(rank))))
    v, _ = np.linalg.qr(rng.normal(size=(int(n_nodes), int(rank))))
    singular_values = np.sort(
        np.maximum(np.abs(rng.normal(loc=0.1, scale=0.05, size=int(rank))), 1.0e-3)
    )[::-1]
    singular_values += (
        np.finfo(float).eps
        * max(1.0, float(np.max(singular_values)))
        * np.arange(rank, 0, -1, dtype=float)
    )
    return point_type(u[:, :rank], singular_values, v[:, :rank].T)


def _low_rank_manifold_point_to_theta(
    point,
    artifacts: ModelArtifacts,
    free_scalar_names: list[str],
    fixed_scalar_params: dict[str, float],
) -> np.ndarray:
    if free_scalar_names:
        field_point = point[0]
        free_scalar_values = np.asarray(point[1], dtype=float).reshape(-1)
    else:
        field_point = point
        free_scalar_values = np.zeros(0, dtype=float)
    u, singular_values, vt = field_point
    theta_parts: dict[str, Any] = {
        "node_factors": np.asarray(vt, dtype=float).T,
        "time_factors": np.asarray(u, dtype=float)
        * np.asarray(singular_values, dtype=float),
    }
    scalar_by_name = dict(fixed_scalar_params)
    scalar_by_name.update(
        {
            name: float(value)
            for name, value in zip(free_scalar_names, free_scalar_values)
        }
    )
    for name in scalar_parameter_names():
        theta_parts[name] = float(scalar_by_name[name])
    return pack_theta(
        theta_parts,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )


def _fit_zero_rank_unconstrained(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    seed: int,
    tol: float,
    theta_init,
    fixed_scalar_params: dict[str, float] | None,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    free_names = free_scalar_parameter_names(fixed)
    rng = np.random.default_rng(seed)
    if theta_init is None:
        initial = rng.normal(0.0, 0.1, size=len(free_names))
    else:
        initial = np.asarray(theta_init, dtype=float)
    initial = _canonicalize_theta(
        initial,
        artifacts,
        fixed_scalar_params=fixed,
    )
    history: list[float] = []

    def objective(theta):
        loss, grad = pseudo_nll(
            x,
            z,
            theta,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed,
        )
        history.append(float(loss))
        return loss, grad

    if initial.size == 0:
        final_loss = _pseudo_loss(
            x,
            z,
            initial,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed,
        )
        result = OptimizeResult(
            x=initial,
            success=True,
            message="CONVERGENCE: no free parameters",
            nit=0,
            nfev=1,
        )
        return initial, [float(final_loss)], result

    result = minimize(
        objective,
        initial,
        method="BFGS",
        jac=True,
        options={"maxiter": int(steps), "gtol": float(tol)},
    )
    theta_hat = _canonicalize_theta(
        result.x,
        artifacts,
        fixed_scalar_params=fixed,
    )
    final_loss = _pseudo_loss(
        x,
        z,
        theta_hat,
        x_0,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fixed_scalar_params=fixed,
    )
    if not history or history[-1] != final_loss:
        history.append(float(final_loss))
    result.x = theta_hat
    return theta_hat, history, result


def _fit_mple_low_rank_manifold(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    seed: int,
    verbose_every: int,
    tol: float,
    logger,
    theta_init,
    fixed_scalar_params: dict[str, float] | None,
    n_starts: int,
    lambda_frobenius: float,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    if lambda_frobenius < 0.0:
        raise ValueError("lambda_frobenius must be nonnegative.")
    if artifacts.latent_rank == 0:
        theta_hat, history, result = _fit_zero_rank_unconstrained(
            x,
            z,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            tol=tol,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
        )
        result["optimizer_mode"] = OPTIMIZER_MODE_MANIFOLD
        result["optimizer"] = "scipy_bfgs_zero_rank"
        result["best_start"] = 0
        result["n_starts"] = 1
        result["iterations"] = int(getattr(result, "nit", 0))
        result["cost_evaluations"] = int(getattr(result, "nfev", len(history)))
        result["lambda_frobenius"] = float(lambda_frobenius)
        result["final_mple_loss"] = float(history[-1])
        result["final_penalized_objective"] = float(history[-1])
        result["frobenius_norm"] = 0.0
        result["normalized_frobenius_norm"] = 0.0
        result["frobenius_norm_normalizer"] = float(
            _frobenius_norm_normalizer(artifacts)
        )
        result["effective_rank"] = 0.0
        result["mple_history"] = list(history)
        result["penalized_history"] = list(history)
        result["start_summaries"] = [
            {
                "start_index": 0,
                "seed": int(seed),
                "initialization_kind": "random",
                "initial_mple_loss": float(history[0]),
                "initial_penalized_objective": float(history[0]),
                "final_mple_loss": float(history[-1]),
                "final_penalized_objective": float(history[-1]),
                "iterations": int(getattr(result, "nit", 0)),
                "cost_evaluations": int(getattr(result, "nfev", len(history))),
                "success": bool(getattr(result, "success", False)),
                "message": str(getattr(result, "message", "")),
            }
        ]
        return theta_hat, history, result

    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    free_names = free_scalar_parameter_names(fixed)
    n_starts = max(1, int(n_starts))
    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    if rank < 1 or rank > min(t_steps, n_nodes):
        raise ValueError(
            f"latent_rank={rank} must lie in [1, {min(t_steps, n_nodes)}] for fixed-rank manifold optimization."
        )
    fixed_rank_manifold = FixedRankEmbedded(t_steps, n_nodes, rank)
    template_point = fixed_rank_manifold.random_point()
    point_type = type(template_point)
    if free_names:
        manifold = Product([fixed_rank_manifold, Euclidean(len(free_names))])
    else:
        manifold = fixed_rank_manifold

    prev_x = np.vstack([x_0, x[:-1, :]])
    beta_feature = np.asarray(z, dtype=float)
    outcome_size = float(np.asarray(x).size)
    frobenius_normalizer = _frobenius_norm_normalizer(artifacts)

    def scalar_values(free_scalar_values: np.ndarray) -> dict[str, float]:
        scalars = dict(fixed)
        scalars.update(
            {
                name: float(value)
                for name, value in zip(
                    free_names, np.asarray(free_scalar_values, dtype=float).reshape(-1)
                )
            }
        )
        return scalars

    def loss_and_grad(
        u: np.ndarray,
        singular_values: np.ndarray,
        vt: np.ndarray,
        free_scalar_values: np.ndarray,
    ):
        scalars = scalar_values(free_scalar_values)
        field_matrix = _fixed_rank_field_matrix(u, singular_values, vt)
        h_x = (
            field_matrix
            + scalars["beta"] * beta_feature
            + scalars["eta"] * prev_x
            + scalars["xi"] * interaction_effect_x
        )
        loss_x = np.logaddexp(h_x, -h_x) - x * h_x
        residual = np.tanh(h_x) - x
        field_gradient = residual / outcome_size
        grad_u = (field_gradient @ vt.T) * singular_values[np.newaxis, :]
        grad_s = np.diag(u.T @ field_gradient @ vt.T)
        grad_vt = singular_values[:, np.newaxis] * (u.T @ field_gradient)
        frobenius_norm = float(np.linalg.norm(singular_values))
        normalized_frobenius_norm = frobenius_norm / frobenius_normalizer
        if lambda_frobenius > 0.0 and frobenius_norm > 0.0:
            grad_s = grad_s + (
                float(lambda_frobenius)
                * singular_values
                / (frobenius_normalizer * frobenius_norm)
            )
        scalar_gradient_lookup = {
            "beta": float((residual * beta_feature).sum()) / outcome_size,
            "xi": float((residual * interaction_effect_x).sum()) / outcome_size,
            "eta": float((residual * prev_x).sum()) / outcome_size,
        }
        scalar_gradient = np.asarray(
            [scalar_gradient_lookup[name] for name in free_names], dtype=float
        )
        smooth_loss = float(loss_x.sum() / outcome_size)
        penalized_loss = (
            smooth_loss + float(lambda_frobenius) * normalized_frobenius_norm
        )
        return (
            float(penalized_loss),
            float(smooth_loss),
            float(frobenius_norm),
            float(normalized_frobenius_norm),
            grad_u,
            grad_s,
            grad_vt,
            scalar_gradient,
        )

    eval_count = [0]
    active_mple_history: list[float] = []
    active_penalized_history: list[float] = []

    if free_names:

        @function.numpy(manifold)
        def cost(u, singular_values, vt, free_scalar_values):
            loss, smooth_loss, *_ = loss_and_grad(
                u, singular_values, vt, free_scalar_values
            )
            active_mple_history.append(smooth_loss)
            active_penalized_history.append(loss)
            if verbose_every and eval_count[0] % verbose_every == 0:
                if logger is None:
                    print(
                        f"Pymanopt eval {eval_count[0]} | Loss: {smooth_loss:.6f} "
                        f"| Penalized: {loss:.6f}"
                    )
                else:
                    logger.info(
                        "Pymanopt eval %s | Loss: %.6f | Penalized: %.6f",
                        eval_count[0],
                        smooth_loss,
                        loss,
                    )
            eval_count[0] += 1
            return loss

        @function.numpy(manifold)
        def euclidean_gradient(u, singular_values, vt, free_scalar_values):
            _, _, _, _, grad_u, grad_s, grad_vt, scalar_gradient = loss_and_grad(
                u, singular_values, vt, free_scalar_values
            )
            return [grad_u, grad_s, grad_vt, scalar_gradient]

    else:

        @function.numpy(manifold)
        def cost(u, singular_values, vt):
            loss, smooth_loss, *_ = loss_and_grad(
                u, singular_values, vt, np.zeros(0, dtype=float)
            )
            active_mple_history.append(smooth_loss)
            active_penalized_history.append(loss)
            if verbose_every and eval_count[0] % verbose_every == 0:
                if logger is None:
                    print(
                        f"Pymanopt eval {eval_count[0]} | Loss: {smooth_loss:.6f} "
                        f"| Penalized: {loss:.6f}"
                    )
                else:
                    logger.info(
                        "Pymanopt eval %s | Loss: %.6f | Penalized: %.6f",
                        eval_count[0],
                        smooth_loss,
                        loss,
                    )
            eval_count[0] += 1
            return loss

        @function.numpy(manifold)
        def euclidean_gradient(u, singular_values, vt):
            _, _, _, _, grad_u, grad_s, grad_vt, _ = loss_and_grad(
                u, singular_values, vt, np.zeros(0, dtype=float)
            )
            return [grad_u, grad_s, grad_vt]

    problem = Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
    )
    optimizer = ConjugateGradient(
        max_iterations=max(1, int(steps)),
        min_gradient_norm=float(tol),
        max_cost_evaluations=max(10, int(steps) * 10),
        verbosity=0,
    )

    base_theta_init = (
        None if theta_init is None else np.asarray(theta_init, dtype=float)
    )
    best_theta: np.ndarray | None = None
    best_mple_history: list[float] = []
    best_penalized_history: list[float] = []
    best_result = None
    best_penalized_objective = np.inf
    best_start_index = 0
    start_summaries: list[dict[str, object]] = []

    for start_index in range(n_starts):
        eval_count[0] = 0
        active_mple_history = []
        active_penalized_history = []
        start_seed = int(seed) + start_index
        rng = np.random.default_rng(start_seed)
        initialization_kind = "random"
        if base_theta_init is not None and start_index == 0:
            theta_parts = unpack_theta(
                base_theta_init,
                artifacts,
                fixed_scalar_params=fixed,
            )
            initial_field = compose_field_matrix_from_theta(theta_parts, artifacts)
            field_point = _fixed_rank_point_from_field(initial_field, rank, point_type)
            initial_scalars = np.asarray(
                [theta_parts[name] for name in free_names], dtype=float
            )
        else:
            field_point = _random_fixed_rank_point(
                rng, t_steps, n_nodes, rank, point_type
            )
            initial_scalars = rng.normal(0.0, 0.1, size=len(free_names))
        initial_point = [field_point, initial_scalars] if free_names else field_point
        initial_penalized_objective = float(problem.cost(initial_point))
        initial_mple_loss = (
            float(active_mple_history[-1])
            if active_mple_history
            else float(initial_penalized_objective)
        )
        if logger is not None:
            logger.info(
                "Pymanopt start %s/%s | seed=%s | initialization=%s | initial_loss=%.6f | initial_penalized=%.6f",
                start_index + 1,
                n_starts,
                start_seed,
                initialization_kind,
                initial_mple_loss,
                initial_penalized_objective,
            )
        result = optimizer.run(problem, initial_point=initial_point)
        theta_hat = _low_rank_manifold_point_to_theta(
            result.point,
            artifacts,
            free_names,
            fixed,
        )
        final_loss = _pseudo_loss(
            x,
            z,
            theta_hat,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed,
        )
        final_field = compose_field_matrix_from_theta(
            unpack_theta(theta_hat, artifacts, fixed_scalar_params=fixed),
            artifacts,
        )
        frobenius_norm = float(np.linalg.norm(final_field, "fro"))
        normalized_frobenius_norm = frobenius_norm / frobenius_normalizer
        final_penalized = float(final_loss) + float(lambda_frobenius) * float(
            normalized_frobenius_norm
        )
        run_mple_history = list(active_mple_history)
        run_penalized_history = list(active_penalized_history)
        if not run_mple_history or run_mple_history[-1] != float(final_loss):
            run_mple_history.append(float(final_loss))
        if not run_penalized_history or run_penalized_history[-1] != float(
            final_penalized
        ):
            run_penalized_history.append(float(final_penalized))
        start_summary = {
            "start_index": start_index,
            "seed": start_seed,
            "initialization_kind": initialization_kind,
            "initial_mple_loss": float(initial_mple_loss),
            "initial_penalized_objective": float(initial_penalized_objective),
            "final_mple_loss": float(final_loss),
            "final_penalized_objective": float(final_penalized),
            "iterations": int(result.iterations),
            "cost_evaluations": int(
                result.cost_evaluations or len(run_penalized_history)
            ),
            "success": "max iterations" not in str(result.stopping_criterion).lower(),
            "message": str(result.stopping_criterion),
        }
        start_summaries.append(start_summary)
        if logger is not None:
            logger.info(
                "Pymanopt start %s/%s complete | final_loss=%.6f | penalized=%.6f | status=%s",
                start_index + 1,
                n_starts,
                final_loss,
                final_penalized,
                result.stopping_criterion,
            )
        if final_penalized < best_penalized_objective:
            best_penalized_objective = final_penalized
            best_theta = theta_hat
            best_mple_history = run_mple_history
            best_penalized_history = run_penalized_history
            best_result = result
            best_start_index = start_index

    if best_theta is None or best_result is None:
        raise RuntimeError("Pymanopt optimizer did not produce a candidate solution.")
    best_start = int(best_start_index)
    final_field = compose_field_matrix_from_theta(
        unpack_theta(best_theta, artifacts, fixed_scalar_params=fixed),
        artifacts,
    )
    final_frobenius_norm = float(np.linalg.norm(final_field, "fro"))
    final_normalized_frobenius_norm = final_frobenius_norm / frobenius_normalizer
    final_mple_loss = float(
        _pseudo_loss(
            x,
            z,
            best_theta,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params=fixed,
        )
    )
    optimize_result = OptimizeResult(
        x=best_theta,
        success=bool(start_summaries[best_start]["success"]),
        message=(
            f"{best_result.stopping_criterion} | best_start={best_start + 1}/{n_starts}"
        ),
        nit=int(best_result.iterations),
        nfev=int(best_result.cost_evaluations or len(best_penalized_history)),
        iterations=int(best_result.iterations),
        cost_evaluations=int(
            best_result.cost_evaluations or len(best_penalized_history)
        ),
        optimizer_mode=OPTIMIZER_MODE_MANIFOLD,
        optimizer="pymanopt_conjugate_gradient",
        lambda_frobenius=float(lambda_frobenius),
        final_penalized_objective=float(
            final_mple_loss + float(lambda_frobenius) * final_normalized_frobenius_norm
        ),
        final_mple_loss=final_mple_loss,
        frobenius_norm=final_frobenius_norm,
        normalized_frobenius_norm=float(final_normalized_frobenius_norm),
        frobenius_norm_normalizer=float(frobenius_normalizer),
        effective_rank=float(artifacts.latent_rank),
        best_start=int(best_start),
        n_starts=int(n_starts),
        mple_history=list(best_mple_history),
        penalized_history=list(best_penalized_history),
        start_summaries=start_summaries,
    )
    return best_theta, best_mple_history, optimize_result


def _fit_mple_alternative_low_rank(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    seed: int,
    verbose_every: int,
    tol: float,
    logger,
    theta_init,
    fixed_scalar_params: dict[str, float] | None,
    n_starts: int,
    lambda_uv_ridge: float,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")
    if artifacts.latent_rank == 0:
        theta_hat, history, result = _fit_zero_rank_unconstrained(
            x,
            z,
            x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            tol=tol,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
        )
        result["optimizer_mode"] = OPTIMIZER_MODE_ALTERNATIVE_LOW_RANK
        result["optimizer"] = "alternating_low_rank_zero_rank"
        result["best_start"] = 0
        result["n_starts"] = 1
        result["iterations"] = int(getattr(result, "nit", 0))
        result["cost_evaluations"] = int(getattr(result, "nfev", len(history)))
        result["lambda_uv_ridge"] = float(lambda_uv_ridge)
        result["final_mple_loss"] = float(history[-1])
        result["final_penalized_objective"] = float(history[-1])
        result["u_frobenius_norm"] = 0.0
        result["v_frobenius_norm"] = 0.0
        result["effective_rank"] = 0.0
        result["mple_history"] = list(history)
        result["penalized_history"] = list(history)
        result["start_summaries"] = [
            {
                "start_index": 0,
                "seed": int(seed),
                "initialization_kind": "random",
                "initial_mple_loss": float(history[0]),
                "initial_penalized_objective": float(history[0]),
                "final_mple_loss": float(history[-1]),
                "final_penalized_objective": float(history[-1]),
                "iterations": int(getattr(result, "nit", 0)),
                "cost_evaluations": int(getattr(result, "nfev", len(history))),
                "success": bool(getattr(result, "success", False)),
                "message": str(getattr(result, "message", "")),
            }
        ]
        return theta_hat, history, result

    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    free_names = free_scalar_parameter_names(fixed)
    n_starts = max(1, int(n_starts))
    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    outcome_size = float(np.asarray(x).size)
    prev_x = np.vstack([x_0, x[:-1, :]])
    beta_feature = np.asarray(z, dtype=float)
    base_theta_init = (
        None if theta_init is None else np.asarray(theta_init, dtype=float)
    )
    outer_iterations = max(1, int(steps))
    subproblem_maxiter = min(25, max(5, outer_iterations))

    def scalar_values_from_vector(free_scalar_values: np.ndarray) -> dict[str, float]:
        scalars = dict(fixed)
        scalars.update(
            {
                name: float(value)
                for name, value in zip(
                    free_names, np.asarray(free_scalar_values, dtype=float).reshape(-1)
                )
            }
        )
        return scalars

    def evaluate_state(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        free_scalar_values: np.ndarray,
    ) -> dict[str, object]:
        scalars = scalar_values_from_vector(free_scalar_values)
        field_matrix = compose_latent_field_matrix(node_factors, time_factors)
        h_x = (
            field_matrix
            + scalars["beta"] * beta_feature
            + scalars["eta"] * prev_x
            + scalars["xi"] * interaction_effect_x
        )
        loss_x = np.logaddexp(h_x, -h_x) - x * h_x
        residual = np.tanh(h_x) - x
        smooth_loss = float(loss_x.sum() / outcome_size)
        u_norm_sq = float(np.sum(np.asarray(time_factors, dtype=float) ** 2))
        v_norm_sq = float(np.sum(np.asarray(node_factors, dtype=float) ** 2))
        ridge_penalty = float(lambda_uv_ridge) * (u_norm_sq + v_norm_sq) / outcome_size
        scalar_gradient_lookup = {
            "beta": float((residual * beta_feature).sum()) / outcome_size,
            "xi": float((residual * interaction_effect_x).sum()) / outcome_size,
            "eta": float((residual * prev_x).sum()) / outcome_size,
        }
        return {
            "smooth_loss": smooth_loss,
            "penalized_loss": smooth_loss + ridge_penalty,
            "field_matrix": field_matrix,
            "residual": residual,
            "time_gradient": (residual @ np.asarray(node_factors, dtype=float))
            / outcome_size
            + (2.0 * float(lambda_uv_ridge) / outcome_size)
            * np.asarray(time_factors, dtype=float),
            "node_gradient": (residual.T @ np.asarray(time_factors, dtype=float))
            / outcome_size
            + (2.0 * float(lambda_uv_ridge) / outcome_size)
            * np.asarray(node_factors, dtype=float),
            "scalar_gradient_lookup": scalar_gradient_lookup,
            "u_frobenius_norm": float(np.linalg.norm(time_factors, ord="fro")),
            "v_frobenius_norm": float(np.linalg.norm(node_factors, ord="fro")),
        }

    def pack_state(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        free_scalar_values: np.ndarray,
    ) -> np.ndarray:
        scalars = scalar_values_from_vector(free_scalar_values)
        return pack_theta(
            {
                "time_factors": np.asarray(time_factors, dtype=float),
                "node_factors": np.asarray(node_factors, dtype=float),
                "beta": float(scalars["beta"]),
                "xi": float(scalars["xi"]),
                "eta": float(scalars["eta"]),
            },
            artifacts,
            fixed_scalar_params=fixed,
        )

    def initial_state_for_start(
        start_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start_seed = int(seed) + start_index
        rng = np.random.default_rng(start_seed)
        if base_theta_init is not None and start_index == 0:
            theta_parts = unpack_theta(
                base_theta_init,
                artifacts,
                fixed_scalar_params=fixed,
            )
            return (
                np.asarray(theta_parts["time_factors"], dtype=float),
                np.asarray(theta_parts["node_factors"], dtype=float),
                np.asarray([theta_parts[name] for name in free_names], dtype=float),
            )
        return (
            rng.normal(0.0, 0.1, size=(t_steps, rank)),
            rng.normal(0.0, 0.1, size=(n_nodes, rank)),
            rng.normal(0.0, 0.1, size=len(free_names)),
        )

    best_theta: np.ndarray | None = None
    best_result: OptimizeResult | None = None
    best_penalized_history: list[float] = []
    best_mple_history: list[float] = []
    best_start = 0
    best_penalized_objective = np.inf
    start_summaries: list[dict[str, object]] = []

    for start_index in range(n_starts):
        time_factors, node_factors, free_scalar_values = initial_state_for_start(
            start_index
        )
        state = evaluate_state(time_factors, node_factors, free_scalar_values)
        initial_mple_loss = float(state["smooth_loss"])
        initial_penalized_objective = float(state["penalized_loss"])
        mple_history = [initial_mple_loss]
        penalized_history = [initial_penalized_objective]
        cost_evaluations = 1
        iterations_completed = 0
        converged = False
        message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"

        if logger is not None:
            logger.info(
                "Alternating low-rank start %s/%s | seed=%s | initial_loss=%.6f | initial_penalized=%.6f",
                start_index + 1,
                n_starts,
                int(seed) + start_index,
                initial_mple_loss,
                initial_penalized_objective,
            )

        for outer_index in range(outer_iterations):
            if free_names:
                scalar_result = minimize(
                    lambda raw: _alt_scalar_objective(
                        raw,
                        time_factors,
                        node_factors,
                        free_names,
                        evaluate_state,
                    ),
                    free_scalar_values,
                    method="BFGS",
                    jac=True,
                    options={"maxiter": subproblem_maxiter, "gtol": float(tol)},
                )
                free_scalar_values = np.asarray(scalar_result.x, dtype=float)
                cost_evaluations += int(getattr(scalar_result, "nfev", 0))

            time_result = minimize(
                lambda raw: _alt_time_objective(
                    raw,
                    node_factors,
                    free_scalar_values,
                    t_steps,
                    rank,
                    evaluate_state,
                ),
                np.asarray(time_factors, dtype=float).reshape(-1),
                method="BFGS",
                jac=True,
                options={"maxiter": subproblem_maxiter, "gtol": float(tol)},
            )
            time_factors = np.asarray(time_result.x, dtype=float).reshape(t_steps, rank)
            cost_evaluations += int(getattr(time_result, "nfev", 0))

            node_result = minimize(
                lambda raw: _alt_node_objective(
                    raw,
                    time_factors,
                    free_scalar_values,
                    n_nodes,
                    rank,
                    evaluate_state,
                ),
                np.asarray(node_factors, dtype=float).reshape(-1),
                method="BFGS",
                jac=True,
                options={"maxiter": subproblem_maxiter, "gtol": float(tol)},
            )
            node_factors = np.asarray(node_result.x, dtype=float).reshape(n_nodes, rank)
            cost_evaluations += int(getattr(node_result, "nfev", 0))

            state = evaluate_state(time_factors, node_factors, free_scalar_values)
            penalized_history.append(float(state["penalized_loss"]))
            mple_history.append(float(state["smooth_loss"]))
            iterations_completed = outer_index + 1

            if verbose_every and outer_index % verbose_every == 0:
                if logger is None:
                    print(
                        f"Alternating low-rank iter {outer_index} | Loss: {state['smooth_loss']:.6f} | Penalized: {state['penalized_loss']:.6f}"
                    )
                else:
                    logger.info(
                        "Alternating low-rank iter %s | Loss: %.6f | Penalized: %.6f",
                        outer_index,
                        state["smooth_loss"],
                        state["penalized_loss"],
                    )

            if len(penalized_history) >= 2:
                improvement = abs(penalized_history[-2] - penalized_history[-1])
                if improvement <= float(tol) * max(1.0, abs(penalized_history[-2])):
                    converged = True
                    message = "CONVERGED: alternating objective tolerance reached"
                    break

        theta_hat = pack_state(time_factors, node_factors, free_scalar_values)
        final_state = evaluate_state(time_factors, node_factors, free_scalar_values)
        start_summary = {
            "start_index": start_index,
            "seed": int(seed) + start_index,
            "initialization_kind": "random",
            "initial_mple_loss": initial_mple_loss,
            "initial_penalized_objective": initial_penalized_objective,
            "final_mple_loss": float(final_state["smooth_loss"]),
            "final_penalized_objective": float(final_state["penalized_loss"]),
            "iterations": int(iterations_completed),
            "cost_evaluations": int(cost_evaluations),
            "success": bool(converged),
            "message": message,
        }
        start_summaries.append(start_summary)
        if float(final_state["penalized_loss"]) < best_penalized_objective:
            best_penalized_objective = float(final_state["penalized_loss"])
            best_start = start_index
            best_theta = theta_hat
            best_mple_history = list(mple_history)
            best_penalized_history = list(penalized_history)
            best_result = OptimizeResult(
                x=theta_hat,
                success=bool(converged),
                message=message,
                nit=int(iterations_completed),
                nfev=int(cost_evaluations),
                iterations=int(iterations_completed),
                cost_evaluations=int(cost_evaluations),
                optimizer_mode=OPTIMIZER_MODE_ALTERNATIVE_LOW_RANK,
                optimizer="alternating_low_rank",
                lambda_uv_ridge=float(lambda_uv_ridge),
                final_mple_loss=float(final_state["smooth_loss"]),
                final_penalized_objective=float(final_state["penalized_loss"]),
                u_frobenius_norm=float(final_state["u_frobenius_norm"]),
                v_frobenius_norm=float(final_state["v_frobenius_norm"]),
                effective_rank=float(
                    np.linalg.matrix_rank(
                        np.asarray(final_state["field_matrix"], dtype=float)
                    )
                ),
                mple_history=list(mple_history),
                penalized_history=list(penalized_history),
                best_start=int(start_index),
                n_starts=int(n_starts),
                start_summaries=start_summaries,
            )

    if best_theta is None or best_result is None:
        raise RuntimeError(
            "Alternating low-rank optimizer did not produce a candidate solution."
        )
    best_result["start_summaries"] = start_summaries
    best_result["best_start"] = int(best_start)
    best_result["n_starts"] = int(n_starts)
    return best_theta, best_mple_history, best_result


def _alt_scalar_objective(
    raw_scalars: np.ndarray,
    time_factors: np.ndarray,
    node_factors: np.ndarray,
    free_names: list[str],
    evaluate_state,
) -> tuple[float, np.ndarray]:
    state = evaluate_state(
        time_factors, node_factors, np.asarray(raw_scalars, dtype=float)
    )
    gradient = np.asarray(
        [state["scalar_gradient_lookup"][name] for name in free_names],
        dtype=float,
    )
    return float(state["penalized_loss"]), gradient


def _alt_time_objective(
    raw_time: np.ndarray,
    node_factors: np.ndarray,
    free_scalar_values: np.ndarray,
    t_steps: int,
    rank: int,
    evaluate_state,
) -> tuple[float, np.ndarray]:
    time_factors = np.asarray(raw_time, dtype=float).reshape(t_steps, rank)
    state = evaluate_state(time_factors, node_factors, free_scalar_values)
    return float(state["penalized_loss"]), np.asarray(
        state["time_gradient"], dtype=float
    ).reshape(-1)


def _alt_node_objective(
    raw_node: np.ndarray,
    time_factors: np.ndarray,
    free_scalar_values: np.ndarray,
    n_nodes: int,
    rank: int,
    evaluate_state,
) -> tuple[float, np.ndarray]:
    node_factors = np.asarray(raw_node, dtype=float).reshape(n_nodes, rank)
    state = evaluate_state(time_factors, node_factors, free_scalar_values)
    return float(state["penalized_loss"]), np.asarray(
        state["node_gradient"], dtype=float
    ).reshape(-1)


def fit_mple(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    s: int,
    param_names: list[str],
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int = 2000,
    seed: int = 0,
    verbose_every: int = 100,
    tol: float = 1e-9,
    logger=None,
    theta_init=None,
    fixed_scalar_params: dict[str, float] | None = None,
    n_starts: int = 1,
    lambda_nuclear: float = 0.0,
    lambda_frobenius: float = 0.0,
    lambda_uv_ridge: float = 0.0,
    proximal_lr: float = 1.0,
):
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError("x and z must both have shape (T, N).")

    t_steps = x.shape[0]
    if t_steps != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
        return _fit_mple_nuclear_norm(
            x,
            z,
            x_0=x_0,
            param_names=param_names,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            verbose_every=verbose_every,
            tol=tol,
            logger=logger,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
            lambda_nuclear=lambda_nuclear,
            proximal_lr=proximal_lr,
        )
    if artifacts.optimizer_mode == OPTIMIZER_MODE_ALTERNATIVE_LOW_RANK:
        return _fit_mple_alternative_low_rank(
            x,
            z,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            verbose_every=verbose_every,
            tol=tol,
            logger=logger,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
            n_starts=n_starts,
            lambda_uv_ridge=lambda_uv_ridge,
        )
    if artifacts.optimizer_mode == OPTIMIZER_MODE_MANIFOLD:
        return _fit_mple_low_rank_manifold(
            x,
            z,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            verbose_every=verbose_every,
            tol=tol,
            logger=logger,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
            n_starts=n_starts,
            lambda_frobenius=lambda_frobenius,
        )
    raise ValueError(f"Unsupported optimizer_mode: {artifacts.optimizer_mode}")


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def scalar_summary_rows(
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    scalar_truths: dict[str, float] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    rows: list[dict[str, object]] = []
    for name in scalar_parameter_names():
        est = float(est_parts[name])
        true = None if scalar_truths is None else scalar_truths.get(name)
        rows.append(
            {
                "category": "scalar",
                "name": name,
                "estimate": est,
                "true": None if true is None else float(true),
                "squared_error": None if true is None else float((est - true) ** 2),
            }
        )
    return rows


def load_truth_context(truth_artifact_dir: str | Path) -> dict[str, object] | None:
    truth_root = Path(truth_artifact_dir)
    metadata_path = truth_root / "experiment_metadata.yaml"
    metadata = (
        OmegaConf.load(metadata_path)
        if metadata_path.exists()
        else OmegaConf.create({})
    )
    if not bool(metadata.get("has_truth", True)):
        return None
    config_path = first_existing_path(
        truth_root / "generation_realized_config.yaml",
        truth_root / "realized_config.yaml",
    )
    truth_config = load_yaml_config(config_path)
    truth_artifacts = load_model_artifacts(truth_root)
    scalar_truths = {
        "beta": float(truth_config.estimation_params.beta),
        "xi": float(truth_config.estimation_params.xi),
        "eta": float(truth_config.estimation_params.eta),
    }
    truth_interaction = compose_interaction_matrix(
        scalar_truths["xi"], truth_artifacts.gamma_matrix
    )
    return {
        "scalar_truths": scalar_truths,
        "field_artifacts": truth_artifacts,
        "field_matrix": truth_artifacts.field_matrix,
        "interaction_matrix": truth_interaction,
    }


def compute_truth_metrics(
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, float]:
    if truth_context is None or truth_context.get("field_matrix") is None:
        return {}
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    est_artifacts = with_theta_field(artifacts, est_parts)
    true_field = np.asarray(truth_context["field_matrix"], dtype=float)
    est_interaction = compose_interaction_matrix(
        est_parts["xi"], artifacts.gamma_matrix
    )
    true_interaction = truth_context.get("interaction_matrix")
    if true_interaction is None:
        interaction_fro_error = None
    elif sparse.issparse(est_interaction):
        interaction_error = est_interaction - true_interaction
        interaction_fro_error = float(
            np.sqrt(interaction_error.multiply(interaction_error).sum())
        )
    else:
        interaction_fro_error = float(
            np.linalg.norm(est_interaction - true_interaction, ord="fro")
        )
    metrics: dict[str, float] = {
        "field_rmse": float(
            np.sqrt(np.mean((np.asarray(est_artifacts.field_matrix) - true_field) ** 2))
        )
    }
    if interaction_fro_error is not None:
        metrics["interaction_fro_error"] = interaction_fro_error
    return metrics


def latent_diagnostic_rows(
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    est_artifacts = with_theta_field(artifacts, est_parts)
    est_field = np.asarray(est_artifacts.field_matrix, dtype=float)
    rows: list[dict[str, object]] = [
        {
            "category": "latent_diagnostic",
            "name": "estimated_field_inf_norm",
            "estimate": latent_field_bound_norm(est_field),
            "true": None,
            "squared_error": None,
        },
        {
            "category": "latent_diagnostic",
            "name": "estimated_field_rank",
            "estimate": float(np.linalg.matrix_rank(est_field)),
            "true": float(artifacts.latent_rank),
            "squared_error": None,
        },
    ]
    if truth_context is not None and truth_context.get("field_matrix") is not None:
        true_field = np.asarray(truth_context["field_matrix"], dtype=float)
        rows.extend(
            [
                {
                    "category": "latent_diagnostic",
                    "name": "true_field_inf_norm",
                    "estimate": latent_field_bound_norm(true_field),
                    "true": None,
                    "squared_error": None,
                },
                {
                    "category": "latent_diagnostic",
                    "name": "true_field_rank",
                    "estimate": float(np.linalg.matrix_rank(true_field)),
                    "true": None,
                    "squared_error": None,
                },
            ]
        )
    return rows


def write_summary_table(
    summary_stem,
    est_theta,
    metrics,
    loss,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
):
    csv_path = Path(f"{summary_stem}.csv")
    md_path = Path(f"{summary_stem}.md")
    rows = scalar_summary_rows(
        est_theta,
        artifacts,
        scalar_truths=(
            None if truth_context is None else truth_context.get("scalar_truths")
        ),
        fixed_scalar_params=fixed_scalar_params,
    )
    rows.extend(
        {
            "category": "metric",
            "name": name,
            "estimate": float(value),
            "true": None,
            "squared_error": None,
        }
        for name, value in {"final_loss": loss, **metrics}.items()
    )
    rows.extend(
        latent_diagnostic_rows(
            est_theta,
            artifacts,
            truth_context=truth_context,
            fixed_scalar_params=fixed_scalar_params,
        )
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["category", "name", "estimate", "true", "squared_error"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| category | name | estimate | true | squared_error |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['category']} | {row['name']} | {_fmt(row['estimate'])} | {_fmt(row['true'])} | {_fmt(row['squared_error'])} |\n"
            )


def write_optimizer_start_summary(path: str | Path, result: OptimizeResult) -> None:
    start_summaries = result.get("start_summaries", [])
    if not start_summaries:
        return
    fieldnames = [
        "start_index",
        "seed",
        "initialization_kind",
        "initial_mple_loss",
        "initial_penalized_objective",
        "final_mple_loss",
        "final_penalized_objective",
        "iterations",
        "cost_evaluations",
        "success",
        "message",
        "is_best",
    ]
    best_start = int(result.get("best_start", 0))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in start_summaries:
            start_index = int(row["start_index"])
            writer.writerow(
                {
                    **row,
                    "is_best": start_index == best_start,
                }
            )


def log_estimates(logger, title, scalar_rows):
    logger.info(title)
    for row in scalar_rows:
        if row["true"] is None:
            logger.info("  %s: %.4f", row["name"], row["estimate"])
        else:
            logger.info(
                "  %s: %.4f (True: %.4f) | SQE: %.6f",
                row["name"],
                row["estimate"],
                row["true"],
                row["squared_error"],
            )


def log_field_diagnostics(
    logger,
    metrics: dict[str, float],
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> None:
    logger.info("Field and interaction diagnostics:")
    for name in ["field_rmse", "interaction_fro_error"]:
        if name in metrics:
            logger.info("  %s: %.6f", name, metrics[name])
    for row in latent_diagnostic_rows(
        est_theta,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    ):
        logger.info("  %s: %s", row["name"], _fmt(row["estimate"]))


def save_estimated_artifacts(
    data_folder: str | Path,
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> None:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    est_artifacts = with_theta_field(
        artifacts,
        est_parts,
    )
    save_field_artifacts(
        Path(data_folder) / "estimated_field_artifacts.npz", est_artifacts
    )
    estimated_interaction = compose_interaction_matrix(
        est_parts["xi"],
        artifacts.gamma_matrix,
    )
    if sparse.issparse(estimated_interaction):
        sparse.save_npz(
            _io_path(Path(data_folder) / "estimated_interaction_matrix_sparse.npz"),
            estimated_interaction,
        )
    else:
        np.save(
            _io_path(Path(data_folder) / "estimated_interaction_matrix.npy"),
            estimated_interaction,
        )
    save_estimated_parameter_bundle(
        Path(data_folder) / "estimated_parameter_bundle.npz",
        beta=float(est_parts["beta"]),
        xi=float(est_parts["xi"]),
        eta=float(est_parts["eta"]),
        latent_rank=int(est_artifacts.latent_rank),
        t_steps=int(est_artifacts.t_steps),
        field_matrix=np.asarray(est_artifacts.field_matrix, dtype=float),
    )
    if truth_context is None or truth_context.get("field_artifacts") is None:
        return
    save_field_artifacts(
        Path(data_folder) / "true_field_artifacts.npz",
        truth_context["field_artifacts"],
    )
    true_interaction = truth_context.get("interaction_matrix")
    if true_interaction is None:
        return
    if sparse.issparse(true_interaction):
        sparse.save_npz(
            _io_path(Path(data_folder) / "true_interaction_matrix_sparse.npz"),
            true_interaction,
        )
    else:
        np.save(
            _io_path(Path(data_folder) / "true_interaction_matrix.npy"),
            true_interaction,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit active conditional-model parameters with MPLE."
    )
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--model_artifact_dir", type=str, default=None)
    parser.add_argument("--truth_artifact_dir", type=str, default=None)
    parser.add_argument("--panel_path", type=str, default=None)
    parser.add_argument("--x0_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_starts", type=int, default=None)
    parser.add_argument("--lambda_nuclear", type=float, default=None)
    parser.add_argument("--lambda_frobenius", type=float, default=None)
    parser.add_argument("--lambda_uv_ridge", type=float, default=None)
    parser.add_argument("--proximal_lr", type=float, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    log_file = args.log_file or str(Path(args.data_folder) / "mple.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Loading data...")
    config_path = (
        Path(args.config_path)
        if args.config_path
        else first_existing_path(
            Path(args.data_folder) / "fit_realized_config.yaml",
            Path(args.data_folder) / "generation_realized_config.yaml",
            Path(args.data_folder) / "realized_config.yaml",
        )
    )
    config = load_yaml_config(config_path)
    fixed_scalar_params = validate_fixed_scalar_params(
        (
            OmegaConf.to_container(
                config.estimation_params.get("fixed_scalar_params", {}),
                resolve=True,
            )
            if "estimation_params" in config
            else {}
        )
    )
    lambda_nuclear = float(
        args.lambda_nuclear
        if args.lambda_nuclear is not None
        else config.global_params.get("lambda_nuclear", 0.0)
    )
    lambda_frobenius = float(
        args.lambda_frobenius
        if args.lambda_frobenius is not None
        else config.global_params.get("lambda_frobenius", 0.0)
    )
    lambda_uv_ridge = float(
        args.lambda_uv_ridge
        if args.lambda_uv_ridge is not None
        else config.global_params.get("lambda_uv_ridge", 0.0)
    )
    optimizer_params = (
        config.optimizer_params
        if "optimizer_params" in config
        else OmegaConf.create({})
    )
    steps = int(
        args.steps if args.steps is not None else optimizer_params.get("steps", 10000)
    )
    tol = float(args.tol if args.tol is not None else optimizer_params.get("tol", 1e-9))
    seed = int(args.seed if args.seed is not None else optimizer_params.get("seed", 0))
    n_starts = int(
        args.n_starts
        if args.n_starts is not None
        else optimizer_params.get("n_starts", 1)
    )
    proximal_lr = float(
        args.proximal_lr
        if args.proximal_lr is not None
        else optimizer_params.get("proximal_lr", 1.0)
    )
    model_artifact_dir = (
        Path(args.model_artifact_dir)
        if args.model_artifact_dir
        else Path(args.data_folder)
    )
    truth_artifact_dir = (
        Path(args.truth_artifact_dir) if args.truth_artifact_dir else model_artifact_dir
    )

    panel_path = (
        Path(args.panel_path)
        if args.panel_path
        else truth_artifact_dir / "panel_data.npz"
    )
    x0_path = Path(args.x0_path) if args.x0_path else truth_artifact_dir / "x_0.npy"
    logger.info("Using panel artifact: %s", panel_path)
    logger.info("Using x_0 artifact: %s", x0_path)
    logger.info("Using fit config: %s", config_path)
    logger.info("Using model artifact directory: %s", model_artifact_dir)
    logger.info("Using truth artifact directory: %s", truth_artifact_dir)
    x_0 = np.load(x0_path)
    panel = load_panel_artifact(panel_path)
    x = panel["x"]
    z = panel["z"]

    gamma_matrix = load_gamma_matrix(model_artifact_dir)
    artifacts = build_fit_model_artifacts(config, gamma_matrix)
    param_keys = parameter_names(
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    truth_context = load_truth_context(truth_artifact_dir)
    interaction_effect_x = interaction_effect(x, artifacts.gamma_matrix)
    logger.info("Configured optimizer mode: %s", artifacts.optimizer_mode)
    logger.info("Configured latent rank: %s", artifacts.latent_rank)
    logger.info("Using a fixed known graph with scalar xi.")
    logger.info("Fit-time hard bounds active: False")
    logger.info("Fixed scalar parameters: %s", fixed_scalar_params or {})
    logger.info(
        "Optimizer settings: steps=%s, tol=%s, n_starts=%s, seed=%s, "
        "lambda_nuclear=%s, lambda_frobenius=%s, lambda_uv_ridge=%s, proximal_lr=%s",
        steps,
        tol,
        n_starts,
        seed,
        lambda_nuclear,
        lambda_frobenius,
        lambda_uv_ridge,
        proximal_lr,
    )

    params_hat, loss_history, result = fit_mple(
        x,
        z,
        x_0=x_0,
        s=int(config.global_params.s),
        param_names=param_keys,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        steps=steps,
        tol=tol,
        seed=seed,
        logger=logger,
        fixed_scalar_params=fixed_scalar_params,
        n_starts=n_starts,
        lambda_nuclear=lambda_nuclear,
        lambda_frobenius=lambda_frobenius,
        lambda_uv_ridge=lambda_uv_ridge,
        proximal_lr=proximal_lr,
    )

    logger.info("Done fitting.")
    logger.info("Optimizer status: %s", result.message)
    logger.info(
        "Best optimizer start: %s / %s",
        int(result.get("best_start", 0)) + 1,
        int(result.get("n_starts", 1)),
    )
    logger.info("Final Loss: %.6f", loss_history[-1])
    scalar_rows = scalar_summary_rows(
        params_hat,
        artifacts,
        scalar_truths=(
            None if truth_context is None else truth_context.get("scalar_truths")
        ),
        fixed_scalar_params=fixed_scalar_params,
    )
    log_estimates(
        logger,
        (
            "Estimated vs True Parameters:"
            if truth_context is not None
            else "Estimated Parameters:"
        ),
        scalar_rows,
    )
    metrics = compute_truth_metrics(
        params_hat,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    if result.get("optimizer_mode") == OPTIMIZER_MODE_NUCLEAR_NORM:
        metrics.update(
            {
                "penalized_objective": float(result["final_penalized_objective"]),
                "mple_loss_without_penalty": float(result["final_mple_loss"]),
                "nuclear_norm": float(result["nuclear_norm"]),
                "normalized_nuclear_norm": float(result["normalized_nuclear_norm"]),
                "nuclear_norm_normalizer": float(result["nuclear_norm_normalizer"]),
                "effective_rank": float(result["effective_rank"]),
                "proximal_iterations": float(result["proximal_iterations"]),
            }
        )
        logger.info("Nuclear-norm optimizer diagnostics:")
        logger.info("  penalized_objective: %.6f", metrics["penalized_objective"])
        logger.info(
            "  mple_loss_without_penalty: %.6f", metrics["mple_loss_without_penalty"]
        )
        logger.info("  nuclear_norm: %.6f", metrics["nuclear_norm"])
        logger.info(
            "  normalized_nuclear_norm: %.6f", metrics["normalized_nuclear_norm"]
        )
        logger.info(
            "  nuclear_norm_normalizer: %.6f", metrics["nuclear_norm_normalizer"]
        )
        logger.info("  effective_rank: %.6f", metrics["effective_rank"])
        logger.info("  proximal_iterations: %.0f", metrics["proximal_iterations"])
    elif result.get("optimizer_mode") == OPTIMIZER_MODE_MANIFOLD:
        metrics.update(
            {
                "penalized_objective": float(result["final_penalized_objective"]),
                "mple_loss_without_penalty": float(result["final_mple_loss"]),
                "lambda_frobenius": float(result["lambda_frobenius"]),
                "frobenius_norm": float(result["frobenius_norm"]),
                "normalized_frobenius_norm": float(result["normalized_frobenius_norm"]),
                "frobenius_norm_normalizer": float(result["frobenius_norm_normalizer"]),
                "effective_rank": float(result["effective_rank"]),
            }
        )
        if float(result["lambda_frobenius"]) > 0.0:
            logger.info("Low-rank Frobenius optimizer diagnostics:")
            logger.info("  penalized_objective: %.6f", metrics["penalized_objective"])
            logger.info(
                "  mple_loss_without_penalty: %.6f",
                metrics["mple_loss_without_penalty"],
            )
            logger.info("  lambda_frobenius: %.6f", metrics["lambda_frobenius"])
            logger.info("  frobenius_norm: %.6f", metrics["frobenius_norm"])
            logger.info(
                "  normalized_frobenius_norm: %.6f",
                metrics["normalized_frobenius_norm"],
            )
            logger.info(
                "  frobenius_norm_normalizer: %.6f",
                metrics["frobenius_norm_normalizer"],
            )
    elif result.get("optimizer_mode") == OPTIMIZER_MODE_ALTERNATIVE_LOW_RANK:
        metrics.update(
            {
                "penalized_objective": float(result["final_penalized_objective"]),
                "mple_loss_without_penalty": float(result["final_mple_loss"]),
                "lambda_uv_ridge": float(result["lambda_uv_ridge"]),
                "u_frobenius_norm": float(result["u_frobenius_norm"]),
                "v_frobenius_norm": float(result["v_frobenius_norm"]),
                "effective_rank": float(result["effective_rank"]),
            }
        )
        logger.info("Alternating low-rank optimizer diagnostics:")
        logger.info("  penalized_objective: %.6f", metrics["penalized_objective"])
        logger.info(
            "  mple_loss_without_penalty: %.6f",
            metrics["mple_loss_without_penalty"],
        )
        logger.info("  lambda_uv_ridge: %.6f", metrics["lambda_uv_ridge"])
        logger.info("  u_frobenius_norm: %.6f", metrics["u_frobenius_norm"])
        logger.info("  v_frobenius_norm: %.6f", metrics["v_frobenius_norm"])
        logger.info("  effective_rank: %.6f", metrics["effective_rank"])
    log_field_diagnostics(
        logger,
        metrics,
        params_hat,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    write_summary_table(
        Path(args.data_folder) / "mple_summary",
        params_hat,
        metrics,
        loss_history[-1],
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    write_optimizer_start_summary(
        Path(args.data_folder) / "optimizer_start_summary.csv",
        result,
    )
    save_estimated_artifacts(
        args.data_folder,
        params_hat,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    logger.info(
        "Saved summary tables to %s and %s",
        Path(args.data_folder) / "mple_summary.csv",
        Path(args.data_folder) / "mple_summary.md",
    )
    logger.info("Log saved to %s", log_file)


if __name__ == "__main__":
    main()
