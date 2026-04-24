"""Fit the latent-only conditional MPLE model for synthetic and real-data experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from scipy.optimize import OptimizeResult, minimize

from pymanopt import Problem, function
from pymanopt.manifolds import Euclidean, FixedRankEmbedded, Product
from pymanopt.optimizers import ConjugateGradient

from io_utils import (
    _fmt,
    first_existing_path,
    io_path,
    load_gamma_matrix,
    load_yaml_config,
)
from loading_utils import save_estimated_parameter_bundle
from model_utils import (
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    ModelArtifacts,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
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


@dataclass(frozen=True)
class _FitEvalContext:
    x: np.ndarray
    prev_x: np.ndarray
    beta_feature: np.ndarray
    interaction_effect_x: np.ndarray
    outcome_size: float
    fixed_scalar_params: dict[str, float]
    free_scalar_names: list[str]


def _build_fit_eval_context(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None,
) -> _FitEvalContext:
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    x_array = np.asarray(x, dtype=float)
    return _FitEvalContext(
        x=x_array,
        prev_x=np.vstack([np.asarray(x_0, dtype=float), x_array[:-1, :]]),
        beta_feature=np.asarray(z, dtype=float),
        interaction_effect_x=np.asarray(interaction_effect_x, dtype=float),
        outcome_size=float(x_array.size),
        fixed_scalar_params=fixed,
        free_scalar_names=free_scalar_parameter_names(fixed),
    )


def _scalar_values_from_free_vector(
    free_scalar_values: np.ndarray,
    context: _FitEvalContext,
) -> dict[str, float]:
    scalars = dict(context.fixed_scalar_params)
    scalars.update(
        {
            name: float(value)
            for name, value in zip(
                context.free_scalar_names,
                np.asarray(free_scalar_values, dtype=float).reshape(-1),
            )
        }
    )
    return scalars


def _resolve_scalar_values(
    *,
    context: _FitEvalContext,
    free_scalar_values: np.ndarray | None = None,
    scalar_values: dict[str, float] | None = None,
) -> dict[str, float]:
    if scalar_values is not None:
        resolved = dict(context.fixed_scalar_params)
        resolved.update({name: float(value) for name, value in scalar_values.items()})
        return resolved
    if free_scalar_values is None:
        raise ValueError("Either free_scalar_values or scalar_values must be provided.")
    return _scalar_values_from_free_vector(free_scalar_values, context)


def _compute_h_x(
    field_matrix: np.ndarray,
    scalar_values: dict[str, float],
    context: _FitEvalContext,
) -> np.ndarray:
    return (
        np.asarray(field_matrix, dtype=float)
        + float(scalar_values["beta"]) * context.beta_feature
        + float(scalar_values["eta"]) * context.prev_x
        + float(scalar_values["xi"]) * context.interaction_effect_x
    )


def _scalar_gradient_from_residual(
    residual: np.ndarray,
    context: _FitEvalContext,
) -> np.ndarray:
    if not context.free_scalar_names:
        return np.zeros(0, dtype=float)
    gradient_lookup = {
        "beta": float((residual * context.beta_feature).sum()) / context.outcome_size,
        "xi": float((residual * context.interaction_effect_x).sum())
        / context.outcome_size,
        "eta": float((residual * context.prev_x).sum()) / context.outcome_size,
    }
    return np.asarray(
        [gradient_lookup[name] for name in context.free_scalar_names],
        dtype=float,
    )


def _evaluate_full_field_loss(
    field_matrix: np.ndarray,
    context: _FitEvalContext,
    *,
    free_scalar_values: np.ndarray | None = None,
    scalar_values: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    resolved_scalars = _resolve_scalar_values(
        context=context,
        free_scalar_values=free_scalar_values,
        scalar_values=scalar_values,
    )
    h_x = _compute_h_x(field_matrix, resolved_scalars, context)
    loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
    residual = np.tanh(h_x) - context.x
    smooth_loss = float(loss_x.sum() / context.outcome_size)
    scalar_gradient = _scalar_gradient_from_residual(residual, context)
    return smooth_loss, residual, scalar_gradient


def _evaluate_scalar_only_loss(
    free_scalar_values: np.ndarray,
    context: _FitEvalContext,
) -> tuple[float, np.ndarray]:
    smooth_loss, residual, scalar_gradient = _evaluate_full_field_loss(
        np.zeros_like(context.x, dtype=float),
        context,
        free_scalar_values=free_scalar_values,
    )
    return smooth_loss, scalar_gradient


def _evaluate_factorized_loss(
    time_factors: np.ndarray,
    node_factors: np.ndarray,
    context: _FitEvalContext,
    *,
    free_scalar_values: np.ndarray | None = None,
    scalar_values: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    field_matrix = compose_latent_field_matrix(node_factors, time_factors)
    smooth_loss, residual, scalar_gradient = _evaluate_full_field_loss(
        field_matrix,
        context,
        free_scalar_values=free_scalar_values,
        scalar_values=scalar_values,
    )
    time_gradient = (
        residual @ np.asarray(node_factors, dtype=float)
    ) / context.outcome_size
    node_gradient = (
        residual.T @ np.asarray(time_factors, dtype=float)
    ) / context.outcome_size
    return smooth_loss, residual, time_gradient, node_gradient, scalar_gradient


def _evaluate_factorized_loss_with_offset(
    time_factors: np.ndarray,
    node_factors: np.ndarray,
    scalar_offset: np.ndarray,
    context: _FitEvalContext,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    field_matrix = time_factors @ node_factors.T
    h_x = field_matrix + scalar_offset
    loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
    residual = np.tanh(h_x) - context.x
    smooth_loss = float(loss_x.sum() / context.outcome_size)
    time_gradient = (residual @ node_factors) / context.outcome_size
    node_gradient = (residual.T @ time_factors) / context.outcome_size
    return smooth_loss, residual, time_gradient, node_gradient


def _prox_threshold_field_matrix(
    field_matrix: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(field_matrix, dtype=float)
    if matrix.size == 0 or threshold <= 0.0:
        nuclear_norm = (
            float(np.linalg.svd(matrix, compute_uv=False).sum()) if matrix.size else 0.0
        )
        return matrix.copy(), nuclear_norm
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - float(threshold), 0.0)
    if not np.any(shrunk):
        return np.zeros_like(matrix, dtype=float), 0.0
    return (u * shrunk) @ vt, float(shrunk.sum())


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
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fixed_scalar_params=context.fixed_scalar_params,
    )
    if uses_full_matrix_parameterization(artifacts):
        smooth_loss, residual, scalar_gradient = _evaluate_full_field_loss(
            np.asarray(theta_parts["field_matrix"], dtype=float),
            context,
            scalar_values={
                "beta": theta_parts["beta"],
                "xi": theta_parts["xi"],
                "eta": theta_parts["eta"],
            },
        )
        field_grad = residual.reshape(-1) / context.outcome_size
    else:
        smooth_loss, _, time_grad, node_grad, scalar_gradient = (
            _evaluate_factorized_loss(
                np.asarray(theta_parts["time_factors"], dtype=float),
                np.asarray(theta_parts["node_factors"], dtype=float),
                context,
                scalar_values={
                    "beta": theta_parts["beta"],
                    "xi": theta_parts["xi"],
                    "eta": theta_parts["eta"],
                },
            )
        )
        field_grad = np.concatenate([node_grad.reshape(-1), time_grad.reshape(-1)])
    return float(smooth_loss), np.concatenate([field_grad, scalar_gradient])


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
    """Fit via proximal gradient descent with nuclear-norm regularization on the field matrix.

    Optimizes the full (unconstrained) field matrix in vectorized theta space, applying a
    soft-threshold proximal operator after each gradient step to promote low-rank solutions.
    Prefer this mode when you want data-driven rank selection rather than a fixed rank.
    """
    if lambda_nuclear < 0.0:
        raise ValueError("lambda_nuclear must be nonnegative.")
    if proximal_lr <= 0.0:
        raise ValueError("proximal_lr must be positive.")

    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    free_scalar_count = len(context.free_scalar_names)
    field_size = int(artifacts.t_steps) * n_nodes
    rng = np.random.default_rng(seed)
    raw_init = (
        rng.normal(0, 0.1, size=len(param_names))
        if theta_init is None
        else np.asarray(theta_init, dtype=float)
    )
    theta = np.asarray(raw_init, dtype=float)
    field_matrix = theta[:field_size].reshape(artifacts.t_steps, n_nodes)
    free_scalars = theta[field_size : field_size + free_scalar_count].copy()
    y_field = field_matrix.copy()
    y_scalars = free_scalars.copy()
    momentum = 1.0
    history: list[float] = []
    penalized_history: list[float] = []
    converged = False
    previous_objective = np.inf
    nuclear_normalizer = _nuclear_norm_normalizer(artifacts)
    initial_smooth_loss, _, _ = _evaluate_full_field_loss(
        field_matrix,
        context,
        free_scalar_values=free_scalars,
    )
    initial_nuclear_norm = _nuclear_norm(field_matrix)
    initial_penalized_obj = float(
        initial_smooth_loss
        + float(lambda_nuclear) * (initial_nuclear_norm / nuclear_normalizer)
    )

    for iteration in range(max(1, int(steps))):
        loss_y, residual_y, scalar_grad_y = _evaluate_full_field_loss(
            y_field,
            context,
            free_scalar_values=y_scalars,
        )
        field_grad_y = residual_y / context.outcome_size
        stepped_field = y_field - float(proximal_lr) * field_grad_y
        stepped_scalars = y_scalars - float(proximal_lr) * scalar_grad_y
        candidate_field, nuclear_norm = _prox_threshold_field_matrix(
            stepped_field,
            threshold=float(proximal_lr) * float(lambda_nuclear) / nuclear_normalizer,
        )
        smooth_loss, _, _ = _evaluate_full_field_loss(
            candidate_field,
            context,
            free_scalar_values=stepped_scalars,
        )
        normalized_nuclear_norm = nuclear_norm / nuclear_normalizer
        penalized_obj = float(
            smooth_loss + float(lambda_nuclear) * normalized_nuclear_norm
        )
        history.append(smooth_loss)
        penalized_history.append(penalized_obj)
        if verbose_every and iteration % verbose_every == 0:
            candidate_theta = np.concatenate(
                [candidate_field.reshape(-1), stepped_scalars]
            )
            message = summarize_theta_for_logging(param_names, candidate_theta)
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
                field_matrix = candidate_field
                free_scalars = stepped_scalars
                break
        previous_objective = penalized_obj

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        y_field = candidate_field + ((momentum - 1.0) / next_momentum) * (
            candidate_field - field_matrix
        )
        y_scalars = stepped_scalars + ((momentum - 1.0) / next_momentum) * (
            stepped_scalars - free_scalars
        )
        field_matrix = candidate_field
        free_scalars = stepped_scalars
        momentum = next_momentum

    smooth_loss, _, _ = _evaluate_full_field_loss(
        field_matrix,
        context,
        free_scalar_values=free_scalars,
    )
    nuclear_norm = _nuclear_norm(field_matrix)
    normalized_nuclear_norm = nuclear_norm / nuclear_normalizer
    penalized_obj = float(smooth_loss + float(lambda_nuclear) * normalized_nuclear_norm)
    scalar_values = _scalar_values_from_free_vector(free_scalars, context)
    theta = pack_theta(
        {
            "field_matrix": field_matrix,
            "beta": scalar_values["beta"],
            "xi": scalar_values["xi"],
            "eta": scalar_values["eta"],
        },
        artifacts,
        fixed_scalar_params=context.fixed_scalar_params,
    )
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
        np.maximum(np.abs(rng.normal(loc=1.0, scale=0.25, size=int(rank))), 1.0e-3)
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
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    rng = np.random.default_rng(seed)
    if theta_init is None:
        initial = rng.normal(0.0, 0.1, size=len(free_names))
    else:
        initial = np.asarray(theta_init, dtype=float)
    initial = np.asarray(initial, dtype=float)
    history: list[float] = []

    def objective(theta):
        loss, grad = _evaluate_scalar_only_loss(theta, context)
        history.append(float(loss))
        return loss, grad

    if initial.size == 0:
        final_loss, _ = _evaluate_scalar_only_loss(initial, context)
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
    theta_hat = np.asarray(result.x, dtype=float)
    final_loss, _ = _evaluate_scalar_only_loss(theta_hat, context)
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
    """Fit using Riemannian conjugate gradient on the fixed-rank matrix manifold (pymanopt).

    Parameterizes the field as a rank-r matrix via its thin SVD factors (U, sigma, Vt) and
    optimizes directly on the FixedRankEmbedded manifold. Use this mode when the rank is known
    or when manifold geometry is expected to give better convergence than nuclear-norm relaxation.
    """
    if lambda_frobenius < 0.0:
        raise ValueError("lambda_frobenius must be nonnegative.")
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
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

    frobenius_normalizer = _nuclear_norm_normalizer(artifacts)
    frobenius_penalty_normalizer = float(context.outcome_size)

    def loss_and_grad(
        u: np.ndarray,
        singular_values: np.ndarray,
        vt: np.ndarray,
        free_scalar_values: np.ndarray,
    ):
        scalars = _scalar_values_from_free_vector(free_scalar_values, context)
        field_matrix = _fixed_rank_field_matrix(u, singular_values, vt)
        smooth_loss, residual, scalar_gradient = _evaluate_full_field_loss(
            field_matrix,
            context,
            scalar_values=scalars,
        )
        field_gradient = residual / context.outcome_size
        grad_u = (field_gradient @ vt.T) * singular_values[np.newaxis, :]
        grad_s = np.diag(u.T @ field_gradient @ vt.T)
        grad_vt = singular_values[:, np.newaxis] * (u.T @ field_gradient)
        frobenius_norm = float(np.linalg.norm(singular_values))
        normalized_frobenius_norm = frobenius_norm / frobenius_normalizer
        squared_normalized_frobenius_norm = (
            frobenius_norm * frobenius_norm / frobenius_penalty_normalizer
        )
        if lambda_frobenius > 0.0:
            grad_s = grad_s + (
                float(lambda_frobenius) * singular_values / frobenius_penalty_normalizer
            )
        penalized_loss = (
            smooth_loss
            + 0.5 * float(lambda_frobenius) * squared_normalized_frobenius_norm
        )
        return (
            float(penalized_loss),
            float(smooth_loss),
            float(frobenius_norm),
            float(normalized_frobenius_norm),
            float(squared_normalized_frobenius_norm),
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
            _, _, _, _, _, grad_u, grad_s, grad_vt, scalar_gradient = loss_and_grad(
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
            _, _, _, _, _, grad_u, grad_s, grad_vt, _ = loss_and_grad(
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
        min_gradient_norm=1e-5,
        max_cost_evaluations=max(50, int(steps) * 25),
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
        (
            initial_penalized_objective,
            initial_mple_loss,
            _,
            _,
            _,
            *_,
        ) = loss_and_grad(
            field_point[0],
            field_point[1],
            field_point[2],
            initial_scalars,
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
        final_point, final_free_scalars = (
            (result.point[0], np.asarray(result.point[1], dtype=float))
            if free_names
            else (result.point, np.zeros(0, dtype=float))
        )
        (
            final_penalized,
            final_loss,
            frobenius_norm,
            normalized_frobenius_norm,
            squared_normalized_frobenius_norm,
            *_,
        ) = loss_and_grad(
            final_point[0],
            final_point[1],
            final_point[2],
            final_free_scalars,
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
    best_point, best_free_scalars = (
        (best_result.point[0], np.asarray(best_result.point[1], dtype=float))
        if free_names
        else (best_result.point, np.zeros(0, dtype=float))
    )
    (
        final_penalized_objective,
        final_mple_loss,
        final_frobenius_norm,
        final_normalized_frobenius_norm,
        final_squared_normalized_frobenius_norm,
        *_,
    ) = loss_and_grad(
        best_point[0],
        best_point[1],
        best_point[2],
        best_free_scalars,
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
        optimizer_mode=OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
        optimizer="pymanopt_conjugate_gradient",
        lambda_frobenius=float(lambda_frobenius),
        final_penalized_objective=float(final_penalized_objective),
        final_mple_loss=final_mple_loss,
        frobenius_norm=final_frobenius_norm,
        normalized_frobenius_norm=float(final_normalized_frobenius_norm),
        squared_normalized_frobenius_norm=float(
            final_squared_normalized_frobenius_norm
        ),
        frobenius_norm_normalizer=float(frobenius_normalizer),
        frobenius_penalty_normalizer=float(frobenius_penalty_normalizer),
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
    """Fit via alternating gradient updates between U, Vt factors and scalar parameters.

    Parameterizes the field as U @ Vt (no explicit singular values) and alternates
    explicit gradient steps for scalars, U, and V with optional ridge regularization.
    Use this mode as an alternative to manifold optimization when pymanopt convergence is poor.
    Requires latent_rank >= 1.
    """
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")

    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    n_starts = max(1, int(n_starts))
    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    base_theta_init = (
        None if theta_init is None else np.asarray(theta_init, dtype=float)
    )
    outer_iterations = max(1, int(steps))
    inner_gradient_steps = 3

    def evaluate_state(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        free_scalar_values: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray, float, float, np.ndarray]:
        smooth_loss, _, time_gradient, node_gradient, scalar_gradient = (
            _evaluate_factorized_loss(
                time_factors,
                node_factors,
                context,
                free_scalar_values=free_scalar_values,
            )
        )
        u_norm_sq = float(np.sum(np.asarray(time_factors, dtype=float) ** 2))
        v_norm_sq = float(np.sum(np.asarray(node_factors, dtype=float) ** 2))
        ridge_penalty = (
            float(lambda_uv_ridge) * (u_norm_sq + v_norm_sq) / context.outcome_size
        )
        return (
            smooth_loss + ridge_penalty,
            smooth_loss,
            time_gradient
            + (2.0 * float(lambda_uv_ridge) / context.outcome_size)
            * np.asarray(time_factors, dtype=float),
            node_gradient
            + (2.0 * float(lambda_uv_ridge) / context.outcome_size)
            * np.asarray(node_factors, dtype=float),
            float(np.linalg.norm(time_factors, ord="fro")),
            float(np.linalg.norm(node_factors, ord="fro")),
            scalar_gradient,
        )

    def penalized_factor_state_with_offset(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        scalar_offset: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        smooth_loss, _, time_gradient, node_gradient = (
            _evaluate_factorized_loss_with_offset(
                time_factors,
                node_factors,
                scalar_offset,
                context,
            )
        )
        ridge_scale = 2.0 * float(lambda_uv_ridge) / context.outcome_size
        ridge_penalty = (
            float(lambda_uv_ridge)
            * (
                float(np.sum(time_factors * time_factors))
                + float(np.sum(node_factors * node_factors))
            )
            / context.outcome_size
        )
        return (
            smooth_loss + ridge_penalty,
            smooth_loss,
            time_gradient + ridge_scale * time_factors,
            node_gradient + ridge_scale * node_factors,
        )

    def scalar_step_size() -> float:
        if not free_names:
            return 0.0
        feature_columns = {
            "beta": context.beta_feature.reshape(-1),
            "xi": context.interaction_effect_x.reshape(-1),
            "eta": context.prev_x.reshape(-1),
        }
        feature_matrix = np.column_stack([feature_columns[name] for name in free_names])
        lipschitz = (
            float(np.linalg.norm(feature_matrix, ord=2) ** 2) / context.outcome_size
        )
        return 1.0 if lipschitz <= 0.0 else 1.0 / lipschitz

    def factor_step_size(fixed_factors: np.ndarray) -> float:
        lipschitz = (
            float(np.linalg.norm(fixed_factors, ord=2) ** 2)
            + 2.0 * float(lambda_uv_ridge)
        ) / context.outcome_size
        return 1.0 if lipschitz <= 0.0 else 1.0 / lipschitz

    scalar_lr = scalar_step_size()

    def pack_state(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        free_scalar_values: np.ndarray,
    ) -> np.ndarray:
        scalars = _scalar_values_from_free_vector(free_scalar_values, context)
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

    def offset_matrix(free_scalar_values: np.ndarray) -> np.ndarray:
        scalars = _scalar_values_from_free_vector(free_scalar_values, context)
        return (
            float(scalars["beta"]) * context.beta_feature
            + float(scalars["eta"]) * context.prev_x
            + float(scalars["xi"]) * context.interaction_effect_x
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
        (
            initial_penalized_objective,
            initial_mple_loss,
            _,
            _,
            _,
            _,
            _,
        ) = evaluate_state(time_factors, node_factors, free_scalar_values)
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
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    scalar_gradient,
                ) = evaluate_state(time_factors, node_factors, free_scalar_values)
                free_scalar_values = free_scalar_values - scalar_lr * scalar_gradient
                cost_evaluations += 1

            current_offset = offset_matrix(free_scalar_values)
            for _ in range(inner_gradient_steps):
                (
                    _,
                    _,
                    time_gradient,
                    _,
                ) = penalized_factor_state_with_offset(
                    time_factors,
                    node_factors,
                    current_offset,
                )
                time_factors = (
                    time_factors - factor_step_size(node_factors) * time_gradient
                )
                cost_evaluations += 1

                (
                    _,
                    _,
                    _,
                    node_gradient,
                ) = penalized_factor_state_with_offset(
                    time_factors,
                    node_factors,
                    current_offset,
                )
                node_factors = (
                    node_factors - factor_step_size(time_factors) * node_gradient
                )
                cost_evaluations += 1

            (
                penalized_loss,
                smooth_loss,
                _,
                _,
                u_frobenius_norm,
                v_frobenius_norm,
                _,
            ) = evaluate_state(time_factors, node_factors, free_scalar_values)
            penalized_history.append(float(penalized_loss))
            mple_history.append(float(smooth_loss))
            iterations_completed = outer_index + 1

            if verbose_every and outer_index % verbose_every == 0:
                if logger is None:
                    print(
                        f"Alternating low-rank iter {outer_index} | Loss: {smooth_loss:.6f} | Penalized: {penalized_loss:.6f}"
                    )
                else:
                    logger.info(
                        "Alternating low-rank iter %s | Loss: %.6f | Penalized: %.6f",
                        outer_index,
                        smooth_loss,
                        penalized_loss,
                    )

            if len(penalized_history) >= 2:
                improvement = abs(penalized_history[-2] - penalized_history[-1])
                if improvement <= float(tol) * max(1.0, abs(penalized_history[-2])):
                    converged = True
                    message = "CONVERGED: alternating objective tolerance reached"
                    break

        theta_hat = pack_state(time_factors, node_factors, free_scalar_values)
        (
            final_penalized_loss,
            final_smooth_loss,
            _,
            _,
            final_u_norm,
            final_v_norm,
            _,
        ) = evaluate_state(time_factors, node_factors, free_scalar_values)
        final_field_matrix = compose_latent_field_matrix(node_factors, time_factors)
        start_summary = {
            "start_index": start_index,
            "seed": int(seed) + start_index,
            "initialization_kind": "random",
            "initial_mple_loss": initial_mple_loss,
            "initial_penalized_objective": initial_penalized_objective,
            "final_mple_loss": float(final_smooth_loss),
            "final_penalized_objective": float(final_penalized_loss),
            "iterations": int(iterations_completed),
            "cost_evaluations": int(cost_evaluations),
            "success": bool(converged),
            "message": message,
        }
        start_summaries.append(start_summary)
        if float(final_penalized_loss) < best_penalized_objective:
            best_penalized_objective = float(final_penalized_loss)
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
                optimizer_mode=OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
                optimizer="alternating_low_rank",
                lambda_uv_ridge=float(lambda_uv_ridge),
                final_mple_loss=float(final_smooth_loss),
                final_penalized_objective=float(final_penalized_loss),
                u_frobenius_norm=float(final_u_norm),
                v_frobenius_norm=float(final_v_norm),
                effective_rank=float(
                    np.linalg.matrix_rank(np.asarray(final_field_matrix, dtype=float))
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


def _fit_mple_concurrent_low_rank(
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
    """Fit the factorized U/V formulation with SciPy L-BFGS-B.

    Uses the same U/V ridge penalty as alternating_latent_rank, but optimizes all
    packed parameters jointly with a quasi-Newton solver.
    """
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")

    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    n_starts = max(1, int(n_starts))
    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    base_theta_init = (
        None if theta_init is None else np.asarray(theta_init, dtype=float)
    )

    if rank < 1 or rank > min(t_steps, n_nodes):
        raise ValueError(
            f"latent_rank={rank} must lie in [1, {min(t_steps, n_nodes)}] for concurrent low-rank optimization."
        )

    def evaluate_theta(
        theta: np.ndarray,
    ) -> tuple[float, float, np.ndarray, float, float, np.ndarray]:
        theta_parts = unpack_theta(
            theta,
            artifacts,
            fixed_scalar_params=fixed,
        )
        time_factors = np.asarray(theta_parts["time_factors"], dtype=float)
        node_factors = np.asarray(theta_parts["node_factors"], dtype=float)
        smooth_loss, _, time_gradient, node_gradient, scalar_gradient = (
            _evaluate_factorized_loss(
                time_factors,
                node_factors,
                context,
                scalar_values={
                    "beta": theta_parts["beta"],
                    "xi": theta_parts["xi"],
                    "eta": theta_parts["eta"],
                },
            )
        )
        ridge_penalty = (
            float(lambda_uv_ridge)
            * (
                float(np.sum(time_factors * time_factors))
                + float(np.sum(node_factors * node_factors))
            )
            / context.outcome_size
        )
        ridge_scale = 2.0 * float(lambda_uv_ridge) / context.outcome_size
        penalized_loss = smooth_loss + ridge_penalty
        grad = np.concatenate(
            [
                (node_gradient + ridge_scale * node_factors).reshape(-1),
                (time_gradient + ridge_scale * time_factors).reshape(-1),
                scalar_gradient,
            ]
        )
        return (
            float(penalized_loss),
            float(smooth_loss),
            grad,
            float(np.linalg.norm(time_factors, ord="fro")),
            float(np.linalg.norm(node_factors, ord="fro")),
            compose_latent_field_matrix(node_factors, time_factors),
        )

    def random_theta_for_start(start_index: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed) + start_index)
        return pack_theta(
            {
                "time_factors": rng.normal(0.0, 0.1, size=(t_steps, rank)),
                "node_factors": rng.normal(0.0, 0.1, size=(n_nodes, rank)),
                **{
                    name: float(value)
                    for name, value in zip(
                        free_names,
                        rng.normal(0.0, 0.1, size=len(free_names)),
                    )
                },
                **fixed,
            },
            artifacts,
            fixed_scalar_params=fixed,
        )

    best_theta: np.ndarray | None = None
    best_result: OptimizeResult | None = None
    best_penalized_history: list[float] = []
    best_mple_history: list[float] = []
    best_start = 0
    best_penalized_objective = np.inf
    start_summaries: list[dict[str, object]] = []

    for start_index in range(n_starts):
        initialization_kind = "random"
        if base_theta_init is not None and start_index == 0:
            initial_theta = np.asarray(base_theta_init, dtype=float)
            initialization_kind = "theta_init"
        else:
            initial_theta = random_theta_for_start(start_index)

        (
            initial_penalized_objective,
            initial_mple_loss,
            _,
            _,
            _,
            _,
        ) = evaluate_theta(initial_theta)
        mple_history = [initial_mple_loss]
        penalized_history = [initial_penalized_objective]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            (
                penalized_loss,
                smooth_loss,
                grad,
                _,
                _,
                _,
            ) = evaluate_theta(np.asarray(theta, dtype=float))
            mple_history.append(float(smooth_loss))
            penalized_history.append(float(penalized_loss))
            return penalized_loss, grad

        if logger is not None:
            logger.info(
                "Concurrent low-rank start %s/%s | seed=%s | initial_loss=%.6f | initial_penalized=%.6f",
                start_index + 1,
                n_starts,
                int(seed) + start_index,
                initial_mple_loss,
                initial_penalized_objective,
            )

        result = minimize(
            objective,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(steps), "gtol": float(tol)},
        )
        theta_hat = np.asarray(result.x, dtype=float)
        (
            final_penalized_loss,
            final_smooth_loss,
            _,
            final_u_norm,
            final_v_norm,
            final_field_matrix,
        ) = evaluate_theta(theta_hat)
        if not penalized_history or penalized_history[-1] != final_penalized_loss:
            penalized_history.append(float(final_penalized_loss))
            mple_history.append(float(final_smooth_loss))
        start_summary = {
            "start_index": start_index,
            "seed": int(seed) + start_index,
            "initialization_kind": initialization_kind,
            "initial_mple_loss": initial_mple_loss,
            "initial_penalized_objective": initial_penalized_objective,
            "final_mple_loss": float(final_smooth_loss),
            "final_penalized_objective": float(final_penalized_loss),
            "iterations": int(getattr(result, "nit", 0)),
            "cost_evaluations": int(getattr(result, "nfev", len(penalized_history))),
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
        }
        start_summaries.append(start_summary)

        if verbose_every and logger is not None:
            logger.info(
                "Concurrent low-rank start %s finished | final_loss=%.6f | final_penalized=%.6f | nit=%s",
                start_index + 1,
                final_smooth_loss,
                final_penalized_loss,
                int(getattr(result, "nit", 0)),
            )

        if float(final_penalized_loss) < best_penalized_objective:
            best_penalized_objective = float(final_penalized_loss)
            best_start = start_index
            best_theta = theta_hat
            best_mple_history = list(mple_history)
            best_penalized_history = list(penalized_history)
            best_result = OptimizeResult(
                x=theta_hat,
                success=bool(getattr(result, "success", False)),
                message=str(getattr(result, "message", "")),
                nit=int(getattr(result, "nit", 0)),
                nfev=int(getattr(result, "nfev", len(penalized_history))),
                iterations=int(getattr(result, "nit", 0)),
                cost_evaluations=int(getattr(result, "nfev", len(penalized_history))),
                optimizer_mode=OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
                optimizer="scipy_lbfgsb_low_rank",
                lambda_uv_ridge=float(lambda_uv_ridge),
                final_mple_loss=float(final_smooth_loss),
                final_penalized_objective=float(final_penalized_loss),
                u_frobenius_norm=float(final_u_norm),
                v_frobenius_norm=float(final_v_norm),
                effective_rank=float(
                    np.linalg.matrix_rank(np.asarray(final_field_matrix, dtype=float))
                ),
                mple_history=list(mple_history),
                penalized_history=list(penalized_history),
                best_start=int(start_index),
                n_starts=int(n_starts),
                start_summaries=start_summaries,
            )

    if best_theta is None or best_result is None:
        raise RuntimeError(
            "Concurrent low-rank optimizer did not produce a candidate solution."
        )
    best_result["start_summaries"] = start_summaries
    best_result["best_start"] = int(best_start)
    best_result["n_starts"] = int(n_starts)
    return best_theta, best_mple_history, best_result


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
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NO_EXTERNAL_FIELD:
        theta_hat, history, result = _fit_zero_rank_unconstrained(
            x,
            z,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            tol=tol,
            theta_init=theta_init,
            fixed_scalar_params=fixed_scalar_params,
        )
        result["optimizer_mode"] = OPTIMIZER_MODE_NO_EXTERNAL_FIELD
        result["optimizer"] = "scipy_bfgs_no_external_field"
        result["best_start"] = 0
        result["n_starts"] = 1
        result["iterations"] = int(getattr(result, "nit", 0))
        result["cost_evaluations"] = int(getattr(result, "nfev", len(history)))
        result["final_mple_loss"] = float(history[-1])
        result["final_penalized_objective"] = float(history[-1])
        result["mple_history"] = list(history)
        result["penalized_history"] = list(history)
        result["effective_rank"] = 0.0
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
    if artifacts.optimizer_mode == OPTIMIZER_MODE_ALTERNATING_LATENT_RANK:
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
    if artifacts.optimizer_mode == OPTIMIZER_MODE_CONCURRENT_LATENT_RANK:
        return _fit_mple_concurrent_low_rank(
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
    if artifacts.optimizer_mode == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD:
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
            "name": "estimated_field_max_abs_entry",
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
                    "name": "true_field_max_abs_entry",
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
            io_path(Path(data_folder) / "estimated_interaction_matrix_sparse.npz"),
            estimated_interaction,
        )
    else:
        np.save(
            io_path(Path(data_folder) / "estimated_interaction_matrix.npy"),
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
            io_path(Path(data_folder) / "true_interaction_matrix_sparse.npz"),
            true_interaction,
        )
    else:
        np.save(
            io_path(Path(data_folder) / "true_interaction_matrix.npy"),
            true_interaction,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit active conditional-model parameters with MPLE."
    )
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    log_file = args.log_file or str(Path(args.data_folder) / "mple.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Loading data...")
    config_path = Path(args.data_folder) / "fit_realized_config.yaml"
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
    lambda_nuclear = float(config.global_params.get("lambda_nuclear", 0.0))
    lambda_frobenius = float(config.global_params.get("lambda_frobenius", 0.0))
    lambda_uv_ridge = float(config.global_params.get("lambda_uv_ridge", 0.0))
    optimizer_params = (
        config.optimizer_params
        if "optimizer_params" in config
        else OmegaConf.create({})
    )
    steps = int(optimizer_params.get("steps", 10000))
    tol = float(optimizer_params.get("tol", 1e-9))
    seed = int(optimizer_params.get("seed", 0))
    n_starts = int(optimizer_params.get("n_starts", 1))
    proximal_lr = float(optimizer_params.get("proximal_lr", 1.0))
    input_artifacts = config.input_artifacts
    model_artifact_dir = Path(str(input_artifacts.model_artifact_dir))
    truth_artifact_dir = Path(str(input_artifacts.truth_artifact_dir))
    panel_path = Path(str(input_artifacts.panel_path))
    x0_path = Path(str(input_artifacts.x0_path))
    logger.info("Using panel artifact: %s", panel_path)
    logger.info("Using x_0 artifact: %s", x0_path)
    logger.info("Using fit config: %s", config_path)
    logger.info("Using model artifact directory: %s", model_artifact_dir)
    logger.info("Using truth artifact directory: %s", truth_artifact_dir)
    # Load Data
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
    elif result.get("optimizer_mode") == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD:
        metrics.update(
            {
                "penalized_objective": float(result["final_penalized_objective"]),
                "mple_loss_without_penalty": float(result["final_mple_loss"]),
                "lambda_frobenius": float(result["lambda_frobenius"]),
                "frobenius_norm": float(result["frobenius_norm"]),
                "normalized_frobenius_norm": float(result["normalized_frobenius_norm"]),
                "squared_normalized_frobenius_norm": float(
                    result["squared_normalized_frobenius_norm"]
                ),
                "frobenius_norm_normalizer": float(result["frobenius_norm_normalizer"]),
                "frobenius_penalty_normalizer": float(
                    result["frobenius_penalty_normalizer"]
                ),
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
                "  squared_normalized_frobenius_norm: %.6f",
                metrics["squared_normalized_frobenius_norm"],
            )
            logger.info(
                "  frobenius_norm_normalizer: %.6f",
                metrics["frobenius_norm_normalizer"],
            )
            logger.info(
                "  frobenius_penalty_normalizer: %.6f",
                metrics["frobenius_penalty_normalizer"],
            )
    elif result.get("optimizer_mode") in {
        OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
        OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    }:
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
        optimizer_title = (
            "Concurrent low-rank optimizer diagnostics:"
            if result.get("optimizer_mode") == OPTIMIZER_MODE_CONCURRENT_LATENT_RANK
            else "Alternating low-rank optimizer diagnostics:"
        )
        logger.info(optimizer_title)
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
