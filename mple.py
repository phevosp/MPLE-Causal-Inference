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

from model_utils import (
    FIELD_MODE_NUCLEAR_NORM,
    ModelArtifacts,
    build_fit_model_artifacts,
    compose_field_matrix_from_theta,
    compose_interaction_matrix,
    free_scalar_parameter_names,
    interaction_effect,
    interaction_matrix_infinity_norm,
    intervention_model_enabled,
    latent_field_bound_norm,
    load_model_artifacts,
    pack_theta,
    parameter_names,
    project_latent_field,
    save_field_artifacts,
    scalar_parameter_names,
    summarize_theta_for_logging,
    unpack_theta,
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
    fit_intervention_model: bool,
    bound_B: float | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    if bound_B is not None:
        for key in scalar_parameter_names(fit_intervention_model):
            value = theta_parts.get(key, None)
            if value is None:
                continue
            theta_parts[key] = float(np.clip(float(value), -bound_B, bound_B))
        gamma_inf = interaction_matrix_infinity_norm(artifacts.gamma_matrix)
        if gamma_inf > 1e-12:
            xi_interaction_bound = float(bound_B) / float(gamma_inf)
            effective_xi_bound = min(float(bound_B), xi_interaction_bound)
            theta_parts["xi"] = float(
                np.clip(
                    float(theta_parts["xi"]), -effective_xi_bound, effective_xi_bound
                )
            )
        if artifacts.field_mode == FIELD_MODE_NUCLEAR_NORM:
            theta_parts["field_matrix"] = np.clip(
                np.asarray(theta_parts["field_matrix"], dtype=float),
                -bound_B,
                bound_B,
            )
        else:
            node_factors, time_factors = project_latent_field(
                theta_parts["node_factors"],
                theta_parts["time_factors"],
                bound_B,
            )
            theta_parts["node_factors"] = node_factors
            theta_parts["time_factors"] = time_factors
    return pack_theta(
        theta_parts,
        artifacts,
        fit_intervention_model,
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
    z_0: np.ndarray,
    s: int,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fit_intervention_model: bool = True,
    beta_mask_pre_intervention: bool = False,
    fixed_scalar_params: dict[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    if x.shape[0] != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([z_0, z[:-1, :]])
    beta_feature = np.asarray(z, dtype=float)
    if beta_mask_pre_intervention:
        beta_mask = np.ones_like(beta_feature)
        beta_mask[:s, :] = 0.0
        beta_feature = beta_feature * beta_mask
    field_matrix = compose_field_matrix_from_theta(theta_parts, artifacts)
    h_x = (
        field_matrix
        + theta_parts["beta"] * beta_feature
        + theta_parts["eta"] * prev_x
        + theta_parts["xi"] * interaction_effect_x
    )
    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x

    if fit_intervention_model:
        h_z = theta_parts["zeta"] * prev_x + theta_parts["psi"] * prev_z
        mask = np.ones_like(z)
        mask[:s, :] = 0
        outcome_size = x.size
        intervention_size = mask.sum()
        gradient_denominator = outcome_size + intervention_size
        total_loss = (
            loss_x.sum() + ((np.logaddexp(h_z, -h_z) - z * h_z) * mask).sum()
        ) / gradient_denominator
        res_z = (np.tanh(h_z) - z) * mask
        zeta_grad = float((res_z * prev_x).sum())
        psi_grad = float((res_z * prev_z).sum())
    else:
        outcome_size = x.size
        gradient_denominator = outcome_size
        total_loss = loss_x.sum() / outcome_size
        zeta_grad = 0.0
        psi_grad = 0.0

    if artifacts.field_mode == FIELD_MODE_NUCLEAR_NORM:
        field_grad = res_x.reshape(-1)
    else:
        node_grad = res_x.T @ theta_parts["time_factors"]
        time_grad = res_x @ theta_parts["node_factors"]
        field_grad = np.concatenate([node_grad.reshape(-1), time_grad.reshape(-1)])

    scalar_grad_lookup = {
        "beta": float((res_x * beta_feature).sum()) / gradient_denominator,
        "xi": float((res_x * interaction_effect_x).sum()) / gradient_denominator,
        "eta": float((res_x * prev_x).sum()) / gradient_denominator,
        "zeta": float(zeta_grad) / gradient_denominator,
        "psi": float(psi_grad) / gradient_denominator,
    }
    grad_parts = [field_grad / gradient_denominator]
    for name in free_scalar_parameter_names(
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    ):
        grad_parts.append(np.array([scalar_grad_lookup[name]], dtype=float))
    return float(total_loss), np.concatenate(grad_parts)


def _pseudo_loss(
    x: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    s: int,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fit_intervention_model: bool,
    beta_mask_pre_intervention: bool,
    fixed_scalar_params: dict[str, float] | None,
) -> float:
    loss, _ = pseudo_nll(
        x,
        z,
        theta,
        x_0,
        z_0,
        s,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fit_intervention_model=fit_intervention_model,
        beta_mask_pre_intervention=beta_mask_pre_intervention,
        fixed_scalar_params=fixed_scalar_params,
    )
    return float(loss)


def _nuclear_norm(field_matrix: np.ndarray) -> float:
    if field_matrix.size == 0:
        return 0.0
    return float(np.linalg.svd(np.asarray(field_matrix, dtype=float), compute_uv=False).sum())


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
    fit_intervention_model: bool,
    bound_B: float | None,
    fixed_scalar_params: dict[str, float] | None,
    threshold: float,
) -> np.ndarray:
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fit_intervention_model,
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
            fit_intervention_model,
            fixed_scalar_params=fixed_scalar_params,
        ),
        artifacts,
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed_scalar_params,
    )


def _penalized_nuclear_objective(
    theta: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    s: int,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fit_intervention_model: bool,
    beta_mask_pre_intervention: bool,
    fixed_scalar_params: dict[str, float] | None,
    lambda_nuclear: float,
) -> tuple[float, float, float]:
    smooth_loss = _pseudo_loss(
        x,
        z,
        theta,
        x_0,
        z_0,
        s,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fit_intervention_model=fit_intervention_model,
        beta_mask_pre_intervention=beta_mask_pre_intervention,
        fixed_scalar_params=fixed_scalar_params,
    )
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    nuclear_norm = _nuclear_norm(np.asarray(theta_parts["field_matrix"], dtype=float))
    return (
        float(smooth_loss + float(lambda_nuclear) * nuclear_norm),
        float(smooth_loss),
        float(nuclear_norm),
    )


def _fit_mple_nuclear_norm(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    s: int,
    param_names: list[str],
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    seed: int,
    verbose_every: int,
    tol: float,
    logger,
    theta_init,
    fit_intervention_model: bool,
    bound_B: float | None,
    beta_mask_pre_intervention: bool,
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
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed_scalar_params,
    )
    y = theta.copy()
    momentum = 1.0
    history: list[float] = []
    penalized_history: list[float] = []
    converged = False
    previous_objective = np.inf

    for iteration in range(max(1, int(steps))):
        loss_y, grad_y = pseudo_nll(
            x,
            z,
            y,
            x_0,
            z_0,
            s,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fit_intervention_model=fit_intervention_model,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
        )
        stepped = y - float(proximal_lr) * grad_y
        candidate = _prox_nuclear_theta(
            stepped,
            artifacts,
            fit_intervention_model,
            bound_B,
            fixed_scalar_params,
            threshold=float(proximal_lr) * float(lambda_nuclear),
        )
        penalized_obj, smooth_loss, nuclear_norm = _penalized_nuclear_objective(
            candidate,
            x,
            z,
            x_0,
            z_0,
            s,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fit_intervention_model=fit_intervention_model,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
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
                    "Nuclear prox iter %s | Loss: %.6f | Penalized: %.6f | nuclear_norm: %.6f",
                    iteration,
                    smooth_loss,
                    penalized_obj,
                    nuclear_norm,
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
            fit_intervention_model,
            bound_B,
            fixed_scalar_params=fixed_scalar_params,
        )
        theta = candidate
        momentum = next_momentum

    theta = _canonicalize_theta(
        theta,
        artifacts,
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed_scalar_params,
    )
    penalized_obj, smooth_loss, nuclear_norm = _penalized_nuclear_objective(
        theta,
        x,
        z,
        x_0,
        z_0,
        s,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        fit_intervention_model=fit_intervention_model,
        beta_mask_pre_intervention=beta_mask_pre_intervention,
        fixed_scalar_params=fixed_scalar_params,
        lambda_nuclear=lambda_nuclear,
    )
    theta_parts = unpack_theta(
        theta,
        artifacts,
        fit_intervention_model,
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
        field_mode=FIELD_MODE_NUCLEAR_NORM,
        lambda_nuclear=float(lambda_nuclear),
        final_penalized_objective=float(penalized_obj),
        final_mple_loss=float(smooth_loss),
        nuclear_norm=float(nuclear_norm),
        effective_rank=float(np.linalg.matrix_rank(field_matrix)),
        proximal_iterations=int(len(history)),
        penalized_history=penalized_history,
    )
    if not history:
        history.append(float(smooth_loss))
    return theta, history, result


def _torch_adam_stage(
    x: np.ndarray,
    z: np.ndarray,
    theta_init: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    s: int,
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    fit_intervention_model: bool,
    bound_B: float | None,
    beta_mask_pre_intervention: bool,
    fixed_scalar_params: dict[str, float] | None,
    steps: int,
    lr: float,
    device: str,
    logger=None,
    verbose_every: int = 100,
    start_index: int = 0,
) -> tuple[np.ndarray, list[float]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for optimizer_params.adam_steps > 0."
        ) from exc

    if steps <= 0:
        return theta_init, []
    if lr <= 0.0:
        raise ValueError("adam_lr must be positive when adam_steps > 0.")

    fixed = validate_fixed_scalar_params(
        fixed_scalar_params, fit_intervention_model=fit_intervention_model
    )
    scalar_names = scalar_parameter_names(fit_intervention_model)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    t_steps = int(artifacts.t_steps)
    latent_rank = int(artifacts.latent_rank)
    n_u = n_nodes * latent_rank
    n_v = t_steps * latent_rank
    outcome_size = float(np.asarray(x).size)
    prev_x_np = np.vstack([x_0, x[:-1, :]])
    prev_z_np = np.vstack([z_0, z[:-1, :]])
    beta_feature_np = np.asarray(z, dtype=float).copy()
    if beta_mask_pre_intervention:
        beta_feature_np[:s, :] = 0.0
    intervention_mask_np = np.ones_like(z, dtype=float)
    intervention_mask_np[:s, :] = 0.0
    intervention_size = float(intervention_mask_np.sum())

    dtype = torch.float64
    x_t = torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)
    z_t = torch.as_tensor(np.asarray(z, dtype=float), dtype=dtype, device=device)
    prev_x_t = torch.as_tensor(prev_x_np, dtype=dtype, device=device)
    prev_z_t = torch.as_tensor(prev_z_np, dtype=dtype, device=device)
    beta_feature_t = torch.as_tensor(beta_feature_np, dtype=dtype, device=device)
    interaction_t = torch.as_tensor(
        np.asarray(interaction_effect_x, dtype=float), dtype=dtype, device=device
    )
    intervention_mask_t = torch.as_tensor(
        intervention_mask_np, dtype=dtype, device=device
    )
    theta = torch.nn.Parameter(
        torch.as_tensor(np.asarray(theta_init, dtype=float), dtype=dtype, device=device)
    )
    optimizer = torch.optim.Adam([theta], lr=float(lr))
    gamma_inf = interaction_matrix_infinity_norm(artifacts.gamma_matrix)
    xi_bound = None
    if bound_B is not None:
        xi_bound = float(bound_B)
        if gamma_inf > 1e-12:
            xi_bound = min(float(bound_B), float(bound_B) / float(gamma_inf))

    def constrained_parts(raw_theta):
        node_factors = raw_theta[:n_u].reshape(n_nodes, latent_rank)
        time_factors = raw_theta[n_u : n_u + n_v].reshape(t_steps, latent_rank)
        cursor = n_u + n_v
        scalars: dict[str, Any] = {}
        for name in scalar_names:
            if name in fixed:
                scalars[name] = torch.as_tensor(
                    float(fixed[name]), dtype=dtype, device=device
                )
            else:
                scalars[name] = raw_theta[cursor]
                cursor += 1
        if bound_B is not None:
            scalar_bound = float(bound_B)
            for name in scalar_names:
                scalars[name] = torch.clamp(
                    scalars[name], -scalar_bound, scalar_bound
                )
            if xi_bound is not None:
                scalars["xi"] = torch.clamp(scalars["xi"], -xi_bound, xi_bound)
            if latent_rank > 0:
                field = time_factors @ node_factors.T
                field_norm = torch.max(torch.abs(field))
                bound = torch.as_tensor(float(bound_B), dtype=dtype, device=device)
                scale = torch.where(
                    field_norm > bound,
                    torch.sqrt(bound / torch.clamp(field_norm, min=1.0e-12)),
                    torch.ones((), dtype=dtype, device=device),
                )
                node_factors = node_factors * scale
                time_factors = time_factors * scale
        return node_factors, time_factors, scalars

    def forward_loss(raw_theta):
        node_factors, time_factors, scalars = constrained_parts(raw_theta)
        if latent_rank > 0:
            field_matrix = time_factors @ node_factors.T
        else:
            field_matrix = torch.zeros(
                (t_steps, n_nodes), dtype=dtype, device=device
            )
        h_x = (
            field_matrix
            + scalars["beta"] * beta_feature_t
            + scalars["eta"] * prev_x_t
            + scalars["xi"] * interaction_t
        )
        loss_x = torch.logaddexp(h_x, -h_x) - x_t * h_x
        if fit_intervention_model and intervention_size > 0.0:
            h_z = scalars["zeta"] * prev_x_t + scalars["psi"] * prev_z_t
            loss_z = (
                torch.logaddexp(h_z, -h_z) - z_t * h_z
            ) * intervention_mask_t
            return (loss_x.sum() + loss_z.sum()) / (
                outcome_size + intervention_size
            )
        return loss_x.sum() / outcome_size

    history: list[float] = []
    for step_index in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(theta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = _canonicalize_theta(
                theta.detach().cpu().numpy(),
                artifacts,
                fit_intervention_model,
                bound_B,
                fixed_scalar_params=fixed,
            )
            theta.copy_(torch.as_tensor(projected, dtype=dtype, device=device))
        loss_value = float(loss.detach().cpu().item())
        history.append(loss_value)
        if verbose_every and step_index % verbose_every == 0:
            if logger is None:
                print(
                    f"Start {start_index + 1} Adam step {step_index} "
                    f"| Loss: {loss_value:.6f}"
                )
            else:
                logger.info(
                    "Start %s Adam step %s | Loss: %.6f",
                    start_index + 1,
                    step_index,
                    loss_value,
                )
    theta_hat = _canonicalize_theta(
        theta.detach().cpu().numpy(),
        artifacts,
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed,
    )
    return theta_hat, history


def _run_lbfgs_stage(
    x: np.ndarray,
    z: np.ndarray,
    theta_init: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    s: int,
    param_names: list[str],
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    steps: int,
    tol: float,
    logger,
    verbose_every: int,
    fit_intervention_model: bool,
    bound_B: float | None,
    beta_mask_pre_intervention: bool,
    fixed_scalar_params: dict[str, float] | None,
    start_index: int,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    history: list[float] = []
    eval_count = [0]

    def objective(theta):
        constrained_theta = _canonicalize_theta(
            theta,
            artifacts,
            fit_intervention_model,
            bound_B,
            fixed_scalar_params=fixed_scalar_params,
        )
        loss, grad = pseudo_nll(
            x,
            z,
            constrained_theta,
            x_0,
            z_0,
            s,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fit_intervention_model=fit_intervention_model,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
        )
        history.append(loss)
        if verbose_every and eval_count[0] % verbose_every == 0:
            message = summarize_theta_for_logging(param_names, constrained_theta)
            if logger is None:
                print(
                    f"Start {start_index + 1} L-BFGS eval {eval_count[0]} "
                    f"| Loss: {loss:.6f}"
                )
                print(message)
            else:
                logger.info(
                    "Start %s L-BFGS eval %s | Loss: %.6f",
                    start_index + 1,
                    eval_count[0],
                    loss,
                )
                logger.info(message)
        eval_count[0] += 1
        return loss, grad

    result = minimize(
        objective,
        theta_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": steps, "ftol": tol, "gtol": tol},
    )
    theta_hat = _canonicalize_theta(
        result.x,
        artifacts,
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed_scalar_params,
    )
    return theta_hat, history, result


def fit_mple(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
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
    fit_intervention_model: bool = True,
    bound_B: float | None = None,
    beta_mask_pre_intervention: bool = False,
    fixed_scalar_params: dict[str, float] | None = None,
    n_starts: int = 1,
    adam_steps: int = 0,
    adam_lr: float = 1.0e-2,
    adam_device: str = "cpu",
    lambda_nuclear: float = 0.0,
    proximal_lr: float = 1.0,
):
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError("x and z must both have shape (T, N).")

    t_steps = x.shape[0]
    if t_steps != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    if artifacts.field_mode == FIELD_MODE_NUCLEAR_NORM:
        return _fit_mple_nuclear_norm(
            x,
            z,
            x_0=x_0,
            z_0=z_0,
            s=s,
            param_names=param_names,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            seed=seed,
            verbose_every=verbose_every,
            tol=tol,
            logger=logger,
            theta_init=theta_init,
            fit_intervention_model=fit_intervention_model,
            bound_B=bound_B,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
            lambda_nuclear=lambda_nuclear,
            proximal_lr=proximal_lr,
        )
    n_starts = max(1, int(n_starts))
    adam_steps = max(0, int(adam_steps))
    adam_lr = float(adam_lr)
    base_theta_init = None if theta_init is None else np.asarray(theta_init, dtype=float)

    best_theta: np.ndarray | None = None
    best_history: list[float] = []
    best_result: OptimizeResult | None = None
    best_loss = np.inf
    start_summaries: list[dict[str, object]] = []

    for start_index in range(n_starts):
        start_seed = int(seed) + start_index
        rng = np.random.default_rng(start_seed)
        raw_init = (
            base_theta_init.copy()
            if base_theta_init is not None and start_index == 0
            else rng.normal(0, 0.1, size=len(param_names))
        )
        start_theta = _canonicalize_theta(
            raw_init,
            artifacts,
            fit_intervention_model,
            bound_B,
            fixed_scalar_params=fixed_scalar_params,
        )
        initial_loss = _pseudo_loss(
            x,
            z,
            start_theta,
            x_0,
            z_0,
            s,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fit_intervention_model=fit_intervention_model,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
        )
        if logger is not None:
            logger.info(
                "Optimizer start %s/%s | seed=%s | initial_loss=%.6f",
                start_index + 1,
                n_starts,
                start_seed,
                initial_loss,
            )

        adam_history: list[float] = []
        adam_final_loss = initial_loss
        if adam_steps > 0:
            start_theta, adam_history = _torch_adam_stage(
                x,
                z,
                start_theta,
                x_0,
                z_0,
                s,
                artifacts=artifacts,
                interaction_effect_x=interaction_effect_x,
                fit_intervention_model=fit_intervention_model,
                bound_B=bound_B,
                beta_mask_pre_intervention=beta_mask_pre_intervention,
                fixed_scalar_params=fixed_scalar_params,
                steps=adam_steps,
                lr=adam_lr,
                device=str(adam_device),
                logger=logger,
                verbose_every=verbose_every,
                start_index=start_index,
            )
            adam_final_loss = _pseudo_loss(
                x,
                z,
                start_theta,
                x_0,
                z_0,
                s,
                artifacts=artifacts,
                interaction_effect_x=interaction_effect_x,
                fit_intervention_model=fit_intervention_model,
                beta_mask_pre_intervention=beta_mask_pre_intervention,
                fixed_scalar_params=fixed_scalar_params,
            )

        theta_hat, lbfgs_history, result = _run_lbfgs_stage(
            x,
            z,
            start_theta,
            x_0,
            z_0,
            s,
            param_names=param_names,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            steps=steps,
            tol=tol,
            logger=logger,
            verbose_every=verbose_every,
            fit_intervention_model=fit_intervention_model,
            bound_B=bound_B,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
            start_index=start_index,
        )
        final_loss = _pseudo_loss(
            x,
            z,
            theta_hat,
            x_0,
            z_0,
            s,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fit_intervention_model=fit_intervention_model,
            beta_mask_pre_intervention=beta_mask_pre_intervention,
            fixed_scalar_params=fixed_scalar_params,
        )
        run_history = [initial_loss, *adam_history, *lbfgs_history, final_loss]
        start_summary = {
            "start_index": start_index,
            "seed": start_seed,
            "initial_loss": initial_loss,
            "adam_steps": adam_steps,
            "adam_final_loss": adam_final_loss,
            "lbfgs_final_loss": final_loss,
            "lbfgs_nit": int(getattr(result, "nit", 0)),
            "lbfgs_nfev": int(getattr(result, "nfev", 0)),
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
        }
        start_summaries.append(start_summary)
        if logger is not None:
            logger.info(
                "Optimizer start %s/%s complete | final_loss=%.6f | status=%s",
                start_index + 1,
                n_starts,
                final_loss,
                result.message,
            )
        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = theta_hat
            best_history = run_history
            best_result = result

    if best_theta is None or best_result is None:
        raise RuntimeError("MPLE optimizer did not produce a candidate solution.")
    best_start = min(
        range(len(start_summaries)),
        key=lambda index: float(start_summaries[index]["lbfgs_final_loss"]),
    )
    best_result["best_start"] = int(best_start)
    best_result["n_starts"] = int(n_starts)
    best_result["adam_steps"] = int(adam_steps)
    best_result["adam_lr"] = float(adam_lr)
    best_result["start_summaries"] = start_summaries
    best_result["message"] = (
        f"{best_result.message} | best_start={best_start + 1}/{n_starts}"
    )
    return best_theta, best_history, best_result


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def scalar_summary_rows(
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    fit_intervention_model: bool,
    scalar_truths: dict[str, float] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    rows: list[dict[str, object]] = []
    for name in scalar_parameter_names(fit_intervention_model):
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


def load_truth_context(
    truth_artifact_dir: str | Path,
    fit_intervention_model: bool,
) -> dict[str, object] | None:
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
    if fit_intervention_model:
        scalar_truths["zeta"] = float(truth_config.estimation_params.zeta)
        scalar_truths["psi"] = float(truth_config.estimation_params.psi)
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
    fit_intervention_model: bool,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, float]:
    if truth_context is None or truth_context.get("field_matrix") is None:
        return {}
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fit_intervention_model=fit_intervention_model,
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
    fit_intervention_model: bool,
    bound_B: float | None,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fit_intervention_model=fit_intervention_model,
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
    if bound_B is not None:
        rows.append(
            {
                "category": "latent_diagnostic",
                "name": "bound_B",
                "estimate": float(bound_B),
                "true": None,
                "squared_error": None,
            }
        )
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
    fit_intervention_model: bool,
    bound_B: float | None,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
):
    csv_path = Path(f"{summary_stem}.csv")
    md_path = Path(f"{summary_stem}.md")
    rows = scalar_summary_rows(
        est_theta,
        artifacts,
        fit_intervention_model=fit_intervention_model,
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
            fit_intervention_model=fit_intervention_model,
            bound_B=bound_B,
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
        "initial_loss",
        "adam_steps",
        "adam_final_loss",
        "lbfgs_final_loss",
        "lbfgs_nit",
        "lbfgs_nfev",
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
    fit_intervention_model: bool,
    bound_B: float | None,
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
        fit_intervention_model=fit_intervention_model,
        bound_B=bound_B,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    ):
        logger.info("  %s: %s", row["name"], _fmt(row["estimate"]))


def save_estimated_artifacts(
    data_folder: str | Path,
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> None:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fit_intervention_model,
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
        zeta=float(est_parts["zeta"]),
        psi=float(est_parts["psi"]),
        fit_intervention_model=fit_intervention_model,
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
        np.save(_io_path(Path(data_folder) / "true_interaction_matrix.npy"), true_interaction)


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
    parser.add_argument("--z0_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_starts", type=int, default=None)
    parser.add_argument("--adam_steps", type=int, default=None)
    parser.add_argument("--adam_lr", type=float, default=None)
    parser.add_argument("--adam_device", type=str, default=None)
    parser.add_argument("--lambda_nuclear", type=float, default=None)
    parser.add_argument("--proximal_lr", type=float, default=None)
    parser.add_argument("--outcome_only", action="store_true")
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
    fit_intervention_model = (
        intervention_model_enabled(config) and not args.outcome_only
    )
    fixed_scalar_params = validate_fixed_scalar_params(
        (
            OmegaConf.to_container(
                config.estimation_params.get("fixed_scalar_params", {}),
                resolve=True,
            )
            if "estimation_params" in config
            else {}
        ),
        fit_intervention_model=fit_intervention_model,
    )
    beta_mask_pre_intervention = (
        bool(config.estimation_params.get("beta_mask_pre_intervention", False))
        if "estimation_params" in config
        else False
    )
    bound_B = float(config.global_params.B) if "B" in config.global_params else None
    lambda_nuclear = float(
        args.lambda_nuclear
        if args.lambda_nuclear is not None
        else config.global_params.get("lambda_nuclear", 0.0)
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
    adam_steps = int(
        args.adam_steps
        if args.adam_steps is not None
        else optimizer_params.get("adam_steps", 0)
    )
    adam_lr = float(
        args.adam_lr
        if args.adam_lr is not None
        else optimizer_params.get("adam_lr", 1.0e-2)
    )
    adam_device = str(
        args.adam_device
        if args.adam_device is not None
        else optimizer_params.get("adam_device", "cpu")
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
    z0_path = Path(args.z0_path) if args.z0_path else truth_artifact_dir / "z_0.npy"
    logger.info("Using panel artifact: %s", panel_path)
    logger.info("Using x_0 artifact: %s", x0_path)
    logger.info("Using z_0 artifact: %s", z0_path)
    logger.info("Using fit config: %s", config_path)
    logger.info("Using model artifact directory: %s", model_artifact_dir)
    logger.info("Using truth artifact directory: %s", truth_artifact_dir)
    x_0 = np.load(x0_path)
    z_0 = np.load(z0_path) if z0_path.exists() else np.zeros_like(x_0)
    panel = load_panel_artifact(panel_path)
    x = panel["x"]
    z = panel["z"]

    gamma_matrix = load_gamma_matrix(model_artifact_dir)
    artifacts = build_fit_model_artifacts(config, gamma_matrix)
    if bound_B is not None and fixed_scalar_params:
        for name in list(fixed_scalar_params):
            fixed_scalar_params[name] = float(
                np.clip(float(fixed_scalar_params[name]), -bound_B, bound_B)
            )
        gamma_inf = interaction_matrix_infinity_norm(artifacts.gamma_matrix)
        if "xi" in fixed_scalar_params and gamma_inf > 1e-12:
            fixed_scalar_params["xi"] = float(
                np.clip(
                    float(fixed_scalar_params["xi"]),
                    -min(bound_B, bound_B / gamma_inf),
                    min(bound_B, bound_B / gamma_inf),
                )
            )
    param_keys = parameter_names(
        artifacts,
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    truth_context = load_truth_context(truth_artifact_dir, fit_intervention_model)
    interaction_effect_x = interaction_effect(x, artifacts.gamma_matrix)
    logger.info("Configured field mode: %s", artifacts.field_mode)
    logger.info("Configured latent rank: %s", artifacts.latent_rank)
    logger.info("Using a fixed known graph with scalar xi.")
    logger.info("Intervention-process model enabled: %s", fit_intervention_model)
    logger.info("Global temperature bound B active: %s", bound_B is not None)
    if bound_B is not None:
        gamma_inf = interaction_matrix_infinity_norm(artifacts.gamma_matrix)
        if gamma_inf > 1e-12:
            logger.info(
                "Effective xi bound from interaction constraint: %.6f",
                min(bound_B, bound_B / gamma_inf),
            )
        else:
            logger.info("Effective xi bound from interaction constraint: %.6f", bound_B)
    logger.info("Beta mask pre-intervention enabled: %s", beta_mask_pre_intervention)
    logger.info("Fixed scalar parameters: %s", fixed_scalar_params or {})
    logger.info(
        "Optimizer settings: n_starts=%s, lbfgs_steps=%s, tol=%s, "
        "adam_steps=%s, adam_lr=%s, adam_device=%s, seed=%s, "
        "lambda_nuclear=%s, proximal_lr=%s",
        n_starts,
        steps,
        tol,
        adam_steps,
        adam_lr,
        adam_device,
        seed,
        lambda_nuclear,
        proximal_lr,
    )

    params_hat, loss_history, result = fit_mple(
        x,
        z,
        x_0=x_0,
        z_0=z_0,
        s=int(config.global_params.s),
        param_names=param_keys,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        steps=steps,
        tol=tol,
        seed=seed,
        logger=logger,
        fit_intervention_model=fit_intervention_model,
        bound_B=bound_B,
        beta_mask_pre_intervention=beta_mask_pre_intervention,
        fixed_scalar_params=fixed_scalar_params,
        n_starts=n_starts,
        adam_steps=adam_steps,
        adam_lr=adam_lr,
        adam_device=adam_device,
        lambda_nuclear=lambda_nuclear,
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
        fit_intervention_model=fit_intervention_model,
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
        fit_intervention_model=fit_intervention_model,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    if result.get("field_mode") == FIELD_MODE_NUCLEAR_NORM:
        metrics.update(
            {
                "penalized_objective": float(result["final_penalized_objective"]),
                "mple_loss_without_penalty": float(result["final_mple_loss"]),
                "nuclear_norm": float(result["nuclear_norm"]),
                "effective_rank": float(result["effective_rank"]),
                "proximal_iterations": float(result["proximal_iterations"]),
            }
        )
        logger.info("Nuclear-norm optimizer diagnostics:")
        logger.info("  penalized_objective: %.6f", metrics["penalized_objective"])
        logger.info("  mple_loss_without_penalty: %.6f", metrics["mple_loss_without_penalty"])
        logger.info("  nuclear_norm: %.6f", metrics["nuclear_norm"])
        logger.info("  effective_rank: %.6f", metrics["effective_rank"])
        logger.info("  proximal_iterations: %.0f", metrics["proximal_iterations"])
    log_field_diagnostics(
        logger,
        metrics,
        params_hat,
        artifacts,
        fit_intervention_model=fit_intervention_model,
        bound_B=bound_B,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    write_summary_table(
        Path(args.data_folder) / "mple_summary",
        params_hat,
        metrics,
        loss_history[-1],
        artifacts,
        fit_intervention_model,
        bound_B,
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
        fit_intervention_model=fit_intervention_model,
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
