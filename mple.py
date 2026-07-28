"""Fit the latent-only conditional MPLE model for synthetic and real-data experiments."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import OptimizeResult, minimize

from pymanopt import Problem, function
from pymanopt.manifolds import Euclidean, FixedRankEmbedded, Product
from pymanopt.optimizers import ConjugateGradient

from utils.t0_config_utils import load_yaml_config
from utils.t1_matrix_io import load_gamma_matrix
from utils.t3_model_artifacts import (
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    ModelArtifacts,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    build_fit_model_artifacts,
)
from utils.t4_scalar_parameters import (
    free_scalar_parameter_names,
    scalar_parameter_names,
    uses_full_matrix_parameterization,
    validate_fixed_scalar_params,
)
from utils.t4_parameter_packing import (
    pack_theta,
    parameter_names,
    summarize_theta_for_logging,
    unpack_theta,
)
from utils.t3_interaction_matrices import interaction_effect
from utils.t3_field_operations import (
    compose_field_matrix_from_theta,
    compose_latent_field_matrix,
)
from utils.t8_fit_outputs import finalize_fit_outputs, load_truth_context


@dataclass(frozen=True)
class _FitEvalContext:
    """Cached arrays and bookkeeping shared across repeated MPLE loss evaluations."""

    x: np.ndarray
    prev_x: np.ndarray
    beta_feature: np.ndarray
    beta_update_mask: np.ndarray
    interaction_effect_x: np.ndarray
    loss_mask: np.ndarray | None
    outcome_size: float
    beta_outcome_size: float
    s: int
    e: int
    beta_mask_pre_s: bool
    beta_mask_post_e: bool
    fixed_scalar_params: dict[str, float]
    free_scalar_names: list[str]


def _build_fit_eval_context(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> _FitEvalContext:
    """Validate fit inputs and cache the arrays used by every fit-time update."""
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    x_array = np.asarray(x, dtype=float)
    t_steps = x_array.shape[0]
    s_index = int(s)
    if s_index < 0 or s_index > t_steps:
        raise ValueError(f"s={s_index} must lie in [0, T={t_steps}].")
    e_index = e if e is not None else t_steps
    e_index = int(e_index)
    if e_index < 0 or e_index > t_steps:
        raise ValueError(f"e={e_index} must lie in [0, T={t_steps}].")
    beta_feature = np.asarray(z, dtype=float)
    beta_update_mask = np.ones_like(x_array, dtype=bool)
    if bool(beta_mask_pre_s) and s_index > 0:
        beta_update_mask[:s_index, :] = False
    if bool(beta_mask_post_e) and e_index < t_steps:
        beta_update_mask[e_index:, :] = False
    resolved_loss_mask: np.ndarray | None = None
    outcome_size = float(x_array.size)
    if loss_mask is not None:
        resolved_loss_mask = np.asarray(loss_mask, dtype=bool)
        if resolved_loss_mask.shape != x_array.shape:
            raise ValueError(
                f"loss_mask shape {resolved_loss_mask.shape} does not match x shape {x_array.shape}."
            )
        outcome_size = float(np.count_nonzero(resolved_loss_mask))
        if outcome_size <= 0.0:
            raise ValueError("loss_mask must contain at least one active entry.")
        beta_update_mask = beta_update_mask & resolved_loss_mask
    beta_outcome_size = float(np.count_nonzero(beta_update_mask))
    free_names = free_scalar_parameter_names(fixed)
    return _FitEvalContext(
        x=x_array,
        prev_x=np.vstack([np.asarray(x_0, dtype=float), x_array[:-1, :]]),
        beta_feature=beta_feature,
        beta_update_mask=beta_update_mask,
        interaction_effect_x=np.asarray(interaction_effect_x, dtype=float),
        loss_mask=resolved_loss_mask,
        outcome_size=outcome_size,
        beta_outcome_size=beta_outcome_size,
        s=s_index,
        e=e_index,
        beta_mask_pre_s=bool(beta_mask_pre_s),
        beta_mask_post_e=bool(beta_mask_post_e),
        fixed_scalar_params=fixed,
        free_scalar_names=free_names,
    )


def _scalar_values_from_free_vector(
    free_scalar_values: np.ndarray,
    context: _FitEvalContext,
) -> dict[str, float]:
    """Merge optimized free scalars with the scalars fixed by configuration."""
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
    """Resolve scalar parameters from either a free vector or an explicit mapping."""
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
    """Assemble the conditional natural-parameter matrix h_t,i(x, z)."""
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
    """Project the residual matrix onto the scalar features used by fit updates."""
    if not context.free_scalar_names:
        return np.zeros(0, dtype=float)
    beta_gradient = 0.0
    if context.beta_outcome_size > 0.0:
        beta_gradient = (
            float(
                (
                    residual
                    * context.beta_feature
                    * np.asarray(context.beta_update_mask, dtype=float)
                ).sum()
            )
            / context.beta_outcome_size
        )
    gradient_lookup = {
        "beta": beta_gradient,
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
    """Evaluate the ordinary MPLE loss and the fit-update gradients."""
    resolved_scalars = _resolve_scalar_values(
        context=context,
        free_scalar_values=free_scalar_values,
        scalar_values=scalar_values,
    )
    h_x = _compute_h_x(field_matrix, resolved_scalars, context)
    loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
    residual = np.tanh(h_x) - context.x
    if context.loss_mask is not None:
        mask = context.loss_mask
        loss_x = loss_x * mask
        residual = residual * mask
    smooth_loss = float(loss_x.sum() / context.outcome_size)
    scalar_gradient = _scalar_gradient_from_residual(residual, context)
    return smooth_loss, residual, scalar_gradient


def _evaluate_scalar_only_loss(
    free_scalar_values: np.ndarray,
    context: _FitEvalContext,
) -> tuple[float, np.ndarray]:
    """Evaluate the zero-field loss with fit-time scalar update gradients."""
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
    """Evaluate loss and gradients for a field parameterized as time/node factors."""
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
    """Evaluate factor gradients with scalar terms folded into the fixed offset."""
    field_matrix = time_factors @ node_factors.T
    h_x = field_matrix + scalar_offset
    loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
    residual = np.tanh(h_x) - context.x
    if context.loss_mask is not None:
        mask = context.loss_mask
        loss_x = loss_x * mask
        residual = residual * mask
    smooth_loss = float(loss_x.sum() / context.outcome_size)
    time_gradient = (residual @ node_factors) / context.outcome_size
    node_gradient = (residual.T @ time_factors) / context.outcome_size
    return smooth_loss, residual, time_gradient, node_gradient


def _project_node_factor_columns_to_l2_ball(
    node_factors: np.ndarray,
    v_column_l2_max: float,
) -> np.ndarray:
    """Project each node-factor column onto the configured L2-radius constraint."""
    projected = np.array(node_factors, dtype=float, copy=True)
    if projected.ndim != 2:
        raise ValueError("node_factors must be a 2D array.")
    radius = float(v_column_l2_max)
    if radius <= 0.0:
        raise ValueError("v_column_l2_max must be positive.")
    if projected.size == 0:
        return projected
    column_norms = np.linalg.norm(projected, axis=0)
    active_columns = column_norms > radius
    if np.any(active_columns):
        projected[:, active_columns] *= (
            radius / column_norms[active_columns]
        )[None, :]
    return projected


def _alternating_factor_step_size(
    *,
    outcome_size: float,
    lambda_uv_ridge: float,
    fixed_factor_blocks: list[np.ndarray],
) -> float:
    """Return the alternating-update step size from the paired factor blocks."""
    spectral_sq = 0.0
    for block in fixed_factor_blocks:
        spectral_sq += float(np.linalg.norm(np.asarray(block, dtype=float), ord=2) ** 2)
    lipschitz = (spectral_sq + 2.0 * float(lambda_uv_ridge)) / float(outcome_size)
    return 1.0 if lipschitz <= 0.0 else 1.0 / lipschitz


def _prox_threshold_field_matrix(
    field_matrix: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    """Apply singular-value soft thresholding and return the shrunk nuclear norm."""
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
    """Create or reuse the file-and-console logger used by standalone MPLE runs."""
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
    """Load the saved `panel_data.npz` bundle for a fit request."""
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
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate the MPLE objective and gradient for a packed theta vector."""
    if x.shape[0] != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
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


def evaluate_mple_loss_from_parts(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    *,
    field_matrix: np.ndarray,
    beta: float,
    xi: float,
    eta: float,
    interaction_effect_x: np.ndarray,
    fixed_scalar_params: dict[str, float] | None = None,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> float:
    """Convenience wrapper for evaluating loss from explicit field/scalar components."""
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    loss, _, _ = _evaluate_full_field_loss(
        np.asarray(field_matrix, dtype=float),
        context,
        scalar_values={"beta": float(beta), "xi": float(xi), "eta": float(eta)},
    )
    return float(loss)


def _nuclear_norm(field_matrix: np.ndarray) -> float:
    """Return the nuclear norm of a field matrix."""
    if field_matrix.size == 0:
        return 0.0
    return float(
        np.linalg.svd(np.asarray(field_matrix, dtype=float), compute_uv=False).sum()
    )


def _nuclear_norm_normalizer(artifacts: ModelArtifacts) -> float:
    """Scale nuclear/Frobenius penalties by the square root of the panel size."""
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
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
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
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
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
    """Reconstruct a field matrix from thin-SVD factors on the fixed-rank manifold."""
    return (
        np.asarray(u, dtype=float) * np.asarray(singular_values, dtype=float)
    ) @ np.asarray(vt, dtype=float)


def _fixed_rank_point_from_field(field_matrix: np.ndarray, rank: int, point_type):
    """Convert a dense field matrix into a valid fixed-rank manifold point."""
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
    """Sample a numerically well-behaved random point on the fixed-rank manifold."""
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
    """Pack a pymanopt manifold point plus scalar values back into theta space."""
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
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    """Fit only the scalar parameters when no external field is being estimated."""
    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    if "beta" in free_names and context.beta_outcome_size <= 0.0:
        raise ValueError(
            "beta-gradient masking removed every eligible beta observation for a free "
            "beta parameter."
        )
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
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
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
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    if "beta" in free_names and context.beta_outcome_size <= 0.0:
        raise ValueError(
            "beta-gradient masking removed every eligible beta observation for a free "
            "beta parameter."
        )
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
    v_column_l2_max: float | None,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    """Fit via alternating gradient updates between U, Vt factors and scalar parameters.

    Parameterizes the field as U @ Vt (no explicit singular values) and alternates
    explicit gradient steps for scalars, U, and V with optional ridge regularization.
    Use this mode as an alternative to manifold optimization when pymanopt convergence is poor.
    Requires latent_rank >= 1.
    """
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")
    if v_column_l2_max is not None and float(v_column_l2_max) <= 0.0:
        raise ValueError("v_column_l2_max must be positive.")

    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    if "beta" in free_names and context.beta_outcome_size <= 0.0:
        raise ValueError(
            "beta-gradient masking removed every eligible beta observation for a free "
            "beta parameter."
        )
    n_starts = max(1, int(n_starts))
    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    projected_v_column_l2_max = (
        None if v_column_l2_max is None else float(v_column_l2_max)
    )
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

    def scalar_step_sizes() -> np.ndarray:
        if not free_names:
            return np.zeros(0, dtype=float)
        active_loss_mask = (
            np.ones_like(context.x, dtype=float)
            if context.loss_mask is None
            else np.asarray(context.loss_mask, dtype=float)
        )
        step_sizes: list[float] = []
        for name in free_names:
            if name == "beta":
                feature = (
                    context.beta_feature
                    * np.asarray(context.beta_update_mask, dtype=float)
                ).reshape(-1)
                normalizer = float(context.beta_outcome_size)
            elif name == "xi":
                feature = (
                    context.interaction_effect_x * active_loss_mask
                ).reshape(-1)
                normalizer = float(context.outcome_size)
            else:
                feature = (context.prev_x * active_loss_mask).reshape(-1)
                normalizer = float(context.outcome_size)
            if normalizer <= 0.0:
                raise ValueError(
                    "beta-gradient masking removed every eligible beta observation for "
                    "a free beta parameter."
                )
            lipschitz = float(np.sum(feature * feature) / normalizer)
            step_sizes.append(1.0 if lipschitz <= 0.0 else 1.0 / lipschitz)
        return np.asarray(step_sizes, dtype=float)

    def factor_step_size(fixed_factors: np.ndarray) -> float:
        return _alternating_factor_step_size(
            outcome_size=context.outcome_size,
            lambda_uv_ridge=lambda_uv_ridge,
            fixed_factor_blocks=[np.asarray(fixed_factors, dtype=float)],
        )

    scalar_lrs = scalar_step_sizes()

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
            node_factors = np.asarray(theta_parts["node_factors"], dtype=float)
            if projected_v_column_l2_max is not None:
                node_factors = _project_node_factor_columns_to_l2_ball(
                    node_factors,
                    projected_v_column_l2_max,
                )
            return (
                np.asarray(theta_parts["time_factors"], dtype=float),
                node_factors,
                np.asarray([theta_parts[name] for name in free_names], dtype=float),
            )
        node_factors = rng.normal(0.0, 0.1, size=(n_nodes, rank))
        if projected_v_column_l2_max is not None:
            node_factors = _project_node_factor_columns_to_l2_ball(
                node_factors,
                projected_v_column_l2_max,
            )
        return (
            rng.normal(0.0, 0.1, size=(t_steps, rank)),
            node_factors,
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
                free_scalar_values = free_scalar_values - scalar_lrs * scalar_gradient
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
                if projected_v_column_l2_max is not None:
                    node_factors = _project_node_factor_columns_to_l2_ball(
                        node_factors,
                        projected_v_column_l2_max,
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


def _fit_mple_treatment_low_rank(
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
    v_column_l2_max: float | None,
    shared_unit: bool,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], OptimizeResult]:
    """Fit treatment-conditioned low-rank baselines with alternating gradient blocks."""
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")
    if v_column_l2_max is not None and float(v_column_l2_max) <= 0.0:
        raise ValueError("v_column_l2_max must be positive.")
    if theta_init is not None:
        raise ValueError(
            "theta_init is not supported for treatment-split low-rank modes because "
            "their external theta schema only serializes the realized field_matrix."
        )

    context = _build_fit_eval_context(
        x,
        z,
        x_0,
        interaction_effect_x,
        fixed_scalar_params,
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    fixed = context.fixed_scalar_params
    free_names = context.free_scalar_names
    if free_names:
        raise ValueError(
            "Treatment-split low-rank modes require beta, xi, and eta to be fixed; "
            f"free scalar parameters were requested for {free_names}."
        )

    rank = int(artifacts.latent_rank)
    t_steps = int(artifacts.t_steps)
    n_nodes = int(artifacts.gamma_matrix.shape[0])
    if rank < 1 or rank > min(t_steps, n_nodes):
        raise ValueError(
            f"latent_rank={rank} must lie in [1, {min(t_steps, n_nodes)}] for "
            "treatment-split low-rank optimization."
        )

    control_mask, treated_mask = _treatment_active_masks(context)
    control_has_support = bool(np.any(control_mask))
    treated_has_support = bool(np.any(treated_mask))
    projected_v_column_l2_max = (
        None if v_column_l2_max is None else float(v_column_l2_max)
    )
    n_starts = max(1, int(n_starts))
    outer_iterations = max(1, int(steps))
    inner_gradient_steps = 3
    optimizer_mode = (
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK
        if shared_unit
        else OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK
    )
    optimizer_name = (
        "alternating_treatment_shared_unit_low_rank"
        if shared_unit
        else "alternating_treatment_split_low_rank"
    )
    mode_label = (
        "Treatment-shared-unit alternating low-rank"
        if shared_unit
        else "Treatment-split alternating low-rank"
    )

    def _maybe_project_node_factors(node_factors: np.ndarray) -> np.ndarray:
        if projected_v_column_l2_max is None:
            return np.asarray(node_factors, dtype=float)
        return _project_node_factor_columns_to_l2_ball(
            np.asarray(node_factors, dtype=float),
            projected_v_column_l2_max,
        )

    def _zero_time() -> np.ndarray:
        return np.zeros((t_steps, rank), dtype=float)

    def _zero_node() -> np.ndarray:
        return np.zeros((n_nodes, rank), dtype=float)

    def _field_from_factors(
        time_factors: np.ndarray,
        node_factors: np.ndarray,
        has_support: bool,
    ) -> np.ndarray:
        if not has_support:
            return np.zeros((t_steps, n_nodes), dtype=float)
        return compose_latent_field_matrix(node_factors, time_factors)

    def evaluate_state(
        control_time_factors: np.ndarray,
        control_node_factors: np.ndarray,
        treated_time_factors: np.ndarray,
        treated_node_factors: np.ndarray | None,
        shared_node_factors: np.ndarray | None,
    ) -> tuple[float, float, dict[str, np.ndarray], dict[str, object]]:
        effective_control_node_factors = (
            np.asarray(shared_node_factors, dtype=float)
            if shared_unit
            else np.asarray(control_node_factors, dtype=float)
        )
        effective_treated_node_factors = (
            np.asarray(shared_node_factors, dtype=float)
            if shared_unit
            else np.asarray(treated_node_factors, dtype=float)
        )
        control_field_matrix = _field_from_factors(
            np.asarray(control_time_factors, dtype=float),
            effective_control_node_factors,
            control_has_support,
        )
        treated_field_matrix = _field_from_factors(
            np.asarray(treated_time_factors, dtype=float),
            effective_treated_node_factors,
            treated_has_support,
        )
        (
            smooth_loss,
            control_residual,
            treated_residual,
            _,
            _,
        ) = _evaluate_treatment_surface_loss(
            control_field_matrix,
            treated_field_matrix,
            context,
        )
        ridge_scale = 2.0 * float(lambda_uv_ridge) / context.outcome_size
        gradients: dict[str, np.ndarray] = {
            "control_time_factors": np.zeros_like(control_time_factors, dtype=float),
            "treated_time_factors": np.zeros_like(treated_time_factors, dtype=float),
        }
        factor_norm_sq = (
            float(np.sum(np.asarray(control_time_factors, dtype=float) ** 2))
            + float(np.sum(np.asarray(treated_time_factors, dtype=float) ** 2))
        )
        if shared_unit:
            shared_node_factors = np.asarray(shared_node_factors, dtype=float)
            factor_norm_sq += float(np.sum(shared_node_factors**2))
            gradients["shared_node_factors"] = ridge_scale * shared_node_factors
            if control_has_support:
                gradients["control_time_factors"] = (
                    control_residual @ shared_node_factors
                ) / context.outcome_size + ridge_scale * np.asarray(
                    control_time_factors,
                    dtype=float,
                )
                gradients["shared_node_factors"] = gradients["shared_node_factors"] + (
                    control_residual.T @ np.asarray(control_time_factors, dtype=float)
                ) / context.outcome_size
            if treated_has_support:
                gradients["treated_time_factors"] = (
                    treated_residual @ shared_node_factors
                ) / context.outcome_size + ridge_scale * np.asarray(
                    treated_time_factors,
                    dtype=float,
                )
                gradients["shared_node_factors"] = gradients["shared_node_factors"] + (
                    treated_residual.T @ np.asarray(treated_time_factors, dtype=float)
                ) / context.outcome_size
            u_frobenius_norm = float(np.linalg.norm(shared_node_factors, ord="fro"))
        else:
            control_node_factors = np.asarray(control_node_factors, dtype=float)
            treated_node_factors = np.asarray(treated_node_factors, dtype=float)
            factor_norm_sq += float(np.sum(control_node_factors**2))
            factor_norm_sq += float(np.sum(treated_node_factors**2))
            gradients["control_node_factors"] = ridge_scale * control_node_factors
            gradients["treated_node_factors"] = ridge_scale * treated_node_factors
            if control_has_support:
                gradients["control_time_factors"] = (
                    control_residual @ control_node_factors
                ) / context.outcome_size + ridge_scale * np.asarray(
                    control_time_factors,
                    dtype=float,
                )
                gradients["control_node_factors"] = (
                    gradients["control_node_factors"]
                    + (control_residual.T @ np.asarray(control_time_factors, dtype=float))
                    / context.outcome_size
                )
            if treated_has_support:
                gradients["treated_time_factors"] = (
                    treated_residual @ treated_node_factors
                ) / context.outcome_size + ridge_scale * np.asarray(
                    treated_time_factors,
                    dtype=float,
                )
                gradients["treated_node_factors"] = (
                    gradients["treated_node_factors"]
                    + (treated_residual.T @ np.asarray(treated_time_factors, dtype=float))
                    / context.outcome_size
                )
            u_frobenius_norm = float(
                np.sqrt(
                    np.sum(control_node_factors**2) + np.sum(treated_node_factors**2)
                )
            )
        v_frobenius_norm = float(
            np.sqrt(
                np.sum(np.asarray(control_time_factors, dtype=float) ** 2)
                + np.sum(np.asarray(treated_time_factors, dtype=float) ** 2)
            )
        )
        realized_field_matrix = compose_realized_treatment_field_matrix(
            control_field_matrix,
            treated_field_matrix,
            context.beta_feature,
        )
        ridge_penalty = float(lambda_uv_ridge) * factor_norm_sq / context.outcome_size
        state_metadata: dict[str, object] = {
            "control_field_matrix": control_field_matrix,
            "treated_field_matrix": treated_field_matrix,
            "realized_field_matrix": realized_field_matrix,
            "u_frobenius_norm": u_frobenius_norm,
            "v_frobenius_norm": v_frobenius_norm,
        }
        return smooth_loss + ridge_penalty, smooth_loss, gradients, state_metadata

    def pack_state(
        realized_field_matrix: np.ndarray,
    ) -> np.ndarray:
        return pack_theta(
            {
                "field_matrix": np.asarray(realized_field_matrix, dtype=float),
                "beta": float(fixed["beta"]),
                "xi": float(fixed["xi"]),
                "eta": float(fixed["eta"]),
            },
            artifacts,
            fixed_scalar_params=fixed,
        )

    def initial_state_for_start(
        start_index: int,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(int(seed) + start_index)
        state: dict[str, np.ndarray] = {}
        if shared_unit:
            shared_node_factors = _zero_node()
            if control_has_support or treated_has_support:
                shared_node_factors = _maybe_project_node_factors(
                    rng.normal(0.0, 0.1, size=(n_nodes, rank))
                )
            state["shared_node_factors"] = shared_node_factors
            state["control_time_factors"] = (
                rng.normal(0.0, 0.1, size=(t_steps, rank))
                if control_has_support
                else _zero_time()
            )
            state["treated_time_factors"] = (
                rng.normal(0.0, 0.1, size=(t_steps, rank))
                if treated_has_support
                else _zero_time()
            )
            state["control_node_factors"] = _zero_node()
            state["treated_node_factors"] = _zero_node()
        else:
            state["control_node_factors"] = (
                _maybe_project_node_factors(rng.normal(0.0, 0.1, size=(n_nodes, rank)))
                if control_has_support
                else _zero_node()
            )
            state["control_time_factors"] = (
                rng.normal(0.0, 0.1, size=(t_steps, rank))
                if control_has_support
                else _zero_time()
            )
            state["treated_node_factors"] = (
                _maybe_project_node_factors(rng.normal(0.0, 0.1, size=(n_nodes, rank)))
                if treated_has_support
                else _zero_node()
            )
            state["treated_time_factors"] = (
                rng.normal(0.0, 0.1, size=(t_steps, rank))
                if treated_has_support
                else _zero_time()
            )
        return state

    def factor_step_size(*fixed_factor_blocks: np.ndarray) -> float:
        return _alternating_factor_step_size(
            outcome_size=context.outcome_size,
            lambda_uv_ridge=lambda_uv_ridge,
            fixed_factor_blocks=[
                np.asarray(block, dtype=float) for block in fixed_factor_blocks
            ],
        )

    best_theta: np.ndarray | None = None
    best_result: OptimizeResult | None = None
    best_penalized_history: list[float] = []
    best_mple_history: list[float] = []
    best_start = 0
    best_penalized_objective = np.inf
    start_summaries: list[dict[str, object]] = []

    for start_index in range(n_starts):
        state = initial_state_for_start(start_index)
        (
            initial_penalized_objective,
            initial_mple_loss,
            _,
            _,
        ) = evaluate_state(
            state["control_time_factors"],
            state["control_node_factors"],
            state["treated_time_factors"],
            state.get("treated_node_factors"),
            state.get("shared_node_factors"),
        )
        mple_history = [float(initial_mple_loss)]
        penalized_history = [float(initial_penalized_objective)]
        cost_evaluations = 1
        iterations_completed = 0
        converged = False
        message = "STOP: TOTAL NO. OF OUTER ITERATIONS REACHED LIMIT"

        if logger is not None:
            logger.info(
                "%s start %s/%s | seed=%s | initial_loss=%.6f | initial_penalized=%.6f",
                mode_label,
                start_index + 1,
                n_starts,
                int(seed) + start_index,
                initial_mple_loss,
                initial_penalized_objective,
            )

        for outer_index in range(outer_iterations):
            for _ in range(inner_gradient_steps):
                if control_has_support:
                    (
                        _,
                        _,
                        gradients,
                        _,
                    ) = evaluate_state(
                        state["control_time_factors"],
                        state["control_node_factors"],
                        state["treated_time_factors"],
                        state.get("treated_node_factors"),
                        state.get("shared_node_factors"),
                    )
                    control_node_factors = (
                        state["shared_node_factors"]
                        if shared_unit
                        else state["control_node_factors"]
                    )
                    state["control_time_factors"] = (
                        np.asarray(state["control_time_factors"], dtype=float)
                        - factor_step_size(control_node_factors)
                        * np.asarray(gradients["control_time_factors"], dtype=float)
                    )
                    cost_evaluations += 1
                else:
                    state["control_time_factors"] = _zero_time()
                    if not shared_unit:
                        state["control_node_factors"] = _zero_node()

                if not shared_unit and control_has_support:
                    (
                        _,
                        _,
                        gradients,
                        _,
                    ) = evaluate_state(
                        state["control_time_factors"],
                        state["control_node_factors"],
                        state["treated_time_factors"],
                        state.get("treated_node_factors"),
                        state.get("shared_node_factors"),
                    )
                    state["control_node_factors"] = _maybe_project_node_factors(
                        np.asarray(state["control_node_factors"], dtype=float)
                        - factor_step_size(state["control_time_factors"])
                        * np.asarray(gradients["control_node_factors"], dtype=float)
                    )
                    cost_evaluations += 1

                if treated_has_support:
                    (
                        _,
                        _,
                        gradients,
                        _,
                    ) = evaluate_state(
                        state["control_time_factors"],
                        state["control_node_factors"],
                        state["treated_time_factors"],
                        state.get("treated_node_factors"),
                        state.get("shared_node_factors"),
                    )
                    treated_node_factors = (
                        state["shared_node_factors"]
                        if shared_unit
                        else state["treated_node_factors"]
                    )
                    state["treated_time_factors"] = (
                        np.asarray(state["treated_time_factors"], dtype=float)
                        - factor_step_size(treated_node_factors)
                        * np.asarray(gradients["treated_time_factors"], dtype=float)
                    )
                    cost_evaluations += 1
                else:
                    state["treated_time_factors"] = _zero_time()
                    if not shared_unit:
                        state["treated_node_factors"] = _zero_node()

                if shared_unit:
                    if control_has_support or treated_has_support:
                        (
                            _,
                            _,
                            gradients,
                            _,
                        ) = evaluate_state(
                            state["control_time_factors"],
                            state["control_node_factors"],
                            state["treated_time_factors"],
                            state.get("treated_node_factors"),
                            state.get("shared_node_factors"),
                        )
                        state["shared_node_factors"] = _maybe_project_node_factors(
                            np.asarray(state["shared_node_factors"], dtype=float)
                            - factor_step_size(
                                state["control_time_factors"],
                                state["treated_time_factors"],
                            )
                            * np.asarray(gradients["shared_node_factors"], dtype=float)
                        )
                        cost_evaluations += 1
                    else:
                        state["shared_node_factors"] = _zero_node()
                elif treated_has_support:
                    (
                        _,
                        _,
                        gradients,
                        _,
                    ) = evaluate_state(
                        state["control_time_factors"],
                        state["control_node_factors"],
                        state["treated_time_factors"],
                        state.get("treated_node_factors"),
                        state.get("shared_node_factors"),
                    )
                    state["treated_node_factors"] = _maybe_project_node_factors(
                        np.asarray(state["treated_node_factors"], dtype=float)
                        - factor_step_size(state["treated_time_factors"])
                        * np.asarray(gradients["treated_node_factors"], dtype=float)
                    )
                    cost_evaluations += 1

            (
                penalized_loss,
                smooth_loss,
                _,
                state_metadata,
            ) = evaluate_state(
                state["control_time_factors"],
                state["control_node_factors"],
                state["treated_time_factors"],
                state.get("treated_node_factors"),
                state.get("shared_node_factors"),
            )
            penalized_history.append(float(penalized_loss))
            mple_history.append(float(smooth_loss))
            iterations_completed = outer_index + 1

            if verbose_every and outer_index % verbose_every == 0:
                if logger is None:
                    print(
                        f"{mode_label} iter {outer_index} | Loss: {smooth_loss:.6f} | Penalized: {penalized_loss:.6f}"
                    )
                else:
                    logger.info(
                        "%s iter %s | Loss: %.6f | Penalized: %.6f",
                        mode_label,
                        outer_index,
                        smooth_loss,
                        penalized_loss,
                    )

            if len(penalized_history) >= 2:
                improvement = abs(penalized_history[-2] - penalized_history[-1])
                if improvement <= float(tol) * max(1.0, abs(penalized_history[-2])):
                    converged = True
                    message = "CONVERGED: alternating treatment objective tolerance reached"
                    break

        (
            final_penalized_loss,
            final_smooth_loss,
            _,
            state_metadata,
        ) = evaluate_state(
            state["control_time_factors"],
            state["control_node_factors"],
            state["treated_time_factors"],
            state.get("treated_node_factors"),
            state.get("shared_node_factors"),
        )
        theta_hat = pack_state(np.asarray(state_metadata["realized_field_matrix"], dtype=float))
        start_summary = {
            "start_index": start_index,
            "seed": int(seed) + start_index,
            "initialization_kind": "random",
            "initial_mple_loss": float(initial_mple_loss),
            "initial_penalized_objective": float(initial_penalized_objective),
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
            result_payload: dict[str, object] = {
                "x": theta_hat,
                "success": bool(converged),
                "message": message,
                "nit": int(iterations_completed),
                "nfev": int(cost_evaluations),
                "iterations": int(iterations_completed),
                "cost_evaluations": int(cost_evaluations),
                "optimizer_mode": optimizer_mode,
                "optimizer": optimizer_name,
                "lambda_uv_ridge": float(lambda_uv_ridge),
                "final_mple_loss": float(final_smooth_loss),
                "final_penalized_objective": float(final_penalized_loss),
                "u_frobenius_norm": float(state_metadata["u_frobenius_norm"]),
                "v_frobenius_norm": float(state_metadata["v_frobenius_norm"]),
                "effective_rank": float(
                    np.linalg.matrix_rank(
                        np.asarray(state_metadata["realized_field_matrix"], dtype=float)
                    )
                ),
                "mple_history": list(mple_history),
                "penalized_history": list(penalized_history),
                "best_start": int(start_index),
                "n_starts": int(n_starts),
                "start_summaries": start_summaries,
                "control_field_matrix": np.asarray(
                    state_metadata["control_field_matrix"],
                    dtype=float,
                ),
                "treated_field_matrix": np.asarray(
                    state_metadata["treated_field_matrix"],
                    dtype=float,
                ),
                "realized_field_matrix": np.asarray(
                    state_metadata["realized_field_matrix"],
                    dtype=float,
                ),
                "control_time_factors": np.asarray(
                    state["control_time_factors"],
                    dtype=float,
                ),
                "treated_time_factors": np.asarray(
                    state["treated_time_factors"],
                    dtype=float,
                ),
            }
            if shared_unit:
                result_payload["shared_node_factors"] = np.asarray(
                    state["shared_node_factors"],
                    dtype=float,
                )
            else:
                result_payload["control_node_factors"] = np.asarray(
                    state["control_node_factors"],
                    dtype=float,
                )
                result_payload["treated_node_factors"] = np.asarray(
                    state["treated_node_factors"],
                    dtype=float,
                )
            best_result = OptimizeResult(**result_payload)

    if best_theta is None or best_result is None:
        raise RuntimeError(
            "Treatment-split low-rank optimizer did not produce a candidate solution."
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
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    loss_mask: np.ndarray | None = None,
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
        s=s,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
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


def _apply_warm_start(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    s: int,
    param_names: list[str],
    artifacts: ModelArtifacts,
    interaction_effect_x: np.ndarray,
    warm_start_steps: int,
    seed: int,
    verbose_every: int,
    tol: float,
    logger,
    theta_init,
    fixed_scalar_params: dict[str, float],
    warm_start_fixed_scalars: dict[str, float],
    lambda_nuclear: float,
    lambda_frobenius: float,
    lambda_uv_ridge: float,
    v_column_l2_max: float | None,
    proximal_lr: float,
    e: int | None,
    beta_mask_pre_s: bool,
    beta_mask_post_e: bool,
    loss_mask: np.ndarray | None,
) -> np.ndarray:
    """Run a short constrained phase and reuse it as the full-fit initialization."""
    phase1_fixed = {**fixed_scalar_params, **warm_start_fixed_scalars}
    phase1_theta, _, _ = fit_mple(
        x,
        z,
        x_0=x_0,
        s=s,
        param_names=param_names,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        steps=warm_start_steps,
        seed=seed,
        verbose_every=verbose_every,
        tol=tol,
        logger=logger,
        theta_init=theta_init,
        fixed_scalar_params=phase1_fixed,
        n_starts=1,
        lambda_nuclear=lambda_nuclear,
        lambda_frobenius=lambda_frobenius,
        lambda_uv_ridge=lambda_uv_ridge,
        v_column_l2_max=v_column_l2_max,
        proximal_lr=proximal_lr,
        e=e,
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        loss_mask=loss_mask,
    )
    theta_parts = unpack_theta(phase1_theta, artifacts, fixed_scalar_params=phase1_fixed)
    return pack_theta(theta_parts, artifacts, fixed_scalar_params=fixed_scalar_params)


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
    v_column_l2_max: float | None = None,
    proximal_lr: float = 1.0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
    warm_start_fixed_scalars: dict[str, float] | None = None,
    warm_start_steps: int = 0,
    loss_mask: np.ndarray | None = None,
):
    """Dispatch to the configured optimizer mode and return theta, history, and status."""
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError("x and z must both have shape (T, N).")
    if v_column_l2_max is not None and float(v_column_l2_max) <= 0.0:
        raise ValueError("v_column_l2_max must be positive.")

    t_steps = x.shape[0]
    if t_steps != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    if (
        bool(beta_mask_pre_s) or bool(beta_mask_post_e)
    ) and artifacts.optimizer_mode not in {
        OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
        OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    }:
        raise ValueError(
            "beta-gradient-only masking is only supported for "
            "optimizer_mode in {'alternating_latent_rank', 'no_external_field'}; "
            "the other optimizer modes are deprecated for masked-beta workflows."
        )

    if artifacts.optimizer_mode in {
        "alternating_treatment_split_latent_rank",
        "alternating_treatment_shared_unit_latent_rank",
    }:
        raise ValueError(
            "Treatment-specific MPLE optimizer modes are no longer supported. "
            "Use 'alternating_latent_rank' or 'snn_treatment_split' instead."
        )

    if warm_start_fixed_scalars and int(warm_start_steps) > 0:
        theta_init = _apply_warm_start(
            x,
            z,
            x_0,
            s,
            param_names,
            artifacts,
            interaction_effect_x,
            int(warm_start_steps),
            seed,
            verbose_every,
            tol,
            logger,
            theta_init,
            validate_fixed_scalar_params(fixed_scalar_params),
            validate_fixed_scalar_params(warm_start_fixed_scalars),
            lambda_nuclear,
            lambda_frobenius,
            lambda_uv_ridge,
            v_column_l2_max,
            proximal_lr,
            e,
            beta_mask_pre_s,
            beta_mask_post_e,
            loss_mask,
        )

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
            s=s,
            e=e,
            beta_mask_pre_s=beta_mask_pre_s,
            beta_mask_post_e=beta_mask_post_e,
            loss_mask=loss_mask,
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
            s=s,
            e=e,
            beta_mask_pre_s=beta_mask_pre_s,
            beta_mask_post_e=beta_mask_post_e,
            loss_mask=loss_mask,
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
            v_column_l2_max=v_column_l2_max,
            s=s,
            e=e,
            beta_mask_pre_s=beta_mask_pre_s,
            beta_mask_post_e=beta_mask_post_e,
            loss_mask=loss_mask,
        )
    # Deprecated for masked-beta workflows because the quasi-Newton objective assumes
    # one shared scalar objective/gradient pair.
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
            s=s,
            e=e,
            beta_mask_pre_s=beta_mask_pre_s,
            beta_mask_post_e=beta_mask_post_e,
            loss_mask=loss_mask,
        )
    # Deprecated for masked-beta workflows because the manifold solver assumes a true
    # scalar objective with a matching Euclidean gradient.
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
            s=s,
            e=e,
            beta_mask_pre_s=beta_mask_pre_s,
            beta_mask_post_e=beta_mask_post_e,
            loss_mask=loss_mask,
        )
    raise ValueError(f"Unsupported optimizer_mode: {artifacts.optimizer_mode}")


def main() -> None:
    """Run the standalone MPLE CLI for one materialized fit directory."""
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
    beta_mask_pre_s = bool(
        config.estimation_params.get("beta_mask_pre_s", False)
        if "estimation_params" in config
        else False
    )
    beta_mask_post_e = bool(
        config.estimation_params.get("beta_mask_post_e", False)
        if "estimation_params" in config
        else False
    )
    warm_start_fixed_scalars = validate_fixed_scalar_params(
        (
            OmegaConf.to_container(
                config.estimation_params.warm_start_fixed_scalars,
                resolve=True,
            )
            if "estimation_params" in config
            and "warm_start_fixed_scalars" in config.estimation_params
            else {}
        )
    )
    warm_start_steps = int(
        config.estimation_params.warm_start_steps
        if "estimation_params" in config
        and "warm_start_steps" in config.estimation_params
        else 0
    )
    lambda_nuclear = float(config.global_params.get("lambda_nuclear", 0.0))
    lambda_frobenius = float(config.global_params.get("lambda_frobenius", 0.0))
    lambda_uv_ridge = float(config.global_params.get("lambda_uv_ridge", 0.0))
    raw_v_column_l2_max = config.global_params.get("v_column_l2_max", None)
    v_column_l2_max = (
        None
        if raw_v_column_l2_max is None
        else float(raw_v_column_l2_max)
    )
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
    raw_loss_mask_path = input_artifacts.get("loss_mask_path", None)
    loss_mask_path = (
        None if raw_loss_mask_path in (None, "") else Path(str(raw_loss_mask_path))
    )
    logger.info("Using panel artifact: %s", panel_path)
    logger.info("Using x_0 artifact: %s", x0_path)
    logger.info("Using fit config: %s", config_path)
    logger.info("Using model artifact directory: %s", model_artifact_dir)
    logger.info("Using truth artifact directory: %s", truth_artifact_dir)
    if loss_mask_path is not None:
        logger.info("Using loss mask artifact: %s", loss_mask_path)
    # Load Data
    x_0 = np.load(x0_path)
    panel = load_panel_artifact(panel_path)
    x = panel["x"]
    z = panel["z"]
    loss_mask = (
        None
        if loss_mask_path is None
        else np.asarray(np.load(loss_mask_path, allow_pickle=False), dtype=bool)
    )

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
        "Warm-start fixed scalars: %s for %s steps",
        warm_start_fixed_scalars or {},
        warm_start_steps,
    )
    logger.info(
        "Beta-gradient mask before s: %s with s=%s",
        beta_mask_pre_s,
        config.global_params.s,
    )
    logger.info(
        "Beta-gradient mask after e: %s with e=%s",
        beta_mask_post_e,
        config.global_params.e,
    )
    if v_column_l2_max is not None:
        if artifacts.optimizer_mode in {
            OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
        }:
            logger.info(
                "Alternating V-column L2-ball constraint active with radius %s",
                v_column_l2_max,
            )
        else:
            logger.info(
                "Ignoring v_column_l2_max=%s for optimizer_mode=%s",
                v_column_l2_max,
                artifacts.optimizer_mode,
            )
    logger.info(
        "Optimizer settings: steps=%s, tol=%s, n_starts=%s, seed=%s, "
        "lambda_nuclear=%s, lambda_frobenius=%s, lambda_uv_ridge=%s, "
        "v_column_l2_max=%s, proximal_lr=%s",
        steps,
        tol,
        n_starts,
        seed,
        lambda_nuclear,
        lambda_frobenius,
        lambda_uv_ridge,
        v_column_l2_max,
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
        v_column_l2_max=v_column_l2_max,
        proximal_lr=proximal_lr,
        e=int(config.global_params.get("e", x.shape[0])),
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        warm_start_fixed_scalars=warm_start_fixed_scalars,
        warm_start_steps=warm_start_steps,
        loss_mask=loss_mask,
    )
    finalize_fit_outputs(
        args.data_folder,
        logger,
        params_hat,
        loss_history,
        result,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    logger.info(
        "Saved summary table to %s",
        Path(args.data_folder) / "mple_summary.csv",
    )
    logger.info("Log saved to %s", log_file)


if __name__ == "__main__":
    main()
