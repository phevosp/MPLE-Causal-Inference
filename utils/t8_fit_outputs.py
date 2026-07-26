"""Fit-output helpers for MPLE diagnostics, summaries, and saved artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from scipy.optimize import OptimizeResult

from utils.t0_config_utils import load_yaml_config
from utils.t0_csv_utils import write_csv
from utils.t0_path_utils import first_existing_path, io_path
from utils.t3_field_operations import latent_field_bound_norm, with_theta_field
from utils.t3_interaction_matrices import compose_interaction_matrix
from utils.t3_model_artifacts import (
    ModelArtifacts,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    TreatmentFieldArtifacts,
    load_model_artifacts,
    save_field_artifacts,
    save_treatment_field_artifacts,
)
from utils.t4_parameter_packing import unpack_theta
from utils.t4_scalar_parameters import scalar_parameter_names
from utils.t5_parameter_bundles import save_estimated_parameter_bundle


def _fmt(value: object) -> str:
    """Format a value for display in CSV/reports."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def scalar_summary_rows(
    est_theta: np.ndarray,
    artifacts: ModelArtifacts,
    scalar_truths: dict[str, float] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Build per-scalar estimate rows for the fit summary table."""
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
    """Load truth-side field and scalar artifacts when the experiment exposes them."""
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
    """Compare the fitted field and interaction matrix against available truth."""
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
    """Summarize rank and scale diagnostics for estimated and true latent fields."""
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


def _merge_optimizer_metrics(
    base_metrics: dict[str, float],
    result: OptimizeResult,
) -> dict[str, float]:
    """Attach optimizer-specific diagnostics to the truth-metric dictionary."""
    metrics = dict(base_metrics)
    optimizer_mode = result.get("optimizer_mode")
    if optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
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
    elif optimizer_mode == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD:
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
    elif optimizer_mode in {
        OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
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
    return metrics


def _log_optimizer_metrics(
    logger,
    result: OptimizeResult,
    metrics: dict[str, float],
) -> None:
    """Write optimizer-specific diagnostics to the MPLE log."""
    optimizer_mode = result.get("optimizer_mode")
    if optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
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
    elif optimizer_mode == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD:
        if float(result["lambda_frobenius"]) <= 0.0:
            return
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
    elif optimizer_mode in {
        OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
        OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    }:
        optimizer_title = (
            "Concurrent low-rank optimizer diagnostics:"
            if optimizer_mode == OPTIMIZER_MODE_CONCURRENT_LATENT_RANK
            else (
                "Treatment-shared-unit alternating low-rank optimizer diagnostics:"
                if optimizer_mode
                == OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK
                else (
                    "Treatment-split alternating low-rank optimizer diagnostics:"
                    if optimizer_mode
                    == OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK
                    else "Alternating low-rank optimizer diagnostics:"
                )
            )
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


def write_summary_table(
    summary_stem: str | Path,
    est_theta: np.ndarray,
    metrics: dict[str, float],
    loss: float,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> None:
    """Persist scalar estimates, fit metrics, and latent diagnostics as CSV."""
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
    write_csv(
        Path(f"{summary_stem}.csv"),
        rows,
        ["category", "name", "estimate", "true", "squared_error"],
    )


def write_optimizer_start_summary(path: str | Path, result: OptimizeResult) -> None:
    """Write one row per optimizer start, flagging the selected best run."""
    start_summaries = result.get("start_summaries", [])
    if not start_summaries:
        return
    best_start = int(result.get("best_start", 0))
    rows = []
    for row in start_summaries:
        start_index = int(row["start_index"])
        rows.append({**row, "is_best": start_index == best_start})
    write_csv(
        Path(path),
        rows,
        [
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
        ],
    )


def log_estimates(
    logger,
    title: str,
    scalar_rows: list[dict[str, object]],
) -> None:
    """Emit estimated scalar parameters, optionally alongside truth values."""
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
    """Log field-level diagnostics after the scalar summary block."""
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
    result: OptimizeResult,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> None:
    """Save fitted field/interactions and truth-side artifacts for later analysis."""
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
    est_artifacts = with_theta_field(
        artifacts,
        est_parts,
    )
    output_root = Path(data_folder)
    save_field_artifacts(output_root / "estimated_field_artifacts.npz", est_artifacts)
    estimated_interaction = compose_interaction_matrix(
        est_parts["xi"],
        artifacts.gamma_matrix,
    )
    if sparse.issparse(estimated_interaction):
        sparse.save_npz(
            io_path(output_root / "estimated_interaction_matrix_sparse.npz"),
            estimated_interaction,
        )
    else:
        np.save(
            io_path(output_root / "estimated_interaction_matrix.npy"),
            estimated_interaction,
        )
    save_estimated_parameter_bundle(
        output_root / "estimated_parameter_bundle.npz",
        beta=float(est_parts["beta"]),
        xi=float(est_parts["xi"]),
        eta=float(est_parts["eta"]),
        latent_rank=int(est_artifacts.latent_rank),
        t_steps=int(est_artifacts.t_steps),
        field_matrix=np.asarray(est_artifacts.field_matrix, dtype=float),
    )
    if artifacts.optimizer_mode in {
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
    }:
        save_treatment_field_artifacts(
            output_root / "estimated_treatment_field_artifacts.npz",
            TreatmentFieldArtifacts(
                optimizer_mode=str(result["optimizer_mode"]),
                latent_rank=int(artifacts.latent_rank),
                control_field_matrix=np.asarray(
                    result["control_field_matrix"],
                    dtype=float,
                ),
                treated_field_matrix=np.asarray(
                    result["treated_field_matrix"],
                    dtype=float,
                ),
                realized_field_matrix=np.asarray(
                    result["realized_field_matrix"],
                    dtype=float,
                ),
                lambda_uv_ridge=float(result["lambda_uv_ridge"]),
                best_start=int(result["best_start"]),
                n_starts=int(result["n_starts"]),
                final_mple_loss=float(result["final_mple_loss"]),
                final_penalized_objective=float(result["final_penalized_objective"]),
                control_node_factors=(
                    None
                    if "control_node_factors" not in result
                    else np.asarray(result["control_node_factors"], dtype=float)
                ),
                control_time_factors=np.asarray(
                    result["control_time_factors"],
                    dtype=float,
                ),
                treated_node_factors=(
                    None
                    if "treated_node_factors" not in result
                    else np.asarray(result["treated_node_factors"], dtype=float)
                ),
                treated_time_factors=np.asarray(
                    result["treated_time_factors"],
                    dtype=float,
                ),
                shared_node_factors=(
                    None
                    if "shared_node_factors" not in result
                    else np.asarray(result["shared_node_factors"], dtype=float)
                ),
            ),
        )
    if truth_context is None or truth_context.get("field_artifacts") is None:
        return
    save_field_artifacts(
        output_root / "true_field_artifacts.npz",
        truth_context["field_artifacts"],
    )
    true_interaction = truth_context.get("interaction_matrix")
    if true_interaction is None:
        return
    if sparse.issparse(true_interaction):
        sparse.save_npz(
            io_path(output_root / "true_interaction_matrix_sparse.npz"),
            true_interaction,
        )
    else:
        np.save(
            io_path(output_root / "true_interaction_matrix.npy"),
            true_interaction,
        )


def finalize_fit_outputs(
    output_root: str | Path,
    logger,
    est_theta: np.ndarray,
    loss_history: list[float],
    result: OptimizeResult,
    artifacts: ModelArtifacts,
    truth_context: dict[str, object] | None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Log, summarize, and persist all standard artifacts after a completed fit."""
    logger.info("Done fitting.")
    logger.info("Optimizer status: %s", result.message)
    logger.info(
        "Best optimizer start: %s / %s",
        int(result.get("best_start", 0)) + 1,
        int(result.get("n_starts", 1)),
    )
    logger.info("Final Loss: %.6f", loss_history[-1])
    scalar_rows = scalar_summary_rows(
        est_theta,
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
    metrics = _merge_optimizer_metrics(
        compute_truth_metrics(
            est_theta,
            artifacts,
            truth_context=truth_context,
            fixed_scalar_params=fixed_scalar_params,
        ),
        result,
    )
    _log_optimizer_metrics(logger, result, metrics)
    log_field_diagnostics(
        logger,
        metrics,
        est_theta,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    write_summary_table(
        Path(output_root) / "mple_summary",
        est_theta,
        metrics,
        loss_history[-1],
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    write_optimizer_start_summary(
        Path(output_root) / "optimizer_start_summary.csv",
        result,
    )
    save_estimated_artifacts(
        output_root,
        est_theta,
        result,
        artifacts,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
    return metrics
