"""Fit the latent-only conditional MPLE model for synthetic and real-data experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from scipy.optimize import minimize

from model_utils import (
    ModelArtifacts,
    build_fit_model_artifacts,
    compose_field_matrix_from_theta,
    compose_interaction_matrix,
    free_scalar_parameter_names,
    interaction_effect,
    interaction_matrix_infinity_norm,
    intervention_model_enabled,
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
                np.clip(float(theta_parts["xi"]), -effective_xi_bound, effective_xi_bound)
            )
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
    beta_mask_rescale: bool = False,
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
    beta_scale = 1.0
    if beta_mask_pre_intervention:
        beta_mask = np.ones_like(beta_feature)
        beta_mask[:s, :] = 0.0
        beta_feature = beta_feature * beta_mask
        if beta_mask_rescale:
            active = float(beta_mask.sum())
            if active > 0.0:
                beta_scale = float(beta_mask.size / active)
    beta_feature *= beta_scale
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
        total_loss = (
            loss_x.sum() + ((np.logaddexp(h_z, -h_z) - z * h_z) * mask).sum()
        ) / (outcome_size + intervention_size)
        res_z = (np.tanh(h_z) - z) * mask
        zeta_grad = float((res_z * prev_x).sum()) / intervention_size
        psi_grad = float((res_z * prev_z).sum()) / intervention_size
    else:
        outcome_size = x.size
        total_loss = loss_x.sum() / outcome_size
        zeta_grad = 0.0
        psi_grad = 0.0

    node_grad = res_x.T @ theta_parts["time_factors"]
    time_grad = res_x @ theta_parts["node_factors"]
    field_grad = np.concatenate([node_grad.reshape(-1), time_grad.reshape(-1)])

    scalar_grad_lookup = {
        "beta": float((res_x * beta_feature).sum()) / outcome_size,
        "xi": float((res_x * interaction_effect_x).sum()) / outcome_size,
        "eta": float((res_x * prev_x).sum()) / outcome_size,
        "zeta": float(zeta_grad),
        "psi": float(psi_grad),
    }
    grad_parts = [field_grad / outcome_size]
    for name in free_scalar_parameter_names(
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    ):
        grad_parts.append(np.array([scalar_grad_lookup[name]], dtype=float))
    return float(total_loss), np.concatenate(grad_parts)


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
    beta_mask_rescale: bool = False,
    fixed_scalar_params: dict[str, float] | None = None,
):
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError("x and z must both have shape (T, N).")

    t_steps = x.shape[0]
    if t_steps != artifacts.t_steps:
        raise ValueError("Panel length does not match artifact t_steps.")
    rng = np.random.default_rng(seed)
    theta_init = (
        rng.normal(0, 0.1, size=len(param_names))
        if theta_init is None
        else np.asarray(theta_init, dtype=float)
    )
    theta_init = _canonicalize_theta(
        theta_init,
        artifacts,
        fit_intervention_model,
        bound_B,
        fixed_scalar_params=fixed_scalar_params,
    )
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
            beta_mask_rescale=beta_mask_rescale,
            fixed_scalar_params=fixed_scalar_params,
        )
        history.append(loss)
        if verbose_every and eval_count[0] % verbose_every == 0:
            message = summarize_theta_for_logging(param_names, constrained_theta)
            if logger is None:
                print(f"Eval {eval_count[0]}  |  Loss: {loss:.6f}")
                print(message)
            else:
                logger.info("Eval %s  |  Loss: %.6f", eval_count[0], loss)
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
    est_interaction = compose_interaction_matrix(est_parts["xi"], artifacts.gamma_matrix)
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
            "estimate": float(np.linalg.norm(est_field, ord=np.inf)),
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
                    "estimate": float(np.linalg.norm(true_field, ord=np.inf)),
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
    est_artifacts = with_theta_field(
        artifacts,
        unpack_theta(
            est_theta,
            artifacts,
            fit_intervention_model,
            fixed_scalar_params=fixed_scalar_params,
        ),
    )
    save_field_artifacts(
        Path(data_folder) / "estimated_field_artifacts.npz", est_artifacts
    )
    estimated_interaction = compose_interaction_matrix(
        unpack_theta(
            est_theta,
            artifacts,
            fit_intervention_model,
            fixed_scalar_params=fixed_scalar_params,
        )["xi"],
        artifacts.gamma_matrix,
    )
    if sparse.issparse(estimated_interaction):
        sparse.save_npz(
            Path(data_folder) / "estimated_interaction_matrix_sparse.npz",
            estimated_interaction,
        )
    else:
        np.save(
            Path(data_folder) / "estimated_interaction_matrix.npy",
            estimated_interaction,
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
            Path(data_folder) / "true_interaction_matrix_sparse.npz", true_interaction
        )
    else:
        np.save(Path(data_folder) / "true_interaction_matrix.npy", true_interaction)


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
    beta_mask_rescale = (
        bool(config.estimation_params.get("beta_mask_rescale", False))
        if "estimation_params" in config
        else False
    )
    bound_B = (
        float(config.global_params.B) if "B" in config.global_params else None
    )
    optimizer_params = (
        config.optimizer_params
        if "optimizer_params" in config
        else OmegaConf.create({})
    )
    steps = int(args.steps if args.steps is not None else optimizer_params.get("steps", 10000))
    tol = float(args.tol if args.tol is not None else optimizer_params.get("tol", 1e-9))
    seed = int(args.seed if args.seed is not None else optimizer_params.get("seed", 0))
    model_artifact_dir = (
        Path(args.model_artifact_dir) if args.model_artifact_dir else Path(args.data_folder)
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
    logger.info("Beta mask rescale enabled: %s", beta_mask_rescale)
    logger.info("Fixed scalar parameters: %s", fixed_scalar_params or {})

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
        beta_mask_rescale=beta_mask_rescale,
        fixed_scalar_params=fixed_scalar_params,
    )

    logger.info("Done fitting.")
    logger.info("Optimizer status: %s", result.message)
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
        "Estimated vs True Parameters:"
        if truth_context is not None
        else "Estimated Parameters:",
        scalar_rows,
    )
    metrics = compute_truth_metrics(
        params_hat,
        artifacts,
        fit_intervention_model=fit_intervention_model,
        truth_context=truth_context,
        fixed_scalar_params=fixed_scalar_params,
    )
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
