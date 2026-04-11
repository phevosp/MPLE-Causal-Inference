"""Fit the active conditional MPLE model for synthetic and real-data experiments."""

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
    compose_field_matrix_from_theta,
    compose_interaction_matrix,
    infer_t_steps_from_theta,
    interaction_effect,
    intervention_model_enabled,
    load_model_artifacts,
    load_true_parameters,
    pack_theta,
    parameter_names,
    project_latent_field,
    save_field_artifacts,
    summary_metrics,
    unpack_theta,
    with_theta_field,
)


def _center_tau(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    return tau - tau.mean() if tau.size else tau


def _smoothness_penalty_and_grad(
    tau: np.ndarray, penalty_lambda: float
) -> tuple[float, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    grad = np.zeros_like(tau)
    if tau.size <= 1 or penalty_lambda <= 0.0:
        return 0.0, grad
    diffs = np.diff(tau)
    penalty = float(penalty_lambda * np.sum(diffs**2))
    grad[0] = -2.0 * penalty_lambda * diffs[0]
    grad[-1] = 2.0 * penalty_lambda * diffs[-1]
    if tau.size > 2:
        grad[1:-1] = 2.0 * penalty_lambda * (diffs[:-1] - diffs[1:])
    return penalty, grad


def _canonicalize_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    t_steps: int,
    fit_intervention_model: bool,
    tau_zero_mean: bool,
    latent_field_bound: float | None,
) -> np.ndarray:
    theta_parts = unpack_theta(theta, artifacts, t_steps, fit_intervention_model)
    if artifacts.field_mode == "latent_feature_matrix":
        if latent_field_bound is not None:
            node_factors, time_factors = project_latent_field(
                theta_parts["node_factors"],
                theta_parts["time_factors"],
                latent_field_bound,
            )
            theta_parts["node_factors"] = node_factors
            theta_parts["time_factors"] = time_factors
        return pack_theta(theta_parts, artifacts, fit_intervention_model)
    if tau_zero_mean and theta_parts["tau"] is not None:
        theta_parts["tau"] = _center_tau(theta_parts["tau"])
    return pack_theta(theta_parts, artifacts, fit_intervention_model)


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
    tau_zero_mean: bool = False,
    tau_smoothness_lambda: float = 0.0,
) -> tuple[float, np.ndarray]:
    t_steps = x.shape[0]
    theta_parts = unpack_theta(theta, artifacts, t_steps, fit_intervention_model)
    if (
        artifacts.field_mode != "latent_feature_matrix"
        and tau_zero_mean
        and theta_parts["tau"] is not None
    ):
        theta_parts["tau"] = _center_tau(theta_parts["tau"])

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([z_0, z[:-1, :]])
    field_matrix = compose_field_matrix_from_theta(theta_parts, artifacts)
    h_x = (
        field_matrix
        + theta_parts["beta"] * z
        + theta_parts["eta"] * prev_x
        + theta_parts["xi"] * interaction_effect_x
    )
    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x

    if fit_intervention_model:
        h_z = theta_parts["zeta"] * prev_x + theta_parts["psi"] * prev_z
        mask = np.ones_like(z)
        mask[:s, :] = 0
        total_size = x.size + mask.sum()
        total_loss = (
            loss_x.sum() + ((np.logaddexp(h_z, -h_z) - z * h_z) * mask).sum()
        ) / total_size
        res_z = (np.tanh(h_z) - z) * mask
        zeta_grad = float((res_z * prev_x).sum())
        psi_grad = float((res_z * prev_z).sum())
    else:
        total_size = x.size
        total_loss = loss_x.sum() / total_size
        zeta_grad = 0.0
        psi_grad = 0.0

    if artifacts.field_mode == "latent_feature_matrix":
        node_grad = res_x.T @ theta_parts["time_factors"]
        time_grad = res_x @ theta_parts["node_factors"]
        field_grad = np.concatenate([node_grad.reshape(-1), time_grad.reshape(-1)])
    else:
        tau_penalty, tau_penalty_grad = _smoothness_penalty_and_grad(
            theta_parts["tau"], float(tau_smoothness_lambda)
        )
        total_loss += tau_penalty / total_size
        tau_grad = res_x.sum(axis=1) + tau_penalty_grad
        if tau_zero_mean and tau_grad.size:
            tau_grad = tau_grad - tau_grad.mean()
        field_grad = np.concatenate(
            [artifacts.field_basis @ res_x.sum(axis=0), tau_grad]
        )

    grad_parts = [
        field_grad,
        np.array([float((res_x * z).sum())], dtype=float),
        np.array([float((res_x * interaction_effect_x).sum())], dtype=float),
        np.array([float((res_x * prev_x).sum())], dtype=float),
    ]
    if fit_intervention_model:
        grad_parts.append(np.array([zeta_grad, psi_grad], dtype=float))
    return float(total_loss), np.concatenate(grad_parts) / total_size


def summarize_theta_for_logging(param_names: list[str], theta: np.ndarray) -> str:
    tau_values = np.asarray(
        [value for name, value in zip(param_names, theta) if name.startswith("tau::")],
        dtype=float,
    )
    if tau_values.size == 0:
        return "  " + ",  ".join(
            f"{key}: {value:+.4f}" for key, value in zip(param_names, theta)
        )
    field_count = sum(name.startswith("field::") for name in param_names)
    non_tau_parts = [
        f"{key}: {value:+.4f}"
        for key, value in zip(param_names, theta)
        if not key.startswith("tau::")
    ]
    non_tau_parts.insert(
        field_count,
        f"tau block: mean={tau_values.mean():+.4f}, std={tau_values.std():.4f}, min={tau_values.min():+.4f}, max={tau_values.max():+.4f}",
    )
    return "  " + ",  ".join(non_tau_parts)


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
    tau_zero_mean: bool = False,
    tau_smoothness_lambda: float = 0.0,
    latent_field_bound: float | None = None,
):
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError("x and z must both have shape (T, N).")

    t_steps = x.shape[0]
    rng = np.random.default_rng(seed)
    theta_init = (
        rng.normal(0, 0.1, size=len(param_names))
        if theta_init is None
        else np.asarray(theta_init, dtype=float)
    )
    theta_init = _canonicalize_theta(
        theta_init,
        artifacts,
        t_steps,
        fit_intervention_model,
        tau_zero_mean,
        latent_field_bound,
    )
    history: list[float] = []
    eval_count = [0]

    def objective(theta):
        constrained_theta = _canonicalize_theta(
            theta,
            artifacts,
            t_steps,
            fit_intervention_model,
            tau_zero_mean,
            latent_field_bound,
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
            tau_zero_mean=tau_zero_mean,
            tau_smoothness_lambda=tau_smoothness_lambda,
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
        t_steps,
        fit_intervention_model,
        tau_zero_mean,
        latent_field_bound,
    )
    return theta_hat, history, result


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def write_summary_table(
    summary_stem, param_names, est_theta, true_theta, metrics, loss
):
    csv_path = Path(f"{summary_stem}.csv")
    md_path = Path(f"{summary_stem}.md")
    rows = [
        {
            "category": "parameter",
            "name": name,
            "estimate": float(est),
            "true": None if true is None else float(true),
            "squared_error": None if true is None else float((est - true) ** 2),
        }
        for name, est, true in zip(param_names, est_theta, true_theta)
    ]
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


def log_estimates(logger, title, param_names, est_theta, true_theta):
    logger.info(title)
    if all(true is None for true in true_theta):
        for key, est in zip(param_names, est_theta):
            logger.info("  %s: %.4f", key, est)
        return
    for key, est, true in zip(param_names, est_theta, true_theta):
        logger.info("  %s: %.4f (True: %.4f)", key, est, true)
        logger.info("  %s SQE: %.6f", key, (est - true) ** 2)


def save_estimated_artifacts(
    data_folder: str | Path,
    est_theta: np.ndarray,
    true_theta,
    artifacts: ModelArtifacts,
    fit_intervention_model: bool = True,
) -> None:
    t_steps = infer_t_steps_from_theta(est_theta, artifacts, fit_intervention_model)
    est_artifacts = with_theta_field(
        artifacts, unpack_theta(est_theta, artifacts, t_steps, fit_intervention_model)
    )
    save_field_artifacts(
        Path(data_folder) / "estimated_field_artifacts.npz", est_artifacts
    )
    estimated_interaction = compose_interaction_matrix(
        unpack_theta(est_theta, artifacts, t_steps, fit_intervention_model)["xi"],
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
    if all(true is None for true in true_theta):
        return
    true_artifacts = with_theta_field(
        artifacts,
        unpack_theta(
            np.asarray(true_theta, dtype=float),
            artifacts,
            t_steps,
            fit_intervention_model,
        ),
    )
    save_field_artifacts(Path(data_folder) / "true_field_artifacts.npz", true_artifacts)
    true_interaction = compose_interaction_matrix(
        unpack_theta(
            np.asarray(true_theta, dtype=float),
            artifacts,
            t_steps,
            fit_intervention_model,
        )["xi"],
        artifacts.gamma_matrix,
    )
    if sparse.issparse(true_interaction):
        sparse.save_npz(
            Path(data_folder) / "true_interaction_matrix_sparse.npz", true_interaction
        )
    else:
        np.save(Path(data_folder) / "true_interaction_matrix.npy", true_interaction)


def maybe_load_true_parameters(
    config, artifacts: ModelArtifacts, has_truth: bool, fit_intervention_model: bool
):
    if not has_truth:
        return None
    if artifacts.field_mode == "latent_feature_matrix":
        if artifacts.node_factors is None or artifacts.time_factors is None:
            return None
    else:
        if artifacts.field_coeffs is None or artifacts.tau is None:
            return None
    try:
        return load_true_parameters(config, artifacts, fit_intervention_model)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit active conditional-model parameters with MPLE."
    )
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--panel_path", type=str, default=None)
    parser.add_argument("--x0_path", type=str, default=None)
    parser.add_argument("--z0_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outcome_only", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    log_file = args.log_file or str(Path(args.data_folder) / "mple.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Loading data...")
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    fit_intervention_model = (
        intervention_model_enabled(config) and not args.outcome_only
    )
    tau_zero_mean = (
        bool(config.estimation_params.get("tau_zero_mean", False))
        if "estimation_params" in config
        else False
    )
    tau_smoothness_lambda = (
        float(config.estimation_params.get("tau_smoothness_lambda", 0.0))
        if "estimation_params" in config
        else 0.0
    )
    latent_field_bound = (
        float(config.global_params.B) if "B" in config.global_params else None
    )
    metadata_path = Path(args.data_folder) / "experiment_metadata.yaml"
    metadata = (
        OmegaConf.load(metadata_path)
        if metadata_path.exists()
        else OmegaConf.create({})
    )
    has_truth = bool(metadata.get("has_truth", True))

    panel_path = (
        Path(args.panel_path)
        if args.panel_path
        else Path(args.data_folder) / "panel_data.npz"
    )
    x0_path = Path(args.x0_path) if args.x0_path else Path(args.data_folder) / "x_0.npy"
    z0_path = Path(args.z0_path) if args.z0_path else Path(args.data_folder) / "z_0.npy"
    logger.info("Using panel artifact: %s", panel_path)
    logger.info("Using x_0 artifact: %s", x0_path)
    logger.info("Using z_0 artifact: %s", z0_path)
    x_0 = np.load(x0_path)
    z_0 = np.load(z0_path) if z0_path.exists() else np.zeros_like(x_0)
    panel = load_panel_artifact(panel_path)
    x = panel["x"]
    z = panel["z"]

    artifacts = load_model_artifacts(args.data_folder)
    param_keys = parameter_names(
        artifacts, x.shape[0], fit_intervention_model=fit_intervention_model
    )
    params_true = maybe_load_true_parameters(
        config, artifacts, has_truth, fit_intervention_model
    )
    interaction_effect_x = interaction_effect(x, artifacts.gamma_matrix)
    logger.info("Loaded field mode: %s", artifacts.field_mode)
    logger.info("Using a fixed known graph with scalar xi.")
    logger.info("Intervention-process model enabled: %s", fit_intervention_model)

    params_hat, loss_history, result = fit_mple(
        x,
        z,
        x_0=x_0,
        z_0=z_0,
        s=int(config.global_params.s),
        param_names=param_keys,
        artifacts=artifacts,
        interaction_effect_x=interaction_effect_x,
        steps=args.steps,
        tol=args.tol,
        seed=args.seed,
        logger=logger,
        fit_intervention_model=fit_intervention_model,
        tau_zero_mean=tau_zero_mean,
        tau_smoothness_lambda=tau_smoothness_lambda,
        latent_field_bound=latent_field_bound,
    )

    logger.info("Done fitting.")
    logger.info("Optimizer status: %s", result.message)
    logger.info("Final Loss: %.6f", loss_history[-1])
    truth_vector = params_true if params_true is not None else [None] * len(param_keys)
    log_estimates(
        logger,
        "Estimated vs True Parameters:" if has_truth else "Estimated Parameters:",
        param_keys,
        params_hat,
        truth_vector,
    )
    metrics = (
        summary_metrics(
            params_hat,
            params_true,
            artifacts,
            fit_intervention_model=fit_intervention_model,
        )
        if params_true is not None
        else {}
    )
    write_summary_table(
        Path(args.data_folder) / "mple_summary",
        param_keys,
        params_hat,
        truth_vector,
        metrics,
        loss_history[-1],
    )
    save_estimated_artifacts(
        args.data_folder,
        params_hat,
        truth_vector,
        artifacts,
        fit_intervention_model=fit_intervention_model,
    )
    logger.info(
        "Saved summary tables to %s and %s",
        Path(args.data_folder) / "mple_summary.csv",
        Path(args.data_folder) / "mple_summary.md",
    )
    logger.info("Log saved to %s", log_file)


if __name__ == "__main__":
    main()
