import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import minimize
from scipy import sparse

from model_utils import (
    BasisExpansion,
    compose_field,
    compose_field_matrix,
    compose_interaction_matrix,
    intervention_model_enabled,
    interaction_basis_count,
    interaction_features,
    load_or_build_basis,
    pack_true_parameters,
    parameter_names,
    scalar_parameter_count,
    summary_metrics,
    unpack_theta,
    validate_basis_infinity_norms,
)


def setup_logger(log_file):
    """Configure a logger that writes to both console and file."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_basis_artifacts(data_folder, config, gamma_matrix):
    """Load saved basis arrays from an experiment folder or rebuild legacy defaults."""
    data_path = Path(data_folder)
    field_basis_path = data_path / "field_basis.npy"
    interaction_basis_path = data_path / "interaction_basis.npy"
    interaction_basis_sparse_path = data_path / "interaction_basis_sparse.npz"
    field_names_path = data_path / "field_basis_names.npy"
    interaction_names_path = data_path / "interaction_basis_names.npy"
    shared_features_path = data_path / "shared_features.npy"
    shared_feature_names_path = data_path / "shared_feature_names.npy"

    if field_basis_path.exists() and (
        interaction_basis_path.exists() or interaction_basis_sparse_path.exists()
    ):
        field_basis = np.load(field_basis_path)
        if interaction_basis_sparse_path.exists():
            interaction_basis = sparse.load_npz(interaction_basis_sparse_path).tocsr()
        else:
            interaction_basis = np.load(interaction_basis_path)
        if field_names_path.exists():
            field_names = tuple(np.load(field_names_path).tolist())
        else:
            field_names = tuple(f"field_{idx}" for idx in range(field_basis.shape[0]))
        if interaction_names_path.exists():
            interaction_names = tuple(np.load(interaction_names_path).tolist())
        else:
            interaction_names = tuple(
                f"interaction_{idx}" for idx in range(interaction_basis_count(interaction_basis))
            )
        if shared_features_path.exists():
            shared_features = np.load(shared_features_path)
        else:
            shared_features = np.empty((0, field_basis.shape[1]), dtype=float)
        if shared_feature_names_path.exists():
            shared_feature_names = tuple(np.load(shared_feature_names_path).tolist())
        else:
            shared_feature_names = tuple(
                f"feature_{idx}" for idx in range(shared_features.shape[0])
            )
        validate_basis_infinity_norms(field_basis, interaction_basis)
        return BasisExpansion(
            field_basis=field_basis,
            interaction_basis=interaction_basis,
            field_names=field_names,
            interaction_names=interaction_names,
            shared_features=shared_features,
            shared_feature_names=shared_feature_names,
        )

    return load_or_build_basis(config, gamma_matrix)


def load_gamma_artifact(data_folder):
    """Load the realized network matrix, supporting dense and sparse storage."""
    data_path = Path(data_folder)
    dense_path = data_path / "gamma_matrix.npy"
    sparse_path = data_path / "gamma_matrix_sparse.npz"
    if sparse_path.exists():
        return sparse.load_npz(sparse_path).tocsr()
    if dense_path.exists():
        return np.load(dense_path)
    raise FileNotFoundError(f"Could not find gamma matrix artifact in {data_folder}.")


def load_panel_artifact(data_folder):
    """Load x and z panel arrays from either a real-data or synthetic-data folder."""
    data_path = Path(data_folder)
    panel_path = data_path / "panel_data.npz"
    legacy_path = data_path / "synthetic_data.npz"
    if panel_path.exists():
        return np.load(panel_path)
    if legacy_path.exists():
        return np.load(legacy_path)
    raise FileNotFoundError(
        f"Could not find panel_data.npz or synthetic_data.npz in {data_folder}."
    )


def _pack_gradient(
    field_grad,
    tau_grad,
    beta_grad,
    interaction_grad,
    eta_grad,
    total_size,
    fit_intervention_model=True,
    zeta_grad=0.0,
    psi_grad=0.0,
):
    """Assemble the flattened gradient vector in the optimizer's parameter order."""
    parts = [
        np.asarray(field_grad, dtype=float),
        np.asarray(tau_grad, dtype=float),
        np.array([beta_grad], dtype=float),
        np.asarray(interaction_grad, dtype=float),
        np.array([eta_grad], dtype=float),
    ]
    if fit_intervention_model:
        parts.append(np.array([zeta_grad, psi_grad], dtype=float))
    return np.concatenate(parts) / total_size


def pseudo_nll(
    x,
    z,
    theta,
    x_0,
    z_0,
    s,
    field_basis,
    interaction_features_x,
    fit_intervention_model=True,
):
    """Compute the conditional-model pseudo-NLL and its analytic gradient."""
    t_steps = x.shape[0]
    n_field = field_basis.shape[0]
    n_interaction = interaction_features_x.shape[0]
    field_coeffs, tau, beta, interaction_coeffs, eta, zeta, psi = unpack_theta(
        theta,
        n_field,
        n_interaction,
        t_steps,
        fit_intervention_model,
    )

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([z_0, z[:-1, :]])
    field_matrix = compose_field_matrix(field_coeffs, tau, field_basis)
    interaction_term = np.tensordot(
        interaction_coeffs,
        interaction_features_x,
        axes=(0, 0),
    )

    h_x = field_matrix + beta * z + eta * prev_x + interaction_term
    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x

    if fit_intervention_model:
        h_z = zeta * prev_x + psi * prev_z
        mask = np.ones_like(z)
        mask[:s, :] = 0
        loss_z = (np.logaddexp(h_z, -h_z) - z * h_z) * mask
        res_z = (np.tanh(h_z) - z) * mask
        total_size = x.size + mask.sum()
        total_loss = (loss_x.sum() + loss_z.sum()) / total_size
        zeta_grad = (res_z * prev_x).sum()
        psi_grad = (res_z * prev_z).sum()
    else:
        total_size = x.size
        total_loss = loss_x.sum() / total_size
        zeta_grad = 0.0
        psi_grad = 0.0

    grad = _pack_gradient(
        field_grad=field_basis @ res_x.sum(axis=0),
        tau_grad=res_x.sum(axis=1),
        beta_grad=(res_x * z).sum(),
        interaction_grad=np.einsum(
            "tn,ktn->k", res_x, interaction_features_x, optimize=True
        ),
        eta_grad=(res_x * prev_x).sum(),
        total_size=total_size,
        fit_intervention_model=fit_intervention_model,
        zeta_grad=zeta_grad,
        psi_grad=psi_grad,
    )

    return total_loss, grad


def fit_mple(
    x,
    z,
    x_0,
    z_0,
    s,
    param_names,
    field_basis,
    interaction_features_x,
    steps=2000,
    seed=0,
    verbose_every=100,
    tol=1e-9,
    logger=None,
    theta_init=None,
    fit_intervention_model=True,
):
    """Optimize the conditional pseudo-likelihood with L-BFGS-B and track the loss history."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (T, N).")
    t_steps, n_nodes = x.shape
    assert z.shape == (t_steps, n_nodes), "z must have the same shape as x."

    rng = np.random.default_rng(seed)
    theta_init = (
        rng.normal(0, 0.1, size=len(param_names))
        if theta_init is None
        else np.asarray(theta_init, dtype=float)
    )

    history = []
    eval_count = [0]

    def objective(theta):
        """Wrap the objective so scipy receives both loss and gradient."""
        loss, grad = pseudo_nll(
            x,
            z,
            theta,
            x_0,
            z_0,
            s,
            field_basis=field_basis,
            interaction_features_x=interaction_features_x,
            fit_intervention_model=fit_intervention_model,
        )
        history.append(loss)
        if verbose_every and eval_count[0] % verbose_every == 0:
            params_str = summarize_theta_for_logging(param_names, theta)
            if logger is not None:
                logger.info("Eval %s  |  Loss: %.6f", eval_count[0], loss)
                logger.info(params_str)
            else:
                print(f"Eval {eval_count[0]}  |  Loss: {loss:.6f}")
                print(params_str)
        eval_count[0] += 1
        return loss, grad

    result = minimize(
        objective,
        theta_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": steps, "ftol": tol, "gtol": tol},
    )

    return result.x, history, result


def _fmt(value):
    """Format numeric summary-table entries while leaving missing values blank."""
    if value is None:
        return ""
    return f"{float(value):.6f}"


def summarize_theta_for_logging(param_names, theta):
    """Summarize parameter blocks compactly, collapsing the temporal field vector."""
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
        (
            "tau block: "
            f"mean={tau_values.mean():+.4f}, std={tau_values.std():.4f}, "
            f"min={tau_values.min():+.4f}, max={tau_values.max():+.4f}"
        ),
    )
    return "  " + ",  ".join(non_tau_parts)


def write_summary_table(summary_stem, param_names, est_theta, true_theta, metrics, loss):
    """Write CSV and Markdown summaries for one fitted estimator."""
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
            handle,
            fieldnames=["category", "name", "estimate", "true", "squared_error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| category | name | estimate | true | squared_error |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['category']} | {row['name']} | {_fmt(row['estimate'])} | "
                f"{_fmt(row['true'])} | {_fmt(row['squared_error'])} |\n"
            )


def log_estimates(logger, title, param_names, est_theta, true_theta):
    """Log estimate-versus-truth comparisons for a fitted parameter vector."""
    logger.info(title)
    if all(true is None for true in true_theta):
        for key, est in zip(param_names, est_theta):
            logger.info("  %s: %.4f", key, est)
        return
    tau_rows = []
    for key, est, true in zip(param_names, est_theta, true_theta):
        if key.startswith("tau::"):
            tau_rows.append((est, true))
            continue
        logger.info("  %s: %.4f (True: %.4f)", key, est, true)
        logger.info("  %s SQE: %.6f", key, (est - true) ** 2)
    if tau_rows:
        tau_est = np.asarray([row[0] for row in tau_rows], dtype=float)
        tau_true = np.asarray([row[1] for row in tau_rows], dtype=float)
        logger.info(
            "  tau block: RMSE %.6f | L2 error %.6f | mean(est)=%.4f | mean(true)=%.4f",
            float(np.sqrt(np.mean((tau_est - tau_true) ** 2))),
            float(np.linalg.norm(tau_est - tau_true, ord=2)),
            float(tau_est.mean()),
            float(tau_true.mean()),
        )


def save_estimated_artifacts(
    data_folder,
    est_theta,
    true_theta,
    field_basis,
    interaction_basis,
    fit_intervention_model=True,
):
    """Save reconstructed field and interaction objects for fitted and true parameters."""
    t_steps = int(
        len(est_theta)
        - field_basis.shape[0]
        - interaction_basis_count(interaction_basis)
        - scalar_parameter_count(fit_intervention_model)
    )
    n_field = field_basis.shape[0]
    n_interaction = interaction_basis_count(interaction_basis)
    est_field_coeffs, est_tau, _, est_interaction_coeffs, _, _, _ = unpack_theta(
        est_theta,
        n_field,
        n_interaction,
        t_steps,
        fit_intervention_model,
    )

    np.save(Path(data_folder) / "estimated_tau.npy", est_tau)
    np.save(
        Path(data_folder) / "estimated_field_vector.npy",
        compose_field(est_field_coeffs, field_basis),
    )
    np.save(
        Path(data_folder) / "estimated_field_matrix.npy",
        compose_field_matrix(est_field_coeffs, est_tau, field_basis),
    )
    estimated_interaction = compose_interaction_matrix(est_interaction_coeffs, interaction_basis)
    if sparse.issparse(estimated_interaction):
        sparse.save_npz(
            Path(data_folder) / "estimated_interaction_matrix_sparse.npz",
            estimated_interaction,
        )
    else:
        np.save(Path(data_folder) / "estimated_interaction_matrix.npy", estimated_interaction)

    if all(true is None for true in true_theta):
        return

    true_theta_array = np.asarray(true_theta, dtype=float)
    true_field_coeffs, true_tau, _, true_interaction_coeffs, _, _, _ = unpack_theta(
        true_theta_array,
        n_field,
        n_interaction,
        t_steps,
        fit_intervention_model,
    )
    np.save(Path(data_folder) / "true_tau.npy", true_tau)
    np.save(
        Path(data_folder) / "true_field_vector.npy",
        compose_field(true_field_coeffs, field_basis),
    )
    np.save(
        Path(data_folder) / "true_field_matrix.npy",
        compose_field_matrix(true_field_coeffs, true_tau, field_basis),
    )
    true_interaction = compose_interaction_matrix(true_interaction_coeffs, interaction_basis)
    if sparse.issparse(true_interaction):
        sparse.save_npz(
            Path(data_folder) / "true_interaction_matrix_sparse.npz",
            true_interaction,
        )
    else:
        np.save(Path(data_folder) / "true_interaction_matrix.npy", true_interaction)

def maybe_pack_true_parameters(
    config,
    field_names,
    interaction_names,
    t_steps,
    has_truth,
    fit_intervention_model,
):
    """Return the true parameter vector when available, otherwise None."""
    if not has_truth:
        return None
    if "estimation_params" not in config:
        return None
    if "field_coefs" not in config.estimation_params:
        return None
    try:
        packed = pack_true_parameters(
            config,
            field_names,
            interaction_names,
            fit_intervention_model=fit_intervention_model,
        )
    except Exception:
        return None
    expected_length = (
        len(field_names)
        + t_steps
        + len(interaction_names)
        + scalar_parameter_count(fit_intervention_model)
    )
    if len(packed) != expected_length:
        return None
    return packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit conditional-model parameters with MPLE."
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        type=str,
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outcome_only",
        action="store_true",
        help="Fit only the x-outcome pseudo-likelihood and do not model the intervention process z.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file. Defaults to <data_folder>/mple.log",
    )
    args = parser.parse_args()

    log_file = args.log_file or str(Path(args.data_folder) / "mple.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Loading data...")
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    fit_intervention_model = intervention_model_enabled(config) and not args.outcome_only
    metadata_path = Path(args.data_folder) / "experiment_metadata.yaml"
    metadata = OmegaConf.load(metadata_path) if metadata_path.exists() else OmegaConf.create({})
    has_truth = bool(metadata.get("has_truth", True))
    gamma_matrix = load_gamma_artifact(args.data_folder)
    x_0 = np.load(f"{args.data_folder}/x_0.npy")
    z_0_path = Path(args.data_folder) / "z_0.npy"
    z_0 = np.load(z_0_path) if z_0_path.exists() else np.zeros_like(x_0)
    data = load_panel_artifact(args.data_folder)
    x = data["x"]
    z = data["z"]

    basis = load_basis_artifacts(args.data_folder, config, gamma_matrix)
    param_keys = parameter_names(
        basis.field_names,
        basis.interaction_names,
        x.shape[0],
        fit_intervention_model=fit_intervention_model,
    )
    params_true = maybe_pack_true_parameters(
        config,
        basis.field_names,
        basis.interaction_names,
        x.shape[0],
        has_truth,
        fit_intervention_model,
    )
    interaction_features_x = interaction_features(x, basis.interaction_basis)
    logger.info(
        "Loaded %s field templates and %s interaction templates.",
        len(basis.field_names),
        len(basis.interaction_names),
    )
    logger.info(
        "Intervention-process model enabled: %s",
        fit_intervention_model,
    )

    logger.info("Running conditional-model MPLE on x with shape=%s", x.shape)
    params_hat, loss_history, result = fit_mple(
        x,
        z,
        x_0=x_0,
        z_0=z_0,
        s=config.global_params.s,
        param_names=param_keys,
        field_basis=basis.field_basis,
        interaction_features_x=interaction_features_x,
        steps=args.steps,
        tol=args.tol,
        seed=args.seed,
        logger=logger,
        fit_intervention_model=fit_intervention_model,
    )

    logger.info("Done fitting.")
    logger.info("Optimizer status: %s", result.message)
    logger.info("Final Loss: %.6f", loss_history[-1])
    log_estimates(
        logger,
        "Estimated vs True Parameters:" if has_truth else "Estimated Parameters:",
        param_keys,
        params_hat,
        params_true if params_true is not None else [None] * len(param_keys),
    )

    metrics = (
        summary_metrics(
            params_hat,
            params_true,
            basis.field_basis,
            basis.interaction_basis,
            fit_intervention_model=fit_intervention_model,
        )
        if params_true is not None
        else {}
    )
    write_summary_table(
        Path(args.data_folder) / "mple_summary",
        param_keys,
        params_hat,
        params_true if params_true is not None else [None] * len(param_keys),
        metrics,
        loss_history[-1],
    )
    save_estimated_artifacts(
        args.data_folder,
        params_hat,
        params_true if params_true is not None else [None] * len(param_keys),
        basis.field_basis,
        basis.interaction_basis,
        fit_intervention_model=fit_intervention_model,
    )
    logger.info(
        "Saved summary tables to %s and %s",
        Path(args.data_folder) / "mple_summary.csv",
        Path(args.data_folder) / "mple_summary.md",
    )
    logger.info("Log saved to %s", log_file)
