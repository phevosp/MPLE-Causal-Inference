import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import minimize

from model_utils import (
    BasisExpansion,
    compose_field,
    interaction_features,
    load_or_build_basis,
    pack_true_parameters,
    parameter_names,
    summary_metrics,
    unpack_theta,
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
    field_names_path = data_path / "field_basis_names.npy"
    interaction_names_path = data_path / "interaction_basis_names.npy"

    if field_basis_path.exists() and interaction_basis_path.exists():
        field_basis = np.load(field_basis_path)
        interaction_basis = np.load(interaction_basis_path)
        if field_names_path.exists():
            field_names = tuple(np.load(field_names_path).tolist())
        else:
            field_names = tuple(f"field_{idx}" for idx in range(field_basis.shape[0]))
        if interaction_names_path.exists():
            interaction_names = tuple(np.load(interaction_names_path).tolist())
        else:
            interaction_names = tuple(
                f"interaction_{idx}" for idx in range(interaction_basis.shape[0])
            )
        return BasisExpansion(
            field_basis=field_basis,
            interaction_basis=interaction_basis,
            field_names=field_names,
            interaction_names=interaction_names,
        )

    return load_or_build_basis(config, gamma_matrix)


def _pack_gradient(
    field_grad,
    beta_grad,
    interaction_grad,
    eta_grad,
    zeta_grad,
    psi_grad,
    total_size,
):
    """Assemble the flattened gradient vector in the optimizer's parameter order."""
    return np.concatenate(
        [
            np.asarray(field_grad, dtype=float),
            np.array([beta_grad], dtype=float),
            np.asarray(interaction_grad, dtype=float),
            np.array([eta_grad, zeta_grad, psi_grad], dtype=float),
        ]
    ) / total_size


def conditional_model_pseudo_nll(
    x,
    z,
    theta,
    x_0,
    s,
    field_basis,
    interaction_features_x,
):
    """Compute the conditional-model pseudo-NLL and its analytic gradient."""
    n_field = field_basis.shape[0]
    n_interaction = interaction_features_x.shape[0]
    field_coeffs, beta, interaction_coeffs, eta, zeta, psi = unpack_theta(
        theta,
        n_field,
        n_interaction,
    )

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([np.zeros_like(x_0), z[:-1, :]])
    field_vector = compose_field(field_coeffs, field_basis)
    interaction_term = np.tensordot(
        interaction_coeffs,
        interaction_features_x,
        axes=(0, 0),
    )

    h_x = field_vector[None, :] + beta * z + eta * prev_x + interaction_term
    h_z = zeta * prev_x + psi * prev_z

    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x

    mask = np.ones_like(z)
    mask[:s, :] = 0

    loss_z = (np.logaddexp(h_z, -h_z) - z * h_z) * mask
    res_z = (np.tanh(h_z) - z) * mask

    total_size = x.size + mask.sum()
    total_loss = (loss_x.sum() + loss_z.sum()) / total_size

    grad = _pack_gradient(
        field_grad=field_basis @ res_x.sum(axis=0),
        beta_grad=(res_x * z).sum(),
        interaction_grad=np.einsum(
            "tn,ktn->k", res_x, interaction_features_x, optimize=True
        ),
        eta_grad=(res_x * prev_x).sum(),
        zeta_grad=(res_z * prev_x).sum(),
        psi_grad=(res_z * prev_z).sum(),
        total_size=total_size,
    )

    return total_loss, grad


def pseudo_nll(
    x,
    z,
    theta,
    x_0,
    s,
    field_basis,
    interaction_features_x,
    conditioning=False,
):
    """Compute the joint Ising pseudo-NLL and its gradient for either estimation stage."""
    n_field = field_basis.shape[0]
    n_interaction = interaction_features_x.shape[0]
    field_coeffs, beta, interaction_coeffs, eta, zeta, psi = unpack_theta(
        theta,
        n_field,
        n_interaction,
    )

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([np.zeros_like(x_0), z[:-1, :]])
    future_x = np.vstack([x[1:, :], np.zeros_like(x_0)])
    future_z = np.vstack([z[1:, :], np.zeros_like(x_0)])
    field_vector = compose_field(field_coeffs, field_basis)
    interaction_term = np.tensordot(
        interaction_coeffs,
        interaction_features_x,
        axes=(0, 0),
    )

    future_z_masked = future_z.copy()
    if s > 1:
        future_z_masked[: s - 1, :] = 0

    h_z = zeta * prev_x + beta * x + psi * (prev_z + future_z)
    h_x = (
        field_vector[None, :]
        + eta * (prev_x + future_x)
        + beta * z
        + interaction_term
        + zeta * future_z_masked
    )

    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    loss_z = np.logaddexp(h_z, -h_z) - z * h_z
    res_x = np.tanh(h_x) - x
    res_z = np.tanh(h_z) - z

    if not conditioning:
        mask = np.ones_like(z)
        mask[:s, :] = 0
        loss_z_masked = loss_z * mask
        res_z_masked = res_z * mask
        total_size = loss_x.size + mask.sum()
        total_loss = (loss_x.sum() + loss_z_masked.sum()) / total_size
        grad = _pack_gradient(
            field_grad=field_basis @ res_x.sum(axis=0),
            beta_grad=(res_x * z).sum() + (res_z_masked * x).sum(),
            interaction_grad=np.einsum(
                "tn,ktn->k", res_x, interaction_features_x, optimize=True
            ),
            eta_grad=(res_x * (prev_x + future_x)).sum(),
            zeta_grad=(res_x * future_z_masked).sum()
            + (res_z_masked * prev_x).sum(),
            psi_grad=(res_z_masked * (prev_z + future_z)).sum(),
            total_size=total_size,
        )
    else:
        total_loss = loss_x[0::2, :].mean()
        total_size = res_x[0::2, :].size
        grad = _pack_gradient(
            field_grad=field_basis @ res_x[0::2, :].sum(axis=0),
            beta_grad=(res_x[0::2, :] * z[0::2, :]).sum(),
            interaction_grad=np.einsum(
                "tn,ktn->k",
                res_x[0::2, :],
                interaction_features_x[:, 0::2, :],
                optimize=True,
            ),
            eta_grad=(res_x[0::2, :] * (prev_x[0::2, :] + future_x[0::2, :])).sum(),
            zeta_grad=(res_x[0::2, :] * future_z_masked[0::2, :]).sum(),
            psi_grad=0.0,
            total_size=total_size,
        )
    return total_loss, grad


def mple_gradient_descent(
    x,
    z,
    x_0,
    s,
    loss_fn,
    param_names,
    loss_fn_kwargs=None,
    steps=2000,
    seed=0,
    verbose_every=100,
    tol=1e-9,
    logger=None,
    theta_init=None,
):
    """Optimize a pseudo-likelihood objective with L-BFGS-B and track the loss history."""
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
    loss_fn_kwargs = loss_fn_kwargs or {}

    history = []
    eval_count = [0]

    def objective(theta):
        """Wrap the objective so scipy receives both loss and gradient."""
        loss, grad = loss_fn(
            x,
            z,
            theta,
            x_0,
            s,
            **loss_fn_kwargs,
        )
        history.append(loss)
        if verbose_every and eval_count[0] % verbose_every == 0:
            params_str = "  " + ",  ".join(
                f"{key}: {value:+.4f}" for key, value in zip(param_names, theta)
            )
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


def write_summary_table(summary_stem, param_names, est_theta, true_theta, metrics, loss):
    """Write CSV and Markdown summaries for one fitted estimator."""
    csv_path = Path(f"{summary_stem}.csv")
    md_path = Path(f"{summary_stem}.md")
    rows = [
        {
            "category": "parameter",
            "name": name,
            "estimate": float(est),
            "true": float(true),
            "squared_error": float((est - true) ** 2),
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
    for key, est, true in zip(param_names, est_theta, true_theta):
        logger.info("  %s: %.4f (True: %.4f)", key, est, true)
        logger.info("  %s SQE: %.6f", key, (est - true) ** 2)


def combine_two_stage_estimates(stage1_theta, stage2_theta, n_field):
    """Use stage-2 field estimates together with stage-1 non-field parameters."""
    combined = stage1_theta.copy()
    combined[:n_field] = stage2_theta[:n_field]
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Ising parameters with MPLE gradient descent."
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
        "--log_file",
        type=str,
        default=None,
        help="Path to log file. Defaults to <data_folder>/mple_full.log",
    )
    parser.add_argument(
        "--use_conditional_npll",
        action="store_true",
        help="Use conditional negative log-likelihood.",
    )
    args = parser.parse_args()

    log_file = args.log_file or str(
        Path(args.data_folder)
        / ("mple_conditional.log" if args.use_conditional_npll else "mple.log")
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Loading data...")
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    gamma_matrix = np.load(f"{args.data_folder}/gamma_matrix.npy")
    x_0 = np.load(f"{args.data_folder}/x_0.npy")
    data = np.load(f"{args.data_folder}/synthetic_data.npz")
    x = data["x"]
    z = data["z"]

    basis = load_basis_artifacts(args.data_folder, config, gamma_matrix)
    param_keys = parameter_names(basis.field_names, basis.interaction_names)
    params_true = pack_true_parameters(
        config,
        basis.field_names,
        basis.interaction_names,
    )
    loss_fn_kwargs = {
        "field_basis": basis.field_basis,
        "interaction_features_x": interaction_features(x, basis.interaction_basis),
    }
    logger.info(
        "Loaded %s field templates and %s interaction templates.",
        len(basis.field_names),
        len(basis.interaction_names),
    )

    if args.use_conditional_npll:
        logger.info("Running conditional-model MPLE on x with shape=%s", x.shape)
        params_hat_conditional, loss_history_conditional, result = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            loss_fn=conditional_model_pseudo_nll,
            param_names=param_keys,
            loss_fn_kwargs=loss_fn_kwargs,
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )

        logger.info("Done fitting.")
        logger.info("Optimizer status: %s", result.message)
        logger.info(
            "Final Loss (Conditional Model): %.6f", loss_history_conditional[-1]
        )
        log_estimates(
            logger,
            "Estimated vs True Parameters (Conditional Model):",
            param_keys,
            params_hat_conditional,
            params_true,
        )

        metrics = summary_metrics(
            params_hat_conditional,
            params_true,
            basis.field_basis,
            basis.interaction_basis,
        )
        write_summary_table(
            Path(args.data_folder) / "mple_conditional_summary",
            param_keys,
            params_hat_conditional,
            params_true,
            metrics,
            loss_history_conditional[-1],
        )
        logger.info(
            "Saved summary tables to %s and %s",
            Path(args.data_folder) / "mple_conditional_summary.csv",
            Path(args.data_folder) / "mple_conditional_summary.md",
        )
    else:
        logger.info("Running Stage 1 MPLE on x with shape=%s", x.shape)
        params_hat, loss_history, result_stage1 = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            loss_fn=pseudo_nll,
            param_names=param_keys,
            loss_fn_kwargs={**loss_fn_kwargs, "conditioning": False},
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )
        logger.info("Running Stage 2 MPLE (conditioning on all z's and even x's)...")
        params_hat_cond, loss_history_cond, result_stage2 = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            loss_fn=pseudo_nll,
            param_names=param_keys,
            loss_fn_kwargs={**loss_fn_kwargs, "conditioning": True},
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )

        combined_params = combine_two_stage_estimates(
            params_hat,
            params_hat_cond,
            basis.field_basis.shape[0],
        )

        logger.info("Done fitting.")
        logger.info("Stage 1 status: %s", result_stage1.message)
        logger.info("Stage 2 status: %s", result_stage2.message)
        logger.info("Final Loss (Unconditioned): %.6f", loss_history[-1])
        log_estimates(
            logger,
            "Estimated vs True Parameters (Unconditioned):",
            param_keys,
            params_hat,
            params_true,
        )
        logger.info("=================================================")
        logger.info("Final Loss (Conditioned): %.6f", loss_history_cond[-1])
        log_estimates(
            logger,
            "Estimated vs True Parameters (Conditioned):",
            param_keys,
            params_hat_cond,
            params_true,
        )
        logger.info("=================================================")
        log_estimates(
            logger,
            "Combined Two-Stage Estimate:",
            param_keys,
            combined_params,
            params_true,
        )

        write_summary_table(
            Path(args.data_folder) / "mple_stage1_summary",
            param_keys,
            params_hat,
            params_true,
            summary_metrics(
                params_hat,
                params_true,
                basis.field_basis,
                basis.interaction_basis,
            ),
            loss_history[-1],
        )
        write_summary_table(
            Path(args.data_folder) / "mple_stage2_summary",
            param_keys,
            params_hat_cond,
            params_true,
            summary_metrics(
                params_hat_cond,
                params_true,
                basis.field_basis,
                basis.interaction_basis,
            ),
            loss_history_cond[-1],
        )
        write_summary_table(
            Path(args.data_folder) / "mple_combined_summary",
            param_keys,
            combined_params,
            params_true,
            summary_metrics(
                combined_params,
                params_true,
                basis.field_basis,
                basis.interaction_basis,
            ),
            loss_history_cond[-1],
        )
        logger.info("Saved stage summaries to %s", Path(args.data_folder))

    logger.info("Log saved to %s", log_file)
