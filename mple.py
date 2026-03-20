import argparse
import logging
from pathlib import Path
import numpy as np
import networkx as nx
from omegaconf import OmegaConf
from scipy.optimize import minimize


def setup_logger(log_file):
    """Configure a logger that writes to both console and file."""
    logger = logging.getLogger("mple_full")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times.
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


# Canonical parameter ordering used to pack/unpack the flat vector for scipy.
_PARAM_KEYS = ("alpha", "beta", "xi", "eta", "zeta", "psi")


def conditional_model_pseudo_nll(x, z, theta, x_0, s, gamma_matrix):
    """The negative log-psuedolikelihood and gradient assuming the conditional generative process

    Args:
        x (np.ndarray): shape (T, N), outcomes in {-1, +1}.
        z (np.ndarray): shape (T, N), interventions in {-1, +1}.
        x_0 (np.ndarray): shape (N,), initial state at t=0.
        s (int): number of initial time steps where z is fixed.
        gamma_matrix (np.ndarray): shape (N, N), normalised network adjacency matrix.

    Returns:
        tuple[float, np.ndarray]: (loss, gradient)
    """

    alpha, beta, xi, eta, zeta, psi = theta

    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([np.zeros_like(x_0), z[:-1, :]])
    m = x @ gamma_matrix.T

    # matched to p(x^(t) | z^(t), x^(t-1))
    h_x = alpha + beta * z + eta * prev_x + xi * m

    # matched to p(z^(t) | x^(t-1), z^(t-1))
    h_z = zeta * prev_x + psi * prev_z

    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    res_x = np.tanh(h_x) - x

    mask = np.ones_like(z)
    mask[:s, :] = 0  # z^(t) fixed for t <= s

    loss_z = (np.logaddexp(h_z, -h_z) - z * h_z) * mask
    res_z = (np.tanh(h_z) - z) * mask

    total_size = x.size + mask.sum()
    total_loss = (loss_x.sum() + loss_z.sum()) / total_size

    grad = (
        np.array(
            [
                res_x.sum(),  # alpha
                (res_x * z).sum(),  # beta
                (res_x * m).sum(),  # xi
                (res_x * prev_x).sum(),  # eta
                (res_z * prev_x).sum(),  # zeta
                (res_z * prev_z).sum(),  # psi
            ]
        )
        / total_size
    )

    return total_loss, grad


def pseudo_nll(x, z, theta, x_0, s, gamma_matrix, conditioning=False):
    """Compute Ising negative log-pseudolikelihood averaged over samples.

    Args:
        x (np.ndarray): shape (T, N), outcomes in {-1, +1}.
        z (np.ndarray): shape (T, N), interventions in {-1, +1}.
        x_0 (np.ndarray): shape (N,), initial state at t=0.
        s (int): number of initial time steps where z is fixed.
        gamma_matrix (np.ndarray): shape (N, N), normalised network adjacency matrix.
        conditioning (bool): if True, compute the pseudo-nll conditioning on all z's and even x's

    Returns:
        tuple[float, np.ndarray]: (loss, gradient)
    """
    alpha, beta, xi, eta, zeta, psi = theta
    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([np.zeros_like(x_0), z[:-1, :]])
    future_x = np.vstack([x[1:, :], np.zeros_like(x_0)])
    future_z = np.vstack([z[1:, :], np.zeros_like(x_0)])
    m = x @ gamma_matrix.T

    # Mask out future z's that are not generated for the first s time steps
    future_z_masked = future_z.copy()
    if s > 1:
        future_z_masked[: s - 1, :] = 0

    # Compute fields for all z
    h_z = psi * prev_z + zeta * prev_x + beta * x + psi * future_z
    # Compute fields for all x
    h_x = (
        alpha
        + eta * prev_x
        + beta * z
        + xi * m
        + zeta
        * future_z_masked  # mask future_z's that are not generated for the first s time steps
        + eta * future_x
    )

    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    loss_z = np.logaddexp(h_z, -h_z) - z * h_z

    res_x = np.tanh(h_x) - x
    res_z = np.tanh(h_z) - z

    if not conditioning:
        # Add masking for first s time steps where z is not generated
        mask = np.ones_like(z)
        mask[:s, :] = 0
        loss_z_masked = loss_z * mask
        res_z_masked = res_z * mask
        total_size = loss_x.size + mask.sum()
        total_loss = (loss_x.sum() + loss_z_masked.sum()) / total_size

        # Gradient vector ordered by _PARAM_KEYS: (alpha, beta, xi, eta, zeta, psi)
        grad = (
            np.array(
                [
                    res_x.sum(),  # alpha
                    (res_x * z).sum() + (res_z_masked * x).sum(),  # beta
                    (res_x * m).sum(),  # xi
                    (res_x * (prev_x + future_x)).sum(),  # eta
                    (
                        (res_x * future_z_masked).sum() + (res_z_masked * prev_x).sum()
                    ),  # zeta
                    (res_z_masked * (prev_z + future_z)).sum(),  # psi
                ]
            )
            / total_size
        )
    else:
        # Average over odd time steps only.
        # Zero-indexing means 0::2 gives us the even-indexed time steps, which correspond to odd time steps
        total_loss = loss_x[0::2, :].mean()
        total_size = res_x[0::2, :].size
        grad = (
            np.array(
                [
                    res_x[0::2, :].sum(),  # alpha
                    (res_x[0::2, :] * z[0::2, :]).sum(),  # beta
                    (res_x[0::2, :] * m[0::2, :]).sum(),  # xi
                    (
                        res_x[0::2, :] * (prev_x[0::2, :] + future_x[0::2, :])
                    ).sum(),  # eta
                    (res_x[0::2, :] * future_z_masked[0::2, :]).sum(),  # zeta
                    0,  # psi is not identifiable when conditioning on all z's
                ]
            )
            / total_size
        )
    return total_loss, grad


def mple_gradient_descent(
    x,
    z,
    x_0,
    gamma_matrix,
    s,
    loss_fn,
    loss_fn_kwargs=None,
    steps=2000,
    seed=0,
    verbose_every=100,
    tol=1e-9,
    logger=None,
):
    """Fit Ising parameters by MPLE using L-BFGS-B (scipy).

    Args:
        x (np.ndarray): shape (T, N), outcomes in {-1, +1}.
        z (np.ndarray): shape (T, N), interventions in {-1, +1}.
        x_0 (np.ndarray): shape (N,), initial state at t=0.
        gamma_matrix (np.ndarray): normalised network adjacency matrix.
        loss_fn (callable): objective returning (loss, gradient).
        loss_fn_kwargs (dict | None): optional keyword args passed to loss_fn.
        steps (int): maximum number of L-BFGS-B iterations.
        seed (int): RNG seed for initialisation.
        verbose_every (int): log objective every this many function evaluations.
        tol (float): convergence tolerance passed to scipy.
        logger (logging.Logger): optional logger for logging progress. If None, logs to console.

    Returns:
        tuple[dict, list[float]]: (params, loss_history)
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (T, N).")
    T, N = x.shape
    assert z.shape == (T, N), "z must have the same shape as x."

    rng = np.random.default_rng(seed)
    # Small Gaussian init: keeps tanh in the linear regime at the start.
    theta_init = rng.normal(0, 0.1, size=len(_PARAM_KEYS))
    loss_fn_kwargs = loss_fn_kwargs or {}

    history = []
    eval_count = [0]

    def objective(theta):
        loss, grad = loss_fn(
            x,
            z,
            theta,
            x_0,
            s,
            gamma_matrix,
            **loss_fn_kwargs,
        )
        history.append(loss)
        if verbose_every and eval_count[0] % verbose_every == 0:
            params_str = "  " + ",  ".join(
                f"{k}: {v:+.4f}" for k, v in zip(_PARAM_KEYS, theta)
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

    return result.x, history


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
    # Params
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    gamma_matrix = np.load(f"{args.data_folder}/gamma_matrix.npy")
    x_0 = np.load(f"{args.data_folder}/x_0.npy")
    logger.info("Loaded gamma_matrix with shape=%s", gamma_matrix.shape)

    # data
    data = np.load(f"{args.data_folder}/synthetic_data.npz")
    x = data["x"]
    z = data["z"]

    params_true = np.array(
        [
            config.estimation_params.alpha,
            config.estimation_params.beta,
            config.estimation_params.xi,
            config.estimation_params.eta,
            config.estimation_params.zeta,
            config.estimation_params.psi,
        ]
    )

    if args.use_conditional_npll:
        logger.info("Running conditional-model MPLE on x with shape=%s", x.shape)
        params_hat_conditional, loss_history_conditional = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            gamma_matrix=gamma_matrix,
            loss_fn=conditional_model_pseudo_nll,
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )

        logger.info("Done fitting.")
        logger.info(
            "Final Loss (Conditional Model): %.6f", loss_history_conditional[-1]
        )
        logger.info("Estimated vs True Parameters (Conditional Model):")
        for key, est, true in zip(_PARAM_KEYS, params_hat_conditional, params_true):
            logger.info("  %s: %.4f (True: %.4f)", key, est, true)
            logger.info("  %s SQE: %.6f", key, (est - true) ** 2)
        logger.info("=================================================")
        logger.info("Final Parameters:")
        for key, est, true in zip(_PARAM_KEYS, params_hat_conditional, params_true):
            logger.info("  %s: %.4f (True: %.4f)", key, est, true)
            logger.info("  %s SQE: %.6f", key, (est - true) ** 2)
    else:
        logger.info("Running Stage 1 MPLE on x with shape=%s", x.shape)
        params_hat, loss_history = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            gamma_matrix=gamma_matrix,
            loss_fn=pseudo_nll,
            loss_fn_kwargs={"conditioning": False},
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )
        logger.info("Running Stage 2 MPLE (conditioning on all z's and even x's)...")
        params_hat_cond, loss_history_cond = mple_gradient_descent(
            x,
            z,
            x_0=x_0,
            s=config.global_params.s,
            gamma_matrix=gamma_matrix,
            loss_fn=pseudo_nll,
            loss_fn_kwargs={"conditioning": True},
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            logger=logger,
        )

        logger.info("Done fitting.")
        logger.info("Final Loss (Unconditioned): %.6f", loss_history[-1])
        logger.info("Estimated vs True Parameters (Unconditioned):")
        for key, est, true in zip(_PARAM_KEYS, params_hat, params_true):
            logger.info("  %s: %.4f (True: %.4f)", key, est, true)
            logger.info("  %s SQE: %.6f", key, (est - true) ** 2)
        logger.info("=================================================")
        logger.info("Final Loss (Conditioned): %.6f", loss_history_cond[-1])
        logger.info("Estimated vs True Parameters (Conditioned):")
        for key, est, true in zip(_PARAM_KEYS, params_hat_cond, params_true):
            logger.info("  %s: %.4f (True: %.4f)", key, est, true)
            logger.info("  %s SQE: %.6f", key, (est - true) ** 2)
        logger.info("=================================================")
        logger.info("Final Parameters:")
        for key in _PARAM_KEYS:
            if key == "alpha":
                est = params_hat_cond[_PARAM_KEYS.index(key)]
            else:
                est = params_hat[_PARAM_KEYS.index(key)]
            true = params_true[_PARAM_KEYS.index(key)]
            logger.info("  %s: %.4f (True: %.4f)", key, est, true)
            logger.info("  %s SQE: %.6f", key, (est - true) ** 2)

    logger.info("Log saved to %s", log_file)
