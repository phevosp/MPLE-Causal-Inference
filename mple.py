import argparse
import logging
from pathlib import Path
import numpy as np
import networkx as nx
from omegaconf import OmegaConf
from scipy.optimize import minimize


def setup_logger(log_file):
    """Configure a logger that writes to both console and file."""
    logger = logging.getLogger("mple")
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


def pseudo_nll(x, z, theta, x_0, s, gamma_matrix):
    """Compute Ising negative log-pseudolikelihood averaged over samples.

    Args:
        theta (np.ndarray): parameter vector ordered by _PARAM_KEYS.
    """
    alpha, beta, xi, eta, zeta, psi = theta
    prev_x = np.vstack([x_0, x[:-1, :]])
    prev_z = np.vstack([np.zeros_like(x_0), z[:-1, :]])
    future_x = np.vstack([x[1:, :], np.zeros_like(x_0)])
    future_z = np.vstack([z[1:, :], np.zeros_like(x_0)])
    m = x @ gamma_matrix.T

    h_z = psi * prev_z + zeta * prev_x + beta * x + psi * future_z
    h_x = alpha + eta * prev_x + beta * z + xi * m + zeta * future_z + eta * future_x

    loss_x = np.logaddexp(h_x, -h_x) - x * h_x
    loss_z = np.logaddexp(h_z, -h_z) - z * h_z

    res_x = np.tanh(h_x) - x
    res_z = np.tanh(h_z) - z

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
                (res_x * future_z).sum() + (res_z_masked * prev_x).sum(),  # zeta
                (res_z_masked * (prev_z + future_z)).sum(),  # psi
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
        steps (int): maximum number of L-BFGS-B iterations.
        seed (int): RNG seed for initialisation.
        verbose_every (int): log objective every this many function evaluations.
        tol (float): convergence tolerance passed to scipy.

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

    history = []
    eval_count = [0]

    def objective(theta):
        loss, grad = pseudo_nll(x, z, theta, x_0, s, gamma_matrix)
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
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
    # Params
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    gamma_matrix = np.load(f"{args.data_folder}/gamma_matrix.npy")
    x_0 = np.load(f"{args.data_folder}/x_0.npy")
    logger.info("Loaded gamma_matrix with shape=%s", gamma_matrix.shape)

    # data
    data = np.load(f"{args.data_folder}/synthetic_data.npz")
    x = data["x"]
    z = data["z"]

    logger.info("Running MPLE on x with shape=%s", x.shape)
    params_hat, loss_history = mple_gradient_descent(
        x,
        z,
        x_0=x_0,
        s=config.global_params.s,
        gamma_matrix=gamma_matrix,
        steps=args.steps,
        tol=args.tol,
        seed=args.seed,
        logger=logger,
    )
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

    logger.info("Done fitting.")
    logger.info("Final loss: %.6f", loss_history[-1])
    logger.info("Estimated vs True parameters:")
    for key, est, true in zip(_PARAM_KEYS, params_hat, params_true):
        logger.info("  %s: %.4f (True: %.4f)", key, est, true)
        logger.info("  %s SE: %.6f", key, (est - true) ** 2)

    logger.info("Log saved to %s", log_file)
