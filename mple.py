import argparse
import logging
from pathlib import Path
import numpy as np
import networkx as nx
from omegaconf import OmegaConf


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


def pseudo_nll(x, z, params, x_0, s, gamma_matrix):
    """Compute Ising negative log-pseudolikelihood averaged over samples."""
    # Params
    alpha, beta, xi, eta, zeta, psi = (
        params["alpha"],
        params["beta"],
        params["xi"],
        params["eta"],
        params["zeta"],
        params["psi"],
    )
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

    # Compute gradients for each param
    # Gradients from x-sites
    g_alpha = res_x.sum()
    g_beta = (res_x * z).sum() + (res_z_masked * x).sum()
    g_xi = (res_x * m).sum()
    g_eta = (res_x * (prev_x + future_x)).sum()
    g_zeta = (res_x * future_z).sum() + (res_z_masked * prev_x).sum()
    g_psi = (res_z_masked * (prev_z + future_z)).sum()

    grad = {
        "alpha": g_alpha / total_size,
        "beta": g_beta / total_size,
        "xi": g_xi / total_size,
        "eta": g_eta / total_size,
        "zeta": g_zeta / total_size,
        "psi": g_psi / total_size,
    }
    return total_loss, grad


def mple_gradient_descent(
    x,
    z,
    x_0,
    gamma_matrix,
    s,
    learning_rate=0.05,
    steps=2000,
    seed=0,
    verbose_every=100,
    logger=None,
):
    """Fit Ising parameters (h, J) by MPLE using gradient descent.

    Args:
        x (np.ndarray): shape (n_samples, n_nodes), outcomes in {-1, +1}.
        z (np.ndarray): shape (n_samples, n_nodes), interventions in {-1, +1}.
        x_0 (np.ndarray): shape (n_nodes,), initial state at t=0.
        gamma_matrix (np.ndarray): the underlying network adjacency matrix on the outcomes.
        learning_rate (float): gradient descent step size.
        steps (int): number of optimization steps.
        seed (int): RNG seed for initialization.
        verbose_every (int): print objective every this many steps.

    Returns:
        tuple[np.ndarray, np.ndarray, list[float]]: (h, J, loss_history)
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_nodes).")

    T, N = x.shape
    assert z.shape == (T, N), "z must have the same shape as x."
    rng = np.random.default_rng(seed)

    params_hat = {
        "alpha": rng.uniform(-1, 1),
        "beta": rng.uniform(-1, 1),
        "xi": rng.uniform(-1, 1),
        "eta": rng.uniform(-1, 1),
        "zeta": rng.uniform(-1, 1),
        "psi": rng.uniform(-1, 1),
    }

    history = []
    for step in range(steps):
        nll, gradient = pseudo_nll(x, z, params_hat, x_0, s, gamma_matrix)
        history.append(nll)

        if verbose_every and step % verbose_every == 0:
            if logger is not None:
                logger.info("Step %s/%s, Loss: %.6f", step, steps, nll)
            else:
                print(f"Step {step}/{steps}, Loss: {nll:.6f}")

        # Update params_hat with gradient descent
        for param in params_hat:
            params_hat[param] -= learning_rate * gradient[param]

    return params_hat, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Ising parameters with MPLE gradient descent."
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        type=str,
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
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
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=args.seed,
        logger=logger,
    )
    params_true = {
        "alpha": config.estimation_params.alpha,
        "beta": config.estimation_params.beta,
        "xi": config.estimation_params.xi,
        "eta": config.estimation_params.eta,
        "zeta": config.estimation_params.zeta,
        "psi": config.estimation_params.psi,
    }

    logger.info("Done fitting.")
    logger.info("Final loss: %.6f", loss_history[-1])
    logger.info("Estimated vs True parameters:")
    for param, value in params_hat.items():
        logger.info("  %s: %.4f (True: %.4f)", param, value, params_true[param])
        logger.info("  %s MSE: %.6f", param, np.mean((value - params_true[param]) ** 2))

    logger.info("Log saved to %s", log_file)
