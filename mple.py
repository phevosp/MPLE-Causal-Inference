import argparse
import numpy as np
import networkx as nx
from omegaconf import OmegaConf


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
        type=str,
        default="data/synthetic_data_20260313_135938",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading data...")
    # Params
    config = OmegaConf.load(f"{args.data_folder}/realized_config.yaml")
    gamma_matrix = np.load(f"{args.data_folder}/gamma_matrix.npy")
    x_0 = np.load(f"{args.data_folder}/x_0.npy")
    print(gamma_matrix)

    # data
    data = np.load(f"{args.data_folder}/synthetic_data.npz")
    x = data["x"]
    z = data["z"]

    print(f"Running MPLE on x with shape={x.shape}")
    params_hat, loss_history = mple_gradient_descent(
        x,
        z,
        x_0=x_0,
        s=config.global_params.s,
        gamma_matrix=gamma_matrix,
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=args.seed,
    )
    params_true = {
        "alpha": config.estimation_params.alpha,
        "beta": config.estimation_params.beta,
        "xi": config.estimation_params.xi,
        "eta": config.estimation_params.eta,
        "zeta": config.estimation_params.zeta,
        "psi": config.estimation_params.psi,
    }

    print("Done fitting.")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print("Estimated vs True parameters:")
    for param, value in params_hat.items():
        print(f"  {param}: {value:.4f} (True: {params_true[param]:.4f})")
        print(f"  {param} MSE: {np.mean((value - params_true[param])**2):.6f}")
