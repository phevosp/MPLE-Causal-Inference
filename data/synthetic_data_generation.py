from omegaconf import OmegaConf
import argparse
import networkx as nx
import numpy as np
from datetime import datetime
import os


def read_and_realize_config(config_name):
    """Function to read and realize (by generating Gamma & x_0) config from relative path 'configs/config_name'

    Args:
        config_name (str): The name of the config file to read

    Returns:
        (OmegaConf, networkx.Graph, np.array): The realized config, the generated Gamma matrix as a networkx graph, and the generated x_0 vector
    """
    config = OmegaConf.load(f"data/configs/{config_name}")

    # Set seed, if specified
    config.generation_params.seed = config.generation_params.get("seed", None)
    if config.generation_params.seed is not None:
        np.random.seed(config.generation_params.seed)

    # Generate Gamma
    if config.global_params.gamma_matrix_generator == "erdos_renyi":
        gamma_matrix = nx.erdos_renyi_graph(
            config.global_params.N,
            config.global_params.gamma_matrix_params.p,
            seed=config.generation_params.seed,
        )
    elif config.global_params.gamma_matrix_generator == "lattice":
        config.global_params.N_side = int(config.global_params.N**0.5)
        gamma_matrix = nx.grid_2d_graph(
            config.global_params.N_side, config.global_params.N_side
        )
    elif config.global_params.gamma_matrix_generator == "complete":
        gamma_matrix = nx.complete_graph(config.global_params.N)
    elif config.global_params.gamma_matrix_generator == "empty":
        gamma_matrix = nx.empty_graph(config.global_params.N)
    else:
        raise ValueError(
            f"Invalid gamma matrix generator: {config.global_params.gamma_matrix_generator}"
        )

    # Generate x_0
    if config.global_params.x_0_generator == "bernoulli":
        p = config.global_params.x_0_params.p
        x_0 = (np.random.rand(config.global_params.N) < p).astype(float) * 2 - 1
    elif config.global_params.x_0_generator == "fixed":
        fixed_val = config.global_params.x_0_params.fixed_val
        x_0 = np.full(config.global_params.N, fixed_val)
    else:
        raise ValueError(f"Invalid x_0_generator: {config.global_params.x_0_generator}")

    return config, gamma_matrix, x_0


def sample_z_t(x, z, config):
    """Sample z^{(t)} given x^{(t-1)} and z^{(t-1)}.
    Since z_t are independent across i, we can sample them in parallel according to a simple logistic

    Args:
        x (np.array): outcomes from time step t-1
        z (np.array): outcomes from time step t-1
        config (OmegaConf): model configuration containing zeta and psi

    Returns:
        np.array: sampled z^{(t)}
    """
    h_z = config.estimation_params.zeta * x + config.estimation_params.psi * z
    p_z = 1 / (1 + np.exp(-2 * h_z))
    z = (np.random.rand(config.global_params.N) < p_z).astype(float) * 2 - 1
    return z


def sample_x_t(x, z, config, gamma_matrix):
    """Sample x^{(t)} given x^{(t-1)} and z^{(t)} using Gibbs sampling.
    For each i, sample x_i^(t) given x_{-i}^(t), z^(t), and x^(t-1) according to a logistic function.

    Args:
        x (np.array): outcomes from time step t-1
        z (np.array): outcomes from time step t
        config (OmegaConf): model configuration containing alpha, beta, eta, xi, and gamma_matrix
        gamma_matrix (networkx.Graph): the network structure as a networkx graph

    Returns:
        np.array: sampled x^{(t)}
    """
    x_t = x.copy()
    for _ in range(config.generation_params.gibbs_sweeps):
        for i in range(config.global_params.N):
            neighbors = list(gamma_matrix.neighbors(i))
            h_x = (
                config.estimation_params.alpha
                + config.estimation_params.beta * z[i]
                + config.estimation_params.eta * x[i]
                + config.estimation_params.xi * np.sum(x[neighbors])
            )
            p_x = 1 / (1 + np.exp(-2 * h_x))
            x_t[i] = (np.random.rand() < p_x).astype(float) * 2 - 1
    return x_t


def main(args):
    print("Starting synthetic data generation...")

    print("Preparing data folder...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = f"data/synthetic_data_{timestamp}"

    print("Reading and realizing config...")
    config, gamma_matrix, x_0 = read_and_realize_config(args.config_name)

    # Initialize
    x = np.zeros((config.global_params.T, config.global_params.N))
    z = np.zeros((config.global_params.T, config.global_params.N))
    z[0, :] = (
        sample_z_t(x_0, np.zeros_like(x_0), config)
        if config.global_params.s == 0
        else np.zeros_like(x_0)
    )
    x[0, :] = sample_x_t(x_0, z[0, :], config, gamma_matrix)

    for t in range(1, config.global_params.T):
        print(f"Sampling time step {t}...")
        z[t, :] = (
            sample_z_t(x[t - 1, :], z[t - 1, :], config)
            if t >= config.global_params.s
            else np.zeros_like(x_0)
        )
        x[t, :] = sample_x_t(x[t - 1, :], z[t, :], config, gamma_matrix)

    os.makedirs(data_folder)
    print("Saving data...")
    np.savez(f"{data_folder}/synthetic_data.npz", x=x, z=z)
    print("Saving config...")
    OmegaConf.save(config, f"{data_folder}/realized_config.yaml")
    print("Saving network")
    nx.write_graphml(gamma_matrix, f"{data_folder}/gamma_matrix.graphml")
    print("Saving x0")
    np.save(f"{data_folder}/x_0.npy", x_0)

    print("Done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Generate synthetic data for MPLE experiments."
    )
    argparser.add_argument(
        "--config_name",
        type=str,
        default="synthetic_data_config.yaml",
        help="The name of the config file to use for data generation (located in data/configs/)",
    )
    args = argparser.parse_args()
    main(args)
