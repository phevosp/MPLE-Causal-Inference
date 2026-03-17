from omegaconf import OmegaConf
import argparse
import networkx as nx
import numpy as np
from datetime import datetime
import os


def load_config(config_name, config_overrides=None):
    """Load a config file and optionally apply dotlist overrides."""
    config = OmegaConf.load(f"data/configs/{config_name}")
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(config_overrides))
    return config


def spin_sample_from_field(h, rng):
    """
    Sample spins in {-1, +1} from local field h.
    P(+1) = sigmoid(2h).
    Works for scalar or array h.
    """
    p = 1.0 / (1.0 + np.exp(-2.0 * h))
    return 2.0 * (rng.random(np.shape(p)) < p).astype(float) - 1.0


def read_and_realize_config(config_name, config_overrides=None):
    """Function to read and realize (by generating Gamma, x_0, and rng) config from relative path 'configs/config_name'

    Args:
        config_name (str): The name of the config file to read
        config_overrides (list[str] | None): optional OmegaConf dotlist overrides

    Returns:
        (OmegaConf, np.array, np.array, np.random.Generator):
            the realized config,
            the generated Gamma network stored as a numpy array,
            the generated x_0 vector,
            and the random number generator
    """
    config = load_config(config_name, config_overrides)

    # Step 1: Set seed, if specified
    seed = config.generation_params.seed
    rng = np.random.default_rng(seed)

    # Step 2: Generate Gamma
    # TODO: add lattice graph
    # TODO: add weights to edges & figure out specification for config
    print("Generating graph...")
    if config.global_params.gamma_matrix_generator == "erdos_renyi":
        gamma_graph = nx.erdos_renyi_graph(
            config.global_params.N,
            config.global_params.gamma_matrix_params.p,
            seed=config.generation_params.seed,
        )
    elif config.global_params.gamma_matrix_generator == "complete":
        gamma_graph = nx.complete_graph(config.global_params.N)
    elif config.global_params.gamma_matrix_generator == "empty":
        gamma_graph = nx.empty_graph(config.global_params.N)
    else:
        raise ValueError(
            f"Invalid gamma matrix generator: {config.global_params.gamma_matrix_generator}"
        )
    # Convert to adjacency matrix and normalize
    print("Converting to adjacency matrix and normalizing...")
    node_order = list(gamma_graph.nodes())
    gamma_matrix = nx.to_numpy_array(gamma_graph, nodelist=node_order)
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2  # Ensure symmetry
    np.fill_diagonal(gamma_matrix, 0)  # Ensure no self-loops
    gamma_matrix = gamma_matrix / np.linalg.norm(gamma_matrix, ord=np.inf)  # Normalize

    # Step 3: Generate x_0
    if config.global_params.x_0_generator == "bernoulli":
        p = config.global_params.x_0_params.p
        x_0 = (rng.random(config.global_params.N) < p).astype(float) * 2 - 1
    elif config.global_params.x_0_generator == "fixed":
        fixed_val = config.global_params.x_0_params.fixed_val
        x_0 = np.full(config.global_params.N, fixed_val)
    else:
        raise ValueError(f"Invalid x_0_generator: {config.global_params.x_0_generator}")

    return config, gamma_matrix, x_0, rng


def sample_z_t(x_prev, z_prev, config, rng):
    """Sample z^{(t)} given x^{(t-1)} and z^{(t-1)}.
    Since z_t are independent across i, we can sample them in parallel according to a simple logistic

    Args:
        x_prev (np.array): outcomes from time step t-1
        z_prev (np.array): outcomes from time step t-1
        config (OmegaConf): model configuration containing zeta and psi

    Returns:
        np.array: sampled z^{(t)}
    """
    h_z = config.estimation_params.zeta * x_prev + config.estimation_params.psi * z_prev
    return spin_sample_from_field(h_z, rng)


def sample_x_t(x_prev, z_curr, config, gamma_matrix, rng):
    """Sample x^{(t)} given x^{(t-1)} and z^{(t)} using Gibbs sampling.
    For each i, sample x_i^(t) given x_{-i}^{(t)}, z^{(t)}, and x^{(t-1)} according to a logistic function.

    Args:
        x_prev (np.array): outcomes from time step t-1
        z_curr (np.array): outcomes from time step t
        config (OmegaConf): model configuration containing alpha, beta, eta, and xi
        gamma_matrix (np.array): the network structure as an adjacency matrix
        rng (np.random.Generator): random number generator

    Returns:
        np.array: sampled x^{(t)}
    """
    x_t = x_prev.copy()
    # Store gamma_matrix @ x_t and update incrementally for efficiency
    gamma_x_t = gamma_matrix @ x_t
    for _ in range(config.generation_params.gibbs_sweeps):
        # Random order mixes faster and ensures symmetry in the influence of nodes
        node_order = rng.permutation(config.global_params.N)
        for i in node_order:
            old_x_i = x_t[i]
            h_x = (
                config.estimation_params.alpha
                + config.estimation_params.beta * z_curr[i]
                + config.estimation_params.eta * x_prev[i]
                + config.estimation_params.xi * gamma_x_t[i]
            )
            x_t[i] = spin_sample_from_field(h_x, rng)
            # Update gamma_x_t for next iteration
            gamma_x_t += (x_t[i] - old_x_i) * gamma_matrix[:, i]

    return x_t


def generate_conditional_model(config, gamma_matrix, x_0, rng):
    # Initialize
    x = np.zeros((config.global_params.T, config.global_params.N))
    z = np.zeros((config.global_params.T, config.global_params.N))
    # Sample initial z and x
    # We assume z^{(0)} is all zeros
    z[0, :] = (
        sample_z_t(x_0, np.zeros_like(x_0), config, rng)
        if config.global_params.s == 0
        else -np.ones_like(x_0)
    )
    x[0, :] = sample_x_t(x_0, z[0, :], config, gamma_matrix, rng)

    # Sample subsequent time steps
    for t in range(1, config.global_params.T):
        print(f"Sampling time step {t}...")
        z[t, :] = (
            sample_z_t(x[t - 1, :], z[t - 1, :], config, rng)
            if t >= config.global_params.s
            else -np.ones_like(x_0)
        )
        x[t, :] = sample_x_t(x[t - 1, :], z[t, :], config, gamma_matrix, rng)

    return x, z


def generate_ising_model(config, gamma_matrix, x_0, rng):
    # Initialize
    x = rng.choice([-1, 1], size=(config.global_params.T, config.global_params.N))
    z = rng.choice([-1, 1], size=(config.global_params.T, config.global_params.N))
    z[: config.global_params.s, :] = (
        -1
    )  # Set first s time steps of z to -1 (no intervention)

    # Precompute gamma_x for efficiency
    gamma_x = x @ gamma_matrix.T
    for g in range(config.generation_params.gibbs_sweeps):
        print(f"Performing Gibbs sweep {g}...")
        # Randomize order for better mixing
        t_order = rng.permutation(config.global_params.T)
        for t in t_order:
            node_order = rng.permutation(config.global_params.N)
            for i in node_order:
                old_x_t_i = x[t, i]
                # fmt: off
                h_x = (
                    config.estimation_params.alpha
                    + config.estimation_params.eta *(x[t - 1, i] if t > 0 else x_0[i])
                    + config.estimation_params.beta * z[t, i]
                    + config.estimation_params.xi * gamma_x[t, i]
                    + config.estimation_params.zeta * (z[t + 1, i] if (t + 1 >= config.global_params.s and t + 1 < config.global_params.T) else 0)
                    + config.estimation_params.eta * (x[t + 1, i] if t < config.global_params.T - 1 else 0)
                )
                x[t, i] = spin_sample_from_field(h_x, rng)
                gamma_x[t, :] += (x[t, i] - old_x_t_i) * gamma_matrix[:, i]
            # Update all z's for time step t in parallel since they are conditionally independent
            if t >= config.global_params.s:
                h_z = (
                    config.estimation_params.psi * (z[t - 1] if t > 0 else 0)
                    + config.estimation_params.zeta * (x[t - 1] if t > 0 else x_0)
                    + config.estimation_params.beta * x[t]
                    + config.estimation_params.psi
                    * (z[t + 1] if t < config.global_params.T - 1 else 0)
                )
                z[t] = spin_sample_from_field(h_z, rng)
                # fmt: on

    return x, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for MPLE experiments."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="base_config.yaml",
        help="The name of the config file to use for data generation (located in data/configs/)",
    )
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Manual config override in OmegaConf dotlist format. Repeat to set multiple values.",
    )
    args = parser.parse_args()

    print("Starting synthetic data generation...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = f"experiments/synthetic_data_{timestamp}"

    print("Reading and realizing config...")
    config, gamma_matrix, x_0, rng = read_and_realize_config(
        args.config_name,
        args.config_override,
    )

    if config.generation_params.process == "conditional":
        print("Generating data with conditional process...")
        x, z = generate_conditional_model(config, gamma_matrix, x_0, rng)
    elif config.generation_params.process == "Ising":
        print("Generating data with Ising process...")
        x, z = generate_ising_model(config, gamma_matrix, x_0, rng)
    else:
        raise ValueError(
            f"Invalid generation process: {config.generation_params.process}"
        )

    os.makedirs(data_folder)
    print("Saving data, config, and network...")
    OmegaConf.save(config, f"{data_folder}/realized_config.yaml")
    np.savez(f"{data_folder}/synthetic_data.npz", x=x, z=z)
    np.save(f"{data_folder}/gamma_matrix.npy", gamma_matrix)
    np.save(f"{data_folder}/x_0.npy", x_0)

    print("Done!")
    print("Frob. Norm of Gamma Matrix:", np.linalg.norm(gamma_matrix, ord="fro"))
