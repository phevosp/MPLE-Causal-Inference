from datetime import datetime
from pathlib import Path
import argparse
import os
import sys
import re

import networkx as nx
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import (
    BasisExpansion,
    compose_field,
    compose_field_matrix,
    compose_interaction_matrix,
    get_field_coeffs,
    get_interaction_coeffs,
    get_temporal_field,
    load_or_build_basis,
)


def slugify(text: str) -> str:
    """Convert free-form experiment labels into filesystem-friendly slugs."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "experiment"


def load_config(config_name, config_overrides=None):
    """Load a config file and optionally apply dotlist overrides."""
    config = OmegaConf.load(f"data/configs/{config_name}")
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(config_overrides))
    return config


def parse_metadata_entries(entries: list[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE metadata arguments into a dictionary."""
    metadata = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Metadata entry '{entry}' must be in KEY=VALUE format.")
        key, value = entry.split("=", 1)
        metadata[key] = value
    return metadata


def spin_sample_from_field(h, rng):
    """Sample spins in {-1, +1} from a local field using the logistic conditional."""
    p = 1.0 / (1.0 + np.exp(-2.0 * h))
    return 2.0 * (rng.random(np.shape(p)) < p).astype(float) - 1.0


def scale_gamma_matrix(config, gamma_matrix: np.ndarray) -> np.ndarray:
    """Normalize the known graph matrix by infinity norm only, without Frobenius re-scaling."""
    gamma_matrix = np.asarray(gamma_matrix, dtype=float)
    norm = np.linalg.norm(gamma_matrix, ord=np.inf)
    return gamma_matrix / norm if norm > 1e-12 else gamma_matrix


def read_and_realize_config(config_name, config_overrides=None):
    """Load the config and realize the graph, initial state, and random generator."""
    config = load_config(config_name, config_overrides)

    seed = config.generation_params.seed
    rng = np.random.default_rng(seed)

    print("Generating graph...")
    if config.global_params.gamma_matrix_generator == "erdos_renyi":
        gamma_graph = nx.erdos_renyi_graph(
            config.global_params.N,
            config.global_params.gamma_matrix_params.p,
            seed=config.generation_params.seed,
        )
    elif config.global_params.gamma_matrix_generator == "complete":
        gamma_graph = nx.complete_graph(config.global_params.N)
    elif config.global_params.gamma_matrix_generator == "cycle":
        gamma_graph = nx.cycle_graph(config.global_params.N)
    elif config.global_params.gamma_matrix_generator == "empty":
        gamma_graph = nx.empty_graph(config.global_params.N)
    else:
        raise ValueError(
            f"Invalid gamma matrix generator: {config.global_params.gamma_matrix_generator}"
        )

    print("Converting to adjacency matrix and normalizing...")
    node_order = list(gamma_graph.nodes())
    gamma_matrix = nx.to_numpy_array(gamma_graph, nodelist=node_order)
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2
    np.fill_diagonal(gamma_matrix, 0)
    gamma_matrix = scale_gamma_matrix(config, gamma_matrix)

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
    """Sample the intervention vector at time t given the previous state and intervention."""
    h_z = config.estimation_params.zeta * x_prev + config.estimation_params.psi * z_prev
    return spin_sample_from_field(h_z, rng)


def sample_x_t(x_prev, z_curr, config, field_vector_t, interaction_matrix, rng):
    """Sample the outcome vector at time t with Gibbs sweeps under the conditional model."""
    x_t = x_prev.copy()
    interaction_x_t = interaction_matrix @ x_t
    for _ in range(config.generation_params.gibbs_sweeps):
        node_order = rng.permutation(config.global_params.N)
        for i in node_order:
            old_x_i = x_t[i]
            h_x = (
                field_vector_t[i]
                + config.estimation_params.beta * z_curr[i]
                + config.estimation_params.eta * x_prev[i]
                + interaction_x_t[i]
            )
            x_t[i] = spin_sample_from_field(h_x, rng)
            interaction_x_t += (x_t[i] - old_x_i) * interaction_matrix[:, i]

    return x_t


def generate_data(config, field_matrix, interaction_matrix, x_0, rng):
    """Generate synthetic outcome and intervention trajectories from the conditional model."""
    x = np.zeros((config.global_params.T, config.global_params.N))
    z = np.zeros((config.global_params.T, config.global_params.N))

    z[0, :] = (
        sample_z_t(x_0, np.zeros_like(x_0), config, rng)
        if config.global_params.s == 0
        else -np.ones_like(x_0)
    )
    x[0, :] = sample_x_t(x_0, z[0, :], config, field_matrix[0, :], interaction_matrix, rng)

    for t in range(1, config.global_params.T):
        print(f"Sampling time step {t}...")
        z[t, :] = (
            sample_z_t(x[t - 1, :], z[t - 1, :], config, rng)
            if t >= config.global_params.s
            else -np.ones_like(x_0)
        )
        x[t, :] = sample_x_t(
            x[t - 1, :],
            z[t, :],
            config,
            field_matrix[t, :],
            interaction_matrix,
            rng,
        )

    return x, z


def save_artifacts(
    data_folder: str,
    config,
    metadata: dict[str, str],
    basis: BasisExpansion,
    gamma_matrix: np.ndarray,
    field_vector: np.ndarray,
    tau: np.ndarray,
    field_matrix: np.ndarray,
    interaction_matrix: np.ndarray,
    x_0: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> None:
    """Persist the realized config, metadata, basis objects, and sampled data."""
    os.makedirs(data_folder)
    print("Saving data, config, and network...")
    OmegaConf.save(config, f"{data_folder}/realized_config.yaml")
    OmegaConf.save(OmegaConf.create(metadata), f"{data_folder}/experiment_metadata.yaml")
    np.savez(f"{data_folder}/panel_data.npz", x=x, z=z)
    np.save(f"{data_folder}/gamma_matrix.npy", gamma_matrix)
    np.save(f"{data_folder}/x_0.npy", x_0)
    np.save(f"{data_folder}/field_basis.npy", basis.field_basis)
    np.save(f"{data_folder}/interaction_basis.npy", basis.interaction_basis)
    np.save(f"{data_folder}/field_vector.npy", field_vector)
    np.save(f"{data_folder}/tau.npy", tau)
    np.save(f"{data_folder}/field_matrix.npy", field_matrix)
    np.save(f"{data_folder}/interaction_matrix.npy", interaction_matrix)
    np.save(f"{data_folder}/shared_features.npy", basis.shared_features)
    np.save(
        f"{data_folder}/field_basis_names.npy",
        np.asarray(basis.field_names, dtype="<U64"),
    )
    np.save(
        f"{data_folder}/interaction_basis_names.npy",
        np.asarray(basis.interaction_names, dtype="<U64"),
    )
    np.save(
        f"{data_folder}/shared_feature_names.npy",
        np.asarray(basis.shared_feature_names, dtype="<U64"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for conditional-model MPLE experiments."
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
    parser.add_argument(
        "--descriptor",
        type=str,
        default=None,
        help="Optional human-readable label used in the output folder name.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional file where the generated experiment folder path will be appended.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional experiment metadata fields to save alongside the generated data.",
    )
    args = parser.parse_args()

    print("Starting synthetic data generation...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = slugify(args.descriptor) if args.descriptor else "synthetic_data"
    data_folder = f"experiments/{descriptor}_{timestamp}"
    extra_metadata = parse_metadata_entries(args.metadata)

    print("Reading and realizing config...")
    config, gamma_matrix, x_0, rng = read_and_realize_config(
        args.config_name,
        args.config_override,
    )
    basis = load_or_build_basis(config, gamma_matrix)
    field_coeffs = get_field_coeffs(config)
    tau = get_temporal_field(config, int(config.global_params.T))
    interaction_coeffs = get_interaction_coeffs(config)
    field_vector = compose_field(field_coeffs, basis.field_basis)
    field_matrix = compose_field_matrix(field_coeffs, tau, basis.field_basis)
    interaction_matrix = compose_interaction_matrix(
        interaction_coeffs,
        basis.interaction_basis,
    )

    print("Generating data with the conditional process...")
    x, z = generate_data(
        config,
        field_matrix,
        interaction_matrix,
        x_0,
        rng,
    )

    metadata = {
        "descriptor": args.descriptor or descriptor,
        "slug": descriptor,
        "timestamp": timestamp,
        "config_name": args.config_name,
        "config_overrides": list(args.config_override),
        "gamma_inf_norm": float(np.linalg.norm(gamma_matrix, ord=np.inf)),
        "gamma_fro_norm": float(np.linalg.norm(gamma_matrix, ord="fro")),
        "field_basis_inf_norms": [
            float(np.linalg.norm(vector, ord=np.inf)) for vector in basis.field_basis
        ],
        "tau_l2_norm": float(np.linalg.norm(tau, ord=2)),
        "interaction_basis_inf_norms": [
            float(np.linalg.norm(matrix, ord=np.inf))
            for matrix in basis.interaction_basis
        ],
        **extra_metadata,
    }
    save_artifacts(
        data_folder=data_folder,
        config=config,
        metadata=metadata,
        basis=basis,
        gamma_matrix=gamma_matrix,
        field_vector=field_vector,
        tau=tau,
        field_matrix=field_matrix,
        interaction_matrix=interaction_matrix,
        x_0=x_0,
        x=x,
        z=z,
    )

    print("Done!")
    print("Infinity Norm of Gamma Matrix:", np.linalg.norm(gamma_matrix, ord=np.inf))
    print(
        "Infinity Norm of Interaction Matrix:",
        np.linalg.norm(interaction_matrix, ord=np.inf),
    )
    print(f"Experiment Folder: {data_folder}")

    if args.manifest_path is not None:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{Path(data_folder).resolve()}\n")
