from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import (
    ModelArtifacts,
    build_synthetic_field,
    compose_interaction_matrix,
    get_xi,
    save_model_artifacts,
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "experiment"


def load_config(config_name: str, config_overrides: list[str] | None = None):
    config = OmegaConf.load(f"data/configs/{config_name}")
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(config_overrides))
    return config


def parse_metadata_entries(entries: list[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Metadata entry '{entry}' must be in KEY=VALUE format.")
        key, value = entry.split("=", 1)
        metadata[key] = value
    return metadata


def spin_sample_from_field(h, rng):
    p = 1.0 / (1.0 + np.exp(-2.0 * h))
    return 2.0 * (rng.random(np.shape(p)) < p).astype(float) - 1.0


def read_and_realize_config(
    config_name: str, config_overrides: list[str] | None = None
):
    config = load_config(config_name, config_overrides)
    rng = np.random.default_rng(int(config.generation_params.seed))

    generator = str(config.global_params.gamma_matrix_generator)
    if generator == "erdos_renyi":
        gamma_graph = nx.erdos_renyi_graph(
            int(config.global_params.N),
            float(config.global_params.gamma_matrix_params.p),
            seed=int(config.generation_params.seed),
        )
    elif generator == "complete":
        gamma_graph = nx.complete_graph(int(config.global_params.N))
    elif generator == "cycle":
        gamma_graph = nx.cycle_graph(int(config.global_params.N))
    elif generator == "empty":
        gamma_graph = nx.empty_graph(int(config.global_params.N))
    else:
        raise ValueError(f"Invalid gamma matrix generator: {generator}")

    gamma_matrix = nx.to_numpy_array(gamma_graph, nodelist=list(gamma_graph.nodes()))
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2.0
    np.fill_diagonal(gamma_matrix, 0.0)

    x0_mode = str(config.global_params.x_0_generator)
    if x0_mode == "bernoulli":
        p = float(config.global_params.x_0_params.p)
        x_0 = (rng.random(int(config.global_params.N)) < p).astype(float) * 2.0 - 1.0
    elif x0_mode == "fixed":
        x_0 = np.full(
            int(config.global_params.N),
            float(config.global_params.x_0_params.fixed_val),
        )
    else:
        raise ValueError(f"Invalid x_0_generator: {x0_mode}")

    return config, gamma_matrix, x_0, rng


def intervention_mode(config) -> str:
    return str(getattr(config.generation_params, "intervention_mode", "generated_z"))


def load_fixed_intervention_artifacts(
    config,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    source = getattr(config.generation_params, "fixed_z_source", None)
    if source is None:
        raise ValueError(
            "generation_params.fixed_z_source is required when intervention_mode='fixed_z'."
        )

    panel_path = Path(str(getattr(source, "panel_path", "")))
    z0_path = Path(str(getattr(source, "z0_path", "")))
    if not panel_path.exists() or not z0_path.exists():
        raise FileNotFoundError(
            "fixed_z_source.panel_path and fixed_z_source.z0_path must both exist."
        )

    with np.load(panel_path) as data:
        if "z" not in data:
            raise KeyError(
                f"Fixed-z panel artifact {panel_path} does not contain a 'z' array."
            )
        z = np.asarray(data["z"], dtype=float)
    z_0 = np.asarray(np.load(z0_path), dtype=float)

    expected_shape = (int(config.global_params.T), int(config.global_params.N))
    if z.shape != expected_shape:
        raise ValueError(
            f"Fixed-z artifact shape {z.shape} does not match configured (T, N)={expected_shape}."
        )
    if z_0.shape != (int(config.global_params.N),):
        raise ValueError(
            f"Fixed-z initial state shape {z_0.shape} does not match configured N={int(config.global_params.N)}."
        )

    metadata = {
        "fixed_z_panel_path": str(panel_path.resolve()),
        "fixed_z_z0_path": str(z0_path.resolve()),
    }
    for key in [
        "shared_panel_dir",
        "outcome_code",
        "intervention_code",
        "lag_code",
        "trim_scope",
    ]:
        value = getattr(source, key, None)
        if value:
            metadata[f"fixed_z_{key}"] = str(value)
    return z, z_0, metadata


def derive_pre_intervention_steps(z: np.ndarray) -> int:
    treated_rows = np.any(np.asarray(z) == 1, axis=1)
    return int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])


def sample_z_t(x_prev, z_prev, config, rng):
    h_z = config.estimation_params.zeta * x_prev + config.estimation_params.psi * z_prev
    return spin_sample_from_field(h_z, rng)


def sample_x_t(x_prev, z_curr, config, field_t, interaction_matrix, rng):
    x_t = x_prev.copy()
    interaction_x_t = interaction_matrix @ x_t
    for _ in range(int(config.generation_params.gibbs_sweeps)):
        for i in rng.permutation(int(config.global_params.N)):
            old_x_i = x_t[i]
            h_x = (
                field_t[i]
                + config.estimation_params.beta * z_curr[i]
                + config.estimation_params.eta * x_prev[i]
                + interaction_x_t[i]
            )
            x_t[i] = spin_sample_from_field(h_x, rng)
            interaction_x_t += (x_t[i] - old_x_i) * interaction_matrix[:, i]
    return x_t


def generate_data(
    config,
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    rng,
    fixed_z: np.ndarray | None = None,
):
    t_steps = int(config.global_params.T)
    n_nodes = int(config.global_params.N)
    interaction_matrix = compose_interaction_matrix(
        get_xi(config), artifacts.gamma_matrix
    )
    x = np.zeros((t_steps, n_nodes))
    z = np.zeros((t_steps, n_nodes))
    z_0 = np.zeros(n_nodes, dtype=float)

    mode = intervention_mode(config)
    if mode == "fixed_z":
        if fixed_z is None:
            raise ValueError(
                "fixed_z must be provided when intervention_mode='fixed_z'."
            )
        z[:, :] = np.asarray(fixed_z, dtype=float)

    if mode == "generated_z":
        z[0, :] = (
            sample_z_t(x_0, z_0, config, rng)
            if int(config.global_params.s) == 0
            else -np.ones_like(x_0)
        )
    x[0, :] = sample_x_t(
        x_0, z[0, :], config, artifacts.field_matrix[0, :], interaction_matrix, rng
    )

    for t in range(1, t_steps):
        print(f"Sampling time step {t}...")
        if mode == "generated_z":
            z[t, :] = (
                sample_z_t(x[t - 1, :], z[t - 1, :], config, rng)
                if t >= int(config.global_params.s)
                else -np.ones_like(x_0)
            )
        x[t, :] = sample_x_t(
            x[t - 1, :],
            z[t, :],
            config,
            artifacts.field_matrix[t, :],
            interaction_matrix,
            rng,
        )

    return x, z, z_0


def save_artifacts(
    data_folder: str,
    config,
    metadata: dict[str, str],
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    z_0: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> None:
    os.makedirs(data_folder)
    OmegaConf.save(config, f"{data_folder}/realized_config.yaml")
    OmegaConf.save(
        OmegaConf.create(metadata), f"{data_folder}/experiment_metadata.yaml"
    )
    np.savez(f"{data_folder}/panel_data.npz", x=x, z=z)
    np.save(f"{data_folder}/x_0.npy", x_0)
    np.save(f"{data_folder}/z_0.npy", z_0)
    save_model_artifacts(data_folder, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for conditional-model MPLE experiments."
    )
    parser.add_argument("--config_name", type=str, default="base_config.yaml")
    parser.add_argument(
        "--config_override", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--descriptor", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = slugify(args.descriptor) if args.descriptor else "synthetic_data"
    data_folder = f"experiments/{descriptor}_{timestamp}"
    extra_metadata = parse_metadata_entries(args.metadata)

    print("Starting synthetic data generation...")
    config, gamma_matrix, x_0, rng = read_and_realize_config(
        args.config_name, args.config_override
    )
    artifacts = build_synthetic_field(config, gamma_matrix)

    fixed_z_metadata: dict[str, str] = {}
    if intervention_mode(config) == "fixed_z":
        fixed_z, z_0, fixed_z_metadata = load_fixed_intervention_artifacts(config)
        config.global_params.s = derive_pre_intervention_steps(fixed_z)
    else:
        fixed_z = None
        z_0 = np.zeros(int(config.global_params.N), dtype=float)

    print("Generating data with the conditional process...")
    x, z, generated_z_0 = generate_data(config, artifacts, x_0, rng, fixed_z=fixed_z)
    if intervention_mode(config) != "fixed_z":
        z_0 = generated_z_0

    metadata = {
        "descriptor": args.descriptor or descriptor,
        "slug": descriptor,
        "timestamp": timestamp,
        "config_name": args.config_name,
        "config_overrides": list(args.config_override),
        "gamma_inf_norm": float(np.linalg.norm(artifacts.gamma_matrix, ord=np.inf)),
        "gamma_fro_norm": float(np.linalg.norm(artifacts.gamma_matrix, ord="fro")),
        "field_mode": artifacts.field_mode,
        "intervention_mode": intervention_mode(config),
        "has_truth": True,
        **extra_metadata,
        **fixed_z_metadata,
    }
    if artifacts.field_basis is not None:
        metadata["field_basis_inf_norms"] = [
            float(np.linalg.norm(vector, ord=np.inf))
            for vector in artifacts.field_basis
        ]
    if artifacts.tau is not None:
        metadata["tau_l2_norm"] = float(np.linalg.norm(artifacts.tau, ord=2))
    if artifacts.field_mode == "latent_feature_matrix":
        metadata["latent_rank"] = int(artifacts.latent_rank)

    save_artifacts(data_folder, config, metadata, artifacts, x_0, z_0, x, z)
    print("Done!")
    print(
        "Infinity Norm of Gamma Matrix:",
        np.linalg.norm(artifacts.gamma_matrix, ord=np.inf),
    )
    print(
        "Infinity Norm of Interaction Matrix:",
        np.linalg.norm(
            compose_interaction_matrix(get_xi(config), artifacts.gamma_matrix),
            ord=np.inf,
        ),
    )
    print(f"Experiment Folder: {data_folder}")

    if args.manifest_path is not None:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{Path(data_folder).resolve()}\n")


if __name__ == "__main__":
    main()
