from __future__ import annotations

import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import (
    ModelArtifacts,
    build_synthetic_field,
    compose_interaction_matrix,
    get_xi,
    interaction_matrix_infinity_norm,
    save_model_artifacts,
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "experiment"


def spin_sample_from_field(h, rng):
    p = 1.0 / (1.0 + np.exp(-2.0 * h))
    return 2.0 * (rng.random(np.shape(p)) < p).astype(float) - 1.0


def realize_generation_inputs(config):
    rng = np.random.default_rng(int(config.generation_params.seed))

    generator = str(config.global_params.gamma_matrix_generator)
    print(
        "Resolving generation inputs:"
        f" gamma_source={generator},"
        f" x0_mode={config.global_params.x_0_generator},"
        f" seed={int(config.generation_params.seed)}"
    )
    fixed_gamma_metadata: dict[str, str] = {}
    if generator == "fixed_artifact":
        source = getattr(config.global_params, "fixed_gamma_source", None)
        if source is None:
            raise ValueError(
                "global_params.fixed_gamma_source is required when gamma_matrix_generator='fixed_artifact'."
            )
        gamma_path = Path(str(getattr(source, "gamma_path", "")))
        if not gamma_path.exists():
            raise FileNotFoundError(
                "fixed_gamma_source.gamma_path must exist when gamma_matrix_generator='fixed_artifact'."
            )
        if gamma_path.suffix == ".npz":
            gamma_matrix = sparse.load_npz(gamma_path).tocsr()
        else:
            gamma_matrix = np.asarray(np.load(gamma_path), dtype=float)
        print(f"Loaded fixed graph artifact from {gamma_path}.")
        expected_n = int(config.global_params.N)
        if gamma_matrix.shape != (expected_n, expected_n):
            raise ValueError(
                f"Fixed gamma artifact shape {gamma_matrix.shape} does not match configured N={expected_n}."
            )
        for key in ["artifact_dir", "network_name", "trim_scope", "node_index_path"]:
            value = getattr(source, key, None)
            if value:
                fixed_gamma_metadata[f"fixed_gamma_{key}"] = str(value)
        fixed_gamma_metadata["fixed_gamma_path"] = str(gamma_path.resolve())
    elif generator == "erdos_renyi":
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

    if generator != "fixed_artifact":
        gamma_matrix = nx.to_numpy_array(
            gamma_graph, nodelist=list(gamma_graph.nodes())
        )
        gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2.0
        np.fill_diagonal(gamma_matrix, 0.0)
        print(
            "Generated graph artifact with shape"
            f" {gamma_matrix.shape} using {generator}."
        )

    x0_mode = str(config.global_params.x_0_generator)
    if x0_mode == "bernoulli":
        p = float(config.global_params.x_0_params.p)
        x_0 = (rng.random(int(config.global_params.N)) < p).astype(float) * 2.0 - 1.0
        print(f"Sampled x_0 from Bernoulli(p={p:.4f}).")
    elif x0_mode == "fixed":
        x_0 = np.full(
            int(config.global_params.N),
            float(config.global_params.x_0_params.fixed_val),
        )
        print(
            "Initialized x_0 to a fixed value of"
            f" {float(config.global_params.x_0_params.fixed_val):.4f}."
        )
    else:
        raise ValueError(f"Invalid x_0_generator: {x0_mode}")

    return config, gamma_matrix, x_0, rng, fixed_gamma_metadata


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
    print(
        "Loaded fixed intervention artifacts from"
        f" panel={panel_path} and z0={z0_path}."
    )

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
        "artifact_dir",
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


def sample_x_t_with_parameters(
    x_prev,
    z_curr,
    beta: float,
    eta: float,
    field_t,
    interaction_matrix,
    rng,
    gibbs_sweeps: int,
):
    x_t = x_prev.copy()
    interaction_x_t = np.asarray(interaction_matrix @ x_t, dtype=float).reshape(-1)
    for _ in range(int(gibbs_sweeps)):
        for i in rng.permutation(int(x_t.shape[0])):
            old_x_i = x_t[i]
            h_x = (
                field_t[i]
                + float(beta) * z_curr[i]
                + float(eta) * x_prev[i]
                + interaction_x_t[i]
            )
            x_t[i] = spin_sample_from_field(h_x, rng)
            delta = x_t[i] - old_x_i
            if sparse.issparse(interaction_matrix):
                interaction_x_t += delta * interaction_matrix[:, i].toarray().ravel()
            else:
                interaction_x_t += delta * interaction_matrix[:, i]
    return x_t


def sample_x_t(x_prev, z_curr, config, field_t, interaction_matrix, rng):
    return sample_x_t_with_parameters(
        x_prev=x_prev,
        z_curr=z_curr,
        beta=float(config.estimation_params.beta),
        eta=float(config.estimation_params.eta),
        field_t=field_t,
        interaction_matrix=interaction_matrix,
        rng=rng,
        gibbs_sweeps=int(config.generation_params.gibbs_sweeps),
    )


def simulate_outcomes_given_fixed_interventions(
    x_0: np.ndarray,
    z: np.ndarray,
    field_matrix: np.ndarray,
    interaction_matrix,
    beta: float,
    eta: float,
    rng,
    gibbs_sweeps: int,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    field_matrix = np.asarray(field_matrix, dtype=float)
    if z.ndim != 2 or field_matrix.shape != z.shape:
        raise ValueError("z and field_matrix must both have shape (T, N).")
    if x_0.shape != (z.shape[1],):
        raise ValueError("x_0 shape must match the panel width.")
    x = np.zeros_like(z, dtype=float)
    x_prev = np.asarray(x_0, dtype=float)
    for t in range(z.shape[0]):
        x_curr = sample_x_t_with_parameters(
            x_prev=x_prev,
            z_curr=z[t, :],
            beta=float(beta),
            eta=float(eta),
            field_t=field_matrix[t, :],
            interaction_matrix=interaction_matrix,
            rng=rng,
            gibbs_sweeps=int(gibbs_sweeps),
        )
        x[t, :] = x_curr
        x_prev = x_curr
    return x


def generate_data(
    config,
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    rng,
    fixed_z: np.ndarray | None = None,
):
    t_steps = int(config.global_params.T)
    n_nodes = int(config.global_params.N)
    print(
        "Generating panel data with"
        f" T={t_steps}, N={n_nodes}, s={int(config.global_params.s)},"
        f" intervention_mode={intervention_mode(config)},"
        f" gibbs_sweeps={int(config.generation_params.gibbs_sweeps)}"
    )
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
        print("Using saved intervention panel z.")

    if mode == "generated_z":
        print("Sampling intervention process z.")
        z[0, :] = (
            sample_z_t(x_0, z_0, config, rng)
            if int(config.global_params.s) == 0
            else -np.ones_like(x_0)
        )
    print("Sampling outcomes x.")
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
    data_folder: Path,
    config,
    metadata: dict[str, str],
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    z_0: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    config_filename: str = "realized_config.yaml",
) -> None:
    print(f"Saving experiment artifacts to {data_folder}...")
    data_folder.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, data_folder / config_filename)
    OmegaConf.save(OmegaConf.create(metadata), data_folder / "experiment_metadata.yaml")
    np.savez(data_folder / "panel_data.npz", x=x, z=z)
    np.save(data_folder / "x_0.npy", x_0)
    np.save(data_folder / "z_0.npy", z_0)
    save_model_artifacts(data_folder, artifacts)
    print("Finished saving generation artifacts.")


def materialize_generation_experiment(
    config,
    data_folder: Path,
    descriptor: str,
    config_label: str = "inline_generation_config",
    extra_metadata: dict[str, object] | None = None,
    config_filename: str = "generation_realized_config.yaml",
) -> dict[str, object]:
    extra_metadata = dict(extra_metadata or {})
    print(f"Materializing generation experiment '{descriptor}'...")
    config, gamma_matrix, x_0, rng, fixed_gamma_metadata = realize_generation_inputs(
        config
    )
    print(
        "Building latent field artifacts with"
        f" latent_rank={int(config.global_params.latent_rank)} and"
        f" B={float(config.global_params.B):.4f}."
    )
    artifacts = build_synthetic_field(config, gamma_matrix)

    fixed_z_metadata: dict[str, str] = {}
    if intervention_mode(config) == "fixed_z":
        fixed_z, z_0, fixed_z_metadata = load_fixed_intervention_artifacts(config)
        config.global_params.s = derive_pre_intervention_steps(fixed_z)
        print(
            "Derived pre-intervention length from fixed z:"
            f" s={int(config.global_params.s)}."
        )
    else:
        fixed_z = None
        z_0 = np.zeros(int(config.global_params.N), dtype=float)
        print("Using generated intervention path with z_0 initialized to zeros.")

    x, z, generated_z_0 = generate_data(config, artifacts, x_0, rng, fixed_z=fixed_z)
    if intervention_mode(config) != "fixed_z":
        z_0 = generated_z_0

    metadata = {
        "descriptor": descriptor,
        "slug": slugify(descriptor),
        "config_name": config_label,
        "gamma_inf_norm": interaction_matrix_infinity_norm(artifacts.gamma_matrix),
        "gamma_fro_norm": (
            float(
                np.sqrt(artifacts.gamma_matrix.multiply(artifacts.gamma_matrix).sum())
            )
            if sparse.issparse(artifacts.gamma_matrix)
            else float(np.linalg.norm(artifacts.gamma_matrix, ord="fro"))
        ),
        "intervention_mode": intervention_mode(config),
        "has_truth": True,
        **extra_metadata,
        **fixed_z_metadata,
        **fixed_gamma_metadata,
    }
    metadata["latent_rank"] = int(artifacts.latent_rank)

    save_artifacts(
        data_folder,
        config,
        metadata,
        artifacts,
        x_0,
        z_0,
        x,
        z,
        config_filename=config_filename,
    )
    print(f"Finished experiment '{descriptor}'.")
    return metadata
