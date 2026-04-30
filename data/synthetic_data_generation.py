from __future__ import annotations

import re
import sys
from dataclasses import dataclass
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
    SpectralLowRankStructure,
    build_synthetic_field,
    compose_interaction_matrix,
    get_xi,
    get_synthetic_field_params,
    get_synthetic_field_mode,
    interaction_matrix_infinity_norm,
    leading_svd_low_rank_structure,
    parse_singular_values,
    resolve_generation_confounded_field_ranks,
    sample_spectral_low_rank_structure,
    save_model_artifacts,
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "experiment"


@dataclass(frozen=True)
class InterventionGenerationArtifacts:
    low_rank_structure: SpectralLowRankStructure
    score_matrix: np.ndarray
    probability_matrix: np.ndarray
    z: np.ndarray
    z_0: np.ndarray


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
    return str(
        getattr(config.generation_params, "intervention_mode", "low_rank_probability")
    )


def intervention_params(config) -> dict[str, object]:
    params = getattr(config.generation_params, "intervention_params", {})
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    return dict(params)


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


def sample_low_rank_probability_interventions(
    config,
) -> InterventionGenerationArtifacts:
    params = intervention_params(config)
    singular_values = parse_singular_values(
        params.get("singular_values"),
        context="generation_params.intervention_params.singular_values",
    )
    amplitude = float(params.get("probability_amplitude", 0.5))
    if amplitude < 0.0 or amplitude > 0.5:
        raise ValueError(
            "generation_params.intervention_params.probability_amplitude must lie in [0, 0.5]."
        )

    structure = sample_spectral_low_rank_structure(
        int(config.global_params.N),
        int(config.global_params.T),
        singular_values,
        np.random.default_rng(int(config.generation_params.seed) + 307),
    )
    score_matrix = np.asarray(structure.matrix, dtype=float)
    max_abs = float(np.max(np.abs(score_matrix))) if score_matrix.size else 0.0
    if max_abs > 0.0:
        score_matrix = score_matrix / max_abs
    else:
        score_matrix = np.zeros_like(score_matrix)
    probability_matrix = 0.5 + amplitude * score_matrix
    probability_matrix = np.clip(probability_matrix, 0.0, 1.0)
    z_rng = np.random.default_rng(int(config.generation_params.seed) + 401)
    z = np.where(
        z_rng.random(probability_matrix.shape) < probability_matrix,
        1.0,
        -1.0,
    )
    z_0 = np.zeros(int(config.global_params.N), dtype=float)
    return InterventionGenerationArtifacts(
        low_rank_structure=structure,
        score_matrix=score_matrix,
        probability_matrix=probability_matrix,
        z=z,
        z_0=z_0,
    )


def derive_fixed_intervention_structure_for_field(
    config,
    fixed_z: np.ndarray,
) -> SpectralLowRankStructure:
    field_params = getattr(config.global_params, "field_params", {}) or {}
    if not isinstance(field_params, dict):
        field_params = dict(field_params)
    field_singular_values = parse_singular_values(
        field_params.get("singular_values"),
        context="global_params.field_params.singular_values",
    )
    return leading_svd_low_rank_structure(
        np.asarray(fixed_z, dtype=float),
        int(field_singular_values.size),
    )


def sample_x_t_with_parameters(
    x_prev,
    z_curr,
    beta: float,
    eta: float,
    field_t,
    interaction_matrix,
    rng,
    gibbs_sweeps: int,
    beta_active: np.ndarray | None = None,
):
    x_t = x_prev.copy()
    interaction_x_t = np.asarray(interaction_matrix @ x_t, dtype=float).reshape(-1)
    beta_feature = (
        np.asarray(z_curr, dtype=float)
        if beta_active is None
        else np.asarray(z_curr, dtype=float) * np.asarray(beta_active, dtype=float)
    )
    for _ in range(int(gibbs_sweeps)):
        for i in rng.permutation(int(x_t.shape[0])):
            old_x_i = x_t[i]
            h_x = (
                field_t[i]
                + float(beta) * beta_feature[i]
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


def _simulate_panel(
    x_0: np.ndarray,
    z_0: np.ndarray,
    field_matrix: np.ndarray,
    interaction_matrix,
    beta: float,
    eta: float,
    rng,
    gibbs_sweeps: int,
    z_sampler,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_0 = np.asarray(x_0, dtype=float)
    z_0 = np.asarray(z_0, dtype=float)
    field_matrix = np.asarray(field_matrix, dtype=float)
    if field_matrix.ndim != 2:
        raise ValueError("field_matrix must have shape (T, N).")

    t_steps, n_nodes = field_matrix.shape
    if x_0.shape != (n_nodes,):
        raise ValueError("x_0 shape must match the panel width.")
    if z_0.shape != (n_nodes,):
        raise ValueError("z_0 shape must match the panel width.")

    resolved_e = e if e is not None else t_steps

    x = np.zeros((t_steps, n_nodes), dtype=float)
    z = np.zeros((t_steps, n_nodes), dtype=float)
    x_prev = x_0
    z_prev = z_0
    for t in range(t_steps):
        z_curr = np.asarray(z_sampler(t, x_prev, z_prev), dtype=float)
        if z_curr.shape != (n_nodes,):
            raise ValueError("Each sampled z_t must have shape (N,).")
        beta_active = np.ones(n_nodes, dtype=float)
        if bool(beta_mask_pre_s) and t < int(s):
            beta_active.fill(0.0)
        if bool(beta_mask_post_e) and t >= int(resolved_e):
            beta_active.fill(0.0)
        x_curr = sample_x_t_with_parameters(
            x_prev=x_prev,
            z_curr=z_curr,
            beta=float(beta),
            eta=float(eta),
            field_t=field_matrix[t, :],
            interaction_matrix=interaction_matrix,
            rng=rng,
            gibbs_sweeps=int(gibbs_sweeps),
            beta_active=beta_active,
        )
        z[t, :] = z_curr
        x[t, :] = x_curr
        x_prev = x_curr
        z_prev = z_curr
    return x, z


def simulate_outcomes_given_fixed_interventions(
    x_0: np.ndarray,
    z: np.ndarray,
    field_matrix: np.ndarray,
    interaction_matrix,
    beta: float,
    eta: float,
    rng,
    gibbs_sweeps: int,
    s: int = 0,
    e: int | None = None,
    beta_mask_pre_s: bool = False,
    beta_mask_post_e: bool = False,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    field_matrix = np.asarray(field_matrix, dtype=float)
    if z.ndim != 2 or field_matrix.shape != z.shape:
        raise ValueError("z and field_matrix must both have shape (T, N).")
    x, _ = _simulate_panel(
        x_0=np.asarray(x_0, dtype=float),
        z_0=np.zeros(z.shape[1], dtype=float),
        field_matrix=field_matrix,
        interaction_matrix=interaction_matrix,
        beta=float(beta),
        eta=float(eta),
        rng=rng,
        gibbs_sweeps=int(gibbs_sweeps),
        z_sampler=lambda t, _x_prev, _z_prev: z[t, :],
        s=int(s),
        e=e,
        beta_mask_pre_s=bool(beta_mask_pre_s),
        beta_mask_post_e=bool(beta_mask_post_e),
    )
    return x


def generate_data(
    config,
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    rng,
    fixed_z: np.ndarray | None = None,
    z_0: np.ndarray | None = None,
):
    t_steps = int(config.global_params.T)
    n_nodes = int(config.global_params.N)
    resolved_z_0 = (
        np.zeros(n_nodes, dtype=float) if z_0 is None else np.asarray(z_0, dtype=float)
    )
    if resolved_z_0.shape != (n_nodes,):
        raise ValueError(f"z_0 shape {resolved_z_0.shape} does not match N={n_nodes}.")
    print(
        "Generating panel data with"
        f" T={t_steps}, N={n_nodes},"
        f" intervention_mode={intervention_mode(config)},"
        f" gibbs_sweeps={int(config.generation_params.gibbs_sweeps)}"
    )
    field_matrix = np.asarray(artifacts.field_matrix, dtype=float)
    interaction_matrix = compose_interaction_matrix(
        get_xi(config), artifacts.gamma_matrix
    )
    beta = float(config.estimation_params.beta)
    eta = float(config.estimation_params.eta)
    gibbs_sweeps = int(config.generation_params.gibbs_sweeps)

    mode = intervention_mode(config)
    if mode == "fixed_z":
        if fixed_z is None:
            raise ValueError(
                "fixed_z must be provided when intervention_mode='fixed_z'."
            )
        z = np.asarray(fixed_z, dtype=float)
        if z.shape != (t_steps, n_nodes):
            raise ValueError(
                f"fixed_z shape {z.shape} does not match configured (T, N)=({t_steps}, {n_nodes})."
            )
        print("Using saved intervention panel z.")
        x = simulate_outcomes_given_fixed_interventions(
            x_0=np.asarray(x_0, dtype=float),
            z=z,
            field_matrix=field_matrix,
            interaction_matrix=interaction_matrix,
            beta=beta,
            eta=eta,
            rng=rng,
            gibbs_sweeps=gibbs_sweeps,
        )
        return x, z, resolved_z_0

    if mode != "low_rank_probability":
        raise ValueError(f"Invalid intervention_mode: {mode}")
    if fixed_z is None:
        intervention_artifacts = sample_low_rank_probability_interventions(config)
        z = np.asarray(intervention_artifacts.z, dtype=float)
        resolved_z_0 = np.asarray(intervention_artifacts.z_0, dtype=float)
    else:
        z = np.asarray(fixed_z, dtype=float)
        if z.shape != (t_steps, n_nodes):
            raise ValueError(
                f"fixed_z shape {z.shape} does not match configured (T, N)=({t_steps}, {n_nodes})."
            )
    x = simulate_outcomes_given_fixed_interventions(
        x_0=np.asarray(x_0, dtype=float),
        z=z,
        field_matrix=field_matrix,
        interaction_matrix=interaction_matrix,
        beta=beta,
        eta=eta,
        rng=rng,
        gibbs_sweeps=gibbs_sweeps,
    )
    return x, z, resolved_z_0


def save_artifacts(
    data_folder: Path,
    config,
    metadata: dict[str, str],
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    z_0: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    intervention_generation_artifacts: InterventionGenerationArtifacts | None = None,
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
    if intervention_generation_artifacts is not None:
        np.savez(
            data_folder / "intervention_generation_artifacts.npz",
            node_factors=np.asarray(
                intervention_generation_artifacts.low_rank_structure.node_factors,
                dtype=float,
            ),
            time_factors=np.asarray(
                intervention_generation_artifacts.low_rank_structure.time_factors,
                dtype=float,
            ),
            singular_values=np.asarray(
                intervention_generation_artifacts.low_rank_structure.singular_values,
                dtype=float,
            ),
            score_matrix=np.asarray(
                intervention_generation_artifacts.score_matrix, dtype=float
            ),
            probability_matrix=np.asarray(
                intervention_generation_artifacts.probability_matrix, dtype=float
            ),
            z=np.asarray(intervention_generation_artifacts.z, dtype=float),
            z_0=np.asarray(intervention_generation_artifacts.z_0, dtype=float),
        )
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

    fixed_z_metadata: dict[str, str] = {}
    intervention_generation_artifacts: InterventionGenerationArtifacts | None = None
    intervention_structure: SpectralLowRankStructure | None = None
    if intervention_mode(config) == "fixed_z":
        fixed_z, z_0, fixed_z_metadata = load_fixed_intervention_artifacts(config)
        if get_synthetic_field_mode(config) == "confounded_low_rank":
            intervention_structure = derive_fixed_intervention_structure_for_field(
                config,
                fixed_z,
            )
        print("Loaded fixed intervention path.")
    elif intervention_mode(config) == "low_rank_probability":
        intervention_generation_artifacts = sample_low_rank_probability_interventions(
            config
        )
        intervention_structure = intervention_generation_artifacts.low_rank_structure
        fixed_z = np.asarray(intervention_generation_artifacts.z, dtype=float)
        z_0 = np.asarray(intervention_generation_artifacts.z_0, dtype=float)
        print("Generated interventions from a low-rank probability matrix.")
    else:
        fixed_z = None
        z_0 = np.zeros(int(config.global_params.N), dtype=float)
        print("Using default generated intervention setup with z_0 initialized to zeros.")

    print(
        "Building latent field artifacts with"
        f" field_mode={get_synthetic_field_mode(config)} and"
        f" B={float(config.global_params.B):.4f}."
    )
    artifacts = build_synthetic_field(
        config,
        gamma_matrix,
        intervention_structure=intervention_structure,
    )

    x, z, z_0 = generate_data(
        config,
        artifacts,
        x_0,
        rng,
        fixed_z=fixed_z,
        z_0=z_0,
    )

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
        "field_mode": get_synthetic_field_mode(config),
        "has_truth": True,
        **extra_metadata,
        **fixed_z_metadata,
        **fixed_gamma_metadata,
    }
    metadata["latent_rank"] = int(artifacts.latent_rank)
    if get_synthetic_field_mode(config) == "confounded_low_rank":
        shared_rank, nonshared_rank = resolve_generation_confounded_field_ranks(
            config,
            intervention_structure,
        )
        metadata["field_shared_rank"] = int(shared_rank)
        metadata["field_nonshared_rank"] = int(nonshared_rank)
        metadata["field_shared_basis_source"] = (
            "fixed_z_svd"
            if intervention_mode(config) == "fixed_z"
            else "generated_intervention_basis"
        )
    else:
        field_singular_values = parse_singular_values(
            get_synthetic_field_params(config).get("singular_values"),
            context="global_params.field_params.singular_values",
        )
        metadata["field_shared_rank"] = 0
        metadata["field_nonshared_rank"] = int(field_singular_values.size)
        metadata["field_shared_basis_source"] = "none"

    save_artifacts(
        data_folder,
        config,
        metadata,
        artifacts,
        x_0,
        z_0,
        x,
        z,
        intervention_generation_artifacts=intervention_generation_artifacts,
        config_filename=config_filename,
    )
    print(f"Finished experiment '{descriptor}'.")
    return metadata
