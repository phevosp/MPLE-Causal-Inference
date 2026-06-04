from __future__ import annotations

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

from utils.t3_model_artifacts import (
    ModelArtifacts,
    save_model_artifacts,
)
from utils.t3_field_generation import (
    ConfoundedFieldLayout,
    build_synthetic_field,
    build_synthetic_field_with_layout,
    leading_svd_low_rank_structure,
    parse_synthetic_field_spec,
    parse_singular_values,
    sample_spectral_low_rank_structure,
    SpectralLowRankStructure,
)
from utils.t3_interaction_matrices import (
    compose_interaction_matrix,
)
from utils.t4_scalar_parameters import (
    get_xi,
)
from utils.t2_normalization import interaction_matrix_infinity_norm, normalize_matrix_by_max_abs_entry
from pipeline_specs import slugify


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
        if gamma_matrix.shape[0] != gamma_matrix.shape[1]:
            raise ValueError(f"Fixed gamma artifact must be square: {gamma_path}")
        if config.global_params.N is None:
            config.global_params.N = int(gamma_matrix.shape[0])
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
    else:
        # For non-fixed generators, we require N to be specified to know how large of a graph to generate.
        if config.global_params.N is None:
            raise ValueError(
                "global_params.N must be resolved before generating a non-fixed graph."
            )
        # Generate graph
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
            raise ValueError(f"Invalid gamma_matrix_generator: {generator}")

        # Ensure numpy array; symmetric; zero-diagonal
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

    if z.ndim != 2:
        raise ValueError(f"Fixed-z artifact must be 2D: {panel_path}")
    if config.global_params.N is None:
        config.global_params.N = int(z.shape[1])
    expected_n = int(config.global_params.N)
    if z_0.shape != (expected_n,):
        raise ValueError(
            f"Fixed-z initial state shape {z_0.shape} does not match configured N={expected_n}."
        )
    if config.global_params.T is None:
        config.global_params.T = int(z.shape[0])
    expected_shape = (int(config.global_params.T), expected_n)
    if z.shape != expected_shape:
        raise ValueError(
            f"Fixed-z artifact shape {z.shape} does not match configured (T, N)={expected_shape}."
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


def sample_low_rank_probability_interventions(
    config,
) -> InterventionGenerationArtifacts:
    params = getattr(config.generation_params, "intervention_params", {}) or {}
    if isinstance(params, dict):
        params = dict(params)
    else:
        params = dict(OmegaConf.to_container(params, resolve=True))

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
    score_matrix = normalize_matrix_by_max_abs_entry(structure.matrix)
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
    fixed_z: np.ndarray,
    target_rank: int,
) -> SpectralLowRankStructure:
    return leading_svd_low_rank_structure(
        np.asarray(fixed_z, dtype=float),
        int(target_rank),
    )


def _apply_interaction_term_to_state(interaction_matrix, x_t: np.ndarray) -> np.ndarray:
    """Compute interaction_matrix @ x_t, handling both sparse and dense matrices."""
    if sparse.issparse(interaction_matrix):
        return np.asarray(interaction_matrix @ x_t, dtype=float).reshape(-1)
    return np.asarray(interaction_matrix @ x_t, dtype=float).reshape(-1)


def _update_interaction_term_on_flip(
    interaction_matrix, interaction_x_t: np.ndarray, node_idx: int, delta: float
) -> np.ndarray:
    """Update interaction_x_t after flipping x_t[node_idx] by delta."""
    if sparse.issparse(interaction_matrix):
        interaction_x_t += delta * interaction_matrix[:, node_idx].toarray().ravel()
    else:
        interaction_x_t += delta * interaction_matrix[:, node_idx]
    return interaction_x_t


def _prepare_intervention_structure(
    config, early_field_spec
) -> tuple[np.ndarray, np.ndarray, InterventionGenerationArtifacts | None, SpectralLowRankStructure | None, dict[str, str]]:
    """Prepare intervention data and structure for generation.

    Returns (fixed_z, z_0, intervention_generation_artifacts, intervention_structure, metadata_dict).
    """
    fixed_z_metadata: dict[str, str] = {}
    intervention_generation_artifacts: InterventionGenerationArtifacts | None = None
    intervention_structure: SpectralLowRankStructure | None = None

    if intervention_mode(config) == "fixed_z":
        fixed_z, z_0, fixed_z_metadata = load_fixed_intervention_artifacts(config)
        if early_field_spec.mode == "confounded_low_rank":
            intervention_structure = derive_fixed_intervention_structure_for_field(
                fixed_z, int(early_field_spec.singular_values.size)
            )
    else:
        intervention_generation_artifacts = sample_low_rank_probability_interventions(config)
        intervention_structure = intervention_generation_artifacts.low_rank_structure
        fixed_z = np.asarray(intervention_generation_artifacts.z, dtype=float)
        z_0 = np.asarray(intervention_generation_artifacts.z_0, dtype=float)
        print("Generated interventions from a low-rank probability matrix.")

    return fixed_z, z_0, intervention_generation_artifacts, intervention_structure, fixed_z_metadata


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
    interaction_x_t = _apply_interaction_term_to_state(interaction_matrix, x_t)
    z_curr_array = np.asarray(z_curr, dtype=float)
    beta_feature = (
        z_curr_array
        if beta_active is None
        else z_curr_array * np.asarray(beta_active, dtype=float)
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
            interaction_x_t = _update_interaction_term_on_flip(
                interaction_matrix, interaction_x_t, i, delta
            )
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

    x = np.zeros((t_steps, n_nodes), dtype=float)
    z = np.zeros((t_steps, n_nodes), dtype=float)
    x_prev = x_0
    z_prev = z_0
    for t in range(t_steps):
        z_curr = np.asarray(z_sampler(t, x_prev, z_prev), dtype=float)
        if z_curr.shape != (n_nodes,):
            raise ValueError("Each sampled z_t must have shape (N,).")
        x_curr = sample_x_t_with_parameters(
            x_prev=x_prev,
            z_curr=z_curr,
            beta=float(beta),
            eta=float(eta),
            field_t=field_matrix[t, :],
            interaction_matrix=interaction_matrix,
            rng=rng,
            gibbs_sweeps=int(gibbs_sweeps),
            beta_active=None,
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
) -> np.ndarray:
    """Simulate outcomes from the realized intervention panel without beta masking."""
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
    )
    return x


def generate_data(
    config,
    artifacts: ModelArtifacts,
    x_0: np.ndarray,
    rng,
    z: np.ndarray,
    z_0: np.ndarray,
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

    if z is None:
        raise ValueError("z must be provided to generate data.")
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


def _build_field_layout_metadata(
    field_spec, field_layout: ConfoundedFieldLayout | None, config
) -> dict[str, object]:
    """Build metadata dict for field layout and confounded field configuration."""
    if field_spec.mode == "confounded_low_rank" and field_layout is not None:
        return {
            "field_shared_rank": int(field_layout.shared_rank),
            "field_nonshared_rank": int(field_layout.nonshared_rank),
            "field_shared_basis_source": (
                "fixed_z_svd" if intervention_mode(config) == "fixed_z"
                else "generated_intervention_basis"
            ),
        }
    return {
        "field_shared_rank": 0,
        "field_nonshared_rank": int(field_spec.singular_values.size),
        "field_shared_basis_source": "none",
    }


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
    early_field_spec = parse_synthetic_field_spec(config)

    # Realize generation inputs, including (fixed) interaction matrix
    config, gamma_matrix, x_0, rng, fixed_gamma_metadata = realize_generation_inputs(
        config
    )

    if config.global_params.N is None:
        raise ValueError("global_params.N must be resolved before generation.")
    if config.global_params.T is None:
        raise ValueError("global_params.T must be resolved before generation.")
    field_spec = parse_synthetic_field_spec(config)

    # Prepare intervention data and structure
    fixed_z, z_0, intervention_generation_artifacts, intervention_structure, fixed_z_metadata = (
        _prepare_intervention_structure(config, early_field_spec)
    )
    if intervention_mode(config) not in ("fixed_z", "low_rank_probability"):
        raise ValueError(f"Invalid intervention_mode: {intervention_mode(config)}")

    print(
        "Building latent field artifacts with"
        f" field_mode={field_spec.mode} and"
        f" B={float(config.global_params.B):.4f}."
    )
    build_result = build_synthetic_field_with_layout(
        config,
        gamma_matrix,
        intervention_structure=intervention_structure,
        field_spec=field_spec,
    )
    artifacts = build_result.artifacts
    field_layout = build_result.confounded_layout

    x, z, z_0 = generate_data(
        config,
        artifacts,
        x_0,
        rng,
        fixed_z,
        z_0,
    )

    metadata = {
        "descriptor": descriptor,
        "slug": slugify(descriptor, fallback="experiment"),
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
        "field_mode": field_spec.mode,
        "has_truth": True,
        "latent_rank": int(artifacts.latent_rank),
        **_build_field_layout_metadata(field_spec, field_layout, config),
        **extra_metadata,
        **fixed_z_metadata,
        **fixed_gamma_metadata,
    }

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
