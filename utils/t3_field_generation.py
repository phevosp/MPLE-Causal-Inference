"""Synthetic field generation and specification parsing."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from utils.t0_config_utils import load_yaml_config, to_plain_mapping
from utils.t2_normalization import normalize_known_graph
from utils.t3_model_artifacts import (
    ModelArtifacts,
    ConfoundedFieldLayout,
    SpectralLowRankStructure,
    SyntheticFieldBuildResult,
    SyntheticFieldSpec,
    SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK,
    SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK,
    _DEGENERACY_THRESHOLD,
    _DEFAULT_FIELD_RMS_FRACTION,
)
from utils.t3_field_operations import (
    compose_latent_field_matrix,
    project_latent_field,
    scale_latent_field_matrix,
    zero_latent_field,
)
from utils.t4_scalar_parameters import get_B


def _get_synthetic_field_params(config) -> dict[str, object]:
    global_params = getattr(config, "global_params", None)
    if global_params is None or "field_params" not in global_params:
        return {}
    return to_plain_mapping(global_params.field_params)


def parse_singular_values(
    raw_values: object | None,
    *,
    context: str,
) -> np.ndarray:
    if raw_values is None:
        return np.zeros(0, dtype=float)
    values = np.asarray(raw_values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{context} must contain only finite numbers.")
    if np.any(values < 0.0):
        raise ValueError(f"{context} must be nonnegative.")
    return values


def _orthonormal_gaussian_factors(
    n_rows: int,
    rank: int,
    rng,
) -> np.ndarray:
    if rank < 0:
        raise ValueError("rank must be nonnegative.")
    if rank == 0:
        return np.zeros((n_rows, 0), dtype=float)
    if rank > n_rows:
        raise ValueError(f"rank={rank} exceeds available dimension {n_rows}.")
    q, _ = np.linalg.qr(rng.normal(size=(n_rows, rank)), mode="reduced")
    return np.asarray(q[:, :rank], dtype=float)


def sample_spectral_low_rank_structure(
    n_nodes: int,
    t_steps: int,
    singular_values: np.ndarray,
    rng,
) -> SpectralLowRankStructure:
    singular_values = np.asarray(singular_values, dtype=float).reshape(-1)
    rank = int(singular_values.size)
    if rank > min(int(n_nodes), int(t_steps)):
        raise ValueError(
            f"rank={rank} exceeds min(N, T)={min(int(n_nodes), int(t_steps))}."
        )
    node_factors = _orthonormal_gaussian_factors(int(n_nodes), rank, rng)
    time_factors = _orthonormal_gaussian_factors(int(t_steps), rank, rng)
    if rank == 0:
        matrix = zero_latent_field(int(n_nodes), int(t_steps))
    else:
        matrix = (time_factors * singular_values[None, :]) @ node_factors.T
    return SpectralLowRankStructure(
        node_factors=node_factors,
        time_factors=time_factors,
        singular_values=singular_values,
        matrix=np.asarray(matrix, dtype=float),
    )


def leading_svd_low_rank_structure(
    matrix: np.ndarray,
    rank: int,
) -> SpectralLowRankStructure:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape (T, N).")
    rank = int(rank)
    t_steps, n_nodes = matrix.shape
    if rank < 0:
        raise ValueError("rank must be nonnegative.")
    if rank > min(t_steps, n_nodes):
        raise ValueError(
            f"rank={rank} exceeds min(T, N)={min(t_steps, n_nodes)} for SVD truncation."
        )
    if rank == 0:
        return SpectralLowRankStructure(
            node_factors=np.zeros((n_nodes, 0), dtype=float),
            time_factors=np.zeros((t_steps, 0), dtype=float),
            singular_values=np.zeros(0, dtype=float),
            matrix=np.zeros_like(matrix, dtype=float),
        )
    time_factors, singular_values, node_factors_t = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    truncated_time_factors = np.asarray(time_factors[:, :rank], dtype=float)
    truncated_singular_values = np.asarray(singular_values[:rank], dtype=float)
    truncated_node_factors = np.asarray(node_factors_t[:rank, :].T, dtype=float)
    truncated_matrix = (
        truncated_time_factors * truncated_singular_values[None, :]
    ) @ truncated_node_factors.T
    return SpectralLowRankStructure(
        node_factors=truncated_node_factors,
        time_factors=truncated_time_factors,
        singular_values=truncated_singular_values,
        matrix=np.asarray(truncated_matrix, dtype=float),
    )


def _parse_optional_nonnegative_int(
    raw_value: object | None,
    *,
    context: str,
) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{context} must be a nonnegative integer.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a nonnegative integer.") from exc
    if not np.isfinite(value) or value < 0.0 or not value.is_integer():
        raise ValueError(f"{context} must be a nonnegative integer.")
    return int(value)


def parse_synthetic_field_spec(config) -> SyntheticFieldSpec:
    global_params = getattr(config, "global_params", None)
    if global_params is None:
        raise KeyError("global_params is required.")
    raw_mode = getattr(global_params, "field_mode", SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK)
    field_mode = str(raw_mode).strip()
    from utils.t3_model_artifacts import VALID_SYNTHETIC_FIELD_MODES
    if field_mode not in VALID_SYNTHETIC_FIELD_MODES:
        raise ValueError(
            "global_params.field_mode must be one of: "
            + ", ".join(sorted(VALID_SYNTHETIC_FIELD_MODES))
        )

    field_params = _get_synthetic_field_params(config)
    singular_values = parse_singular_values(
        field_params.get("singular_values"),
        context="global_params.field_params.singular_values",
    )
    target_rms_fraction = float(
        field_params.get("target_rms_fraction", _DEFAULT_FIELD_RMS_FRACTION)
    )
    if target_rms_fraction < 0.0:
        raise ValueError(
            "global_params.field_params.target_rms_fraction must be nonnegative."
        )
    shared_rank = _parse_optional_nonnegative_int(
        field_params.get("shared_rank"),
        context="global_params.field_params.shared_rank",
    )
    if (
        shared_rank is not None
        and field_mode != SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
    ):
        raise ValueError(
            "global_params.field_params.shared_rank is only valid when "
            "global_params.field_mode='confounded_low_rank'."
        )
    return SyntheticFieldSpec(
        mode=field_mode,
        singular_values=singular_values,
        target_rms_fraction=target_rms_fraction,
        shared_rank=shared_rank,
        B=get_B(config),
        seed=int(config.generation_params.seed),
        n_nodes=(
            None
            if getattr(config.global_params, "N", None) is None
            else int(config.global_params.N)
        ),
        t_steps=(
            None
            if getattr(config.global_params, "T", None) is None
            else int(config.global_params.T)
        ),
    )


def _effective_structure_rank(structure: SpectralLowRankStructure) -> int:
    return int(np.count_nonzero(np.abs(structure.singular_values) > _DEGENERACY_THRESHOLD))


def resolve_confounded_field_layout(
    field_spec: SyntheticFieldSpec,
    intervention_structure: SpectralLowRankStructure | None,
) -> ConfoundedFieldLayout:
    if intervention_structure is None:
        raise ValueError(
            "field_mode='confounded_low_rank' requires low-rank intervention factors, "
            "either generated directly or derived from a fixed intervention panel."
        )
    total_rank = int(field_spec.singular_values.size)
    if total_rank == 0:
        raise ValueError(
            "global_params.field_params.singular_values must be provided for this field mode."
        )
    available_shared_rank = _effective_structure_rank(intervention_structure)
    if field_spec.shared_rank is None:
        shared_rank = available_shared_rank
        if total_rank != available_shared_rank:
            raise ValueError(
                "global_params.field_params.singular_values must have the same length as the "
                "shared intervention low-rank basis for field_mode='confounded_low_rank' "
                "when shared_rank is omitted."
            )
    else:
        shared_rank = int(field_spec.shared_rank)
        if shared_rank > total_rank:
            raise ValueError(
                "global_params.field_params.shared_rank must not exceed the total field rank "
                "defined by global_params.field_params.singular_values."
            )
        if shared_rank > available_shared_rank:
            raise ValueError(
                "global_params.field_params.shared_rank must not exceed the available "
                "intervention basis rank for field_mode='confounded_low_rank'."
            )
    return ConfoundedFieldLayout(
        total_rank=total_rank,
        available_shared_rank=available_shared_rank,
        shared_rank=shared_rank,
        nonshared_rank=int(total_rank - shared_rank),
    )


def _field_rng(field_spec: SyntheticFieldSpec, offset: int):
    return np.random.default_rng(int(field_spec.seed) + int(offset))


def _scale_spectral_field(
    field_matrix: np.ndarray,
    field_spec: SyntheticFieldSpec,
) -> np.ndarray:
    from utils.t2_normalization import normalize_matrix_max_abs
    field_matrix = normalize_matrix_max_abs(field_matrix, max_abs=1.0)
    target_rms = float(field_spec.target_rms_fraction) * float(field_spec.B)
    return scale_latent_field_matrix(field_matrix, target_rms)


def _build_random_low_rank_field(
    field_spec: SyntheticFieldSpec,
) -> tuple[np.ndarray, int]:
    if field_spec.singular_values.size == 0:
        return zero_latent_field(field_spec.n_nodes, field_spec.t_steps), 0
    structure = sample_spectral_low_rank_structure(
        field_spec.n_nodes,
        field_spec.t_steps,
        field_spec.singular_values,
        _field_rng(field_spec, 101),
    )
    field_matrix = _scale_spectral_field(structure.matrix, field_spec)
    return field_matrix, int(field_spec.singular_values.size)


def _orthonormal_complement_gaussian_factors(
    n_rows: int,
    rank: int,
    existing_factors: np.ndarray,
    rng,
) -> np.ndarray:
    if rank < 0:
        raise ValueError("rank must be nonnegative.")
    if rank == 0:
        return np.zeros((n_rows, 0), dtype=float)
    existing = np.asarray(existing_factors, dtype=float)
    if existing.ndim != 2 or existing.shape[0] != int(n_rows):
        raise ValueError("existing_factors must have shape (n_rows, existing_rank).")
    existing_rank = int(existing.shape[1])
    if existing_rank + rank > int(n_rows):
        raise ValueError(
            "Requested orthogonal-complement rank exceeds the remaining ambient dimension."
        )
    if existing_rank == 0:
        return _orthonormal_gaussian_factors(int(n_rows), int(rank), rng)

    existing_q, _ = np.linalg.qr(existing, mode="reduced")
    for _ in range(8):
        proposal = rng.normal(size=(int(n_rows), int(rank)))
        projected = proposal - existing_q @ (existing_q.T @ proposal)
        q, r = np.linalg.qr(projected, mode="reduced")
        diag = np.abs(np.diag(r))
        if diag.size >= int(rank) and np.all(diag[:rank] > _DEGENERACY_THRESHOLD):
            return np.asarray(q[:, :rank], dtype=float)
    raise ValueError(
        "Unable to sample a stable orthogonal complement for the requested nonshared rank."
    )


def _build_confounded_low_rank_field(
    field_spec: SyntheticFieldSpec,
    intervention_structure: SpectralLowRankStructure | None,
) -> tuple[np.ndarray, int, ConfoundedFieldLayout]:
    layout = resolve_confounded_field_layout(
        field_spec,
        intervention_structure,
    )
    shared_time_factors = np.asarray(
        intervention_structure.time_factors[:, : layout.shared_rank],
        dtype=float,
    )
    shared_node_factors = np.asarray(
        intervention_structure.node_factors[:, : layout.shared_rank],
        dtype=float,
    )
    rng = _field_rng(field_spec, 211)
    nonshared_time_factors = _orthonormal_complement_gaussian_factors(
        intervention_structure.time_factors.shape[0],
        layout.nonshared_rank,
        shared_time_factors,
        rng,
    )
    nonshared_node_factors = _orthonormal_complement_gaussian_factors(
        intervention_structure.node_factors.shape[0],
        layout.nonshared_rank,
        shared_node_factors,
        rng,
    )
    time_factors = np.concatenate(
        [shared_time_factors, nonshared_time_factors],
        axis=1,
    )
    node_factors = np.concatenate(
        [shared_node_factors, nonshared_node_factors],
        axis=1,
    )
    field_matrix = (time_factors * field_spec.singular_values[None, :]) @ node_factors.T
    field_matrix = _scale_spectral_field(field_matrix, field_spec)
    return field_matrix, int(field_spec.singular_values.size), layout


def build_synthetic_field(
    config,
    gamma_matrix,
    intervention_structure: SpectralLowRankStructure | None = None,
    field_spec: SyntheticFieldSpec | None = None,
) -> ModelArtifacts:
    return build_synthetic_field_with_layout(
        config,
        gamma_matrix,
        intervention_structure=intervention_structure,
        field_spec=field_spec,
    ).artifacts


def build_synthetic_field_with_layout(
    config,
    gamma_matrix,
    intervention_structure: SpectralLowRankStructure | None = None,
    field_spec: SyntheticFieldSpec | None = None,
) -> SyntheticFieldBuildResult:
    from utils.t2_normalization import validate_graph_infinity_norm
    field_spec = parse_synthetic_field_spec(config) if field_spec is None else field_spec
    if field_spec.n_nodes is None or field_spec.t_steps is None:
        raise ValueError("global_params.N and global_params.T must be resolved before building the synthetic field.")
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    confounded_layout: ConfoundedFieldLayout | None = None
    if field_spec.mode == SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK:
        field_matrix, latent_rank = _build_random_low_rank_field(field_spec)
    elif field_spec.mode == SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK:
        field_matrix, latent_rank, confounded_layout = _build_confounded_low_rank_field(
            field_spec,
            intervention_structure,
        )
    else:
        raise ValueError(
            "Unsupported synthetic field_mode: "
            f"{field_spec.mode}"
        )
    return SyntheticFieldBuildResult(
        artifacts=ModelArtifacts(
            gamma_matrix=gamma_matrix,
            t_steps=field_spec.t_steps,
            latent_rank=int(latent_rank),
            optimizer_mode="no_external_field",
            field_matrix=field_matrix,
        ),
        confounded_layout=confounded_layout,
    )
