"""Minimal shared helpers for the latent-only conditional MPLE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from io_utils import io_path


DEFAULT_LATENT_RANK = 0
_DEGENERACY_THRESHOLD = 1e-12   # norms below this are treated as zero/degenerate
_DEFAULT_FIELD_RMS_FRACTION = 0.4
SCALAR_PARAMETER_ORDER = ("beta", "xi", "eta")
SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK = "random_low_rank"
SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK = "confounded_low_rank"
VALID_SYNTHETIC_FIELD_MODES = {
    SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK,
    SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK,
}
OPTIMIZER_MODE_NO_EXTERNAL_FIELD = "no_external_field"
OPTIMIZER_MODE_NUCLEAR_NORM = "nuclear_norm"
OPTIMIZER_MODE_EXACT_RANK_MANIFOLD = "exact_rank_manifold"
OPTIMIZER_MODE_ALTERNATING_LATENT_RANK = "alternating_latent_rank"
OPTIMIZER_MODE_CONCURRENT_LATENT_RANK = "concurrent_latent_rank"
VALID_OPTIMIZER_MODES = {
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
}


@dataclass(frozen=True)
class ModelArtifacts:
    """Latent-field artifacts for one experiment."""

    gamma_matrix: object
    t_steps: int
    latent_rank: int = 0
    optimizer_mode: str = OPTIMIZER_MODE_EXACT_RANK_MANIFOLD
    field_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class SpectralLowRankStructure:
    """Low-rank matrix structure with canonical panel orientation (T, N)."""

    node_factors: np.ndarray
    time_factors: np.ndarray
    singular_values: np.ndarray
    matrix: np.ndarray


def scalar_parameter_names() -> list[str]:
    return list(SCALAR_PARAMETER_ORDER)


def validate_fixed_scalar_params(
    fixed_scalar_params: dict[str, float] | None,
) -> dict[str, float]:
    fixed_scalar_params = {
        str(key): float(value) for key, value in (fixed_scalar_params or {}).items()
    }
    invalid = sorted(set(fixed_scalar_params) - set(SCALAR_PARAMETER_ORDER))
    if invalid:
        raise ValueError(f"Unknown fixed scalar parameter(s): {', '.join(invalid)}.")
    return fixed_scalar_params


def free_scalar_parameter_names(
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    return [name for name in scalar_parameter_names() if name not in fixed]


def interaction_matrix_infinity_norm(matrix) -> float:
    if sparse.issparse(matrix):
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        return float(row_sums.max()) if row_sums.size else 0.0
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def get_latent_rank(config) -> int:
    if "latent_rank" not in config.global_params:
        return DEFAULT_LATENT_RANK
    rank = int(config.global_params.latent_rank)
    if rank < 0:
        raise ValueError("global_params.latent_rank must be nonnegative.")
    return rank


def get_optimizer_mode(config) -> str:
    global_params = getattr(config, "global_params", None)
    if global_params is None or "optimizer_mode" not in global_params:
        if global_params is not None and int(global_params.get("latent_rank", 0)) > 0:
            return OPTIMIZER_MODE_EXACT_RANK_MANIFOLD
        return OPTIMIZER_MODE_NO_EXTERNAL_FIELD
    optimizer_mode = str(global_params.optimizer_mode)
    if optimizer_mode not in VALID_OPTIMIZER_MODES:
        raise ValueError(
            "global_params.optimizer_mode must be one of: "
            + ", ".join(sorted(VALID_OPTIMIZER_MODES))
        )
    return optimizer_mode


def uses_full_matrix_parameterization(
    artifacts: ModelArtifacts,
) -> bool:
    return artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM


def get_B(config) -> float:
    if "B" not in config.global_params:
        raise KeyError("global_params.B is required.")
    return float(config.global_params.B)


def get_xi(config) -> float:
    if "xi" not in config.estimation_params:
        raise KeyError("estimation_params.xi is required.")
    return float(config.estimation_params.xi)


def get_synthetic_field_mode(config) -> str:
    global_params = getattr(config, "global_params", None)
    if global_params is None or "field_mode" not in global_params:
        return SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK
    field_mode = str(global_params.field_mode).strip()
    if field_mode not in VALID_SYNTHETIC_FIELD_MODES:
        raise ValueError(
            "global_params.field_mode must be one of: "
            + ", ".join(sorted(VALID_SYNTHETIC_FIELD_MODES))
        )
    return field_mode


def get_synthetic_field_params(config) -> dict[str, object]:
    global_params = getattr(config, "global_params", None)
    if global_params is None or "field_params" not in global_params:
        return {}
    field_params = global_params.field_params
    if field_params is None:
        return {}
    if isinstance(field_params, dict):
        return dict(field_params)
    return dict(field_params)


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


def _normalize_dense_graph(gamma_matrix: np.ndarray) -> np.ndarray:
    gamma_matrix = np.asarray(gamma_matrix, dtype=float)
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2.0
    np.fill_diagonal(gamma_matrix, 0.0)
    norm = float(np.linalg.norm(gamma_matrix, ord=np.inf))
    if norm < _DEGENERACY_THRESHOLD:
        return np.zeros_like(gamma_matrix)
    return gamma_matrix / norm


def _normalize_sparse_graph(gamma_matrix) -> sparse.csr_matrix:
    normalized = sparse.csr_matrix(gamma_matrix, dtype=float)
    normalized = ((normalized + normalized.T) * 0.5).tocsr()
    normalized.setdiag(0.0)
    normalized.eliminate_zeros()
    norm = interaction_matrix_infinity_norm(normalized)
    if norm < _DEGENERACY_THRESHOLD:
        return sparse.csr_matrix(normalized.shape, dtype=float)
    return normalized.multiply(1.0 / norm).tocsr()


def normalize_known_graph(gamma_matrix):
    if sparse.issparse(gamma_matrix):
        return _normalize_sparse_graph(gamma_matrix)
    return _normalize_dense_graph(np.asarray(gamma_matrix, dtype=float))


def validate_graph_infinity_norm(gamma_matrix, tol: float = 1e-8) -> None:
    gamma_norm = interaction_matrix_infinity_norm(gamma_matrix)
    if gamma_norm < tol:
        raise ValueError("Known graph is degenerate.")
    if not np.isclose(gamma_norm, 1.0, atol=tol, rtol=0.0):
        raise ValueError("The known graph must have infinity norm one.")


def compose_latent_field_matrix(
    node_factors: np.ndarray, time_factors: np.ndarray
) -> np.ndarray:
    """Return the (T × N) field matrix as time_factors @ node_factors.T.

    Both node_factors (N × r) and time_factors (T × r) share r latent dimensions.
    The resulting matrix has entry [t, n] = <time_factors[t], node_factors[n]>.
    """
    return (
        np.asarray(time_factors, dtype=float) @ np.asarray(node_factors, dtype=float).T
    )


def latent_field_bound_norm(field_matrix: np.ndarray) -> float:
    """Maximum absolute entry used for the latent-field B constraint."""
    field_matrix = np.asarray(field_matrix, dtype=float)
    if field_matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(field_matrix)))


def zero_latent_field(n_nodes: int, t_steps: int) -> np.ndarray:
    return np.zeros((t_steps, n_nodes), dtype=float)


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


def normalize_matrix_max_abs(
    matrix: np.ndarray,
    *,
    max_abs: float = 1.0,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if max_abs < 0.0:
        raise ValueError("max_abs must be nonnegative.")
    current = latent_field_bound_norm(matrix)
    if current < _DEGENERACY_THRESHOLD or max_abs == 0.0:
        return np.zeros_like(matrix)
    return matrix * (float(max_abs) / current)


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


def truncate_matrix_rank(field_matrix: np.ndarray, rank: int) -> np.ndarray:
    """Project field_matrix onto its best rank-r approximation via truncated SVD."""
    field_matrix = np.asarray(field_matrix, dtype=float)
    max_rank = min(max(int(rank), 0), *field_matrix.shape)
    if max_rank == 0:
        return np.zeros_like(field_matrix)
    u, singular_values, vt = np.linalg.svd(field_matrix, full_matrices=False)
    return (u[:, :max_rank] * singular_values[:max_rank]) @ vt[:max_rank, :]


def scale_latent_field_matrix(
    field_matrix: np.ndarray,
    target_rms: float,
) -> np.ndarray:
    """Rescale field_matrix to have RMS ≈ target_rms.

    Returns the zero matrix if the initial RMS is below _DEGENERACY_THRESHOLD.
    """
    field_matrix = np.asarray(field_matrix, dtype=float)
    if target_rms < 0.0:
        raise ValueError("target_rms must be nonnegative.")
    rms = float(np.sqrt(np.mean(field_matrix**2)))
    if rms < _DEGENERACY_THRESHOLD:
        return np.zeros_like(field_matrix)
    return field_matrix * (target_rms / rms)


def project_latent_field(
    node_factors: np.ndarray, time_factors: np.ndarray, bound: float
) -> tuple[np.ndarray, np.ndarray]:
    node_factors = np.asarray(node_factors, dtype=float).copy()
    time_factors = np.asarray(time_factors, dtype=float).copy()
    field_matrix = compose_latent_field_matrix(node_factors, time_factors)
    norm = latent_field_bound_norm(field_matrix)
    if norm <= bound or norm < _DEGENERACY_THRESHOLD:
        return node_factors, time_factors
    scale = np.sqrt(bound / norm)
    return node_factors * scale, time_factors * scale


def _resolve_generation_field_singular_values(
    config,
    *,
    allow_empty: bool = True,
) -> np.ndarray:
    field_params = get_synthetic_field_params(config)
    singular_values = parse_singular_values(
        field_params.get("singular_values"),
        context="global_params.field_params.singular_values",
    )
    if singular_values.size == 0 and not allow_empty:
        raise ValueError(
            "global_params.field_params.singular_values must be provided for this field mode."
        )
    return singular_values


def _resolve_generation_field_rms_fraction(config) -> float:
    field_params = get_synthetic_field_params(config)
    fraction = float(
        field_params.get("target_rms_fraction", _DEFAULT_FIELD_RMS_FRACTION)
    )
    if fraction < 0.0:
        raise ValueError(
            "global_params.field_params.target_rms_fraction must be nonnegative."
        )
    return fraction


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


def _resolve_generation_field_shared_rank(config) -> int | None:
    field_params = get_synthetic_field_params(config)
    return _parse_optional_nonnegative_int(
        field_params.get("shared_rank"),
        context="global_params.field_params.shared_rank",
    )


def resolve_generation_confounded_field_ranks(
    config,
    intervention_structure: SpectralLowRankStructure | None,
) -> tuple[int, int]:
    if intervention_structure is None:
        raise ValueError(
            "field_mode='confounded_low_rank' requires low-rank intervention factors, "
            "either generated directly or derived from a fixed intervention panel."
        )
    singular_values = _resolve_generation_field_singular_values(
        config,
        allow_empty=False,
    )
    total_rank = int(singular_values.size)
    available_rank = int(intervention_structure.singular_values.size)
    requested_shared_rank = _resolve_generation_field_shared_rank(config)
    if requested_shared_rank is None:
        shared_rank = available_rank
        if total_rank != available_rank:
            raise ValueError(
                "global_params.field_params.singular_values must have the same length as the "
                "shared intervention low-rank basis for field_mode='confounded_low_rank' "
                "when shared_rank is omitted."
            )
    else:
        shared_rank = int(requested_shared_rank)
        if shared_rank > total_rank:
            raise ValueError(
                "global_params.field_params.shared_rank must not exceed the total field rank "
                "defined by global_params.field_params.singular_values."
            )
        if shared_rank > available_rank:
            raise ValueError(
                "global_params.field_params.shared_rank must not exceed the available "
                "intervention basis rank for field_mode='confounded_low_rank'."
            )
    return shared_rank, int(total_rank - shared_rank)


def _scale_spectral_field(field_matrix: np.ndarray, config) -> np.ndarray:
    field_matrix = normalize_matrix_max_abs(field_matrix, max_abs=1.0)
    target_rms = _resolve_generation_field_rms_fraction(config) * get_B(config)
    return scale_latent_field_matrix(field_matrix, target_rms)


def _sample_random_low_rank_field(
    config,
    n_nodes: int,
    t_steps: int,
) -> tuple[np.ndarray, int]:
    singular_values = _resolve_generation_field_singular_values(config)
    if singular_values.size == 0:
        return zero_latent_field(n_nodes, t_steps), 0
    structure = sample_spectral_low_rank_structure(
        n_nodes,
        t_steps,
        singular_values,
        np.random.default_rng(int(config.generation_params.seed) + 101),
    )
    field_matrix = _scale_spectral_field(structure.matrix, config)
    return field_matrix, int(singular_values.size)


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


def _sample_confounded_low_rank_field(
    config,
    intervention_structure: SpectralLowRankStructure | None,
) -> tuple[np.ndarray, int]:
    singular_values = _resolve_generation_field_singular_values(
        config,
        allow_empty=False,
    )
    shared_rank, nonshared_rank = resolve_generation_confounded_field_ranks(
        config,
        intervention_structure,
    )
    shared_time_factors = np.asarray(
        intervention_structure.time_factors[:, :shared_rank],
        dtype=float,
    )
    shared_node_factors = np.asarray(
        intervention_structure.node_factors[:, :shared_rank],
        dtype=float,
    )
    rng = np.random.default_rng(int(config.generation_params.seed) + 211)
    nonshared_time_factors = _orthonormal_complement_gaussian_factors(
        intervention_structure.time_factors.shape[0],
        nonshared_rank,
        shared_time_factors,
        rng,
    )
    nonshared_node_factors = _orthonormal_complement_gaussian_factors(
        intervention_structure.node_factors.shape[0],
        nonshared_rank,
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
    field_matrix = (time_factors * singular_values[None, :]) @ node_factors.T
    field_matrix = _scale_spectral_field(field_matrix, config)
    return field_matrix, int(singular_values.size)


def build_synthetic_field(
    config,
    gamma_matrix,
    intervention_structure: SpectralLowRankStructure | None = None,
) -> ModelArtifacts:
    n_nodes = int(config.global_params.N)
    t_steps = int(config.global_params.T)
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    field_mode = get_synthetic_field_mode(config)
    if (
        _resolve_generation_field_shared_rank(config) is not None
        and field_mode != SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
    ):
        raise ValueError(
            "global_params.field_params.shared_rank is only valid when "
            "global_params.field_mode='confounded_low_rank'."
        )
    if field_mode == SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK:
        field_matrix, latent_rank = _sample_random_low_rank_field(
            config, n_nodes, t_steps
        )
    elif field_mode == SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK:
        field_matrix, latent_rank = _sample_confounded_low_rank_field(
            config,
            intervention_structure,
        )
    else:
        raise ValueError(
            "Unsupported synthetic field_mode: "
            f"{field_mode}"
        )
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=t_steps,
        latent_rank=int(latent_rank),
        optimizer_mode=OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        field_matrix=field_matrix,
    )


def build_fit_model_artifacts(config, gamma_matrix) -> ModelArtifacts:
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    optimizer_mode = get_optimizer_mode(config)
    latent_rank = get_latent_rank(config)
    if optimizer_mode in {
        OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        OPTIMIZER_MODE_NUCLEAR_NORM,
    }:
        latent_rank = 0
    elif (
        optimizer_mode in {
            OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
            OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
        }
        and latent_rank <= 0
    ):
        raise ValueError(
            "global_params.latent_rank must be positive for optimizer_mode="
            f"'{optimizer_mode}'."
        )
    elif optimizer_mode == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD and latent_rank <= 0:
        raise ValueError(
            "global_params.latent_rank must be positive for optimizer_mode='exact_rank_manifold'."
        )
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(config.global_params.T),
        latent_rank=latent_rank,
        optimizer_mode=optimizer_mode,
    )


def save_field_artifacts(path: str | Path, artifacts: ModelArtifacts) -> None:
    payload: dict[str, np.ndarray] = {
        "latent_rank": np.asarray(int(artifacts.latent_rank), dtype=int),
        "t_steps": np.asarray(int(artifacts.t_steps), dtype=int),
    }
    if artifacts.field_matrix is not None:
        payload["field_matrix"] = np.asarray(artifacts.field_matrix, dtype=float)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(io_path(target), **payload)


def load_field_artifacts(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as data:
        result: dict[str, object] = {
            "latent_rank": int(data["latent_rank"]),
            "t_steps": int(data["t_steps"]),
        }
        for key in ["field_matrix"]:
            if key in data:
                result[key] = data[key]
    return result


def save_model_artifacts(data_folder: str | Path, artifacts: ModelArtifacts) -> None:
    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)
    if sparse.issparse(artifacts.gamma_matrix):
        sparse.save_npz(
            data_path / "gamma_matrix_sparse.npz",
            sparse.csr_matrix(artifacts.gamma_matrix),
        )
    else:
        np.save(
            data_path / "gamma_matrix.npy",
            np.asarray(artifacts.gamma_matrix, dtype=float),
        )
    save_field_artifacts(data_path / "field_artifacts.npz", artifacts)


def load_model_artifacts(data_folder: str | Path) -> ModelArtifacts:
    data_path = Path(data_folder)
    field_path = data_path / "field_artifacts.npz"
    if not field_path.exists():
        raise FileNotFoundError(f"Missing field_artifacts.npz in {data_path}.")
    gamma_sparse = data_path / "gamma_matrix_sparse.npz"
    gamma_dense = data_path / "gamma_matrix.npy"
    if gamma_sparse.exists():
        gamma_matrix = sparse.load_npz(gamma_sparse).tocsr()
    elif gamma_dense.exists():
        gamma_matrix = np.load(gamma_dense)
    else:
        raise FileNotFoundError(f"Missing gamma matrix artifact in {data_path}.")
    payload = load_field_artifacts(field_path)
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(payload["t_steps"]),
        latent_rank=int(payload.get("latent_rank", 0)),
        optimizer_mode=OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        field_matrix=payload.get("field_matrix"),
    )


def compose_interaction_matrix(xi: float, gamma_matrix):
    if sparse.issparse(gamma_matrix):
        interaction_matrix = (
            sparse.csr_matrix(gamma_matrix, dtype=float).multiply(xi).tocsr()
        )
        interaction_matrix = ((interaction_matrix + interaction_matrix.T) * 0.5).tocsr()
        interaction_matrix.setdiag(0.0)
        interaction_matrix.eliminate_zeros()
        return interaction_matrix
    interaction_matrix = xi * np.asarray(gamma_matrix, dtype=float)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
    np.fill_diagonal(interaction_matrix, 0.0)
    return interaction_matrix


def _apply_interaction_matrix(x: np.ndarray, interaction_matrix) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    if sparse.issparse(interaction_matrix):
        return np.asarray(x_array @ sparse.csr_matrix(interaction_matrix).T)
    return x_array @ np.asarray(interaction_matrix, dtype=float).T


def interaction_effect(x: np.ndarray, gamma_matrix) -> np.ndarray:
    """Apply the canonical Gamma interaction operator without the xi scalar."""
    return _apply_interaction_matrix(
        x,
        compose_interaction_matrix(1.0, gamma_matrix),
    )


def interaction_term(x: np.ndarray, xi: float, gamma_matrix) -> np.ndarray:
    """Apply the full xi-scaled interaction term used by predictive h(x)."""
    return _apply_interaction_matrix(
        x,
        compose_interaction_matrix(xi, gamma_matrix),
    )


def parameter_names(
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    n_nodes = artifacts.gamma_matrix.shape[0]
    if uses_full_matrix_parameterization(artifacts):
        field_keys = [
            f"F::time_{time_idx}::node_{node_idx}"
            for time_idx in range(artifacts.t_steps)
            for node_idx in range(n_nodes)
        ]
    else:
        field_keys = [
            f"U::node_{node_idx}::r_{rank_idx}"
            for node_idx in range(n_nodes)
            for rank_idx in range(artifacts.latent_rank)
        ]
        field_keys.extend(
            f"V::time_{time_idx}::r_{rank_idx}"
            for time_idx in range(artifacts.t_steps)
            for rank_idx in range(artifacts.latent_rank)
        )
    return field_keys + free_scalar_parameter_names(fixed_scalar_params)


def summarize_theta_for_logging(param_names: list[str], theta: np.ndarray) -> str:
    scalar_names = set(SCALAR_PARAMETER_ORDER)
    scalar_parts = [
        f"{key}: {value:+.4f}"
        for key, value in zip(param_names, theta)
        if key in scalar_names
    ]
    if scalar_parts:
        return "  " + ",  ".join(scalar_parts)
    return "  no free scalar parameters"


def unpack_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, object]:
    theta = np.asarray(theta, dtype=float)
    n_nodes = artifacts.gamma_matrix.shape[0]
    t_steps = artifacts.t_steps
    if uses_full_matrix_parameterization(artifacts):
        n_field = t_steps * n_nodes
        n_u = 0
        n_v = 0
    else:
        n_field = 0
        n_u = n_nodes * artifacts.latent_rank
        n_v = t_steps * artifacts.latent_rank
    tail = len(free_scalar_parameter_names(fixed_scalar_params))
    expected_length = n_field + n_u + n_v + tail
    if len(theta) != expected_length:
        raise ValueError(
            f"Theta length {len(theta)} does not match expected length {expected_length}."
        )
    if uses_full_matrix_parameterization(artifacts):
        field_matrix = theta[:n_field].reshape(t_steps, n_nodes)
        node_factors = np.zeros((n_nodes, 0), dtype=float)
        time_factors = np.zeros((t_steps, 0), dtype=float)
        cursor = n_field
    else:
        field_matrix = None
        node_factors = theta[:n_u].reshape(n_nodes, artifacts.latent_rank)
        time_factors = theta[n_u : n_u + n_v].reshape(t_steps, artifacts.latent_rank)
        cursor = n_u + n_v
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    scalar_values: dict[str, float] = {}
    for name in scalar_parameter_names():
        if name in fixed:
            scalar_values[name] = float(fixed[name])
        else:
            scalar_values[name] = float(theta[cursor])
            cursor += 1
    return {
        "node_factors": node_factors,
        "time_factors": time_factors,
        "field_matrix": field_matrix,
        "beta": scalar_values["beta"],
        "xi": scalar_values["xi"],
        "eta": scalar_values["eta"],
    }


def pack_theta(
    theta_parts: dict[str, object],
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if uses_full_matrix_parameterization(artifacts):
        if theta_parts.get("field_matrix") is None:
            raise ValueError("Full-matrix parameterization requires field_matrix.")
        field_block = np.asarray(theta_parts["field_matrix"], dtype=float).reshape(-1)
    else:
        if theta_parts["node_factors"] is None or theta_parts["time_factors"] is None:
            raise ValueError(
                "Factorized parameterization requires node_factors and time_factors."
            )
        field_block = np.concatenate(
            [
                np.asarray(theta_parts["node_factors"], dtype=float).reshape(-1),
                np.asarray(theta_parts["time_factors"], dtype=float).reshape(-1),
            ]
        )
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    tail = [
        np.array([float(theta_parts[name])], dtype=float)
        for name in scalar_parameter_names()
        if name not in fixed
    ]
    return np.concatenate([field_block, *tail])


def compose_field_matrix_from_theta(
    theta_parts: dict[str, object], artifacts: ModelArtifacts
) -> np.ndarray:
    if uses_full_matrix_parameterization(artifacts):
        return np.asarray(theta_parts["field_matrix"], dtype=float)
    return compose_latent_field_matrix(
        theta_parts["node_factors"], theta_parts["time_factors"]
    )


def with_theta_field(
    artifacts: ModelArtifacts, theta_parts: dict[str, object]
) -> ModelArtifacts:
    field_matrix = compose_field_matrix_from_theta(theta_parts, artifacts)
    return ModelArtifacts(
        gamma_matrix=artifacts.gamma_matrix,
        t_steps=artifacts.t_steps,
        latent_rank=artifacts.latent_rank,
        optimizer_mode=artifacts.optimizer_mode,
        field_matrix=field_matrix,
    )


def load_true_parameters(
    config,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if artifacts.field_matrix is None:
        raise ValueError("Missing latent truth field in field_artifacts.")
    truth_artifacts = ModelArtifacts(
        gamma_matrix=artifacts.gamma_matrix,
        t_steps=artifacts.t_steps,
        latent_rank=0,
        optimizer_mode=OPTIMIZER_MODE_NUCLEAR_NORM,
        field_matrix=artifacts.field_matrix,
    )
    theta_parts = {
        "field_matrix": artifacts.field_matrix,
        "beta": float(config.estimation_params.beta),
        "xi": float(get_xi(config)),
        "eta": float(config.estimation_params.eta),
    }
    return pack_theta(
        theta_parts,
        truth_artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
