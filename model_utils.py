"""Minimal shared helpers for the latent-only conditional MPLE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import make_low_rank_matrix


DEFAULT_LATENT_RANK = 0
_DEGENERACY_THRESHOLD = 1e-12   # norms below this are treated as zero/degenerate
_RMS_SCALE_FACTOR = 0.4         # targets initial field RMS at _RMS_SCALE_FACTOR * B
_TAIL_STRENGTH = 0.5            # tail_strength arg for sklearn make_low_rank_matrix
SCALAR_PARAMETER_ORDER = ("beta", "xi", "eta")
SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK = "random_low_rank"
SYNTHETIC_FIELD_MODE_NODE_BIAS_PLUS_SMOOTH_TIME_DRIFT = (
    "node_bias_plus_smooth_time_drift"
)
VALID_SYNTHETIC_FIELD_MODES = {
    SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK,
    SYNTHETIC_FIELD_MODE_NODE_BIAS_PLUS_SMOOTH_TIME_DRIFT,
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


def _smooth_time_trend(t_steps: int, sharpness: float = 2.0) -> np.ndarray:
    if t_steps <= 0:
        raise ValueError("t_steps must be positive.")
    if t_steps == 1:
        return np.zeros(1, dtype=float)
    grid = np.linspace(-1.0, 1.0, int(t_steps), dtype=float)
    trend = np.tanh(float(sharpness) * grid)
    trend = trend - float(np.mean(trend))
    rms = float(np.sqrt(np.mean(trend**2)))
    if rms < _DEGENERACY_THRESHOLD:
        return np.zeros_like(trend)
    return trend / rms


def truncate_matrix_rank(field_matrix: np.ndarray, rank: int) -> np.ndarray:
    """Project field_matrix onto its best rank-r approximation via truncated SVD."""
    field_matrix = np.asarray(field_matrix, dtype=float)
    max_rank = min(max(int(rank), 0), *field_matrix.shape)
    if max_rank == 0:
        return np.zeros_like(field_matrix)
    u, singular_values, vt = np.linalg.svd(field_matrix, full_matrices=False)
    return (u[:, :max_rank] * singular_values[:max_rank]) @ vt[:max_rank, :]


def scale_latent_field_matrix(
    field_matrix: np.ndarray, target_rms: float, bound: float
) -> np.ndarray:
    """Rescale field_matrix to have RMS ≈ target_rms, then clip to inf-norm ≤ bound.

    Returns the zero matrix if the initial RMS is below _DEGENERACY_THRESHOLD.
    """
    field_matrix = np.asarray(field_matrix, dtype=float)
    rms = float(np.sqrt(np.mean(field_matrix**2)))
    if rms < _DEGENERACY_THRESHOLD:
        return np.zeros_like(field_matrix)
    field_matrix = field_matrix * (target_rms / rms)
    norm = latent_field_bound_norm(field_matrix)
    if norm > bound and norm >= 1e-12:
        field_matrix = field_matrix * (bound / norm)
    return field_matrix


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


def _sample_random_low_rank_field(config, n_nodes: int, t_steps: int) -> np.ndarray:
    rank = get_latent_rank(config)
    if rank == 0:
        return zero_latent_field(n_nodes, t_steps)
    field_matrix = make_low_rank_matrix(
        n_samples=t_steps,
        n_features=n_nodes,
        effective_rank=rank,
        tail_strength=_TAIL_STRENGTH,
        random_state=int(config.generation_params.seed) + 101,
    )
    # Add bias to ensure nonzero mean and rescale to target RMS before truncation.
    rng = np.random.default_rng(int(config.generation_params.seed) + 202)
    bias = rng.normal(loc=0.0, scale=get_B(config), size=n_nodes)
    field_matrix = field_matrix + bias[None, :]
    field_matrix = np.asarray(field_matrix, dtype=float)
    target_rms = _RMS_SCALE_FACTOR * get_B(config)
    field_matrix = truncate_matrix_rank(field_matrix, rank)
    return scale_latent_field_matrix(field_matrix, target_rms, get_B(config))


def _sample_node_bias_plus_smooth_time_drift_field(
    config,
    n_nodes: int,
    t_steps: int,
) -> np.ndarray:
    field_params = get_synthetic_field_params(config)
    node_bias_scale = float(field_params.get("node_bias_scale", 1.0))
    drift_scale = float(field_params.get("drift_scale", 0.5))
    time_trend_sharpness = float(field_params.get("time_trend_sharpness", 2.0))
    rng = np.random.default_rng(int(config.generation_params.seed) + 211)

    node_bias = rng.normal(size=n_nodes)
    drift_loading = rng.normal(size=n_nodes)
    node_bias_norm = float(np.linalg.norm(node_bias))
    if node_bias_norm < _DEGENERACY_THRESHOLD:
        node_bias = np.ones(n_nodes, dtype=float)
        node_bias_norm = float(np.linalg.norm(node_bias))
    node_bias = node_bias / node_bias_norm

    drift_norm = float(np.linalg.norm(drift_loading))
    if drift_norm < _DEGENERACY_THRESHOLD:
        drift_loading = np.zeros(n_nodes, dtype=float)
        drift_loading[0] = 1.0
        drift_norm = 1.0
    if node_bias_norm >= _DEGENERACY_THRESHOLD:
        projection = float(np.dot(drift_loading, node_bias))
        drift_loading = drift_loading - projection * node_bias
        drift_norm = float(np.linalg.norm(drift_loading))
        if drift_norm < _DEGENERACY_THRESHOLD:
            drift_loading = rng.normal(size=n_nodes)
            projection = float(np.dot(drift_loading, node_bias))
            drift_loading = drift_loading - projection * node_bias
            drift_norm = float(np.linalg.norm(drift_loading))
    if drift_norm < _DEGENERACY_THRESHOLD:
        drift_loading = np.zeros(n_nodes, dtype=float)
        drift_loading[min(1, n_nodes - 1)] = 1.0
        drift_norm = float(np.linalg.norm(drift_loading))
    drift_loading = drift_loading / drift_norm

    trend = _smooth_time_trend(t_steps, sharpness=time_trend_sharpness)
    field_matrix = (
        node_bias_scale * node_bias[None, :]
        + drift_scale * trend[:, None] * drift_loading[None, :]
    )
    target_rms = _RMS_SCALE_FACTOR * get_B(config)
    return scale_latent_field_matrix(field_matrix, target_rms, get_B(config))


def build_synthetic_field(config, gamma_matrix) -> ModelArtifacts:
    n_nodes = int(config.global_params.N)
    t_steps = int(config.global_params.T)
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    field_mode = get_synthetic_field_mode(config)
    if field_mode == SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK:
        field_matrix = _sample_random_low_rank_field(config, n_nodes, t_steps)
        latent_rank = get_latent_rank(config)
    elif field_mode == SYNTHETIC_FIELD_MODE_NODE_BIAS_PLUS_SMOOTH_TIME_DRIFT:
        field_matrix = _sample_node_bias_plus_smooth_time_drift_field(
            config, n_nodes, t_steps
        )
        latent_rank = min(2, n_nodes, t_steps)
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
    np.savez(Path(path), **payload)


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


def interaction_effect(x: np.ndarray, gamma_matrix) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sparse.issparse(gamma_matrix):
        return np.asarray(x @ sparse.csr_matrix(gamma_matrix).T)
    return x @ np.asarray(gamma_matrix, dtype=float).T


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
