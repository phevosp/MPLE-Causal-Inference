"""Minimal shared helpers for the latent-only conditional MPLE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


DEFAULT_LATENT_RANK = 0
SCALAR_PARAMETER_ORDER = ("beta", "xi", "eta", "zeta", "psi")


@dataclass(frozen=True)
class ModelArtifacts:
    """Latent-field artifacts for one experiment."""

    gamma_matrix: object
    t_steps: int
    latent_rank: int = 0
    field_matrix: np.ndarray | None = None
    node_factors: np.ndarray | None = None
    time_factors: np.ndarray | None = None


def intervention_model_enabled(config) -> bool:
    estimation_params = getattr(config, "estimation_params", None)
    if estimation_params is None or "fit_intervention_model" not in estimation_params:
        return True
    return bool(estimation_params.fit_intervention_model)


def scalar_parameter_names(fit_intervention_model: bool = True) -> list[str]:
    if fit_intervention_model:
        return list(SCALAR_PARAMETER_ORDER)
    return list(SCALAR_PARAMETER_ORDER[:3])


def validate_fixed_scalar_params(
    fixed_scalar_params: dict[str, float] | None,
    fit_intervention_model: bool = True,
) -> dict[str, float]:
    fixed_scalar_params = {
        str(key): float(value) for key, value in (fixed_scalar_params or {}).items()
    }
    invalid = sorted(set(fixed_scalar_params) - set(SCALAR_PARAMETER_ORDER))
    if invalid:
        raise ValueError(
            f"Unknown fixed scalar parameter(s): {', '.join(invalid)}."
        )
    if not fit_intervention_model:
        blocked = [name for name in ["zeta", "psi"] if name in fixed_scalar_params]
        if blocked:
            raise ValueError(
                "fixed_scalar_params cannot include zeta or psi when fit_intervention_model=false."
            )
    return fixed_scalar_params


def free_scalar_parameter_names(
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    fixed = validate_fixed_scalar_params(
        fixed_scalar_params, fit_intervention_model=fit_intervention_model
    )
    return [
        name
        for name in scalar_parameter_names(fit_intervention_model)
        if name not in fixed
    ]


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


def get_B(config) -> float:
    if "B" not in config.global_params:
        raise KeyError("global_params.B is required.")
    return float(config.global_params.B)


def get_xi(config) -> float:
    if "xi" not in config.estimation_params:
        raise KeyError("estimation_params.xi is required.")
    return float(config.estimation_params.xi)


def _normalize_dense_graph(gamma_matrix: np.ndarray) -> np.ndarray:
    gamma_matrix = np.asarray(gamma_matrix, dtype=float)
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2.0
    np.fill_diagonal(gamma_matrix, 0.0)
    norm = float(np.linalg.norm(gamma_matrix, ord=np.inf))
    if norm < 1e-12:
        return np.zeros_like(gamma_matrix)
    return gamma_matrix / norm


def _normalize_sparse_graph(gamma_matrix) -> sparse.csr_matrix:
    normalized = sparse.csr_matrix(gamma_matrix, dtype=float)
    normalized = ((normalized + normalized.T) * 0.5).tocsr()
    normalized.setdiag(0.0)
    normalized.eliminate_zeros()
    norm = interaction_matrix_infinity_norm(normalized)
    if norm < 1e-12:
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
    return (
        np.asarray(time_factors, dtype=float) @ np.asarray(node_factors, dtype=float).T
    )


def latent_field_bound_norm(field_matrix: np.ndarray) -> float:
    """Entrywise infinity norm used for the latent-field B constraint."""
    field_matrix = np.asarray(field_matrix, dtype=float)
    if field_matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(field_matrix)))


def zero_latent_factors(
    n_nodes: int, t_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((n_nodes, 0), dtype=float),
        np.zeros((t_steps, 0), dtype=float),
    )


def project_latent_field(
    node_factors: np.ndarray, time_factors: np.ndarray, bound: float
) -> tuple[np.ndarray, np.ndarray]:
    node_factors = np.asarray(node_factors, dtype=float).copy()
    time_factors = np.asarray(time_factors, dtype=float).copy()
    field_matrix = compose_latent_field_matrix(node_factors, time_factors)
    norm = latent_field_bound_norm(field_matrix)
    if norm <= bound or norm < 1e-12:
        return node_factors, time_factors
    scale = np.sqrt(bound / norm)
    return node_factors * scale, time_factors * scale


def _sample_latent_factors(
    config, n_nodes: int, t_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    rank = get_latent_rank(config)
    if rank == 0:
        return zero_latent_factors(n_nodes, t_steps)
    factor_bound = float(np.sqrt(get_B(config)))
    rng = np.random.default_rng(int(config.generation_params.seed) + 101)
    node_factors = rng.normal(size=(n_nodes, rank))
    time_factors = rng.normal(size=(t_steps, rank))
    node_norm = float(np.linalg.norm(node_factors, ord=np.inf))
    time_norm = float(np.linalg.norm(time_factors, ord=np.inf))
    if node_norm > factor_bound and node_norm > 1e-12:
        node_factors *= factor_bound / node_norm
    if time_norm > factor_bound and time_norm > 1e-12:
        time_factors *= factor_bound / time_norm
    return project_latent_field(node_factors, time_factors, get_B(config))


def build_synthetic_field(config, gamma_matrix) -> ModelArtifacts:
    n_nodes = int(config.global_params.N)
    t_steps = int(config.global_params.T)
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    node_factors, time_factors = _sample_latent_factors(config, n_nodes, t_steps)
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=t_steps,
        latent_rank=get_latent_rank(config),
        field_matrix=compose_latent_field_matrix(node_factors, time_factors),
        node_factors=node_factors,
        time_factors=time_factors,
    )


def build_fit_model_artifacts(config, gamma_matrix) -> ModelArtifacts:
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(config.global_params.T),
        latent_rank=get_latent_rank(config),
    )


def save_field_artifacts(path: str | Path, artifacts: ModelArtifacts) -> None:
    payload: dict[str, np.ndarray] = {
        "latent_rank": np.asarray(int(artifacts.latent_rank), dtype=int),
        "t_steps": np.asarray(int(artifacts.t_steps), dtype=int),
    }
    if artifacts.field_matrix is not None:
        payload["field_matrix"] = np.asarray(artifacts.field_matrix, dtype=float)
    if artifacts.node_factors is not None:
        payload["node_factors"] = np.asarray(artifacts.node_factors, dtype=float)
    if artifacts.time_factors is not None:
        payload["time_factors"] = np.asarray(artifacts.time_factors, dtype=float)
    np.savez(Path(path), **payload)


def load_field_artifacts(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as data:
        result: dict[str, object] = {
            "latent_rank": int(data["latent_rank"]),
            "t_steps": int(data["t_steps"]),
        }
        for key in ["field_matrix", "node_factors", "time_factors"]:
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
        field_matrix=payload.get("field_matrix"),
        node_factors=payload.get("node_factors"),
        time_factors=payload.get("time_factors"),
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
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    n_nodes = artifacts.gamma_matrix.shape[0]
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
    return field_keys + free_scalar_parameter_names(
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )


def summarize_theta_for_logging(param_names: list[str], theta: np.ndarray) -> str:
    scalar_names = {"beta", "xi", "eta", "zeta", "psi"}
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
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, object]:
    theta = np.asarray(theta, dtype=float)
    n_nodes = artifacts.gamma_matrix.shape[0]
    t_steps = artifacts.t_steps
    n_u = n_nodes * artifacts.latent_rank
    n_v = t_steps * artifacts.latent_rank
    tail = len(
        free_scalar_parameter_names(
            fit_intervention_model=fit_intervention_model,
            fixed_scalar_params=fixed_scalar_params,
        )
    )
    expected_length = n_u + n_v + tail
    if len(theta) != expected_length:
        raise ValueError(
            f"Theta length {len(theta)} does not match expected length {expected_length}."
        )
    node_factors = theta[:n_u].reshape(n_nodes, artifacts.latent_rank)
    time_factors = theta[n_u : n_u + n_v].reshape(t_steps, artifacts.latent_rank)
    cursor = n_u + n_v
    fixed = validate_fixed_scalar_params(
        fixed_scalar_params, fit_intervention_model=fit_intervention_model
    )
    scalar_values: dict[str, float] = {}
    for name in scalar_parameter_names(fit_intervention_model):
        if name in fixed:
            scalar_values[name] = float(fixed[name])
        else:
            scalar_values[name] = float(theta[cursor])
            cursor += 1
    return {
        "node_factors": node_factors,
        "time_factors": time_factors,
        "beta": scalar_values["beta"],
        "xi": scalar_values["xi"],
        "eta": scalar_values["eta"],
        "zeta": float(scalar_values.get("zeta", 0.0)),
        "psi": float(scalar_values.get("psi", 0.0)),
    }


def pack_theta(
    theta_parts: dict[str, object],
    artifacts: ModelArtifacts,
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if theta_parts["node_factors"] is None or theta_parts["time_factors"] is None:
        raise ValueError("Latent mode requires node_factors and time_factors.")
    field_block = np.concatenate(
        [
            np.asarray(theta_parts["node_factors"], dtype=float).reshape(-1),
            np.asarray(theta_parts["time_factors"], dtype=float).reshape(-1),
        ]
    )
    fixed = validate_fixed_scalar_params(
        fixed_scalar_params, fit_intervention_model=fit_intervention_model
    )
    tail = [
        np.array([float(theta_parts[name])], dtype=float)
        for name in scalar_parameter_names(fit_intervention_model)
        if name not in fixed
    ]
    return np.concatenate([field_block, *tail])


def compose_field_matrix_from_theta(
    theta_parts: dict[str, object], artifacts: ModelArtifacts
) -> np.ndarray:
    return compose_latent_field_matrix(
        theta_parts["node_factors"], theta_parts["time_factors"]
    )


def with_theta_field(
    artifacts: ModelArtifacts, theta_parts: dict[str, object]
) -> ModelArtifacts:
    return ModelArtifacts(
        gamma_matrix=artifacts.gamma_matrix,
        t_steps=artifacts.t_steps,
        latent_rank=artifacts.latent_rank,
        field_matrix=compose_latent_field_matrix(
            theta_parts["node_factors"], theta_parts["time_factors"]
        ),
        node_factors=np.asarray(theta_parts["node_factors"], dtype=float),
        time_factors=np.asarray(theta_parts["time_factors"], dtype=float),
    )


def load_true_parameters(
    config,
    artifacts: ModelArtifacts,
    fit_intervention_model: bool | None = None,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if fit_intervention_model is None:
        fit_intervention_model = intervention_model_enabled(config)
    if artifacts.node_factors is None or artifacts.time_factors is None:
        raise ValueError("Missing latent truth in field_artifacts.")
    theta_parts = {
        "node_factors": artifacts.node_factors,
        "time_factors": artifacts.time_factors,
        "beta": float(config.estimation_params.beta),
        "xi": float(get_xi(config)),
        "eta": float(config.estimation_params.eta),
        "zeta": float(getattr(config.estimation_params, "zeta", 0.0)),
        "psi": float(getattr(config.estimation_params, "psi", 0.0)),
    }
    return pack_theta(
        theta_parts,
        artifacts,
        fit_intervention_model=fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )


def summary_metrics(
    est_theta: np.ndarray,
    true_theta: np.ndarray,
    artifacts: ModelArtifacts,
    fit_intervention_model: bool = True,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, float]:
    est_parts = unpack_theta(
        est_theta,
        artifacts,
        fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    true_parts = unpack_theta(
        true_theta,
        artifacts,
        fit_intervention_model,
        fixed_scalar_params=fixed_scalar_params,
    )
    est_artifacts = with_theta_field(artifacts, est_parts)
    true_artifacts = with_theta_field(artifacts, true_parts)
    est_interaction = compose_interaction_matrix(
        est_parts["xi"], artifacts.gamma_matrix
    )
    true_interaction = compose_interaction_matrix(
        true_parts["xi"], artifacts.gamma_matrix
    )
    if sparse.issparse(est_interaction):
        interaction_error = est_interaction - true_interaction
        interaction_fro_error = float(
            np.sqrt(interaction_error.multiply(interaction_error).sum())
        )
    else:
        interaction_fro_error = float(
            np.linalg.norm(est_interaction - true_interaction, ord="fro")
        )

    return {
        "field_rmse": float(
            np.sqrt(
                np.mean((est_artifacts.field_matrix - true_artifacts.field_matrix) ** 2)
            )
        ),
        "field_l2_error": float(
            np.linalg.norm(
                (est_artifacts.field_matrix - true_artifacts.field_matrix).reshape(-1),
                ord=2,
            )
        ),
        "interaction_fro_error": interaction_fro_error,
        "parameter_rmse": float(np.sqrt(np.mean((est_theta - true_theta) ** 2))),
    }
