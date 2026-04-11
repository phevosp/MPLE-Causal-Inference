"""Minimal shared helpers for the active conditional MPLE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


DEFAULT_NUM_SHARED_FEATURES = 5
DEFAULT_FIELD_MODE = "uniform"
DEFAULT_LATENT_RANK = 10
LATENT_FIELD_MODE = "latent_feature_matrix"


@dataclass(frozen=True)
class ModelArtifacts:
    """Mode-aware model artifacts for one experiment."""

    field_mode: str
    gamma_matrix: object
    field_basis: np.ndarray | None = None
    field_names: tuple[str, ...] = ()
    shared_features: np.ndarray | None = None
    latent_rank: int = 0
    field_matrix: np.ndarray | None = None
    field_coeffs: np.ndarray | None = None
    tau: np.ndarray | None = None
    field_vector: np.ndarray | None = None
    node_factors: np.ndarray | None = None
    time_factors: np.ndarray | None = None


def intervention_model_enabled(config) -> bool:
    """Return whether the MPLE fit should include the intervention process."""
    estimation_params = getattr(config, "estimation_params", None)
    if estimation_params is None or "fit_intervention_model" not in estimation_params:
        return True
    return bool(estimation_params.fit_intervention_model)


def scalar_parameter_count(fit_intervention_model: bool) -> int:
    """Return the number of scalar tail parameters in theta."""
    return 5 if fit_intervention_model else 3


def interaction_matrix_infinity_norm(matrix) -> float:
    """Compute the matrix infinity norm for dense or sparse matrices."""
    if sparse.issparse(matrix):
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        return float(row_sums.max()) if row_sums.size else 0.0
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def _basis_setting(config, key: str, default):
    basis_params = getattr(config.global_params, "basis_params", None)
    if basis_params is None or key not in basis_params:
        return default
    return basis_params[key]


def get_field_mode(config) -> str:
    """Return the configured field mode."""
    return str(_basis_setting(config, "field_mode", DEFAULT_FIELD_MODE))


def get_latent_rank(config) -> int:
    """Return the configured latent rank."""
    return int(_basis_setting(config, "latent_rank", DEFAULT_LATENT_RANK))


def get_B(config) -> float:
    """Return the configured upper bound B."""
    if "B" not in config.global_params:
        raise KeyError("global_params.B is required.")
    return float(config.global_params.B)


def get_xi(config) -> float:
    """Load the scalar graph-temperature parameter xi from config."""
    if "xi" not in config.estimation_params:
        raise KeyError("estimation_params.xi is required.")
    return float(config.estimation_params.xi)


def _center_and_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    values = values - values.mean()
    norm = float(np.linalg.norm(values, ord=np.inf))
    if norm < 1e-12:
        return np.zeros_like(values)
    return values / norm


def _centered_quadratic(feature: np.ndarray) -> np.ndarray:
    squared = np.asarray(feature, dtype=float) ** 2
    return squared - squared.mean()


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
    """Return the fixed known graph normalized to infinity norm one."""
    if sparse.issparse(gamma_matrix):
        return _normalize_sparse_graph(gamma_matrix)
    return _normalize_dense_graph(np.asarray(gamma_matrix, dtype=float))


def validate_basis_infinity_norms(
    field_basis: np.ndarray | None,
    gamma_matrix,
    tol: float = 1e-8,
) -> None:
    """Ensure the additive field basis rows and known graph are normalized."""
    if field_basis is not None:
        field_basis = np.asarray(field_basis, dtype=float)
        if field_basis.ndim != 2:
            raise ValueError("field_basis must be a 2D array.")
        if field_basis.shape[0] > 0:
            field_norms = np.linalg.norm(field_basis, ord=np.inf, axis=1)
            if np.any(field_norms < tol):
                raise ValueError("Field basis contains a degenerate zero vector.")
            if not np.allclose(field_norms, 1.0, atol=tol, rtol=0.0):
                raise ValueError("Each field basis vector must have infinity norm one.")

    gamma_norm = interaction_matrix_infinity_norm(gamma_matrix)
    if gamma_norm < tol:
        raise ValueError("Known graph is degenerate.")
    if not np.isclose(gamma_norm, 1.0, atol=tol, rtol=0.0):
        raise ValueError("The known graph must have infinity norm one.")


def _shared_features(config, n_nodes: int) -> np.ndarray:
    count = int(
        _basis_setting(config, "num_shared_features", DEFAULT_NUM_SHARED_FEATURES)
    )
    seed = int(
        _basis_setting(config, "shared_feature_seed", config.generation_params.seed)
    )
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, n_nodes))
    return np.vstack([_center_and_normalize(feature) for feature in raw])


def _additive_basis(
    config, field_mode: str, n_nodes: int
) -> tuple[np.ndarray | None, tuple[str, ...], np.ndarray | None]:
    if field_mode == "uniform":
        return np.empty((0, n_nodes), dtype=float), (), None
    if field_mode != "shared_feature_field":
        raise ValueError(f"Unknown field_mode '{field_mode}'.")

    shared_features = _shared_features(config, n_nodes)
    basis_rows: list[np.ndarray] = []
    names: list[str] = []
    for feature_index, feature in enumerate(shared_features, start=1):
        linear = _center_and_normalize(feature)
        quadratic = _center_and_normalize(_centered_quadratic(feature))
        if np.linalg.norm(linear, ord=np.inf) >= 1e-12:
            basis_rows.append(linear)
            names.append(f"linear::feature_{feature_index}")
        if np.linalg.norm(quadratic, ord=np.inf) >= 1e-12:
            basis_rows.append(quadratic)
            names.append(f"quadratic::feature_{feature_index}")
    field_basis = (
        np.vstack(basis_rows).astype(float)
        if basis_rows
        else np.empty((0, n_nodes), dtype=float)
    )
    return field_basis, tuple(names), shared_features


def compose_field(field_coeffs: np.ndarray, field_basis: np.ndarray) -> np.ndarray:
    """Map additive field coefficients to the realized node-wise field."""
    return np.asarray(field_coeffs, dtype=float) @ np.asarray(field_basis, dtype=float)


def compose_additive_field_matrix(
    field_coeffs: np.ndarray, tau: np.ndarray, field_basis: np.ndarray
) -> np.ndarray:
    """Compose the additive T x N field."""
    return (
        compose_field(field_coeffs, field_basis)[None, :]
        + np.asarray(tau, dtype=float)[:, None]
    )


def compose_latent_field_matrix(
    node_factors: np.ndarray, time_factors: np.ndarray
) -> np.ndarray:
    """Compose the latent T x N field."""
    return (
        np.asarray(time_factors, dtype=float) @ np.asarray(node_factors, dtype=float).T
    )


def project_latent_field(
    node_factors: np.ndarray, time_factors: np.ndarray, bound: float
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly rescale latent factors so the realized field stays within the matrix-infinity ball."""
    node_factors = np.asarray(node_factors, dtype=float).copy()
    time_factors = np.asarray(time_factors, dtype=float).copy()
    field_matrix = compose_latent_field_matrix(node_factors, time_factors)
    norm = float(np.linalg.norm(field_matrix, ord=np.inf))
    if norm <= bound or norm < 1e-12:
        return node_factors, time_factors
    scale = np.sqrt(bound / norm)
    return node_factors * scale, time_factors * scale


def _sample_tau(config, t_steps: int) -> np.ndarray:
    estimation_params = config.estimation_params
    if "tau_params" not in estimation_params or estimation_params.tau_params is None:
        return np.zeros(t_steps, dtype=float)
    tau_params = estimation_params.tau_params
    mode = str(tau_params.mode)
    if mode == "fixed":
        tau = np.asarray(tau_params["vector"], dtype=float)
        if tau.shape != (t_steps,):
            raise ValueError(
                "Fixed tau vector must have length equal to global_params.T."
            )
        return tau
    if mode == "uniform_random":
        lower = float(tau_params.lower)
        upper = float(tau_params.upper)
        if lower > upper:
            raise ValueError(
                "tau_params.lower must be less than or equal to tau_params.upper."
            )
        seed = int(getattr(tau_params, "seed", config.generation_params.seed))
        return np.random.default_rng(seed).uniform(lower, upper, size=t_steps)
    raise ValueError(f"Unknown tau generation mode '{mode}'.")


def _sample_latent_factors(
    config, n_nodes: int, t_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    rank = get_latent_rank(config)
    if rank <= 0:
        raise ValueError(
            "basis_params.latent_rank must be positive in latent field mode."
        )
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
    """Build the normalized known graph and realized synthetic field artifacts."""
    field_mode = get_field_mode(config)
    n_nodes = int(config.global_params.N)
    t_steps = int(config.global_params.T)
    gamma_matrix = normalize_known_graph(gamma_matrix)

    if field_mode == LATENT_FIELD_MODE:
        validate_basis_infinity_norms(None, gamma_matrix)
        node_factors, time_factors = _sample_latent_factors(config, n_nodes, t_steps)
        return ModelArtifacts(
            field_mode=field_mode,
            gamma_matrix=gamma_matrix,
            latent_rank=get_latent_rank(config),
            field_matrix=compose_latent_field_matrix(node_factors, time_factors),
            node_factors=node_factors,
            time_factors=time_factors,
        )

    field_basis, field_names, shared_features = _additive_basis(
        config, field_mode, n_nodes
    )
    validate_basis_infinity_norms(field_basis, gamma_matrix)
    field_coeffs = np.asarray(config.estimation_params.field_coefs, dtype=float)
    if len(field_coeffs) != field_basis.shape[0]:
        raise ValueError(
            "Field coefficient count does not match the configured field basis."
        )
    tau = _sample_tau(config, t_steps)
    field_vector = compose_field(field_coeffs, field_basis)
    return ModelArtifacts(
        field_mode=field_mode,
        gamma_matrix=gamma_matrix,
        field_basis=field_basis,
        field_names=field_names,
        shared_features=shared_features,
        field_matrix=compose_additive_field_matrix(field_coeffs, tau, field_basis),
        field_coeffs=field_coeffs,
        tau=tau,
        field_vector=field_vector,
    )


def save_field_artifacts(path: str | Path, artifacts: ModelArtifacts) -> None:
    """Save one field artifact bundle."""
    payload: dict[str, np.ndarray] = {
        "field_mode": np.asarray(artifacts.field_mode),
        "latent_rank": np.asarray(int(artifacts.latent_rank), dtype=int),
    }
    if artifacts.field_basis is not None:
        payload["field_basis"] = np.asarray(artifacts.field_basis, dtype=float)
        payload["field_names"] = np.asarray(artifacts.field_names, dtype="<U128")
    if artifacts.shared_features is not None:
        payload["shared_features"] = np.asarray(artifacts.shared_features, dtype=float)
    if artifacts.field_matrix is not None:
        payload["field_matrix"] = np.asarray(artifacts.field_matrix, dtype=float)
    if artifacts.field_coeffs is not None:
        payload["field_coeffs"] = np.asarray(artifacts.field_coeffs, dtype=float)
    if artifacts.tau is not None:
        payload["tau"] = np.asarray(artifacts.tau, dtype=float)
    if artifacts.field_vector is not None:
        payload["field_vector"] = np.asarray(artifacts.field_vector, dtype=float)
    if artifacts.node_factors is not None:
        payload["node_factors"] = np.asarray(artifacts.node_factors, dtype=float)
    if artifacts.time_factors is not None:
        payload["time_factors"] = np.asarray(artifacts.time_factors, dtype=float)
    np.savez(Path(path), **payload)


def load_field_artifacts(path: str | Path) -> dict[str, object]:
    """Load one field artifact bundle into a plain dictionary."""
    with np.load(Path(path), allow_pickle=False) as data:
        result: dict[str, object] = {"field_mode": str(data["field_mode"].item())}
        if "latent_rank" in data:
            result["latent_rank"] = int(data["latent_rank"])
        for key in [
            "field_basis",
            "shared_features",
            "field_matrix",
            "field_coeffs",
            "tau",
            "field_vector",
            "node_factors",
            "time_factors",
        ]:
            if key in data:
                result[key] = data[key]
        if "field_names" in data:
            result["field_names"] = tuple(data["field_names"].tolist())
        else:
            result["field_names"] = ()
    return result


def save_model_artifacts(data_folder: str | Path, artifacts: ModelArtifacts) -> None:
    """Save graph and field artifacts for one experiment."""
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
    """Load graph and field artifacts for one experiment."""
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
        field_mode=str(payload["field_mode"]),
        gamma_matrix=gamma_matrix,
        field_basis=payload.get("field_basis"),
        field_names=tuple(payload.get("field_names", ())),
        shared_features=payload.get("shared_features"),
        latent_rank=int(payload.get("latent_rank", 0)),
        field_matrix=payload.get("field_matrix"),
        field_coeffs=payload.get("field_coeffs"),
        tau=payload.get("tau"),
        field_vector=payload.get("field_vector"),
        node_factors=payload.get("node_factors"),
        time_factors=payload.get("time_factors"),
    )


def compose_interaction_matrix(xi: float, gamma_matrix):
    """Scale the fixed known graph by the scalar temperature xi."""
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
    """Precompute the single known-graph interaction effect."""
    x = np.asarray(x, dtype=float)
    if sparse.issparse(gamma_matrix):
        return np.asarray(x @ sparse.csr_matrix(gamma_matrix).T)
    return x @ np.asarray(gamma_matrix, dtype=float).T


def parameter_names(
    artifacts: ModelArtifacts, t_steps: int, fit_intervention_model: bool = True
) -> list[str]:
    """Create human-readable parameter labels matching the flattened theta vector."""
    if artifacts.field_mode == LATENT_FIELD_MODE:
        n_nodes = artifacts.gamma_matrix.shape[0]
        field_keys = [
            f"U::node_{node_idx}::r_{rank_idx}"
            for node_idx in range(n_nodes)
            for rank_idx in range(artifacts.latent_rank)
        ]
        field_keys.extend(
            f"V::time_{time_idx}::r_{rank_idx}"
            for time_idx in range(t_steps)
            for rank_idx in range(artifacts.latent_rank)
        )
    else:
        field_keys = [f"field::{name}" for name in artifacts.field_names]
        field_keys.extend(f"tau::t_{idx}" for idx in range(t_steps))
    tail_keys = ["eta", "zeta", "psi"] if fit_intervention_model else ["eta"]
    return field_keys + ["beta", "xi"] + tail_keys


def summarize_theta_for_logging(param_names: list[str], theta: np.ndarray) -> str:
    """Format theta for compact optimizer logging without dumping high-dimensional fields."""
    scalar_names = {"beta", "xi", "eta", "zeta", "psi"}
    scalar_parts = [
        f"{key}: {value:+.4f}"
        for key, value in zip(param_names, theta)
        if key in scalar_names
    ]
    if scalar_parts:
        return "  " + ",  ".join(scalar_parts)
    tau_values = np.asarray(
        [value for name, value in zip(param_names, theta) if name.startswith("tau::")],
        dtype=float,
    )
    if tau_values.size == 0:
        return "  " + ",  ".join(
            f"{key}: {value:+.4f}" for key, value in zip(param_names, theta)
        )
    field_count = sum(name.startswith("field::") for name in param_names)
    non_tau_parts = [
        f"{key}: {value:+.4f}"
        for key, value in zip(param_names, theta)
        if not key.startswith("tau::")
    ]
    non_tau_parts.insert(
        field_count,
        (
            f"tau block: mean={tau_values.mean():+.4f}, std={tau_values.std():.4f}, "
            f"min={tau_values.min():+.4f}, max={tau_values.max():+.4f}"
        ),
    )
    return "  " + ",  ".join(non_tau_parts)


def infer_t_steps_from_theta(
    theta: np.ndarray, artifacts: ModelArtifacts, fit_intervention_model: bool = True
) -> int:
    """Infer T from a flattened theta vector and model structure."""
    theta_length = len(np.asarray(theta, dtype=float))
    tail = scalar_parameter_count(fit_intervention_model)
    if artifacts.field_mode == LATENT_FIELD_MODE:
        n_nodes = artifacts.gamma_matrix.shape[0]
        numerator = theta_length - n_nodes * artifacts.latent_rank - tail
        if numerator < 0 or numerator % artifacts.latent_rank != 0:
            raise ValueError(
                "Latent theta length is incompatible with the configured rank."
            )
        return numerator // artifacts.latent_rank
    if artifacts.field_basis is None:
        raise ValueError("Additive mode requires field_basis.")
    t_steps = theta_length - artifacts.field_basis.shape[0] - tail
    if t_steps < 0:
        raise ValueError("Theta length is too short for the additive field block.")
    return t_steps


def unpack_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    t_steps: int,
    fit_intervention_model: bool = True,
) -> dict[str, object]:
    """Split theta into field, scalar, and intervention-process blocks."""
    theta = np.asarray(theta, dtype=float)
    if artifacts.field_mode == LATENT_FIELD_MODE:
        n_nodes = artifacts.gamma_matrix.shape[0]
        n_u = n_nodes * artifacts.latent_rank
        n_v = t_steps * artifacts.latent_rank
        node_factors = theta[:n_u].reshape(n_nodes, artifacts.latent_rank)
        time_factors = theta[n_u : n_u + n_v].reshape(t_steps, artifacts.latent_rank)
        cursor = n_u + n_v
        field_coeffs = None
        tau = None
    else:
        if artifacts.field_basis is None:
            raise ValueError("Additive mode requires field_basis.")
        n_field = artifacts.field_basis.shape[0]
        field_coeffs = theta[:n_field]
        tau = theta[n_field : n_field + t_steps]
        cursor = n_field + t_steps
        node_factors = None
        time_factors = None
    beta = float(theta[cursor])
    xi = float(theta[cursor + 1])
    tail = np.asarray(theta[cursor + 2 :], dtype=float)
    if fit_intervention_model:
        eta, zeta, psi = tail
    else:
        eta = float(tail[0])
        zeta = 0.0
        psi = 0.0
    return {
        "field_coeffs": (
            None if field_coeffs is None else np.asarray(field_coeffs, dtype=float)
        ),
        "tau": None if tau is None else np.asarray(tau, dtype=float),
        "node_factors": node_factors,
        "time_factors": time_factors,
        "beta": beta,
        "xi": xi,
        "eta": float(eta),
        "zeta": float(zeta),
        "psi": float(psi),
    }


def pack_theta(
    theta_parts: dict[str, object],
    artifacts: ModelArtifacts,
    fit_intervention_model: bool = True,
) -> np.ndarray:
    """Pack structured theta parts into the optimizer ordering."""
    if artifacts.field_mode == LATENT_FIELD_MODE:
        if theta_parts["node_factors"] is None or theta_parts["time_factors"] is None:
            raise ValueError("Latent mode requires node_factors and time_factors.")
        field_block = np.concatenate(
            [
                np.asarray(theta_parts["node_factors"], dtype=float).reshape(-1),
                np.asarray(theta_parts["time_factors"], dtype=float).reshape(-1),
            ]
        )
    else:
        if theta_parts["field_coeffs"] is None or theta_parts["tau"] is None:
            raise ValueError("Additive mode requires field_coeffs and tau.")
        field_block = np.concatenate(
            [
                np.asarray(theta_parts["field_coeffs"], dtype=float),
                np.asarray(theta_parts["tau"], dtype=float),
            ]
        )
    tail = [
        np.array([float(theta_parts["beta"])], dtype=float),
        np.array([float(theta_parts["xi"])], dtype=float),
        np.array([float(theta_parts["eta"])], dtype=float),
    ]
    if fit_intervention_model:
        tail.append(
            np.array(
                [float(theta_parts["zeta"]), float(theta_parts["psi"])], dtype=float
            )
        )
    return np.concatenate([field_block, *tail])


def compose_field_matrix_from_theta(
    theta_parts: dict[str, object], artifacts: ModelArtifacts
) -> np.ndarray:
    """Compose the realized field matrix from unpacked theta parts."""
    if artifacts.field_mode == LATENT_FIELD_MODE:
        return compose_latent_field_matrix(
            theta_parts["node_factors"], theta_parts["time_factors"]
        )
    if artifacts.field_basis is None:
        raise ValueError("Additive mode requires field_basis.")
    return compose_additive_field_matrix(
        theta_parts["field_coeffs"], theta_parts["tau"], artifacts.field_basis
    )


def with_theta_field(
    artifacts: ModelArtifacts, theta_parts: dict[str, object]
) -> ModelArtifacts:
    """Return one artifact view with field quantities reconstructed from theta."""
    if artifacts.field_mode == LATENT_FIELD_MODE:
        return ModelArtifacts(
            field_mode=artifacts.field_mode,
            gamma_matrix=artifacts.gamma_matrix,
            latent_rank=artifacts.latent_rank,
            field_matrix=compose_latent_field_matrix(
                theta_parts["node_factors"], theta_parts["time_factors"]
            ),
            node_factors=np.asarray(theta_parts["node_factors"], dtype=float),
            time_factors=np.asarray(theta_parts["time_factors"], dtype=float),
        )
    if artifacts.field_basis is None:
        raise ValueError("Additive mode requires field_basis.")
    field_coeffs = np.asarray(theta_parts["field_coeffs"], dtype=float)
    tau = np.asarray(theta_parts["tau"], dtype=float)
    field_vector = compose_field(field_coeffs, artifacts.field_basis)
    return ModelArtifacts(
        field_mode=artifacts.field_mode,
        gamma_matrix=artifacts.gamma_matrix,
        field_basis=artifacts.field_basis,
        field_names=artifacts.field_names,
        shared_features=artifacts.shared_features,
        field_matrix=compose_additive_field_matrix(
            field_coeffs, tau, artifacts.field_basis
        ),
        field_coeffs=field_coeffs,
        tau=tau,
        field_vector=field_vector,
    )


def load_true_parameters(
    config, artifacts: ModelArtifacts, fit_intervention_model: bool | None = None
) -> np.ndarray:
    """Pack the saved true parameter vector from model artifacts and config scalars."""
    if fit_intervention_model is None:
        fit_intervention_model = intervention_model_enabled(config)
    if artifacts.field_mode == LATENT_FIELD_MODE:
        if artifacts.node_factors is None or artifacts.time_factors is None:
            raise ValueError("Missing latent truth in field_artifacts.")
        field_block = np.concatenate(
            [artifacts.node_factors.reshape(-1), artifacts.time_factors.reshape(-1)]
        )
    else:
        if artifacts.field_coeffs is None or artifacts.tau is None:
            raise ValueError("Missing additive truth in field_artifacts.")
        field_block = np.concatenate([artifacts.field_coeffs, artifacts.tau])
    return np.concatenate(
        [
            field_block,
            np.array([config.estimation_params.beta], dtype=float),
            np.array([get_xi(config)], dtype=float),
            np.array([config.estimation_params.eta], dtype=float),
            *(
                [
                    np.array(
                        [config.estimation_params.zeta, config.estimation_params.psi],
                        dtype=float,
                    )
                ]
                if fit_intervention_model
                else []
            ),
        ]
    )


def summary_metrics(
    est_theta: np.ndarray,
    true_theta: np.ndarray,
    artifacts: ModelArtifacts,
    fit_intervention_model: bool = True,
) -> dict[str, float]:
    """Compute reconstruction metrics for fitted parameters, fields, and interactions."""
    t_steps = infer_t_steps_from_theta(est_theta, artifacts, fit_intervention_model)
    est_parts = unpack_theta(est_theta, artifacts, t_steps, fit_intervention_model)
    true_parts = unpack_theta(true_theta, artifacts, t_steps, fit_intervention_model)
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

    metrics = {
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
    if artifacts.field_mode != LATENT_FIELD_MODE:
        metrics["static_field_rmse"] = float(
            np.sqrt(
                np.mean((est_artifacts.field_vector - true_artifacts.field_vector) ** 2)
            )
        )
        metrics["tau_rmse"] = float(
            np.sqrt(np.mean((est_artifacts.tau - true_artifacts.tau) ** 2))
        )
    return metrics
