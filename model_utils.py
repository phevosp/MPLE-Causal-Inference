"""Minimal shared helpers for the active conditional MPLE pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


DEFAULT_NUM_SHARED_FEATURES = 5
DEFAULT_FIELD_MODE = "uniform"
DEFAULT_INTERACTION_MODE = "known_graph"


@dataclass(frozen=True)
class BasisExpansion:
    """Container for one field basis and one fixed interaction template."""

    field_basis: np.ndarray
    interaction_basis: object
    field_names: tuple[str, ...]
    interaction_names: tuple[str, ...]
    shared_features: np.ndarray
    shared_feature_names: tuple[str, ...]


def intervention_model_enabled(config) -> bool:
    """Return whether the MPLE fit should include the intervention process."""
    estimation_params = getattr(config, "estimation_params", None)
    if estimation_params is None or "fit_intervention_model" not in estimation_params:
        return True
    return bool(estimation_params.fit_intervention_model)


def scalar_parameter_count(fit_intervention_model: bool) -> int:
    """Return the number of scalar tail parameters in theta."""
    return 4 if fit_intervention_model else 2


def interaction_basis_count(interaction_basis) -> int:
    """The active pipeline keeps exactly one interaction template."""
    _ = interaction_basis
    return 1


def interaction_matrix_infinity_norm(matrix) -> float:
    """Compute the matrix infinity norm for dense or sparse interaction matrices."""
    if sparse.issparse(matrix):
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        return float(row_sums.max()) if row_sums.size else 0.0
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    """Center and scale one vector by infinity norm."""
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    norm = np.linalg.norm(values, ord=np.inf)
    if norm < 1e-12:
        return np.zeros_like(values)
    return values / norm


def _normalize_dense_interaction(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric zero-diagonal dense interaction template with inf norm one."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    norm = np.linalg.norm(matrix, ord=np.inf)
    if norm < 1e-12:
        return np.zeros_like(matrix)
    return matrix / norm


def _normalize_sparse_interaction(matrix) -> sparse.csr_matrix:
    """Return a symmetric zero-diagonal sparse interaction template with inf norm one."""
    interaction = sparse.csr_matrix(matrix, dtype=float)
    interaction = ((interaction + interaction.T) * 0.5).tocsr()
    interaction.setdiag(0.0)
    interaction.eliminate_zeros()
    norm = interaction_matrix_infinity_norm(interaction)
    if norm < 1e-12:
        return sparse.csr_matrix(interaction.shape, dtype=float)
    return interaction.multiply(1.0 / norm).tocsr()


def validate_basis_infinity_norms(
    field_basis: np.ndarray,
    interaction_basis,
    tol: float = 1e-8,
) -> None:
    """Ensure the active field basis rows and interaction template are normalized."""
    field_basis = np.asarray(field_basis, dtype=float)
    if field_basis.ndim != 2:
        raise ValueError("field_basis must be a 2D array.")
    if field_basis.shape[0] > 0:
        field_norms = np.linalg.norm(field_basis, ord=np.inf, axis=1)
        if np.any(field_norms < tol):
            raise ValueError("Field basis contains a degenerate zero vector.")
        if not np.allclose(field_norms, 1.0, atol=tol, rtol=0.0):
            raise ValueError("Each field basis vector must have infinity norm one.")

    interaction_norm = interaction_matrix_infinity_norm(interaction_basis)
    if interaction_norm < tol:
        raise ValueError("Interaction template is degenerate.")
    if not np.isclose(interaction_norm, 1.0, atol=tol, rtol=0.0):
        raise ValueError("The interaction basis matrix must have infinity norm one.")


def _basis_params(config):
    """Return the basis-parameter section, if present."""
    if "basis_params" in config.global_params:
        return config.global_params.basis_params
    return None


def _basis_setting(config, key: str, default):
    """Read one basis setting with a default fallback."""
    basis_params = _basis_params(config)
    if basis_params is None or key not in basis_params:
        return default
    return basis_params[key]


def _centered_quadratic(feature: np.ndarray) -> np.ndarray:
    """Return a centered quadratic transform of one shared feature."""
    squared = np.asarray(feature, dtype=float) ** 2
    return squared - squared.mean()


def build_shared_features(config, n_nodes: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the shared node features used by the additive field basis."""
    num_features = int(_basis_setting(config, "num_shared_features", DEFAULT_NUM_SHARED_FEATURES))
    feature_seed = int(_basis_setting(config, "shared_feature_seed", config.generation_params.seed))
    rng = np.random.default_rng(feature_seed)
    raw_features = rng.normal(size=(num_features, n_nodes))
    shared_features = np.vstack([_normalize_vector(feature) for feature in raw_features])
    feature_names = tuple(f"feature_{idx + 1}" for idx in range(num_features))
    return shared_features, feature_names


def build_basis_expansion(config, gamma_matrix) -> BasisExpansion:
    """Build the active field basis and fixed known-graph interaction template."""
    gamma_is_sparse = sparse.issparse(gamma_matrix)
    n_nodes = gamma_matrix.shape[0]
    field_mode = str(_basis_setting(config, "field_mode", DEFAULT_FIELD_MODE))
    interaction_mode = str(_basis_setting(config, "interaction_mode", DEFAULT_INTERACTION_MODE))
    if interaction_mode != "known_graph":
        raise ValueError("Only interaction_mode='known_graph' is supported in the active pipeline.")

    shared_features, shared_feature_names = build_shared_features(config, n_nodes)
    field_rows: list[np.ndarray] = []
    field_names: list[str] = []
    if field_mode == "uniform":
        pass
    elif field_mode == "shared_feature_field":
        for feature_name, feature in zip(shared_feature_names, shared_features):
            linear = _normalize_vector(feature)
            quadratic = _normalize_vector(_centered_quadratic(feature))
            if np.linalg.norm(linear, ord=np.inf) >= 1e-12:
                field_rows.append(linear)
                field_names.append(f"linear::{feature_name}")
            if np.linalg.norm(quadratic, ord=np.inf) >= 1e-12:
                field_rows.append(quadratic)
                field_names.append(f"quadratic::{feature_name}")
    else:
        raise ValueError(f"Unknown field_mode '{field_mode}'.")

    field_basis = (
        np.vstack(field_rows).astype(float)
        if field_rows
        else np.empty((0, n_nodes), dtype=float)
    )
    interaction_basis = (
        _normalize_sparse_interaction(gamma_matrix)
        if gamma_is_sparse
        else _normalize_dense_interaction(np.asarray(gamma_matrix, dtype=float))
    )
    validate_basis_infinity_norms(field_basis, interaction_basis)
    return BasisExpansion(
        field_basis=field_basis,
        interaction_basis=interaction_basis,
        field_names=tuple(field_names),
        interaction_names=("adjacency",),
        shared_features=shared_features,
        shared_feature_names=shared_feature_names,
    )


def compose_field(field_coeffs: np.ndarray, field_basis: np.ndarray) -> np.ndarray:
    """Map field coefficients to the realized node-wise external field."""
    return np.asarray(field_coeffs, dtype=float) @ np.asarray(field_basis, dtype=float)


def compose_field_matrix(
    field_coeffs: np.ndarray,
    tau: np.ndarray,
    field_basis: np.ndarray,
) -> np.ndarray:
    """Compose the realized T x N external field from node and time components."""
    static_field = compose_field(field_coeffs, field_basis)
    return static_field[None, :] + np.asarray(tau, dtype=float)[:, None]


def compose_interaction_matrix(interaction_coeffs: np.ndarray, interaction_basis):
    """Scale the single fixed interaction template by its scalar coefficient."""
    coeffs = np.asarray(interaction_coeffs, dtype=float)
    if coeffs.shape != (1,):
        raise ValueError("The active pipeline expects exactly one interaction coefficient.")
    scale = float(coeffs[0])
    if sparse.issparse(interaction_basis):
        interaction_matrix = sparse.csr_matrix(interaction_basis, dtype=float).multiply(scale).tocsr()
        interaction_matrix = ((interaction_matrix + interaction_matrix.T) * 0.5).tocsr()
        interaction_matrix.setdiag(0.0)
        interaction_matrix.eliminate_zeros()
        return interaction_matrix
    interaction_matrix = scale * np.asarray(interaction_basis, dtype=float)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
    np.fill_diagonal(interaction_matrix, 0.0)
    return interaction_matrix


def interaction_features(x: np.ndarray, interaction_basis) -> np.ndarray:
    """Precompute the single interaction feature tensor expected by MPLE."""
    x = np.asarray(x, dtype=float)
    if sparse.issparse(interaction_basis):
        features = np.asarray(x @ sparse.csr_matrix(interaction_basis).T)
    else:
        features = x @ np.asarray(interaction_basis, dtype=float).T
    return features.reshape(1, x.shape[0], x.shape[1])


def get_field_coeffs(config) -> np.ndarray:
    """Load field coefficients from config."""
    if "field_coefs" not in config.estimation_params:
        raise KeyError("estimation_params.field_coefs is required.")
    return np.asarray(config.estimation_params.field_coefs, dtype=float)


def get_interaction_coeffs(config) -> np.ndarray:
    """Load the single interaction coefficient from config."""
    if "interaction_coefs" not in config.estimation_params:
        raise KeyError("estimation_params.interaction_coefs is required.")
    coeffs = np.asarray(config.estimation_params.interaction_coefs, dtype=float)
    if coeffs.shape != (1,):
        raise ValueError("The active pipeline expects estimation_params.interaction_coefs to have length one.")
    return coeffs


def get_temporal_field(config, t_steps: int) -> np.ndarray:
    """Load the realized shared time-varying field from config."""
    estimation_params = config.estimation_params
    if "tau_params" not in estimation_params or estimation_params.tau_params is None:
        return np.zeros(t_steps, dtype=float)

    tau_params = estimation_params.tau_params
    mode = str(tau_params.mode)
    if mode == "fixed":
        tau = np.asarray(tau_params["vector"], dtype=float)
        if tau.shape != (t_steps,):
            raise ValueError("Fixed tau vector must have length equal to global_params.T.")
        return tau
    if mode == "uniform_random":
        lower = float(tau_params.lower)
        upper = float(tau_params.upper)
        if lower > upper:
            raise ValueError("tau_params.lower must be less than or equal to tau_params.upper.")
        tau_seed = int(getattr(tau_params, "seed", config.generation_params.seed))
        rng = np.random.default_rng(tau_seed)
        return rng.uniform(lower, upper, size=t_steps)
    raise ValueError(f"Unknown tau generation mode '{mode}'.")


def load_or_build_basis(config, gamma_matrix) -> BasisExpansion:
    """Build the active basis expansion and validate it."""
    basis = build_basis_expansion(config, gamma_matrix)
    validate_basis_infinity_norms(basis.field_basis, basis.interaction_basis)
    return basis


def parameter_names(
    field_names: tuple[str, ...],
    interaction_names: tuple[str, ...],
    t_steps: int,
    fit_intervention_model: bool = True,
) -> list[str]:
    """Create human-readable parameter labels matching the flattened theta vector."""
    field_keys = [f"field::{name}" for name in field_names]
    temporal_keys = [f"tau::t_{idx}" for idx in range(t_steps)]
    interaction_keys = [f"interaction::{name}" for name in interaction_names]
    tail_keys = ["eta", "zeta", "psi"] if fit_intervention_model else ["eta"]
    return field_keys + temporal_keys + ["beta"] + interaction_keys + tail_keys


def pack_true_parameters(
    config,
    field_names: tuple[str, ...],
    interaction_names: tuple[str, ...],
    fit_intervention_model: bool | None = None,
) -> np.ndarray:
    """Pack the configured true parameters in the optimizer's flat ordering."""
    if fit_intervention_model is None:
        fit_intervention_model = intervention_model_enabled(config)
    field_coeffs = get_field_coeffs(config)
    tau = get_temporal_field(config, int(config.global_params.T))
    interaction_coeffs = get_interaction_coeffs(config)
    if len(field_coeffs) != len(field_names):
        raise ValueError("Field coefficient count does not match the configured field basis.")
    if len(interaction_names) != 1:
        raise ValueError("The active pipeline expects exactly one interaction name.")
    return np.concatenate(
        [
            field_coeffs,
            tau,
            np.array([config.estimation_params.beta], dtype=float),
            interaction_coeffs,
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


def unpack_theta(
    theta: np.ndarray,
    n_field: int,
    n_interaction: int,
    t_steps: int,
    fit_intervention_model: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, float]:
    """Split theta into field, temporal, scalar, and intervention-process blocks."""
    if n_interaction != 1:
        raise ValueError("The active pipeline expects exactly one interaction parameter.")
    field_coeffs = np.asarray(theta[:n_field], dtype=float)
    tau = np.asarray(theta[n_field : n_field + t_steps], dtype=float)
    beta = float(theta[n_field + t_steps])
    interaction_coeffs = np.asarray(theta[n_field + t_steps + 1 : n_field + t_steps + 2], dtype=float)
    tail = np.asarray(theta[n_field + t_steps + 2 :], dtype=float)
    if fit_intervention_model:
        eta, zeta, psi = tail
    else:
        eta = float(tail[0])
        zeta = 0.0
        psi = 0.0
    return field_coeffs, tau, beta, interaction_coeffs, eta, zeta, psi


def summary_metrics(
    est_theta: np.ndarray,
    true_theta: np.ndarray,
    field_basis: np.ndarray,
    interaction_basis,
    fit_intervention_model: bool = True,
) -> dict[str, float]:
    """Compute reconstruction metrics for fitted parameters, fields, and interactions."""
    t_steps = int(
        len(est_theta)
        - field_basis.shape[0]
        - interaction_basis_count(interaction_basis)
        - scalar_parameter_count(fit_intervention_model)
    )
    if t_steps < 0:
        raise ValueError("Parameter vector is too short for the configured model blocks.")
    n_field = field_basis.shape[0]
    n_interaction = interaction_basis_count(interaction_basis)
    est_field_coeffs, est_tau, _, est_interaction_coeffs, _, _, _ = unpack_theta(
        est_theta, n_field, n_interaction, t_steps, fit_intervention_model
    )
    true_field_coeffs, true_tau, _, true_interaction_coeffs, _, _, _ = unpack_theta(
        true_theta, n_field, n_interaction, t_steps, fit_intervention_model
    )
    est_field = compose_field(est_field_coeffs, field_basis)
    true_field = compose_field(true_field_coeffs, field_basis)
    est_field_matrix = compose_field_matrix(est_field_coeffs, est_tau, field_basis)
    true_field_matrix = compose_field_matrix(true_field_coeffs, true_tau, field_basis)
    est_interaction = compose_interaction_matrix(est_interaction_coeffs, interaction_basis)
    true_interaction = compose_interaction_matrix(true_interaction_coeffs, interaction_basis)

    if sparse.issparse(est_interaction):
        interaction_error = est_interaction - true_interaction
        interaction_fro_error = float(np.sqrt(interaction_error.multiply(interaction_error).sum()))
    else:
        interaction_fro_error = float(np.linalg.norm(est_interaction - true_interaction, ord="fro"))

    return {
        "field_rmse": float(np.sqrt(np.mean((est_field_matrix - true_field_matrix) ** 2))),
        "field_l2_error": float(np.linalg.norm((est_field_matrix - true_field_matrix).reshape(-1), ord=2)),
        "static_field_rmse": float(np.sqrt(np.mean((est_field - true_field) ** 2))),
        "tau_rmse": float(np.sqrt(np.mean((est_tau - true_tau) ** 2))),
        "interaction_fro_error": interaction_fro_error,
        "parameter_rmse": float(np.sqrt(np.mean((est_theta - true_theta) ** 2))),
    }
