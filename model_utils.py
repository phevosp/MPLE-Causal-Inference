from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


DEFAULT_NUM_SHARED_FEATURES = 5
DEFAULT_FIELD_MODE = "uniform"
DEFAULT_INTERACTION_MODE = "known_graph"


@dataclass(frozen=True)
class BasisExpansion:
    """Container for the field and interaction bases used in one experiment."""

    field_basis: np.ndarray
    interaction_basis: np.ndarray
    field_names: tuple[str, ...]
    interaction_names: tuple[str, ...]
    shared_features: np.ndarray
    shared_feature_names: tuple[str, ...]


def intervention_model_enabled(config) -> bool:
    """Return whether the MPLE fit should include a model for the intervention process z."""
    if "estimation_params" not in config:
        return True
    estimation_params = config.estimation_params
    if "fit_intervention_model" not in estimation_params:
        return True
    return bool(estimation_params.fit_intervention_model)


def scalar_parameter_count(fit_intervention_model: bool) -> int:
    """Return the number of non-basis scalar parameters in the flattened optimizer vector."""
    return 4 if fit_intervention_model else 2


def interaction_basis_count(interaction_basis) -> int:
    """Return the number of interaction templates represented by one basis object."""
    if sparse.issparse(interaction_basis):
        return 1
    basis_array = np.asarray(interaction_basis)
    if basis_array.ndim == 2:
        return 1
    return int(basis_array.shape[0])


def interaction_matrix_infinity_norm(matrix) -> float:
    """Compute the infinity norm of one dense or sparse interaction matrix."""
    if sparse.issparse(matrix):
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        return float(row_sums.max()) if row_sums.size else 0.0
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def validate_basis_infinity_norms(
    field_basis: np.ndarray,
    interaction_basis,
    tol: float = 1e-8,
) -> None:
    """Ensure every nondegenerate basis element has infinity norm one."""
    field_norms = np.linalg.norm(np.asarray(field_basis, dtype=float), ord=np.inf, axis=1)
    if sparse.issparse(interaction_basis):
        interaction_norms = np.array([interaction_matrix_infinity_norm(interaction_basis)])
    else:
        interaction_array = np.asarray(interaction_basis, dtype=float)
        if interaction_array.ndim == 2:
            interaction_norms = np.array([interaction_matrix_infinity_norm(interaction_array)])
        else:
            interaction_norms = np.array(
                [interaction_matrix_infinity_norm(matrix) for matrix in interaction_array]
            )

    if np.any(field_norms < tol) or np.any(interaction_norms < tol):
        raise ValueError("Basis construction produced a degenerate zero template.")
    if not np.allclose(field_norms, 1.0, atol=tol, rtol=0.0):
        raise ValueError("Each field basis vector must have infinity norm one.")
    if not np.allclose(interaction_norms, 1.0, atol=tol, rtol=0.0):
        raise ValueError("Each interaction basis matrix must have infinity norm one.")


def _safe_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector by infinity norm, returning zeros if it is degenerate."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, ord=np.inf)
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _safe_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize, zero the diagonal, and normalize a matrix by infinity norm."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    norm = np.linalg.norm(matrix, ord=np.inf)
    if norm < 1e-12:
        return np.zeros_like(matrix)
    return matrix / norm


def _stack_vector_basis(vectors: list[np.ndarray]) -> np.ndarray:
    """Stack field templates after infinity-norm normalization without re-scaling them."""
    normalized = [_safe_normalize_vector(vector) for vector in vectors]
    return np.vstack(normalized)


def _stack_matrix_basis(matrices: list[np.ndarray]) -> np.ndarray:
    """Stack interaction templates after infinity-norm normalization without re-scaling them."""
    normalized = [_safe_normalize_matrix(matrix) for matrix in matrices]
    return np.stack(normalized)


def _basis_params(config):
    """Return the basis configuration section, if present."""
    if "basis_params" in config.global_params:
        return config.global_params.basis_params
    return None


def _get_basis_setting(config, key: str, default):
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
    """Generate the shared node features used by both the field and interaction bases."""
    num_features = int(_get_basis_setting(config, "num_shared_features", DEFAULT_NUM_SHARED_FEATURES))
    feature_seed = int(
        _get_basis_setting(config, "shared_feature_seed", config.generation_params.seed)
    )
    rng = np.random.default_rng(feature_seed)
    raw_features = rng.normal(size=(num_features, n_nodes))
    centered = raw_features - raw_features.mean(axis=1, keepdims=True)
    shared_features = np.vstack(
        [_safe_normalize_vector(feature) for feature in centered]
    )
    feature_names = tuple(f"feature_{idx + 1}" for idx in range(num_features))
    return shared_features, feature_names


def _interaction_distance_kernel(
    feature: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Construct a distance-kernel interaction template from one shared feature."""
    pairwise_distance = np.abs(feature[:, None] - feature[None, :])
    return np.exp(-decay * pairwise_distance)


def _interaction_cross_similarity(feature: np.ndarray) -> np.ndarray:
    """Construct a simple similarity template from one shared feature."""
    return np.outer(feature, feature)


def build_basis_expansion(config, gamma_matrix: np.ndarray) -> BasisExpansion:
    """Construct infinity-normalized field and interaction bases from shared features."""
    n_nodes = gamma_matrix.shape[0]
    shared_features, shared_feature_names = build_shared_features(config, n_nodes)
    field_mode = str(_get_basis_setting(config, "field_mode", DEFAULT_FIELD_MODE))
    interaction_mode = str(
        _get_basis_setting(config, "interaction_mode", DEFAULT_INTERACTION_MODE)
    )
    distance_decay = float(_get_basis_setting(config, "distance_kernel_decay", 3.0))

    field_vectors = [np.ones(n_nodes, dtype=float)]
    field_names = ["intercept"]
    if field_mode == "shared_feature_field":
        for feature_name, feature in zip(shared_feature_names, shared_features):
            field_vectors.append(feature)
            field_names.append(f"linear::{feature_name}")
            field_vectors.append(_centered_quadratic(feature))
            field_names.append(f"quadratic::{feature_name}")
    elif field_mode != "uniform":
        raise ValueError(f"Unknown field_mode '{field_mode}'.")

    interaction_matrices = [gamma_matrix]
    interaction_names = ["adjacency"]
    if interaction_mode == "shared_feature_interactions":
        for feature_name, feature in zip(shared_feature_names, shared_features):
            interaction_matrices.append(_interaction_distance_kernel(feature, distance_decay))
            interaction_names.append(f"distance_kernel::{feature_name}")
            interaction_matrices.append(_interaction_cross_similarity(feature))
            interaction_names.append(f"cross_similarity::{feature_name}")
    elif interaction_mode != "known_graph":
        raise ValueError(f"Unknown interaction_mode '{interaction_mode}'.")

    field_basis = _stack_vector_basis(field_vectors)
    interaction_basis = _stack_matrix_basis(interaction_matrices)
    validate_basis_infinity_norms(field_basis, interaction_basis)

    return BasisExpansion(
        field_basis=field_basis,
        interaction_basis=interaction_basis,
        field_names=tuple(field_names),
        interaction_names=tuple(interaction_names),
        shared_features=shared_features,
        shared_feature_names=shared_feature_names,
    )


def compose_field(field_coeffs: np.ndarray, field_basis: np.ndarray) -> np.ndarray:
    """Map field coefficients to the realized node-wise external field."""
    return np.asarray(field_coeffs, dtype=float) @ np.asarray(field_basis, dtype=float)


def compose_interaction_matrix(
    interaction_coeffs: np.ndarray,
    interaction_basis,
):
    """Map interaction coefficients to the realized symmetric interaction matrix."""
    coeffs = np.asarray(interaction_coeffs, dtype=float)
    if sparse.issparse(interaction_basis):
        if coeffs.shape != (1,):
            raise ValueError("Sparse interaction bases currently support exactly one template.")
        interaction_matrix = interaction_basis.multiply(float(coeffs[0])).tocsr()
        interaction_matrix = (interaction_matrix + interaction_matrix.T) * 0.5
        interaction_matrix.setdiag(0.0)
        interaction_matrix.eliminate_zeros()
        return interaction_matrix

    basis_array = np.asarray(interaction_basis, dtype=float)
    if basis_array.ndim == 2:
        if coeffs.shape != (1,):
            raise ValueError("A single dense interaction matrix requires exactly one coefficient.")
        interaction_matrix = coeffs[0] * basis_array
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
        np.fill_diagonal(interaction_matrix, 0.0)
        return interaction_matrix

    interaction_matrix = np.tensordot(
        coeffs,
        basis_array,
        axes=(0, 0),
    )
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
    np.fill_diagonal(interaction_matrix, 0.0)
    return interaction_matrix


def interaction_features(
    x: np.ndarray,
    interaction_basis,
) -> np.ndarray:
    """Precompute basis-specific interaction features for each time step and node."""
    if sparse.issparse(interaction_basis):
        features = np.asarray(x @ interaction_basis.T)
        return features.reshape(1, x.shape[0], x.shape[1])

    basis_array = np.asarray(interaction_basis, dtype=float)
    if basis_array.ndim == 2:
        features = np.asarray(x @ basis_array.T)
        return features.reshape(1, x.shape[0], x.shape[1])

    return np.einsum("tn,kmn->ktm", x, basis_array, optimize=True)


def get_field_coeffs(config) -> np.ndarray:
    """Load field coefficients from config, with legacy support for scalar alpha."""
    if "field_coefs" in config.estimation_params:
        return np.asarray(config.estimation_params.field_coefs, dtype=float)
    return np.array([float(config.estimation_params.alpha)], dtype=float)


def get_interaction_coeffs(config) -> np.ndarray:
    """Load interaction coefficients from config, with legacy support for scalar xi."""
    if "interaction_coefs" in config.estimation_params:
        return np.asarray(config.estimation_params.interaction_coefs, dtype=float)
    return np.array([float(config.estimation_params.xi)], dtype=float)


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
            raise ValueError(
                "Fixed tau vector must have length equal to global_params.T."
            )
        return tau

    if mode == "uniform_random":
        lower = float(tau_params.lower)
        upper = float(tau_params.upper)
        if lower > upper:
            raise ValueError("tau_params.lower must be less than or equal to upper.")
        tau_seed = int(getattr(tau_params, "seed", config.generation_params.seed))
        rng = np.random.default_rng(tau_seed)
        return rng.uniform(lower, upper, size=t_steps)

    raise ValueError(f"Unknown tau generation mode '{mode}'.")


def load_or_build_basis(config, gamma_matrix: np.ndarray) -> BasisExpansion:
    """Build the configured basis or fall back to the original scalar setup."""
    field_coeffs = get_field_coeffs(config)
    interaction_coeffs = get_interaction_coeffs(config)
    basis_params = _basis_params(config)
    if basis_params is None and len(field_coeffs) == 1 and len(interaction_coeffs) == 1:
        basis = BasisExpansion(
            field_basis=_stack_vector_basis([np.ones(gamma_matrix.shape[0], dtype=float)]),
            interaction_basis=_stack_matrix_basis([gamma_matrix]),
            field_names=("intercept",),
            interaction_names=("adjacency",),
            shared_features=np.empty((0, gamma_matrix.shape[0]), dtype=float),
            shared_feature_names=(),
        )
        validate_basis_infinity_norms(basis.field_basis, basis.interaction_basis)
        return basis
    basis = build_basis_expansion(config, gamma_matrix)
    validate_basis_infinity_norms(basis.field_basis, basis.interaction_basis)
    return basis


def parameter_names(
    field_names: tuple[str, ...],
    interaction_names: tuple[str, ...],
    t_steps: int,
    fit_intervention_model: bool = True,
) -> list[str]:
    """Create human-readable parameter labels matching the flattened optimizer vector."""
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
    """Pack the true configuration parameters into the optimizer's flat ordering."""
    if fit_intervention_model is None:
        fit_intervention_model = intervention_model_enabled(config)
    field_coeffs = get_field_coeffs(config)
    tau = get_temporal_field(config, int(config.global_params.T))
    interaction_coeffs = get_interaction_coeffs(config)
    if len(field_coeffs) != len(field_names):
        raise ValueError(
            "Number of field coefficients does not match the configured field basis."
        )
    if len(interaction_coeffs) != len(interaction_names):
        raise ValueError(
            "Number of interaction coefficients does not match the configured interaction basis."
        )
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
                        [
                            config.estimation_params.zeta,
                            config.estimation_params.psi,
                        ],
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
    """Split the optimizer vector into field, treatment, interaction, and temporal blocks."""
    field_coeffs = np.asarray(theta[:n_field], dtype=float)
    tau = np.asarray(theta[n_field : n_field + t_steps], dtype=float)
    beta = float(theta[n_field + t_steps])
    interaction_start = n_field + t_steps + 1
    interaction_end = interaction_start + n_interaction
    interaction_coeffs = np.asarray(theta[interaction_start:interaction_end], dtype=float)
    tail = np.asarray(theta[interaction_end:], dtype=float)
    if fit_intervention_model:
        eta, zeta, psi = tail
    else:
        eta = float(tail[0])
        zeta = 0.0
        psi = 0.0
    return field_coeffs, tau, beta, interaction_coeffs, eta, zeta, psi


def compose_field_matrix(
    field_coeffs: np.ndarray,
    tau: np.ndarray,
    field_basis: np.ndarray,
) -> np.ndarray:
    """Compose the realized T x N external field with node and time components."""
    static_field = compose_field(field_coeffs, field_basis)
    return static_field[None, :] + np.asarray(tau, dtype=float)[:, None]


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
    est_interaction = compose_interaction_matrix(
        est_interaction_coeffs, interaction_basis
    )
    true_interaction = compose_interaction_matrix(
        true_interaction_coeffs, interaction_basis
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
        "field_rmse": float(np.sqrt(np.mean((est_field_matrix - true_field_matrix) ** 2))),
        "field_l2_error": float(
            np.linalg.norm((est_field_matrix - true_field_matrix).reshape(-1), ord=2)
        ),
        "static_field_rmse": float(np.sqrt(np.mean((est_field - true_field) ** 2))),
        "tau_rmse": float(np.sqrt(np.mean((est_tau - true_tau) ** 2))),
        "interaction_fro_error": interaction_fro_error,
        "parameter_rmse": float(np.sqrt(np.mean((est_theta - true_theta) ** 2))),
    }
