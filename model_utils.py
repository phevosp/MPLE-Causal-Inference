from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_FIELD_TEMPLATES = ("intercept", "linear", "quadratic")
DEFAULT_INTERACTION_TEMPLATES = (
    "adjacency",
    "distance_kernel",
    "cross_similarity",
)


@dataclass(frozen=True)
class BasisExpansion:
    """Container for the known low-dimensional field and interaction templates."""

    field_basis: np.ndarray
    interaction_basis: np.ndarray
    field_names: tuple[str, ...]
    interaction_names: tuple[str, ...]


def _safe_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector by infinity norm, returning zeros if it is degenerate."""
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


def _node_coordinate(n_nodes: int) -> np.ndarray:
    """Create a deterministic one-dimensional node coordinate on [-1, 1]."""
    if n_nodes == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-1.0, 1.0, n_nodes, dtype=float)


def _orthonormalize_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Apply Gram-Schmidt to a list of vectors and return an orthonormal row stack."""
    basis = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=float).copy()
        for prev in basis:
            residual -= np.dot(residual, prev) * prev
        norm = np.linalg.norm(residual)
        if norm < 1e-12:
            raise ValueError("Configured field templates are linearly dependent.")
        basis.append(residual / norm)
    return np.vstack(basis)


def _orthonormalize_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Apply Gram-Schmidt to symmetric matrices under the Frobenius inner product."""
    basis_flat = []
    basis_mats = []
    for matrix in matrices:
        residual = np.asarray(matrix, dtype=float).copy().reshape(-1)
        for prev in basis_flat:
            residual -= np.dot(residual, prev) * prev
        norm = np.linalg.norm(residual)
        if norm < 1e-12:
            raise ValueError("Configured interaction templates are linearly dependent.")
        residual /= norm
        basis_flat.append(residual)
        basis_mats.append(residual.reshape(matrix.shape))
    return np.stack(basis_mats)


def _template_names(
    config_section, key: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    """Read a tuple of configured template names, falling back to defaults when absent."""
    if config_section is None or key not in config_section:
        return default
    return tuple(config_section[key])


def build_basis_expansion(config, gamma_matrix: np.ndarray) -> BasisExpansion:
    """Construct the configured field and interaction bases from the observed network."""
    basis_config = (
        config.global_params.basis_params
        if "basis_params" in config.global_params
        else None
    )
    field_templates = _template_names(
        basis_config,
        "field_templates",
        DEFAULT_FIELD_TEMPLATES,
    )
    interaction_templates = _template_names(
        basis_config,
        "interaction_templates",
        DEFAULT_INTERACTION_TEMPLATES,
    )

    n_nodes = gamma_matrix.shape[0]
    u = _node_coordinate(n_nodes)
    quadratic = u**2 - np.mean(u**2)
    pairwise_distance = np.abs(u[:, None] - u[None, :])
    cross_similarity = 0.5 * (np.outer(u, quadratic) + np.outer(quadratic, u))

    field_template_map = {
        "intercept": np.ones(n_nodes, dtype=float),
        "linear": u,
        "quadratic": quadratic,
    }
    interaction_template_map = {
        "adjacency": gamma_matrix,
        "distance_kernel": np.exp(-3.0 * pairwise_distance),
        "cross_similarity": cross_similarity,
    }

    field_basis = _orthonormalize_vectors(
        [_safe_normalize_vector(field_template_map[name]) for name in field_templates]
    )
    interaction_basis = _orthonormalize_matrices(
        [
            _safe_normalize_matrix(interaction_template_map[name])
            for name in interaction_templates
        ]
    )

    return BasisExpansion(
        field_basis=field_basis,
        interaction_basis=interaction_basis,
        field_names=field_templates,
        interaction_names=interaction_templates,
    )


def compose_field(field_coeffs: np.ndarray, field_basis: np.ndarray) -> np.ndarray:
    """Map field coefficients to the realized node-wise external field."""
    return np.asarray(field_coeffs, dtype=float) @ np.asarray(field_basis, dtype=float)


def compose_interaction_matrix(
    interaction_coeffs: np.ndarray,
    interaction_basis: np.ndarray,
) -> np.ndarray:
    """Map interaction coefficients to the realized symmetric interaction matrix."""
    interaction_matrix = np.tensordot(
        np.asarray(interaction_coeffs, dtype=float),
        np.asarray(interaction_basis, dtype=float),
        axes=(0, 0),
    )
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
    np.fill_diagonal(interaction_matrix, 0.0)
    return interaction_matrix


def interaction_features(
    x: np.ndarray,
    interaction_basis: np.ndarray,
) -> np.ndarray:
    """Precompute basis-specific interaction features for each time step and node."""
    return np.einsum("tn,kmn->ktm", x, interaction_basis, optimize=True)


def compose_interaction_term(
    x: np.ndarray,
    interaction_coeffs: np.ndarray,
    interaction_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full interaction term and return it alongside the basis features."""
    features = interaction_features(x, interaction_basis)
    interaction_term = np.tensordot(
        np.asarray(interaction_coeffs, dtype=float),
        features,
        axes=(0, 0),
    )
    return interaction_term, features


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


def load_or_build_basis(config, gamma_matrix: np.ndarray) -> BasisExpansion:
    """Load the configured basis or fall back to the original scalar-field/scalar-network model."""
    field_coeffs = get_field_coeffs(config)
    interaction_coeffs = get_interaction_coeffs(config)
    if len(field_coeffs) == 1 and len(interaction_coeffs) == 1:
        return BasisExpansion(
            field_basis=np.ones((1, gamma_matrix.shape[0]), dtype=float),
            interaction_basis=gamma_matrix[None, :, :],
            field_names=("intercept",),
            interaction_names=("adjacency",),
        )
    return build_basis_expansion(config, gamma_matrix)


def parameter_names(
    field_names: tuple[str, ...],
    interaction_names: tuple[str, ...],
) -> list[str]:
    """Create human-readable parameter labels matching the flattened optimizer vector."""
    field_keys = [f"field::{name}" for name in field_names]
    interaction_keys = [f"interaction::{name}" for name in interaction_names]
    return field_keys + ["beta"] + interaction_keys + ["eta", "zeta", "psi"]


def pack_true_parameters(
    config,
    field_names: tuple[str, ...],
    interaction_names: tuple[str, ...],
) -> np.ndarray:
    """Pack the true configuration parameters into the optimizer's flat ordering."""
    field_coeffs = get_field_coeffs(config)
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
            np.array([config.estimation_params.beta], dtype=float),
            interaction_coeffs,
            np.array(
                [
                    config.estimation_params.eta,
                    config.estimation_params.zeta,
                    config.estimation_params.psi,
                ],
                dtype=float,
            ),
        ]
    )


def unpack_theta(
    theta: np.ndarray,
    n_field: int,
    n_interaction: int,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float]:
    """Split the optimizer vector into field, treatment, interaction, and temporal blocks."""
    field_coeffs = np.asarray(theta[:n_field], dtype=float)
    beta = float(theta[n_field])
    interaction_start = n_field + 1
    interaction_end = interaction_start + n_interaction
    interaction_coeffs = np.asarray(
        theta[interaction_start:interaction_end], dtype=float
    )
    eta, zeta, psi = np.asarray(theta[interaction_end:], dtype=float)
    return field_coeffs, beta, interaction_coeffs, eta, zeta, psi


def summary_metrics(
    est_theta: np.ndarray,
    true_theta: np.ndarray,
    field_basis: np.ndarray,
    interaction_basis: np.ndarray,
) -> dict[str, float]:
    """Compute reconstruction metrics for fitted parameters, fields, and interactions."""
    n_field = field_basis.shape[0]
    n_interaction = interaction_basis.shape[0]
    est_field_coeffs, _, est_interaction_coeffs, _, _, _ = unpack_theta(
        est_theta, n_field, n_interaction
    )
    true_field_coeffs, _, true_interaction_coeffs, _, _, _ = unpack_theta(
        true_theta, n_field, n_interaction
    )
    est_field = compose_field(est_field_coeffs, field_basis)
    true_field = compose_field(true_field_coeffs, field_basis)
    est_interaction = compose_interaction_matrix(
        est_interaction_coeffs, interaction_basis
    )
    true_interaction = compose_interaction_matrix(
        true_interaction_coeffs, interaction_basis
    )
    return {
        "field_rmse": float(np.sqrt(np.mean((est_field - true_field) ** 2))),
        "interaction_fro_error": float(
            np.linalg.norm(est_interaction - true_interaction, ord="fro")
        ),
        "parameter_rmse": float(np.sqrt(np.mean((est_theta - true_theta) ** 2))),
    }
