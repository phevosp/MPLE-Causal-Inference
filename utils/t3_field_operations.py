"""Field matrix operations (composition, scaling, projection, truncation)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from utils.t2_normalization import (
    latent_field_bound_norm,
    normalize_matrix_max_abs,
    normalize_matrix_by_max_abs_entry,
)
from utils.t3_model_artifacts import (
    FULL_FIELD_SERIALIZER_OPTIMIZER_MODES,
    ModelArtifacts,
    SpectralLowRankStructure,
    _DEGENERACY_THRESHOLD,
)


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


def zero_latent_field(n_nodes: int, t_steps: int) -> np.ndarray:
    return np.zeros((t_steps, n_nodes), dtype=float)


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


def compose_field_matrix_from_theta(
    theta_parts: dict[str, object], artifacts: ModelArtifacts
) -> np.ndarray:
    if artifacts.optimizer_mode in FULL_FIELD_SERIALIZER_OPTIMIZER_MODES:
        return np.asarray(theta_parts["field_matrix"], dtype=float)
    return compose_latent_field_matrix(
        theta_parts["node_factors"], theta_parts["time_factors"]
    )


def compose_realized_treatment_field_matrix(
    control_field_matrix: np.ndarray,
    treated_field_matrix: np.ndarray,
    treatment_matrix: np.ndarray,
) -> np.ndarray:
    """Merge untreated and treated surfaces slotwise under the observed treatment panel."""
    control = np.asarray(control_field_matrix, dtype=float)
    treated = np.asarray(treated_field_matrix, dtype=float)
    treatment = np.asarray(treatment_matrix, dtype=float)
    if control.shape != treated.shape or control.shape != treatment.shape:
        raise ValueError(
            "control_field_matrix, treated_field_matrix, and treatment_matrix must share a shape."
        )
    return np.where(treatment > 0.5, treated, control)


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
