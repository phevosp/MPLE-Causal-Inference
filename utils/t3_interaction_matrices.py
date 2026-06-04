"""Interaction matrix composition and application."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from utils.t2_normalization import interaction_matrix_infinity_norm


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
