"""Normalization utilities for matrices, vectors, and graphs."""

from __future__ import annotations

import numpy as np
from scipy import sparse

_DEGENERACY_THRESHOLD = 1e-12


def interaction_matrix_infinity_norm(matrix) -> float:
    """Compute the infinity norm of an interaction matrix (dense or sparse)."""
    if sparse.issparse(matrix):
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        return float(row_sums.max()) if row_sums.size else 0.0
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def latent_field_bound_norm(field_matrix: np.ndarray) -> float:
    """Maximum absolute entry used for the latent-field B constraint."""
    field_matrix = np.asarray(field_matrix, dtype=float)
    if field_matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(field_matrix)))


def _normalize_dense_graph(gamma_matrix: np.ndarray) -> np.ndarray:
    """Normalize a dense gamma matrix to infinity norm one."""
    gamma_matrix = np.asarray(gamma_matrix, dtype=float)
    gamma_matrix = (gamma_matrix + gamma_matrix.T) / 2.0
    np.fill_diagonal(gamma_matrix, 0.0)
    norm = float(np.linalg.norm(gamma_matrix, ord=np.inf))
    if norm < _DEGENERACY_THRESHOLD:
        return np.zeros_like(gamma_matrix)
    return gamma_matrix / norm


def _normalize_sparse_graph(gamma_matrix) -> sparse.csr_matrix:
    """Normalize a sparse gamma matrix to infinity norm one."""
    normalized = sparse.csr_matrix(gamma_matrix, dtype=float)
    normalized = ((normalized + normalized.T) * 0.5).tocsr()
    normalized.setdiag(0.0)
    normalized.eliminate_zeros()
    norm = interaction_matrix_infinity_norm(normalized)
    if norm < _DEGENERACY_THRESHOLD:
        return sparse.csr_matrix(normalized.shape, dtype=float)
    return normalized.multiply(1.0 / norm).tocsr()


def normalize_known_graph(gamma_matrix):
    """Normalize a known adjacency matrix (dense or sparse) to infinity norm one."""
    if sparse.issparse(gamma_matrix):
        return _normalize_sparse_graph(gamma_matrix)
    return _normalize_dense_graph(np.asarray(gamma_matrix, dtype=float))


def validate_graph_infinity_norm(gamma_matrix, tol: float = 1e-8) -> None:
    """Validate that a graph has infinity norm one."""
    gamma_norm = interaction_matrix_infinity_norm(gamma_matrix)
    if gamma_norm < tol:
        raise ValueError("Known graph is degenerate.")
    if not np.isclose(gamma_norm, 1.0, atol=tol, rtol=0.0):
        raise ValueError("The known graph must have infinity norm one.")


def normalize_matrix_max_abs(
    matrix: np.ndarray,
    *,
    max_abs: float = 1.0,
) -> np.ndarray:
    """Normalize a matrix by its maximum absolute entry to a target bound."""
    matrix = np.asarray(matrix, dtype=float)
    if max_abs < 0.0:
        raise ValueError("max_abs must be nonnegative.")
    current = latent_field_bound_norm(matrix)
    if current < _DEGENERACY_THRESHOLD or max_abs == 0.0:
        return np.zeros_like(matrix)
    return matrix * (float(max_abs) / current)


def normalize_matrix_by_max_abs_entry(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix by dividing by its maximum absolute entry, returning zeros if degenerate."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.zeros_like(matrix)
    max_abs = float(np.max(np.abs(matrix)))
    if max_abs < _DEGENERACY_THRESHOLD:
        return np.zeros_like(matrix)
    return matrix / max_abs
