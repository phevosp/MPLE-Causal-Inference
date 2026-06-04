"""Matrix loading and loss mask saving utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from utils.t0_path_utils import io_path


def load_gamma_matrix(data_folder: str | Path):
    """Load a graph adjacency matrix from either sparse (.npz) or dense (.npy) artifact."""
    data_path = Path(data_folder)
    gamma_sparse = data_path / "gamma_matrix_sparse.npz"
    gamma_dense = data_path / "gamma_matrix.npy"
    if gamma_sparse.exists():
        return sparse.load_npz(gamma_sparse).tocsr()
    if gamma_dense.exists():
        return np.load(gamma_dense, allow_pickle=False)
    raise FileNotFoundError(f"Missing gamma matrix artifact in {data_path}.")


def save_loss_mask(path: str | Path, loss_mask: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(io_path(output_path), np.asarray(loss_mask, dtype=bool))
    return output_path
