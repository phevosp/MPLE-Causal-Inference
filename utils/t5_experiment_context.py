"""Experiment panel context loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.t0_path_utils import io_path
from utils.t1_matrix_io import load_gamma_matrix
from utils.t6_intervention_utils import derive_post_intervention_steps, derive_pre_intervention_steps


def load_panel_context_from_artifacts(
    panel_path: str | Path,
    x0_path: str | Path,
    z0_path: str | Path,
) -> dict[str, object]:
    with np.load(io_path(panel_path), allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    x_0 = np.asarray(np.load(io_path(x0_path)), dtype=float)
    z_0 = np.asarray(np.load(io_path(z0_path)), dtype=float)

    return {
        "x": x,
        "z": z,
        "x_0": x_0,
        "z_0": z_0,
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "s": derive_pre_intervention_steps(z),
        "e": derive_post_intervention_steps(z),
    }


def load_experiment_panel_context(experiment_root: str | Path) -> dict[str, object]:
    experiment_path = Path(experiment_root)
    return load_panel_context_from_artifacts(
        experiment_path / "panel_data.npz",
        experiment_path / "x_0.npy",
        experiment_path / "z_0.npy",
    )
