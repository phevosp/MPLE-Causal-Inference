"""Shared I/O utilities used across pipeline entry points and report scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse


def load_yaml_config(path: str | Path):
    return OmegaConf.load(Path(path))


def first_existing_path(*paths: str | Path) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of the expected paths: "
        + ", ".join(str(Path(path)) for path in paths)
    )


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


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _metric_or_inf(value: object) -> float:
    parsed = _as_float(value)
    return math.inf if parsed is None else parsed


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in columns} for row in rows]
        )


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(
    handle, rows: list[dict[str, object]], columns: list[str]
) -> None:
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        handle.write(
            "| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |\n"
        )
    handle.write("\n")
