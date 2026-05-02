"""Shared I/O utilities used across pipeline entry points and report scripts."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse


def load_yaml_config(path: str | Path):
    return OmegaConf.load(io_path(path))


def io_path(path: str | Path) -> str:
    candidate = Path(path)
    if os.name == "nt":
        resolved = str(candidate.resolve())
        if not resolved.startswith("\\\\?\\"):
            return "\\\\?\\" + resolved
        return resolved
    if candidate.is_absolute():
        return str(candidate)
    return os.path.abspath(os.fspath(candidate))


def path_exists(path: str | Path) -> bool:
    return os.path.exists(io_path(path))


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
    with open(io_path(path), "w", encoding="utf-8", newline="") as handle:
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


def write_predictive_stats_tables(
    output_root: str | Path,
    stat_rows: list[dict[str, object]],
) -> Path:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "posterior_predictive_stats.csv"
    columns = [
        "statistic",
        "observed_value",
        "sample_mean",
        "sample_std",
        "z_score",
        "tail_probability",
        "q025",
        "q500",
        "q975",
        "in_95_interval",
    ]
    write_csv(csv_path, stat_rows, columns)
    return csv_path


def _finite_summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "sample_mean": "",
            "sample_std": "",
            "q025": "",
            "q500": "",
            "q975": "",
            "num_finite_samples": 0,
        }
    q025, q500, q975 = np.quantile(finite, [0.025, 0.5, 0.975])
    return {
        "sample_mean": float(np.mean(finite)),
        "sample_std": float(np.std(finite, ddof=0)),
        "q025": float(q025),
        "q500": float(q500),
        "q975": float(q975),
        "num_finite_samples": int(finite.size),
    }


def write_counterfactual_summary_tables(
    output_root: str | Path,
    *,
    sample_summaries: dict[str, np.ndarray],
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_npz_path = output_path / "counterfactual_sample_summaries.npz"
    summary_csv_path = output_path / "counterfactual_summary.csv"
    unit_csv_path = output_path / "counterfactual_unit_summary.csv"
    time_csv_path = output_path / "counterfactual_time_summary.csv"
    np.savez(io_path(sample_npz_path), **sample_summaries)

    summary_rows = []
    for key in [
        "overall_mean_magnetization",
        "post_intervention_mean_magnetization",
    ]:
        row = {"statistic": key}
        row.update(_finite_summary(np.asarray(sample_summaries[key], dtype=float)))
        summary_rows.append(row)
    write_csv(
        summary_csv_path,
        summary_rows,
        [
            "statistic",
            "sample_mean",
            "sample_std",
            "q025",
            "q500",
            "q975",
            "num_finite_samples",
        ],
    )

    unit_values = np.asarray(sample_summaries["unit_mean_magnetization"], dtype=float)
    unit_rows = []
    for unit_index in range(unit_values.shape[1]):
        row = {"unit_index": int(unit_index)}
        row.update(_finite_summary(unit_values[:, unit_index]))
        unit_rows.append(row)
    write_csv(
        unit_csv_path,
        unit_rows,
        [
            "unit_index",
            "sample_mean",
            "sample_std",
            "q025",
            "q500",
            "q975",
            "num_finite_samples",
        ],
    )

    time_values = np.asarray(sample_summaries["time_mean_magnetization"], dtype=float)
    time_rows = []
    for time_index in range(time_values.shape[1]):
        row = {"time_index": int(time_index)}
        row.update(_finite_summary(time_values[:, time_index]))
        time_rows.append(row)
    write_csv(
        time_csv_path,
        time_rows,
        [
            "time_index",
            "sample_mean",
            "sample_std",
            "q025",
            "q500",
            "q975",
            "num_finite_samples",
        ],
    )
    return sample_npz_path, summary_csv_path, unit_csv_path, time_csv_path
