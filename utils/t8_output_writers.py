"""Output table writers for posterior predictive results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from utils.t0_csv_utils import write_csv
from utils.t0_path_utils import io_path
from utils.t2_summary_statistics import finite_scalar_summary


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _metric_or_inf(value: object) -> float:
    parsed = _as_float(value)
    return math.inf if parsed is None else parsed


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


def write_observed_predictive_summary_tables(
    output_root: str | Path,
    *,
    sample_summaries: dict[str, np.ndarray],
    mean_rows: list[dict[str, object]],
    unit_rows: list[dict[str, object]],
    time_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_npz_path = output_path / "posterior_predictive_sample_summaries.npz"
    mean_csv_path = output_path / "posterior_predictive_mean_summary.csv"
    unit_csv_path = output_path / "posterior_predictive_unit_summary.csv"
    time_csv_path = output_path / "posterior_predictive_time_summary.csv"
    np.savez(io_path(sample_npz_path), **sample_summaries)
    write_csv(
        mean_csv_path,
        mean_rows,
        [
            "statistic",
            "observed_value",
            "sample_mean",
            "sample_std",
            "abs_error",
            "q025",
            "q500",
            "q975",
            "in_95_interval",
            "num_finite_samples",
        ],
    )
    write_csv(
        unit_csv_path,
        unit_rows,
        [
            "unit_index",
            "observed_value",
            "sample_mean",
            "sample_std",
            "abs_error",
            "squared_error",
            "q025",
            "q500",
            "q975",
            "in_95_interval",
            "num_finite_samples",
        ],
    )
    write_csv(
        time_csv_path,
        time_rows,
        [
            "time_index",
            "observed_value",
            "sample_mean",
            "sample_std",
            "abs_error",
            "squared_error",
            "q025",
            "q500",
            "q975",
            "in_95_interval",
            "num_finite_samples",
        ],
    )
    return sample_npz_path, mean_csv_path, unit_csv_path, time_csv_path


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
        row.update(finite_scalar_summary(np.asarray(sample_summaries[key], dtype=float)))
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
        row.update(finite_scalar_summary(unit_values[:, unit_index]))
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
        row.update(finite_scalar_summary(time_values[:, time_index]))
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
