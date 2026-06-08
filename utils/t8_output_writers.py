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


def _ensure_output_path(output_root: str | Path) -> Path:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _write_summary_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
) -> Path:
    write_csv(csv_path, rows, columns)
    return csv_path


def _write_sample_summaries_npz(
    output_path: Path,
    filename: str,
    sample_summaries: dict[str, np.ndarray],
) -> Path:
    sample_npz_path = output_path / filename
    np.savez(io_path(sample_npz_path), **sample_summaries)
    return sample_npz_path


def _finite_scalar_summary_rows(
    sample_summaries: dict[str, np.ndarray],
    statistic_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for statistic_name in statistic_names:
        row = {"statistic": statistic_name}
        row.update(
            finite_scalar_summary(
                np.asarray(sample_summaries[statistic_name], dtype=float)
            )
        )
        rows.append(row)
    return rows


def _finite_index_summary_rows(
    sample_values: np.ndarray,
    *,
    index_name: str,
) -> list[dict[str, object]]:
    values = np.asarray(sample_values, dtype=float)
    rows: list[dict[str, object]] = []
    for item_index in range(values.shape[1]):
        row = {index_name: int(item_index)}
        row.update(finite_scalar_summary(values[:, item_index]))
        rows.append(row)
    return rows


def write_predictive_stats_tables(
    output_root: str | Path,
    stat_rows: list[dict[str, object]],
) -> Path:
    output_path = _ensure_output_path(output_root)
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
    return _write_summary_csv(csv_path, stat_rows, columns)


def write_observed_predictive_summary_tables(
    output_root: str | Path,
    *,
    sample_summaries: dict[str, np.ndarray],
    mean_rows: list[dict[str, object]],
    unit_rows: list[dict[str, object]],
    time_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path, Path]:
    output_path = _ensure_output_path(output_root)
    sample_npz_path = _write_sample_summaries_npz(
        output_path,
        "posterior_predictive_sample_summaries.npz",
        sample_summaries,
    )
    mean_csv_path = output_path / "posterior_predictive_mean_summary.csv"
    unit_csv_path = output_path / "posterior_predictive_unit_summary.csv"
    time_csv_path = output_path / "posterior_predictive_time_summary.csv"
    _write_summary_csv(
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
    _write_summary_csv(
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
    _write_summary_csv(
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
    output_path = _ensure_output_path(output_root)
    sample_npz_path = _write_sample_summaries_npz(
        output_path,
        "counterfactual_sample_summaries.npz",
        sample_summaries,
    )
    summary_csv_path = output_path / "counterfactual_summary.csv"
    unit_csv_path = output_path / "counterfactual_unit_summary.csv"
    time_csv_path = output_path / "counterfactual_time_summary.csv"
    summary_rows = _finite_scalar_summary_rows(
        sample_summaries,
        [
            "overall_mean_magnetization",
            "post_intervention_mean_magnetization",
        ],
    )
    _write_summary_csv(
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

    unit_rows = _finite_index_summary_rows(
        sample_summaries["unit_mean_magnetization"],
        index_name="unit_index",
    )
    _write_summary_csv(
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

    time_rows = _finite_index_summary_rows(
        sample_summaries["time_mean_magnetization"],
        index_name="time_index",
    )
    _write_summary_csv(
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
