"""Posterior-predictive summaries, manifest refresh, ranking, and CSV reporting."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.t0_config_utils import load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows, write_csv, write_csv_rows
from utils.t2_summary_statistics import (
    finite_scalar_summary,
    finite_vector_summaries,
)
from utils.t6_posterior_predictive_summary import (
    POSTERIOR_PREDICTIVE_MANIFEST_NAME,
    manifest_row_from_metadata,
)
from utils.t8_output_writers import _as_float, _metric_or_inf

COUNTERFACTUAL_SUMMARY_ROOT_NAME = "counterfactual_summaries"
POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME = "posterior_predictive_summaries"
POSTERIOR_PREDICTIVE_PER_EXPERIMENT_SUMMARY_NAME = "posterior_predictive_summary.csv"
POSTERIOR_PREDICTIVE_WINNERS_SUMMARY_NAME = (
    "best_posterior_predictive_by_experiment.csv"
)

PER_EXPERIMENT_COLUMNS = [
    "experiment_name",
    "descriptor",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "latent_rank",
    "rank_in_experiment",
    "is_best",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "output_path",
]
WINNER_COLUMNS = [
    "experiment_name",
    "descriptor",
    "intervention_source",
    "graph_source",
    "N",
    "T",
    "s",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "latent_rank",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "output_path",
]
MANIFEST_COLUMNS = [
    "experiment_name",
    "experiment_slug",
    "descriptor",
    "experiment_path",
    "intervention_source",
    "graph_source",
    "N",
    "T",
    "s",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "target_intervention_source",
    "target_intervention_name",
    "target_intervention_slug",
    "latent_rank",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "seed",
    "output_path",
]
INTERVENTION_SUMMARY_COLUMNS = [
    "source_type",
    "source_name",
    "source_slug",
    "run_name",
    "run_slug",
    "num_samples",
    "gibbs_sweeps",
    "s",
    "overall_mean_magnetization_mean",
    "overall_mean_magnetization_std",
    "overall_mean_magnetization_q025",
    "overall_mean_magnetization_q500",
    "overall_mean_magnetization_q975",
    "post_intervention_mean_magnetization_mean",
    "post_intervention_mean_magnetization_std",
    "post_intervention_mean_magnetization_q025",
    "post_intervention_mean_magnetization_q500",
    "post_intervention_mean_magnetization_q975",
    "truth_overall_mean_magnetization_abs_error",
    "truth_post_intervention_mean_magnetization_abs_error",
    "truth_overall_mean_in_95_interval",
    "truth_post_intervention_mean_in_95_interval",
    "truth_unit_mean_abs_error_mean",
    "truth_unit_mean_squared_error_mean",
    "truth_unit_mean_rmse",
    "truth_unit_mean_max_abs_error",
    "truth_unit_mean_correlation",
    "truth_unit_mean_95_interval_coverage_rate",
    "truth_time_mean_abs_error_mean",
    "truth_time_mean_squared_error_mean",
    "truth_time_mean_rmse",
    "truth_time_mean_max_abs_error",
    "truth_time_mean_correlation",
    "truth_time_mean_95_interval_coverage_rate",
    "truth_rank_in_run",
    "truth_is_best",
]

_INTERVENTION_METRIC_COLUMNS = [
    "overall_mean_magnetization_mean",
    "overall_mean_magnetization_std",
    "overall_mean_magnetization_q025",
    "overall_mean_magnetization_q500",
    "overall_mean_magnetization_q975",
    "post_intervention_mean_magnetization_mean",
    "post_intervention_mean_magnetization_std",
    "post_intervention_mean_magnetization_q025",
    "post_intervention_mean_magnetization_q500",
    "post_intervention_mean_magnetization_q975",
]
_TRUTH_COMPARISON_COLUMNS = [
    "truth_overall_mean_magnetization_abs_error",
    "truth_post_intervention_mean_magnetization_abs_error",
    "truth_overall_mean_in_95_interval",
    "truth_post_intervention_mean_in_95_interval",
    "truth_unit_mean_abs_error_mean",
    "truth_unit_mean_squared_error_mean",
    "truth_unit_mean_rmse",
    "truth_unit_mean_max_abs_error",
    "truth_unit_mean_correlation",
    "truth_unit_mean_95_interval_coverage_rate",
    "truth_time_mean_abs_error_mean",
    "truth_time_mean_squared_error_mean",
    "truth_time_mean_rmse",
    "truth_time_mean_max_abs_error",
    "truth_time_mean_correlation",
    "truth_time_mean_95_interval_coverage_rate",
]
_TRUTH_RANKING_COLUMNS = [
    "truth_rank_in_run",
    "truth_is_best",
]


def summarize_observed_mean_statistics(
    observed_summary: dict[str, np.ndarray | float],
    sample_summaries: dict[str, np.ndarray],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, float],
]:
    mean_rows: list[dict[str, object]] = []
    scalar_metrics: dict[str, float] = {}
    for key, metric_name in [
        ("overall_mean_magnetization", "overall_mean_abs_error"),
        (
            "post_intervention_mean_magnetization",
            "post_intervention_mean_abs_error",
        ),
    ]:
        observed_value = float(observed_summary[key])
        sample_values = np.asarray(sample_summaries[key], dtype=float)
        row = {"statistic": key}
        row.update(finite_scalar_summary(sample_values, observed_value=observed_value))
        mean_rows.append(row)
        scalar_metrics[metric_name] = (
            math.nan if row["abs_error"] == "" else float(row["abs_error"])
        )

    unit_rows, unit_metrics = finite_vector_summaries(
        np.asarray(observed_summary["unit_mean_magnetization"], dtype=float),
        np.asarray(sample_summaries["unit_mean_magnetization"], dtype=float),
        index_name="unit_index",
    )
    time_rows, time_metrics = finite_vector_summaries(
        np.asarray(observed_summary["time_mean_magnetization"], dtype=float),
        np.asarray(sample_summaries["time_mean_magnetization"], dtype=float),
        index_name="time_index",
    )
    summary = {
        **scalar_metrics,
        "unit_mean_abs_error_mean": unit_metrics["abs_error_mean"],
        "unit_mean_rmse": unit_metrics["rmse"],
        "unit_mean_max_abs_error": unit_metrics["max_abs_error"],
        "unit_mean_95_interval_coverage_rate": unit_metrics["coverage_rate"],
        "time_mean_abs_error_mean": time_metrics["abs_error_mean"],
        "time_mean_rmse": time_metrics["rmse"],
        "time_mean_max_abs_error": time_metrics["max_abs_error"],
        "time_mean_95_interval_coverage_rate": time_metrics["coverage_rate"],
    }
    return mean_rows, unit_rows, time_rows, summary


def summarize_predictive_statistics(
    observed_stats: dict[str, float | None],
    simulated_stats: list[dict[str, float | None]],
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    abs_zscores: list[float] = []
    covered: list[float] = []
    for stat_name, observed_value in observed_stats.items():
        sample_values = np.asarray(
            [
                stat[stat_name]
                for stat in simulated_stats
                if stat.get(stat_name) is not None
            ],
            dtype=float,
        )
        if observed_value is None or sample_values.size == 0:
            continue
        sample_mean = float(np.mean(sample_values))
        sample_std = float(np.std(sample_values, ddof=0))
        if sample_std < 1e-12:
            if abs(float(observed_value) - sample_mean) < 1e-12:
                z_score = 0.0
            else:
                z_score = math.copysign(math.inf, float(observed_value) - sample_mean)
        else:
            z_score = (float(observed_value) - sample_mean) / sample_std
        q025, q500, q975 = np.quantile(sample_values, [0.025, 0.5, 0.975])
        left_tail = float(np.mean(sample_values <= float(observed_value)))
        right_tail = float(np.mean(sample_values >= float(observed_value)))
        tail_probability = min(1.0, 2.0 * min(left_tail, right_tail))
        in_interval = float(q025 <= float(observed_value) <= q975)
        rows.append(
            {
                "statistic": stat_name,
                "observed_value": float(observed_value),
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "z_score": z_score,
                "tail_probability": tail_probability,
                "q025": float(q025),
                "q500": float(q500),
                "q975": float(q975),
                "in_95_interval": bool(in_interval),
            }
        )
        abs_zscores.append(abs(z_score))
        covered.append(in_interval)

    summary = {
        "mean_abs_zscore": float(np.mean(abs_zscores)) if abs_zscores else math.inf,
        "max_abs_zscore": float(np.max(abs_zscores)) if abs_zscores else math.inf,
        "coverage_rate": float(np.mean(covered)) if covered else 0.0,
        "num_statistics": len(rows),
    }
    return rows, summary


def collect_predictive_rows(manifest_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in read_csv_rows(manifest_path):
        target_intervention_source = (
            str(
                manifest_row.get("target_intervention_source", "observed_experiment")
            ).strip()
            or "observed_experiment"
        )
        if target_intervention_source != "observed_experiment":
            continue
        row: dict[str, object] = dict(manifest_row)
        for key in [
            "mean_abs_zscore",
            "max_abs_zscore",
            "coverage_rate",
        ]:
            row[key] = _as_float(manifest_row.get(key))
        for key in [
            "N",
            "T",
            "s",
            "latent_rank",
            "num_statistics",
            "num_samples",
            "gibbs_sweeps",
        ]:
            value = manifest_row.get(key)
            row[key] = int(value) if value not in (None, "") else ""
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("run_name", "")),
            str(row.get("source_name", "")),
        )
    )
    return rows


def collect_manifest_rows_from_outputs(
    generation_manifest_path: str | Path,
) -> tuple[Path, list[dict[str, object]]]:
    manifest_root = Path(generation_manifest_path).resolve().parent
    rows: list[dict[str, object]] = []
    for experiment_row in read_csv_rows(generation_manifest_path):
        experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
        observed_metadata_paths = experiment_root.glob(
            "posterior_predictive/*/*/posterior_predictive_metadata.yaml"
        )
        counterfactual_metadata_paths = experiment_root.glob(
            "counterfactual/*/*/*/counterfactual_metadata.yaml"
        )
        for metadata_path in list(observed_metadata_paths) + list(counterfactual_metadata_paths):
            metadata = load_yaml_mapping(metadata_path)
            rows.append(
                manifest_row_from_metadata(
                    experiment_row,
                    metadata,
                    metadata_path.parent,
                )
            )
    rows.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("target_intervention_source", "")),
            str(row.get("target_intervention_name", "")),
            str(row.get("run_name", "")),
            str(row.get("source_name", "")),
        )
    )
    manifest_path = manifest_root / POSTERIOR_PREDICTIVE_MANIFEST_NAME
    write_csv_rows(
        manifest_path,
        [{column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in rows],
    )
    return manifest_path, rows


def ranking_key(row: dict[str, object]) -> tuple[float, float, str, str]:
    return (
        _metric_or_inf(row.get("mean_abs_zscore")),
        _metric_or_inf(row.get("max_abs_zscore")),
        str(row.get("run_name", "")),
        str(row.get("source_name", "")),
    )


def _read_magnetization_stats(
    output_path: Path, intervention_source: str
) -> dict[str, float | None]:
    if intervention_source == "observed_experiment":
        csv_path = output_path / "posterior_predictive_stats.csv"
    else:
        csv_path = output_path / "counterfactual_summary.csv"

    if not csv_path.exists():
        return {col: None for col in _INTERVENTION_METRIC_COLUMNS}

    stats: dict[str, float | None] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            statistic = str(row.get("statistic", "")).strip()
            if statistic == "overall_mean_magnetization":
                stats["overall_mean_magnetization_mean"] = _as_float(row.get("sample_mean"))
                stats["overall_mean_magnetization_std"] = _as_float(row.get("sample_std"))
                stats["overall_mean_magnetization_q025"] = _as_float(row.get("q025"))
                stats["overall_mean_magnetization_q500"] = _as_float(row.get("q500"))
                stats["overall_mean_magnetization_q975"] = _as_float(row.get("q975"))
            elif statistic == "post_intervention_mean_magnetization":
                stats["post_intervention_mean_magnetization_mean"] = _as_float(row.get("sample_mean"))
                stats["post_intervention_mean_magnetization_std"] = _as_float(row.get("sample_std"))
                stats["post_intervention_mean_magnetization_q025"] = _as_float(row.get("q025"))
                stats["post_intervention_mean_magnetization_q500"] = _as_float(row.get("q500"))
                stats["post_intervention_mean_magnetization_q975"] = _as_float(row.get("q975"))

    for col in _INTERVENTION_METRIC_COLUMNS:
        if col not in stats:
            stats[col] = None
    return stats


def _read_counterfactual_unit_summary(
    output_path: Path,
    intervention_source: str,
) -> dict[str, np.ndarray | None]:
    if intervention_source == "observed_experiment":
        return {
            "unit_mean_sample_mean": None,
            "unit_mean_q025": None,
            "unit_mean_q975": None,
        }
    csv_path = output_path / "counterfactual_unit_summary.csv"
    if not csv_path.exists():
        return {
            "unit_mean_sample_mean": None,
            "unit_mean_q025": None,
            "unit_mean_q975": None,
        }

    sample_means: list[float] = []
    q025_values: list[float] = []
    q975_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_mean = _as_float(row.get("sample_mean"))
            q025 = _as_float(row.get("q025"))
            q975 = _as_float(row.get("q975"))
            sample_means.append(np.nan if sample_mean is None else float(sample_mean))
            q025_values.append(np.nan if q025 is None else float(q025))
            q975_values.append(np.nan if q975 is None else float(q975))
    return {
        "unit_mean_sample_mean": np.asarray(sample_means, dtype=float),
        "unit_mean_q025": np.asarray(q025_values, dtype=float),
        "unit_mean_q975": np.asarray(q975_values, dtype=float),
    }


def _read_counterfactual_time_summary(
    output_path: Path,
    intervention_source: str,
) -> dict[str, np.ndarray | None]:
    if intervention_source == "observed_experiment":
        return {
            "time_mean_sample_mean": None,
            "time_mean_q025": None,
            "time_mean_q975": None,
        }
    csv_path = output_path / "counterfactual_time_summary.csv"
    if not csv_path.exists():
        return {
            "time_mean_sample_mean": None,
            "time_mean_q025": None,
            "time_mean_q975": None,
        }

    sample_means: list[float] = []
    q025_values: list[float] = []
    q975_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_mean = _as_float(row.get("sample_mean"))
            q025 = _as_float(row.get("q025"))
            q975 = _as_float(row.get("q975"))
            sample_means.append(np.nan if sample_mean is None else float(sample_mean))
            q025_values.append(np.nan if q025 is None else float(q025))
            q975_values.append(np.nan if q975 is None else float(q975))
    return {
        "time_mean_sample_mean": np.asarray(sample_means, dtype=float),
        "time_mean_q025": np.asarray(q025_values, dtype=float),
        "time_mean_q975": np.asarray(q975_values, dtype=float),
    }


def _blank_truth_metrics() -> dict[str, object]:
    return {column: "" for column in [*_TRUTH_COMPARISON_COLUMNS, *_TRUTH_RANKING_COLUMNS]}


def _interval_contains(
    value: object,
    lower: object,
    upper: object,
) -> float | None:
    parsed_value = _as_float(value)
    parsed_lower = _as_float(lower)
    parsed_upper = _as_float(upper)
    if parsed_value is None or parsed_lower is None or parsed_upper is None:
        return None
    return float(parsed_lower <= parsed_value <= parsed_upper)


def _safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if not np.any(valid):
        return None
    left_valid = left[valid]
    right_valid = right[valid]
    left_centered = left_valid - float(np.mean(left_valid))
    right_centered = right_valid - float(np.mean(right_valid))
    left_scale = float(np.linalg.norm(left_centered))
    right_scale = float(np.linalg.norm(right_centered))
    if left_scale < 1e-12 or right_scale < 1e-12:
        return 1.0 if np.allclose(left_valid, right_valid, atol=1e-12, rtol=0.0) else 0.0
    return float((left_centered @ right_centered) / (left_scale * right_scale))


def _compute_truth_metrics(
    candidate_row: dict[str, object],
    truth_row: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "truth_overall_mean_magnetization_abs_error": None,
        "truth_post_intervention_mean_magnetization_abs_error": None,
        "truth_overall_mean_in_95_interval": None,
        "truth_post_intervention_mean_in_95_interval": None,
        "truth_unit_mean_abs_error_mean": None,
        "truth_unit_mean_squared_error_mean": None,
        "truth_unit_mean_rmse": None,
        "truth_unit_mean_max_abs_error": None,
        "truth_unit_mean_correlation": None,
        "truth_unit_mean_95_interval_coverage_rate": None,
        "truth_time_mean_abs_error_mean": None,
        "truth_time_mean_squared_error_mean": None,
        "truth_time_mean_rmse": None,
        "truth_time_mean_max_abs_error": None,
        "truth_time_mean_correlation": None,
        "truth_time_mean_95_interval_coverage_rate": None,
    }
    overall_mean = _as_float(candidate_row.get("overall_mean_magnetization_mean"))
    truth_overall_mean = _as_float(truth_row.get("overall_mean_magnetization_mean"))
    if overall_mean is not None and truth_overall_mean is not None:
        metrics["truth_overall_mean_magnetization_abs_error"] = abs(
            overall_mean - truth_overall_mean
        )
    metrics["truth_overall_mean_in_95_interval"] = _interval_contains(
        truth_row.get("overall_mean_magnetization_mean"),
        candidate_row.get("overall_mean_magnetization_q025"),
        candidate_row.get("overall_mean_magnetization_q975"),
    )

    post_mean = _as_float(candidate_row.get("post_intervention_mean_magnetization_mean"))
    truth_post_mean = _as_float(truth_row.get("post_intervention_mean_magnetization_mean"))
    if post_mean is not None and truth_post_mean is not None:
        metrics["truth_post_intervention_mean_magnetization_abs_error"] = abs(
            post_mean - truth_post_mean
        )
    metrics["truth_post_intervention_mean_in_95_interval"] = _interval_contains(
        truth_row.get("post_intervention_mean_magnetization_mean"),
        candidate_row.get("post_intervention_mean_magnetization_q025"),
        candidate_row.get("post_intervention_mean_magnetization_q975"),
    )

    candidate_unit_means = candidate_row.get("unit_mean_sample_mean")
    truth_unit_means = truth_row.get("unit_mean_sample_mean")
    if (
        isinstance(candidate_unit_means, np.ndarray)
        and isinstance(truth_unit_means, np.ndarray)
        and candidate_unit_means.shape == truth_unit_means.shape
    ):
        valid = np.isfinite(candidate_unit_means) & np.isfinite(truth_unit_means)
        if np.any(valid):
            abs_errors = np.abs(candidate_unit_means[valid] - truth_unit_means[valid])
            squared_errors = abs_errors**2
            metrics["truth_unit_mean_abs_error_mean"] = float(np.mean(abs_errors))
            metrics["truth_unit_mean_squared_error_mean"] = float(np.mean(squared_errors))
            metrics["truth_unit_mean_rmse"] = float(np.sqrt(np.mean(squared_errors)))
            metrics["truth_unit_mean_max_abs_error"] = float(np.max(abs_errors))
            metrics["truth_unit_mean_correlation"] = _safe_correlation(
                candidate_unit_means[valid],
                truth_unit_means[valid],
            )

    candidate_q025 = candidate_row.get("unit_mean_q025")
    candidate_q975 = candidate_row.get("unit_mean_q975")
    if (
        isinstance(candidate_q025, np.ndarray)
        and isinstance(candidate_q975, np.ndarray)
        and isinstance(truth_unit_means, np.ndarray)
        and candidate_q025.shape == candidate_q975.shape == truth_unit_means.shape
    ):
        valid = (
            np.isfinite(candidate_q025)
            & np.isfinite(candidate_q975)
            & np.isfinite(truth_unit_means)
        )
        if np.any(valid):
            covered = (
                (candidate_q025[valid] <= truth_unit_means[valid])
                & (truth_unit_means[valid] <= candidate_q975[valid])
            )
            metrics["truth_unit_mean_95_interval_coverage_rate"] = float(
                np.mean(covered)
            )

    candidate_time_means = candidate_row.get("time_mean_sample_mean")
    truth_time_means = truth_row.get("time_mean_sample_mean")
    if (
        isinstance(candidate_time_means, np.ndarray)
        and isinstance(truth_time_means, np.ndarray)
        and candidate_time_means.shape == truth_time_means.shape
    ):
        valid = np.isfinite(candidate_time_means) & np.isfinite(truth_time_means)
        if np.any(valid):
            abs_errors = np.abs(candidate_time_means[valid] - truth_time_means[valid])
            squared_errors = abs_errors**2
            metrics["truth_time_mean_abs_error_mean"] = float(np.mean(abs_errors))
            metrics["truth_time_mean_squared_error_mean"] = float(np.mean(squared_errors))
            metrics["truth_time_mean_rmse"] = float(np.sqrt(np.mean(squared_errors)))
            metrics["truth_time_mean_max_abs_error"] = float(np.max(abs_errors))
            metrics["truth_time_mean_correlation"] = _safe_correlation(
                candidate_time_means[valid],
                truth_time_means[valid],
            )

    candidate_time_q025 = candidate_row.get("time_mean_q025")
    candidate_time_q975 = candidate_row.get("time_mean_q975")
    if (
        isinstance(candidate_time_q025, np.ndarray)
        and isinstance(candidate_time_q975, np.ndarray)
        and isinstance(truth_time_means, np.ndarray)
        and candidate_time_q025.shape == candidate_time_q975.shape == truth_time_means.shape
    ):
        valid = (
            np.isfinite(candidate_time_q025)
            & np.isfinite(candidate_time_q975)
            & np.isfinite(truth_time_means)
        )
        if np.any(valid):
            covered = (
                (candidate_time_q025[valid] <= truth_time_means[valid])
                & (truth_time_means[valid] <= candidate_time_q975[valid])
            )
            metrics["truth_time_mean_95_interval_coverage_rate"] = float(
                np.mean(covered)
            )
    return metrics


def _truth_row_self_metrics(truth_row: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {
        "truth_overall_mean_magnetization_abs_error": 0.0,
        "truth_post_intervention_mean_magnetization_abs_error": None,
        "truth_overall_mean_in_95_interval": None,
        "truth_post_intervention_mean_in_95_interval": None,
        "truth_unit_mean_abs_error_mean": None,
        "truth_unit_mean_squared_error_mean": None,
        "truth_unit_mean_rmse": None,
        "truth_unit_mean_max_abs_error": None,
        "truth_unit_mean_correlation": None,
        "truth_unit_mean_95_interval_coverage_rate": None,
        "truth_time_mean_abs_error_mean": None,
        "truth_time_mean_squared_error_mean": None,
        "truth_time_mean_rmse": None,
        "truth_time_mean_max_abs_error": None,
        "truth_time_mean_correlation": None,
        "truth_time_mean_95_interval_coverage_rate": None,
    }
    if _as_float(truth_row.get("overall_mean_magnetization_mean")) is not None:
        metrics["truth_overall_mean_in_95_interval"] = 1.0
    if _as_float(truth_row.get("post_intervention_mean_magnetization_mean")) is not None:
        metrics["truth_post_intervention_mean_magnetization_abs_error"] = 0.0
        metrics["truth_post_intervention_mean_in_95_interval"] = 1.0
    truth_unit_means = truth_row.get("unit_mean_sample_mean")
    if isinstance(truth_unit_means, np.ndarray) and np.isfinite(truth_unit_means).any():
        metrics["truth_unit_mean_abs_error_mean"] = 0.0
        metrics["truth_unit_mean_squared_error_mean"] = 0.0
        metrics["truth_unit_mean_rmse"] = 0.0
        metrics["truth_unit_mean_max_abs_error"] = 0.0
        metrics["truth_unit_mean_correlation"] = 1.0
        metrics["truth_unit_mean_95_interval_coverage_rate"] = 1.0
    truth_time_means = truth_row.get("time_mean_sample_mean")
    if isinstance(truth_time_means, np.ndarray) and np.isfinite(truth_time_means).any():
        metrics["truth_time_mean_abs_error_mean"] = 0.0
        metrics["truth_time_mean_squared_error_mean"] = 0.0
        metrics["truth_time_mean_rmse"] = 0.0
        metrics["truth_time_mean_max_abs_error"] = 0.0
        metrics["truth_time_mean_correlation"] = 1.0
        metrics["truth_time_mean_95_interval_coverage_rate"] = 1.0
    return metrics


def _truth_ranking_key(row: dict[str, object]) -> tuple[float, float, float, float, str]:
    return (
        _metric_or_inf(row.get("truth_unit_mean_squared_error_mean")),
        _metric_or_inf(row.get("truth_unit_mean_max_abs_error")),
        _metric_or_inf(row.get("truth_overall_mean_magnetization_abs_error")),
        _metric_or_inf(row.get("truth_post_intervention_mean_magnetization_abs_error")),
        str(row.get("source_name", "")),
    )


def _apply_truth_metrics_and_ranking(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_run[str(row.get("run_slug", ""))].append(row)

    enriched_rows: list[dict[str, object]] = []
    for _, run_rows in rows_by_run.items():
        truth_rows = [row for row in run_rows if str(row.get("source_type", "")) == "truth"]
        truth_row = truth_rows[0] if len(truth_rows) == 1 else None
        can_compare = (
            truth_row is not None
            and str(truth_row.get("target_intervention_source", "")) == "saved_intervention"
        )
        comparable_rows: list[dict[str, object]] = []
        for row in run_rows:
            enriched = dict(row)
            enriched.update(_blank_truth_metrics())
            if can_compare:
                if str(enriched.get("source_type", "")) == "truth":
                    enriched.update(_truth_row_self_metrics(enriched))
                elif str(enriched.get("target_intervention_source", "")) == "saved_intervention":
                    enriched.update(_compute_truth_metrics(enriched, truth_row))
                    comparable_rows.append(enriched)
            enriched_rows.append(enriched)
        if comparable_rows:
            ordered = sorted(comparable_rows, key=_truth_ranking_key)
            for index, row in enumerate(ordered, start=1):
                row["truth_rank_in_run"] = index
                row["truth_is_best"] = index == 1
    return enriched_rows


def _build_intervention_row(manifest_row: dict[str, object]) -> dict[str, object]:
    output_path = Path(str(manifest_row.get("output_path", "")))
    intervention_source = str(manifest_row.get("target_intervention_source", ""))
    stats = _read_magnetization_stats(output_path, intervention_source)
    unit_stats = _read_counterfactual_unit_summary(output_path, intervention_source)
    time_stats = _read_counterfactual_time_summary(output_path, intervention_source)
    return {
        "source_type": manifest_row.get("source_type", ""),
        "source_name": manifest_row.get("source_name", ""),
        "source_slug": manifest_row.get("source_slug", ""),
        "run_name": manifest_row.get("run_name", ""),
        "run_slug": manifest_row.get("run_slug", ""),
        "num_samples": manifest_row.get("num_samples", ""),
        "gibbs_sweeps": manifest_row.get("gibbs_sweeps", ""),
        "s": manifest_row.get("s", ""),
        "target_intervention_source": intervention_source,
        **stats,
        **unit_stats,
        **time_stats,
    }


def rank_rows_within_experiment(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=ranking_key)
    ranked_rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        ranked = dict(row)
        ranked["rank_in_experiment"] = index
        ranked["is_best"] = index == 1
        ranked_rows.append(ranked)
    return ranked_rows


def group_and_rank_predictive_rows(
    rows: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_path"])].append(row)

    ranked_groups: dict[str, list[dict[str, object]]] = {}
    winners: list[dict[str, object]] = []
    for experiment_path, group_rows in grouped.items():
        ranked = rank_rows_within_experiment(group_rows)
        ranked_groups[experiment_path] = ranked
        winners.append(dict(ranked[0]))
    winners.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("source_name", "")),
        )
    )
    return ranked_groups, winners


def write_intervention_summaries(
    experiment_path: str | Path,
    rows: list[dict[str, object]],
    *,
    summary_dir_name: str = COUNTERFACTUAL_SUMMARY_ROOT_NAME,
) -> dict[str, dict[str, str]]:
    experiment_root = Path(experiment_path)
    summary_dir = experiment_root / str(summary_dir_name)
    summary_dir.mkdir(exist_ok=True)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        slug = str(row.get("target_intervention_slug", "unknown"))
        grouped[slug].append(row)

    outputs: dict[str, dict[str, str]] = {}
    for slug, group_rows in grouped.items():
        built_rows = [_build_intervention_row(r) for r in group_rows]
        built_rows = _apply_truth_metrics_and_ranking(built_rows)
        csv_path = summary_dir / f"{slug}.csv"
        write_csv(csv_path, built_rows, INTERVENTION_SUMMARY_COLUMNS)
        outputs[slug] = {"csv": str(csv_path)}
    return outputs


def write_per_experiment_summary(
    experiment_path: str | Path,
    rows: list[dict[str, object]],
) -> Path:
    experiment_root = Path(experiment_path)
    csv_path = (
        experiment_root
        / POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME
        / POSTERIOR_PREDICTIVE_PER_EXPERIMENT_SUMMARY_NAME
    )
    write_csv(csv_path, rows, PER_EXPERIMENT_COLUMNS)
    return csv_path


def write_cross_experiment_summary(
    manifest_path: str | Path,
    winner_rows: list[dict[str, object]],
) -> Path:
    manifest_root = Path(manifest_path).resolve().parent
    csv_path = (
        manifest_root
        / POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME
        / POSTERIOR_PREDICTIVE_WINNERS_SUMMARY_NAME
    )
    write_csv(csv_path, winner_rows, WINNER_COLUMNS)
    return csv_path


def write_posterior_predictive_reports(manifest_path: str | Path) -> dict[str, object]:
    rows = collect_predictive_rows(manifest_path)
    if not rows:
        raise ValueError(
            f"No posterior-predictive runs were found in manifest {manifest_path}."
        )

    ranked_groups, winners = group_and_rank_predictive_rows(rows)
    per_experiment_outputs: dict[str, dict[str, str]] = {}
    for experiment_path, ranked_rows in ranked_groups.items():
        csv_path = write_per_experiment_summary(experiment_path, ranked_rows)
        per_experiment_outputs[experiment_path] = {
            "csv": str(csv_path),
        }
    winners_csv = write_cross_experiment_summary(manifest_path, winners)
    return {
        "per_experiment": per_experiment_outputs,
        "winners_csv": str(winners_csv),
    }


def refresh_and_write_posterior_predictive_reports(
    generation_manifest_path: str | Path,
) -> dict[str, object]:
    manifest_path, all_rows = collect_manifest_rows_from_outputs(generation_manifest_path)
    outputs = {
        "manifest_path": str(manifest_path),
        "num_manifest_rows": len(all_rows),
    }

    by_experiment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_experiment[str(row["experiment_path"])].append(row)
    counterfactual_outputs: dict[str, dict[str, dict[str, str]]] = {}
    posterior_predictive_target_outputs: dict[str, dict[str, dict[str, str]]] = {}
    for exp_path, exp_rows in by_experiment.items():
        counterfactual_rows = [
            row
            for row in exp_rows
            if str(row.get("target_intervention_source", "")).strip()
            == "saved_intervention"
        ]
        observed_rows = [
            row
            for row in exp_rows
            if str(row.get("target_intervention_source", "")).strip()
            == "observed_experiment"
        ]
        counterfactual_outputs[exp_path] = write_intervention_summaries(
            exp_path,
            counterfactual_rows,
            summary_dir_name=COUNTERFACTUAL_SUMMARY_ROOT_NAME,
        )
        posterior_predictive_target_outputs[exp_path] = write_intervention_summaries(
            exp_path,
            observed_rows,
            summary_dir_name=POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME,
        )
    outputs["counterfactual_summaries"] = counterfactual_outputs
    outputs["posterior_predictive_summaries"] = posterior_predictive_target_outputs

    predictive_rows = collect_predictive_rows(manifest_path)
    if predictive_rows:
        outputs.update(write_posterior_predictive_reports(manifest_path))
    return outputs
