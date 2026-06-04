"""Posterior predictive summary statistics reporting."""

from __future__ import annotations

import math

import numpy as np

from utils.t2_summary_statistics import finite_scalar_summary, finite_vector_summaries, mean_on_mask


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
