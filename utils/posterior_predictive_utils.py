"""Posterior-predictive simulation and statistics helpers."""

from __future__ import annotations

import math

import numpy as np

from data.synthetic_data_generation import simulate_outcomes_given_fixed_interventions
from utils.loading_utils import OutcomeParameterBundle
from utils.model_utils import compose_interaction_matrix, interaction_effect


def simulate_outcomes_for_bundle(
    bundle: OutcomeParameterBundle,
    *,
    x_0: np.ndarray,
    z: np.ndarray,
    gibbs_sweeps: int,
    seed: int,
) -> np.ndarray:
    """Simulate from the predictive model using the realized intervention panel.

    Fit-time beta masking is intentionally ignored here because it is only a
    parameter-estimation choice, not part of the generative model.
    """
    rng = np.random.default_rng(seed)
    interaction_matrix = compose_interaction_matrix(bundle.xi, bundle.gamma_matrix)
    return simulate_outcomes_given_fixed_interventions(
        x_0=np.asarray(x_0, dtype=float),
        z=np.asarray(z, dtype=float),
        field_matrix=np.asarray(bundle.field_matrix, dtype=float),
        interaction_matrix=interaction_matrix,
        beta=float(bundle.beta),
        eta=float(bundle.eta),
        rng=rng,
        gibbs_sweeps=int(gibbs_sweeps),
    )


def _graph_energy(x: np.ndarray, gamma_matrix) -> np.ndarray:
    interaction_x = interaction_effect(np.asarray(x, dtype=float), gamma_matrix)
    return np.sum(np.asarray(x, dtype=float) * interaction_x, axis=1) / x.shape[1]


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def compute_panel_statistics(
    x: np.ndarray,
    *,
    z: np.ndarray,
    x_0: np.ndarray,
    s: int,
    field_matrix: np.ndarray,
    gamma_matrix,
) -> dict[str, float | None]:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    field_matrix = np.asarray(field_matrix, dtype=float)
    prev_x = np.vstack([np.asarray(x_0, dtype=float), x[:-1, :]])
    graph_energy = _graph_energy(x, gamma_matrix)
    post_mask = slice(int(s), x.shape[0])
    pre_mask = slice(0, int(s))

    return {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": _mean_or_none(x[post_mask]),
        "intervention_alignment": float(np.mean(x * z)),
        "lag1_persistence": float(np.mean(x * prev_x)),
        "graph_interaction_energy": float(np.mean(graph_energy)),
        "field_alignment": float(np.mean(x * field_matrix)),
        "pre_intervention_alignment": _mean_or_none(
            (x[pre_mask] * z[pre_mask]).reshape(-1)
        ),
        "post_intervention_alignment": _mean_or_none(
            (x[post_mask] * z[post_mask]).reshape(-1)
        ),
        "pre_graph_interaction_energy": _mean_or_none(graph_energy[pre_mask]),
        "post_graph_interaction_energy": _mean_or_none(graph_energy[post_mask]),
    }


def compute_counterfactual_sample_summary(
    x: np.ndarray,
    *,
    s: int,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=float)
    post_value = float(np.mean(x[int(s) :, :])) if int(s) < x.shape[0] else math.nan
    return {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": post_value,
        "unit_mean_magnetization": np.mean(x, axis=0),
        "time_mean_magnetization": np.mean(x, axis=1),
    }


def compute_observed_sample_summary(
    x: np.ndarray,
    *,
    s: int,
) -> dict[str, np.ndarray | float]:
    """Compute draw-level mean magnetization summaries for observed-run evaluation."""
    return compute_counterfactual_sample_summary(x, s=s)


def _finite_scalar_summary(
    observed_value: float,
    sample_values: np.ndarray,
) -> dict[str, object]:
    observed = float(observed_value)
    finite = np.asarray(sample_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(observed) or finite.size == 0:
        return {
            "observed_value": observed if np.isfinite(observed) else "",
            "sample_mean": "",
            "sample_std": "",
            "abs_error": "",
            "q025": "",
            "q500": "",
            "q975": "",
            "in_95_interval": "",
            "num_finite_samples": int(finite.size),
        }
    q025, q500, q975 = np.quantile(finite, [0.025, 0.5, 0.975])
    sample_mean = float(np.mean(finite))
    return {
        "observed_value": observed,
        "sample_mean": sample_mean,
        "sample_std": float(np.std(finite, ddof=0)),
        "abs_error": abs(observed - sample_mean),
        "q025": float(q025),
        "q500": float(q500),
        "q975": float(q975),
        "in_95_interval": bool(float(q025) <= observed <= float(q975)),
        "num_finite_samples": int(finite.size),
    }


def _finite_vector_summaries(
    observed_values: np.ndarray,
    sample_values: np.ndarray,
    *,
    index_name: str,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    observed = np.asarray(observed_values, dtype=float).reshape(-1)
    samples = np.asarray(sample_values, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected a 2D sample array for {index_name} summaries, got shape {samples.shape}."
        )
    if samples.shape[1] != observed.shape[0]:
        raise ValueError(
            f"Observed {index_name} values have length {observed.shape[0]}, but sample array has "
            f"shape {samples.shape}."
        )

    rows: list[dict[str, object]] = []
    abs_errors: list[float] = []
    squared_errors: list[float] = []
    covered: list[float] = []
    for item_index in range(observed.shape[0]):
        row = {index_name: int(item_index)}
        row.update(
            _finite_scalar_summary(
                float(observed[item_index]),
                samples[:, item_index],
            )
        )
        if row["abs_error"] != "":
            abs_error = float(row["abs_error"])
            abs_errors.append(abs_error)
            squared_errors.append(abs_error**2)
        if row["in_95_interval"] != "":
            covered.append(float(bool(row["in_95_interval"])))
        row["squared_error"] = (
            ""
            if row["abs_error"] == ""
            else float(row["abs_error"]) ** 2
        )
        rows.append(row)

    aggregates = {
        "abs_error_mean": float(np.mean(abs_errors)) if abs_errors else math.nan,
        "rmse": float(np.sqrt(np.mean(squared_errors))) if squared_errors else math.nan,
        "max_abs_error": float(np.max(abs_errors)) if abs_errors else math.nan,
        "coverage_rate": float(np.mean(covered)) if covered else math.nan,
    }
    return rows, aggregates


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
        row.update(_finite_scalar_summary(observed_value, sample_values))
        mean_rows.append(row)
        scalar_metrics[metric_name] = (
            math.nan if row["abs_error"] == "" else float(row["abs_error"])
        )

    unit_rows, unit_metrics = _finite_vector_summaries(
        np.asarray(observed_summary["unit_mean_magnetization"], dtype=float),
        np.asarray(sample_summaries["unit_mean_magnetization"], dtype=float),
        index_name="unit_index",
    )
    time_rows, time_metrics = _finite_vector_summaries(
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

