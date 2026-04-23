"""Posterior-predictive simulation and statistics helpers."""

from __future__ import annotations

import math

import numpy as np

from data.synthetic_data_generation import simulate_outcomes_given_fixed_interventions
from loading_utils import OutcomeParameterBundle
from model_utils import compose_interaction_matrix, interaction_effect


def simulate_outcomes_for_bundle(
    bundle: OutcomeParameterBundle,
    *,
    x_0: np.ndarray,
    z: np.ndarray,
    gibbs_sweeps: int,
    seed: int,
) -> np.ndarray:
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
    }


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
