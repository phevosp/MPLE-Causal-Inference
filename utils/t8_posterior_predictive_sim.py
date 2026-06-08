"""Outcome simulation and panel statistics computation."""

from __future__ import annotations

import math

import numpy as np

from data.synthetic_data_generation import simulate_outcomes_given_fixed_interventions
from utils.t2_summary_statistics import mean_on_mask
from utils.t3_interaction_matrices import compose_interaction_matrix, interaction_effect
from utils.t5_parameter_bundles import OutcomeParameterBundle


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
    post_mask_bool = np.zeros(x.shape, dtype=bool)
    post_mask_bool[int(s) :, :] = True
    pre_mask_bool = np.zeros(x.shape, dtype=bool)
    pre_mask_bool[0 : int(s), :] = True
    post_time_mask = np.zeros(x.shape[0], dtype=bool)
    post_time_mask[int(s) :] = True
    pre_time_mask = np.zeros(x.shape[0], dtype=bool)
    pre_time_mask[0 : int(s)] = True

    return {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": mean_on_mask(x, post_mask_bool),
        "intervention_alignment": float(np.mean(x * z)),
        "lag1_persistence": float(np.mean(x * prev_x)),
        "graph_interaction_energy": float(np.mean(graph_energy)),
        "field_alignment": float(np.mean(x * field_matrix)),
        "pre_intervention_alignment": mean_on_mask((x * z).reshape(-1), pre_mask_bool.reshape(-1)),
        "post_intervention_alignment": mean_on_mask((x * z).reshape(-1), post_mask_bool.reshape(-1)),
        "pre_graph_interaction_energy": mean_on_mask(graph_energy, pre_time_mask),
        "post_graph_interaction_energy": mean_on_mask(graph_energy, post_time_mask),
    }


def compute_counterfactual_sample_summary(
    x: np.ndarray,
    *,
    s: int,
) -> dict[str, np.ndarray | float]:
    """Compute draw-level mean magnetization summaries for one simulated panel."""
    x = np.asarray(x, dtype=float)
    post_value = float(np.mean(x[int(s) :, :])) if int(s) < x.shape[0] else math.nan
    return {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": post_value,
        "unit_mean_magnetization": np.mean(x, axis=0),
        "time_mean_magnetization": np.mean(x, axis=1),
    }
