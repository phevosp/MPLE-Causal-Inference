"""Shared validation metric helpers for CV and single-fold validation flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from data.synthetic_data_generation import spin_sample_from_field
from loading_utils import OutcomeParameterBundle, load_experiment_panel_context, load_fit_parameter_bundle
from model_utils import compose_interaction_matrix, interaction_effect
from mple import evaluate_mple_loss_from_parts


DEFAULT_VALIDATION_SAMPLING = {
    "num_samples": 16,
    "gibbs_sweeps": 10,
    "seed": 0,
}
ECE_NUM_BINS = 10

FULL_VALIDATION_METRIC_SPECS = (
    ("validation_loss", "num_validation_slots"),
    ("validation_brier_score", "num_validation_slots"),
    ("validation_ece", "num_validation_slots"),
    ("validation_mean_magnetization_abs_diff", "num_validation_slots"),
)
POST_S_VALIDATION_METRIC_SPECS = (
    ("post_s_validation_loss", "num_post_s_validation_slots"),
    ("post_s_validation_brier_score", "num_post_s_validation_slots"),
    ("post_s_validation_ece", "num_post_s_validation_slots"),
    ("post_s_validation_mean_magnetization_abs_diff", "num_post_s_validation_slots"),
)


def resolve_validation_sampling(config: dict[str, Any] | None) -> dict[str, int]:
    resolved = dict(DEFAULT_VALIDATION_SAMPLING)
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return {
        "num_samples": int(resolved["num_samples"]),
        "gibbs_sweeps": int(resolved["gibbs_sweeps"]),
        "seed": int(resolved["seed"]),
    }


def time_window_mask(*, t_steps: int, n_nodes: int, start_t: int = 0) -> np.ndarray:
    mask = np.zeros((int(t_steps), int(n_nodes)), dtype=bool)
    if int(start_t) < int(t_steps):
        mask[int(start_t) :, :] = True
    return mask


def validation_brier_score(
    *,
    x: np.ndarray,
    h_x: np.ndarray,
    loss_mask: np.ndarray,
) -> float:
    x_array = np.asarray(x, dtype=float)
    h_array = np.asarray(h_x, dtype=float)
    mask = np.asarray(loss_mask, dtype=bool)
    if x_array.shape != h_array.shape or x_array.shape != mask.shape:
        raise ValueError(
            "x, h_x, and loss_mask must all have the same shape for Brier evaluation."
        )
    if not np.any(mask):
        raise ValueError("loss_mask must contain at least one active entry.")
    observed_positive = (x_array + 1.0) / 2.0
    predicted_positive = (1.0 + np.tanh(h_array)) / 2.0
    squared_error = (observed_positive - predicted_positive) ** 2
    return float(np.mean(squared_error[mask]))


def validation_expected_calibration_error(
    *,
    x: np.ndarray,
    h_x: np.ndarray,
    loss_mask: np.ndarray,
    num_bins: int = ECE_NUM_BINS,
) -> float:
    x_array = np.asarray(x, dtype=float)
    h_array = np.asarray(h_x, dtype=float)
    mask = np.asarray(loss_mask, dtype=bool)
    if x_array.shape != h_array.shape or x_array.shape != mask.shape:
        raise ValueError(
            "x, h_x, and loss_mask must all have the same shape for ECE evaluation."
        )
    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive for ECE evaluation.")
    if not np.any(mask):
        raise ValueError("loss_mask must contain at least one active entry.")
    observed_positive = ((x_array + 1.0) / 2.0)[mask]
    predicted_positive = ((1.0 + np.tanh(h_array)) / 2.0)[mask]
    bin_indices = np.minimum(
        np.floor(predicted_positive * float(num_bins)).astype(int),
        int(num_bins - 1),
    )
    total_count = float(predicted_positive.size)
    ece = 0.0
    for bin_index in range(int(num_bins)):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        bin_fraction = float(np.count_nonzero(in_bin)) / total_count
        empirical_rate = float(np.mean(observed_positive[in_bin]))
        mean_predicted_probability = float(np.mean(predicted_positive[in_bin]))
        ece += bin_fraction * abs(empirical_rate - mean_predicted_probability)
    return float(ece)


def _mean_on_mask(x: np.ndarray, mask: np.ndarray) -> float | None:
    x_array = np.asarray(x, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if x_array.shape != mask_array.shape:
        raise ValueError("x and mask must have the same shape when averaging over a mask.")
    if not np.any(mask_array):
        return None
    return float(np.mean(x_array[mask_array]))


def _interaction_column(interaction_matrix, column_index: int) -> np.ndarray:
    if sparse.issparse(interaction_matrix):
        return np.asarray(interaction_matrix[:, int(column_index)].toarray()).reshape(-1)
    return np.asarray(interaction_matrix[:, int(column_index)], dtype=float).reshape(-1)


def _sample_validation_time_step(
    observed_x_t: np.ndarray,
    *,
    x_prev: np.ndarray,
    z_t: np.ndarray,
    field_t: np.ndarray,
    interaction_matrix,
    beta: float,
    eta: float,
    rng: np.random.Generator,
    validation_nodes: np.ndarray,
    gibbs_sweeps: int,
) -> np.ndarray:
    x_t = np.asarray(observed_x_t, dtype=float).copy()
    active_validation_nodes = np.asarray(validation_nodes, dtype=int)
    if active_validation_nodes.size == 0:
        return x_t
    interaction_x_t = np.asarray(interaction_matrix @ x_t, dtype=float).reshape(-1)
    beta_feature = np.asarray(z_t, dtype=float)
    for _ in range(int(gibbs_sweeps)):
        for node_index in rng.permutation(active_validation_nodes):
            old_x_i = float(x_t[node_index])
            h_x = (
                float(field_t[node_index])
                + float(beta) * float(beta_feature[node_index])
                + float(eta) * float(x_prev[node_index])
                + float(interaction_x_t[node_index])
            )
            x_t[node_index] = spin_sample_from_field(h_x, rng)
            delta = float(x_t[node_index] - old_x_i)
            if abs(delta) > 0.0:
                interaction_x_t = interaction_x_t + delta * _interaction_column(
                    interaction_matrix,
                    int(node_index),
                )
    return x_t


def sample_validation_panel_conditional(
    *,
    panel_context: dict[str, object],
    bundle: OutcomeParameterBundle,
    validation_loss_mask: np.ndarray,
    gibbs_sweeps: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(panel_context["x"], dtype=float)
    z = np.asarray(panel_context["z"], dtype=float)
    x_0 = np.asarray(panel_context["x_0"], dtype=float)
    validation_mask = np.asarray(validation_loss_mask, dtype=bool)
    if x.shape != validation_mask.shape:
        raise ValueError(
            f"validation_loss_mask shape {validation_mask.shape} does not match x shape {x.shape}."
        )
    interaction_matrix = compose_interaction_matrix(float(bundle.xi), bundle.gamma_matrix)
    sampled_x = np.asarray(x, dtype=float).copy()
    rng = np.random.default_rng(int(seed))
    x_prev = x_0
    for time_index in range(x.shape[0]):
        validation_nodes = np.flatnonzero(validation_mask[time_index, :]).astype(int)
        sampled_x[time_index, :] = _sample_validation_time_step(
            x[time_index, :],
            x_prev=np.asarray(x_prev, dtype=float),
            z_t=z[time_index, :],
            field_t=np.asarray(bundle.field_matrix[time_index, :], dtype=float),
            interaction_matrix=interaction_matrix,
            beta=float(bundle.beta),
            eta=float(bundle.eta),
            rng=rng,
            validation_nodes=validation_nodes,
            gibbs_sweeps=int(gibbs_sweeps),
        )
        x_prev = sampled_x[time_index, :]
    sampled_x[~validation_mask] = x[~validation_mask]
    return sampled_x


def _sample_validation_magnetization_metrics(
    *,
    panel_context: dict[str, object],
    bundle: OutcomeParameterBundle,
    validation_loss_mask: np.ndarray,
    validation_sampling: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    sampling = resolve_validation_sampling(validation_sampling)
    x = np.asarray(panel_context["x"], dtype=float)
    validation_mask = np.asarray(validation_loss_mask, dtype=bool)
    post_s_mask = validation_mask & time_window_mask(
        t_steps=x.shape[0],
        n_nodes=x.shape[1],
        start_t=int(panel_context["s"]),
    )
    observed_validation_mean = _mean_on_mask(x, validation_mask)
    observed_post_s_mean = _mean_on_mask(x, post_s_mask)

    validation_sample_means: list[float] = []
    post_s_sample_means: list[float] = []
    for sample_index in range(int(sampling["num_samples"])):
        sampled_x = sample_validation_panel_conditional(
            panel_context=panel_context,
            bundle=bundle,
            validation_loss_mask=validation_mask,
            gibbs_sweeps=int(sampling["gibbs_sweeps"]),
            seed=int(sampling["seed"]) + int(sample_index),
        )
        sample_validation_mean = _mean_on_mask(sampled_x, validation_mask)
        if sample_validation_mean is not None:
            validation_sample_means.append(float(sample_validation_mean))
        sample_post_s_mean = _mean_on_mask(sampled_x, post_s_mask)
        if sample_post_s_mean is not None:
            post_s_sample_means.append(float(sample_post_s_mean))

    sampled_validation_mean = (
        float(np.mean(np.asarray(validation_sample_means, dtype=float)))
        if validation_sample_means
        else None
    )
    sampled_post_s_mean = (
        float(np.mean(np.asarray(post_s_sample_means, dtype=float)))
        if post_s_sample_means
        else None
    )
    return {
        "validation_observed_mean_magnetization": observed_validation_mean,
        "validation_sampled_mean_magnetization_mean": sampled_validation_mean,
        "validation_mean_magnetization_abs_diff": (
            None
            if observed_validation_mean is None or sampled_validation_mean is None
            else abs(float(observed_validation_mean) - float(sampled_validation_mean))
        ),
        "post_s_validation_observed_mean_magnetization": observed_post_s_mean,
        "post_s_validation_sampled_mean_magnetization_mean": sampled_post_s_mean,
        "post_s_validation_mean_magnetization_abs_diff": (
            None
            if observed_post_s_mean is None or sampled_post_s_mean is None
            else abs(float(observed_post_s_mean) - float(sampled_post_s_mean))
        ),
    }


def _build_loss_kwargs(
    bundle: OutcomeParameterBundle,
    panel_context: dict[str, object],
    *,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    interaction_effect_x: np.ndarray,
) -> dict[str, Any]:
    return {
        "x": x,
        "z": z,
        "x_0": x_0,
        "field_matrix": np.asarray(bundle.field_matrix, dtype=float),
        "beta": float(bundle.beta),
        "xi": float(bundle.xi),
        "eta": float(bundle.eta),
        "interaction_effect_x": interaction_effect_x,
        "fixed_scalar_params": {},
        "s": int(panel_context["s"]),
        "e": int(panel_context["e"]),
        "beta_mask_pre_s": bool(bundle.beta_mask_pre_s),
        "beta_mask_post_e": bool(bundle.beta_mask_post_e),
    }


def evaluate_fold_metrics(
    *,
    panel_context: dict[str, object],
    bundle: OutcomeParameterBundle,
    training_loss_mask: np.ndarray,
    validation_loss_mask: np.ndarray,
    validation_sampling: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    x = np.asarray(panel_context["x"], dtype=float)
    z = np.asarray(panel_context["z"], dtype=float)
    x_0 = np.asarray(panel_context["x_0"], dtype=float)
    training_mask = np.asarray(training_loss_mask, dtype=bool)
    validation_mask = np.asarray(validation_loss_mask, dtype=bool)
    if x.shape != training_mask.shape or x.shape != validation_mask.shape:
        raise ValueError("Training and validation masks must match the panel shape.")

    interaction_effect_x = interaction_effect(x, bundle.gamma_matrix)
    common_kwargs = _build_loss_kwargs(
        bundle,
        panel_context,
        x=x,
        z=z,
        x_0=x_0,
        interaction_effect_x=interaction_effect_x,
    )
    fit_loss = float(
        evaluate_mple_loss_from_parts(
            loss_mask=training_mask,
            **common_kwargs,
        )
    )
    validation_loss = float(
        evaluate_mple_loss_from_parts(
            loss_mask=validation_mask,
            **common_kwargs,
        )
    )
    post_s_validation_loss_mask = validation_mask & time_window_mask(
        t_steps=x.shape[0],
        n_nodes=x.shape[1],
        start_t=int(panel_context["s"]),
    )
    num_post_s_validation_slots = int(np.count_nonzero(post_s_validation_loss_mask))
    post_s_validation_loss = None
    if num_post_s_validation_slots > 0:
        post_s_validation_loss = float(
            evaluate_mple_loss_from_parts(
                loss_mask=post_s_validation_loss_mask,
                **common_kwargs,
            )
        )

    h_x = _compute_h_x_from_bundle(bundle, panel_context)
    validation_brier = validation_brier_score(
        x=x,
        h_x=h_x,
        loss_mask=validation_mask,
    )
    validation_ece = validation_expected_calibration_error(
        x=x,
        h_x=h_x,
        loss_mask=validation_mask,
    )
    post_s_validation_brier = None
    post_s_validation_ece = None
    if num_post_s_validation_slots > 0:
        post_s_validation_brier = validation_brier_score(
            x=x,
            h_x=h_x,
            loss_mask=post_s_validation_loss_mask,
        )
        post_s_validation_ece = validation_expected_calibration_error(
            x=x,
            h_x=h_x,
            loss_mask=post_s_validation_loss_mask,
        )

    magnetization_metrics = _sample_validation_magnetization_metrics(
        panel_context=panel_context,
        bundle=bundle,
        validation_loss_mask=validation_mask,
        validation_sampling=validation_sampling,
    )
    return {
        "fit_loss": fit_loss,
        "validation_loss": validation_loss,
        "validation_brier_score": float(validation_brier),
        "validation_ece": float(validation_ece),
        "num_post_s_validation_slots": int(num_post_s_validation_slots),
        "post_s_validation_loss": post_s_validation_loss,
        "post_s_validation_brier_score": post_s_validation_brier,
        "post_s_validation_ece": post_s_validation_ece,
        **magnetization_metrics,
    }


def evaluate_saved_fit_fold_metrics(
    fit_root: str | Path,
    experiment_root: str | Path,
    *,
    training_loss_mask: np.ndarray,
    validation_loss_mask: np.ndarray,
    validation_sampling: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    panel_context = load_experiment_panel_context(experiment_root)
    bundle = load_fit_parameter_bundle(fit_root, experiment_root)
    return evaluate_fold_metrics(
        panel_context=panel_context,
        bundle=bundle,
        training_loss_mask=training_loss_mask,
        validation_loss_mask=validation_loss_mask,
        validation_sampling=validation_sampling,
    )


def evaluate_test_metrics(
    *,
    panel_context: dict[str, object],
    bundle: OutcomeParameterBundle,
    training_loss_mask: np.ndarray,
    test_loss_mask: np.ndarray,
    sampling: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    metrics = evaluate_fold_metrics(
        panel_context=panel_context,
        bundle=bundle,
        training_loss_mask=training_loss_mask,
        validation_loss_mask=test_loss_mask,
        validation_sampling=sampling,
    )
    training_mask = np.asarray(training_loss_mask, dtype=bool)
    scored_test_mask = np.asarray(test_loss_mask, dtype=bool)
    return {
        "training_loss": float(metrics["fit_loss"]),
        "num_training_slots": int(np.count_nonzero(training_mask)),
        "test_loss": float(metrics["validation_loss"]),
        "test_brier_score": float(metrics["validation_brier_score"]),
        "test_ece": float(metrics["validation_ece"]),
        "num_test_slots": int(np.count_nonzero(scored_test_mask)),
        "post_s_test_loss": metrics["post_s_validation_loss"],
        "post_s_test_brier_score": metrics["post_s_validation_brier_score"],
        "post_s_test_ece": metrics["post_s_validation_ece"],
        "num_post_s_test_slots": int(metrics["num_post_s_validation_slots"]),
        "test_mean_magnetization_abs_diff": metrics["validation_mean_magnetization_abs_diff"],
        "test_observed_mean_magnetization": metrics["validation_observed_mean_magnetization"],
        "test_sampled_mean_magnetization_mean": metrics[
            "validation_sampled_mean_magnetization_mean"
        ],
        "post_s_test_mean_magnetization_abs_diff": metrics[
            "post_s_validation_mean_magnetization_abs_diff"
        ],
        "post_s_test_observed_mean_magnetization": metrics[
            "post_s_validation_observed_mean_magnetization"
        ],
        "post_s_test_sampled_mean_magnetization_mean": metrics[
            "post_s_validation_sampled_mean_magnetization_mean"
        ],
    }


def evaluate_test_metrics_by_treatment(
    *,
    panel_context: dict[str, object],
    bundle: OutcomeParameterBundle,
    test_loss_mask: np.ndarray,
    sampling: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    x = np.asarray(panel_context["x"], dtype=float)
    z = np.asarray(panel_context["z"], dtype=float)
    x_0 = np.asarray(panel_context["x_0"], dtype=float)
    test_mask = np.asarray(test_loss_mask, dtype=bool)
    s = int(panel_context["s"])
    post_s_window = time_window_mask(
        t_steps=x.shape[0], n_nodes=x.shape[1], start_t=s
    )

    interaction_effect_x = interaction_effect(x, bundle.gamma_matrix)
    common_kwargs = _build_loss_kwargs(
        bundle,
        panel_context,
        x=x,
        z=z,
        x_0=x_0,
        interaction_effect_x=interaction_effect_x,
    )

    treated_mask = z > 0.5
    untreated_mask = z <= 0.5

    h_x = _compute_h_x_from_bundle(bundle, panel_context)
    metrics = {}

    for treatment_name, treatment_selector in [("treated", treated_mask), ("untreated", untreated_mask)]:
        test_and_treatment = test_mask & treatment_selector
        test_and_treatment_post_s = test_and_treatment & post_s_window

        if np.any(test_and_treatment):
            test_loss = float(
                evaluate_mple_loss_from_parts(
                    loss_mask=test_and_treatment,
                    **common_kwargs,
                )
            )
            test_brier = validation_brier_score(
                x=x,
                h_x=h_x,
                loss_mask=test_and_treatment,
            )
            test_ece = validation_expected_calibration_error(
                x=x,
                h_x=h_x,
                loss_mask=test_and_treatment,
            )
            num_test_slots = int(np.count_nonzero(test_and_treatment))

            metrics[f"test_loss_{treatment_name}"] = test_loss
            metrics[f"test_brier_score_{treatment_name}"] = test_brier
            metrics[f"test_ece_{treatment_name}"] = test_ece
            metrics[f"num_test_slots_{treatment_name}"] = num_test_slots
        else:
            metrics[f"test_loss_{treatment_name}"] = None
            metrics[f"test_brier_score_{treatment_name}"] = None
            metrics[f"test_ece_{treatment_name}"] = None
            metrics[f"num_test_slots_{treatment_name}"] = 0

        magnetization_metrics = _compute_magnetization_metrics(
            x=x,
            bundle=bundle,
            loss_mask=test_and_treatment,
            sampling=sampling,
            panel_context=panel_context,
        )

        if np.any(test_and_treatment):
            metrics[f"test_mean_magnetization_abs_diff_{treatment_name}"] = (
                magnetization_metrics["magnetization_abs_diff"]
            )
        else:
            metrics[f"test_mean_magnetization_abs_diff_{treatment_name}"] = None

        if np.any(test_and_treatment_post_s):
            post_s_test_loss = float(
                evaluate_mple_loss_from_parts(
                    loss_mask=test_and_treatment_post_s,
                    **common_kwargs,
                )
            )
            post_s_test_brier = validation_brier_score(
                x=x,
                h_x=h_x,
                loss_mask=test_and_treatment_post_s,
            )
            post_s_test_ece = validation_expected_calibration_error(
                x=x,
                h_x=h_x,
                loss_mask=test_and_treatment_post_s,
            )
            num_post_s_test_slots = int(np.count_nonzero(test_and_treatment_post_s))

            metrics[f"post_s_test_loss_{treatment_name}"] = post_s_test_loss
            metrics[f"post_s_test_brier_score_{treatment_name}"] = post_s_test_brier
            metrics[f"post_s_test_ece_{treatment_name}"] = post_s_test_ece
            metrics[f"post_s_test_mean_magnetization_abs_diff_{treatment_name}"] = (
                magnetization_metrics["post_s_magnetization_abs_diff"]
            )
            metrics[f"num_post_s_test_slots_{treatment_name}"] = num_post_s_test_slots
        else:
            metrics[f"post_s_test_loss_{treatment_name}"] = None
            metrics[f"post_s_test_brier_score_{treatment_name}"] = None
            metrics[f"post_s_test_ece_{treatment_name}"] = None
            metrics[f"post_s_test_mean_magnetization_abs_diff_{treatment_name}"] = None
            metrics[f"num_post_s_test_slots_{treatment_name}"] = 0

    return metrics


def _compute_magnetization_metrics(
    *,
    x: np.ndarray,
    bundle: OutcomeParameterBundle,
    loss_mask: np.ndarray,
    sampling: dict[str, Any] | None = None,
    panel_context: dict[str, object],
) -> dict[str, float | None]:
    """Compute magnetization metrics for a given loss mask and post-s restricted window."""
    sampling_config = resolve_validation_sampling(sampling)
    mask = np.asarray(loss_mask, dtype=bool)
    post_s_mask = mask & time_window_mask(
        t_steps=x.shape[0],
        n_nodes=x.shape[1],
        start_t=int(panel_context["s"]),
    )
    observed_mean = _mean_on_mask(x, mask)
    observed_post_s_mean = _mean_on_mask(x, post_s_mask)

    sample_means: list[float] = []
    post_s_sample_means: list[float] = []
    for sample_index in range(int(sampling_config["num_samples"])):
        sampled_x = sample_validation_panel_conditional(
            panel_context=panel_context,
            bundle=bundle,
            validation_loss_mask=mask,
            gibbs_sweeps=int(sampling_config["gibbs_sweeps"]),
            seed=int(sampling_config["seed"]) + int(sample_index),
        )
        sample_mean = _mean_on_mask(sampled_x, mask)
        if sample_mean is not None:
            sample_means.append(float(sample_mean))
        post_s_sample_mean = _mean_on_mask(sampled_x, post_s_mask)
        if post_s_sample_mean is not None:
            post_s_sample_means.append(float(post_s_sample_mean))

    sampled_mean = (
        float(np.mean(np.asarray(sample_means, dtype=float)))
        if sample_means
        else None
    )
    sampled_post_s_mean = (
        float(np.mean(np.asarray(post_s_sample_means, dtype=float)))
        if post_s_sample_means
        else None
    )
    return {
        "magnetization_abs_diff": (
            None
            if observed_mean is None or sampled_mean is None
            else abs(float(observed_mean) - float(sampled_mean))
        ),
        "post_s_magnetization_abs_diff": (
            None
            if observed_post_s_mean is None or sampled_post_s_mean is None
            else abs(float(observed_post_s_mean) - float(sampled_post_s_mean))
        ),
    }


def _compute_h_x_from_bundle(bundle: OutcomeParameterBundle, panel_context: dict[str, object]) -> np.ndarray:
    x = np.asarray(panel_context["x"], dtype=float)
    z = np.asarray(panel_context["z"], dtype=float)
    x_0 = np.asarray(panel_context["x_0"], dtype=float)
    field_matrix = np.asarray(bundle.field_matrix, dtype=float)
    interaction_effect_x = interaction_effect(x, bundle.gamma_matrix)
    prev_x = np.vstack([x_0, x[:-1, :]])
    h_x = (
        field_matrix
        + float(bundle.beta) * z
        + float(bundle.xi) * interaction_effect_x
        + float(bundle.eta) * prev_x
    )
    return h_x


def evaluate_saved_fit_test_metrics(
    fit_root: str | Path,
    experiment_root: str | Path,
    *,
    training_loss_mask: np.ndarray,
    test_loss_mask: np.ndarray,
    sampling: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    panel_context = load_experiment_panel_context(experiment_root)
    bundle = load_fit_parameter_bundle(fit_root, experiment_root)
    metrics = evaluate_test_metrics(
        panel_context=panel_context,
        bundle=bundle,
        training_loss_mask=training_loss_mask,
        test_loss_mask=test_loss_mask,
        sampling=sampling,
    )
    stratified_metrics = evaluate_test_metrics_by_treatment(
        panel_context=panel_context,
        bundle=bundle,
        test_loss_mask=test_loss_mask,
        sampling=sampling,
    )
    metrics.update(stratified_metrics)
    return metrics


def _blank_aggregate_metrics() -> dict[str, object]:
    return {
        "weighted_mean_validation_loss": "",
        "mean_fold_validation_loss": "",
        "weighted_mean_validation_brier_score": "",
        "mean_fold_validation_brier_score": "",
        "standard_error_fold_validation_brier_score": "",
        "weighted_mean_validation_ece": "",
        "mean_fold_validation_ece": "",
        "weighted_mean_validation_mean_magnetization_abs_diff": "",
        "mean_fold_validation_mean_magnetization_abs_diff": "",
        "standard_error_fold_validation_mean_magnetization_abs_diff": "",
        "total_validation_slots": "",
        "weighted_mean_post_s_validation_loss": "",
        "mean_fold_post_s_validation_loss": "",
        "weighted_mean_post_s_validation_brier_score": "",
        "mean_fold_post_s_validation_brier_score": "",
        "standard_error_fold_post_s_validation_brier_score": "",
        "weighted_mean_post_s_validation_ece": "",
        "mean_fold_post_s_validation_ece": "",
        "weighted_mean_post_s_validation_mean_magnetization_abs_diff": "",
        "mean_fold_post_s_validation_mean_magnetization_abs_diff": "",
        "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": "",
        "total_post_s_validation_slots": "",
    }


def _weighted_and_mean(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    weight_key: str,
) -> tuple[float, float]:
    weights = np.asarray([int(row[weight_key]) for row in rows], dtype=float)
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    return (
        float(np.sum(weights * values) / np.sum(weights)),
        float(np.mean(values)),
    )


def _mean_and_standard_error(
    rows: list[dict[str, object]],
    *,
    value_key: str,
) -> tuple[float, float]:
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    mean_value = float(np.mean(values))
    if values.size <= 1:
        return mean_value, 0.0
    return mean_value, float(np.std(values, ddof=1) / np.sqrt(values.size))


def build_candidate_score_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_rows: list[dict[str, object]],
    *,
    expected_num_folds: int,
) -> dict[str, object]:
    success_rows = [row for row in fold_rows if row.get("status") == "completed"]
    base_row: dict[str, object] = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "execution_mode": search.get("_execution_mode", "cv"),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
    }
    if len(success_rows) != int(expected_num_folds):
        return {
            **base_row,
            "status": "failed",
            "num_completed_folds": int(len(success_rows)),
            **_blank_aggregate_metrics(),
        }

    weighted_validation_loss, mean_validation_loss = _weighted_and_mean(
        success_rows,
        value_key="validation_loss",
        weight_key="num_validation_slots",
    )
    weighted_validation_brier, mean_validation_brier = _weighted_and_mean(
        success_rows,
        value_key="validation_brier_score",
        weight_key="num_validation_slots",
    )
    _, se_validation_brier = _mean_and_standard_error(
        success_rows,
        value_key="validation_brier_score",
    )
    weighted_validation_ece, mean_validation_ece = _weighted_and_mean(
        success_rows,
        value_key="validation_ece",
        weight_key="num_validation_slots",
    )
    weighted_validation_mag_diff, mean_validation_mag_diff = _weighted_and_mean(
        success_rows,
        value_key="validation_mean_magnetization_abs_diff",
        weight_key="num_validation_slots",
    )
    _, se_validation_mag_diff = _mean_and_standard_error(
        success_rows,
        value_key="validation_mean_magnetization_abs_diff",
    )

    aggregated: dict[str, object] = {
        **base_row,
        "status": "completed",
        "num_completed_folds": int(len(success_rows)),
        "weighted_mean_validation_loss": weighted_validation_loss,
        "mean_fold_validation_loss": mean_validation_loss,
        "weighted_mean_validation_brier_score": weighted_validation_brier,
        "mean_fold_validation_brier_score": mean_validation_brier,
        "standard_error_fold_validation_brier_score": se_validation_brier,
        "weighted_mean_validation_ece": weighted_validation_ece,
        "mean_fold_validation_ece": mean_validation_ece,
        "weighted_mean_validation_mean_magnetization_abs_diff": weighted_validation_mag_diff,
        "mean_fold_validation_mean_magnetization_abs_diff": mean_validation_mag_diff,
        "standard_error_fold_validation_mean_magnetization_abs_diff": se_validation_mag_diff,
        "total_validation_slots": int(
            np.sum(np.asarray([int(row["num_validation_slots"]) for row in success_rows], dtype=int))
        ),
    }

    post_s_rows = [
        row
        for row in success_rows
        if int(row.get("num_post_s_validation_slots", 0)) > 0
    ]
    if not post_s_rows:
        aggregated.update(
            {
                "weighted_mean_post_s_validation_loss": "",
                "mean_fold_post_s_validation_loss": "",
                "weighted_mean_post_s_validation_brier_score": "",
                "mean_fold_post_s_validation_brier_score": "",
                "standard_error_fold_post_s_validation_brier_score": "",
                "weighted_mean_post_s_validation_ece": "",
                "mean_fold_post_s_validation_ece": "",
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": "",
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "total_post_s_validation_slots": 0,
            }
        )
        return aggregated

    weighted_post_s_loss, mean_post_s_loss = _weighted_and_mean(
        post_s_rows,
        value_key="post_s_validation_loss",
        weight_key="num_post_s_validation_slots",
    )
    weighted_post_s_brier, mean_post_s_brier = _weighted_and_mean(
        post_s_rows,
        value_key="post_s_validation_brier_score",
        weight_key="num_post_s_validation_slots",
    )
    _, se_post_s_brier = _mean_and_standard_error(
        post_s_rows,
        value_key="post_s_validation_brier_score",
    )
    weighted_post_s_ece, mean_post_s_ece = _weighted_and_mean(
        post_s_rows,
        value_key="post_s_validation_ece",
        weight_key="num_post_s_validation_slots",
    )
    weighted_post_s_mag_diff, mean_post_s_mag_diff = _weighted_and_mean(
        post_s_rows,
        value_key="post_s_validation_mean_magnetization_abs_diff",
        weight_key="num_post_s_validation_slots",
    )
    _, se_post_s_mag_diff = _mean_and_standard_error(
        post_s_rows,
        value_key="post_s_validation_mean_magnetization_abs_diff",
    )
    aggregated.update(
        {
            "weighted_mean_post_s_validation_loss": weighted_post_s_loss,
            "mean_fold_post_s_validation_loss": mean_post_s_loss,
            "weighted_mean_post_s_validation_brier_score": weighted_post_s_brier,
            "mean_fold_post_s_validation_brier_score": mean_post_s_brier,
            "standard_error_fold_post_s_validation_brier_score": se_post_s_brier,
            "weighted_mean_post_s_validation_ece": weighted_post_s_ece,
            "mean_fold_post_s_validation_ece": mean_post_s_ece,
            "weighted_mean_post_s_validation_mean_magnetization_abs_diff": weighted_post_s_mag_diff,
            "mean_fold_post_s_validation_mean_magnetization_abs_diff": mean_post_s_mag_diff,
            "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": se_post_s_mag_diff,
            "total_post_s_validation_slots": int(
                np.sum(
                    np.asarray(
                        [int(row["num_post_s_validation_slots"]) for row in post_s_rows],
                        dtype=int,
                    )
                )
            ),
        }
    )
    return aggregated


def candidate_score_sort_key(row: dict[str, object]) -> tuple[float, float, int]:
    mag_diff = row.get("weighted_mean_post_s_validation_mean_magnetization_abs_diff", "")
    brier = row.get("weighted_mean_post_s_validation_brier_score", "")
    loss = row.get("weighted_mean_post_s_validation_loss", "")
    if mag_diff in ("", None):
        mag_diff = row["weighted_mean_validation_mean_magnetization_abs_diff"]
    if brier in ("", None):
        brier = row["weighted_mean_validation_brier_score"]
    if loss in ("", None):
        loss = row["weighted_mean_validation_loss"]
    return (
        float(mag_diff),
        float(brier),
        float(loss),
        int(row["candidate_index"]),
    )
