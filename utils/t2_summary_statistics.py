"""Summary statistics and masking utilities."""

from __future__ import annotations

import math

import numpy as np


def finite_scalar_summary(
    sample_values: np.ndarray,
    observed_value: object = None,
) -> dict[str, object]:
    """Compute quantiles and statistics for sample values, optionally with an observed value.

    Args:
        sample_values: Array of sample values.
        observed_value: Optional observed value to compute error against.

    Returns:
        Dictionary with keys: sample_mean, sample_std, q025, q500, q975, num_finite_samples.
        If observed_value is not None, also includes: observed_value, abs_error, in_95_interval.
    """
    finite = np.asarray(sample_values, dtype=float)
    finite = finite[np.isfinite(finite)]

    base_result = {
        "sample_mean": "",
        "sample_std": "",
        "q025": "",
        "q500": "",
        "q975": "",
        "num_finite_samples": int(finite.size),
    }

    if finite.size == 0:
        if observed_value is not None:
            base_result.update({
                "observed_value": observed_value if isinstance(observed_value, (int, float)) and np.isfinite(float(observed_value)) else "",
                "abs_error": "",
                "in_95_interval": "",
            })
        return base_result

    q025, q500, q975 = np.quantile(finite, [0.025, 0.5, 0.975])
    sample_mean = float(np.mean(finite))
    result = {
        "sample_mean": sample_mean,
        "sample_std": float(np.std(finite, ddof=0)),
        "q025": float(q025),
        "q500": float(q500),
        "q975": float(q975),
        "num_finite_samples": int(finite.size),
    }

    if observed_value is not None:
        observed = float(observed_value)
        if np.isfinite(observed):
            result.update({
                "observed_value": observed,
                "abs_error": abs(observed - sample_mean),
                "in_95_interval": bool(float(q025) <= observed <= float(q975)),
            })
        else:
            result.update({
                "observed_value": "",
                "abs_error": "",
                "in_95_interval": "",
            })

    return result


def finite_vector_summaries(
    observed_values: np.ndarray,
    sample_values: np.ndarray,
    *,
    index_name: str,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Compute summaries for a vector of observed values against 2D samples.

    Args:
        observed_values: 1D array of observed values (length N).
        sample_values: 2D array of samples (shape M x N, where M is num samples).
        index_name: Name for the index column (e.g. "unit_index", "time_index").

    Returns:
        Tuple of (rows, aggregates) where:
        - rows: List of dicts with per-item statistics
        - aggregates: Dict with abs_error_mean, rmse, max_abs_error, coverage_rate
    """
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
            finite_scalar_summary(
                samples[:, item_index],
                observed_value=float(observed[item_index]),
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


def mean_on_mask(x: np.ndarray, mask: np.ndarray) -> float | None:
    """Compute mean of x over positions where mask is True.

    Returns None if mask selects no entries. Raises ValueError on shape mismatch.
    """
    x_array = np.asarray(x, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if x_array.shape != mask_array.shape:
        raise ValueError(
            "x and mask must have the same shape when averaging over a mask."
        )
    if not np.any(mask_array):
        return None
    return float(np.mean(x_array[mask_array]))


def time_window_mask(*, t_steps: int, n_nodes: int, start_t: int = 0) -> np.ndarray:
    """Create a boolean mask for a time window [start_t, t_steps) across all nodes.

    Args:
        t_steps: Total number of time steps.
        n_nodes: Total number of nodes.
        start_t: Start time index (default 0).

    Returns:
        Boolean array of shape (t_steps, n_nodes) with True for times >= start_t.
    """
    mask = np.zeros((int(t_steps), int(n_nodes)), dtype=bool)
    if int(start_t) < int(t_steps):
        mask[int(start_t) :, :] = True
    return mask
