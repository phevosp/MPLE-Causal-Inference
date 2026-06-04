"""CV fold aggregation and candidate scoring."""

from __future__ import annotations

from typing import Any

import numpy as np


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
            np.sum(
                np.asarray(
                    [int(row["num_validation_slots"]) for row in success_rows],
                    dtype=int,
                )
            )
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
                        [
                            int(row["num_post_s_validation_slots"])
                            for row in post_s_rows
                        ],
                        dtype=int,
                    )
                )
            ),
        }
    )
    return aggregated


def candidate_score_sort_key(row: dict[str, object]) -> tuple[float, float, float, int]:
    mag_diff = row.get(
        "weighted_mean_post_s_validation_mean_magnetization_abs_diff", ""
    )
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
