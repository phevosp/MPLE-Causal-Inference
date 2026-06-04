"""Run spatiotemporal hyperparameter search in CV or single-validation mode."""

from __future__ import annotations

import argparse
import ast
import itertools
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

import build_cv_folds as cv_folds
from utils.t1_matrix_io import save_loss_mask
from utils.t0_path_utils import io_path
from utils.t5_experiment_context import load_experiment_panel_context
from pipeline_specs import (
    deep_merge,
    expand_named_entries,
    load_spec,
    read_csv_manifest,
    slugify,
    validate_cv_spec,
    validate_fit_variant_dict,
    write_csv_manifest,
)
from run_fit_pipeline import execute_fit_root, materialize_fit_root
from utils.t6_split_management import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    SPLIT_SOURCE_CV_FOLDS,
    load_model_selection_split_masks,
    normalize_split_source,
)
from utils.t2_summary_statistics import time_window_mask
from utils.t7_cv_aggregation import (
    build_candidate_score_row,
    candidate_score_sort_key,
)
from utils.t7_validation_metrics import (
    evaluate_saved_fit_fold_metrics,
    resolve_validation_sampling,
)


DEFAULT_NUM_FOLDS = 5
CV_REQUESTS_NAME = "cv_requests.csv"
VALIDATION_REQUESTS_NAME = "validation_requests.csv"
EXECUTION_MODE_CV = "cv"
EXECUTION_MODE_VALIDATION = "validation"
VALID_EXECUTION_MODES = frozenset({EXECUTION_MODE_CV, EXECUTION_MODE_VALIDATION})

AGGREGATED_METRIC_KEYS = (
    "weighted_mean_validation_loss",
    "mean_fold_validation_loss",
    "weighted_mean_validation_brier_score",
    "mean_fold_validation_brier_score",
    "standard_error_fold_validation_brier_score",
    "weighted_mean_validation_ece",
    "mean_fold_validation_ece",
    "weighted_mean_validation_mean_magnetization_abs_diff",
    "mean_fold_validation_mean_magnetization_abs_diff",
    "standard_error_fold_validation_mean_magnetization_abs_diff",
    "total_validation_slots",
    "weighted_mean_post_s_validation_loss",
    "mean_fold_post_s_validation_loss",
    "weighted_mean_post_s_validation_brier_score",
    "mean_fold_post_s_validation_brier_score",
    "standard_error_fold_post_s_validation_brier_score",
    "weighted_mean_post_s_validation_ece",
    "mean_fold_post_s_validation_ece",
    "weighted_mean_post_s_validation_mean_magnetization_abs_diff",
    "mean_fold_post_s_validation_mean_magnetization_abs_diff",
    "standard_error_fold_post_s_validation_mean_magnetization_abs_diff",
    "total_post_s_validation_slots",
)
FOLD_METRIC_KEYS = (
    "fit_loss",
    "validation_loss",
    "validation_brier_score",
    "validation_ece",
    "num_post_s_validation_slots",
    "post_s_validation_loss",
    "post_s_validation_brier_score",
    "post_s_validation_ece",
    "validation_mean_magnetization_abs_diff",
    "validation_observed_mean_magnetization",
    "validation_sampled_mean_magnetization_mean",
    "post_s_validation_mean_magnetization_abs_diff",
    "post_s_validation_observed_mean_magnetization",
    "post_s_validation_sampled_mean_magnetization_mean",
)

WINNER_MAG_DIFF_MEAN_KEY = "mean_fold_post_s_validation_mean_magnetization_abs_diff"
WINNER_MAG_DIFF_SE_KEY = "standard_error_fold_post_s_validation_mean_magnetization_abs_diff"
WINNER_BRIER_MEAN_KEY = "mean_fold_post_s_validation_brier_score"
WINNER_BRIER_SE_KEY = "standard_error_fold_post_s_validation_brier_score"


def _normalize_execution_mode(execution_mode: str) -> str:
    normalized = str(execution_mode).strip().lower()
    if normalized not in VALID_EXECUTION_MODES:
        raise ValueError(
            f"execution_mode must be one of {sorted(VALID_EXECUTION_MODES)}, got '{execution_mode}'."
        )
    return normalized


def _get_num_folds_from_search(search: dict[str, Any]) -> int:
    num_folds = search.get("num_folds")
    if num_folds is not None:
        return int(num_folds)
    return DEFAULT_NUM_FOLDS


def _expand_searches(cv_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_cv_spec(cv_spec_path)
    searches = expand_named_entries(cv_spec_path, "searches")
    if not searches:
        raise ValueError(f"No searches found in CV spec {cv_spec_path}.")
    for search in searches:
        search["_spec_path"] = str(Path(cv_spec_path).resolve())
    return searches


def _execution_manifest_path_from_search(
    search: dict[str, Any],
    *,
    execution_mode: str,
) -> Path:
    normalized_mode = _normalize_execution_mode(execution_mode)
    if normalized_mode == EXECUTION_MODE_CV:
        return Path(str(search["cv_manifest_path"]))
    validation_manifest_path = search.get("validation_manifest_path")
    if validation_manifest_path not in (None, ""):
        return Path(str(validation_manifest_path))
    return Path(str(search["cv_manifest_path"])).with_name("validation_manifest.csv")


def model_selection_manifest_path_for_spec(
    cv_spec_path: str | Path,
    *,
    execution_mode: str = EXECUTION_MODE_CV,
) -> Path:
    searches = _expand_searches(cv_spec_path)
    return _execution_manifest_path_from_search(
        searches[0],
        execution_mode=execution_mode,
    )


def cv_manifest_path_for_spec(cv_spec_path: str | Path) -> Path:
    return model_selection_manifest_path_for_spec(
        cv_spec_path,
        execution_mode=EXECUTION_MODE_CV,
    )


def validation_manifest_path_for_spec(cv_spec_path: str | Path) -> Path:
    return model_selection_manifest_path_for_spec(
        cv_spec_path,
        execution_mode=EXECUTION_MODE_VALIDATION,
    )


def model_selection_requests_path_for_spec(
    cv_spec_path: str | Path,
    *,
    execution_mode: str = EXECUTION_MODE_CV,
) -> Path:
    manifest_path = model_selection_manifest_path_for_spec(
        cv_spec_path,
        execution_mode=execution_mode,
    )
    # Derive requests name from manifest name by replacing "cv_manifest" or "validation_manifest"
    manifest_name = manifest_path.stem  # e.g., "cv_manifest" or "cv_manifest_xi_zero"
    prefix = "cv" if _normalize_execution_mode(execution_mode) == EXECUTION_MODE_CV else "validation"
    manifest_prefix = "cv_manifest" if "cv_manifest" in manifest_name else "validation_manifest"
    requests_name = manifest_name.replace(manifest_prefix, f"{prefix}_requests") + ".csv"
    return manifest_path.with_name(requests_name)


def cv_requests_path_for_spec(cv_spec_path: str | Path) -> Path:
    return model_selection_requests_path_for_spec(
        cv_spec_path,
        execution_mode=EXECUTION_MODE_CV,
    )


def validation_requests_path_for_spec(cv_spec_path: str | Path) -> Path:
    return model_selection_requests_path_for_spec(
        cv_spec_path,
        execution_mode=EXECUTION_MODE_VALIDATION,
    )


def _grid_leaf_entries(
    node: dict[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], list[Any]]]:
    leaves: list[tuple[tuple[str, ...], list[Any]]] = []
    for key, value in node.items():
        key_path = (*path, str(key))
        if isinstance(value, dict):
            leaves.extend(_grid_leaf_entries(value, path=key_path))
            continue
        if not isinstance(value, list) or not value:
            dotted = ".".join(key_path)
            raise ValueError(f"Grid leaf '{dotted}' must be a non-empty list.")
        leaves.append((key_path, list(value)))
    return leaves


def _assign_nested_value(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value


def expand_search_candidates(search: dict[str, Any]) -> list[dict[str, Any]]:
    grid = dict(search.get("grid", {}) or {})
    if not grid:
        raise ValueError(f"Search '{search.get('name', '<unnamed>')}' must define grid.")
    leaves = _grid_leaf_entries(grid)
    search_base = {key: value for key, value in search.items() if key != "grid"}
    candidates: list[dict[str, Any]] = []
    for candidate_index, values in enumerate(
        itertools.product(*(leaf_values for _, leaf_values in leaves)),
        start=1,
    ):
        overrides: dict[str, Any] = {}
        flat_params: dict[str, Any] = {}
        for (path, _), value in zip(leaves, values):
            _assign_nested_value(overrides, path, value)
            flat_params[".".join(path)] = value
        candidate = deep_merge(search_base, overrides)
        candidate_name = f"{search['name']}__candidate_{candidate_index:04d}"
        candidate["name"] = candidate_name
        candidate["slug"] = slugify(candidate_name)
        candidate["_candidate_index"] = int(candidate_index)
        candidate["_flat_params"] = flat_params
        validate_fit_variant_dict(candidate)
        candidates.append(candidate)
    return candidates


def _search_split_source(search: dict[str, Any]) -> str:
    return normalize_split_source(search.get("split_source", SPLIT_SOURCE_CV_FOLDS))


def _search_outer_num_folds(search: dict[str, Any]) -> int:
    value = search.get("outer_num_folds")
    if value is None:
        return DEFAULT_OUTER_NUM_FOLDS
    return int(value)


def _search_test_fold_id(search: dict[str, Any]) -> int:
    value = search.get("test_fold_id")
    if value is None:
        return DEFAULT_TEST_FOLD_ID
    return int(value)


def _output_root_name(search: dict[str, Any], *, execution_mode: str) -> str:
    normalized_mode = _normalize_execution_mode(execution_mode)
    if normalized_mode == EXECUTION_MODE_CV:
        return str(search.get("cv_root_name", "cv_runs"))
    return str(search.get("validation_root_name", "validation_runs"))


def _candidate_output_root(
    experiment_root: str | Path,
    search: dict[str, Any],
    *,
    execution_mode: str,
) -> Path:
    return Path(experiment_root) / _output_root_name(
        search,
        execution_mode=execution_mode,
    ) / str(search["slug"])


def _candidate_fit_root(
    experiment_root: str | Path,
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_id: int,
    *,
    execution_mode: str,
) -> Path:
    return (
        _candidate_output_root(
            experiment_root,
            search,
            execution_mode=execution_mode,
        )
        / "candidates"
        / str(candidate["slug"])
        / f"fold_{int(fold_id)}"
    )


def _selected_fold_ids(num_folds: int, *, execution_mode: str) -> tuple[int, ...]:
    normalized_mode = _normalize_execution_mode(execution_mode)
    if normalized_mode == EXECUTION_MODE_VALIDATION:
        return (1,)
    return tuple(range(1, int(num_folds) + 1))


def _nonempty_fold_ids_from_masks(
    training_masks: np.ndarray,
    validation_masks: np.ndarray,
) -> tuple[int, ...]:
    training_array = np.asarray(training_masks, dtype=bool)
    validation_array = np.asarray(validation_masks, dtype=bool)
    supported_fold_ids: list[int] = []
    for fold_index in range(int(training_array.shape[0])):
        if int(np.count_nonzero(training_array[fold_index])) <= 0:
            continue
        if int(np.count_nonzero(validation_array[fold_index])) <= 0:
            continue
        supported_fold_ids.append(int(fold_index + 1))
    return tuple(supported_fold_ids)


def _resolved_fold_ids_for_experiment(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    *,
    execution_mode: str,
) -> tuple[int, ...]:
    configured_num_folds = _get_num_folds_from_search(search)
    requested_fold_ids = _selected_fold_ids(
        configured_num_folds,
        execution_mode=execution_mode,
    )
    experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
    split_artifacts = load_model_selection_split_masks(
        experiment_root,
        split_source=_search_split_source(search),
        num_folds=configured_num_folds,
        outer_num_folds=_search_outer_num_folds(search),
        test_fold_id=_search_test_fold_id(search),
    )
    supported_fold_ids = _nonempty_fold_ids_from_masks(
        np.asarray(split_artifacts["training_masks"], dtype=bool),
        np.asarray(split_artifacts["validation_masks"], dtype=bool),
    )
    if not supported_fold_ids:
        raise ValueError(
            f"No usable fold ids remain for {experiment_root} with split_source="
            f"{_search_split_source(search)}."
        )
    if _normalize_execution_mode(execution_mode) == EXECUTION_MODE_VALIDATION:
        return (supported_fold_ids[0],)
    return tuple(
        fold_id for fold_id in requested_fold_ids if int(fold_id) in set(supported_fold_ids)
    )


def _cv_request_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_id: int,
    *,
    execution_mode: str,
    configured_num_folds: int,
) -> dict[str, object]:
    fit_root = _candidate_fit_root(
        experiment_row["experiment_path"],
        search,
        candidate,
        fold_id,
        execution_mode=execution_mode,
    )
    return {
        "execution_mode": _normalize_execution_mode(execution_mode),
        "configured_num_folds": int(configured_num_folds),
        "split_source": _search_split_source(search),
        "outer_num_folds": int(_search_outer_num_folds(search)),
        "test_fold_id": int(_search_test_fold_id(search)),
        "cv_spec_path": str(Path(search["_spec_path"]).resolve()),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "experiment_path": experiment_row.get("experiment_path", ""),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
        "cv_fold_id": int(fold_id),
        "fit_path": str(fit_root.resolve()),
    }


def write_cv_requests(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    *,
    execution_mode: str = EXECUTION_MODE_CV,
) -> Path:
    normalized_mode = _normalize_execution_mode(execution_mode)
    generation_rows = read_csv_manifest(generation_manifest_path)
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {generation_manifest_path}."
        )
    request_rows: list[dict[str, object]] = []
    for search in _expand_searches(cv_spec_path):
        configured_num_folds = _get_num_folds_from_search(search)
        for candidate in expand_search_candidates(search):
            for experiment_row in generation_rows:
                for fold_id in _resolved_fold_ids_for_experiment(
                    experiment_row,
                    search,
                    execution_mode=normalized_mode,
                ):
                    request_rows.append(
                        _cv_request_row(
                            experiment_row,
                            search,
                            candidate,
                            fold_id,
                            execution_mode=normalized_mode,
                            configured_num_folds=configured_num_folds,
                        )
                    )
    request_path = model_selection_requests_path_for_spec(
        cv_spec_path,
        execution_mode=normalized_mode,
    )
    write_csv_manifest(request_path, request_rows)
    return request_path



def _parse_manifest_scalar(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


def _candidate_grid_row(
    search: dict[str, Any],
    candidate: dict[str, Any],
    *,
    execution_mode: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "execution_mode": _normalize_execution_mode(execution_mode),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "split_source": _search_split_source(search),
        "outer_num_folds": int(_search_outer_num_folds(search)),
        "test_fold_id": int(_search_test_fold_id(search)),
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
    }
    for key, value in sorted(candidate["_flat_params"].items()):
        row[key] = value
    return row


def _load_scored_candidates(
    output_root: str | Path,
    request_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        candidate_slug = str(row["candidate_slug"])
        candidate_map[candidate_slug] = {
            "name": row["candidate_name"],
            "slug": candidate_slug,
            "_candidate_index": int(row["candidate_index"]),
            "_flat_params": {},
        }

    candidate_grid_path = Path(output_root) / "candidate_grid.csv"
    if candidate_grid_path.exists():
        base_fields = {
            "execution_mode",
            "search_name",
            "search_slug",
            "split_source",
            "outer_num_folds",
            "test_fold_id",
            "candidate_name",
            "candidate_slug",
            "candidate_index",
        }
        for row in read_csv_manifest(candidate_grid_path):
            candidate_slug = str(row["candidate_slug"])
            candidate = candidate_map.setdefault(
                candidate_slug,
                {
                    "name": row["candidate_name"],
                    "slug": candidate_slug,
                    "_candidate_index": int(row["candidate_index"]),
                    "_flat_params": {},
                },
            )
            candidate["_flat_params"] = {
                key: _parse_manifest_scalar(value)
                for key, value in row.items()
                if key not in base_fields and value not in {"", None}
            }

    return sorted(
        candidate_map.values(),
        key=lambda candidate: int(candidate["_candidate_index"]),
    )


def _metric_or_blank(value: object) -> object:
    return "" if value is None else value


def _blank_fold_metric_values() -> dict[str, object]:
    return {key: "" for key in FOLD_METRIC_KEYS}


def _manifest_row_from_best_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    best_candidate: dict[str, Any],
    best_row: dict[str, object],
    output_root: Path,
    *,
    execution_mode: str,
) -> dict[str, object]:
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "experiment_path": str(Path(experiment_row["experiment_path"]).resolve()),
        "execution_mode": _normalize_execution_mode(execution_mode),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "split_source": _search_split_source(search),
        "outer_num_folds": int(_search_outer_num_folds(search)),
        "test_fold_id": int(_search_test_fold_id(search)),
        "output_path": str(output_root.resolve()),
        "best_candidate_name": best_candidate["name"],
        "best_candidate_slug": best_candidate["slug"],
        **{key: best_row[key] for key in AGGREGATED_METRIC_KEYS},
        "status": "completed",
    }


def _best_candidate_payload(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    best_candidate: dict[str, Any],
    best_row: dict[str, object],
    *,
    execution_mode: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "execution_mode": _normalize_execution_mode(execution_mode),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "split_source": _search_split_source(search),
        "outer_num_folds": int(_search_outer_num_folds(search)),
        "test_fold_id": int(_search_test_fold_id(search)),
        "candidate_name": best_candidate["name"],
        "candidate_slug": best_candidate["slug"],
        "candidate_index": int(best_candidate["_candidate_index"]),
        "hyperparameters": {
            key: value for key, value in sorted(best_candidate["_flat_params"].items())
        },
    }
    for key in AGGREGATED_METRIC_KEYS:
        value = best_row[key]
        payload[key] = "" if value == "" else value
    return payload


def _candidate_regularization_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    v_column_l2_max = candidate.get("v_column_l2_max")
    max_column_penalty = (
        -float("inf")
        if v_column_l2_max in (None, "")
        else -float(v_column_l2_max)
    )
    return (
        float(candidate.get("lambda_nuclear", 0.0) or 0.0),
        float(candidate.get("lambda_frobenius", 0.0) or 0.0),
        float(candidate.get("lambda_uv_ridge", 0.0) or 0.0),
        max_column_penalty,
        int(candidate["_candidate_index"]),
    )


def _select_best_candidate_within_standard_error(
    candidates: list[dict[str, Any]],
    candidate_score_rows: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, object]]:
    candidate_by_slug = {str(candidate["slug"]): candidate for candidate in candidates}
    completed_pairs = [
        (candidate_by_slug[str(row["candidate_slug"])], row)
        for row in candidate_score_rows
        if row.get("status") == "completed"
        and str(row.get("candidate_slug", "")) in candidate_by_slug
    ]
    if not completed_pairs:
        raise RuntimeError("No completed candidate score rows are available.")

    winner_mag_diff_mean_key = WINNER_MAG_DIFF_MEAN_KEY
    winner_mag_diff_se_key = WINNER_MAG_DIFF_SE_KEY
    winner_brier_mean_key = WINNER_BRIER_MEAN_KEY
    winner_brier_se_key = WINNER_BRIER_SE_KEY
    if any(
        pair[1].get(winner_mag_diff_mean_key, "") in ("", None)
        or pair[1].get(winner_mag_diff_se_key, "") in ("", None)
        or pair[1].get(winner_brier_mean_key, "") in ("", None)
        or pair[1].get(winner_brier_se_key, "") in ("", None)
        for pair in completed_pairs
    ):
        winner_mag_diff_mean_key = "mean_fold_validation_mean_magnetization_abs_diff"
        winner_mag_diff_se_key = "standard_error_fold_validation_mean_magnetization_abs_diff"
        winner_brier_mean_key = "mean_fold_validation_brier_score"
        winner_brier_se_key = "standard_error_fold_validation_brier_score"

    best_mag_candidate, best_mag_row = min(
        completed_pairs,
        key=lambda pair: (
            float(pair[1][winner_mag_diff_mean_key]),
            float(pair[1][winner_brier_mean_key]),
            _candidate_regularization_sort_key(pair[0]),
        ),
    )
    mag_diff_threshold = float(best_mag_row[winner_mag_diff_mean_key]) + float(
        best_mag_row[winner_mag_diff_se_key]
    )
    mag_eligible_pairs = [
        pair
        for pair in completed_pairs
        if float(pair[1][winner_mag_diff_mean_key]) <= mag_diff_threshold
    ]
    if not mag_eligible_pairs:
        return best_mag_candidate, best_mag_row

    _, best_brier_row = min(
        mag_eligible_pairs,
        key=lambda pair: (
            float(pair[1][winner_brier_mean_key]),
            _candidate_regularization_sort_key(pair[0]),
        ),
    )
    brier_threshold = float(best_brier_row[winner_brier_mean_key]) + float(
        best_brier_row[winner_brier_se_key]
    )
    brier_eligible_pairs = [
        pair
        for pair in mag_eligible_pairs
        if float(pair[1][winner_brier_mean_key]) <= brier_threshold
    ]
    return min(
        brier_eligible_pairs,
        key=lambda pair: (
            _candidate_regularization_sort_key(pair[0]),
            float(pair[1][winner_mag_diff_mean_key]),
            float(pair[1][winner_brier_mean_key]),
        ),
    )


def _write_search_score_artifacts(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidates: list[dict[str, Any]],
    fold_rows: list[dict[str, object]],
    *,
    expected_num_folds: int,
    output_root: str | Path,
    execution_mode: str,
) -> dict[str, object]:
    output_root_path = Path(output_root)
    candidate_score_rows: list[dict[str, object]] = []

    score_search = dict(search)
    score_search["_execution_mode"] = _normalize_execution_mode(execution_mode)
    for candidate in candidates:
        candidate_fold_rows = [
            row for row in fold_rows if row.get("candidate_slug") == candidate["slug"]
        ]
        candidate_score = build_candidate_score_row(
            experiment_row,
            score_search,
            candidate,
            candidate_fold_rows,
            expected_num_folds=expected_num_folds,
        )
        candidate_score_rows.append(candidate_score)

    write_csv_manifest(output_root_path / "fold_scores.csv", fold_rows)
    completed_rows = sorted(
        [row for row in candidate_score_rows if row.get("status") == "completed"],
        key=candidate_score_sort_key,
    )
    for rank, row in enumerate(completed_rows, start=1):
        row["rank"] = int(rank)
    write_csv_manifest(output_root_path / "candidate_scores.csv", candidate_score_rows)

    try:
        best_candidate, best_row = _select_best_candidate_within_standard_error(
            candidates,
            candidate_score_rows,
        )
    except RuntimeError:
        raise RuntimeError(
            f"All candidates failed for experiment '{experiment_row.get('experiment_name', '')}' "
            f"search '{search['name']}'. See {output_root_path / 'fold_scores.csv'}."
        )

    OmegaConf.save(
        OmegaConf.create(
            _best_candidate_payload(
                experiment_row,
                search,
                best_candidate,
                best_row,
                execution_mode=execution_mode,
            )
        ),
        io_path(output_root_path / "best_candidate.yaml"),
    )
    return _manifest_row_from_best_row(
        experiment_row,
        search,
        best_candidate,
        best_row,
        output_root_path,
        execution_mode=execution_mode,
    )


def _fit_metric_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidate: dict[str, Any],
    *,
    execution_mode: str,
    fold_id: int,
    fit_root: Path,
    num_training_slots: int,
    num_validation_slots: int,
    num_post_s_validation_slots: int,
) -> dict[str, object]:
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "execution_mode": _normalize_execution_mode(execution_mode),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "split_source": _search_split_source(search),
        "outer_num_folds": int(_search_outer_num_folds(search)),
        "test_fold_id": int(_search_test_fold_id(search)),
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
        "cv_fold_id": int(fold_id),
        "num_training_slots": int(num_training_slots),
        "num_validation_slots": int(num_validation_slots),
        "num_post_s_validation_slots": int(num_post_s_validation_slots),
        "fit_path": str(fit_root.resolve()),
    }


def _evaluate_and_store_fold_metrics(
    row: dict[str, object],
    *,
    fit_root: Path,
    experiment_root: Path,
    training_loss_mask: np.ndarray,
    validation_loss_mask: np.ndarray,
    validation_sampling: dict[str, Any],
) -> None:
    metrics = evaluate_saved_fit_fold_metrics(
        fit_root,
        experiment_root,
        training_loss_mask=training_loss_mask,
        validation_loss_mask=validation_loss_mask,
        validation_sampling=validation_sampling,
    )
    row.update(
        {
            "status": "completed",
            "fit_loss": float(metrics["fit_loss"]),
            "validation_loss": float(metrics["validation_loss"]),
            "validation_brier_score": float(metrics["validation_brier_score"]),
            "validation_ece": float(metrics["validation_ece"]),
            "num_post_s_validation_slots": int(metrics["num_post_s_validation_slots"]),
            "post_s_validation_loss": _metric_or_blank(metrics["post_s_validation_loss"]),
            "post_s_validation_brier_score": _metric_or_blank(
                metrics["post_s_validation_brier_score"]
            ),
            "post_s_validation_ece": _metric_or_blank(metrics["post_s_validation_ece"]),
            "validation_mean_magnetization_abs_diff": float(
                metrics["validation_mean_magnetization_abs_diff"]
            ),
            "validation_observed_mean_magnetization": float(
                metrics["validation_observed_mean_magnetization"]
            ),
            "validation_sampled_mean_magnetization_mean": float(
                metrics["validation_sampled_mean_magnetization_mean"]
            ),
            "post_s_validation_mean_magnetization_abs_diff": _metric_or_blank(
                metrics["post_s_validation_mean_magnetization_abs_diff"]
            ),
            "post_s_validation_observed_mean_magnetization": _metric_or_blank(
                metrics["post_s_validation_observed_mean_magnetization"]
            ),
            "post_s_validation_sampled_mean_magnetization_mean": _metric_or_blank(
                metrics["post_s_validation_sampled_mean_magnetization_mean"]
            ),
        }
    )


def _load_search_from_spec(
    cv_spec_path: str | Path,
    search_slug: str,
) -> dict[str, Any]:
    searches = _expand_searches(cv_spec_path)
    search = next((item for item in searches if item.get("slug") == search_slug), None)
    if search is None:
        raise ValueError(f"Search slug '{search_slug}' not found in {cv_spec_path}.")
    return search


def _run_search_for_experiment(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    *,
    execution_mode: str = EXECUTION_MODE_CV,
    num_folds: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    normalized_mode = _normalize_execution_mode(execution_mode)
    configured_num_folds = (
        int(num_folds) if num_folds is not None else _get_num_folds_from_search(search)
    )
    selected_fold_ids = _resolved_fold_ids_for_experiment(
        experiment_row,
        {**search, "num_folds": int(configured_num_folds)},
        execution_mode=normalized_mode,
    )
    experiment_root = Path(experiment_row["experiment_path"]).resolve()
    output_root = _candidate_output_root(
        experiment_root,
        search,
        execution_mode=normalized_mode,
    )
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_artifacts = load_model_selection_split_masks(
        experiment_root,
        split_source=_search_split_source(search),
        num_folds=configured_num_folds,
        outer_num_folds=_search_outer_num_folds(search),
        test_fold_id=_search_test_fold_id(search),
    )
    training_masks = np.asarray(split_artifacts["training_masks"], dtype=bool)
    validation_masks = np.asarray(split_artifacts["validation_masks"], dtype=bool)
    separator_masks = np.asarray(split_artifacts["separator_masks"], dtype=bool)
    panel_context = load_experiment_panel_context(experiment_root)
    if training_masks.shape[1] != int(panel_context["T"]) or training_masks.shape[2] != int(
        panel_context["N"]
    ):
        raise ValueError(
            f"Split mask tensor shape {training_masks.shape} does not match panel dimensions "
            f"(T={panel_context['T']}, N={panel_context['N']}) for {experiment_root}."
        )

    validation_sampling = resolve_validation_sampling(search.get("validation_sampling"))
    candidates = expand_search_candidates(search)
    candidate_grid_rows = [
        _candidate_grid_row(
            search,
            candidate,
            execution_mode=normalized_mode,
        )
        for candidate in candidates
    ]
    write_csv_manifest(output_root / "candidate_grid.csv", candidate_grid_rows)

    fold_rows: list[dict[str, object]] = []
    for candidate in candidates:
        for fold_id in selected_fold_ids:
            fold_root = _candidate_fit_root(
                experiment_root,
                search,
                candidate,
                fold_id,
                execution_mode=normalized_mode,
            )
            if fold_root.exists():
                if overwrite:
                    shutil.rmtree(fold_root)
                else:
                    raise FileExistsError(
                        f"{fold_root} already exists. Re-run with --overwrite to rebuild it."
                    )
            fold_root.mkdir(parents=True, exist_ok=False)
            training_loss_mask = np.asarray(training_masks[int(fold_id) - 1], dtype=bool)
            validation_loss_mask = np.asarray(validation_masks[int(fold_id) - 1], dtype=bool)
            separator_loss_mask = np.asarray(separator_masks[int(fold_id) - 1], dtype=bool)
            num_training_slots = int(np.count_nonzero(training_loss_mask))
            num_validation_slots = int(np.count_nonzero(validation_loss_mask))
            post_s_validation_loss_mask = validation_loss_mask & time_window_mask(
                t_steps=training_loss_mask.shape[0],
                n_nodes=training_loss_mask.shape[1],
                start_t=int(panel_context["s"]),
            )
            num_post_s_validation_slots = int(
                np.count_nonzero(post_s_validation_loss_mask)
            )
            if num_training_slots <= 0 or num_validation_slots <= 0:
                raise ValueError(
                    f"Fold {fold_id} for {experiment_root} has empty training or validation support."
                )
            mask_path = save_loss_mask(fold_root / "loss_mask.npy", training_loss_mask)
            extra_metadata = {
                "execution_mode": normalized_mode,
                "search_name": search["name"],
                "search_slug": search["slug"],
                "cv_spec_path": str(Path(search["_spec_path"]).resolve()),
                "configured_num_folds": int(configured_num_folds),
                "split_source": _search_split_source(search),
                "outer_num_folds": int(_search_outer_num_folds(search)),
                "test_fold_id": int(_search_test_fold_id(search)),
                "cv_fold_id": int(fold_id),
                "candidate_name": candidate["name"],
                "candidate_slug": candidate["slug"],
                "candidate_index": int(candidate["_candidate_index"]),
                "num_training_slots": num_training_slots,
                "num_separator_slots": int(np.count_nonzero(separator_loss_mask)),
                "num_validation_slots": num_validation_slots,
                "num_post_s_validation_slots": num_post_s_validation_slots,
            }
            row = _fit_metric_row(
                experiment_row,
                search,
                candidate,
                execution_mode=normalized_mode,
                fold_id=fold_id,
                fit_root=fold_root,
                num_training_slots=num_training_slots,
                num_validation_slots=num_validation_slots,
                num_post_s_validation_slots=num_post_s_validation_slots,
            )
            try:
                materialize_fit_root(
                    experiment_row,
                    candidate,
                    fold_root,
                    extra_input_artifacts={"loss_mask_path": str(mask_path.resolve())},
                    extra_metadata=extra_metadata,
                )
                execute_fit_root(fold_root)
                _evaluate_and_store_fold_metrics(
                    row,
                    fit_root=fold_root,
                    experiment_root=experiment_root,
                    training_loss_mask=training_loss_mask,
                    validation_loss_mask=validation_loss_mask,
                    validation_sampling=validation_sampling,
                )
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "status": "failed",
                        **_blank_fold_metric_values(),
                        "error_message": str(exc),
                    }
                )
            fold_rows.append(row)
    return _write_search_score_artifacts(
        experiment_row,
        search,
        candidates,
        fold_rows,
        expected_num_folds=len(selected_fold_ids),
        output_root=output_root,
        execution_mode=normalized_mode,
    )


def run_cv_folds(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    *,
    execution_mode: str = EXECUTION_MODE_CV,
    overwrite: bool = False,
) -> Path:
    normalized_mode = _normalize_execution_mode(execution_mode)
    request_path = write_cv_requests(
        generation_manifest_path,
        cv_spec_path,
        execution_mode=normalized_mode,
    )
    generation_rows = read_csv_manifest(generation_manifest_path)
    searches = _expand_searches(cv_spec_path)
    manifest_rows: list[dict[str, object]] = []
    print(f"Loaded {len(generation_rows)} experiment(s) from {generation_manifest_path}.")
    print(
        f"Loaded {len(searches)} {normalized_mode} search(es) from {cv_spec_path}.",
        flush=True,
    )
    for experiment_row in generation_rows:
        for search in searches:
            print(
                f"Running {normalized_mode} search '{search['name']}' for experiment "
                f"'{experiment_row.get('experiment_name', experiment_row.get('experiment_slug', ''))}'...",
                flush=True,
            )
            manifest_rows.append(
                _run_search_for_experiment(
                    experiment_row,
                    search,
                    execution_mode=normalized_mode,
                    overwrite=overwrite,
                )
            )
    manifest_path = model_selection_manifest_path_for_spec(
        cv_spec_path,
        execution_mode=normalized_mode,
    )
    write_csv_manifest(manifest_path, manifest_rows)
    print(f"{normalized_mode} requests: {request_path}")
    print(f"{normalized_mode} manifest: {manifest_path}")
    return manifest_path


def run_cv_search_for_experiment_slug(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    experiment_slug: str,
    search_slug: str,
    *,
    execution_mode: str = EXECUTION_MODE_CV,
    num_folds: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    generation_rows = read_csv_manifest(generation_manifest_path)
    experiment_row = next(
        (row for row in generation_rows if row.get("experiment_slug") == experiment_slug),
        None,
    )
    if experiment_row is None:
        raise ValueError(
            f"Experiment slug '{experiment_slug}' not found in {generation_manifest_path}."
        )
    search = _load_search_from_spec(cv_spec_path, search_slug)
    resolved_num_folds = (
        int(num_folds) if num_folds is not None else _get_num_folds_from_search(search)
    )
    return _run_search_for_experiment(
        experiment_row,
        search,
        execution_mode=execution_mode,
        num_folds=resolved_num_folds,
        overwrite=overwrite,
    )


def refresh_cv_scores_from_requests(
    cv_requests_path: str | Path,
    *,
    execution_mode: str | None = None,
    cv_manifest_path: str | Path | None = None,
) -> Path:
    request_rows = read_csv_manifest(cv_requests_path)
    if not request_rows:
        raise ValueError(f"No CV requests found in {cv_requests_path}.")

    normalized_mode = (
        _normalize_execution_mode(execution_mode)
        if execution_mode is not None
        else _normalize_execution_mode(request_rows[0].get("execution_mode", EXECUTION_MODE_CV))
    )
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in request_rows:
        key = (str(row["experiment_slug"]), str(row["search_slug"]))
        grouped_rows.setdefault(key, []).append(row)

    manifest_rows: list[dict[str, object]] = []
    for _, group_rows in grouped_rows.items():
        first_row = group_rows[0]
        experiment_row = {
            "experiment_name": first_row.get("experiment_name", ""),
            "experiment_slug": first_row.get("experiment_slug", ""),
            "experiment_path": first_row.get("experiment_path", ""),
        }
        cv_spec_path = first_row.get("cv_spec_path", "")
        if not cv_spec_path:
            raise ValueError(
                f"Request row for experiment '{experiment_row['experiment_slug']}' is missing cv_spec_path."
            )
        search = _load_search_from_spec(cv_spec_path, str(first_row["search_slug"]))
        configured_num_folds = int(
            first_row.get("configured_num_folds", _get_num_folds_from_search(search))
        )
        search["split_source"] = str(
            first_row.get("split_source", _search_split_source(search))
        )
        search["outer_num_folds"] = int(
            first_row.get("outer_num_folds", _search_outer_num_folds(search))
        )
        search["test_fold_id"] = int(
            first_row.get("test_fold_id", _search_test_fold_id(search))
        )
        experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
        output_roots = {
            Path(str(row["fit_path"])).resolve().parents[2]
            for row in group_rows
        }
        if len(output_roots) != 1:
            raise ValueError(
                f"Expected a single output root for experiment '{experiment_row['experiment_slug']}' "
                f"search '{search['slug']}', found {len(output_roots)}."
            )
        output_root = next(iter(output_roots))
        output_root.mkdir(parents=True, exist_ok=True)
        candidates = _load_scored_candidates(output_root, group_rows)
        expected_num_folds = len({int(row["cv_fold_id"]) for row in group_rows})

        split_artifacts = load_model_selection_split_masks(
            experiment_root,
            split_source=_search_split_source(search),
            num_folds=configured_num_folds,
            outer_num_folds=_search_outer_num_folds(search),
            test_fold_id=_search_test_fold_id(search),
        )
        training_masks = np.asarray(split_artifacts["training_masks"], dtype=bool)
        validation_masks = np.asarray(split_artifacts["validation_masks"], dtype=bool)
        panel_context = load_experiment_panel_context(experiment_root)
        if training_masks.shape[1] != int(panel_context["T"]) or training_masks.shape[2] != int(
            panel_context["N"]
        ):
            raise ValueError(
                f"Split mask tensor shape {training_masks.shape} does not match panel dimensions "
                f"(T={panel_context['T']}, N={panel_context['N']}) for {experiment_root}."
            )
        validation_sampling = resolve_validation_sampling(search.get("validation_sampling"))

        fold_rows: list[dict[str, object]] = []
        sorted_group_rows = sorted(
            group_rows,
            key=lambda row: (int(row["candidate_index"]), int(row["cv_fold_id"])),
        )
        for request_row in sorted_group_rows:
            fold_id = int(request_row["cv_fold_id"])
            fit_root = Path(str(request_row["fit_path"])).resolve()
            training_loss_mask = np.asarray(training_masks[fold_id - 1], dtype=bool)
            validation_loss_mask = np.asarray(validation_masks[fold_id - 1], dtype=bool)
            post_s_validation_loss_mask = validation_loss_mask & time_window_mask(
                t_steps=training_loss_mask.shape[0],
                n_nodes=training_loss_mask.shape[1],
                start_t=int(panel_context["s"]),
            )
            candidate = {
                "name": request_row["candidate_name"],
                "slug": request_row["candidate_slug"],
                "_candidate_index": int(request_row["candidate_index"]),
            }
            row = _fit_metric_row(
                experiment_row,
                search,
                candidate,
                execution_mode=normalized_mode,
                fold_id=fold_id,
                fit_root=fit_root,
                num_training_slots=int(np.count_nonzero(training_loss_mask)),
                num_validation_slots=int(np.count_nonzero(validation_loss_mask)),
                num_post_s_validation_slots=int(np.count_nonzero(post_s_validation_loss_mask)),
            )
            try:
                _evaluate_and_store_fold_metrics(
                    row,
                    fit_root=fit_root,
                    experiment_root=experiment_root,
                    training_loss_mask=training_loss_mask,
                    validation_loss_mask=validation_loss_mask,
                    validation_sampling=validation_sampling,
                )
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "status": "failed",
                        **_blank_fold_metric_values(),
                        "error_message": str(exc),
                    }
                )
            fold_rows.append(row)

        manifest_rows.append(
            _write_search_score_artifacts(
                experiment_row,
                search,
                candidates,
                fold_rows,
                expected_num_folds=expected_num_folds,
                output_root=output_root,
                execution_mode=normalized_mode,
            )
        )

    manifest_path = (
        Path(cv_manifest_path)
        if cv_manifest_path is not None
        else model_selection_manifest_path_for_spec(
            first_row["cv_spec_path"],
            execution_mode=normalized_mode,
        )
    )
    write_csv_manifest(manifest_path, manifest_rows)
    return manifest_path


def collect_cv_manifest_from_requests(
    cv_requests_path: str | Path,
    *,
    execution_mode: str | None = None,
    cv_manifest_path: str | Path | None = None,
) -> Path:
    request_rows = read_csv_manifest(cv_requests_path)
    if not request_rows:
        raise ValueError(f"No CV requests found in {cv_requests_path}.")

    normalized_mode = (
        _normalize_execution_mode(execution_mode)
        if execution_mode is not None
        else _normalize_execution_mode(request_rows[0].get("execution_mode", EXECUTION_MODE_CV))
    )
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in request_rows:
        key = (str(row["experiment_slug"]), str(row["search_slug"]))
        grouped_rows.setdefault(key, []).append(row)

    manifest_rows: list[dict[str, object]] = []
    first_row: dict[str, str] | None = None
    for _, group_rows in grouped_rows.items():
        first_row = group_rows[0]
        experiment_row = {
            "experiment_name": first_row.get("experiment_name", ""),
            "experiment_slug": first_row.get("experiment_slug", ""),
            "experiment_path": first_row.get("experiment_path", ""),
        }
        cv_spec_path = first_row.get("cv_spec_path", "")
        if not cv_spec_path:
            raise ValueError(
                f"Request row for experiment '{experiment_row['experiment_slug']}' is missing cv_spec_path."
            )
        search = _load_search_from_spec(cv_spec_path, str(first_row["search_slug"]))
        configured_num_folds = int(
            first_row.get("configured_num_folds", _get_num_folds_from_search(search))
        )
        search["split_source"] = str(
            first_row.get("split_source", _search_split_source(search))
        )
        search["outer_num_folds"] = int(
            first_row.get("outer_num_folds", _search_outer_num_folds(search))
        )
        search["test_fold_id"] = int(
            first_row.get("test_fold_id", _search_test_fold_id(search))
        )
        output_roots = {
            Path(str(row["fit_path"])).resolve().parents[2]
            for row in group_rows
        }
        if len(output_roots) != 1:
            raise ValueError(
                f"Expected a single output root for experiment '{experiment_row['experiment_slug']}' "
                f"search '{search['slug']}', found {len(output_roots)}."
            )
        output_root = next(iter(output_roots))
        fold_scores_path = output_root / "fold_scores.csv"
        if not fold_scores_path.exists():
            raise FileNotFoundError(
                f"Missing fold-score artifact for experiment '{experiment_row['experiment_slug']}' "
                f"search '{search['slug']}': {fold_scores_path}"
            )
        candidates = expand_search_candidates(search)
        fold_rows = read_csv_manifest(fold_scores_path)
        expected_num_folds = len({int(row["cv_fold_id"]) for row in group_rows})
        manifest_rows.append(
            _write_search_score_artifacts(
                experiment_row,
                search,
                candidates,
                fold_rows,
                expected_num_folds=expected_num_folds,
                output_root=output_root,
                execution_mode=normalized_mode,
            )
        )

    manifest_path = (
        Path(cv_manifest_path)
        if cv_manifest_path is not None
        else model_selection_manifest_path_for_spec(
            first_row["cv_spec_path"],
            execution_mode=normalized_mode,
        )
    )
    write_csv_manifest(manifest_path, manifest_rows)
    return manifest_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CV or single-fold validation search over prebuilt spatiotemporal fold artifacts."
    )
    parser.add_argument("--generation_manifest_path", type=str)
    parser.add_argument("--cv_spec_path", type=str)
    parser.add_argument(
        "--cv_requests_path",
        type=str,
        help="Path to an existing request CSV file (used with --refresh_scores or --refresh_manifest).",
    )
    parser.add_argument(
        "--execution_mode",
        type=str,
        default=EXECUTION_MODE_CV,
        choices=sorted(VALID_EXECUTION_MODES),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs and print planned model-selection work without executing fits.",
    )
    parser.add_argument(
        "--write_requests",
        action="store_true",
        help="Write request CSVs for the configured generation manifest and CV spec.",
    )
    parser.add_argument(
        "--run_request",
        action="store_true",
        help="Run a specific experiment+search job.",
    )
    parser.add_argument(
        "--refresh_scores",
        action="store_true",
        help="Recompute fold and candidate scores from an existing request CSV without rerunning fits.",
    )
    parser.add_argument(
        "--refresh_manifest",
        action="store_true",
        help="Rebuild only the top-level model-selection manifest from existing best_candidate.yaml artifacts.",
    )
    parser.add_argument("--experiment_slug", type=str, help="Experiment slug (used with --run_request).")
    parser.add_argument("--search_slug", type=str, help="Search slug (used with --run_request).")
    parser.add_argument("--num_folds", type=int, help="Override number of folds from cv_spec.")
    args = parser.parse_args()

    normalized_mode = _normalize_execution_mode(args.execution_mode)

    if args.refresh_scores:
        if not args.cv_requests_path:
            raise ValueError("--refresh_scores requires --cv_requests_path.")
        manifest_path = refresh_cv_scores_from_requests(
            args.cv_requests_path,
            execution_mode=normalized_mode,
        )
        print(f"{normalized_mode} manifest: {manifest_path}")
        return

    if args.refresh_manifest:
        if not args.cv_requests_path:
            raise ValueError("--refresh_manifest requires --cv_requests_path.")
        manifest_path = collect_cv_manifest_from_requests(
            args.cv_requests_path,
            execution_mode=normalized_mode,
        )
        print(f"{normalized_mode} manifest: {manifest_path}")
        return

    if args.dry_run:
        if not args.generation_manifest_path or not args.cv_spec_path:
            raise ValueError("--dry_run requires both --generation_manifest_path and --cv_spec_path.")
        generation_rows = read_csv_manifest(args.generation_manifest_path)
        searches = _expand_searches(args.cv_spec_path)
        total_folds = 0
        total_candidates = 0
        for search in searches:
            configured_num_folds = (
                int(args.num_folds)
                if args.num_folds is not None
                else _get_num_folds_from_search(search)
            )
            candidates = expand_search_candidates(search)
            total_folds += len(candidates) * len(
                _selected_fold_ids(
                    configured_num_folds,
                    execution_mode=normalized_mode,
                )
            )
            total_candidates += len(candidates)
        print(
            f"Dry run: {len(generation_rows)} experiment(s) × {total_candidates} candidate(s) "
            f"× mode={normalized_mode} = {total_folds} total fit(s) planned."
        )
        return

    if args.write_requests:
        if not args.generation_manifest_path or not args.cv_spec_path:
            raise ValueError(
                "--write_requests requires both --generation_manifest_path and --cv_spec_path."
            )
        request_path = write_cv_requests(
            args.generation_manifest_path,
            args.cv_spec_path,
            execution_mode=normalized_mode,
        )
        print(f"{normalized_mode} requests: {request_path}")
        return

    if args.run_request:
        if not args.generation_manifest_path or not args.cv_spec_path:
            raise ValueError(
                "--run_request requires both --generation_manifest_path and --cv_spec_path."
            )
        if not args.experiment_slug or not args.search_slug:
            raise ValueError("--run_request requires both --experiment_slug and --search_slug.")
        run_cv_search_for_experiment_slug(
            args.generation_manifest_path,
            args.cv_spec_path,
            args.experiment_slug,
            args.search_slug,
            execution_mode=normalized_mode,
            num_folds=args.num_folds,
            overwrite=args.overwrite,
        )
        return

    if not args.generation_manifest_path or not args.cv_spec_path:
        raise ValueError(
            "Default execution requires both --generation_manifest_path and --cv_spec_path."
        )

    run_cv_folds(
        args.generation_manifest_path,
        args.cv_spec_path,
        execution_mode=normalized_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

