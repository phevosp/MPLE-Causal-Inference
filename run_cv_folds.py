"""Run 5-fold spatiotemporal cross-validation over hyperparameter grids."""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

import build_cv_folds as cv_folds
from io_utils import io_path
from loading_utils import load_experiment_panel_context, load_fit_parameter_bundle
from model_utils import interaction_effect
from mple import evaluate_mple_loss_from_parts
from pipeline_specs import (
    deep_merge,
    expand_named_entries,
    read_csv_manifest,
    slugify,
    validate_cv_spec,
    validate_fit_variant_dict,
    write_csv_manifest,
)
from run_fit_pipeline import execute_fit_root, materialize_fit_root


DEFAULT_NUM_FOLDS = 5
CV_REQUESTS_NAME = "cv_requests.csv"
ECE_NUM_BINS = 10


def _get_num_folds_from_search(search: dict[str, Any]) -> int:
    """Extract num_folds from search configuration, defaults to DEFAULT_NUM_FOLDS."""
    num_folds = search.get("num_folds")
    if num_folds is not None:
        return int(num_folds)
    return DEFAULT_NUM_FOLDS


def _read_yaml_mapping(path: str | Path) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping data in {path}.")
    return loaded


def _expand_searches(cv_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_cv_spec(cv_spec_path)
    searches = expand_named_entries(cv_spec_path, "searches")
    if not searches:
        raise ValueError(f"No searches found in CV spec {cv_spec_path}.")
    for search in searches:
        search["_spec_path"] = str(Path(cv_spec_path).resolve())
    return searches


def cv_manifest_path_for_spec(cv_spec_path: str | Path) -> Path:
    searches = _expand_searches(cv_spec_path)
    return Path(str(searches[0]["cv_manifest_path"]))


def cv_requests_path_for_spec(cv_spec_path: str | Path) -> Path:
    return cv_manifest_path_for_spec(cv_spec_path).with_name(CV_REQUESTS_NAME)


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


def _flatten_mapping(mapping: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, prefix=dotted))
        else:
            flat[dotted] = value
    return flat


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


def _load_fold_roles(experiment_root: str | Path, *, num_folds: int) -> np.ndarray:
    output_root = Path(experiment_root) / "cv_folds" / f"folds_{num_folds}"
    blanket_summary = _read_yaml_mapping(output_root / "markov_blanket_summary.yaml")
    if not bool(blanket_summary.get("blanket_validation_passed", False)):
        raise ValueError(
            f"CV folds at {output_root} failed Markov blanket validation."
        )
    metadata = _read_yaml_mapping(output_root / "spatiotemporal_cv_metadata.yaml")
    if int(metadata.get("num_cv_folds", 0)) != num_folds:
        raise ValueError(
            f"Expected {num_folds} folds in {output_root}; found "
            f"{metadata.get('num_cv_folds')}."
        )
    with np.load(output_root / "fold_roles.npz", allow_pickle=False) as data:
        role_codes = np.asarray(data["role_codes"], dtype=np.int8)
    if role_codes.ndim != 3 or role_codes.shape[0] != num_folds:
        raise ValueError(
            f"fold_roles.npz at {output_root} has invalid role tensor shape {role_codes.shape}."
        )
    return role_codes


def _candidate_output_root(experiment_root: str | Path, search: dict[str, Any]) -> Path:
    cv_root_name = str(search.get("cv_root_name", "cv_runs"))
    return Path(experiment_root) / cv_root_name / str(search["slug"])


def _candidate_fit_root(
    experiment_root: str | Path,
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_id: int,
) -> Path:
    return (
        _candidate_output_root(experiment_root, search)
        / "candidates"
        / str(candidate["slug"])
        / f"fold_{int(fold_id)}"
    )


def _cv_request_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_id: int,
) -> dict[str, object]:
    fit_root = _candidate_fit_root(experiment_row["experiment_path"], search, candidate, fold_id)
    return {
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
) -> Path:
    generation_rows = read_csv_manifest(generation_manifest_path)
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {generation_manifest_path}."
        )
    request_rows: list[dict[str, object]] = []
    for search in _expand_searches(cv_spec_path):
        num_folds = _get_num_folds_from_search(search)
        for candidate in expand_search_candidates(search):
            for experiment_row in generation_rows:
                for fold_id in range(1, num_folds + 1):
                    request_rows.append(
                        _cv_request_row(experiment_row, search, candidate, fold_id)
                    )
    request_path = cv_requests_path_for_spec(cv_spec_path)
    write_csv_manifest(request_path, request_rows)
    return request_path


def _save_loss_mask(path: str | Path, loss_mask: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(io_path(output_path), np.asarray(loss_mask, dtype=bool))
    return output_path


def _masked_beta_feature(
    z: np.ndarray,
    *,
    s: int,
    e: int,
    beta_mask_pre_s: bool,
    beta_mask_post_e: bool,
) -> np.ndarray:
    beta_feature = np.asarray(z, dtype=float).copy()
    if bool(beta_mask_pre_s) and int(s) > 0:
        beta_feature[: int(s), :] = 0.0
    if bool(beta_mask_post_e) and int(e) < beta_feature.shape[0]:
        beta_feature[int(e) :, :] = 0.0
    return beta_feature


def _validation_brier_score(
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


def _validation_expected_calibration_error(
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
    if num_bins <= 0:
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
    for bin_index in range(num_bins):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        bin_fraction = float(np.count_nonzero(in_bin)) / total_count
        empirical_rate = float(np.mean(observed_positive[in_bin]))
        mean_predicted_probability = float(np.mean(predicted_positive[in_bin]))
        ece += bin_fraction * abs(empirical_rate - mean_predicted_probability)
    return float(ece)


def _evaluate_fold_metrics(
    fit_root: str | Path,
    experiment_root: str | Path,
    *,
    training_loss_mask: np.ndarray,
    validation_loss_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    panel_context = load_experiment_panel_context(experiment_root)
    bundle = load_fit_parameter_bundle(fit_root, experiment_root)
    x = np.asarray(panel_context["x"], dtype=float)
    z = np.asarray(panel_context["z"], dtype=float)
    x_0 = np.asarray(panel_context["x_0"], dtype=float)
    field_matrix = np.asarray(bundle.field_matrix, dtype=float)
    beta = float(bundle.beta)
    xi = float(bundle.xi)
    eta = float(bundle.eta)
    interaction_effect_x = interaction_effect(x, bundle.gamma_matrix)
    common_kwargs = {
        "x": x,
        "z": z,
        "x_0": x_0,
        "field_matrix": field_matrix,
        "beta": beta,
        "xi": xi,
        "eta": eta,
        "interaction_effect_x": interaction_effect_x,
        "fixed_scalar_params": {},
        "s": int(panel_context["s"]),
        "e": int(panel_context["e"]),
        "beta_mask_pre_s": bool(bundle.beta_mask_pre_s),
        "beta_mask_post_e": bool(bundle.beta_mask_post_e),
    }
    fit_loss = evaluate_mple_loss_from_parts(
        loss_mask=np.asarray(training_loss_mask, dtype=bool),
        **common_kwargs,
    )
    validation_loss = evaluate_mple_loss_from_parts(
        loss_mask=np.asarray(validation_loss_mask, dtype=bool),
        **common_kwargs,
    )
    prev_x = np.vstack([x_0, x[:-1, :]])
    h_x = (
        field_matrix
        + beta
        * _masked_beta_feature(
            z,
            s=int(panel_context["s"]),
            e=int(panel_context["e"]),
            beta_mask_pre_s=bool(bundle.beta_mask_pre_s),
            beta_mask_post_e=bool(bundle.beta_mask_post_e),
        )
        + xi * interaction_effect_x
        + eta * prev_x
    )
    validation_brier_score = _validation_brier_score(
        x=x,
        h_x=h_x,
        loss_mask=np.asarray(validation_loss_mask, dtype=bool),
    )
    validation_ece = _validation_expected_calibration_error(
        x=x,
        h_x=h_x,
        loss_mask=np.asarray(validation_loss_mask, dtype=bool),
    )
    return (
        float(fit_loss),
        float(validation_loss),
        float(validation_brier_score),
        float(validation_ece),
    )


def _candidate_grid_row(search: dict[str, Any], candidate: dict[str, Any]) -> dict[str, object]:
    row: dict[str, object] = {
        "search_name": search["name"],
        "search_slug": search["slug"],
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
    }
    for key, value in sorted(candidate["_flat_params"].items()):
        row[key] = value
    return row


def _candidate_score_row(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    candidate: dict[str, Any],
    fold_rows: list[dict[str, object]],
    *,
    num_folds: int,
) -> dict[str, object]:
    success_rows = [row for row in fold_rows if row.get("status") == "completed"]
    if len(success_rows) != num_folds:
        return {
            "experiment_name": experiment_row.get("experiment_name", ""),
            "experiment_slug": experiment_row.get("experiment_slug", ""),
            "search_name": search["name"],
            "search_slug": search["slug"],
            "candidate_name": candidate["name"],
            "candidate_slug": candidate["slug"],
            "candidate_index": int(candidate["_candidate_index"]),
            "status": "failed",
            "num_completed_folds": int(len(success_rows)),
            "weighted_mean_validation_loss": "",
            "mean_fold_validation_loss": "",
            "weighted_mean_validation_brier_score": "",
            "mean_fold_validation_brier_score": "",
            "weighted_mean_validation_ece": "",
            "mean_fold_validation_ece": "",
            "total_validation_slots": "",
        }
    validation_slots = np.asarray(
        [int(row["num_validation_slots"]) for row in success_rows],
        dtype=float,
    )
    validation_losses = np.asarray(
        [float(row["validation_loss"]) for row in success_rows],
        dtype=float,
    )
    validation_brier_scores = np.asarray(
        [float(row["validation_brier_score"]) for row in success_rows],
        dtype=float,
    )
    validation_eces = np.asarray(
        [float(row["validation_ece"]) for row in success_rows],
        dtype=float,
    )
    weighted_mean = float(
        np.sum(validation_slots * validation_losses) / np.sum(validation_slots)
    )
    mean_fold = float(np.mean(validation_losses))
    weighted_mean_brier = float(
        np.sum(validation_slots * validation_brier_scores) / np.sum(validation_slots)
    )
    mean_fold_brier = float(np.mean(validation_brier_scores))
    weighted_mean_ece = float(
        np.sum(validation_slots * validation_eces) / np.sum(validation_slots)
    )
    mean_fold_ece = float(np.mean(validation_eces))
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "candidate_name": candidate["name"],
        "candidate_slug": candidate["slug"],
        "candidate_index": int(candidate["_candidate_index"]),
        "status": "completed",
        "num_completed_folds": int(len(success_rows)),
        "weighted_mean_validation_loss": weighted_mean,
        "mean_fold_validation_loss": mean_fold,
        "weighted_mean_validation_brier_score": weighted_mean_brier,
        "mean_fold_validation_brier_score": mean_fold_brier,
        "weighted_mean_validation_ece": weighted_mean_ece,
        "mean_fold_validation_ece": mean_fold_ece,
        "total_validation_slots": int(np.sum(validation_slots)),
    }


def _candidate_score_sort_key(row: dict[str, object]) -> tuple[float, float, int]:
    return (
        float(row["weighted_mean_validation_brier_score"]),
        float(row["weighted_mean_validation_loss"]),
        int(row["candidate_index"]),
    )


def _run_search_for_experiment(
    experiment_row: dict[str, str],
    search: dict[str, Any],
    *,
    num_folds: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if num_folds is None:
        num_folds = _get_num_folds_from_search(search)
    experiment_root = Path(experiment_row["experiment_path"]).resolve()
    output_root = _candidate_output_root(experiment_root, search)
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    role_codes = _load_fold_roles(experiment_root, num_folds=num_folds)
    panel_context = load_experiment_panel_context(experiment_root)
    if role_codes.shape[1] != int(panel_context["T"]) or role_codes.shape[2] != int(
        panel_context["N"]
    ):
        raise ValueError(
            f"CV fold tensor shape {role_codes.shape} does not match panel dimensions "
            f"(T={panel_context['T']}, N={panel_context['N']}) for {experiment_root}."
        )

    candidates = expand_search_candidates(search)
    candidate_grid_rows = [_candidate_grid_row(search, candidate) for candidate in candidates]
    write_csv_manifest(output_root / "candidate_grid.csv", candidate_grid_rows)

    fold_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_candidate: dict[str, Any] | None = None

    for candidate in candidates:
        candidate_fold_rows: list[dict[str, object]] = []
        for fold_id in range(1, num_folds + 1):
            fold_root = _candidate_fit_root(experiment_root, search, candidate, fold_id)
            if fold_root.exists():
                if overwrite:
                    shutil.rmtree(fold_root)
                else:
                    raise FileExistsError(
                        f"{fold_root} already exists. Re-run with --overwrite to rebuild it."
                    )
            fold_root.mkdir(parents=True, exist_ok=False)
            fold_roles = np.asarray(role_codes[fold_id - 1], dtype=np.int8)
            training_loss_mask = fold_roles == cv_folds.ROLE_CODE_TRAINING
            validation_loss_mask = fold_roles == cv_folds.ROLE_CODE_VALIDATION
            num_training_slots = int(np.count_nonzero(training_loss_mask))
            num_validation_slots = int(np.count_nonzero(validation_loss_mask))
            if num_training_slots <= 0 or num_validation_slots <= 0:
                raise ValueError(
                    f"Fold {fold_id} for {experiment_root} has empty training or validation support."
                )
            mask_path = _save_loss_mask(fold_root / "loss_mask.npy", training_loss_mask)
            extra_metadata = {
                "search_name": search["name"],
                "search_slug": search["slug"],
                "cv_spec_path": str(Path(search["_spec_path"]).resolve()),
                "cv_fold_id": int(fold_id),
                "candidate_name": candidate["name"],
                "candidate_slug": candidate["slug"],
                "candidate_index": int(candidate["_candidate_index"]),
                "num_training_slots": num_training_slots,
                "num_separator_slots": int(
                    np.count_nonzero(fold_roles == cv_folds.ROLE_CODE_SEPARATOR)
                ),
                "num_validation_slots": num_validation_slots,
            }
            row = {
                "experiment_name": experiment_row.get("experiment_name", ""),
                "experiment_slug": experiment_row.get("experiment_slug", ""),
                "search_name": search["name"],
                "search_slug": search["slug"],
                "candidate_name": candidate["name"],
                "candidate_slug": candidate["slug"],
                "candidate_index": int(candidate["_candidate_index"]),
                "cv_fold_id": int(fold_id),
                "num_training_slots": num_training_slots,
                "num_validation_slots": num_validation_slots,
                "fit_path": str(fold_root.resolve()),
            }
            try:
                materialize_fit_root(
                    experiment_row,
                    candidate,
                    fold_root,
                    extra_input_artifacts={"loss_mask_path": str(mask_path.resolve())},
                    extra_metadata=extra_metadata,
                )
                execute_fit_root(fold_root)
                (
                    fit_loss,
                    validation_loss,
                    validation_brier_score,
                    validation_ece,
                ) = _evaluate_fold_metrics(
                    fold_root,
                    experiment_root,
                    training_loss_mask=training_loss_mask,
                    validation_loss_mask=validation_loss_mask,
                )
                row.update(
                    {
                        "status": "completed",
                        "fit_loss": float(fit_loss),
                        "validation_loss": float(validation_loss),
                        "validation_brier_score": float(validation_brier_score),
                        "validation_ece": float(validation_ece),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "status": "failed",
                        "fit_loss": "",
                        "validation_loss": "",
                        "validation_brier_score": "",
                        "validation_ece": "",
                        "error_message": str(exc),
                    }
                )
            fold_rows.append(row)
            candidate_fold_rows.append(row)

        candidate_score = _candidate_score_row(
            experiment_row,
            search,
            candidate,
            candidate_fold_rows,
            num_folds=num_folds,
        )
        candidate_score_rows.append(candidate_score)
        if candidate_score["status"] != "completed":
            continue
        if best_row is None or _candidate_score_sort_key(candidate_score) < _candidate_score_sort_key(
            best_row
        ):
            best_row = candidate_score
            best_candidate = candidate

    write_csv_manifest(output_root / "fold_scores.csv", fold_rows)
    completed_rows = sorted(
        [row for row in candidate_score_rows if row.get("status") == "completed"],
        key=_candidate_score_sort_key,
    )
    for rank, row in enumerate(completed_rows, start=1):
        row["rank"] = int(rank)
    write_csv_manifest(output_root / "candidate_scores.csv", candidate_score_rows)

    if best_row is None or best_candidate is None:
        raise RuntimeError(
            f"All candidates failed for experiment '{experiment_row.get('experiment_name', '')}' "
            f"search '{search['name']}'. See {output_root / 'fold_scores.csv'}."
        )

    best_candidate_payload = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "candidate_name": best_candidate["name"],
        "candidate_slug": best_candidate["slug"],
        "candidate_index": int(best_candidate["_candidate_index"]),
        "weighted_mean_validation_loss": float(best_row["weighted_mean_validation_loss"]),
        "mean_fold_validation_loss": float(best_row["mean_fold_validation_loss"]),
        "weighted_mean_validation_brier_score": float(
            best_row["weighted_mean_validation_brier_score"]
        ),
        "mean_fold_validation_brier_score": float(
            best_row["mean_fold_validation_brier_score"]
        ),
        "weighted_mean_validation_ece": float(best_row["weighted_mean_validation_ece"]),
        "mean_fold_validation_ece": float(best_row["mean_fold_validation_ece"]),
        "total_validation_slots": int(best_row["total_validation_slots"]),
        "hyperparameters": {
            key: value for key, value in sorted(best_candidate["_flat_params"].items())
        },
    }
    OmegaConf.save(
        OmegaConf.create(best_candidate_payload),
        io_path(output_root / "best_candidate.yaml"),
    )
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "experiment_path": str(experiment_root),
        "search_name": search["name"],
        "search_slug": search["slug"],
        "output_path": str(output_root.resolve()),
        "best_candidate_name": best_candidate["name"],
        "best_candidate_slug": best_candidate["slug"],
        "weighted_mean_validation_loss": float(best_row["weighted_mean_validation_loss"]),
        "mean_fold_validation_loss": float(best_row["mean_fold_validation_loss"]),
        "weighted_mean_validation_brier_score": float(
            best_row["weighted_mean_validation_brier_score"]
        ),
        "mean_fold_validation_brier_score": float(
            best_row["mean_fold_validation_brier_score"]
        ),
        "weighted_mean_validation_ece": float(best_row["weighted_mean_validation_ece"]),
        "mean_fold_validation_ece": float(best_row["mean_fold_validation_ece"]),
        "total_validation_slots": int(best_row["total_validation_slots"]),
        "status": "completed",
    }


def run_cv_folds(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    request_path = write_cv_requests(generation_manifest_path, cv_spec_path)
    generation_rows = read_csv_manifest(generation_manifest_path)
    searches = _expand_searches(cv_spec_path)
    manifest_rows: list[dict[str, object]] = []
    print(f"Loaded {len(generation_rows)} experiment(s) from {generation_manifest_path}.")
    print(f"Loaded {len(searches)} CV search(es) from {cv_spec_path}.")
    for experiment_row in generation_rows:
        for search in searches:
            print(
                f"Running CV search '{search['name']}' for experiment "
                f"'{experiment_row.get('experiment_name', experiment_row.get('experiment_slug', ''))}'..."
            )
            manifest_rows.append(
                _run_search_for_experiment(
                    experiment_row,
                    search,
                    overwrite=overwrite,
                )
            )
    manifest_path = cv_manifest_path_for_spec(cv_spec_path)
    write_csv_manifest(manifest_path, manifest_rows)
    print(f"CV requests: {request_path}")
    print(f"CV manifest: {manifest_path}")
    return manifest_path


def run_cv_search_for_experiment_slug(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    experiment_slug: str,
    search_slug: str,
    *,
    num_folds: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run CV for a specific experiment and search combination.

    Args:
        num_folds: Override the number of folds from cv_spec. If None, uses cv_spec default.
    """
    generation_rows = read_csv_manifest(generation_manifest_path)
    experiment_row = next(
        (row for row in generation_rows if row.get("experiment_slug") == experiment_slug),
        None,
    )
    if experiment_row is None:
        raise ValueError(f"Experiment slug '{experiment_slug}' not found in {generation_manifest_path}.")

    searches = _expand_searches(cv_spec_path)
    search = next(
        (s for s in searches if s.get("slug") == search_slug),
        None,
    )
    if search is None:
        raise ValueError(f"Search slug '{search_slug}' not found in {cv_spec_path}.")

    if num_folds is None:
        num_folds = _get_num_folds_from_search(search)
    return _run_search_for_experiment(
        experiment_row,
        search,
        num_folds=num_folds,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5-fold hyperparameter CV over prebuilt spatiotemporal fold artifacts."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--cv_spec_path", required=True, type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs and print planned CV work without executing fits.",
    )
    parser.add_argument(
        "--write_requests",
        action="store_true",
        help="Write cv_requests.csv for the configured generation manifest and CV spec.",
    )
    parser.add_argument(
        "--run_request",
        action="store_true",
        help="Run a specific experiment+search CV job.",
    )
    parser.add_argument("--experiment_slug", type=str, help="Experiment slug (used with --run_request).")
    parser.add_argument("--search_slug", type=str, help="Search slug (used with --run_request).")
    parser.add_argument("--num_folds", type=int, help="Override number of CV folds from cv_spec.")
    args = parser.parse_args()

    if args.dry_run:
        generation_rows = read_csv_manifest(args.generation_manifest_path)
        searches = _expand_searches(args.cv_spec_path)
        total_folds = 0
        total_candidates = 0
        for search in searches:
            num_folds = _get_num_folds_from_search(search)
            candidates = expand_search_candidates(search)
            total_folds += len(candidates) * num_folds
            total_candidates += len(candidates)
        print(
            f"Dry run: {len(generation_rows)} experiment(s) × {total_candidates} candidate(s) "
            f"× variable folds = {total_folds} total fold(s) planned."
        )
        return

    if args.write_requests:
        request_path = write_cv_requests(
            args.generation_manifest_path,
            args.cv_spec_path,
        )
        print(f"CV requests: {request_path}")
        return

    if args.run_request:
        if not args.experiment_slug or not args.search_slug:
            raise ValueError("--run_request requires both --experiment_slug and --search_slug.")
        generation_rows = read_csv_manifest(args.generation_manifest_path)
        experiment_row = next(
            (row for row in generation_rows if row.get("experiment_slug") == args.experiment_slug),
            None,
        )
        if experiment_row is None:
            raise ValueError(
                f"Experiment slug '{args.experiment_slug}' not found in {args.generation_manifest_path}."
            )

        searches = _expand_searches(args.cv_spec_path)
        search = next(
            (s for s in searches if s.get("slug") == args.search_slug),
            None,
        )
        if search is None:
            raise ValueError(f"Search slug '{args.search_slug}' not found in {args.cv_spec_path}.")

        num_folds = args.num_folds if args.num_folds is not None else _get_num_folds_from_search(search)
        _run_search_for_experiment(
            experiment_row,
            search,
            num_folds=num_folds,
            overwrite=args.overwrite,
        )
        return

    run_cv_folds(
        args.generation_manifest_path,
        args.cv_spec_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
