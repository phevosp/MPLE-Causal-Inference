"""Run MPLE fits over a generation manifest.

Supports two complementary execution modes:

- ``standard``: fit explicit variants from a fits spec
- ``outer_masked``: fit one best-CV candidate per experiment on the outer
  training region defined by a split bundle

Both modes support the same staged workflows:

- plan requests into a CSV
- execute one planned request
- refresh a manifest from completed fit outputs and rebuild reports
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from utils.t0_config_utils import deep_merge_mappings, load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows, write_csv_rows
from utils.t0_path_utils import io_path, path_exists
from utils.t1_matrix_io import save_loss_mask
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t6_fit_materialization import (
    execute_fit_root,
    materialize_fit_root,
    path_text,
)
from utils.t6_pipeline_spec_utils import (
    best_candidate_path_for_search,
    expand_named_entries,
    load_search_from_spec,
    validate_cv_spec,
    validate_fits_spec,
)
from utils.t6_split_management import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    load_outer_training_split_masks,
    normalize_split_kind,
)
from utils.t8_parameter_recovery_reporting import write_fit_reports

FIT_REQUESTS_NAME = "fit_requests.csv"
FIT_MODE_STANDARD = "standard"
FIT_MODE_OUTER_MASKED = "outer_masked"
VALID_FIT_MODES = frozenset({FIT_MODE_STANDARD, FIT_MODE_OUTER_MASKED})
DEFAULT_TRAIN_NUM_FOLDS = 5
TRAIN_FIT_ROOT_NAME = "train_fits"
TRAIN_FIT_MANIFEST_NAME = "train_fit_manifest.csv"
TRAIN_FIT_SUMMARY_NAME = "train_fit_summary.csv"
BEST_TRAIN_FIT_BY_EXPERIMENT_NAME = "best_train_fit_by_experiment.csv"


def _normalize_fit_mode(fit_mode: str) -> str:
    normalized = str(fit_mode).strip().lower()
    if normalized not in VALID_FIT_MODES:
        raise ValueError(
            f"fit_mode must be one of {sorted(VALID_FIT_MODES)}, got '{fit_mode}'."
        )
    return normalized


def _expand_fit_variants(fits_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_fits_spec(fits_spec_path)
    variants = expand_named_entries(fits_spec_path, "variants")
    if not variants:
        raise ValueError(f"No variants found in fit spec {fits_spec_path}.")
    for variant in variants:
        variant["_spec_path"] = path_text(fits_spec_path)
    return variants


def _expand_train_fit_searches(cv_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_cv_spec(cv_spec_path)
    searches = expand_named_entries(cv_spec_path, "searches")
    if not searches:
        raise ValueError(f"No searches found in CV spec {cv_spec_path}.")
    for search in searches:
        search["_spec_path"] = str(Path(cv_spec_path).resolve())
    return searches


def fit_manifest_path_for_spec(fits_spec_path: str | Path) -> Path:
    """Return path to the fit manifest for a given fits spec."""
    variants = _expand_fit_variants(fits_spec_path)
    return Path(str(variants[0]["fit_manifest_path"]))


def fit_requests_path_for_spec(fits_spec_path: str | Path) -> Path:
    """Return path to the fit requests for a given fits spec."""
    return fit_manifest_path_for_spec(fits_spec_path).with_name(FIT_REQUESTS_NAME)


def train_fit_manifest_path_for_scope(
    generation_manifest_path: str | Path,
    search_slug: str | None = None,
) -> Path:
    """Return the refreshed train-fit manifest path for an outer-masked fit scope."""
    return _train_fit_manifest_path(generation_manifest_path, search_slug=search_slug)


def _train_fit_requests_path(
    manifest_path: str | Path,
    search_slug: str,
) -> Path:
    return Path(manifest_path).resolve().parent / (
        f"train_fit_requests__{str(search_slug).strip()}.csv"
    )


def _train_fit_manifest_path(
    manifest_path: str | Path,
    search_slug: str | None = None,
) -> Path:
    if search_slug in (None, ""):
        return Path(manifest_path).resolve().parent / TRAIN_FIT_MANIFEST_NAME
    return Path(manifest_path).resolve().parent / (
        f"train_fit_manifest__{str(search_slug).strip()}.csv"
    )


def _train_fit_report_filenames(
    search_slug: str | None = None,
) -> tuple[str, str]:
    if search_slug in (None, ""):
        return TRAIN_FIT_SUMMARY_NAME, BEST_TRAIN_FIT_BY_EXPERIMENT_NAME
    slug = str(search_slug).strip()
    return (
        f"train_fit_summary__{slug}.csv",
        f"best_train_fit_by_experiment__{slug}.csv",
    )


def _requested_train_fit_search_slugs(
    cv_spec_path: str | Path,
    search_slug: str | None,
) -> list[str]:
    normalized = str(search_slug or "").strip()
    if normalized:
        return [str(load_search_from_spec(cv_spec_path, normalized)["slug"])]
    return [str(search["slug"]) for search in _expand_train_fit_searches(cv_spec_path)]


def _fit_request_row(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    generation_manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> dict[str, object]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_path = experiment_root / str(variant["fit_root_name"]) / str(variant["slug"])
    return {
        "generation_manifest_path": path_text(generation_manifest_path),
        "fits_spec_path": path_text(fits_spec_path),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "variant_name": str(variant["name"]),
        "variant_slug": str(variant["slug"]),
        "fit_path": path_text(fit_path),
    }


def _select_generation_row(
    manifest_path: str | Path,
    experiment_slug: str,
) -> dict[str, str]:
    matches = [
        row
        for row in read_csv_rows(manifest_path)
        if row.get("experiment_slug", "") == experiment_slug
    ]
    if not matches:
        raise ValueError(
            f"No experiment with slug '{experiment_slug}' found in {manifest_path}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Experiment slug '{experiment_slug}' is not unique in {manifest_path}."
        )
    return matches[0]


def _select_fit_variant(
    fits_spec_path: str | Path,
    variant_slug: str,
) -> dict[str, Any]:
    matches = [
        variant
        for variant in _expand_fit_variants(fits_spec_path)
        if str(variant["slug"]) == variant_slug
    ]
    if not matches:
        raise ValueError(
            f"No fit variant with slug '{variant_slug}' found in {fits_spec_path}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Fit variant slug '{variant_slug}' is not unique in {fits_spec_path}."
        )
    return matches[0]


def _assign_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in str(dotted_path).split(".") if part]
    cursor = target
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    if parts:
        cursor[parts[-1]] = value


def _load_best_candidate_payload(best_candidate_path: str | Path) -> dict[str, Any]:
    if not path_exists(best_candidate_path):
        raise FileNotFoundError(f"Best candidate YAML not found: {best_candidate_path}")
    config = OmegaConf.load(io_path(best_candidate_path))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Best candidate YAML at {best_candidate_path} must be a mapping.")
    return dict(payload)


def _variant_from_best_candidate_payload(
    payload: dict[str, Any],
    *,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_config = payload.get("candidate_config")
    hyperparameters = payload.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}

    if isinstance(candidate_config, dict):
        variant = dict(candidate_config)
    elif search is not None:
        search_base = {key: value for key, value in dict(search).items() if key != "grid"}
        overrides: dict[str, Any] = {}
        for dotted_key, value in hyperparameters.items():
            _assign_nested_value(overrides, str(dotted_key), value)
        variant = deep_merge_mappings(search_base, overrides)
    else:
        variant = {}
        for dotted_key, value in hyperparameters.items():
            _assign_nested_value(variant, str(dotted_key), value)

    variant["name"] = str(payload.get("candidate_name", variant.get("name", "train_fit_candidate")))
    variant["slug"] = str(payload.get("candidate_slug", variant.get("slug", "train_fit_candidate")))
    variant["_candidate_index"] = int(payload.get("candidate_index", 0))
    variant["_flat_params"] = dict(hyperparameters)
    if search is not None and search.get("_spec_path"):
        variant["_cv_spec_path"] = str(search["_spec_path"])
    return variant


def _resolve_train_fit_split_settings(
    search: dict[str, Any],
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> dict[str, int | str]:
    resolved_split_kind = normalize_split_kind(
        split_kind if split_kind not in (None, "") else search.get("split_kind", "train_cv")
    )
    resolved_num_folds = (
        int(num_folds)
        if num_folds is not None
        else int(search.get("num_folds", DEFAULT_TRAIN_NUM_FOLDS))
    )
    resolved_outer_num_folds = (
        int(outer_num_folds)
        if outer_num_folds is not None
        else int(search.get("outer_num_folds", DEFAULT_OUTER_NUM_FOLDS))
    )
    resolved_test_fold_id = (
        int(test_fold_id)
        if test_fold_id is not None
        else int(search.get("test_fold_id", DEFAULT_TEST_FOLD_ID))
    )
    return {
        "split_kind": resolved_split_kind,
        "num_folds": resolved_num_folds,
        "outer_num_folds": resolved_outer_num_folds,
        "test_fold_id": resolved_test_fold_id,
    }


def _train_fit_output_root(
    experiment_root: str | Path,
    *,
    search_slug: str,
    split_kind: str,
    num_folds: int,
    outer_num_folds: int,
    test_fold_id: int,
    candidate_slug: str,
) -> Path:
    experiment_path = Path(experiment_root).resolve()
    if str(split_kind) == "train_cv":
        split_dir = f"train_cv__folds_{int(num_folds)}"
    else:
        split_dir = (
            f"test_train_cv__outer_{int(outer_num_folds)}"
            f"__test_{int(test_fold_id)}__inner_{int(num_folds)}"
        )
    return (
        experiment_path
        / TRAIN_FIT_ROOT_NAME
        / str(search_slug)
        / split_dir
        / str(candidate_slug)
    )


def _train_fit_request_row(
    experiment_row: dict[str, str],
    *,
    search: dict[str, Any],
    candidate: dict[str, Any],
    best_candidate_path: str | Path,
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    split_kind: str,
    num_folds: int,
    outer_num_folds: int,
    test_fold_id: int,
) -> dict[str, object]:
    fit_path = _train_fit_output_root(
        experiment_row["experiment_path"],
        search_slug=str(search["slug"]),
        split_kind=str(split_kind),
        num_folds=int(num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        candidate_slug=str(candidate["slug"]),
    )
    row: dict[str, object] = {
        "generation_manifest_path": path_text(generation_manifest_path),
        "cv_spec_path": path_text(cv_spec_path),
        "search_name": str(search["name"]),
        "search_slug": str(search["slug"]),
        "best_candidate_path": path_text(best_candidate_path),
        "split_kind": str(split_kind),
        "num_folds": int(num_folds),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "variant_name": str(candidate["name"]),
        "variant_slug": str(candidate["slug"]),
        "fit_path": path_text(fit_path),
    }
    if str(split_kind) == "test_train_cv":
        row["outer_num_folds"] = int(outer_num_folds)
        row["test_fold_id"] = int(test_fold_id)
    return row


def _planned_train_fit_request_rows(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    generation_rows = read_csv_rows(manifest_path)
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {manifest_path}."
        )
    search = load_search_from_spec(cv_spec_path, search_slug)
    split_settings = _resolve_train_fit_split_settings(
        search,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    request_rows: list[dict[str, object]] = []
    for experiment_row in generation_rows:
        experiment_root = Path(experiment_row["experiment_path"]).resolve()
        best_candidate_path = best_candidate_path_for_search(
            experiment_root,
            cv_spec_path,
            str(search["slug"]),
        )
        payload = _load_best_candidate_payload(best_candidate_path)
        candidate = _variant_from_best_candidate_payload(payload, search=search)
        request_rows.append(
            _train_fit_request_row(
                experiment_row,
                search=search,
                candidate=candidate,
                best_candidate_path=best_candidate_path,
                generation_manifest_path=manifest_path,
                cv_spec_path=cv_spec_path,
                split_kind=str(split_settings["split_kind"]),
                num_folds=int(split_settings["num_folds"]),
                outer_num_folds=int(split_settings["outer_num_folds"]),
                test_fold_id=int(split_settings["test_fold_id"]),
            )
        )
    return search, request_rows


def _select_request_row(
    request_rows: list[dict[str, str]],
    *,
    experiment_slug: str,
    variant_slug: str = "",
) -> dict[str, str]:
    matches = [
        row
        for row in request_rows
        if str(row.get("experiment_slug", "")).strip() == str(experiment_slug).strip()
        and (
            not str(variant_slug).strip()
            or str(row.get("variant_slug", "")).strip() == str(variant_slug).strip()
        )
    ]
    if not matches:
        raise ValueError(
            f"No fit request matched experiment_slug={experiment_slug!r}"
            + (
                f" and variant_slug={variant_slug!r}."
                if str(variant_slug).strip()
                else "."
            )
        )
    if len(matches) != 1:
        raise ValueError(
            f"Fit request selection was not unique for experiment_slug={experiment_slug!r}"
            + (
                f" and variant_slug={variant_slug!r}."
                if str(variant_slug).strip()
                else "."
            )
        )
    return matches[0]


def _manifest_row_from_completed_fit(request_row: dict[str, str]) -> dict[str, object]:
    fit_root = Path(path_text(request_row["fit_path"]))
    metadata_path = fit_root / "fit_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fit metadata: {metadata_path}")
    metadata = load_yaml_mapping(metadata_path)
    fixed_scalar_params = metadata.get("fixed_scalar_params", {})
    row = {
        "experiment_name": str(metadata.get("experiment_name", "")),
        "experiment_slug": str(
            metadata.get("experiment_slug", request_row.get("experiment_slug", ""))
        ),
        "descriptor": str(
            metadata.get("descriptor", metadata.get("experiment_name", ""))
        ),
        "experiment_path": str(metadata.get("experiment_path", "")),
        "intervention_source": str(metadata.get("intervention_source", "")),
        "graph_source": str(metadata.get("graph_source", "")),
        "field_mode": str(metadata.get("field_mode", "")),
        "variant_name": str(metadata.get("variant_name", "")),
        "variant_slug": str(
            metadata.get("variant_slug", request_row.get("variant_slug", ""))
        ),
        "fit_path": str(fit_root),
        "N": int(metadata.get("N", 0)),
        "T": int(metadata.get("T", 0)),
        "s": int(metadata.get("s", 0)),
        "latent_rank": int(metadata.get("latent_rank", 0)),
        "optimizer_mode": str(metadata.get("optimizer_mode", "no_external_field")),
        "lambda_nuclear": float(metadata.get("lambda_nuclear", 0.0)),
        "lambda_frobenius": float(metadata.get("lambda_frobenius", 0.0)),
        "lambda_uv_ridge": float(metadata.get("lambda_uv_ridge", 0.0)),
        "fixed_scalar_params": str(fixed_scalar_params),
        "status": "completed",
    }
    for key in [
        "execution_mode",
        "search_name",
        "search_slug",
        "split_kind",
        "num_folds",
        "outer_num_folds",
        "test_fold_id",
        "best_candidate_path",
        "num_training_slots",
        "cv_spec_path",
    ]:
        value = metadata.get(key, "")
        if value not in (None, ""):
            row[key] = value
    return row


def write_fit_requests(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    generation_rows = read_csv_rows(manifest_path)
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {manifest_path}."
        )
    variants = _expand_fit_variants(fits_spec_path)
    request_rows: list[dict[str, object]] = []
    for experiment_row in generation_rows:
        for variant in variants:
            request_rows.append(
                _fit_request_row(
                    experiment_row,
                    variant,
                    manifest_path,
                    fits_spec_path,
                )
            )
    request_path = fit_requests_path_for_spec(fits_spec_path)
    write_csv_rows(request_path, request_rows)
    return request_path


def write_train_fit_requests(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> Path:
    search, request_rows = _planned_train_fit_request_rows(
        manifest_path,
        cv_spec_path,
        search_slug,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    request_path = _train_fit_requests_path(manifest_path, str(search["slug"]))
    write_csv_rows(request_path, request_rows)
    return request_path


def write_train_fit_requests_for_scope(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str | None = None,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> list[Path]:
    request_paths: list[Path] = []
    for resolved_search_slug in _requested_train_fit_search_slugs(
        cv_spec_path,
        search_slug,
    ):
        request_paths.append(
            write_train_fit_requests(
                manifest_path,
                cv_spec_path,
                resolved_search_slug,
                split_kind=split_kind,
                num_folds=num_folds,
                outer_num_folds=outer_num_folds,
                test_fold_id=test_fold_id,
            )
        )
    return request_paths


def _ensure_train_fit_request_paths(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str | None = None,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> list[Path]:
    request_paths: list[Path] = []
    for resolved_search_slug in _requested_train_fit_search_slugs(
        cv_spec_path,
        search_slug,
    ):
        request_path = _train_fit_requests_path(manifest_path, resolved_search_slug)
        if not request_path.exists():
            write_train_fit_requests(
                manifest_path,
                cv_spec_path,
                resolved_search_slug,
                split_kind=split_kind,
                num_folds=num_folds,
                outer_num_folds=outer_num_folds,
                test_fold_id=test_fold_id,
            )
        request_paths.append(request_path)
    return request_paths


def collect_train_fit_request_rows(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str | None = None,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> list[dict[str, str]]:
    request_rows: list[dict[str, str]] = []
    for request_path in _ensure_train_fit_request_paths(
        manifest_path,
        cv_spec_path,
        search_slug,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    ):
        request_rows.extend(read_csv_rows(request_path))
    return request_rows


def run_fit_variant(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_root = experiment_root / str(variant["fit_root_name"]) / str(variant["slug"])
    if fit_root.exists():
        if overwrite:
            shutil.rmtree(fit_root)
        else:
            raise FileExistsError(
                f"{fit_root} already exists. Re-run with --overwrite to rebuild it."
            )
    fit_root.mkdir(parents=True, exist_ok=False)
    fit_root_path, _, _ = materialize_fit_root(
        experiment_row,
        variant,
        fit_root,
    )
    execute_fit_root(fit_root_path)
    return fit_root_path


def run_train_fit(
    experiment_row: dict[str, str],
    *,
    candidate: dict[str, Any],
    best_candidate_path: str | Path,
    search_slug: str,
    split_kind: str,
    num_folds: int,
    outer_num_folds: int,
    test_fold_id: int,
    fit_root: str | Path | None = None,
    cv_spec_path: str | Path | None = None,
    search_name: str = "",
    overwrite: bool = False,
) -> Path:
    experiment_root = Path(experiment_row["experiment_path"]).resolve()
    output_root = (
        Path(fit_root).resolve()
        if fit_root not in (None, "")
        else _train_fit_output_root(
            experiment_root,
            search_slug=str(search_slug),
            split_kind=str(split_kind),
            num_folds=int(num_folds),
            outer_num_folds=int(outer_num_folds),
            test_fold_id=int(test_fold_id),
            candidate_slug=str(candidate["slug"]),
        )
    )
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    split_artifacts = load_outer_training_split_masks(
        experiment_root,
        split_kind=str(split_kind),
        num_folds=int(num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
    )
    training_mask = np.asarray(split_artifacts["training_mask"], dtype=bool)
    panel_context = load_experiment_panel_context(experiment_root)
    if training_mask.shape != (int(panel_context["T"]), int(panel_context["N"])):
        raise ValueError(
            f"Training mask shape {training_mask.shape} does not match panel dimensions "
            f"(T={panel_context['T']}, N={panel_context['N']})."
        )
    num_training_slots = int(np.count_nonzero(training_mask))
    if num_training_slots <= 0:
        raise ValueError("Training mask is empty.")

    mask_path = save_loss_mask(output_root / "loss_mask.npy", training_mask)
    extra_metadata: dict[str, object] = {
        "execution_mode": "train_fit",
        "search_name": str(search_name),
        "search_slug": str(search_slug),
        "best_candidate_path": path_text(best_candidate_path),
        "split_kind": str(split_artifacts["split_kind"]),
        "num_folds": int(num_folds),
        "num_training_slots": int(num_training_slots),
        "candidate_name": candidate.get("name", ""),
        "candidate_slug": candidate.get("slug", ""),
        "candidate_index": int(candidate.get("_candidate_index", 0)),
        "split_output_root": str(split_artifacts["output_root"]),
    }
    if cv_spec_path not in (None, ""):
        extra_metadata["cv_spec_path"] = path_text(cv_spec_path)
    if str(split_artifacts["split_kind"]) == "test_train_cv":
        extra_metadata["outer_num_folds"] = int(outer_num_folds)
        extra_metadata["test_fold_id"] = int(test_fold_id)

    materialize_fit_root(
        experiment_row,
        candidate,
        output_root,
        extra_input_artifacts={"loss_mask_path": str(mask_path.resolve())},
        extra_metadata=extra_metadata,
    )
    execute_fit_root(output_root)
    return output_root


def run_fit_request(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    experiment_slug: str,
    variant_slug: str,
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_row = _select_generation_row(manifest_path, experiment_slug)
    variant = _select_fit_variant(fits_spec_path, variant_slug)
    request_row = _fit_request_row(
        experiment_row,
        variant,
        manifest_path,
        fits_spec_path,
    )
    run_fit_variant(experiment_row, variant, overwrite=overwrite)
    return _manifest_row_from_completed_fit(
        {key: str(value) for key, value in request_row.items()}
    )


def _run_train_fit_from_request_row(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    request_row: dict[str, str],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_row = _select_generation_row(manifest_path, request_row["experiment_slug"])
    payload = _load_best_candidate_payload(request_row["best_candidate_path"])
    search = load_search_from_spec(cv_spec_path, request_row["search_slug"])
    candidate = _variant_from_best_candidate_payload(payload, search=search)
    run_train_fit(
        experiment_row,
        candidate=candidate,
        best_candidate_path=request_row["best_candidate_path"],
        search_slug=request_row["search_slug"],
        split_kind=request_row["split_kind"],
        num_folds=int(request_row["num_folds"]),
        outer_num_folds=int(request_row.get("outer_num_folds", DEFAULT_OUTER_NUM_FOLDS)),
        test_fold_id=int(request_row.get("test_fold_id", DEFAULT_TEST_FOLD_ID)),
        fit_root=request_row["fit_path"],
        cv_spec_path=cv_spec_path,
        search_name=request_row.get("search_name", ""),
        overwrite=overwrite,
    )
    return _manifest_row_from_completed_fit(request_row)


def run_train_fit_request(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str,
    experiment_slug: str,
    *,
    variant_slug: str = "",
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    request_rows = collect_train_fit_request_rows(
        manifest_path,
        cv_spec_path,
        search_slug,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    request_row = _select_request_row(
        request_rows,
        experiment_slug=experiment_slug,
        variant_slug=variant_slug,
    )
    return _run_train_fit_from_request_row(
        manifest_path,
        cv_spec_path,
        request_row,
        overwrite=overwrite,
    )


def refresh_fit_manifest(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    request_path = fit_requests_path_for_spec(fits_spec_path)
    if not request_path.exists():
        write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_rows(request_path)
    fit_rows = [
        _manifest_row_from_completed_fit(request_row) for request_row in request_rows
    ]
    fit_manifest_path = fit_manifest_path_for_spec(fits_spec_path)
    write_csv_rows(fit_manifest_path, fit_rows)
    write_fit_reports(fit_manifest_path)
    return fit_manifest_path


def refresh_train_fit_manifest(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str | None = None,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
) -> Path:
    normalized_search_slug = str(search_slug or "").strip()
    request_rows = collect_train_fit_request_rows(
        manifest_path,
        cv_spec_path,
        normalized_search_slug or None,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    fit_rows = [
        _manifest_row_from_completed_fit(request_row) for request_row in request_rows
    ]
    train_fit_manifest_path = _train_fit_manifest_path(
        manifest_path,
        normalized_search_slug or None,
    )
    write_csv_rows(train_fit_manifest_path, fit_rows)
    per_experiment_filename, winners_filename = _train_fit_report_filenames(
        normalized_search_slug or None
    )
    write_fit_reports(
        train_fit_manifest_path,
        per_experiment_filename=per_experiment_filename,
        winners_filename=winners_filename,
    )
    return train_fit_manifest_path


def run_fits(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    overwrite: bool = False,
) -> Path:
    request_path = write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_rows(request_path)
    print(f"Loaded {len(request_rows)} fit request(s) from {request_path}.")
    for request_row in request_rows:
        experiment_name = request_row.get(
            "experiment_name", request_row["experiment_slug"]
        )
        variant_name = request_row.get("variant_name", request_row["variant_slug"])
        print(f"Running fit '{variant_name}' for experiment '{experiment_name}'...")
        run_fit_request(
            manifest_path,
            fits_spec_path,
            request_row["experiment_slug"],
            request_row["variant_slug"],
            overwrite=overwrite,
        )
    fit_manifest_path = refresh_fit_manifest(manifest_path, fits_spec_path)
    print(f"Wrote fit manifest to {fit_manifest_path}.")
    return fit_manifest_path


def run_train_fits(
    manifest_path: str | Path,
    cv_spec_path: str | Path,
    search_slug: str | None = None,
    *,
    split_kind: str | None = None,
    num_folds: int | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
    overwrite: bool = False,
) -> Path:
    normalized_search_slug = str(search_slug or "").strip()
    request_paths = write_train_fit_requests_for_scope(
        manifest_path,
        cv_spec_path,
        normalized_search_slug or None,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    request_rows: list[dict[str, str]] = []
    for request_path in request_paths:
        request_rows.extend(read_csv_rows(request_path))
    if normalized_search_slug:
        print(f"Loaded {len(request_rows)} train-fit request(s) from {request_paths[0]}.")
    else:
        print(
            f"Loaded {len(request_rows)} train-fit request(s) from {len(request_paths)} search request file(s)."
        )
    for request_row in request_rows:
        experiment_name = request_row.get(
            "experiment_name", request_row["experiment_slug"]
        )
        variant_name = request_row.get("variant_name", request_row["variant_slug"])
        print(
            f"Running train fit '{variant_name}' for experiment '{experiment_name}'..."
        )
        _run_train_fit_from_request_row(
            manifest_path,
            cv_spec_path,
            request_row,
            overwrite=overwrite,
        )
    train_fit_manifest_path = refresh_train_fit_manifest(
        manifest_path,
        cv_spec_path,
        normalized_search_slug or None,
        split_kind=split_kind,
        num_folds=num_folds,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
    )
    print(f"Wrote train-fit manifest to {train_fit_manifest_path}.")
    return train_fit_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MPLE fits over a generation manifest."
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the generation manifest CSV produced by run_generation_pipeline.py.",
    )
    parser.add_argument(
        "--fits_spec_path",
        type=str,
        default="data/configs/quickstart_fits_spec.yaml",
        help="Path to the fits YAML spec defining optimizer variants (default: data/configs/quickstart_fits_spec.yaml).",
    )
    parser.add_argument(
        "--cv_spec_path",
        type=str,
        default="",
        help="Path to the CV YAML spec used to locate best_candidate.yaml in outer_masked mode.",
    )
    parser.add_argument(
        "--search_slug",
        type=str,
        default="",
        help="CV search slug used in outer_masked mode.",
    )
    parser.add_argument(
        "--fit_mode",
        type=str,
        default=FIT_MODE_STANDARD,
        help="Fit execution mode: standard or outer_masked.",
    )
    parser.add_argument("--split_kind", type=str, default="")
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--outer_num_folds", type=int, default=None)
    parser.add_argument("--test_fold_id", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete and rebuild existing fit directories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs and print the planned work without executing any fits.",
    )
    parser.add_argument(
        "--write_requests",
        action="store_true",
        help="Write request CSV for the configured mode.",
    )
    parser.add_argument(
        "--run_request",
        action="store_true",
        help="Run one planned fit request selected by --experiment_slug and optionally --variant_slug.",
    )
    parser.add_argument(
        "--refresh_manifest",
        action="store_true",
        help="Refresh the manifest from completed fit outputs and rebuild reports.",
    )
    parser.add_argument("--experiment_slug", type=str, default="")
    parser.add_argument("--variant_slug", type=str, default="")
    args = parser.parse_args()

    fit_mode = _normalize_fit_mode(args.fit_mode)
    normalized_search_slug = str(args.search_slug).strip()
    if fit_mode == FIT_MODE_OUTER_MASKED:
        if not str(args.cv_spec_path).strip():
            raise ValueError("--cv_spec_path is required when --fit_mode=outer_masked.")
        if args.run_request and not normalized_search_slug:
            raise ValueError("--search_slug is required when --fit_mode=outer_masked.")

    if args.dry_run:
        generation_rows = read_csv_rows(args.manifest_path)
        if fit_mode == FIT_MODE_STANDARD:
            variants = _expand_fit_variants(args.fits_spec_path)
            print(
                f"Dry run: {len(generation_rows)} experiment(s) × {len(variants)} variant(s) "
                f"= {len(generation_rows) * len(variants)} fit(s) planned."
            )
            for row in generation_rows:
                for variant in variants:
                    print(f"  {row.get('experiment_name', '?')} / {variant['name']}")
        else:
            searches, request_rows = [], []
            for resolved_search_slug in _requested_train_fit_search_slugs(
                args.cv_spec_path,
                normalized_search_slug or None,
            ):
                search, current_rows = _planned_train_fit_request_rows(
                    args.manifest_path,
                    args.cv_spec_path,
                    resolved_search_slug,
                    split_kind=args.split_kind or None,
                    num_folds=args.num_folds,
                    outer_num_folds=args.outer_num_folds,
                    test_fold_id=args.test_fold_id,
                )
                searches.append(search)
                request_rows.extend(current_rows)
            if normalized_search_slug:
                print(
                    f"Dry run: {len(request_rows)} train fit(s) planned for search '{searches[0]['name']}'."
                )
            else:
                print(
                    f"Dry run: {len(request_rows)} train fit(s) planned across {len(searches)} search(es)."
                )
            for row in request_rows:
                print(
                    f"  {row.get('experiment_name', '?')} / {row['search_slug']} / {row['variant_name']} / {row['split_kind']}"
                )
        return

    if args.write_requests:
        if fit_mode == FIT_MODE_STANDARD:
            request_path = write_fit_requests(args.manifest_path, args.fits_spec_path)
            print(f"Fit requests: {request_path}")
        else:
            request_paths = write_train_fit_requests_for_scope(
                args.manifest_path,
                args.cv_spec_path,
                normalized_search_slug or None,
                split_kind=args.split_kind or None,
                num_folds=args.num_folds,
                outer_num_folds=args.outer_num_folds,
                test_fold_id=args.test_fold_id,
            )
            for request_path in request_paths:
                print(f"Train-fit requests: {request_path}")
        return

    if args.run_request:
        if not args.experiment_slug.strip():
            raise ValueError("--experiment_slug is required when --run_request is set.")
        if fit_mode == FIT_MODE_STANDARD:
            if not args.variant_slug.strip():
                raise ValueError(
                    "--variant_slug is required when --run_request is set in standard mode."
                )
            row = run_fit_request(
                args.manifest_path,
                args.fits_spec_path,
                args.experiment_slug.strip(),
                args.variant_slug.strip(),
                overwrite=args.overwrite,
            )
        else:
            row = run_train_fit_request(
                args.manifest_path,
                args.cv_spec_path,
                normalized_search_slug,
                args.experiment_slug.strip(),
                variant_slug=args.variant_slug.strip(),
                split_kind=args.split_kind or None,
                num_folds=args.num_folds,
                outer_num_folds=args.outer_num_folds,
                test_fold_id=args.test_fold_id,
                overwrite=args.overwrite,
            )
        print(f"Completed fit: {row['fit_path']}")
        return

    if args.refresh_manifest:
        if fit_mode == FIT_MODE_STANDARD:
            fit_manifest_path = refresh_fit_manifest(
                args.manifest_path,
                args.fits_spec_path,
            )
            print(f"Fit manifest: {fit_manifest_path}")
        else:
            fit_manifest_path = refresh_train_fit_manifest(
                args.manifest_path,
                args.cv_spec_path,
                normalized_search_slug or None,
                split_kind=args.split_kind or None,
                num_folds=args.num_folds,
                outer_num_folds=args.outer_num_folds,
                test_fold_id=args.test_fold_id,
            )
            print(f"Train-fit manifest: {fit_manifest_path}")
        return

    if fit_mode == FIT_MODE_STANDARD:
        fit_manifest_path = run_fits(
            args.manifest_path,
            args.fits_spec_path,
            overwrite=args.overwrite,
        )
        print(f"Fit manifest: {fit_manifest_path}")
    else:
        fit_manifest_path = run_train_fits(
            args.manifest_path,
            args.cv_spec_path,
            normalized_search_slug or None,
            split_kind=args.split_kind or None,
            num_folds=args.num_folds,
            outer_num_folds=args.outer_num_folds,
            test_fold_id=args.test_fold_id,
            overwrite=args.overwrite,
        )
        print(f"Train-fit manifest: {fit_manifest_path}")


if __name__ == "__main__":
    main()
