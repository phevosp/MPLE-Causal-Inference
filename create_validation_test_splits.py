"""Build validation/test split artifacts: outer test fold + inner validation folds."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

import build_cv_folds as cv_folds
from io_utils import io_path, write_csv
from pipeline_specs import read_csv_manifest
from split_artifact_utils import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    validation_test_split_output_root,
)


DEFAULT_INNER_NUM_FOLDS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create held-out test masks and inner validation masks for every "
            "experiment listed in a generation manifest."
        )
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--outer_num_folds", type=int, default=DEFAULT_OUTER_NUM_FOLDS)
    parser.add_argument("--test_fold_id", type=int, default=DEFAULT_TEST_FOLD_ID)
    parser.add_argument("--inner_num_folds", type=int, default=DEFAULT_INNER_NUM_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--tolerance", type=float, default=cv_folds.DEFAULT_GAMMA_TOLERANCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _generation_manifest_rows(
    generation_manifest_path: str | Path,
) -> list[dict[str, str]]:
    rows = read_csv_manifest(generation_manifest_path)
    if not rows:
        raise ValueError(f"No rows found in generation manifest {generation_manifest_path}.")
    return rows


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    resolved_output_root = output_root.resolve()
    if resolved_output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {resolved_output_root} already exists. Pass --overwrite to replace it."
            )
        expected_parent = resolved_output_root.parent
        if expected_parent not in resolved_output_root.parents:
            raise ValueError(f"Refusing to overwrite unexpected path {resolved_output_root}.")
        shutil.rmtree(io_path(resolved_output_root), ignore_errors=True)
    os.makedirs(io_path(resolved_output_root), exist_ok=True)


def _stable_spatial_partition_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    stable = dict(metadata)
    stable.pop("runtime_seconds", None)
    return stable


def _role_masks_for_fold(role_codes: np.ndarray, fold_id: int) -> dict[str, np.ndarray]:
    fold_index = int(fold_id) - 1
    if fold_index < 0 or fold_index >= int(role_codes.shape[0]):
        raise ValueError(
            f"fold_id must be between 1 and {int(role_codes.shape[0])}, got {fold_id}."
        )
    fold_roles = np.asarray(role_codes[fold_index], dtype=np.int8)
    return {
        "training_mask": np.asarray(
            fold_roles == cv_folds.ROLE_CODE_TRAINING,
            dtype=bool,
        ),
        "separator_mask": np.asarray(
            fold_roles == cv_folds.ROLE_CODE_SEPARATOR,
            dtype=bool,
        ),
        "validation_mask": np.asarray(
            fold_roles == cv_folds.ROLE_CODE_VALIDATION,
            dtype=bool,
        ),
    }


def _build_inner_masks(
    inner_role_codes: np.ndarray,
    outer_training_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    outer_training = np.asarray(outer_training_mask, dtype=bool)
    training_masks = np.asarray(
        (np.asarray(inner_role_codes, dtype=np.int8) == cv_folds.ROLE_CODE_TRAINING)
        & outer_training[None, :, :],
        dtype=bool,
    )
    separator_masks = np.asarray(
        (np.asarray(inner_role_codes, dtype=np.int8) == cv_folds.ROLE_CODE_SEPARATOR)
        & outer_training[None, :, :],
        dtype=bool,
    )
    validation_masks = np.asarray(
        (np.asarray(inner_role_codes, dtype=np.int8) == cv_folds.ROLE_CODE_VALIDATION)
        & outer_training[None, :, :],
        dtype=bool,
    )
    inactive_mask = np.asarray(~outer_training, dtype=bool)
    return {
        "training_masks": training_masks,
        "separator_masks": separator_masks,
        "validation_masks": validation_masks,
        "active_mask": outer_training,
        "inactive_mask": inactive_mask,
    }


def _count_true(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _build_inner_fold_summary_rows(
    inner_masks: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    training_masks = np.asarray(inner_masks["training_masks"], dtype=bool)
    separator_masks = np.asarray(inner_masks["separator_masks"], dtype=bool)
    validation_masks = np.asarray(inner_masks["validation_masks"], dtype=bool)
    active_mask = np.asarray(inner_masks["active_mask"], dtype=bool)
    inactive_mask = np.asarray(inner_masks["inactive_mask"], dtype=bool)
    num_active_slots = _count_true(active_mask)
    num_inactive_slots = _count_true(inactive_mask)

    rows: list[dict[str, object]] = []
    for fold_index in range(int(training_masks.shape[0])):
        num_training_slots = _count_true(training_masks[fold_index])
        num_separator_slots = _count_true(separator_masks[fold_index])
        num_validation_slots = _count_true(validation_masks[fold_index])
        rows.append(
            {
                "inner_fold_id": int(fold_index + 1),
                "num_active_slots": int(num_active_slots),
                "num_inactive_slots": int(num_inactive_slots),
                "num_training_slots": int(num_training_slots),
                "num_separator_slots": int(num_separator_slots),
                "num_validation_slots": int(num_validation_slots),
                "validation_fraction_within_active": (
                    float(num_validation_slots) / float(num_active_slots)
                    if num_active_slots > 0
                    else 0.0
                ),
            }
        )
    return rows


def _write_split_artifacts(
    output_root: Path,
    *,
    outer_masks: dict[str, np.ndarray],
    outer_metadata: dict[str, Any],
    inner_masks: dict[str, np.ndarray],
    inner_metadata: dict[str, Any],
    inner_fold_summary_rows: list[dict[str, object]],
) -> None:
    np.savez(
        io_path(output_root / "outer_test_mask.npz"),
        training_mask=np.asarray(outer_masks["training_mask"], dtype=bool),
        separator_mask=np.asarray(outer_masks["separator_mask"], dtype=bool),
        test_mask=np.asarray(outer_masks["validation_mask"], dtype=bool),
        time_block_ids=np.asarray(outer_metadata["time_block_ids"], dtype=np.int16),
        is_transition_step=np.asarray(outer_metadata["is_transition_step"], dtype=bool),
    )
    cv_folds._write_yaml(output_root / "outer_test_metadata.yaml", outer_metadata)

    np.savez(
        io_path(output_root / "inner_validation_masks.npz"),
        training_masks=np.asarray(inner_masks["training_masks"], dtype=bool),
        separator_masks=np.asarray(inner_masks["separator_masks"], dtype=bool),
        validation_masks=np.asarray(inner_masks["validation_masks"], dtype=bool),
        active_mask=np.asarray(inner_masks["active_mask"], dtype=bool),
    )
    cv_folds._write_yaml(output_root / "inner_validation_metadata.yaml", inner_metadata)
    write_csv(
        output_root / "inner_fold_summary.csv",
        inner_fold_summary_rows,
        [
            "inner_fold_id",
            "num_active_slots",
            "num_inactive_slots",
            "num_training_slots",
            "num_separator_slots",
            "num_validation_slots",
            "validation_fraction_within_active",
        ],
    )


def build_validation_test_splits_for_experiment(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = DEFAULT_INNER_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = cv_folds.DEFAULT_GAMMA_TOLERANCE,
    overwrite: bool = False,
) -> Path:
    if int(outer_num_folds) < 1:
        raise ValueError(f"outer_num_folds must be >= 1, got {outer_num_folds}.")
    if int(inner_num_folds) < 1:
        raise ValueError(f"inner_num_folds must be >= 1, got {inner_num_folds}.")
    if int(test_fold_id) < 1 or int(test_fold_id) > int(outer_num_folds):
        raise ValueError(
            f"test_fold_id must be between 1 and {int(outer_num_folds)}, got {test_fold_id}."
        )

    experiment_path = Path(experiment_root).resolve()
    output_root = validation_test_split_output_root(
        experiment_path,
        test_fold_id=test_fold_id,
        inner_num_folds=inner_num_folds,
    )
    _prepare_output_root(output_root, overwrite=overwrite)

    outer_artifacts = cv_folds._build_cv_fold_artifacts(
        experiment_path,
        num_folds=int(outer_num_folds),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )
    inner_artifacts = cv_folds._build_cv_fold_artifacts(
        experiment_path,
        num_folds=int(inner_num_folds),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )

    outer_masks = _role_masks_for_fold(
        np.asarray(outer_artifacts["role_codes"], dtype=np.int8),
        int(test_fold_id),
    )
    inner_masks = _build_inner_masks(
        np.asarray(inner_artifacts["role_codes"], dtype=np.int8),
        np.asarray(outer_masks["training_mask"], dtype=bool),
    )
    inner_fold_summary_rows = _build_inner_fold_summary_rows(inner_masks)

    outer_time_plan = outer_artifacts["time_plan"]
    inner_time_plan = inner_artifacts["time_plan"]
    outer_metadata = {
        "experiment_root": str(experiment_path),
        "output_root": str(output_root),
        "outer_num_folds": int(outer_num_folds),
        "test_fold_id": int(test_fold_id),
        "seed": int(seed),
        "recursive": bool(recursive),
        "contiguous": bool(contiguous),
        "tolerance": float(tolerance),
        "role_code_map": dict(outer_artifacts["spatiotemporal_metadata"]["role_code_map"]),
        "outer_spatial_partition_metadata": _stable_spatial_partition_metadata(
            dict(outer_artifacts["spatial_partition_metadata"])
        ),
        "outer_spatiotemporal_metadata": dict(
            outer_artifacts["spatiotemporal_metadata"]
        ),
        "outer_markov_blanket_summary": dict(
            outer_artifacts["markov_blanket_summary"]
        ),
        "time_block_ids": np.asarray(
            outer_time_plan["time_block_ids"],
            dtype=np.int16,
        ).tolist(),
        "is_transition_step": np.asarray(
            outer_time_plan["is_transition_step"],
            dtype=bool,
        ).tolist(),
        "num_training_slots": _count_true(outer_masks["training_mask"]),
        "num_separator_slots": _count_true(outer_masks["separator_mask"]),
        "num_test_slots": _count_true(outer_masks["validation_mask"]),
    }
    inner_metadata = {
        "experiment_root": str(experiment_path),
        "output_root": str(output_root),
        "outer_num_folds": int(outer_num_folds),
        "test_fold_id": int(test_fold_id),
        "inner_num_folds": int(inner_num_folds),
        "seed": int(seed),
        "recursive": bool(recursive),
        "contiguous": bool(contiguous),
        "tolerance": float(tolerance),
        "role_code_map": dict(inner_artifacts["spatiotemporal_metadata"]["role_code_map"]),
        "active_mask_comes_from_outer_training_fold": int(test_fold_id),
        "inner_spatial_partition_metadata": _stable_spatial_partition_metadata(
            dict(inner_artifacts["spatial_partition_metadata"])
        ),
        "inner_spatiotemporal_metadata": dict(
            inner_artifacts["spatiotemporal_metadata"]
        ),
        "inner_markov_blanket_summary": dict(
            inner_artifacts["markov_blanket_summary"]
        ),
        "time_block_ids": np.asarray(
            inner_time_plan["time_block_ids"],
            dtype=np.int16,
        ).tolist(),
        "is_transition_step": np.asarray(
            inner_time_plan["is_transition_step"],
            dtype=bool,
        ).tolist(),
        "num_active_slots": _count_true(inner_masks["active_mask"]),
        "num_inactive_slots": _count_true(inner_masks["inactive_mask"]),
        "num_training_slots_by_fold": [
            _count_true(inner_masks["training_masks"][fold_index])
            for fold_index in range(int(inner_masks["training_masks"].shape[0]))
        ],
        "num_separator_slots_by_fold": [
            _count_true(inner_masks["separator_masks"][fold_index])
            for fold_index in range(int(inner_masks["separator_masks"].shape[0]))
        ],
        "num_validation_slots_by_fold": [
            _count_true(inner_masks["validation_masks"][fold_index])
            for fold_index in range(int(inner_masks["validation_masks"].shape[0]))
        ],
    }

    _write_split_artifacts(
        output_root,
        outer_masks=outer_masks,
        outer_metadata=outer_metadata,
        inner_masks=inner_masks,
        inner_metadata=inner_metadata,
        inner_fold_summary_rows=inner_fold_summary_rows,
    )
    return output_root


def create_validation_test_splits(
    generation_manifest_path: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = DEFAULT_INNER_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = cv_folds.DEFAULT_GAMMA_TOLERANCE,
    overwrite: bool = False,
) -> list[Path]:
    output_paths: list[Path] = []
    for row in _generation_manifest_rows(generation_manifest_path):
        experiment_path = str(row.get("experiment_path", "")).strip()
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        output_root = build_validation_test_splits_for_experiment(
            experiment_path,
            outer_num_folds=outer_num_folds,
            test_fold_id=test_fold_id,
            inner_num_folds=inner_num_folds,
            seed=seed,
            recursive=recursive,
            contiguous=contiguous,
            tolerance=tolerance,
            overwrite=overwrite,
        )
        output_paths.append(output_root)
        if experiment_name:
            print(
                f"Built validation/test splits for {experiment_name}: {output_root}",
                flush=True,
            )
    return output_paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_paths = create_validation_test_splits(
        args.generation_manifest_path,
        outer_num_folds=args.outer_num_folds,
        test_fold_id=args.test_fold_id,
        inner_num_folds=args.inner_num_folds,
        seed=args.seed,
        recursive=args.recursive,
        contiguous=args.contiguous,
        tolerance=args.tolerance,
        overwrite=args.overwrite,
    )
    print(f"Built validation/test split artifacts for {len(output_paths)} experiments.")


if __name__ == "__main__":
    main(sys.argv[1:])
