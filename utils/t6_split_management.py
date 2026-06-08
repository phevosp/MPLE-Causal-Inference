"""Shared helpers for locating and loading unified split bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.t0_config_utils import load_yaml_mapping
from utils.t0_path_utils import io_path, path_exists
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t6_split_engine import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    SPLIT_KIND_TEST_TRAIN_CV,
    SPLIT_KIND_TRAIN_CV,
    VALID_SPLIT_KINDS,
    test_train_cv_split_output_root,
    train_cv_split_output_root,
)




def normalize_split_kind(split_kind: str | None) -> str:
    normalized = str(split_kind or SPLIT_KIND_TRAIN_CV).strip().lower()
    if normalized not in VALID_SPLIT_KINDS:
        raise ValueError(
            f"split_kind must be one of {sorted(VALID_SPLIT_KINDS)}, got '{split_kind}'."
        )
    return normalized


def _expected_panel_shape(experiment_root: str | Path) -> tuple[int, int]:
    panel_context = load_experiment_panel_context(experiment_root)
    return int(panel_context["T"]), int(panel_context["N"])


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    if not path_exists(path):
        return {}
    return load_yaml_mapping(path)


def _validate_fold_mask_tensor(
    name: str,
    tensor: np.ndarray,
    *,
    expected_num_folds: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(tensor, dtype=bool)
    expected_tensor_shape = (int(expected_num_folds), *expected_shape)
    if array.shape != expected_tensor_shape:
        raise ValueError(
            f"{name} has shape {array.shape}, expected {expected_tensor_shape}."
        )
    return array


def _validate_single_mask(
    name: str,
    mask: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {array.shape}, expected {expected_shape}."
        )
    return array


def _bundle_root(
    experiment_root: str | Path,
    *,
    split_kind: str,
    num_folds: int,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
) -> Path:
    normalized_kind = normalize_split_kind(split_kind)
    if normalized_kind == SPLIT_KIND_TRAIN_CV:
        return train_cv_split_output_root(experiment_root, num_folds=int(num_folds))
    return test_train_cv_split_output_root(
        experiment_root,
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(num_folds),
    )


def load_model_selection_split_masks(
    experiment_root: str | Path,
    *,
    split_kind: str | None,
    num_folds: int,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
) -> dict[str, Any]:
    normalized_kind = normalize_split_kind(split_kind)
    expected_shape = _expected_panel_shape(experiment_root)
    output_root = _bundle_root(
        experiment_root,
        split_kind=normalized_kind,
        num_folds=int(num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
    )
    mask_path = output_root / "model_selection_folds.npz"
    if not path_exists(mask_path):
        raise FileNotFoundError(f"Missing split artifact: {mask_path}")
    with np.load(io_path(mask_path), allow_pickle=False) as data:
        training_masks = _validate_fold_mask_tensor(
            "training_masks",
            data["training_masks"],
            expected_num_folds=int(num_folds),
            expected_shape=expected_shape,
        )
        validation_masks = _validate_fold_mask_tensor(
            "validation_masks",
            data["validation_masks"],
            expected_num_folds=int(num_folds),
            expected_shape=expected_shape,
        )
        separator_masks = _validate_fold_mask_tensor(
            "separator_masks",
            data["separator_masks"],
            expected_num_folds=int(num_folds),
            expected_shape=expected_shape,
        )
    metadata = _load_optional_metadata(output_root / "bundle_metadata.yaml")
    if normalized_kind == SPLIT_KIND_TRAIN_CV:
        metadata_num_folds = int(metadata.get("num_folds", int(num_folds)))
        if metadata_num_folds != int(num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports num_folds={metadata_num_folds}, "
                f"expected {num_folds}."
            )
    else:
        metadata_inner_num_folds = int(metadata.get("inner_num_folds", int(num_folds)))
        if metadata_inner_num_folds != int(num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports inner_num_folds={metadata_inner_num_folds}, "
                f"expected {num_folds}."
            )
        metadata_outer_num_folds = int(metadata.get("outer_num_folds", int(outer_num_folds)))
        if metadata_outer_num_folds != int(outer_num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports outer_num_folds={metadata_outer_num_folds}, "
                f"expected {outer_num_folds}."
            )
        metadata_test_fold_id = int(metadata.get("test_fold_id", int(test_fold_id)))
        if metadata_test_fold_id != int(test_fold_id):
            raise ValueError(
                f"Split metadata at {output_root} reports test_fold_id={metadata_test_fold_id}, "
                f"expected {test_fold_id}."
            )
    return {
        "split_kind": normalized_kind,
        "output_root": output_root.resolve(),
        "training_masks": training_masks,
        "validation_masks": validation_masks,
        "separator_masks": separator_masks,
        "metadata": metadata,
    }


def load_outer_test_split_masks(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int,
) -> dict[str, Any]:
    loaded = load_outer_training_split_masks(
        experiment_root,
        split_kind=SPLIT_KIND_TEST_TRAIN_CV,
        num_folds=int(inner_num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
    )
    return {
        "split_kind": loaded["split_kind"],
        "output_root": loaded["output_root"],
        "training_mask": loaded["training_mask"],
        "separator_mask": loaded["separator_mask"],
        "test_mask": loaded["test_mask"],
        "metadata": loaded["metadata"],
    }


def load_outer_training_split_masks(
    experiment_root: str | Path,
    *,
    split_kind: str | None,
    num_folds: int,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
) -> dict[str, Any]:
    normalized_kind = normalize_split_kind(split_kind)
    expected_shape = _expected_panel_shape(experiment_root)
    output_root = _bundle_root(
        experiment_root,
        split_kind=normalized_kind,
        num_folds=int(num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
    )
    mask_path = output_root / "outer_layer.npz"
    if not path_exists(mask_path):
        raise FileNotFoundError(f"Missing outer split artifact: {mask_path}")
    with np.load(io_path(mask_path), allow_pickle=False) as data:
        training_mask = _validate_single_mask(
            "outer_active_mask",
            data["outer_active_mask"],
            expected_shape=expected_shape,
        )
        separator_mask = _validate_single_mask(
            "outer_separator_mask",
            data["outer_separator_mask"],
            expected_shape=expected_shape,
        )
        test_mask = _validate_single_mask(
            "outer_test_mask",
            data["outer_test_mask"],
            expected_shape=expected_shape,
        )
    metadata = _load_optional_metadata(output_root / "bundle_metadata.yaml")
    metadata_split_kind = str(metadata.get("split_kind", normalized_kind)).strip().lower()
    if metadata_split_kind != normalized_kind:
        raise ValueError(
            f"Split metadata at {output_root} reports split_kind={metadata_split_kind}, "
            f"expected {normalized_kind}."
        )
    if normalized_kind == SPLIT_KIND_TRAIN_CV:
        metadata_num_folds = int(metadata.get("num_folds", int(num_folds)))
        if metadata_num_folds != int(num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports num_folds={metadata_num_folds}, "
                f"expected {num_folds}."
            )
    else:
        metadata_outer_num_folds = int(
            metadata.get("outer_num_folds", int(outer_num_folds))
        )
        if metadata_outer_num_folds != int(outer_num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports outer_num_folds={metadata_outer_num_folds}, "
                f"expected {outer_num_folds}."
            )
        metadata_test_fold_id = int(metadata.get("test_fold_id", int(test_fold_id)))
        if metadata_test_fold_id != int(test_fold_id):
            raise ValueError(
                f"Split metadata at {output_root} reports test_fold_id={metadata_test_fold_id}, "
                f"expected {test_fold_id}."
            )
        metadata_inner_num_folds = int(metadata.get("inner_num_folds", int(num_folds)))
        if metadata_inner_num_folds != int(num_folds):
            raise ValueError(
                f"Split metadata at {output_root} reports inner_num_folds={metadata_inner_num_folds}, "
                f"expected {num_folds}."
            )
    return {
        "split_kind": normalized_kind,
        "output_root": output_root.resolve(),
        "training_mask": training_mask,
        "separator_mask": separator_mask,
        "test_mask": test_mask,
        "metadata": metadata,
    }
