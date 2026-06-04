"""Shared helpers for CV-fold and validation/test split artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from utils.io_utils import io_path, path_exists
from utils.loading_utils import load_experiment_panel_context


SPLIT_SOURCE_CV_FOLDS = "cv_folds"
SPLIT_SOURCE_VALIDATION_TEST_SPLITS = "validation_test_splits"
VALID_SPLIT_SOURCES = frozenset(
    {SPLIT_SOURCE_CV_FOLDS, SPLIT_SOURCE_VALIDATION_TEST_SPLITS}
)
DEFAULT_OUTER_NUM_FOLDS = 5
DEFAULT_TEST_FOLD_ID = 1


def normalize_split_source(split_source: str | None) -> str:
    normalized = str(split_source or SPLIT_SOURCE_CV_FOLDS).strip().lower()
    if normalized not in VALID_SPLIT_SOURCES:
        raise ValueError(
            f"split_source must be one of {sorted(VALID_SPLIT_SOURCES)}, got '{split_source}'."
        )
    return normalized


def validation_test_split_output_root(
    experiment_root: str | Path,
    *,
    test_fold_id: int,
    inner_num_folds: int,
) -> Path:
    experiment_path = Path(experiment_root).resolve()
    return (
        experiment_path
        / SPLIT_SOURCE_VALIDATION_TEST_SPLITS
        / f"test_fold_{int(test_fold_id)}__inner_folds_{int(inner_num_folds)}"
    )


def _expected_panel_shape(experiment_root: str | Path) -> tuple[int, int]:
    panel_context = load_experiment_panel_context(experiment_root)
    return int(panel_context["T"]), int(panel_context["N"])


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    if not path_exists(path):
        return {}
    metadata = OmegaConf.to_container(OmegaConf.load(io_path(path)), resolve=True)
    return dict(metadata) if isinstance(metadata, dict) else {}


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


def load_model_selection_split_masks(
    experiment_root: str | Path,
    *,
    split_source: str | None,
    num_folds: int,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
) -> dict[str, Any]:
    normalized_source = normalize_split_source(split_source)
    expected_shape = _expected_panel_shape(experiment_root)

    if normalized_source == SPLIT_SOURCE_CV_FOLDS:
        import build_cv_folds as cv_folds

        output_root = Path(experiment_root) / "cv_folds" / f"folds_{int(num_folds)}"
        blanket_summary = _load_optional_metadata(output_root / "markov_blanket_summary.yaml")
        if not bool(blanket_summary.get("blanket_validation_passed", False)):
            raise ValueError(
                f"CV folds at {output_root} failed Markov blanket validation."
            )
        metadata = _load_optional_metadata(output_root / "spatiotemporal_cv_metadata.yaml")
        if int(metadata.get("num_cv_folds", 0)) != int(num_folds):
            raise ValueError(
                f"Expected {num_folds} folds in {output_root}; found "
                f"{metadata.get('num_cv_folds')}."
            )
        fold_roles_path = output_root / "fold_roles.npz"
        if not path_exists(fold_roles_path):
            raise FileNotFoundError(f"Missing CV fold artifact: {fold_roles_path}")
        with np.load(io_path(fold_roles_path), allow_pickle=False) as data:
            role_codes = np.asarray(data["role_codes"], dtype=np.int8)
        expected_tensor_shape = (int(num_folds), *expected_shape)
        if role_codes.shape != expected_tensor_shape:
            raise ValueError(
                f"fold_roles.npz at {output_root} has invalid role tensor shape "
                f"{role_codes.shape}; expected {expected_tensor_shape}."
            )
        return {
            "split_source": normalized_source,
            "output_root": output_root.resolve(),
            "training_masks": np.asarray(
                role_codes == cv_folds.ROLE_CODE_TRAINING,
                dtype=bool,
            ),
            "validation_masks": np.asarray(
                role_codes == cv_folds.ROLE_CODE_VALIDATION,
                dtype=bool,
            ),
            "separator_masks": np.asarray(
                role_codes == cv_folds.ROLE_CODE_SEPARATOR,
                dtype=bool,
            ),
            "metadata": dict(metadata),
            "blanket_summary": dict(blanket_summary),
        }

    output_root = validation_test_split_output_root(
        experiment_root,
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(num_folds),
    )
    mask_path = output_root / "inner_validation_masks.npz"
    if not path_exists(mask_path):
        raise FileNotFoundError(f"Missing inner validation split artifact: {mask_path}")
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
        active_mask = _validate_single_mask(
            "active_mask",
            data["active_mask"],
            expected_shape=expected_shape,
        )
    metadata = _load_optional_metadata(output_root / "inner_validation_metadata.yaml")
    metadata_num_folds = int(metadata.get("inner_num_folds", int(num_folds)))
    if metadata_num_folds != int(num_folds):
        raise ValueError(
            f"Inner validation metadata at {output_root} reports {metadata_num_folds} folds, "
            f"expected {num_folds}."
        )
    metadata_outer_num_folds = int(
        metadata.get("outer_num_folds", int(outer_num_folds))
    )
    if metadata_outer_num_folds != int(outer_num_folds):
        raise ValueError(
            f"Inner validation metadata at {output_root} reports outer_num_folds="
            f"{metadata_outer_num_folds}, expected {outer_num_folds}."
        )
    metadata_test_fold_id = int(metadata.get("test_fold_id", int(test_fold_id)))
    if metadata_test_fold_id != int(test_fold_id):
        raise ValueError(
            f"Inner validation metadata at {output_root} reports test_fold_id="
            f"{metadata_test_fold_id}, expected {test_fold_id}."
        )
    return {
        "split_source": normalized_source,
        "output_root": output_root.resolve(),
        "training_masks": training_masks,
        "validation_masks": validation_masks,
        "separator_masks": separator_masks,
        "active_mask": active_mask,
        "metadata": metadata,
    }


def load_outer_test_split_masks(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int,
) -> dict[str, Any]:
    expected_shape = _expected_panel_shape(experiment_root)
    output_root = validation_test_split_output_root(
        experiment_root,
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(inner_num_folds),
    )
    mask_path = output_root / "outer_test_mask.npz"
    if not path_exists(mask_path):
        raise FileNotFoundError(f"Missing outer test split artifact: {mask_path}")
    with np.load(io_path(mask_path), allow_pickle=False) as data:
        training_mask = _validate_single_mask(
            "training_mask",
            data["training_mask"],
            expected_shape=expected_shape,
        )
        separator_mask = _validate_single_mask(
            "separator_mask",
            data["separator_mask"],
            expected_shape=expected_shape,
        )
        test_mask = _validate_single_mask(
            "test_mask",
            data["test_mask"],
            expected_shape=expected_shape,
        )
        time_block_ids = np.asarray(data["time_block_ids"], dtype=np.int16)
        is_transition_step = np.asarray(data["is_transition_step"], dtype=bool)
    metadata = _load_optional_metadata(output_root / "outer_test_metadata.yaml")
    metadata_outer_num_folds = int(
        metadata.get("outer_num_folds", int(outer_num_folds))
    )
    if metadata_outer_num_folds != int(outer_num_folds):
        raise ValueError(
            f"Outer test metadata at {output_root} reports outer_num_folds="
            f"{metadata_outer_num_folds}, expected {outer_num_folds}."
        )
    metadata_test_fold_id = int(metadata.get("test_fold_id", int(test_fold_id)))
    if metadata_test_fold_id != int(test_fold_id):
        raise ValueError(
            f"Outer test metadata at {output_root} reports test_fold_id="
            f"{metadata_test_fold_id}, expected {test_fold_id}."
        )
    return {
        "split_source": SPLIT_SOURCE_VALIDATION_TEST_SPLITS,
        "output_root": output_root.resolve(),
        "training_mask": training_mask,
        "separator_mask": separator_mask,
        "test_mask": test_mask,
        "time_block_ids": time_block_ids,
        "is_transition_step": is_transition_step,
        "metadata": metadata,
    }

