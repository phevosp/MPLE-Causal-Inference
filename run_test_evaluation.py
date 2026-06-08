"""Evaluate held-out test-set metrics for a saved fit root."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.t0_config_utils import load_yaml_mapping
from utils.t0_path_utils import io_path, path_exists
from utils.t6_split_management import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    load_outer_test_split_masks,
)
from utils.t7_validation_metrics import (
    DEFAULT_VALIDATION_SAMPLING,
    evaluate_saved_fit_test_metrics,
    resolve_validation_sampling,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out test-set metrics for a saved fit root."
    )
    parser.add_argument("--fit_path", required=True, type=str)
    parser.add_argument("--experiment_path", type=str, default=None)
    parser.add_argument("--outer_num_folds", type=int, default=None)
    parser.add_argument("--test_fold_id", type=int, default=None)
    parser.add_argument("--inner_num_folds", type=int, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=int(DEFAULT_VALIDATION_SAMPLING["num_samples"]),
    )
    parser.add_argument(
        "--gibbs_sweeps",
        type=int,
        default=int(DEFAULT_VALIDATION_SAMPLING["gibbs_sweeps"]),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(DEFAULT_VALIDATION_SAMPLING["seed"]),
    )
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args(argv)


def _load_fit_metadata(fit_root: str | Path) -> dict[str, Any]:
    metadata_path = Path(fit_root) / "fit_metadata.yaml"
    if not path_exists(metadata_path):
        raise FileNotFoundError(
            f"Could not infer experiment_path because {metadata_path} does not exist."
        )
    return load_yaml_mapping(metadata_path)


def _resolve_experiment_path(
    fit_root: str | Path,
    experiment_path: str | Path | None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if experiment_path not in (None, ""):
        return Path(str(experiment_path)).resolve()
    resolved_metadata = metadata if metadata is not None else _load_fit_metadata(fit_root)
    resolved = str(resolved_metadata.get("experiment_path", "")).strip()
    if not resolved:
        raise ValueError(
            f"fit_metadata.yaml under {fit_root} does not contain experiment_path."
        )
    return Path(resolved).resolve()


def _optional_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _resolve_split_settings(
    fit_root: str | Path,
    *,
    metadata: dict[str, Any],
    outer_num_folds: int | None,
    test_fold_id: int | None,
    inner_num_folds: int | None,
) -> dict[str, int]:
    execution_mode = str(metadata.get("execution_mode", "")).strip().lower()
    split_kind = str(metadata.get("split_kind", "")).strip().lower()
    resolved_outer_num_folds = (
        int(outer_num_folds) if outer_num_folds is not None else None
    )
    resolved_test_fold_id = int(test_fold_id) if test_fold_id is not None else None
    resolved_inner_num_folds = (
        int(inner_num_folds) if inner_num_folds is not None else None
    )

    if execution_mode == "train_fit" and split_kind == "test_train_cv":
        if resolved_outer_num_folds is None:
            resolved_outer_num_folds = _optional_int(metadata, "outer_num_folds")
        if resolved_test_fold_id is None:
            resolved_test_fold_id = _optional_int(metadata, "test_fold_id")
        if resolved_inner_num_folds is None:
            resolved_inner_num_folds = _optional_int(metadata, "num_folds")

    if resolved_outer_num_folds is None:
        resolved_outer_num_folds = int(DEFAULT_OUTER_NUM_FOLDS)
    if resolved_test_fold_id is None:
        resolved_test_fold_id = int(DEFAULT_TEST_FOLD_ID)
    if resolved_inner_num_folds is None:
        resolved_inner_num_folds = int(DEFAULT_OUTER_NUM_FOLDS)

    return {
        "outer_num_folds": int(resolved_outer_num_folds),
        "test_fold_id": int(resolved_test_fold_id),
        "inner_num_folds": int(resolved_inner_num_folds),
    }


def _default_output_path(
    fit_root: str | Path,
    *,
    test_fold_id: int,
    inner_num_folds: int,
) -> Path:
    return (
        Path(fit_root).resolve()
        / "test_set_evaluation"
        / f"test_fold_{int(test_fold_id)}__inner_folds_{int(inner_num_folds)}"
        / "test_metrics.yaml"
    )


def run_test_evaluation(
    fit_path: str | Path,
    *,
    experiment_path: str | Path | None = None,
    outer_num_folds: int | None = None,
    test_fold_id: int | None = None,
    inner_num_folds: int | None = None,
    sampling: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    fit_root = Path(fit_path).resolve()
    fit_metadata = _load_fit_metadata(fit_root)
    resolved_experiment_root = _resolve_experiment_path(
        fit_root,
        experiment_path,
        metadata=fit_metadata,
    )
    split_settings = _resolve_split_settings(
        fit_root,
        metadata=fit_metadata,
        outer_num_folds=outer_num_folds,
        test_fold_id=test_fold_id,
        inner_num_folds=inner_num_folds,
    )
    resolved_sampling = resolve_validation_sampling(sampling)
    split_artifacts = load_outer_test_split_masks(
        resolved_experiment_root,
        outer_num_folds=int(split_settings["outer_num_folds"]),
        test_fold_id=int(split_settings["test_fold_id"]),
        inner_num_folds=int(split_settings["inner_num_folds"]),
    )
    metrics = evaluate_saved_fit_test_metrics(
        fit_root,
        resolved_experiment_root,
        training_loss_mask=split_artifacts["training_mask"],
        test_loss_mask=split_artifacts["test_mask"],
        sampling=resolved_sampling,
    )
    report_path = (
        Path(output_path).resolve()
        if output_path not in (None, "")
        else _default_output_path(
            fit_root,
            test_fold_id=int(split_settings["test_fold_id"]),
            inner_num_folds=int(split_settings["inner_num_folds"]),
        )
    )
    payload = {
        "fit_path": str(fit_root),
        "fit_name": fit_root.name,
        "experiment_path": str(resolved_experiment_root),
        "split_kind": str(split_artifacts["split_kind"]),
        "outer_num_folds": int(split_settings["outer_num_folds"]),
        "test_fold_id": int(split_settings["test_fold_id"]),
        "inner_num_folds": int(split_settings["inner_num_folds"]),
        "sampling": dict(resolved_sampling),
        "split_output_root": str(split_artifacts["output_root"]),
        "split_metadata": dict(split_artifacts["metadata"]),
        **metrics,
    }
    os.makedirs(io_path(report_path.parent), exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), io_path(report_path))
    return report_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report_path = run_test_evaluation(
        args.fit_path,
        experiment_path=args.experiment_path,
        outer_num_folds=args.outer_num_folds,
        test_fold_id=args.test_fold_id,
        inner_num_folds=args.inner_num_folds,
        sampling={
            "num_samples": int(args.num_samples),
            "gibbs_sweeps": int(args.gibbs_sweeps),
            "seed": int(args.seed),
        },
        output_path=args.output_path,
    )
    print(f"Test metrics report: {report_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
