"""Evaluate held-out test-set metrics for a saved fit root."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.t0_path_utils import io_path, path_exists
from utils.t6_split_management import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    load_outer_test_split_masks,
)
from utils.validation_metric_utils import (
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
    parser.add_argument("--outer_num_folds", type=int, default=DEFAULT_OUTER_NUM_FOLDS)
    parser.add_argument("--test_fold_id", type=int, default=DEFAULT_TEST_FOLD_ID)
    parser.add_argument("--inner_num_folds", type=int, default=DEFAULT_OUTER_NUM_FOLDS)
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
    metadata = OmegaConf.to_container(OmegaConf.load(io_path(metadata_path)), resolve=True)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _resolve_experiment_path(
    fit_root: str | Path,
    experiment_path: str | Path | None,
) -> Path:
    if experiment_path not in (None, ""):
        return Path(str(experiment_path)).resolve()
    metadata = _load_fit_metadata(fit_root)
    resolved = str(metadata.get("experiment_path", "")).strip()
    if not resolved:
        raise ValueError(
            f"fit_metadata.yaml under {fit_root} does not contain experiment_path."
        )
    return Path(resolved).resolve()


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
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    sampling: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    fit_root = Path(fit_path).resolve()
    resolved_experiment_root = _resolve_experiment_path(fit_root, experiment_path)
    resolved_sampling = resolve_validation_sampling(sampling)
    split_artifacts = load_outer_test_split_masks(
        resolved_experiment_root,
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(inner_num_folds),
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
            test_fold_id=int(test_fold_id),
            inner_num_folds=int(inner_num_folds),
        )
    )
    payload = {
        "fit_path": str(fit_root),
        "fit_name": fit_root.name,
        "experiment_path": str(resolved_experiment_root),
        "split_source": str(split_artifacts["split_source"]),
        "outer_num_folds": int(outer_num_folds),
        "test_fold_id": int(test_fold_id),
        "inner_num_folds": int(inner_num_folds),
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

