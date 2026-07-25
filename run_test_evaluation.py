"""Evaluate held-out test-set metrics for saved fits listed in a fit manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.t0_config_utils import load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows
from utils.t0_path_utils import io_path, path_exists
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t6_split_management import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    load_outer_test_split_masks,
)
from utils.t7_validation_metrics import (
    DEFAULT_VALIDATION_SAMPLING,
    evaluate_saved_fit_test_metrics,
    evaluate_test_baseline_metrics,
    resolve_validation_sampling,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out test-set metrics for every eligible fit in a fit manifest."
    )
    parser.add_argument("--fit_manifest_path", required=True, type=str)
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
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help="Recompute only baseline metrics and merge them into an existing test report.",
    )
    return parser.parse_args(argv)


def _load_fit_metadata(fit_root: str | Path) -> dict[str, Any]:
    metadata_path = Path(fit_root) / "fit_metadata.yaml"
    if not path_exists(metadata_path):
        raise FileNotFoundError(
            f"Could not infer fit metadata because {metadata_path} does not exist."
        )
    return load_yaml_mapping(metadata_path)


def _optional_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _resolved_split_kind(
    manifest_row: dict[str, str],
    metadata: dict[str, Any],
) -> str:
    split_kind = str(
        manifest_row.get("split_kind", metadata.get("split_kind", ""))
    ).strip().lower()
    if not split_kind:
        raise ValueError(
            "Could not infer split_kind from the fit manifest row or fit metadata."
        )
    return split_kind


def _resolve_experiment_path(
    manifest_row: dict[str, str],
    metadata: dict[str, Any],
    *,
    fit_root: str | Path,
) -> Path:
    resolved = str(
        manifest_row.get("experiment_path", metadata.get("experiment_path", ""))
    ).strip()
    if not resolved:
        raise ValueError(
            f"Could not infer experiment_path for fit {fit_root} from the manifest row or fit metadata."
        )
    return Path(resolved).resolve()


def _resolve_split_settings(
    manifest_row: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, int]:
    resolved_outer_num_folds = _optional_int(manifest_row, "outer_num_folds")
    if resolved_outer_num_folds is None:
        resolved_outer_num_folds = _optional_int(metadata, "outer_num_folds")
    resolved_test_fold_id = _optional_int(manifest_row, "test_fold_id")
    if resolved_test_fold_id is None:
        resolved_test_fold_id = _optional_int(metadata, "test_fold_id")
    resolved_inner_num_folds = _optional_int(manifest_row, "num_folds")
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


def _evaluate_manifest_row(
    manifest_row: dict[str, str],
    *,
    sampling: dict[str, Any] | None = None,
    baselines_only: bool = False,
) -> Path:
    fit_path = str(manifest_row.get("fit_path", "")).strip()
    if not fit_path:
        raise ValueError("Fit manifest row is missing fit_path.")
    fit_root = Path(fit_path).resolve()
    fit_metadata = _load_fit_metadata(fit_root)
    split_kind = _resolved_split_kind(manifest_row, fit_metadata)
    if split_kind != "test_train_cv":
        raise ValueError(
            f"Fit {fit_root} is not eligible for held-out test evaluation because split_kind={split_kind!r}."
        )

    resolved_experiment_root = _resolve_experiment_path(
        manifest_row,
        fit_metadata,
        fit_root=fit_root,
    )
    split_settings = _resolve_split_settings(manifest_row, fit_metadata)
    resolved_sampling = resolve_validation_sampling(sampling)
    split_artifacts = load_outer_test_split_masks(
        resolved_experiment_root,
        outer_num_folds=int(split_settings["outer_num_folds"]),
        test_fold_id=int(split_settings["test_fold_id"]),
        inner_num_folds=int(split_settings["inner_num_folds"]),
    )
    report_path = _default_output_path(
        fit_root,
        test_fold_id=int(split_settings["test_fold_id"]),
        inner_num_folds=int(split_settings["inner_num_folds"]),
    )
    panel_context = load_experiment_panel_context(resolved_experiment_root)
    baseline_metrics = evaluate_test_baseline_metrics(
        panel_context=panel_context,
        training_loss_mask=split_artifacts["training_mask"],
        test_loss_mask=split_artifacts["test_mask"],
    )
    if baselines_only:
        if not path_exists(report_path):
            raise FileNotFoundError(
                f"Cannot update baselines because {report_path} does not exist. "
                "Run the default test evaluation first."
            )
        payload = load_yaml_mapping(report_path)
        payload["baselines"] = baseline_metrics
        OmegaConf.save(OmegaConf.create(payload), io_path(report_path))
        return report_path

    metrics = evaluate_saved_fit_test_metrics(
        fit_root,
        resolved_experiment_root,
        training_loss_mask=split_artifacts["training_mask"],
        test_loss_mask=split_artifacts["test_mask"],
        sampling=resolved_sampling,
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
        "baselines": baseline_metrics,
    }
    os.makedirs(io_path(report_path.parent), exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), io_path(report_path))
    return report_path


def run_test_evaluation(
    fit_manifest_path: str | Path,
    *,
    sampling: dict[str, Any] | None = None,
    baselines_only: bool = False,
) -> dict[str, object]:
    manifest_rows = read_csv_rows(fit_manifest_path)
    if not manifest_rows:
        raise ValueError(f"No rows found in fit manifest {fit_manifest_path}.")

    evaluated_report_paths: list[str] = []
    skipped_rows: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        fit_path = str(manifest_row.get("fit_path", "")).strip()
        if not fit_path:
            raise ValueError("Fit manifest row is missing fit_path.")
        fit_root = Path(fit_path).resolve()
        fit_metadata = _load_fit_metadata(fit_root)
        split_kind = _resolved_split_kind(manifest_row, fit_metadata)
        if split_kind != "test_train_cv":
            skipped_rows.append(
                {
                    "fit_path": str(fit_root),
                    "split_kind": str(split_kind),
                    "reason": "split_kind is not test_train_cv",
                }
            )
            continue
        report_path = _evaluate_manifest_row(
            manifest_row,
            sampling=sampling,
            baselines_only=baselines_only,
        )
        evaluated_report_paths.append(str(report_path))

    return {
        "fit_manifest_path": str(Path(fit_manifest_path).resolve()),
        "evaluated_report_paths": evaluated_report_paths,
        "num_evaluated_rows": int(len(evaluated_report_paths)),
        "skipped_rows": skipped_rows,
        "num_skipped_rows": int(len(skipped_rows)),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run_test_evaluation(
        args.fit_manifest_path,
        sampling={
            "num_samples": int(args.num_samples),
            "gibbs_sweeps": int(args.gibbs_sweeps),
            "seed": int(args.seed),
        },
        baselines_only=bool(args.baselines_only),
    )
    for report_path in results["evaluated_report_paths"]:
        print(f"Test metrics report: {report_path}")
    print(
        f"Evaluated {results['num_evaluated_rows']} fit(s); skipped {results['num_skipped_rows']} ineligible row(s)."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
