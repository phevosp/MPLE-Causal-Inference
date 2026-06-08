"""Build split bundles for train_cv and test_train_cv searches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.t0_csv_utils import read_csv_rows
from utils.t6_pipeline_spec_utils import expand_named_entries, validate_cv_spec
from utils.t6_split_engine import (
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    SPLIT_KIND_TEST_TRAIN_CV,
    SPLIT_KIND_TRAIN_CV,
    _build_split_for_experiment,
    build_test_train_cv_bundle,
    build_train_cv_bundle,
    test_train_cv_split_output_root,
    train_cv_split_output_root,
    write_split_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build split bundles for every experiment and split request defined in a CV spec."
        )
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--cv_spec_path", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _manifest_rows(generation_manifest_path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(generation_manifest_path)
    if not rows:
        raise ValueError(f"No rows found in generation manifest {generation_manifest_path}.")
    return rows


def _searches(cv_spec_path: str | Path) -> list[dict[str, object]]:
    validate_cv_spec(cv_spec_path)
    searches = expand_named_entries(cv_spec_path, "searches")
    if not searches:
        raise ValueError(f"No searches found in CV spec {cv_spec_path}.")
    return searches


def _build_requests_from_searches(searches: list[dict[str, object]]) -> list[dict[str, int | str]]:
    seen: set[tuple[str, int, int, int]] = set()
    requests: list[dict[str, int | str]] = []
    for search in searches:
        split_kind = str(search.get("split_kind", SPLIT_KIND_TRAIN_CV)).strip().lower()
        num_folds = int(search.get("num_folds", 5))
        if split_kind == SPLIT_KIND_TRAIN_CV:
            key = (split_kind, int(num_folds), 0, 0)
            if key in seen:
                continue
            seen.add(key)
            requests.append(
                {
                    "split_kind": split_kind,
                    "num_folds": int(num_folds),
                    "outer_num_folds": 0,
                    "test_fold_id": 0,
                }
            )
            continue

        outer_num_folds = int(search.get("outer_num_folds", DEFAULT_OUTER_NUM_FOLDS))
        test_fold_id = int(search.get("test_fold_id", DEFAULT_TEST_FOLD_ID))
        key = (split_kind, int(num_folds), int(outer_num_folds), int(test_fold_id))
        if key in seen:
            continue
        seen.add(key)
        requests.append(
            {
                "split_kind": split_kind,
                "num_folds": int(num_folds),
                "outer_num_folds": int(outer_num_folds),
                "test_fold_id": int(test_fold_id),
            }
        )
    return requests


def build_splits(
    generation_manifest_path: str | Path,
    cv_spec_path: str | Path,
    *,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = 1.0e-12,
    overwrite: bool = False,
) -> list[Path]:
    rows = _manifest_rows(generation_manifest_path)
    requests = _build_requests_from_searches(_searches(cv_spec_path))
    output_paths: list[Path] = []
    for row in rows:
        experiment_path = str(row.get("experiment_path", "")).strip()
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        experiment_root = Path(experiment_path).resolve()
        for request in requests:
            split_kind = str(request["split_kind"])
            if split_kind == SPLIT_KIND_TRAIN_CV:
                bundle = build_train_cv_bundle(
                    experiment_root,
                    num_folds=int(request["num_folds"]),
                    seed=int(seed),
                    recursive=bool(recursive),
                    contiguous=bool(contiguous),
                    tolerance=float(tolerance),
                )
                output_root = train_cv_split_output_root(
                    experiment_root,
                    num_folds=int(request["num_folds"]),
                )
            else:
                bundle = build_test_train_cv_bundle(
                    experiment_root,
                    outer_num_folds=int(request["outer_num_folds"]),
                    test_fold_id=int(request["test_fold_id"]),
                    inner_num_folds=int(request["num_folds"]),
                    seed=int(seed),
                    recursive=bool(recursive),
                    contiguous=bool(contiguous),
                    tolerance=float(tolerance),
                )
                output_root = test_train_cv_split_output_root(
                    experiment_root,
                    outer_num_folds=int(request["outer_num_folds"]),
                    test_fold_id=int(request["test_fold_id"]),
                    inner_num_folds=int(request["num_folds"]),
                )
            write_split_bundle(output_root, bundle, overwrite=bool(overwrite))
            output_paths.append(output_root)
            if experiment_name:
                print(f"Built {split_kind} split for {experiment_name}: {output_root}", flush=True)
    return output_paths


def build_train_cv_splits(
    generation_manifest_path: str | Path,
    *,
    num_folds: int = 5,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = 1.0e-12,
    overwrite: bool = False,
) -> list[Path]:
    output_paths: list[Path] = []
    for row in _manifest_rows(generation_manifest_path):
        experiment_path = str(row.get("experiment_path", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        output_paths.append(
            _build_split_for_experiment(
                experiment_path,
                split_kind=SPLIT_KIND_TRAIN_CV,
                num_folds=int(num_folds),
                seed=int(seed),
                recursive=bool(recursive),
                contiguous=bool(contiguous),
                tolerance=float(tolerance),
                overwrite=bool(overwrite),
            )
        )
    return output_paths


def build_test_train_cv_splits(
    generation_manifest_path: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = 5,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = 1.0e-12,
    overwrite: bool = False,
) -> list[Path]:
    output_paths: list[Path] = []
    for row in _manifest_rows(generation_manifest_path):
        experiment_path = str(row.get("experiment_path", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        output_paths.append(
            _build_split_for_experiment(
                experiment_path,
                split_kind=SPLIT_KIND_TEST_TRAIN_CV,
                num_folds=int(inner_num_folds),
                outer_num_folds=int(outer_num_folds),
                test_fold_id=int(test_fold_id),
                seed=int(seed),
                recursive=bool(recursive),
                contiguous=bool(contiguous),
                tolerance=float(tolerance),
                overwrite=bool(overwrite),
            )
        )
    return output_paths


def build_validation_test_splits_for_experiment(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = 5,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = 1.0e-12,
    overwrite: bool = False,
) -> Path:
    if int(outer_num_folds) < 1:
        raise ValueError(f"outer_num_folds must be >= 1, got {outer_num_folds}.")
    if int(inner_num_folds) < 1:
        raise ValueError(f"inner_num_folds must be >= 1, got {inner_num_folds}.")
    return _build_split_for_experiment(
        experiment_root,
        split_kind=SPLIT_KIND_TEST_TRAIN_CV,
        num_folds=int(inner_num_folds),
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
        overwrite=bool(overwrite),
    )


def create_validation_test_splits(
    generation_manifest_path: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int = 5,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = 1.0e-12,
    overwrite: bool = False,
) -> list[Path]:
    return build_test_train_cv_splits(
        generation_manifest_path,
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(inner_num_folds),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
        overwrite=bool(overwrite),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_paths = build_splits(
        args.generation_manifest_path,
        args.cv_spec_path,
        seed=args.seed,
        recursive=args.recursive,
        contiguous=args.contiguous,
        tolerance=args.tolerance,
        overwrite=args.overwrite,
    )
    print(f"Built {len(output_paths)} split bundles.")


if __name__ == "__main__":
    main(sys.argv[1:])
