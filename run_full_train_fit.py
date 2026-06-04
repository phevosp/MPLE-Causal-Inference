"""Train on the full outer training region using best-candidate hyperparameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from utils.io_utils import save_loss_mask
from utils.t0_path_utils import io_path, path_exists
from utils.loading_utils import load_experiment_panel_context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on the full outer training region using best-candidate hyperparameters."
    )
    parser.add_argument("--experiment_path", required=True, type=str)
    parser.add_argument(
        "--best_candidate_path",
        type=str,
        required=True,
        help="Path to best_candidate.yaml from a CV run.",
    )
    parser.add_argument("--outer_num_folds", type=int, default=5)
    parser.add_argument("--test_fold_id", type=int, default=1)
    parser.add_argument(
        "--inner_num_folds",
        type=int,
        default=5,
        help="Number of inner folds (to locate the split artifacts).",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_best_candidate_from_yaml(
    best_candidate_path: str | Path,
) -> dict[str, Any]:
    if not path_exists(best_candidate_path):
        raise FileNotFoundError(f"Best candidate YAML not found: {best_candidate_path}")
    config = OmegaConf.load(io_path(best_candidate_path))
    candidate_dict = dict(OmegaConf.to_container(config, resolve=True))

    reconstructed = {
        "name": candidate_dict.get("candidate_name", "full_train_candidate"),
        "slug": candidate_dict.get("candidate_slug", "full_train_candidate"),
        "_candidate_index": candidate_dict.get("candidate_index", 0),
        "_flat_params": {},
    }

    if "hyperparameters" in candidate_dict and isinstance(candidate_dict["hyperparameters"], dict):
        hyperparameters = candidate_dict["hyperparameters"]
        reconstructed["_flat_params"] = dict(hyperparameters)
        reconstructed.update(hyperparameters)

    return reconstructed



def _default_output_path(
    experiment_root: str | Path,
    *,
    test_fold_id: int,
    inner_num_folds: int,
    candidate_slug: str,
) -> Path:
    return (
        Path(experiment_root).resolve()
        / "full_train_fits"
        / f"test_fold_{int(test_fold_id)}__inner_folds_{int(inner_num_folds)}"
        / str(candidate_slug)
    )


def run_full_train_fit(
    experiment_path: str | Path,
    *,
    best_candidate_path: str | Path,
    outer_num_folds: int = 5,
    test_fold_id: int = 1,
    inner_num_folds: int = 5,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    from utils.t6_split_management import load_outer_test_split_masks

    experiment_root = Path(experiment_path).resolve()
    candidate = _load_best_candidate_from_yaml(best_candidate_path)

    output_root = (
        Path(output_path).resolve()
        if output_path not in (None, "")
        else _default_output_path(
            experiment_root,
            test_fold_id=test_fold_id,
            inner_num_folds=inner_num_folds,
            candidate_slug=candidate["slug"],
        )
    )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_root} already exists. Pass --overwrite to replace it."
            )
        import shutil
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    split_artifacts = load_outer_test_split_masks(
        experiment_root,
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        inner_num_folds=int(inner_num_folds),
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

    experiment_row = {
        "experiment_name": "",
        "experiment_slug": experiment_root.name,
        "experiment_path": str(experiment_root),
    }

    extra_metadata = {
        "execution_mode": "full_train",
        "outer_num_folds": int(outer_num_folds),
        "test_fold_id": int(test_fold_id),
        "inner_num_folds": int(inner_num_folds),
        "candidate_name": candidate.get("name", ""),
        "candidate_slug": candidate.get("slug", ""),
        "candidate_index": candidate.get("_candidate_index", 0),
        "num_training_slots": num_training_slots,
    }

    from run_fit_pipeline import execute_fit_root, materialize_fit_root

    materialize_fit_root(
        experiment_row,
        candidate,
        output_root,
        extra_input_artifacts={"loss_mask_path": str(mask_path.resolve())},
        extra_metadata=extra_metadata,
    )
    execute_fit_root(output_root)

    print(f"Full-train fit complete: {output_root}")
    return output_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_full_train_fit(
        args.experiment_path,
        best_candidate_path=args.best_candidate_path,
        outer_num_folds=args.outer_num_folds,
        test_fold_id=args.test_fold_id,
        inner_num_folds=args.inner_num_folds,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

