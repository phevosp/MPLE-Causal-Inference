"""Run MPLE fit variants over a generation manifest.

Reads a generation manifest and a fits_spec.yaml, builds per-variant fit configs,
invokes mple.py via subprocess for each (experiment, variant) pair, and writes a
fit_manifest.csv and per-experiment fit summary reports.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from data.synthetic_data_generation import derive_pre_intervention_steps
from pipeline_specs import (
    expand_named_entries,
    read_csv_manifest,
    validate_fits_spec,
    write_csv_manifest,
)
from report_parameter_recovery_detailed import write_fit_reports


REPO_ROOT = Path(__file__).resolve().parent


def infer_panel_dimensions(experiment_path: str | Path) -> dict[str, int]:
    experiment_root = Path(experiment_path)
    panel_path = experiment_root / "panel_data.npz"
    x0_path = experiment_root / "x_0.npy"
    with np.load(panel_path, allow_pickle=False) as panel:
        x = np.asarray(panel["x"], dtype=float)
        z = np.asarray(panel["z"], dtype=float)
    x_0 = np.asarray(np.load(x0_path), dtype=float)
    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError(
            f"Experiment panel must contain matching 2D x/z arrays: {panel_path}"
        )
    if x_0.shape != (x.shape[1],):
        raise ValueError(
            f"x_0 shape {x_0.shape} does not match panel width {x.shape[1]}."
        )
    return {
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "s": derive_pre_intervention_steps(z),
    }


def build_fit_config(
    variant: dict[str, Any],
    dims: dict[str, int],
) -> object:
    optimizer = dict(variant.get("optimizer", {}) or {})
    optimizer_mode = str(variant.get("optimizer_mode", "no_external_field"))
    if optimizer_mode not in {
        "no_external_field",
        "nuclear_norm",
        "exact_rank_manifold",
        "alternating_latent_rank",
        "concurrent_latent_rank",
    }:
        raise ValueError(
            "optimizer_mode must be one of 'no_external_field', 'nuclear_norm', "
            "'exact_rank_manifold', 'alternating_latent_rank', or "
            "'concurrent_latent_rank'."
        )
    latent_rank = (
        0
        if optimizer_mode in {"no_external_field", "nuclear_norm"}
        else int(variant.get("latent_rank", 0))
    )
    if optimizer_mode in {"alternating_latent_rank", "concurrent_latent_rank"} and latent_rank <= 0:
        raise ValueError(
            f"latent_rank must be positive for optimizer_mode='{optimizer_mode}'."
        )
    if optimizer_mode == "exact_rank_manifold" and latent_rank <= 0:
        raise ValueError(
            "latent_rank must be positive for optimizer_mode='exact_rank_manifold'."
        )
    lambda_nuclear = float(variant.get("lambda_nuclear", 0.0))
    if lambda_nuclear < 0.0:
        raise ValueError("lambda_nuclear must be nonnegative.")
    lambda_frobenius = float(variant.get("lambda_frobenius", 0.0))
    if lambda_frobenius < 0.0:
        raise ValueError("lambda_frobenius must be nonnegative.")
    lambda_uv_ridge = float(variant.get("lambda_uv_ridge", 0.0))
    if lambda_uv_ridge < 0.0:
        raise ValueError("lambda_uv_ridge must be nonnegative.")
    if optimizer_mode != "nuclear_norm" and lambda_nuclear != 0.0:
        raise ValueError("lambda_nuclear is only valid for optimizer_mode='nuclear_norm'.")
    if optimizer_mode != "exact_rank_manifold" and lambda_frobenius != 0.0:
        raise ValueError(
            "lambda_frobenius is only valid for optimizer_mode='exact_rank_manifold'."
        )
    if (
        optimizer_mode not in {"alternating_latent_rank", "concurrent_latent_rank"}
        and lambda_uv_ridge != 0.0
    ):
        raise ValueError(
            "lambda_uv_ridge is only valid for optimizer_mode='alternating_latent_rank' "
            "or 'concurrent_latent_rank'."
        )

    optimizer_config: dict[str, Any] = {
        "steps": int(optimizer["steps"]),
        "tol": float(optimizer["tol"]),
        "seed": int(optimizer["seed"]),
        "n_starts": int(optimizer.get("n_starts", 1)),
        "proximal_lr": float(optimizer.get("proximal_lr", 1.0)),
    }

    config_dict: dict[str, Any] = {
        "global_params": {
            "N": dims["N"],
            "T": dims["T"],
            "s": dims["s"],
            "latent_rank": latent_rank,
            "optimizer_mode": optimizer_mode,
            "lambda_nuclear": lambda_nuclear,
            "lambda_frobenius": lambda_frobenius,
            "lambda_uv_ridge": lambda_uv_ridge,
        },
        "estimation_params": {
            "fixed_scalar_params": dict(
                (variant.get("estimation", {}) or {}).get("fixed_scalar_params", {})
                or {}
            ),
        },
        "optimizer_params": optimizer_config,
    }
    return OmegaConf.create(config_dict)


def run_fit_variant(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_root = experiment_root / str(variant["fit_root_name"]) / variant["slug"]
    if fit_root.exists():
        if overwrite:
            shutil.rmtree(fit_root)
        else:
            raise FileExistsError(
                f"{fit_root} already exists. Re-run with --overwrite to rebuild it."
            )
    fit_root.mkdir(parents=True, exist_ok=False)

    dims = infer_panel_dimensions(experiment_root)
    fit_config = build_fit_config(variant, dims)
    fit_config.input_artifacts = OmegaConf.create(
        {
            "model_artifact_dir": str(experiment_root.resolve()),
            "truth_artifact_dir": str(experiment_root.resolve()),
            "panel_path": str((experiment_root / "panel_data.npz").resolve()),
            "x0_path": str((experiment_root / "x_0.npy").resolve()),
        }
    )
    config_path = fit_root / "fit_realized_config.yaml"
    fit_metadata = {
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fits_spec_path": str(Path(variant["_spec_path"]).resolve()),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_path": str(experiment_root.resolve()),
        "field_mode": experiment_row.get("field_mode", ""),
        "model_artifact_dir": str(experiment_root.resolve()),
        "truth_artifact_dir": str(experiment_root.resolve()),
        "panel_path": str((experiment_root / "panel_data.npz").resolve()),
        "x0_path": str((experiment_root / "x_0.npy").resolve()),
        "latent_rank": int(fit_config.global_params.latent_rank),
        "optimizer_mode": str(fit_config.global_params.optimizer_mode),
        "lambda_nuclear": float(fit_config.global_params.lambda_nuclear),
        "lambda_frobenius": float(fit_config.global_params.lambda_frobenius),
        "lambda_uv_ridge": float(fit_config.global_params.lambda_uv_ridge),
        **dims,
    }
    OmegaConf.save(fit_config, config_path)
    OmegaConf.save(OmegaConf.create(fit_metadata), fit_root / "fit_metadata.yaml")

    command = [
        sys.executable,
        str(REPO_ROOT / "mple.py"),
        "--data_folder",
        str(fit_root),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": str(experiment_root.resolve()),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "field_mode": experiment_row.get("field_mode", ""),
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fit_path": str(fit_root.resolve()),
        "N": dims["N"],
        "T": dims["T"],
        "s": dims["s"],
        "latent_rank": int(fit_config.global_params.latent_rank),
        "optimizer_mode": str(fit_config.global_params.optimizer_mode),
        "lambda_nuclear": float(fit_config.global_params.lambda_nuclear),
        "lambda_frobenius": float(fit_config.global_params.lambda_frobenius),
        "lambda_uv_ridge": float(fit_config.global_params.lambda_uv_ridge),
        "fixed_scalar_params": str(
            OmegaConf.to_container(
                fit_config.estimation_params.fixed_scalar_params, resolve=True
            )
        ),
        "status": "completed",
    }


def run_fits(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    overwrite: bool = False,
) -> Path:
    validate_fits_spec(fits_spec_path)
    generation_rows = read_csv_manifest(manifest_path)
    variants = expand_named_entries(fits_spec_path, "variants")
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {manifest_path}."
        )
    if not variants:
        raise ValueError(f"No variants found in fit spec {fits_spec_path}.")
    for variant in variants:
        variant["_spec_path"] = str(Path(fits_spec_path))

    fit_rows: list[dict[str, object]] = []
    fit_manifest_path = Path(str(variants[0]["fit_manifest_path"]))
    for experiment_row in generation_rows:
        for variant in variants:
            fit_rows.append(
                run_fit_variant(experiment_row, variant, overwrite=overwrite)
            )

    write_csv_manifest(fit_manifest_path, fit_rows)
    write_fit_reports(fit_manifest_path)
    return fit_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MPLE fit variants over a generation manifest."
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
        default="data/configs/fits_spec.yaml",
        help="Path to the fits YAML spec defining optimizer variants (default: data/configs/fits_spec.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete and rebuild existing fit directories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs and print the planned (experiment, variant) work without executing any fits.",
    )
    args = parser.parse_args()

    if args.dry_run:
        generation_rows = read_csv_manifest(args.manifest_path)
        variants = expand_named_entries(args.fits_spec_path, "variants")
        print(
            f"Dry run: {len(generation_rows)} experiment(s) × {len(variants)} variant(s) "
            f"= {len(generation_rows) * len(variants)} fit(s) planned."
        )
        for row in generation_rows:
            for variant in variants:
                print(f"  {row.get('experiment_name', '?')} / {variant['name']}")
        return

    fit_manifest_path = run_fits(
        args.manifest_path,
        args.fits_spec_path,
        overwrite=args.overwrite,
    )
    print(f"Fit manifest: {fit_manifest_path}")


if __name__ == "__main__":
    main()
