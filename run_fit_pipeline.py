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
from pipeline_specs import expand_named_entries, read_csv_manifest, write_csv_manifest


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
        raise ValueError(f"Experiment panel must contain matching 2D x/z arrays: {panel_path}")
    if x_0.shape != (x.shape[1],):
        raise ValueError(f"x_0 shape {x_0.shape} does not match panel width {x.shape[1]}.")
    return {"N": int(x.shape[1]), "T": int(x.shape[0]), "s": derive_pre_intervention_steps(z)}

def build_fit_config(
    variant: dict[str, Any],
    dims: dict[str, int],
) -> tuple[object, float]:
    estimation = dict(variant.get("estimation", {}) or {})
    optimizer = dict(variant.get("optimizer", {}) or {})
    bound_B = float(variant["B"])
    latent_rank = int(variant.get("latent_rank", 0))
    if latent_rank < 0:
        raise ValueError("latent_rank must be nonnegative.")

    config_dict: dict[str, Any] = {
        "global_params": {
            "N": dims["N"],
            "T": dims["T"],
            "s": dims["s"],
            "B": bound_B,
            "latent_rank": latent_rank,
        },
        "estimation_params": {
            "fit_intervention_model": bool(estimation["fit_intervention_model"]),
            "beta_mask_pre_intervention": bool(estimation["beta_mask_pre_intervention"]),
            "beta_mask_rescale": bool(estimation["beta_mask_rescale"]),
            "fixed_scalar_params": dict(estimation.get("fixed_scalar_params", {}) or {}),
        },
        "optimizer_params": {
            "steps": int(optimizer["steps"]),
            "tol": float(optimizer["tol"]),
            "seed": int(optimizer["seed"]),
        },
    }
    return OmegaConf.create(config_dict), bound_B


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
    fit_config, resolved_B = build_fit_config(variant, dims)
    config_path = fit_root / "fit_realized_config.yaml"
    fit_metadata = {
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fits_spec_path": str(Path(variant["_spec_path"]).resolve()),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_path": str(experiment_root.resolve()),
        "model_artifact_dir": str(experiment_root.resolve()),
        "truth_artifact_dir": str(experiment_root.resolve()),
        "panel_path": str((experiment_root / "panel_data.npz").resolve()),
        "x0_path": str((experiment_root / "x_0.npy").resolve()),
        "z0_path": str((experiment_root / "z_0.npy").resolve()),
        "requested_B": variant.get("B"),
        "resolved_B": float(resolved_B),
        "latent_rank": int(fit_config.global_params.latent_rank),
        **dims,
    }
    OmegaConf.save(fit_config, config_path)
    OmegaConf.save(OmegaConf.create(fit_metadata), fit_root / "fit_metadata.yaml")

    command = [
        sys.executable,
        str(REPO_ROOT / "mple.py"),
        "--data_folder",
        str(fit_root),
        "--config_path",
        str(config_path),
        "--model_artifact_dir",
        str(experiment_root),
        "--truth_artifact_dir",
        str(experiment_root),
        "--panel_path",
        str(experiment_root / "panel_data.npz"),
        "--x0_path",
        str(experiment_root / "x_0.npy"),
        "--z0_path",
        str(experiment_root / "z_0.npy"),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_path": str(experiment_root.resolve()),
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fit_path": str(fit_root.resolve()),
        "N": dims["N"],
        "T": dims["T"],
        "s": dims["s"],
        "B": float(resolved_B),
        "latent_rank": int(fit_config.global_params.latent_rank),
        "fit_intervention_model": bool(fit_config.estimation_params.fit_intervention_model),
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
    generation_rows = read_csv_manifest(manifest_path)
    variants = expand_named_entries(fits_spec_path, "variants")
    if not generation_rows:
        raise ValueError(f"No experiments found in generation manifest {manifest_path}.")
    if not variants:
        raise ValueError(f"No variants found in fit spec {fits_spec_path}.")
    for variant in variants:
        variant["_spec_path"] = str(Path(fits_spec_path))

    fit_rows: list[dict[str, object]] = []
    fit_manifest_path = Path(str(variants[0]["fit_manifest_path"]))
    for experiment_row in generation_rows:
        for variant in variants:
            fit_rows.append(run_fit_variant(experiment_row, variant, overwrite=overwrite))

    write_csv_manifest(fit_manifest_path, fit_rows)
    return fit_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MPLE fit variants over a generation manifest."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument(
        "--fits_spec_path",
        type=str,
        default="data/configs/fits_spec.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fit_manifest_path = run_fits(
        args.manifest_path,
        args.fits_spec_path,
        overwrite=args.overwrite,
    )
    print(f"Fit manifest: {fit_manifest_path}")


if __name__ == "__main__":
    main()
