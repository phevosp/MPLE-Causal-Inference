"""Run MPLE fit variants over a generation manifest.

Supports three complementary workflows:

- plan fit requests into ``fit_requests.csv``
- execute one planned fit request by experiment/variant slug
- refresh ``fit_manifest.csv`` from completed fit outputs and rebuild fit reports

The sequential ``run_fits(...)`` entry point now reuses the same request-planning,
single-fit execution, and manifest-refresh helpers used by the shell/SLURM
wrappers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from data.synthetic_data_generation import derive_pre_intervention_steps
from io_utils import io_path
from intervention_utils import derive_post_intervention_steps
from pipeline_specs import (
    expand_named_entries,
    load_spec,
    read_csv_manifest,
    validate_fit_variant_dict,
    validate_fits_spec,
    write_csv_manifest,
)
from report_parameter_recovery_detailed import write_fit_reports


REPO_ROOT = Path(__file__).resolve().parent
FIT_REQUESTS_NAME = "fit_requests.csv"


def _path_text(path: str | Path) -> str:
    candidate = Path(path)
    if os.name == "nt":
        return str(candidate.resolve())
    if candidate.is_absolute():
        return str(candidate)
    return os.path.abspath(os.fspath(candidate))


def _expand_fit_variants(fits_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_fits_spec(fits_spec_path)
    variants = expand_named_entries(fits_spec_path, "variants")
    if not variants:
        raise ValueError(f"No variants found in fit spec {fits_spec_path}.")
    for variant in variants:
        variant["_spec_path"] = _path_text(fits_spec_path)
    return variants


def fit_manifest_path_for_spec(fits_spec_path: str | Path) -> Path:
    variants = _expand_fit_variants(fits_spec_path)
    return Path(str(variants[0]["fit_manifest_path"]))


def fit_requests_path_for_spec(fits_spec_path: str | Path) -> Path:
    return fit_manifest_path_for_spec(fits_spec_path).with_name(FIT_REQUESTS_NAME)


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
        "e": derive_post_intervention_steps(z),
    }


def build_fit_config(
    variant: dict[str, Any],
    dims: dict[str, int],
) -> object:
    validate_fit_variant_dict(variant)
    estimation = dict(variant.get("estimation", {}) or {})
    optimizer = dict(variant.get("optimizer", {}) or {})
    optimizer_mode = str(variant.get("optimizer_mode", "no_external_field"))
    latent_rank = (
        0
        if optimizer_mode in {"no_external_field", "nuclear_norm"}
        else int(variant.get("latent_rank", 0))
    )
    lambda_nuclear = float(variant.get("lambda_nuclear", 0.0))
    lambda_frobenius = float(variant.get("lambda_frobenius", 0.0))
    lambda_uv_ridge = float(variant.get("lambda_uv_ridge", 0.0))
    raw_v_column_l2_max = variant.get("v_column_l2_max", None)
    v_column_l2_max = (
        None if raw_v_column_l2_max is None else float(raw_v_column_l2_max)
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
            "e": dims["e"],
            "latent_rank": latent_rank,
            "optimizer_mode": optimizer_mode,
            "lambda_nuclear": lambda_nuclear,
            "lambda_frobenius": lambda_frobenius,
            "lambda_uv_ridge": lambda_uv_ridge,
            "v_column_l2_max": v_column_l2_max,
        },
        "estimation_params": {
            "fixed_scalar_params": dict(estimation.get("fixed_scalar_params", {}) or {}),
            "warm_start_fixed_scalars": dict(
                estimation.get("warm_start_fixed_scalars", {}) or {}
            ),
            "warm_start_steps": int(estimation.get("warm_start_steps", 0)),
            "beta_mask_pre_s": bool(estimation.get("beta_mask_pre_s", False)),
            "beta_mask_post_e": bool(estimation.get("beta_mask_post_e", False)),
        },
        "optimizer_params": optimizer_config,
    }
    return OmegaConf.create(config_dict)


def _fit_input_artifacts(
    experiment_root: str | Path,
    *,
    extra_input_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    experiment_path = Path(experiment_root)
    artifacts = {
        "model_artifact_dir": _path_text(experiment_path),
        "truth_artifact_dir": _path_text(experiment_path),
        "panel_path": _path_text(experiment_path / "panel_data.npz"),
        "x0_path": _path_text(experiment_path / "x_0.npy"),
    }
    if extra_input_artifacts:
        artifacts.update(extra_input_artifacts)
    return artifacts


def materialize_fit_root(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    fit_root: str | Path,
    *,
    extra_input_artifacts: dict[str, object] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> tuple[Path, object, dict[str, object]]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_root_path = Path(fit_root)
    dims = infer_panel_dimensions(experiment_root)
    fit_config = build_fit_config(variant, dims)
    fit_config.input_artifacts = OmegaConf.create(
        _fit_input_artifacts(
            experiment_root,
            extra_input_artifacts=extra_input_artifacts,
        )
    )
    config_path = fit_root_path / "fit_realized_config.yaml"
    fixed_scalar_params = OmegaConf.to_container(
        fit_config.estimation_params.fixed_scalar_params, resolve=True
    )
    fit_metadata = {
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fits_spec_path": _path_text(variant["_spec_path"]) if variant.get("_spec_path") else "",
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get(
            "descriptor", experiment_row.get("experiment_name", "")
        ),
        "experiment_path": _path_text(experiment_root),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "field_mode": experiment_row.get("field_mode", ""),
        "model_artifact_dir": _path_text(experiment_root),
        "truth_artifact_dir": _path_text(experiment_root),
        "panel_path": _path_text(experiment_root / "panel_data.npz"),
        "x0_path": _path_text(experiment_root / "x_0.npy"),
        "latent_rank": int(fit_config.global_params.latent_rank),
        "optimizer_mode": str(fit_config.global_params.optimizer_mode),
        "lambda_nuclear": float(fit_config.global_params.lambda_nuclear),
        "lambda_frobenius": float(fit_config.global_params.lambda_frobenius),
        "lambda_uv_ridge": float(fit_config.global_params.lambda_uv_ridge),
        "fixed_scalar_params": fixed_scalar_params,
        **dims,
    }
    if extra_metadata:
        fit_metadata.update(extra_metadata)
    OmegaConf.save(fit_config, io_path(config_path))
    OmegaConf.save(
        OmegaConf.create(fit_metadata),
        io_path(fit_root_path / "fit_metadata.yaml"),
    )
    return fit_root_path, fit_config, fit_metadata


def execute_fit_root(fit_root: str | Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "mple.py"),
        "--data_folder",
        str(Path(fit_root)),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _fit_request_row(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    generation_manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> dict[str, object]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_path = experiment_root / str(variant["fit_root_name"]) / str(variant["slug"])
    return {
        "generation_manifest_path": _path_text(generation_manifest_path),
        "fits_spec_path": _path_text(fits_spec_path),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "variant_name": str(variant["name"]),
        "variant_slug": str(variant["slug"]),
        "fit_path": _path_text(fit_path),
    }


def write_fit_requests(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    generation_rows = read_csv_manifest(manifest_path)
    if not generation_rows:
        raise ValueError(f"No experiments found in generation manifest {manifest_path}.")
    variants = _expand_fit_variants(fits_spec_path)
    request_rows: list[dict[str, object]] = []
    for experiment_row in generation_rows:
        for variant in variants:
            request_rows.append(
                _fit_request_row(
                    experiment_row,
                    variant,
                    manifest_path,
                    fits_spec_path,
                )
            )
    request_path = fit_requests_path_for_spec(fits_spec_path)
    write_csv_manifest(request_path, request_rows)
    return request_path


def _select_generation_row(
    manifest_path: str | Path,
    experiment_slug: str,
) -> dict[str, str]:
    matches = [
        row
        for row in read_csv_manifest(manifest_path)
        if row.get("experiment_slug", "") == experiment_slug
    ]
    if not matches:
        raise ValueError(
            f"No experiment with slug '{experiment_slug}' found in {manifest_path}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Experiment slug '{experiment_slug}' is not unique in {manifest_path}."
        )
    return matches[0]


def _select_fit_variant(
    fits_spec_path: str | Path,
    variant_slug: str,
) -> dict[str, Any]:
    matches = [
        variant
        for variant in _expand_fit_variants(fits_spec_path)
        if str(variant["slug"]) == variant_slug
    ]
    if not matches:
        raise ValueError(
            f"No fit variant with slug '{variant_slug}' found in {fits_spec_path}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Fit variant slug '{variant_slug}' is not unique in {fits_spec_path}."
        )
    return matches[0]


def run_fit_variant(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_root = experiment_root / str(variant["fit_root_name"]) / str(variant["slug"])
    if fit_root.exists():
        if overwrite:
            shutil.rmtree(fit_root)
        else:
            raise FileExistsError(
                f"{fit_root} already exists. Re-run with --overwrite to rebuild it."
            )
    fit_root.mkdir(parents=True, exist_ok=False)
    fit_root_path, fit_config, _ = materialize_fit_root(
        experiment_row,
        variant,
        fit_root,
    )
    execute_fit_root(fit_root_path)
    dims = infer_panel_dimensions(experiment_root)
    fixed_scalar_params = OmegaConf.to_container(
        fit_config.estimation_params.fixed_scalar_params, resolve=True
    )
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": _path_text(experiment_root),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "field_mode": experiment_row.get("field_mode", ""),
        "variant_name": variant["name"],
        "variant_slug": variant["slug"],
        "fit_path": _path_text(fit_root),
        "N": dims["N"],
        "T": dims["T"],
        "s": dims["s"],
        "latent_rank": int(fit_config.global_params.latent_rank),
        "optimizer_mode": str(fit_config.global_params.optimizer_mode),
        "lambda_nuclear": float(fit_config.global_params.lambda_nuclear),
        "lambda_frobenius": float(fit_config.global_params.lambda_frobenius),
        "lambda_uv_ridge": float(fit_config.global_params.lambda_uv_ridge),
        "fixed_scalar_params": str(fixed_scalar_params),
        "status": "completed",
    }


def run_fit_request(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    experiment_slug: str,
    variant_slug: str,
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_row = _select_generation_row(manifest_path, experiment_slug)
    variant = _select_fit_variant(fits_spec_path, variant_slug)
    return run_fit_variant(experiment_row, variant, overwrite=overwrite)


def _manifest_row_from_completed_fit(request_row: dict[str, str]) -> dict[str, object]:
    fit_root = Path(_path_text(request_row["fit_path"]))
    metadata_path = fit_root / "fit_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fit metadata: {metadata_path}")
    metadata = load_spec(metadata_path)
    fixed_scalar_params = metadata.get("fixed_scalar_params", {})
    return {
        "experiment_name": str(metadata.get("experiment_name", "")),
        "experiment_slug": str(
            metadata.get("experiment_slug", request_row.get("experiment_slug", ""))
        ),
        "descriptor": str(
            metadata.get("descriptor", metadata.get("experiment_name", ""))
        ),
        "experiment_path": str(metadata.get("experiment_path", "")),
        "intervention_source": str(metadata.get("intervention_source", "")),
        "graph_source": str(metadata.get("graph_source", "")),
        "field_mode": str(metadata.get("field_mode", "")),
        "variant_name": str(metadata.get("variant_name", "")),
        "variant_slug": str(
            metadata.get("variant_slug", request_row.get("variant_slug", ""))
        ),
        "fit_path": str(fit_root),
        "N": int(metadata.get("N", 0)),
        "T": int(metadata.get("T", 0)),
        "s": int(metadata.get("s", 0)),
        "latent_rank": int(metadata.get("latent_rank", 0)),
        "optimizer_mode": str(metadata.get("optimizer_mode", "no_external_field")),
        "lambda_nuclear": float(metadata.get("lambda_nuclear", 0.0)),
        "lambda_frobenius": float(metadata.get("lambda_frobenius", 0.0)),
        "lambda_uv_ridge": float(metadata.get("lambda_uv_ridge", 0.0)),
        "fixed_scalar_params": str(fixed_scalar_params),
        "status": "completed",
    }


def refresh_fit_manifest(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    request_path = fit_requests_path_for_spec(fits_spec_path)
    if not request_path.exists():
        write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_manifest(request_path)
    fit_rows = [_manifest_row_from_completed_fit(request_row) for request_row in request_rows]
    fit_manifest_path = fit_manifest_path_for_spec(fits_spec_path)
    write_csv_manifest(fit_manifest_path, fit_rows)
    write_fit_reports(fit_manifest_path)
    return fit_manifest_path


def run_fits(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    overwrite: bool = False,
) -> Path:
    request_path = write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_manifest(request_path)
    print(f"Loaded {len(request_rows)} fit request(s) from {request_path}.")
    for request_row in request_rows:
        experiment_name = request_row.get("experiment_name", request_row["experiment_slug"])
        variant_name = request_row.get("variant_name", request_row["variant_slug"])
        print(f"Running fit '{variant_name}' for experiment '{experiment_name}'...")
        run_fit_request(
            manifest_path,
            fits_spec_path,
            request_row["experiment_slug"],
            request_row["variant_slug"],
            overwrite=overwrite,
        )
    fit_manifest_path = refresh_fit_manifest(manifest_path, fits_spec_path)
    print(f"Wrote fit manifest to {fit_manifest_path}.")
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
    parser.add_argument(
        "--write_requests",
        action="store_true",
        help="Write fit_requests.csv for the configured generation manifest and fit spec.",
    )
    parser.add_argument(
        "--run_request",
        action="store_true",
        help="Run one planned fit request selected by --experiment_slug and --variant_slug.",
    )
    parser.add_argument(
        "--refresh_manifest",
        action="store_true",
        help="Refresh fit_manifest.csv from completed fit outputs and rebuild reports.",
    )
    parser.add_argument("--experiment_slug", type=str, default="")
    parser.add_argument("--variant_slug", type=str, default="")
    args = parser.parse_args()

    if args.dry_run:
        generation_rows = read_csv_manifest(args.manifest_path)
        variants = _expand_fit_variants(args.fits_spec_path)
        print(
            f"Dry run: {len(generation_rows)} experiment(s) × {len(variants)} variant(s) "
            f"= {len(generation_rows) * len(variants)} fit(s) planned."
        )
        for row in generation_rows:
            for variant in variants:
                print(f"  {row.get('experiment_name', '?')} / {variant['name']}")
        return

    if args.write_requests:
        request_path = write_fit_requests(args.manifest_path, args.fits_spec_path)
        print(f"Fit requests: {request_path}")
        return

    if args.run_request:
        if not args.experiment_slug.strip():
            raise ValueError("--experiment_slug is required when --run_request is set.")
        if not args.variant_slug.strip():
            raise ValueError("--variant_slug is required when --run_request is set.")
        row = run_fit_request(
            args.manifest_path,
            args.fits_spec_path,
            args.experiment_slug.strip(),
            args.variant_slug.strip(),
            overwrite=args.overwrite,
        )
        print(f"Completed fit: {row['fit_path']}")
        return

    if args.refresh_manifest:
        fit_manifest_path = refresh_fit_manifest(
            args.manifest_path,
            args.fits_spec_path,
        )
        print(f"Fit manifest: {fit_manifest_path}")
        return

    fit_manifest_path = run_fits(
        args.manifest_path,
        args.fits_spec_path,
        overwrite=args.overwrite,
    )
    print(f"Fit manifest: {fit_manifest_path}")


if __name__ == "__main__":
    main()
