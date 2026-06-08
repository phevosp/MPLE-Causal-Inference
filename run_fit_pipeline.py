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
import shutil
from pathlib import Path
from typing import Any

from utils.t6_fit_materialization import (
    execute_fit_root,
    materialize_fit_root,
    path_text,
)
from utils.t0_config_utils import load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows, write_csv_rows
from utils.t6_pipeline_spec_utils import expand_named_entries, validate_fits_spec
from report_parameter_recovery_detailed import write_fit_reports

FIT_REQUESTS_NAME = "fit_requests.csv"


def _expand_fit_variants(fits_spec_path: str | Path) -> list[dict[str, Any]]:
    validate_fits_spec(fits_spec_path)
    variants = expand_named_entries(fits_spec_path, "variants")
    if not variants:
        raise ValueError(f"No variants found in fit spec {fits_spec_path}.")
    for variant in variants:
        variant["_spec_path"] = path_text(fits_spec_path)
    return variants


def fit_manifest_path_for_spec(fits_spec_path: str | Path) -> Path:
    """Return path to the fit manifest for a given fits spec."""
    variants = _expand_fit_variants(fits_spec_path)
    return Path(str(variants[0]["fit_manifest_path"]))


def fit_requests_path_for_spec(fits_spec_path: str | Path) -> Path:
    """Return path to the fit requests for a given fits spec."""
    return fit_manifest_path_for_spec(fits_spec_path).with_name(FIT_REQUESTS_NAME)


def _fit_request_row(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    generation_manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> dict[str, object]:
    experiment_root = Path(experiment_row["experiment_path"])
    fit_path = experiment_root / str(variant["fit_root_name"]) / str(variant["slug"])
    return {
        "generation_manifest_path": path_text(generation_manifest_path),
        "fits_spec_path": path_text(fits_spec_path),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "variant_name": str(variant["name"]),
        "variant_slug": str(variant["slug"]),
        "fit_path": path_text(fit_path),
    }


def _select_generation_row(
    manifest_path: str | Path,
    experiment_slug: str,
) -> dict[str, str]:
    matches = [
        row
        for row in read_csv_rows(manifest_path)
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


def _manifest_row_from_completed_fit(request_row: dict[str, str]) -> dict[str, object]:
    fit_root = Path(path_text(request_row["fit_path"]))
    metadata_path = fit_root / "fit_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fit metadata: {metadata_path}")
    metadata = load_yaml_mapping(metadata_path)
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


def write_fit_requests(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    generation_rows = read_csv_rows(manifest_path)
    if not generation_rows:
        raise ValueError(
            f"No experiments found in generation manifest {manifest_path}."
        )
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
    write_csv_rows(request_path, request_rows)
    return request_path


def run_fit_variant(
    experiment_row: dict[str, str],
    variant: dict[str, Any],
    overwrite: bool = False,
) -> Path:
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
    fit_root_path, _, _ = materialize_fit_root(
        experiment_row,
        variant,
        fit_root,
    )
    execute_fit_root(fit_root_path)
    return fit_root_path


def run_fit_request(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    experiment_slug: str,
    variant_slug: str,
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_row = _select_generation_row(manifest_path, experiment_slug)
    variant = _select_fit_variant(fits_spec_path, variant_slug)
    request_row = _fit_request_row(
        experiment_row,
        variant,
        manifest_path,
        fits_spec_path,
    )
    run_fit_variant(experiment_row, variant, overwrite=overwrite)
    return _manifest_row_from_completed_fit(
        {key: str(value) for key, value in request_row.items()}
    )


def refresh_fit_manifest(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
) -> Path:
    request_path = fit_requests_path_for_spec(fits_spec_path)
    if not request_path.exists():
        write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_rows(request_path)
    fit_rows = [
        _manifest_row_from_completed_fit(request_row) for request_row in request_rows
    ]
    fit_manifest_path = fit_manifest_path_for_spec(fits_spec_path)
    write_csv_rows(fit_manifest_path, fit_rows)
    write_fit_reports(fit_manifest_path)
    return fit_manifest_path


def run_fits(
    manifest_path: str | Path,
    fits_spec_path: str | Path,
    overwrite: bool = False,
) -> Path:
    request_path = write_fit_requests(manifest_path, fits_spec_path)
    request_rows = read_csv_rows(request_path)
    print(f"Loaded {len(request_rows)} fit request(s) from {request_path}.")
    for request_row in request_rows:
        experiment_name = request_row.get(
            "experiment_name", request_row["experiment_slug"]
        )
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
        default="data/configs/quickstart_fits_spec.yaml",
        help="Path to the fits YAML spec defining optimizer variants (default: data/configs/quickstart_fits_spec.yaml).",
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
        generation_rows = read_csv_rows(args.manifest_path)
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
