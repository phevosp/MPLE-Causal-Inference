"""Shared fit config/materialization helpers used by fit-related entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.t0_path_utils import io_path
from utils.t5_experiment_context import infer_panel_dimensions
from utils.t6_pipeline_spec_utils import validate_fit_variant_dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def path_text(path: str | Path) -> str:
    candidate = Path(path)
    if os.name == "nt":
        return str(candidate.resolve())
    if candidate.is_absolute():
        return str(candidate)
    return os.path.abspath(os.fspath(candidate))


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
            "fixed_scalar_params": dict(
                estimation.get("fixed_scalar_params", {}) or {}
            ),
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
        "model_artifact_dir": path_text(experiment_path),
        "truth_artifact_dir": path_text(experiment_path),
        "panel_path": path_text(experiment_path / "panel_data.npz"),
        "x0_path": path_text(experiment_path / "x_0.npy"),
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
        "fits_spec_path": (
            path_text(variant["_spec_path"]) if variant.get("_spec_path") else ""
        ),
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get(
            "descriptor", experiment_row.get("experiment_name", "")
        ),
        "experiment_path": path_text(experiment_root),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "field_mode": experiment_row.get("field_mode", ""),
        "model_artifact_dir": path_text(experiment_root),
        "truth_artifact_dir": path_text(experiment_root),
        "panel_path": path_text(experiment_root / "panel_data.npz"),
        "x0_path": path_text(experiment_root / "x_0.npy"),
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
