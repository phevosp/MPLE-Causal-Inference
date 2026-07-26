"""Parameter bundle dataclass and persistence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.t0_path_utils import first_existing_path, io_path, path_exists
from utils.t0_config_utils import load_yaml_config
from utils.t1_matrix_io import load_gamma_matrix
from utils.t3_model_artifacts import (
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
    TreatmentFieldArtifacts,
    load_model_artifacts,
    load_treatment_field_artifacts,
)


GENERATION_CONFIG_FILENAMES = (
    "generation_realized_config.yaml",
    "realized_config.yaml",
)


@dataclass(frozen=True)
class OutcomeParameterBundle:
    """Estimated or truth parameters plus fit-time beta-gradient metadata.

    The beta-mask flags are preserved for optimizers that support beta-gradient-only
    masking. Reported losses and predictive evaluation use the realized intervention
    panel without changing the forward model.
    """
    source_type: str
    source_name: str
    beta: float
    xi: float
    eta: float
    beta_mask_pre_s: bool
    beta_mask_post_e: bool
    latent_rank: int
    t_steps: int
    field_matrix: np.ndarray
    gamma_matrix: object


def save_estimated_parameter_bundle(
    path: str | Path,
    *,
    beta: float,
    xi: float,
    eta: float,
    latent_rank: int,
    t_steps: int,
    field_matrix: np.ndarray,
) -> None:
    np.savez(
        io_path(path),
        beta=np.asarray(float(beta)),
        xi=np.asarray(float(xi)),
        eta=np.asarray(float(eta)),
        latent_rank=np.asarray(int(latent_rank), dtype=int),
        t_steps=np.asarray(int(t_steps), dtype=int),
        field_matrix=np.asarray(field_matrix, dtype=float),
    )


def _load_scalar_estimates_from_summary(summary_path: Path) -> dict[str, float]:
    estimates: dict[str, float] = {}
    with open(io_path(summary_path), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name in {"beta", "xi", "eta"} and row.get("estimate"):
                estimates[name] = float(row["estimate"])
    return estimates


def _fit_beta_mask_flag(fit_root: Path, key: str) -> bool:
    config_path = fit_root / "fit_realized_config.yaml"
    if not path_exists(config_path):
        return False
    config = load_yaml_config(config_path)
    estimation_params = getattr(config, "estimation_params", None)
    if estimation_params is None:
        return False
    return bool(estimation_params.get(str(key), False))


def _fit_optimizer_mode(fit_root: Path) -> str | None:
    config_path = fit_root / "fit_realized_config.yaml"
    if not path_exists(config_path):
        return None
    config = load_yaml_config(config_path)
    global_params = getattr(config, "global_params", None)
    if global_params is None:
        return None
    optimizer_mode = global_params.get("optimizer_mode", None)
    if optimizer_mode in (None, ""):
        return None
    return str(optimizer_mode)


def load_fit_treatment_field_artifacts(
    fit_root: str | Path,
) -> TreatmentFieldArtifacts | None:
    fit_path = Path(fit_root)
    artifact_path = fit_path / "estimated_treatment_field_artifacts.npz"
    optimizer_mode = _fit_optimizer_mode(fit_path)
    treatment_modes = {
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
        OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
    }
    if not path_exists(artifact_path):
        if optimizer_mode in treatment_modes:
            raise FileNotFoundError(
                f"Fit {fit_path} uses optimizer_mode={optimizer_mode!r} but is missing "
                f"{artifact_path.name}."
            )
        return None
    artifacts = load_treatment_field_artifacts(artifact_path)
    if optimizer_mode is not None and str(artifacts.optimizer_mode) != optimizer_mode:
        raise ValueError(
            f"Treatment field artifact optimizer_mode={artifacts.optimizer_mode!r} does "
            f"not match fit optimizer_mode={optimizer_mode!r} for {fit_path}."
        )
    return artifacts


def _build_outcome_parameter_bundle(
    *,
    source_type: str,
    source_name: str,
    beta: float,
    xi: float,
    eta: float,
    beta_mask_pre_s: bool,
    beta_mask_post_e: bool,
    latent_rank: int,
    t_steps: int,
    field_matrix: np.ndarray,
    gamma_matrix: object,
) -> OutcomeParameterBundle:
    return OutcomeParameterBundle(
        source_type=str(source_type),
        source_name=str(source_name),
        beta=float(beta),
        xi=float(xi),
        eta=float(eta),
        beta_mask_pre_s=bool(beta_mask_pre_s),
        beta_mask_post_e=bool(beta_mask_post_e),
        latent_rank=int(latent_rank),
        t_steps=int(t_steps),
        field_matrix=np.asarray(field_matrix, dtype=float),
        gamma_matrix=gamma_matrix,
    )


def load_truth_parameter_bundle(experiment_root: str | Path) -> OutcomeParameterBundle:
    experiment_path = Path(experiment_root)
    config_path = first_existing_path(
        *(experiment_path / name for name in GENERATION_CONFIG_FILENAMES)
    )
    config = load_yaml_config(config_path)
    artifacts = load_model_artifacts(experiment_path)
    if artifacts.field_matrix is None:
        raise ValueError(f"Missing truth field matrix in {experiment_path}.")
    return _build_outcome_parameter_bundle(
        source_type="truth",
        source_name="truth",
        beta=config.estimation_params.beta,
        xi=config.estimation_params.xi,
        eta=config.estimation_params.eta,
        beta_mask_pre_s=False,
        beta_mask_post_e=False,
        latent_rank=artifacts.latent_rank,
        t_steps=artifacts.t_steps,
        field_matrix=artifacts.field_matrix,
        gamma_matrix=artifacts.gamma_matrix,
    )


def load_fit_parameter_bundle(
    fit_root: str | Path,
    experiment_root: str | Path,
) -> OutcomeParameterBundle:
    fit_path = Path(fit_root)
    experiment_path = Path(experiment_root)
    bundle_path = fit_path / "estimated_parameter_bundle.npz"
    gamma_matrix = load_gamma_matrix(experiment_path)
    beta_mask_pre_s = _fit_beta_mask_flag(fit_path, "beta_mask_pre_s")
    beta_mask_post_e = _fit_beta_mask_flag(fit_path, "beta_mask_post_e")

    if path_exists(bundle_path):
        with np.load(io_path(bundle_path), allow_pickle=False) as data:
            return _build_outcome_parameter_bundle(
                source_type="fit",
                source_name=fit_path.name,
                beta=data["beta"],
                xi=data["xi"],
                eta=data["eta"],
                beta_mask_pre_s=beta_mask_pre_s,
                beta_mask_post_e=beta_mask_post_e,
                latent_rank=data["latent_rank"],
                t_steps=data["t_steps"],
                field_matrix=data["field_matrix"],
                gamma_matrix=gamma_matrix,
            )

    summary_path = fit_path / "mple_summary.csv"
    estimates = _load_scalar_estimates_from_summary(summary_path)
    estimated_field_path = fit_path / "estimated_field_artifacts.npz"
    if not path_exists(estimated_field_path):
        raise FileNotFoundError(
            f"Missing estimated_parameter_bundle.npz and estimated_field_artifacts.npz in {fit_path}."
        )
    with np.load(io_path(estimated_field_path), allow_pickle=False) as data:
        field_matrix = np.asarray(data["field_matrix"], dtype=float)
        latent_rank = int(data["latent_rank"])
        t_steps = int(data["t_steps"])
    return _build_outcome_parameter_bundle(
        source_type="fit",
        source_name=fit_path.name,
        beta=estimates["beta"],
        xi=estimates["xi"],
        eta=estimates["eta"],
        beta_mask_pre_s=beta_mask_pre_s,
        beta_mask_post_e=beta_mask_post_e,
        latent_rank=latent_rank,
        t_steps=t_steps,
        field_matrix=field_matrix,
        gamma_matrix=gamma_matrix,
    )
