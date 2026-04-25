"""Artifact loading and parameter-bundle persistence helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from io_utils import first_existing_path, io_path, load_gamma_matrix, load_yaml_config
from model_utils import load_model_artifacts


GENERATION_CONFIG_FILENAMES = (
    "generation_realized_config.yaml",
    "realized_config.yaml",
)


@dataclass(frozen=True)
class OutcomeParameterBundle:
    source_type: str
    source_name: str
    beta: float
    xi: float
    eta: float
    beta_mask_pre_s: bool
    latent_rank: int
    t_steps: int
    field_matrix: np.ndarray
    gamma_matrix: object


def load_panel_context_from_artifacts(
    panel_path: str | Path,
    x0_path: str | Path,
    z0_path: str | Path,
) -> dict[str, object]:
    with np.load(Path(panel_path), allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    x_0 = np.asarray(np.load(Path(x0_path)), dtype=float)
    z_0 = np.asarray(np.load(Path(z0_path)), dtype=float)

    from data.synthetic_data_generation import derive_pre_intervention_steps

    return {
        "x": x,
        "z": z,
        "x_0": x_0,
        "z_0": z_0,
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "s": derive_pre_intervention_steps(z),
    }


def load_experiment_panel_context(experiment_root: str | Path) -> dict[str, object]:
    experiment_path = Path(experiment_root)
    return load_panel_context_from_artifacts(
        experiment_path / "panel_data.npz",
        experiment_path / "x_0.npy",
        experiment_path / "z_0.npy",
    )


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
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name in {"beta", "xi", "eta"} and row.get("estimate"):
                estimates[name] = float(row["estimate"])
    return estimates


def _fit_beta_mask_pre_s(fit_root: Path) -> bool:
    config_path = fit_root / "fit_realized_config.yaml"
    if not config_path.exists():
        return False
    config = load_yaml_config(config_path)
    estimation_params = getattr(config, "estimation_params", None)
    if estimation_params is None:
        return False
    return bool(estimation_params.get("beta_mask_pre_s", False))


def load_truth_parameter_bundle(experiment_root: str | Path) -> OutcomeParameterBundle:
    experiment_path = Path(experiment_root)
    config_path = first_existing_path(
        *(experiment_path / name for name in GENERATION_CONFIG_FILENAMES)
    )
    config = load_yaml_config(config_path)
    artifacts = load_model_artifacts(experiment_path)
    if artifacts.field_matrix is None:
        raise ValueError(f"Missing truth field matrix in {experiment_path}.")
    return OutcomeParameterBundle(
        source_type="truth",
        source_name="truth",
        beta=float(config.estimation_params.beta),
        xi=float(config.estimation_params.xi),
        eta=float(config.estimation_params.eta),
        beta_mask_pre_s=False,
        latent_rank=int(artifacts.latent_rank),
        t_steps=int(artifacts.t_steps),
        field_matrix=np.asarray(artifacts.field_matrix, dtype=float),
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
    beta_mask_pre_s = _fit_beta_mask_pre_s(fit_path)

    if bundle_path.exists():
        with np.load(bundle_path, allow_pickle=False) as data:
            return OutcomeParameterBundle(
                source_type="fit",
                source_name=fit_path.name,
                beta=float(data["beta"]),
                xi=float(data["xi"]),
                eta=float(data["eta"]),
                beta_mask_pre_s=beta_mask_pre_s,
                latent_rank=int(data["latent_rank"]),
                t_steps=int(data["t_steps"]),
                field_matrix=np.asarray(data["field_matrix"], dtype=float),
                gamma_matrix=gamma_matrix,
            )

    summary_path = fit_path / "mple_summary.csv"
    estimates = _load_scalar_estimates_from_summary(summary_path)
    field_artifacts = load_model_artifacts(experiment_path)
    estimated_field_path = fit_path / "estimated_field_artifacts.npz"
    if not estimated_field_path.exists():
        raise FileNotFoundError(
            f"Missing estimated_parameter_bundle.npz and estimated_field_artifacts.npz in {fit_path}."
        )
    with np.load(estimated_field_path, allow_pickle=False) as data:
        field_matrix = np.asarray(data["field_matrix"], dtype=float)
        latent_rank = int(data["latent_rank"])
        t_steps = int(data["t_steps"])
    return OutcomeParameterBundle(
        source_type="fit",
        source_name=fit_path.name,
        beta=float(estimates["beta"]),
        xi=float(estimates["xi"]),
        eta=float(estimates["eta"]),
        beta_mask_pre_s=beta_mask_pre_s,
        latent_rank=latent_rank,
        t_steps=t_steps,
        field_matrix=field_matrix,
        gamma_matrix=gamma_matrix if gamma_matrix is not None else field_artifacts.gamma_matrix,
    )
