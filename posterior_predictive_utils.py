"""Utilities for posterior-predictive outcome simulation and summary statistics."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

from data.synthetic_data_generation import (
    derive_pre_intervention_steps,
    simulate_outcomes_given_fixed_interventions,
)
from model_utils import (
    compose_interaction_matrix,
    interaction_effect,
    load_model_artifacts,
)


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
    zeta: float
    psi: float
    fit_intervention_model: bool
    latent_rank: int
    t_steps: int
    field_matrix: np.ndarray
    gamma_matrix: object


def first_existing_path(*paths: str | Path) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of the expected paths: "
        + ", ".join(str(Path(path)) for path in paths)
    )


def load_yaml_config(path: str | Path):
    return OmegaConf.load(Path(path))


def load_gamma_matrix(data_folder: str | Path):
    data_path = Path(data_folder)
    gamma_sparse = data_path / "gamma_matrix_sparse.npz"
    gamma_dense = data_path / "gamma_matrix.npy"
    if gamma_sparse.exists():
        return sparse.load_npz(gamma_sparse).tocsr()
    if gamma_dense.exists():
        return np.load(gamma_dense, allow_pickle=False)
    raise FileNotFoundError(f"Missing gamma matrix artifact in {data_path}.")


def load_experiment_panel_context(experiment_root: str | Path) -> dict[str, object]:
    experiment_path = Path(experiment_root)
    panel_path = experiment_path / "panel_data.npz"
    x0_path = experiment_path / "x_0.npy"
    z0_path = experiment_path / "z_0.npy"
    with np.load(panel_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    x_0 = np.asarray(np.load(x0_path), dtype=float)
    z_0 = np.asarray(np.load(z0_path), dtype=float)
    return {
        "x": x,
        "z": z,
        "x_0": x_0,
        "z_0": z_0,
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "s": derive_pre_intervention_steps(z),
    }


def save_estimated_parameter_bundle(
    path: str | Path,
    *,
    beta: float,
    xi: float,
    eta: float,
    zeta: float,
    psi: float,
    fit_intervention_model: bool,
    latent_rank: int,
    t_steps: int,
    field_matrix: np.ndarray,
) -> None:
    np.savez(
        Path(path),
        beta=np.asarray(float(beta)),
        xi=np.asarray(float(xi)),
        eta=np.asarray(float(eta)),
        zeta=np.asarray(float(zeta)),
        psi=np.asarray(float(psi)),
        fit_intervention_model=np.asarray(bool(fit_intervention_model)),
        latent_rank=np.asarray(int(latent_rank), dtype=int),
        t_steps=np.asarray(int(t_steps), dtype=int),
        field_matrix=np.asarray(field_matrix, dtype=float),
    )


def _load_scalar_estimates_from_summary(summary_path: Path) -> dict[str, float]:
    estimates: dict[str, float] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name in {"beta", "xi", "eta", "zeta", "psi"} and row.get("estimate"):
                estimates[name] = float(row["estimate"])
    return estimates


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
        zeta=float(getattr(config.estimation_params, "zeta", 0.0)),
        psi=float(getattr(config.estimation_params, "psi", 0.0)),
        fit_intervention_model=True,
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
    config = load_yaml_config(fit_path / "fit_realized_config.yaml")
    fit_intervention_model = bool(config.estimation_params.fit_intervention_model)
    gamma_matrix = load_gamma_matrix(experiment_path)

    if bundle_path.exists():
        with np.load(bundle_path, allow_pickle=False) as data:
            return OutcomeParameterBundle(
                source_type="fit",
                source_name=fit_path.name,
                beta=float(data["beta"]),
                xi=float(data["xi"]),
                eta=float(data["eta"]),
                zeta=float(data["zeta"]),
                psi=float(data["psi"]),
                fit_intervention_model=bool(data["fit_intervention_model"]),
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
        zeta=float(estimates.get("zeta", 0.0)),
        psi=float(estimates.get("psi", 0.0)),
        fit_intervention_model=fit_intervention_model,
        latent_rank=latent_rank,
        t_steps=t_steps,
        field_matrix=field_matrix,
        gamma_matrix=(
            gamma_matrix if gamma_matrix is not None else field_artifacts.gamma_matrix
        ),
    )


def simulate_outcomes_for_bundle(
    bundle: OutcomeParameterBundle,
    *,
    x_0: np.ndarray,
    z: np.ndarray,
    gibbs_sweeps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    interaction_matrix = compose_interaction_matrix(bundle.xi, bundle.gamma_matrix)
    return simulate_outcomes_given_fixed_interventions(
        x_0=np.asarray(x_0, dtype=float),
        z=np.asarray(z, dtype=float),
        field_matrix=np.asarray(bundle.field_matrix, dtype=float),
        interaction_matrix=interaction_matrix,
        beta=float(bundle.beta),
        eta=float(bundle.eta),
        rng=rng,
        gibbs_sweeps=int(gibbs_sweeps),
    )


def _graph_energy(x: np.ndarray, gamma_matrix) -> np.ndarray:
    interaction_x = interaction_effect(np.asarray(x, dtype=float), gamma_matrix)
    return np.sum(np.asarray(x, dtype=float) * interaction_x, axis=1) / x.shape[1]


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def compute_panel_statistics(
    x: np.ndarray,
    *,
    z: np.ndarray,
    x_0: np.ndarray,
    s: int,
    field_matrix: np.ndarray,
    gamma_matrix,
) -> dict[str, float | None]:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    field_matrix = np.asarray(field_matrix, dtype=float)
    prev_x = np.vstack([np.asarray(x_0, dtype=float), x[:-1, :]])
    graph_energy = _graph_energy(x, gamma_matrix)
    post_mask = slice(int(s), x.shape[0])
    pre_mask = slice(0, int(s))

    stats: dict[str, float | None] = {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": _mean_or_none(x[post_mask]),
        "intervention_alignment": float(np.mean(x * z)),
        "lag1_persistence": float(np.mean(x * prev_x)),
        "graph_interaction_energy": float(np.mean(graph_energy)),
        "field_alignment": float(np.mean(x * field_matrix)),
        "pre_intervention_alignment": _mean_or_none(
            (x[pre_mask] * z[pre_mask]).reshape(-1)
        ),
        "post_intervention_alignment": _mean_or_none(
            (x[post_mask] * z[post_mask]).reshape(-1)
        ),
        "pre_graph_interaction_energy": _mean_or_none(graph_energy[pre_mask]),
        "post_graph_interaction_energy": _mean_or_none(graph_energy[post_mask]),
    }
    return stats


def summarize_predictive_statistics(
    observed_stats: dict[str, float | None],
    simulated_stats: list[dict[str, float | None]],
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    abs_zscores: list[float] = []
    covered: list[float] = []
    for stat_name, observed_value in observed_stats.items():
        sample_values = np.asarray(
            [
                stat[stat_name]
                for stat in simulated_stats
                if stat.get(stat_name) is not None
            ],
            dtype=float,
        )
        if observed_value is None or sample_values.size == 0:
            continue
        sample_mean = float(np.mean(sample_values))
        sample_std = float(np.std(sample_values, ddof=0))
        if sample_std < 1e-12:
            if abs(float(observed_value) - sample_mean) < 1e-12:
                z_score = 0.0
            else:
                z_score = math.copysign(math.inf, float(observed_value) - sample_mean)
        else:
            z_score = (float(observed_value) - sample_mean) / sample_std
        q025, q500, q975 = np.quantile(sample_values, [0.025, 0.5, 0.975])
        left_tail = float(np.mean(sample_values <= float(observed_value)))
        right_tail = float(np.mean(sample_values >= float(observed_value)))
        tail_probability = min(1.0, 2.0 * min(left_tail, right_tail))
        in_interval = float(q025 <= float(observed_value) <= q975)
        rows.append(
            {
                "statistic": stat_name,
                "observed_value": float(observed_value),
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "z_score": z_score,
                "tail_probability": tail_probability,
                "q025": float(q025),
                "q500": float(q500),
                "q975": float(q975),
                "in_95_interval": bool(in_interval),
            }
        )
        abs_zscores.append(abs(z_score))
        covered.append(in_interval)

    summary = {
        "mean_abs_zscore": float(np.mean(abs_zscores)) if abs_zscores else math.inf,
        "max_abs_zscore": float(np.max(abs_zscores)) if abs_zscores else math.inf,
        "coverage_rate": float(np.mean(covered)) if covered else 0.0,
        "num_statistics": len(rows),
    }
    return rows, summary


def write_predictive_stats_tables(
    output_root: str | Path,
    stat_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    output_path = Path(output_root)
    csv_path = output_path / "posterior_predictive_stats.csv"
    md_path = output_path / "posterior_predictive_stats.md"
    fieldnames = [
        "statistic",
        "observed_value",
        "sample_mean",
        "sample_std",
        "z_score",
        "tail_probability",
        "q025",
        "q500",
        "q975",
        "in_95_interval",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stat_rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in stat_rows:
            formatted = []
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    formatted.append(f"{value:.6f}")
                else:
                    formatted.append(str(value))
            handle.write("| " + " | ".join(formatted) + " |\n")
    return csv_path, md_path
