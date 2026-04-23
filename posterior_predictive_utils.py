"""Utilities for posterior-predictive outcome simulation and summary statistics."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from data.synthetic_data_generation import (
    derive_pre_intervention_steps,
    simulate_outcomes_given_fixed_interventions,
)
from model_utils import (
    compose_interaction_matrix,
    interaction_effect,
    load_model_artifacts,
)
from pipeline_specs import slugify
from io_utils import first_existing_path, load_gamma_matrix, load_yaml_config


GENERATION_CONFIG_FILENAMES = (
    "generation_realized_config.yaml",
    "realized_config.yaml",
)
INTERVENTION_LIBRARY_ROOT_NAME = "intervention_library"
COUNTERFACTUAL_ROOT_NAME = "counterfactual"
COUNTERFACTUAL_MANIFEST_NAME = "counterfactual_manifest.csv"


@dataclass(frozen=True)
class OutcomeParameterBundle:
    source_type: str
    source_name: str
    beta: float
    xi: float
    eta: float
    latent_rank: int
    t_steps: int
    field_matrix: np.ndarray
    gamma_matrix: object


@dataclass(frozen=True)
class InterventionContext:
    source_kind: str
    intervention_name: str
    intervention_slug: str
    z: np.ndarray
    z_0: np.ndarray
    s: int
    metadata: dict[str, object]


def _io_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


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


def _validate_intervention_panel(z: np.ndarray, z_0: np.ndarray) -> None:
    if z.ndim != 2:
        raise ValueError("Intervention panel z must be 2D.")
    if z_0.shape != (z.shape[1],):
        raise ValueError(
            f"Intervention z_0 shape {z_0.shape} does not match panel width {z.shape[1]}."
        )
    if not np.all(np.isin(z, (-1.0, 1.0))):
        raise ValueError("Intervention panel z must use -1/+1 coding only.")
    if not np.all(np.isin(z_0, (-1.0, 0.0, 1.0))):
        raise ValueError("Intervention z_0 must use legacy 0 or -1/+1 coding only.")


def save_intervention_artifact(
    output_root: str | Path,
    *,
    intervention_name: str,
    experiment_name: str,
    z: np.ndarray,
    z_0: np.ndarray,
    s: int,
    source_kind: str,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    artifact_root = Path(output_root)
    z = np.asarray(z, dtype=float)
    z_0 = np.asarray(z_0, dtype=float)
    _validate_intervention_panel(z, z_0)
    artifact_root.mkdir(parents=True, exist_ok=False)
    np.savez(artifact_root / "intervention_panel.npz", z=z)
    np.save(artifact_root / "z_0.npy", z_0)
    metadata = {
        "intervention_name": intervention_name,
        "intervention_slug": slugify(intervention_name),
        "experiment_name": experiment_name,
        "N": int(z.shape[1]),
        "T": int(z.shape[0]),
        "s": int(s),
        "source_kind": source_kind,
        **dict(extra_metadata or {}),
    }
    with open(_io_path(artifact_root / "intervention_metadata.yaml"), "w", encoding="utf-8") as handle:
        OmegaConf.save(OmegaConf.create(metadata), handle)
    return artifact_root


def build_full_on_intervention(
    n_nodes: int,
    t_steps: int,
    s: int,
    *,
    activation_scope: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    z = -np.ones((int(t_steps), int(n_nodes)), dtype=float)
    z_0 = -np.ones(int(n_nodes), dtype=float)
    scope = str(activation_scope)
    if scope == "all_time":
        z[:, :] = 1.0
        z_0[:] = 1.0
    elif scope == "no_time":
        pass
    elif scope == "from_s":
        z[int(s) :, :] = 1.0
    else:
        raise ValueError(f"Unsupported full_on activation_scope '{activation_scope}'.")
    return z, z_0, derive_pre_intervention_steps(z)


def build_single_unit_on_intervention(
    n_nodes: int,
    t_steps: int,
    s: int,
    *,
    unit_index: int,
    activation_scope: str,
    start_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if int(unit_index) < 0 or int(unit_index) >= int(n_nodes):
        raise ValueError(
            f"unit_index={unit_index} is out of bounds for N={n_nodes}."
        )
    z = -np.ones((int(t_steps), int(n_nodes)), dtype=float)
    z_0 = -np.ones(int(n_nodes), dtype=float)
    scope = str(activation_scope)
    if scope == "all_time":
        z[:, int(unit_index)] = 1.0
        z_0[int(unit_index)] = 1.0
    elif scope == "no_time":
        pass
    elif scope == "from_s":
        z[int(s) :, int(unit_index)] = 1.0
    elif scope == "from_step":
        if start_step is None:
            raise ValueError("start_step is required when activation_scope='from_step'.")
        if int(start_step) < 0 or int(start_step) > int(t_steps):
            raise ValueError(
                f"start_step={start_step} must lie in [0, {t_steps}]."
            )
        z[int(start_step) :, int(unit_index)] = 1.0
    else:
        raise ValueError(
            f"Unsupported single_unit_on activation_scope '{activation_scope}'."
        )
    return z, z_0, derive_pre_intervention_steps(z)


def load_saved_intervention_context(
    experiment_root: str | Path,
    intervention_name: str,
) -> InterventionContext:
    experiment_path = Path(experiment_root)
    intervention_slug = slugify(intervention_name)
    artifact_root = (
        experiment_path / INTERVENTION_LIBRARY_ROOT_NAME / intervention_slug
    )
    panel_path = artifact_root / "intervention_panel.npz"
    z0_path = artifact_root / "z_0.npy"
    metadata_path = artifact_root / "intervention_metadata.yaml"
    if not panel_path.exists() or not z0_path.exists():
        raise FileNotFoundError(
            f"Saved intervention artifact '{intervention_name}' not found under {artifact_root}."
        )
    with np.load(panel_path, allow_pickle=False) as data:
        if "z" not in data:
            raise KeyError(f"Intervention artifact {panel_path} does not contain 'z'.")
        z = np.asarray(data["z"], dtype=float)
    z_0 = np.asarray(np.load(z0_path), dtype=float)
    _validate_intervention_panel(z, z_0)
    metadata = {}
    if metadata_path.exists():
        with open(_io_path(metadata_path), "r", encoding="utf-8") as handle:
            metadata = OmegaConf.to_container(OmegaConf.load(handle), resolve=True)
    if not isinstance(metadata, dict):
        metadata = {}
    return InterventionContext(
        source_kind="saved_intervention",
        intervention_name=str(metadata.get("intervention_name", intervention_name)),
        intervention_slug=str(metadata.get("intervention_slug", intervention_slug)),
        z=z,
        z_0=z_0,
        s=int(metadata.get("s", derive_pre_intervention_steps(z))),
        metadata=metadata,
    )


def resolve_intervention_context(
    experiment_root: str | Path,
    *,
    intervention_source: str,
    intervention_name: str | None = None,
    panel_context: dict[str, object] | None = None,
) -> InterventionContext:
    source = str(intervention_source).strip().lower()
    if source == "observed_experiment":
        resolved_panel_context = (
            panel_context
            if panel_context is not None
            else load_experiment_panel_context(experiment_root)
        )
        return InterventionContext(
            source_kind="observed_experiment",
            intervention_name="observed_experiment",
            intervention_slug="observed_experiment",
            z=np.asarray(resolved_panel_context["z"], dtype=float),
            z_0=np.asarray(resolved_panel_context["z_0"], dtype=float),
            s=int(resolved_panel_context["s"]),
            metadata={"source_kind": "observed_experiment"},
        )
    if source == "saved_intervention":
        if not intervention_name or not str(intervention_name).strip():
            raise ValueError(
                "saved_intervention targets must provide intervention_name."
            )
        context = load_saved_intervention_context(experiment_root, intervention_name)
        if panel_context is not None:
            if context.z.shape != (
                int(panel_context["T"]),
                int(panel_context["N"]),
            ):
                raise ValueError(
                    f"Saved intervention '{intervention_name}' has shape {context.z.shape},"
                    f" expected {(int(panel_context['T']), int(panel_context['N']))}."
                )
        return context
    raise ValueError(f"Unsupported intervention_source '{intervention_source}'.")


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
        _io_path(path),
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

    if bundle_path.exists():
        with np.load(bundle_path, allow_pickle=False) as data:
            return OutcomeParameterBundle(
                source_type="fit",
                source_name=fit_path.name,
                beta=float(data["beta"]),
                xi=float(data["xi"]),
                eta=float(data["eta"]),
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


def compute_counterfactual_sample_summary(
    x: np.ndarray,
    *,
    s: int,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=float)
    post_value = float(np.mean(x[int(s) :, :])) if int(s) < x.shape[0] else math.nan
    return {
        "overall_mean_magnetization": float(np.mean(x)),
        "post_intervention_mean_magnetization": post_value,
        "unit_mean_magnetization": np.mean(x, axis=0),
    }


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


def _finite_summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "sample_mean": "",
            "sample_std": "",
            "q025": "",
            "q500": "",
            "q975": "",
            "num_finite_samples": 0,
        }
    q025, q500, q975 = np.quantile(finite, [0.025, 0.5, 0.975])
    return {
        "sample_mean": float(np.mean(finite)),
        "sample_std": float(np.std(finite, ddof=0)),
        "q025": float(q025),
        "q500": float(q500),
        "q975": float(q975),
        "num_finite_samples": int(finite.size),
    }


def write_counterfactual_summary_tables(
    output_root: str | Path,
    *,
    sample_summaries: dict[str, np.ndarray],
) -> tuple[Path, Path, Path]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_npz_path = output_path / "counterfactual_sample_summaries.npz"
    summary_csv_path = output_path / "counterfactual_summary.csv"
    unit_csv_path = output_path / "counterfactual_unit_summary.csv"
    np.savez(_io_path(sample_npz_path), **sample_summaries)

    summary_rows = []
    for key in [
        "overall_mean_magnetization",
        "post_intervention_mean_magnetization",
    ]:
        row = {"statistic": key}
        row.update(_finite_summary(np.asarray(sample_summaries[key], dtype=float)))
        summary_rows.append(row)
    with open(_io_path(summary_csv_path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "statistic",
                "sample_mean",
                "sample_std",
                "q025",
                "q500",
                "q975",
                "num_finite_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    unit_values = np.asarray(sample_summaries["unit_mean_magnetization"], dtype=float)
    unit_rows = []
    for unit_index in range(unit_values.shape[1]):
        row = {"unit_index": int(unit_index)}
        row.update(_finite_summary(unit_values[:, unit_index]))
        unit_rows.append(row)
    with open(_io_path(unit_csv_path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_index",
                "sample_mean",
                "sample_std",
                "q025",
                "q500",
                "q975",
                "num_finite_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(unit_rows)
    return sample_npz_path, summary_csv_path, unit_csv_path
