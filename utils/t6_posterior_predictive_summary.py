"""Posterior predictive run summary and manifest row building."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.t6_posterior_predictive_manifest import (
    POSTERIOR_PREDICTIVE_MANIFEST_NAME,
    POSTERIOR_PREDICTIVE_ROOT_NAME,
)


def _build_shared_manifest_row(
    *,
    experiment_row: dict[str, str],
    n_units: object,
    t_steps: object,
    s: object,
    run_name: object,
    run_slug: object,
    source_type: object,
    source_name: object,
    source_slug: object,
    target_intervention_source: object,
    target_intervention_name: object,
    target_intervention_slug: object,
    latent_rank: object,
    num_samples: object,
    gibbs_sweeps: object,
    seed: object,
    output_root: str | Path,
) -> dict[str, object]:
    return {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": str(Path(str(experiment_row["experiment_path"])).resolve()),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "N": n_units,
        "T": t_steps,
        "s": s,
        "run_name": run_name,
        "run_slug": run_slug,
        "source_type": source_type,
        "source_name": source_name,
        "source_slug": source_slug,
        "target_intervention_source": target_intervention_source,
        "target_intervention_name": target_intervention_name,
        "target_intervention_slug": target_intervention_slug,
        "latent_rank": latent_rank,
        "num_samples": num_samples,
        "gibbs_sweeps": gibbs_sweeps,
        "seed": seed,
        "output_path": str(Path(output_root).resolve()),
    }


def _append_predictive_metrics(
    row: dict[str, object],
    *,
    target_intervention_source: str,
    summary: dict[str, object],
) -> dict[str, object]:
    if target_intervention_source == "observed_experiment":
        row.update(
            {
                "mean_abs_zscore": float(summary["mean_abs_zscore"]),
                "max_abs_zscore": float(summary["max_abs_zscore"]),
                "coverage_rate": float(summary["coverage_rate"]),
                "num_statistics": int(summary["num_statistics"]),
            }
        )
    else:
        row.update(
            {
                "mean_abs_zscore": "",
                "max_abs_zscore": "",
                "coverage_rate": "",
                "num_statistics": "",
            }
        )
    return row


def build_manifest_row(
    *,
    experiment_row: dict[str, str],
    panel_context: dict[str, Any],
    target: dict[str, object],
    run_spec: dict[str, Any],
    latent_rank: int,
    num_samples: int,
    gibbs_sweeps: int,
    seed: int,
    output_root: str | Path,
    summary: dict[str, object],
) -> dict[str, object]:
    target_intervention_source = str(target["intervention_source"])
    row = _build_shared_manifest_row(
        experiment_row=experiment_row,
        n_units=panel_context["N"],
        t_steps=panel_context["T"],
        s=int(summary.get("s", panel_context.get("s", 0))),
        run_name=run_spec["name"],
        run_slug=run_spec["slug"],
        source_type=target["source_type"],
        source_name=target["source_name"],
        source_slug=target["source_slug"],
        target_intervention_source=target_intervention_source,
        target_intervention_name=target["intervention_name"],
        target_intervention_slug=target["intervention_slug"],
        latent_rank=int(latent_rank),
        num_samples=int(num_samples),
        gibbs_sweeps=int(gibbs_sweeps),
        seed=int(seed),
        output_root=output_root,
    )
    return _append_predictive_metrics(
        row,
        target_intervention_source=target_intervention_source,
        summary=summary,
    )


def manifest_row_from_metadata(
    experiment_row: dict[str, str],
    metadata: dict[str, Any],
    output_root: str | Path,
) -> dict[str, object]:
    summary = dict(metadata.get("summary", {}) or {})
    target_intervention_source = str(metadata.get("intervention_source", "")).strip()
    row = _build_shared_manifest_row(
        experiment_row=experiment_row,
        n_units=metadata.get("num_units", experiment_row.get("N", "")),
        t_steps=metadata.get("num_time_steps", experiment_row.get("T", "")),
        s=metadata.get("s", ""),
        run_name=metadata.get("run_name", ""),
        run_slug=metadata.get("run_slug", ""),
        source_type=metadata.get("source_type", ""),
        source_name=metadata.get("source_name", ""),
        source_slug=metadata.get("source_slug", ""),
        target_intervention_source=target_intervention_source,
        target_intervention_name=metadata.get("intervention_name", ""),
        target_intervention_slug=metadata.get("intervention_slug", ""),
        latent_rank=metadata.get("latent_rank", ""),
        num_samples=metadata.get("num_samples", ""),
        gibbs_sweeps=metadata.get("gibbs_sweeps", ""),
        seed=metadata.get("seed", ""),
        output_root=output_root,
    )
    return _append_predictive_metrics(
        row,
        target_intervention_source=target_intervention_source,
        summary=summary,
    )
