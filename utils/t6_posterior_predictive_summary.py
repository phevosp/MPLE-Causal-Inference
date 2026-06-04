"""Posterior predictive run summary and manifest row building."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from utils.t0_path_utils import io_path
from utils.t6_posterior_predictive_manifest import (
    POSTERIOR_PREDICTIVE_ROOT_NAME,
    POSTERIOR_PREDICTIVE_MANIFEST_NAME,
)


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
    row = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": str(Path(str(experiment_row["experiment_path"])).resolve()),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "N": panel_context["N"],
        "T": panel_context["T"],
        "s": int(summary.get("s", panel_context.get("s", 0))),
        "run_name": run_spec["name"],
        "run_slug": run_spec["slug"],
        "source_type": target["source_type"],
        "source_name": target["source_name"],
        "source_slug": target["source_slug"],
        "target_intervention_source": target["intervention_source"],
        "target_intervention_name": target["intervention_name"],
        "target_intervention_slug": target["intervention_slug"],
        "latent_rank": int(latent_rank),
        "num_samples": int(num_samples),
        "gibbs_sweeps": int(gibbs_sweeps),
        "seed": int(seed),
        "output_path": str(Path(output_root).resolve()),
    }
    if str(target["intervention_source"]) == "observed_experiment":
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


def manifest_row_from_metadata(
    experiment_row: dict[str, str],
    metadata: dict[str, Any],
    output_root: str | Path,
) -> dict[str, object]:
    summary = dict(metadata.get("summary", {}) or {})
    target_intervention_source = str(metadata.get("intervention_source", "")).strip()
    row = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": str(Path(str(experiment_row["experiment_path"])).resolve()),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "N": metadata.get("num_units", experiment_row.get("N", "")),
        "T": metadata.get("num_time_steps", experiment_row.get("T", "")),
        "s": metadata.get("s", ""),
        "run_name": metadata.get("run_name", ""),
        "run_slug": metadata.get("run_slug", ""),
        "source_type": metadata.get("source_type", ""),
        "source_name": metadata.get("source_name", ""),
        "source_slug": metadata.get("source_slug", ""),
        "target_intervention_source": target_intervention_source,
        "target_intervention_name": metadata.get("intervention_name", ""),
        "target_intervention_slug": metadata.get("intervention_slug", ""),
        "latent_rank": metadata.get("latent_rank", ""),
        "num_samples": metadata.get("num_samples", ""),
        "gibbs_sweeps": metadata.get("gibbs_sweeps", ""),
        "seed": metadata.get("seed", ""),
        "output_path": str(Path(output_root).resolve()),
    }
    if target_intervention_source == "observed_experiment":
        row.update(
            {
                "mean_abs_zscore": summary.get("mean_abs_zscore", ""),
                "max_abs_zscore": summary.get("max_abs_zscore", ""),
                "coverage_rate": summary.get("coverage_rate", ""),
                "num_statistics": summary.get("num_statistics", ""),
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
