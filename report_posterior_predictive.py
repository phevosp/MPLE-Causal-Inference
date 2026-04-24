"""Refresh posterior-predictive manifests from outputs and write grouped summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from io_utils import _as_float, _fmt, _metric_or_inf, write_csv, write_markdown_table
from io_utils import io_path
from pipeline_specs import read_csv_manifest, write_csv_manifest
from posterior_predictive_job_utils import (
    POSTERIOR_PREDICTIVE_MANIFEST_NAME,
    manifest_row_from_metadata,
)


PER_EXPERIMENT_COLUMNS = [
    "experiment_name",
    "descriptor",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "latent_rank",
    "rank_in_experiment",
    "is_best",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "output_path",
]
WINNER_COLUMNS = [
    "experiment_name",
    "descriptor",
    "intervention_source",
    "graph_source",
    "N",
    "T",
    "s",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "latent_rank",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "output_path",
]
MANIFEST_COLUMNS = [
    "experiment_name",
    "experiment_slug",
    "descriptor",
    "experiment_path",
    "intervention_source",
    "graph_source",
    "N",
    "T",
    "s",
    "run_name",
    "run_slug",
    "source_type",
    "source_name",
    "source_slug",
    "target_intervention_source",
    "target_intervention_name",
    "target_intervention_slug",
    "latent_rank",
    "mean_abs_zscore",
    "max_abs_zscore",
    "coverage_rate",
    "num_statistics",
    "num_samples",
    "gibbs_sweeps",
    "seed",
    "output_path",
]


def collect_predictive_rows(manifest_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in read_csv_manifest(manifest_path):
        target_intervention_source = (
            str(
                manifest_row.get("target_intervention_source", "observed_experiment")
            ).strip()
            or "observed_experiment"
        )
        if target_intervention_source != "observed_experiment":
            continue
        row: dict[str, object] = dict(manifest_row)
        for key in [
            "mean_abs_zscore",
            "max_abs_zscore",
            "coverage_rate",
        ]:
            row[key] = _as_float(manifest_row.get(key))
        for key in [
            "N",
            "T",
            "s",
            "latent_rank",
            "num_statistics",
            "num_samples",
            "gibbs_sweeps",
        ]:
            value = manifest_row.get(key)
            row[key] = int(value) if value not in (None, "") else ""
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("run_name", "")),
            str(row.get("source_name", "")),
        )
    )
    return rows


def collect_manifest_rows_from_outputs(
    generation_manifest_path: str | Path,
) -> tuple[Path, list[dict[str, object]]]:
    manifest_root = Path(generation_manifest_path).resolve().parent
    rows: list[dict[str, object]] = []
    for experiment_row in read_csv_manifest(generation_manifest_path):
        experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
        observed_metadata_paths = experiment_root.glob(
            "posterior_predictive/*/*/posterior_predictive_metadata.yaml"
        )
        counterfactual_metadata_paths = experiment_root.glob(
            "counterfactual/*/*/*/counterfactual_metadata.yaml"
        )
        for metadata_path in list(observed_metadata_paths) + list(counterfactual_metadata_paths):
            with open(io_path(metadata_path), "r", encoding="utf-8") as handle:
                metadata = OmegaConf.to_container(OmegaConf.load(handle), resolve=True)
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata file {metadata_path} did not contain a mapping.")
            rows.append(
                manifest_row_from_metadata(
                    experiment_row,
                    metadata,
                    metadata_path.parent,
                )
            )
    rows.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("target_intervention_source", "")),
            str(row.get("target_intervention_name", "")),
            str(row.get("run_name", "")),
            str(row.get("source_name", "")),
        )
    )
    manifest_path = manifest_root / POSTERIOR_PREDICTIVE_MANIFEST_NAME
    write_csv_manifest(
        manifest_path,
        [{column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in rows],
    )
    return manifest_path, rows


def ranking_key(row: dict[str, object]) -> tuple[float, float, str, str]:
    return (
        _metric_or_inf(row.get("mean_abs_zscore")),
        _metric_or_inf(row.get("max_abs_zscore")),
        str(row.get("run_name", "")),
        str(row.get("source_name", "")),
    )


def rank_rows_within_experiment(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=ranking_key)
    ranked_rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        ranked = dict(row)
        ranked["rank_in_experiment"] = index
        ranked["is_best"] = index == 1
        ranked_rows.append(ranked)
    return ranked_rows


def group_and_rank_predictive_rows(
    rows: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_path"])].append(row)

    ranked_groups: dict[str, list[dict[str, object]]] = {}
    winners: list[dict[str, object]] = []
    for experiment_path, group_rows in grouped.items():
        ranked = rank_rows_within_experiment(group_rows)
        ranked_groups[experiment_path] = ranked
        winners.append(dict(ranked[0]))
    winners.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("source_name", "")),
        )
    )
    return ranked_groups, winners


def write_per_experiment_summary(
    experiment_path: str | Path,
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    experiment_root = Path(experiment_path)
    csv_path = experiment_root / "posterior_predictive_summary.csv"
    md_path = experiment_root / "posterior_predictive_summary.md"
    write_csv(csv_path, rows, PER_EXPERIMENT_COLUMNS)
    best_row = next(row for row in rows if row.get("is_best"))
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Posterior Predictive Summary: "
            f"{best_row.get('experiment_name', experiment_root.name)}\n\n"
        )
        handle.write(
            f"- Best source: `{best_row.get('source_name', '')}`\n"
            f"- Best run: `{best_row.get('run_name', '')}`\n"
            f"- Mean absolute z-score: `{_fmt(best_row.get('mean_abs_zscore'))}`\n\n"
        )
        write_markdown_table(handle, rows, PER_EXPERIMENT_COLUMNS)
    return csv_path, md_path


def write_cross_experiment_summary(
    manifest_path: str | Path,
    winner_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    manifest_root = Path(manifest_path).resolve().parent
    csv_path = manifest_root / "best_posterior_predictive_by_experiment.csv"
    md_path = manifest_root / "best_posterior_predictive_by_experiment.md"
    write_csv(csv_path, winner_rows, WINNER_COLUMNS)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Best Posterior Predictive Source By Experiment\n\n")
        handle.write(
            "Each row is the top-ranked posterior-predictive source within one generated experiment.\n\n"
        )
        write_markdown_table(handle, winner_rows, WINNER_COLUMNS)
    return csv_path, md_path


def write_posterior_predictive_reports(manifest_path: str | Path) -> dict[str, object]:
    rows = collect_predictive_rows(manifest_path)
    if not rows:
        raise ValueError(
            f"No posterior-predictive runs were found in manifest {manifest_path}."
        )

    ranked_groups, winners = group_and_rank_predictive_rows(rows)
    per_experiment_outputs: dict[str, dict[str, str]] = {}
    for experiment_path, ranked_rows in ranked_groups.items():
        csv_path, md_path = write_per_experiment_summary(experiment_path, ranked_rows)
        per_experiment_outputs[experiment_path] = {
            "csv": str(csv_path),
            "md": str(md_path),
        }
    winners_csv, winners_md = write_cross_experiment_summary(manifest_path, winners)
    return {
        "per_experiment": per_experiment_outputs,
        "winners_csv": str(winners_csv),
        "winners_md": str(winners_md),
    }


def refresh_and_write_posterior_predictive_reports(
    generation_manifest_path: str | Path,
) -> dict[str, object]:
    manifest_path, all_rows = collect_manifest_rows_from_outputs(generation_manifest_path)
    outputs = {
        "manifest_path": str(manifest_path),
        "num_manifest_rows": len(all_rows),
    }
    predictive_rows = collect_predictive_rows(manifest_path)
    if predictive_rows:
        outputs.update(write_posterior_predictive_reports(manifest_path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh posterior-predictive manifest rows from outputs and write grouped summaries."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    args = parser.parse_args()
    outputs = refresh_and_write_posterior_predictive_reports(
        args.generation_manifest_path
    )
    print(f"Wrote posterior predictive manifest: {outputs['manifest_path']}")


if __name__ == "__main__":
    main()
