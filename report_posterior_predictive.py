"""Aggregate posterior-predictive simulation results into grouped summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from pipeline_specs import read_csv_manifest


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


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _metric_or_inf(value: object) -> float:
    parsed = _as_float(value)
    return math.inf if parsed is None else parsed


def collect_predictive_rows(manifest_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in read_csv_manifest(manifest_path):
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


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in columns} for row in rows]
        )


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(
    handle, rows: list[dict[str, object]], columns: list[str]
) -> None:
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        handle.write(
            "| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |\n"
        )
    handle.write("\n")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write grouped posterior-predictive summaries from a predictive manifest."
    )
    parser.add_argument("--manifest", required=True, type=str)
    args = parser.parse_args()
    outputs = write_posterior_predictive_reports(args.manifest)
    print(f"Wrote winners summary: {outputs['winners_csv']}")


if __name__ == "__main__":
    main()
