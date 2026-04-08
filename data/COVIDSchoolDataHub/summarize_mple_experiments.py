"""Summarize Ohio COVIDSchoolDataHub MPLE experiment folders into tabular reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def load_summary_values(summary_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row["estimate"]:
                continue
            values[row["name"]] = float(row["estimate"])
    return values


def load_binary_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    table = pd.read_csv(summary_path)
    values: dict[str, object] = {}
    for _, row in table.iterrows():
        prefix = str(row["variable"])
        values[f"{prefix}_rule"] = row["rule"]
        values[f"{prefix}_positive_share"] = row["positive_share"]
        values[f"{prefix}_variance"] = row["variance"]
        values[f"{prefix}_transition_rate"] = row["transition_rate"]
    return values


def collect_rows(experiments_root: Path) -> list[dict[str, object]]:
    manifest_path = experiments_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    if not manifest.empty and "experiment_name" in manifest.columns:
        summary_paths = [
            experiments_root / str(experiment_name) / "mple_summary.csv"
            for experiment_name in manifest["experiment_name"].tolist()
            if (experiments_root / str(experiment_name) / "mple_summary.csv").exists()
        ]
    else:
        summary_paths = sorted(experiments_root.rglob("mple_summary.csv"))

    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        experiment_dir = summary_path.parent
        if experiment_dir.name == "reports":
            continue
        metadata_path = experiment_dir / "experiment_metadata.yaml"
        metadata = OmegaConf.load(metadata_path) if metadata_path.exists() else OmegaConf.create({})
        row: dict[str, object] = {
            "experiment_name": experiment_dir.name,
            "panel_frequency": metadata.get("panel_frequency", ""),
            "intervention_source": metadata.get("intervention_source", ""),
            "intervention_rule": metadata.get("intervention_rule", ""),
            "outcome_rule": metadata.get("outcome_rule", ""),
            "lag_period": metadata.get("lag_period", ""),
            "lag_steps": metadata.get("lag_steps", ""),
            "fit_intervention_model": metadata.get("fit_intervention_model", ""),
            "tau_zero_mean": metadata.get("tau_zero_mean", ""),
            "tau_smoothness_lambda": metadata.get("tau_smoothness_lambda", ""),
            "node_count": metadata.get("node_count", ""),
            "time_steps": metadata.get("time_steps", ""),
            "pre_intervention_steps": metadata.get("pre_intervention_steps", ""),
        }
        row.update(load_binary_summary(experiment_dir / "binary_definition_summary.csv"))
        row.update(load_summary_values(summary_path))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished Ohio CSDH experiments were found.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished Ohio CSDH experiments were found.")
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Ohio CSDH MPLE Experiment Summary\n\n")
        handle.write(
            "This table summarizes every finished Ohio COVIDSchoolDataHub experiment folder "
            "that contains an `mple_summary.csv` file.\n\n"
        )
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered: list[str] = []
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, float):
                    rendered.append(f"{value:.6f}")
                else:
                    rendered.append(str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Ohio COVIDSchoolDataHub MPLE experiment folders."
    )
    parser.add_argument(
        "--experiments_root",
        type=Path,
        default=Path("experiments/COVIDSchoolDataHub_OH"),
        help="Root directory containing Ohio COVIDSchoolDataHub experiment folders.",
    )
    args = parser.parse_args()

    experiments_root = args.experiments_root.resolve()
    rows = collect_rows(experiments_root)
    reports_dir = experiments_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(reports_dir / "MPLE_experiment_summary.csv", rows)
    write_markdown(reports_dir / "MPLE_experiment_summary.md", rows)


if __name__ == "__main__":
    main()
