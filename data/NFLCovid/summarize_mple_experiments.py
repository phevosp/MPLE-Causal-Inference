"""Summarize nationwide NFL + COVID county MPLE experiment folders."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EXPERIMENT_ROOT,
    intervention_label_from_code,
    outcome_label_from_code,
    parse_experiment_name,
)


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
        for column in ["positive_share", "variance", "transition_rate", "weeks", "counties"]:
            values[f"{prefix}_{column}"] = row[column]
        values[f"{prefix}_rule"] = row["rule"]
    return values


def collect_rows(experiments_root: Path) -> list[dict[str, object]]:
    manifest_path = experiments_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_lookup = manifest.set_index("experiment_name").to_dict(orient="index") if not manifest.empty else {}

    rows: list[dict[str, object]] = []
    if not manifest.empty:
        summary_paths = [
            experiments_root / str(experiment_name_value) / "mple_summary.csv"
            for experiment_name_value in manifest["experiment_name"].tolist()
            if (experiments_root / str(experiment_name_value) / "mple_summary.csv").exists()
        ]
    else:
        summary_paths = sorted(experiments_root.rglob("mple_summary.csv"))

    for summary_path in summary_paths:
        experiment_dir = summary_path.parent
        if experiment_dir.name == "reports":
            continue
        parsed = parse_experiment_name(experiment_dir.name)
        metadata_path = experiment_dir / "experiment_metadata.yaml"
        metadata = OmegaConf.load(metadata_path) if metadata_path.exists() else OmegaConf.create({})
        manifest_row = manifest_lookup.get(experiment_dir.name, {})
        estimates = load_summary_values(summary_path)
        binary_summary = load_binary_summary(experiment_dir / "binary_definition_summary.csv")
        full_fit_status = manifest_row.get("full_fit_status", "completed")
        if full_fit_status == "not_run":
            full_fit_status = "completed_existing"
        row: dict[str, object] = {
            "experiment_name": experiment_dir.name,
            "outcome_code": parsed["outcome_code"],
            "outcome_label": outcome_label_from_code(parsed["outcome_code"]),
            "intervention_code": parsed["intervention_code"],
            "intervention_label": intervention_label_from_code(parsed["intervention_code"]),
            "lag_code": parsed["lag_code"],
            "network_name": parsed["network_name"],
            "field_basis_mode": metadata.get("field_basis_mode", ""),
            "tau_zero_mean": metadata.get("tau_zero_mean", ""),
            "tau_smoothness_lambda": metadata.get("tau_smoothness_lambda", ""),
            "realized_start_date": metadata.get("realized_week_start_date", ""),
            "realized_end_date": metadata.get("realized_week_end_date", ""),
            "requested_calendar_weeks": metadata.get("requested_calendar_weeks", ""),
            "realized_calendar_weeks": metadata.get("realized_calendar_weeks", ""),
            "weeks_dropped_due_to_missing_or_lag": metadata.get("weeks_dropped_due_to_missing_or_lag", ""),
            "time_steps": metadata.get("time_steps", ""),
            "requested_node_count": metadata.get("requested_node_count", ""),
            "node_count": metadata.get("node_count", ""),
            "dropped_node_count": metadata.get("dropped_node_count", ""),
            "full_fit_status": full_fit_status,
            "outcome_only_fit_status": manifest_row.get("outcome_only_fit_status", ""),
            "fallback_run": manifest_row.get("fallback_run", False),
        }
        row.update(binary_summary)
        for key, value in sorted(estimates.items()):
            row[key] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished NFL COVID experiments were found.")
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
        raise ValueError("No finished NFL COVID experiments were found.")
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# NFL COVID MPLE Experiment Summary\n\n")
        handle.write(
            "This table summarizes every finished nationwide NFL + COVID county experiment folder "
            "that contains an `mple_summary.csv` file. `fallback_run = True` means the full fit "
            "failed and the folder was rerun with `--outcome_only`.\n\n"
        )
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered = []
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, float):
                    rendered.append(f"{value:.6f}")
                else:
                    rendered.append(str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize nationwide NFL + COVID county MPLE experiment folders."
    )
    parser.add_argument(
        "--experiments_root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root directory containing NFL COVID experiment folders.",
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
