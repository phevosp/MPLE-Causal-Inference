"""Summarize fitted synthetic experiments from a manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


def read_manifest(manifest_path: Path) -> list[Path]:
    return [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_values(summary_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["estimate"]:
                values[row["name"]] = float(row["estimate"])
    return values


def collect_rows(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for folder in read_manifest(manifest_path):
        summary_path = folder / "mple_summary.csv"
        if not summary_path.exists():
            continue
        metadata = OmegaConf.load(folder / "experiment_metadata.yaml") if (folder / "experiment_metadata.yaml").exists() else OmegaConf.create({})
        config = OmegaConf.load(folder / "realized_config.yaml") if (folder / "realized_config.yaml").exists() else OmegaConf.create({})
        values = summary_values(summary_path)
        row: dict[str, object] = {
            "folder": str(folder),
            "descriptor": str(metadata.get("descriptor", folder.name)),
            "field_mode": str(metadata.get("field_mode", "")),
            "intervention_mode": str(metadata.get("intervention_mode", "")),
            "N": int(config.global_params.N) if "global_params" in config and "N" in config.global_params else "",
            "T": int(config.global_params.T) if "global_params" in config and "T" in config.global_params else "",
        }
        row.update(values)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished experiments were found in the manifest.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished experiments were found in the manifest.")
    headers = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic Experiment Summary\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered = []
            for key in headers:
                value = row.get(key, "")
                rendered.append(f"{value:.6f}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic MPLE runs from a manifest.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--report_stem", required=True, type=str)
    args = parser.parse_args()

    rows = collect_rows(Path(args.manifest))
    report_stem = Path(args.report_stem)
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(f"{args.report_stem}.csv"), rows)
    write_markdown(Path(f"{args.report_stem}.md"), rows)


if __name__ == "__main__":
    main()
