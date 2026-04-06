from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTCOME_LABELS = {
    "i_drugs_gt_0_pm1": "Drug crime indicator: zero drug crimes vs at least one drug crime",
    "i_drugs_gt_district_mean_pm1": "Drug crime indicator: at or below district-average drug crime vs above district average",
    "any_crime_gt_0_pm1": "Any-crime indicator: zero total crimes vs at least one total crime",
    "any_crime_gt_1_pm1": "Any-crime indicator: at most one total crime vs more than one total crime",
    "any_crime_gt_2_pm1": "Any-crime indicator: at most two total crimes vs more than two total crimes",
    "any_crime_gt_3_pm1": "Any-crime indicator: at most three total crimes vs more than three total crimes",
    "any_crime_gt_district_mean_pm1": "Any-crime indicator: at or below district-average total crime vs above district average",
    "any_crime_gt_block_mean_pm1": "Any-crime indicator: at or below the block-specific mean total crime vs above the block-specific mean",
}

OUTCOME_DEFINITIONS = {
    "i_drugs_gt_0_pm1": "+1 means no drug crime in the period; -1 means at least one drug crime in the period.",
    "i_drugs_gt_district_mean_pm1": "+1 means drug crime is at or below the neighborhood-district mean; -1 means above the district mean.",
    "any_crime_gt_0_pm1": "+1 means no recorded crime in the period; -1 means at least one recorded crime in the period.",
    "any_crime_gt_1_pm1": "+1 means there is at most one recorded crime in the period; -1 means there are two or more recorded crimes in the period.",
    "any_crime_gt_2_pm1": "+1 means there are at most two recorded crimes in the period; -1 means there are three or more recorded crimes in the period.",
    "any_crime_gt_3_pm1": "+1 means there are at most three recorded crimes in the period; -1 means there are four or more recorded crimes in the period.",
    "any_crime_gt_district_mean_pm1": "+1 means total crime is at or below the neighborhood-district mean; -1 means above the district mean.",
    "any_crime_gt_block_mean_pm1": "+1 means total crime is at or below that exact block's own time-average total crime; -1 means above the block-specific mean.",
}

NETWORK_LABELS = {
    "contiguity": "Geographic contiguity network",
    "knn_8": "8-nearest-neighbor centroid graph",
    "knn_16": "16-nearest-neighbor centroid graph",
    "centroid_distance_kernel_8": "Centroid-distance kernel with 8-nearest-neighbor support",
    "centroid_distance_kernel_16": "Centroid-distance kernel with 16-nearest-neighbor support",
}

INTERACTION_LABELS = {
    "contiguity": "Contiguity interaction matrix",
    "knn_8": "8-nearest-neighbor interaction matrix",
    "knn_16": "16-nearest-neighbor interaction matrix",
    "centroid_distance_kernel_8": "Centroid-distance-kernel interaction matrix (8-neighbor support)",
    "centroid_distance_kernel_16": "Centroid-distance-kernel interaction matrix (16-neighbor support)",
}

INTERVENTION_DEFINITION = (
    "z = +1 means the block is under intervention in that quarter; "
    "z = -1 means no intervention in that quarter."
)


def parse_experiment_name(experiment_name: str) -> tuple[str, str]:
    """Split one SeattleDMI experiment folder name into outcome and network."""
    if "__" not in experiment_name:
        raise ValueError(
            f"Experiment folder '{experiment_name}' does not match the expected outcome__network pattern."
        )
    outcome, network = experiment_name.split("__", 1)
    return outcome, network


def load_summary_rows(summary_path: Path) -> dict[str, float]:
    """Load one mple_summary.csv file into a flat mapping from name to estimate."""
    values: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row["estimate"]:
                continue
            values[row["name"]] = float(row["estimate"])
    return values


def readable_outcome_label(outcome: str) -> str:
    """Return a report-friendly English label for one outcome code."""
    return OUTCOME_LABELS.get(outcome, outcome.replace("_", " "))


def readable_outcome_definition(outcome: str) -> str:
    """Return a plain-English explanation of one binary outcome."""
    return OUTCOME_DEFINITIONS.get(outcome, "")


def readable_network_label(network: str) -> str:
    """Return a report-friendly English label for one network code."""
    return NETWORK_LABELS.get(network, network.replace("_", " "))


def readable_interaction_label(network: str) -> str:
    """Return a report-friendly English label for the known interaction matrix."""
    return INTERACTION_LABELS.get(network, network.replace("_", " "))


def collect_experiment_rows(experiments_root: Path) -> list[dict[str, float | str]]:
    """Collect one flat experiment-summary row per SeattleDMI experiment folder."""
    experiment_dirs = sorted(path for path in experiments_root.iterdir() if path.is_dir())
    if not experiment_dirs:
        return []

    all_keys: set[str] = set()
    raw_rows: list[tuple[str, str, str, dict[str, float]]] = []

    for experiment_dir in experiment_dirs:
        summary_path = experiment_dir / "mple_summary.csv"
        if not summary_path.exists():
            continue
        outcome, network = parse_experiment_name(experiment_dir.name)
        values = load_summary_rows(summary_path)
        all_keys.update(values.keys())
        raw_rows.append((experiment_dir.name, outcome, network, values))

    ordered_keys = sorted(
        key for key in all_keys if key != "final_loss"
    ) + (["final_loss"] if "final_loss" in all_keys else [])

    rows: list[dict[str, float | str]] = []
    for experiment_name, outcome, network, values in raw_rows:
        row: dict[str, float | str] = {
            "experiment_name": experiment_name,
            "outcome_code": outcome,
            "outcome_label": readable_outcome_label(outcome),
            "outcome_definition": readable_outcome_definition(outcome),
            "intervention_definition": INTERVENTION_DEFINITION,
            "network_code": network,
            "network_label": readable_network_label(network),
            "interaction_matrix_label": readable_interaction_label(network),
        }
        for key in ordered_keys:
            row[key] = values.get(key, "")
        rows.append(row)
    return rows


def write_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write the consolidated experiment summary as CSV."""
    if not rows:
        raise ValueError("No experiment rows were found to summarize.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write the consolidated experiment summary as a Markdown table."""
    if not rows:
        raise ValueError("No experiment rows were found to summarize.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SeattleDMI MPLE Experiment Summary\n\n")
        handle.write(
            "This table summarizes one MPLE fit per outcome/network pair. "
            "For the binary outcome, `x = +1` denotes the better lower-crime state and "
            "`x = -1` denotes the worse higher-crime state. "
            "For the intervention, `z = +1` denotes intervention and `z = -1` denotes no intervention.\n\n"
        )
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            rendered = []
            for key in fieldnames:
                value = row[key]
                if isinstance(value, float):
                    rendered.append(f"{value:.6f}")
                else:
                    rendered.append(str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    """Create one experiment-level summary table across the SeattleDMI MPLE runs."""
    parser = argparse.ArgumentParser(
        description="Summarize SeattleDMI MPLE experiment folders into one table."
    )
    parser.add_argument(
        "--experiments_root",
        type=Path,
        default=Path("experiments/SeattleDMI"),
        help="Root directory containing one SeattleDMI experiment folder per outcome/network pair.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Where to write the consolidated CSV summary.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=None,
        help="Where to write the consolidated Markdown summary.",
    )
    args = parser.parse_args()

    experiments_root = args.experiments_root.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else (experiments_root / "reports" / "MPLE_experiment_summary.csv").resolve()
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else (experiments_root / "reports" / "MPLE_experiment_summary.md").resolve()
    )

    rows = collect_experiment_rows(experiments_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows)


if __name__ == "__main__":
    main()
