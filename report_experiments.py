import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def read_manifest(manifest_path: Path) -> list[Path]:
    """Load experiment folder paths from a manifest file."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    """Read one estimator summary CSV into a list of row dictionaries."""
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_experiment(folder: Path) -> dict:
    """Collect metadata, configuration, metrics, and parameter rows for one experiment."""
    metadata = OmegaConf.load(folder / "experiment_metadata.yaml")
    config = OmegaConf.load(folder / "realized_config.yaml")
    rows = load_summary_rows(folder / "mple_summary.csv")

    parameter_rows = [row for row in rows if row["category"] == "parameter"]
    metric_rows = [row for row in rows if row["category"] == "metric"]
    metrics = {row["name"]: float(row["estimate"]) for row in metric_rows}

    return {
        "folder": folder,
        "metadata": metadata,
        "config": config,
        "parameter_rows": parameter_rows,
        "metrics": metrics,
    }


def grouped_metric_summary(experiments: list[dict], metadata_key: str) -> dict[str, dict[str, float]]:
    """Average the main fit metrics over experiments that share one metadata value."""
    grouped = defaultdict(list)
    for experiment in experiments:
        value = str(experiment["metadata"].get(metadata_key, "unknown"))
        grouped[value].append(experiment["metrics"])

    summary = {}
    for value, metric_list in grouped.items():
        summary[value] = {
            "parameter_rmse": float(np.mean([m["parameter_rmse"] for m in metric_list])),
            "field_rmse": float(np.mean([m["field_rmse"] for m in metric_list])),
            "interaction_fro_error": float(
                np.mean([m["interaction_fro_error"] for m in metric_list])
            ),
            "count": len(metric_list),
        }
    return summary


def write_overall_csv(report_stem: Path, experiments: list[dict]) -> None:
    """Write a single-row-per-experiment CSV summary."""
    csv_path = Path(f"{report_stem}.csv")
    fieldnames = [
        "descriptor",
        "folder",
        "N_regime",
        "T_regime",
        "temperature_regime",
        "fro_regime",
        "graph_family",
        "gamma_fro_norm",
        "field_complexity",
        "interaction_complexity",
        "final_loss",
        "field_rmse",
        "interaction_fro_error",
        "parameter_rmse",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            metadata = experiment["metadata"]
            writer.writerow(
                {
                    "descriptor": metadata.descriptor,
                    "folder": str(experiment["folder"]),
                    "N_regime": metadata.get("N_regime", ""),
                    "T_regime": metadata.get("T_regime", ""),
                    "temperature_regime": metadata.get("temperature_regime", ""),
                    "fro_regime": metadata.get("fro_regime", ""),
                    "graph_family": metadata.get("graph_family", ""),
                    "gamma_fro_norm": metadata.get("gamma_fro_norm", ""),
                    "field_complexity": metadata.get("field_complexity", ""),
                    "interaction_complexity": metadata.get("interaction_complexity", ""),
                    "final_loss": experiment["metrics"]["final_loss"],
                    "field_rmse": experiment["metrics"]["field_rmse"],
                    "interaction_fro_error": experiment["metrics"]["interaction_fro_error"],
                    "parameter_rmse": experiment["metrics"]["parameter_rmse"],
                }
            )


def write_group_section(handle, title: str, grouped_metrics: dict[str, dict[str, float]]) -> None:
    """Write one grouped summary table to the Markdown report."""
    handle.write(f"## {title}\n\n")
    handle.write("| value | experiments | mean parameter rmse | mean field rmse | mean interaction fro error |\n")
    handle.write("| --- | ---: | ---: | ---: | ---: |\n")
    for value, metrics in sorted(grouped_metrics.items()):
        handle.write(
            f"| {value} | {metrics['count']} | {metrics['parameter_rmse']:.4f} | "
            f"{metrics['field_rmse']:.4f} | {metrics['interaction_fro_error']:.4f} |\n"
        )
    handle.write("\n")


def write_markdown_report(report_stem: Path, experiments: list[dict]) -> None:
    """Write a Markdown report with factor-level takeaways and per-experiment details."""
    md_path = Path(f"{report_stem}.md")

    best_overall = min(experiments, key=lambda exp: exp["metrics"]["parameter_rmse"])
    worst_overall = max(experiments, key=lambda exp: exp["metrics"]["parameter_rmse"])
    best_field = min(experiments, key=lambda exp: exp["metrics"]["field_rmse"])
    worst_interaction = max(
        experiments, key=lambda exp: exp["metrics"]["interaction_fro_error"]
    )

    grouped_by_n = grouped_metric_summary(experiments, "N_regime")
    grouped_by_t = grouped_metric_summary(experiments, "T_regime")
    grouped_by_temp = grouped_metric_summary(experiments, "temperature_regime")
    grouped_by_fro = grouped_metric_summary(experiments, "fro_regime")
    grouped_by_field = grouped_metric_summary(experiments, "field_complexity")
    grouped_by_interaction = grouped_metric_summary(experiments, "interaction_complexity")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Conditional MPLE Grid Report\n\n")
        handle.write("## High-Level Takeaways\n\n")
        handle.write(
            f"- Best overall parameter recovery came from **{best_overall['metadata'].descriptor}** "
            f"with parameter RMSE {best_overall['metrics']['parameter_rmse']:.4f}.\n"
        )
        handle.write(
            f"- The hardest experiment overall was **{worst_overall['metadata'].descriptor}** "
            f"with parameter RMSE {worst_overall['metrics']['parameter_rmse']:.4f}.\n"
        )
        handle.write(
            f"- Best field reconstruction came from **{best_field['metadata'].descriptor}** "
            f"with field RMSE {best_field['metrics']['field_rmse']:.4f}.\n"
        )
        handle.write(
            f"- The hardest interaction recovery came from **{worst_interaction['metadata'].descriptor}** "
            f"with interaction Frobenius error {worst_interaction['metrics']['interaction_fro_error']:.4f}.\n\n"
        )

        write_group_section(handle, "Grouped by N", grouped_by_n)
        write_group_section(handle, "Grouped by T", grouped_by_t)
        write_group_section(handle, "Grouped by Temperature Regime", grouped_by_temp)
        write_group_section(handle, "Grouped by Graph Frobenius Regime", grouped_by_fro)
        write_group_section(handle, "Grouped by Field Complexity", grouped_by_field)
        write_group_section(
            handle, "Grouped by Interaction Complexity", grouped_by_interaction
        )

        handle.write("## Overall Experiment Summary\n\n")
        handle.write(
            "| experiment | N | T | temp | fro | graph | gamma fro | field complexity | interaction complexity | parameter rmse | field rmse | interaction fro error |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |\n")
        for experiment in experiments:
            metadata = experiment["metadata"]
            handle.write(
                f"| {metadata.descriptor} | {metadata.get('N_regime', '')} | "
                f"{metadata.get('T_regime', '')} | {metadata.get('temperature_regime', '')} | "
                f"{metadata.get('fro_regime', '')} | {metadata.get('graph_family', '')} | "
                f"{float(metadata.get('gamma_fro_norm', 0.0)):.4f} | "
                f"{metadata.get('field_complexity', '')} | "
                f"{metadata.get('interaction_complexity', '')} | "
                f"{experiment['metrics']['parameter_rmse']:.4f} | "
                f"{experiment['metrics']['field_rmse']:.4f} | "
                f"{experiment['metrics']['interaction_fro_error']:.4f} |\n"
            )

        for experiment in experiments:
            metadata = experiment["metadata"]
            config = experiment["config"]
            handle.write(f"\n## {metadata.descriptor}\n\n")
            handle.write(
                f"- Folder: `{experiment['folder']}`\n"
                f"- N={config.global_params.N}, T={config.global_params.T}, s={config.global_params.s}, "
                f"seed={config.generation_params.seed}\n"
                f"- Temperature regime: {metadata.get('temperature_regime', '')}\n"
                f"- Graph Frobenius regime: {metadata.get('fro_regime', '')}\n"
                f"- Graph family: {metadata.get('graph_family', '')}\n"
                f"- Realized gamma Frobenius norm: {float(metadata.get('gamma_fro_norm', 0.0)):.4f}\n"
                f"- Field complexity: {metadata.get('field_complexity', '')}\n"
                f"- Interaction complexity: {metadata.get('interaction_complexity', '')}\n"
                f"- Metrics: final loss={experiment['metrics']['final_loss']:.4f}, "
                f"field RMSE={experiment['metrics']['field_rmse']:.4f}, "
                f"interaction Frobenius error={experiment['metrics']['interaction_fro_error']:.4f}, "
                f"parameter RMSE={experiment['metrics']['parameter_rmse']:.4f}\n\n"
            )
            handle.write("| parameter | true | estimate | abs error | squared error |\n")
            handle.write("| --- | ---: | ---: | ---: | ---: |\n")
            for row in experiment["parameter_rows"]:
                estimate = float(row["estimate"])
                true = float(row["true"])
                abs_error = abs(estimate - true)
                squared_error = float(row["squared_error"])
                handle.write(
                    f"| {row['name']} | {true:.4f} | {estimate:.4f} | {abs_error:.4f} | {squared_error:.4f} |\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a report from a manifest of conditional MPLE experiments."
    )
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument(
        "--report_stem",
        required=True,
        type=str,
        help="Output path without extension for the generated CSV and Markdown report.",
    )
    args = parser.parse_args()

    experiments = [
        summarize_experiment(folder)
        for folder in read_manifest(Path(args.manifest))
        if (folder / "mple_summary.csv").exists()
    ]
    write_overall_csv(Path(args.report_stem), experiments)
    write_markdown_report(Path(args.report_stem), experiments)
