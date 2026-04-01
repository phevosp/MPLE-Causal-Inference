from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def read_manifest(manifest_path: Path) -> list[Path]:
    """Read experiment folders from a manifest file."""
    return [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_experiment(folder: Path) -> dict:
    """Load one experiment folder into metadata, configuration, metrics, and parameter rows."""
    metadata = OmegaConf.load(folder / "experiment_metadata.yaml")
    config = OmegaConf.load(folder / "realized_config.yaml")
    metrics: dict[str, float] = {}
    parameter_rows: list[dict[str, str]] = []

    with (folder / "mple_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == "metric":
                metrics[row["name"]] = float(row["estimate"])
            elif row["category"] == "parameter":
                parameter_rows.append(row)

    return {
        "folder": folder,
        "metadata": metadata,
        "config": config,
        "metrics": metrics,
        "parameter_rows": parameter_rows,
    }


def parameter_block(name: str) -> str:
    """Map a flattened parameter name to a readable block label."""
    if name.startswith("field::"):
        return "field"
    if name.startswith("interaction::"):
        return "interaction"
    return name


def factor_value(exp: dict, key: str) -> str:
    """Return the comparison value for one factor, using config for N/T and metadata otherwise."""
    if key in {"N", "T"}:
        return str(getattr(exp["config"].global_params, key))
    return str(exp["metadata"].get(key, ""))


def parameter_abs_errors(exp: dict) -> dict[str, float]:
    """Collect absolute errors by parameter name for one experiment."""
    errors: dict[str, float] = {}
    for row in exp["parameter_rows"]:
        errors[row["name"]] = abs(float(row["estimate"]) - float(row["true"]))
    return errors


def parameter_block_mean_error(exp: dict, block: str) -> float:
    """Average absolute error for one parameter block."""
    errors = parameter_abs_errors(exp)
    values = [err for name, err in errors.items() if parameter_block(name) == block]
    return mean(values)


def scalar_abs_error(exp: dict, name: str) -> float:
    """Absolute error for one scalar parameter."""
    errors = parameter_abs_errors(exp)
    return float(errors[name])


def mean(values: list[float]) -> float:
    """Return the mean of a numeric list."""
    return float(statistics.mean(values)) if values else float("nan")


def summarize_factor(experiments: list[dict], key: str, value: str) -> dict[str, float]:
    """Summarize recovery metrics for experiments matching one factor value."""
    subset = [exp for exp in experiments if factor_value(exp, key) == value]
    return {
        "count": len(subset),
        "parameter_rmse": mean([exp["metrics"]["parameter_rmse"] for exp in subset]),
        "field_rmse": mean([exp["metrics"]["field_rmse"] for exp in subset]),
        "interaction_fro_error": mean([exp["metrics"]["interaction_fro_error"] for exp in subset]),
        "avg_field_coef_abs_error": mean(
            [parameter_block_mean_error(exp, "field") for exp in subset]
        ),
        "avg_interaction_coef_abs_error": mean(
            [parameter_block_mean_error(exp, "interaction") for exp in subset]
        ),
        "beta_abs_error": mean([scalar_abs_error(exp, "beta") for exp in subset]),
        "eta_abs_error": mean([scalar_abs_error(exp, "eta") for exp in subset]),
        "zeta_abs_error": mean([scalar_abs_error(exp, "zeta") for exp in subset]),
        "psi_abs_error": mean([scalar_abs_error(exp, "psi") for exp in subset]),
    }


def write_parameter_rows_csv(report_dir: Path, experiments: list[dict]) -> Path:
    """Write a flattened table with one row per parameter estimate."""
    csv_path = report_dir / "parameter_recovery_parameter_rows.csv"
    fieldnames = [
        "descriptor",
        "folder",
        "N",
        "T",
        "temperature_regime",
        "fro_regime",
        "graph_family",
        "field_complexity",
        "interaction_complexity",
        "gamma_fro_norm",
        "parameter_block",
        "name",
        "true",
        "estimate",
        "abs_error",
        "squared_error",
        "parameter_rmse",
        "field_rmse",
        "interaction_fro_error",
    ]

    rows = []
    for exp in experiments:
        metadata = exp["metadata"]
        config = exp["config"]
        metrics = exp["metrics"]
        for row in exp["parameter_rows"]:
            rows.append(
                {
                    "descriptor": str(metadata.get("descriptor", "")),
                    "folder": str(exp["folder"]),
                    "N": int(config.global_params.N),
                    "T": int(config.global_params.T),
                    "temperature_regime": str(metadata.get("temperature_regime", "")),
                    "fro_regime": str(metadata.get("fro_regime", "")),
                    "graph_family": str(metadata.get("graph_family", "")),
                    "field_complexity": str(metadata.get("field_complexity", "")),
                    "interaction_complexity": str(metadata.get("interaction_complexity", "")),
                    "gamma_fro_norm": float(metadata.get("gamma_fro_norm", float("nan"))),
                    "parameter_block": parameter_block(row["name"]),
                    "name": row["name"],
                    "true": float(row["true"]),
                    "estimate": float(row["estimate"]),
                    "abs_error": abs(float(row["estimate"]) - float(row["true"])),
                    "squared_error": float(row["squared_error"]),
                    "parameter_rmse": float(metrics["parameter_rmse"]),
                    "field_rmse": float(metrics["field_rmse"]),
                    "interaction_fro_error": float(metrics["interaction_fro_error"]),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_metric(value: float) -> str:
    """Format a numeric value for Markdown tables."""
    return f"{value:.4f}"


def select_experiment(experiments: list[dict], descriptor: str) -> dict:
    """Return the experiment with the requested descriptor."""
    for exp in experiments:
        if str(exp["metadata"].get("descriptor", "")) == descriptor:
            return exp
    raise KeyError(f"Could not find experiment '{descriptor}'.")


def summarize_known_graph(experiments: list[dict]) -> dict[str, dict[str, float]]:
    """Summarize the known-graph experiments by factor values."""
    core = [exp for exp in experiments if str(exp["metadata"].get("interaction_complexity", "")) == "known_graph"]
    summary = {
        "N": {},
        "T": {},
        "temperature_regime": {},
        "fro_regime": {},
        "field_complexity": {},
    }
    for key in summary:
        values = sorted({factor_value(exp, key) for exp in core})
        summary[key] = {value: summarize_factor(core, key, value) for value in values}
    return summary


def write_report(report_path: Path, experiments: list[dict]) -> None:
    """Write a detailed markdown report centered on parameter recovery."""
    known_graph = [
        exp for exp in experiments if str(exp["metadata"].get("interaction_complexity", "")) == "known_graph"
    ]
    core_summary = summarize_known_graph(experiments)
    flat_best = min(known_graph, key=lambda exp: exp["metrics"]["parameter_rmse"])
    flat_worst = max(known_graph, key=lambda exp: exp["metrics"]["parameter_rmse"])
    shared_field_example = select_experiment(
        experiments, "N5000_T100_high_temp_fro_large_shared_feature_field_known_graph"
    )
    fro_small = select_experiment(
        experiments, "N5000_T100_baseline_temp_fro_small_uniform_known_graph"
    )
    fro_medium = select_experiment(
        experiments, "N5000_T100_baseline_temp_fro_medium_uniform_known_graph"
    )
    fro_large = select_experiment(
        experiments, "N5000_T100_baseline_temp_fro_large_uniform_known_graph"
    )

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Parameter Recovery Report\n\n")
        handle.write(
            "This report focuses on direct true-vs-estimated parameter recovery, not on loss.\n\n"
        )
        handle.write(
            "The full flattened parameter table is available in `reports/parameter_recovery_parameter_rows.csv`.\n\n"
        )
        handle.write("## Main Takeaways\n\n")
        handle.write(
            f"- In the core known-graph setting, larger `N` and larger `T` both improve recovery substantially. "
            f"The best baseline experiment is `{flat_best['metadata'].descriptor}` with parameter RMSE "
            f"{flat_best['metrics']['parameter_rmse']:.4f}.\n"
        )
        handle.write(
            f"- Low temperature is the hardest regime. The worst known-graph experiment is "
            f"`{flat_worst['metadata'].descriptor}` with parameter RMSE {flat_worst['metrics']['parameter_rmse']:.4f}.\n"
        )
        handle.write(
            f"- `shared_feature_field` improves recovery in this grid rather than hurting it. "
            f"It lowers mean parameter RMSE in the known-graph setting from "
            f"{core_summary['field_complexity']['uniform']['parameter_rmse']:.4f} to "
            f"{core_summary['field_complexity']['shared_feature_field']['parameter_rmse']:.4f}.\n"
        )
        handle.write(
            f"- The Frobenius effect is not monotone. Very small Frobenius norm is bad for recovery, "
            f"but the medium Frobenius regime is often best overall.\n\n"
        )

        handle.write("## Levers\n\n")
        handle.write("### `N`\n\n")
        N_rows = []
        for value, summary in core_summary["N"].items():
            N_rows.append(
                [
                    value,
                    str(summary["count"]),
                    format_metric(summary["parameter_rmse"]),
                    format_metric(summary["field_rmse"]),
                    format_metric(summary["interaction_fro_error"]),
                    format_metric(summary["avg_field_coef_abs_error"]),
                    format_metric(summary["avg_interaction_coef_abs_error"]),
                ]
            )
        handle.write(
            render_table(
                ["N", "experiments", "parameter_rmse", "field_rmse", "interaction_fro_error", "avg field abs err", "avg interaction abs err"],
                N_rows,
            )
            + "\n\n"
        )

        handle.write("### `T`\n\n")
        T_rows = []
        for value, summary in core_summary["T"].items():
            T_rows.append(
                [
                    value,
                    str(summary["count"]),
                    format_metric(summary["parameter_rmse"]),
                    format_metric(summary["field_rmse"]),
                    format_metric(summary["interaction_fro_error"]),
                    format_metric(summary["beta_abs_error"]),
                ]
            )
        handle.write(
            render_table(
                ["T", "experiments", "parameter_rmse", "field_rmse", "interaction_fro_error", "beta abs err"],
                T_rows,
            )
            + "\n\n"
        )

        handle.write("### Temperature\n\n")
        temp_rows = []
        for value, summary in core_summary["temperature_regime"].items():
            temp_rows.append(
                [
                    value,
                    str(summary["count"]),
                    format_metric(summary["parameter_rmse"]),
                    format_metric(summary["field_rmse"]),
                    format_metric(summary["interaction_fro_error"]),
                    format_metric(summary["beta_abs_error"]),
                    format_metric(summary["eta_abs_error"]),
                    format_metric(summary["zeta_abs_error"]),
                    format_metric(summary["psi_abs_error"]),
                ]
            )
        handle.write(
            render_table(
                [
                    "temperature",
                    "experiments",
                    "parameter_rmse",
                    "field_rmse",
                    "interaction_fro_error",
                    "beta abs err",
                    "eta abs err",
                    "zeta abs err",
                    "psi abs err",
                ],
                temp_rows,
            )
            + "\n\n"
        )

        handle.write("### Frobenius Norm\n\n")
        fro_rows = []
        for value, summary in core_summary["fro_regime"].items():
            fro_rows.append(
                [
                    value,
                    str(summary["count"]),
                    format_metric(summary["parameter_rmse"]),
                    format_metric(summary["field_rmse"]),
                    format_metric(summary["interaction_fro_error"]),
                    format_metric(summary["avg_field_coef_abs_error"]),
                    format_metric(summary["avg_interaction_coef_abs_error"]),
                    format_metric(summary["beta_abs_error"]),
                ]
            )
        handle.write(
            render_table(
                [
                    "fro regime",
                    "experiments",
                    "parameter_rmse",
                    "field_rmse",
                    "interaction_fro_error",
                    "avg field abs err",
                    "avg interaction abs err",
                    "beta abs err",
                ],
                fro_rows,
            )
            + "\n\n"
        )
        handle.write(
            "The data support a limited version of the Frobenius hypothesis: moving away from the complete-graph regime helps, "
            "but the best overall recovery is usually at the medium regime rather than the largest Frobenius regime.\n\n"
        )

        handle.write("### `shared_feature_field`\n\n")
        field_rows = []
        for value, summary in core_summary["field_complexity"].items():
            field_rows.append(
                [
                    value,
                    str(summary["count"]),
                    format_metric(summary["parameter_rmse"]),
                    format_metric(summary["field_rmse"]),
                    format_metric(summary["interaction_fro_error"]),
                    format_metric(summary["beta_abs_error"]),
                    format_metric(summary["avg_field_coef_abs_error"]),
                ]
            )
        handle.write(
            render_table(
                ["field complexity", "experiments", "parameter_rmse", "field_rmse", "interaction_fro_error", "beta abs err", "avg field abs err"],
                field_rows,
            )
            + "\n\n"
        )

        handle.write("## Representative True vs Estimated Tables\n\n")
        examples = [
            ("Best baseline case", flat_best),
            ("Hard low-temperature case", flat_worst),
            ("Rich field case", shared_field_example),
        ]
        for title, exp in examples:
            metadata = exp["metadata"]
            handle.write(f"### {title}\n\n")
            handle.write(
                f"- `{metadata.descriptor}`\n"
                f"- parameter_rmse = {exp['metrics']['parameter_rmse']:.4f}\n"
                f"- field_rmse = {exp['metrics']['field_rmse']:.4f}\n"
                f"- interaction_fro_error = {exp['metrics']['interaction_fro_error']:.4f}\n\n"
            )
            param_rows = []
            for row in exp["parameter_rows"]:
                param_rows.append(
                    [
                        row["name"],
                        f"{float(row['true']):.4f}",
                        f"{float(row['estimate']):.4f}",
                        f"{abs(float(row['estimate']) - float(row['true'])):.4f}",
                        f"{float(row['squared_error']):.4f}",
                    ]
                )
            handle.write(
                render_table(
                    ["parameter", "true", "estimate", "abs error", "squared error"],
                    param_rows,
                )
                + "\n\n"
            )

        handle.write("## Frobenius Comparison at Fixed `N`, `T`, and Temperature\n\n")
        handle.write(
            "These three experiments differ only in graph family, so they isolate the effect of the graph's Frobenius norm.\n\n"
        )
        for exp in [fro_small, fro_medium, fro_large]:
            metadata = exp["metadata"]
            handle.write(f"### {metadata.descriptor}\n\n")
            handle.write(
                f"- graph family = {metadata.get('graph_family', '')}\n"
                f"- gamma Frobenius norm = {float(metadata.get('gamma_fro_norm', float('nan'))):.4f}\n"
                f"- parameter_rmse = {exp['metrics']['parameter_rmse']:.4f}\n"
                f"- field_rmse = {exp['metrics']['field_rmse']:.4f}\n"
                f"- interaction_fro_error = {exp['metrics']['interaction_fro_error']:.4f}\n\n"
            )
            param_rows = []
            for row in exp["parameter_rows"]:
                param_rows.append(
                    [
                        row["name"],
                        f"{float(row['true']):.4f}",
                        f"{float(row['estimate']):.4f}",
                        f"{abs(float(row['estimate']) - float(row['true'])):.4f}",
                        f"{float(row['squared_error']):.4f}",
                    ]
                )
            handle.write(
                render_table(
                    ["parameter", "true", "estimate", "abs error", "squared error"],
                    param_rows,
                )
                + "\n\n"
            )


def main() -> None:
    """Entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate a detailed parameter-recovery report from a manifest."
    )
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--report_stem", required=True, type=str)
    args = parser.parse_args()

    report_stem = Path(args.report_stem)
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    report_dir = report_stem.parent

    experiments = [load_experiment(folder) for folder in read_manifest(Path(args.manifest))]
    write_parameter_rows_csv(report_dir, experiments)
    write_report(Path(f"{args.report_stem}.md"), experiments)


if __name__ == "__main__":
    main()
