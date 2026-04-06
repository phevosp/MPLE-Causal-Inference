"""Generate detailed synthetic parameter-recovery summaries and reports."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def read_manifest(manifest_path: Path) -> list[Path]:
    """Read experiment folders from a manifest file."""
    return [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def metric(exp: dict, name: str) -> float:
    """Return one metric value, defaulting to NaN when unavailable."""
    return float(exp["metrics"].get(name, float("nan")))


def mean(values: list[float]) -> float:
    """Return the mean of the finite numeric values in a list."""
    finite = [float(value) for value in values if not math.isnan(float(value))]
    return float(statistics.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    """Return the median of the finite numeric values in a list."""
    finite = [float(value) for value in values if not math.isnan(float(value))]
    return float(statistics.median(finite)) if finite else float("nan")


def format_metric(value: float) -> str:
    """Format a numeric value for Markdown tables."""
    return "nan" if math.isnan(value) else f"{value:.4f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parameter_block(name: str) -> str:
    """Map a flattened parameter name to a readable block label."""
    if name.startswith("field::"):
        return "field"
    if name.startswith("tau::"):
        return "tau"
    if name.startswith("interaction::"):
        return "interaction"
    return "scalar"


def factor_value(exp: dict, key: str) -> str:
    """Return the comparison value for one factor, using config for N/T and metadata otherwise."""
    if key in {"N", "T"}:
        return str(getattr(exp["config"].global_params, key))
    return str(exp["metadata"].get(key, ""))


def parameter_abs_errors(exp: dict) -> dict[str, float]:
    """Collect absolute errors by parameter name for one experiment."""
    return {
        row["name"]: abs(float(row["estimate"]) - float(row["true"]))
        for row in exp["parameter_rows"]
    }


def parameter_block_mean_error(exp: dict, block: str) -> float:
    """Average absolute error for one parameter block."""
    errors = parameter_abs_errors(exp)
    values = [err for name, err in errors.items() if parameter_block(name) == block]
    return mean(values)


def scalar_abs_error(exp: dict, name: str) -> float:
    """Absolute error for one scalar parameter."""
    errors = parameter_abs_errors(exp)
    return float(errors[name]) if name in errors else float("nan")


def subset_for_factor(experiments: list[dict], key: str, value: str) -> list[dict]:
    """Return all experiments matching one factor value."""
    return [exp for exp in experiments if factor_value(exp, key) == value]


def summarize_factor(experiments: list[dict], key: str, value: str) -> dict[str, float]:
    """Summarize recovery metrics for experiments matching one factor value."""
    subset = subset_for_factor(experiments, key, value)
    return {
        "count": len(subset),
        "final_loss": mean([metric(exp, "final_loss") for exp in subset]),
        "parameter_rmse": mean([metric(exp, "parameter_rmse") for exp in subset]),
        "field_rmse": mean([metric(exp, "field_rmse") for exp in subset]),
        "field_l2_error": mean([metric(exp, "field_l2_error") for exp in subset]),
        "static_field_rmse": mean([metric(exp, "static_field_rmse") for exp in subset]),
        "tau_rmse": mean([metric(exp, "tau_rmse") for exp in subset]),
        "interaction_fro_error": mean(
            [metric(exp, "interaction_fro_error") for exp in subset]
        ),
        "avg_alpha_coef_abs_error": mean(
            [parameter_block_mean_error(exp, "field") for exp in subset]
        ),
        "avg_tau_abs_error": mean(
            [parameter_block_mean_error(exp, "tau") for exp in subset]
        ),
        "avg_interaction_coef_abs_error": mean(
            [parameter_block_mean_error(exp, "interaction") for exp in subset]
        ),
        "beta_abs_error": mean([scalar_abs_error(exp, "beta") for exp in subset]),
        "eta_abs_error": mean([scalar_abs_error(exp, "eta") for exp in subset]),
        "zeta_abs_error": mean([scalar_abs_error(exp, "zeta") for exp in subset]),
        "psi_abs_error": mean([scalar_abs_error(exp, "psi") for exp in subset]),
    }


def available_factor_values(experiments: list[dict], key: str) -> list[str]:
    """Return sorted nonempty values for one grouping factor."""
    values = sorted({factor_value(exp, key) for exp in experiments if factor_value(exp, key) != ""})
    return values


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
        "tau_mode",
        "interaction_complexity",
        "gamma_fro_norm",
        "parameter_block",
        "name",
        "true",
        "estimate",
        "abs_error",
        "squared_error",
        "final_loss",
        "parameter_rmse",
        "field_rmse",
        "field_l2_error",
        "static_field_rmse",
        "tau_rmse",
        "interaction_fro_error",
    ]

    rows = []
    for exp in experiments:
        metadata = exp["metadata"]
        config = exp["config"]
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
                    "tau_mode": str(metadata.get("tau_mode", "")),
                    "interaction_complexity": str(
                        metadata.get("interaction_complexity", "")
                    ),
                    "gamma_fro_norm": float(
                        metadata.get("gamma_fro_norm", float("nan"))
                    ),
                    "parameter_block": parameter_block(row["name"]),
                    "name": row["name"],
                    "true": float(row["true"]),
                    "estimate": float(row["estimate"]),
                    "abs_error": abs(float(row["estimate"]) - float(row["true"])),
                    "squared_error": float(row["squared_error"]),
                    "final_loss": metric(exp, "final_loss"),
                    "parameter_rmse": metric(exp, "parameter_rmse"),
                    "field_rmse": metric(exp, "field_rmse"),
                    "field_l2_error": metric(exp, "field_l2_error"),
                    "static_field_rmse": metric(exp, "static_field_rmse"),
                    "tau_rmse": metric(exp, "tau_rmse"),
                    "interaction_fro_error": metric(exp, "interaction_fro_error"),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_experiment_summary_csv(report_dir: Path, experiments: list[dict]) -> Path:
    """Write one row per experiment with the main field-recovery metrics."""
    csv_path = report_dir / "parameter_recovery_experiment_summary.csv"
    fieldnames = [
        "descriptor",
        "folder",
        "N",
        "T",
        "temperature_regime",
        "fro_regime",
        "graph_family",
        "field_complexity",
        "tau_mode",
        "interaction_complexity",
        "gamma_fro_norm",
        "final_loss",
        "parameter_rmse",
        "field_rmse",
        "field_l2_error",
        "static_field_rmse",
        "tau_rmse",
        "interaction_fro_error",
        "avg_alpha_coef_abs_error",
        "avg_tau_abs_error",
        "avg_interaction_coef_abs_error",
        "beta_abs_error",
        "eta_abs_error",
        "zeta_abs_error",
        "psi_abs_error",
    ]

    rows = []
    for exp in experiments:
        metadata = exp["metadata"]
        config = exp["config"]
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
                "tau_mode": str(metadata.get("tau_mode", "")),
                "interaction_complexity": str(
                    metadata.get("interaction_complexity", "")
                ),
                "gamma_fro_norm": float(metadata.get("gamma_fro_norm", float("nan"))),
                "final_loss": metric(exp, "final_loss"),
                "parameter_rmse": metric(exp, "parameter_rmse"),
                "field_rmse": metric(exp, "field_rmse"),
                "field_l2_error": metric(exp, "field_l2_error"),
                "static_field_rmse": metric(exp, "static_field_rmse"),
                "tau_rmse": metric(exp, "tau_rmse"),
                "interaction_fro_error": metric(exp, "interaction_fro_error"),
                "avg_alpha_coef_abs_error": parameter_block_mean_error(exp, "field"),
                "avg_tau_abs_error": parameter_block_mean_error(exp, "tau"),
                "avg_interaction_coef_abs_error": parameter_block_mean_error(
                    exp, "interaction"
                ),
                "beta_abs_error": scalar_abs_error(exp, "beta"),
                "eta_abs_error": scalar_abs_error(exp, "eta"),
                "zeta_abs_error": scalar_abs_error(exp, "zeta"),
                "psi_abs_error": scalar_abs_error(exp, "psi"),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_factor_summary_csv(report_dir: Path, experiments: list[dict]) -> Path:
    """Write grouped summaries for the main levers."""
    csv_path = report_dir / "parameter_recovery_factor_summary.csv"
    fieldnames = [
        "subset",
        "lever",
        "value",
        "n_experiments",
        "mean_final_loss",
        "mean_parameter_rmse",
        "mean_field_rmse",
        "mean_field_l2_error",
        "mean_static_field_rmse",
        "mean_tau_rmse",
        "mean_interaction_fro_error",
        "mean_alpha_coef_abs_error",
        "mean_tau_abs_error",
        "mean_interaction_coef_abs_error",
        "mean_beta_abs_error",
        "mean_eta_abs_error",
        "mean_zeta_abs_error",
        "mean_psi_abs_error",
    ]
    levers = ["field_complexity", "tau_mode", "N", "T", "temperature_regime", "fro_regime", "interaction_complexity"]
    rows = []
    for lever in levers:
        for value in available_factor_values(experiments, lever):
            summary = summarize_factor(experiments, lever, value)
            rows.append(
                {
                    "subset": "all",
                    "lever": lever,
                    "value": value,
                    "n_experiments": summary["count"],
                    "mean_final_loss": summary["final_loss"],
                    "mean_parameter_rmse": summary["parameter_rmse"],
                    "mean_field_rmse": summary["field_rmse"],
                    "mean_field_l2_error": summary["field_l2_error"],
                    "mean_static_field_rmse": summary["static_field_rmse"],
                    "mean_tau_rmse": summary["tau_rmse"],
                    "mean_interaction_fro_error": summary["interaction_fro_error"],
                    "mean_alpha_coef_abs_error": summary["avg_alpha_coef_abs_error"],
                    "mean_tau_abs_error": summary["avg_tau_abs_error"],
                    "mean_interaction_coef_abs_error": summary["avg_interaction_coef_abs_error"],
                    "mean_beta_abs_error": summary["beta_abs_error"],
                    "mean_eta_abs_error": summary["eta_abs_error"],
                    "mean_zeta_abs_error": summary["zeta_abs_error"],
                    "mean_psi_abs_error": summary["psi_abs_error"],
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def pick_best(experiments: list[dict], metric_name: str) -> dict:
    """Pick the experiment minimizing one metric."""
    return min(experiments, key=lambda exp: metric(exp, metric_name))


def pick_worst(experiments: list[dict], metric_name: str) -> dict:
    """Pick the experiment maximizing one metric."""
    return max(experiments, key=lambda exp: metric(exp, metric_name))


def select_representative_examples(experiments: list[dict]) -> list[tuple[str, dict]]:
    """Choose a few representative experiments without hardcoded descriptors."""
    examples = [("Best full-field recovery", pick_best(experiments, "field_l2_error"))]
    examples.append(("Worst full-field recovery", pick_worst(experiments, "field_l2_error")))

    with_tau = [exp for exp in experiments if any(parameter_block(row["name"]) == "tau" for row in exp["parameter_rows"])]
    if with_tau:
        examples.append(("Best tau recovery", pick_best(with_tau, "tau_rmse")))

    shared_field = [
        exp
        for exp in experiments
        if str(exp["metadata"].get("field_complexity", "")) == "shared_feature_field"
    ]
    if shared_field:
        examples.append(("Representative shared-field case", pick_best(shared_field, "field_l2_error")))

    deduped: list[tuple[str, dict]] = []
    seen = set()
    for title, exp in examples:
        folder = str(exp["folder"])
        if folder in seen:
            continue
        seen.add(folder)
        deduped.append((title, exp))
    return deduped


def grouped_section(handle, experiments: list[dict], lever: str, title: str) -> None:
    """Write one factor-summary section if the lever varies across experiments."""
    values = available_factor_values(experiments, lever)
    if len(values) <= 1:
        return

    rows = []
    for value in values:
        summary = summarize_factor(experiments, lever, value)
        rows.append(
            [
                value,
                str(summary["count"]),
                format_metric(summary["field_l2_error"]),
                format_metric(summary["field_rmse"]),
                format_metric(summary["static_field_rmse"]),
                format_metric(summary["tau_rmse"]),
                format_metric(summary["parameter_rmse"]),
                format_metric(summary["interaction_fro_error"]),
                format_metric(summary["avg_alpha_coef_abs_error"]),
                format_metric(summary["avg_tau_abs_error"]),
            ]
        )

    handle.write(f"## {title}\n\n")
    handle.write(
        render_table(
            [
                "value",
                "experiments",
                "field_l2_error",
                "field_rmse",
                "static_field_rmse",
                "tau_rmse",
                "parameter_rmse",
                "interaction_fro_error",
                "avg alpha abs err",
                "avg tau abs err",
            ],
            rows,
        )
        + "\n\n"
    )


def split_parameter_rows(exp: dict) -> tuple[list[list[str]], list[list[str]]]:
    """Split non-tau and tau parameter rows into display tables."""
    base_rows: list[list[str]] = []
    tau_rows: list[list[str]] = []
    for row in exp["parameter_rows"]:
        formatted = [
            row["name"],
            f"{float(row['true']):.4f}",
            f"{float(row['estimate']):.4f}",
            f"{abs(float(row['estimate']) - float(row['true'])):.4f}",
            f"{float(row['squared_error']):.4f}",
        ]
        if parameter_block(row["name"]) == "tau":
            tau_rows.append(formatted)
        else:
            base_rows.append(formatted)
    return base_rows, tau_rows


def write_report(report_path: Path, experiments: list[dict]) -> None:
    """Write a detailed markdown report centered on external-field recovery."""
    if not experiments:
        raise ValueError("No experiments were found in the manifest.")

    best_field = pick_best(experiments, "field_l2_error")
    worst_field = pick_worst(experiments, "field_l2_error")
    best_tau = pick_best(experiments, "tau_rmse")
    best_static = pick_best(experiments, "static_field_rmse")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# External Field Recovery Report\n\n")
        handle.write(
            "This report focuses on recovery of the realized external field "
            "`alpha_i + tau_t`, rather than only on coefficient recovery or loss.\n\n"
        )
        handle.write(
            "Artifacts written alongside this report:\n\n"
            "- `parameter_recovery_parameter_rows.csv`: one row per fitted parameter\n"
            "- `parameter_recovery_experiment_summary.csv`: one row per experiment\n"
            "- `parameter_recovery_factor_summary.csv`: grouped summaries by lever\n\n"
        )
        handle.write("## Reported Statistics\n\n")
        handle.write(
            "- `field_l2_error`: L2 error of the full realized external field matrix `(alpha_i + tau_t)` flattened over all `(t, i)` entries\n"
            "- `field_rmse`: RMSE of the full realized external field matrix `(alpha_i + tau_t)`\n"
            "- `static_field_rmse`: RMSE of the unit-specific static field `alpha_i`\n"
            "- `tau_rmse`: RMSE of the time-specific field `tau_t`\n"
            "- `interaction_fro_error`: Frobenius error of the realized interaction matrix\n"
            "- `parameter_rmse`: RMSE of the full optimizer parameter vector\n"
            "- `final_loss`: final conditional pseudo-NLL\n\n"
        )
        handle.write("## Main Takeaways\n\n")
        handle.write(
            f"- Best full external-field recovery by L2 error is "
            f"`{best_field['metadata'].get('descriptor', best_field['folder'].name)}` "
            f"with `field_l2_error = {metric(best_field, 'field_l2_error'):.4f}` and "
            f"`field_rmse = {metric(best_field, 'field_rmse'):.4f}`.\n"
        )
        handle.write(
            f"- Worst full external-field recovery by L2 error is "
            f"`{worst_field['metadata'].get('descriptor', worst_field['folder'].name)}` "
            f"with `field_l2_error = {metric(worst_field, 'field_l2_error'):.4f}`.\n"
        )
        handle.write(
            f"- Best time-specific recovery is "
            f"`{best_tau['metadata'].get('descriptor', best_tau['folder'].name)}` "
            f"with `tau_rmse = {metric(best_tau, 'tau_rmse'):.4f}`.\n"
        )
        handle.write(
            f"- Best unit-specific static-field recovery is "
            f"`{best_static['metadata'].get('descriptor', best_static['folder'].name)}` "
            f"with `static_field_rmse = {metric(best_static, 'static_field_rmse'):.4f}`.\n\n"
        )

        grouped_section(handle, experiments, "field_complexity", "Field Complexity")
        grouped_section(handle, experiments, "tau_mode", "Tau Mode")
        grouped_section(handle, experiments, "N", "N")
        grouped_section(handle, experiments, "T", "T")
        grouped_section(handle, experiments, "temperature_regime", "Temperature Regime")
        grouped_section(handle, experiments, "fro_regime", "Graph Frobenius Regime")
        grouped_section(handle, experiments, "interaction_complexity", "Interaction Complexity")

        handle.write("## Experiment Summary\n\n")
        summary_rows = []
        for exp in experiments:
            metadata = exp["metadata"]
            summary_rows.append(
                [
                    str(metadata.get("descriptor", exp["folder"].name)),
                    str(getattr(exp["config"].global_params, "N")),
                    str(getattr(exp["config"].global_params, "T")),
                    str(metadata.get("field_complexity", "")),
                    str(metadata.get("tau_mode", "")),
                    format_metric(metric(exp, "field_l2_error")),
                    format_metric(metric(exp, "field_rmse")),
                    format_metric(metric(exp, "static_field_rmse")),
                    format_metric(metric(exp, "tau_rmse")),
                    format_metric(metric(exp, "parameter_rmse")),
                ]
            )
        handle.write(
            render_table(
                [
                    "experiment",
                    "N",
                    "T",
                    "field complexity",
                    "tau mode",
                    "field_l2_error",
                    "field_rmse",
                    "static_field_rmse",
                    "tau_rmse",
                    "parameter_rmse",
                ],
                summary_rows,
            )
            + "\n\n"
        )

        handle.write("## Representative True vs Estimated Tables\n\n")
        for title, exp in select_representative_examples(experiments):
            metadata = exp["metadata"]
            descriptor = str(metadata.get("descriptor", exp["folder"].name))
            handle.write(f"### {title}\n\n")
            handle.write(
                f"- `{descriptor}`\n"
                f"- field_l2_error = {metric(exp, 'field_l2_error'):.4f}\n"
                f"- field_rmse = {metric(exp, 'field_rmse'):.4f}\n"
                f"- static_field_rmse = {metric(exp, 'static_field_rmse'):.4f}\n"
                f"- tau_rmse = {metric(exp, 'tau_rmse'):.4f}\n"
                f"- interaction_fro_error = {metric(exp, 'interaction_fro_error'):.4f}\n"
                f"- parameter_rmse = {metric(exp, 'parameter_rmse'):.4f}\n\n"
            )

            base_rows, tau_rows = split_parameter_rows(exp)
            if base_rows:
                handle.write(
                    render_table(
                        ["parameter", "true", "estimate", "abs error", "squared error"],
                        base_rows,
                    )
                    + "\n\n"
                )
            if tau_rows:
                tau_abs_errors = [
                    abs(float(row["estimate"]) - float(row["true"]))
                    for row in exp["parameter_rows"]
                    if parameter_block(row["name"]) == "tau"
                ]
                handle.write(
                    f"Tau block summary: mean abs error = {mean(tau_abs_errors):.4f}, "
                    f"median abs error = {median(tau_abs_errors):.4f}, "
                    f"max abs error = {max(tau_abs_errors):.4f}.\n\n"
                )
                handle.write(
                    render_table(
                        ["tau entry", "true", "estimate", "abs error", "squared error"],
                        tau_rows,
                    )
                    + "\n\n"
                )


def main() -> None:
    """Entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate a detailed external-field recovery report from a manifest."
    )
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--report_stem", required=True, type=str)
    args = parser.parse_args()

    report_stem = Path(args.report_stem)
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    report_dir = report_stem.parent

    experiments = [load_experiment(folder) for folder in read_manifest(Path(args.manifest))]
    write_parameter_rows_csv(report_dir, experiments)
    write_experiment_summary_csv(report_dir, experiments)
    write_factor_summary_csv(report_dir, experiments)
    write_report(Path(f"{args.report_stem}.md"), experiments)


if __name__ == "__main__":
    main()
