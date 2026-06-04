"""Build a paper-ready FINAL synthetic MPLE recovery table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from utils.io_utils import write_csv
from pipeline_specs import read_csv_manifest

EXPECTED_EXPERIMENT_COUNT = 10
EXPECTED_VARIANTS = (
    "alternating_rank_3_uv_5_e2",
    "alternating_rank_3_uv_5_e2_xi_zero",
    "base",
)

SUMMARY_COLUMNS = [
    "row_key",
    "row_label",
    "beta_mean",
    "beta_se",
    "xi_mean",
    "xi_se",
    "eta_mean",
    "eta_se",
    "a_rmse_mean",
    "a_rmse_se",
    "gte_mean",
    "gte_se",
    "num_experiments",
]

ROW_SPECS = (
    {
        "row_key": "truth",
        "row_label": r"$\theta^\ast$",
        "variant_slug": None,
        "counterfactual_dir": "truth",
        "bold": False,
        "show_parameter_error_bars": False,
    },
    {
        "row_key": "theta_hat",
        "row_label": r"$\boldsymbol{\hat{\theta}}$",
        "variant_slug": "alternating_rank_3_uv_5_e2",
        "counterfactual_dir": "fit_alternating_rank_3_uv_5_e2",
        "bold": True,
        "show_parameter_error_bars": True,
    },
    {
        "row_key": "theta_a_zero",
        "row_label": r"$\hat{\theta}_{A=0}$",
        "variant_slug": "base",
        "counterfactual_dir": "fit_base",
        "bold": False,
        "show_parameter_error_bars": True,
    },
    {
        "row_key": "theta_xi_zero",
        "row_label": r"$\theta_{\xi=0}$",
        "variant_slug": "alternating_rank_3_uv_5_e2_xi_zero",
        "counterfactual_dir": "fit_alternating_rank_3_uv_5_e2_xi_zero",
        "bold": False,
        "show_parameter_error_bars": True,
    },
)


def _resolve_experiment_root(
    manifest_row: dict[str, str],
    *,
    manifest_root: Path,
) -> Path:
    experiment_slug = str(
        manifest_row.get("experiment_slug") or manifest_row.get("experiment_name") or ""
    ).strip()
    local_candidate = manifest_root / experiment_slug
    if experiment_slug and local_candidate.exists():
        return local_candidate
    experiment_path = str(manifest_row.get("experiment_path", "")).strip()
    if experiment_path:
        path_candidate = Path(experiment_path)
        if path_candidate.exists():
            return path_candidate
    raise FileNotFoundError(
        f"Could not resolve a local experiment directory for '{experiment_slug}'."
    )


def _read_summary_value(summary_path: Path, name: str) -> float:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("name") == name and row.get("estimate") not in (None, ""):
                return float(row["estimate"])
    raise ValueError(f"Could not find '{name}' in {summary_path}.")


def _read_truth_scalars(experiment_root: Path) -> dict[str, float]:
    config = OmegaConf.load(experiment_root / "generation_realized_config.yaml")
    return {
        "beta": float(config.estimation_params.beta),
        "xi": float(config.estimation_params.xi),
        "eta": float(config.estimation_params.eta),
    }


def _read_counterfactual_mean(counterfactual_summary_path: Path) -> float:
    with counterfactual_summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("statistic") == "overall_mean_magnetization":
                return float(row["sample_mean"])
    raise ValueError(
        "Could not find 'overall_mean_magnetization' in "
        f"{counterfactual_summary_path}."
    )


def _read_counterfactual_overall_stats(
    counterfactual_summary_path: Path,
) -> dict[str, float]:
    with counterfactual_summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("statistic") != "overall_mean_magnetization":
                continue
            sample_mean = row.get("sample_mean")
            sample_std = row.get("sample_std")
            num_finite_samples = row.get("num_finite_samples")
            if sample_mean in (None, ""):
                raise ValueError(
                    "Missing 'sample_mean' for 'overall_mean_magnetization' in "
                    f"{counterfactual_summary_path}."
                )
            if sample_std in (None, ""):
                raise ValueError(
                    "Missing 'sample_std' for 'overall_mean_magnetization' in "
                    f"{counterfactual_summary_path}."
                )
            if num_finite_samples in (None, ""):
                raise ValueError(
                    "Missing 'num_finite_samples' for 'overall_mean_magnetization' in "
                    f"{counterfactual_summary_path}."
                )
            return {
                "sample_mean": float(sample_mean),
                "sample_std": float(sample_std),
                "num_finite_samples": float(num_finite_samples),
            }
    raise ValueError(
        "Could not find 'overall_mean_magnetization' in "
        f"{counterfactual_summary_path}."
    )


def _sample_mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Expected at least one value when computing a sample summary.")
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return mean_value, 0.0
    sample_std = float(np.std(values, ddof=1))
    return mean_value, sample_std / float(np.sqrt(len(values)))


def _sample_variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _collect_experiment_groups(manifest_path: str | Path) -> dict[str, dict[str, object]]:
    manifest_root = Path(manifest_path).resolve().parent
    grouped: dict[str, dict[str, object]] = {}
    for row in read_csv_manifest(manifest_path):
        if str(row.get("status", "")).strip().lower() != "completed":
            continue
        experiment_root = _resolve_experiment_root(row, manifest_root=manifest_root)
        experiment_key = experiment_root.name
        variant_slug = str(row.get("variant_slug", "")).strip()
        if variant_slug not in EXPECTED_VARIANTS:
            continue
        group = grouped.setdefault(
            experiment_key,
            {
                "experiment_root": experiment_root,
                "variants": {},
            },
        )
        variants = group["variants"]
        if variant_slug in variants:
            raise ValueError(
                f"Found duplicate manifest entries for {experiment_key} / {variant_slug}."
            )
        variants[variant_slug] = row
    if len(grouped) != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_EXPERIMENT_COUNT} FINAL synthetic experiments, found {len(grouped)}."
        )
    for experiment_key, group in grouped.items():
        variants = group["variants"]
        missing = sorted(set(EXPECTED_VARIANTS) - set(variants))
        if missing:
            raise ValueError(
                f"Experiment '{experiment_key}' is missing expected variants: {missing}."
            )
    return grouped


def _collect_row_values(
    experiment_groups: dict[str, dict[str, object]],
    row_spec: dict[str, object],
) -> dict[str, object]:
    beta_values: list[float] = []
    xi_values: list[float] = []
    eta_values: list[float] = []
    a_rmse_values: list[float] = []
    gte_values: list[float] = []
    gte_mc_variances: list[float] = []

    variant_slug = row_spec["variant_slug"]
    counterfactual_dir = str(row_spec["counterfactual_dir"])

    for group in experiment_groups.values():
        experiment_root = Path(group["experiment_root"])
        if variant_slug is None:
            scalars = _read_truth_scalars(experiment_root)
            a_rmse_values.append(0.0)
        else:
            summary_path = experiment_root / "fits" / str(variant_slug) / "mple_summary.csv"
            scalars = {
                "beta": _read_summary_value(summary_path, "beta"),
                "xi": _read_summary_value(summary_path, "xi"),
                "eta": _read_summary_value(summary_path, "eta"),
            }
            a_rmse_values.append(_read_summary_value(summary_path, "field_rmse"))
        beta_values.append(scalars["beta"])
        xi_values.append(scalars["xi"])
        eta_values.append(scalars["eta"])

        all_ones_summary = _read_counterfactual_overall_stats(
            experiment_root
            / "counterfactual"
            / counterfactual_dir
            / "all_ones"
            / "default"
            / "counterfactual_summary.csv"
        )
        all_zeros_summary = _read_counterfactual_overall_stats(
            experiment_root
            / "counterfactual"
            / counterfactual_dir
            / "all_zeros"
            / "default"
            / "counterfactual_summary.csv"
        )
        gte_values.append(
            float(all_ones_summary["sample_mean"]) - float(all_zeros_summary["sample_mean"])
        )
        gte_mc_variances.append(
            float(all_ones_summary["sample_std"]) ** 2
            / float(all_ones_summary["num_finite_samples"])
            + float(all_zeros_summary["sample_std"]) ** 2
            / float(all_zeros_summary["num_finite_samples"])
        )

    beta_mean, beta_se = _sample_mean_se(beta_values)
    xi_mean, xi_se = _sample_mean_se(xi_values)
    eta_mean, eta_se = _sample_mean_se(eta_values)
    a_rmse_mean, a_rmse_se = _sample_mean_se(a_rmse_values)
    gte_mean = float(np.mean(gte_values))
    num_experiments = len(experiment_groups)
    gte_variance = (
        _sample_variance(gte_values) / num_experiments
        + float(np.mean(gte_mc_variances)) / num_experiments
    )
    gte_se = float(np.sqrt(gte_variance))
    return {
        "row_key": row_spec["row_key"],
        "row_label": row_spec["row_label"],
        "bold": bool(row_spec["bold"]),
        "show_parameter_error_bars": bool(row_spec["show_parameter_error_bars"]),
        "beta_mean": beta_mean,
        "beta_se": beta_se,
        "xi_mean": xi_mean,
        "xi_se": xi_se,
        "eta_mean": eta_mean,
        "eta_se": eta_se,
        "a_rmse_mean": a_rmse_mean,
        "a_rmse_se": a_rmse_se,
        "gte_mean": gte_mean,
        "gte_se": gte_se,
        "num_experiments": num_experiments,
    }


def build_summary_rows(manifest_path: str | Path) -> list[dict[str, object]]:
    experiment_groups = _collect_experiment_groups(manifest_path)
    return [_collect_row_values(experiment_groups, row_spec) for row_spec in ROW_SPECS]


def _format_value(value: float) -> str:
    return f"{value:.3f}"


def _format_display(mean: float, std: float, *, show_error_bar: bool, bold: bool) -> str:
    rendered = _format_value(mean)
    if show_error_bar:
        rendered = f"{rendered} \\pm {_format_value(std)}"
    if bold:
        rendered = rf"\boldsymbol{{{rendered}}}"
    return f"${rendered}$"


def _render_latex_row(row: dict[str, object]) -> str:
    label = str(row["row_label"])
    bold = bool(row["bold"])
    show_parameter_error_bars = bool(row["show_parameter_error_bars"])
    cells = [
        _format_display(
            float(row["beta_mean"]),
            float(row["beta_se"]),
            show_error_bar=show_parameter_error_bars,
            bold=bold,
        ),
        _format_display(
            float(row["xi_mean"]),
            float(row["xi_se"]),
            show_error_bar=show_parameter_error_bars,
            bold=bold,
        ),
        _format_display(
            float(row["eta_mean"]),
            float(row["eta_se"]),
            show_error_bar=show_parameter_error_bars,
            bold=bold,
        ),
        _format_display(
            float(row["a_rmse_mean"]),
            float(row["a_rmse_se"]),
            show_error_bar=True,
            bold=bold,
        ),
        _format_display(
            float(row["gte_mean"]),
            float(row["gte_se"]),
            show_error_bar=True,
            bold=bold,
        ),
    ]
    return (
        f"        {label}\n"
        f"            & {cells[0]}\n"
        f"            & {cells[1]}\n"
        f"            & {cells[2]}\n"
        f"            & {cells[3]}\n"
        f"            & {cells[4]} \\\\\n"
    )


def render_latex_table(rows: list[dict[str, object]]) -> str:
    row_blocks = "\n".join(_render_latex_row(row) for row in rows)
    return (
        "\\begin{table}[hbt]\n"
        "    \\centering\n"
        "    \\begin{tabular}{lccccc}\n"
        "        \\hline\n"
        "        & $\\beta$ & $\\xi$ & $\\eta$ & $A$ RMSE & $\\GTE({\\bf 1},{\\bf -1})$ \\\\\n"
        "        \\hline\n\n"
        f"{row_blocks}\n"
        "        \\hline\n"
        "    \\end{tabular}\n"
        "    \\caption{Parameter, latent-field, and GTE recovery for FINAL synthetic MPLE fits. "
        "Entries report means across the 10 synthetic runs with standard errors. "
        "The reported GTE uses the mean outcome under the all_ones intervention "
        "minus the mean outcome under the all_zeroes intervention. For the latent field $A$, the "
        "reported value is the RMSE.}\n"
        "    \\label{tab:final_synth_parameter_gate_recovery}\n"
        "\\end{table}\n"
    )


def write_outputs(
    manifest_path: str | Path,
    *,
    output_tex: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    manifest_root = Path(manifest_path).resolve().parent
    tex_path = (
        Path(output_tex)
        if output_tex is not None
        else manifest_root / "parameter_gate_recovery_table.tex"
    )
    csv_path = (
        Path(output_csv)
        if output_csv is not None
        else manifest_root / "parameter_gate_recovery_table.csv"
    )

    rows = build_summary_rows(manifest_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(render_latex_table(rows), encoding="utf-8")
    write_csv(csv_path, rows, SUMMARY_COLUMNS)
    return {
        "tex_path": str(tex_path),
        "csv_path": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the FINAL synthetic MPLE recovery LaTeX table and CSV."
    )
    parser.add_argument("--fit-manifest", required=True, type=str)
    parser.add_argument("--output-tex", default=None, type=str)
    parser.add_argument("--output-csv", default=None, type=str)
    args = parser.parse_args()

    outputs = write_outputs(
        args.fit_manifest,
        output_tex=args.output_tex,
        output_csv=args.output_csv,
    )
    print(f"Wrote LaTeX table to {outputs['tex_path']}")
    print(f"Wrote CSV summary to {outputs['csv_path']}")


if __name__ == "__main__":
    main()
