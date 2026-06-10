"""Refresh posterior-predictive reports and optionally generate plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.t8_posterior_predictive_plotting import (
    write_posterior_predictive_plot_reports,
)
from utils.t8_posterior_predictive_reporting import (
    refresh_and_write_posterior_predictive_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh posterior-predictive reports and optionally generate time-series plots."
        )
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--plot_posterior_predictive", action="store_true")
    parser.add_argument(
        "--plot_intervention_summaries",
        "--plot_counterfactual_summaries",
        dest="plot_intervention_summaries",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="",
    )
    return parser.parse_args(argv)


def run_report_posterior_predictive(
    generation_manifest_path: str | Path,
    *,
    plot_posterior_predictive: bool = False,
    plot_intervention_summaries: bool = False,
    output_dir_name: str = "",
) -> dict[str, object]:
    outputs = refresh_and_write_posterior_predictive_reports(generation_manifest_path)
    if plot_posterior_predictive or plot_intervention_summaries:
        outputs["plot_outputs"] = write_posterior_predictive_plot_reports(
            outputs["manifest_path"],
            plot_posterior_predictive=plot_posterior_predictive,
            plot_intervention_summaries=plot_intervention_summaries,
            output_dir_name=output_dir_name,
        )
    return outputs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_report_posterior_predictive(
        args.generation_manifest_path,
        plot_posterior_predictive=args.plot_posterior_predictive,
        plot_intervention_summaries=args.plot_intervention_summaries,
        output_dir_name=args.output_dir_name,
    )
    print(f"Posterior predictive manifest: {outputs['manifest_path']}")
    print(f"Manifest rows: {outputs['num_manifest_rows']}")
    if "winners_csv" in outputs:
        print(f"Posterior predictive winners: {outputs['winners_csv']}")
    plot_outputs = outputs.get("plot_outputs")
    if isinstance(plot_outputs, dict):
        for message in plot_outputs.get("messages", []):
            print(message)
        for plot_path in plot_outputs.get("posterior_predictive_plot_paths", []):
            print(f"Posterior predictive plot: {plot_path}")
        for plot_path in plot_outputs.get("counterfactual_summary_plot_paths", []):
            print(f"Counterfactual summary plot: {plot_path}")
        for plot_path in plot_outputs.get("intervention_share_plot_paths", []):
            print(f"Intervention share plot: {plot_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
