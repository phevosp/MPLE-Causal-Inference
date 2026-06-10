"""Aggregate trial-level statistics across replicated experiment cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.t9_trial_aggregation import write_trial_aggregation_reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate trial-level statistics across replicated experiments."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--fit_manifest_path", required=True, type=str)
    parser.add_argument("--cohort_label", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--write_wide", action="store_true")
    return parser.parse_args(argv)


def run_trial_aggregation(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    *,
    cohort_label: str,
    output_dir: str | Path | None = None,
    write_wide: bool = False,
) -> dict[str, object]:
    return write_trial_aggregation_reports(
        generation_manifest_path,
        fit_manifest_path,
        cohort_label=cohort_label,
        output_dir=output_dir,
        write_wide=write_wide,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir if str(args.output_dir).strip() else None
    outputs = run_trial_aggregation(
        args.generation_manifest_path,
        args.fit_manifest_path,
        cohort_label=args.cohort_label,
        output_dir=output_dir,
        write_wide=bool(args.write_wide),
    )
    print(f"Trial statistics: {outputs['trial_statistics_path']}")
    print(f"Summary: {outputs['summary_path']}")
    print(f"Trial rows: {outputs['num_trial_rows']}")
    print(f"Summary rows: {outputs['num_summary_rows']}")
    print(f"Warnings: {outputs['warnings_path']}")
    print(f"Warning rows: {outputs['num_warning_rows']}")
    if "wide_path" in outputs:
        print(f"Wide summary: {outputs['wide_path']}")
        print(f"Wide rows: {outputs['num_wide_rows']}")
    if int(outputs.get("num_warning_rows", 0)) > 0:
        print(
            "Incomplete summary statistics omitted: "
            f"{outputs['num_incomplete_summary_groups']}. "
            f"See {outputs['warnings_path']} for details."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
