"""Cross-trial aggregation for replicated experiment cohorts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.t0_csv_utils import read_csv_rows, write_csv, write_csv_rows
from utils.t0_orcd_path_remap import resolve_orcd_local_path
from utils.t8_output_writers import _as_float
from utils.t8_parameter_recovery_reporting import collect_fit_rows


TRIAL_STATISTICS_COLUMNS = [
    "cohort_label",
    "experiment_name",
    "experiment_slug",
    "descriptor",
    "source_type",
    "source_name",
    "source_slug",
    "run_name",
    "run_slug",
    "statistic_name",
    "statistic_value",
]
SUMMARY_COLUMNS = [
    "cohort_label",
    "source_type",
    "source_name",
    "source_slug",
    "statistic_name",
    "num_trials",
    "mean",
    "sample_std",
    "standard_error",
    "q025",
    "q500",
    "q975",
]
_GTE_COLUMN = "overall_mean_magnetization_mean"


def _cohort_rows(
    generation_manifest_path: str | Path,
    *,
    cohort_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in read_csv_rows(generation_manifest_path):
        experiment_root = resolve_orcd_local_path(manifest_row["experiment_path"])
        rows.append(
            {
                "cohort_label": str(cohort_label),
                "experiment_name": manifest_row.get("experiment_name", ""),
                "experiment_slug": manifest_row.get("experiment_slug", ""),
                "descriptor": manifest_row.get("descriptor", ""),
                "experiment_root": experiment_root,
            }
        )
    if not rows:
        raise ValueError(
            f"No rows found in generation manifest {generation_manifest_path}."
        )
    return rows


def _fit_source_slug(variant_slug: object) -> str:
    return f"fit_{str(variant_slug)}"


def _parameter_trial_rows(
    fit_manifest_path: str | Path,
    cohort_by_slug: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fit_row in collect_fit_rows(fit_manifest_path):
        experiment_slug = str(
            fit_row.get("experiment_slug", fit_row.get("experiment_name", ""))
        )
        cohort_row = cohort_by_slug.get(experiment_slug)
        if cohort_row is None:
            continue
        for statistic_name in (
            "beta_estimate",
            "xi_estimate",
            "eta_estimate",
            "field_rmse",
        ):
            statistic_value = _as_float(fit_row.get(statistic_name))
            if statistic_value is None:
                continue
            rows.append(
                {
                    "cohort_label": cohort_row["cohort_label"],
                    "experiment_name": cohort_row["experiment_name"],
                    "experiment_slug": cohort_row["experiment_slug"],
                    "descriptor": cohort_row["descriptor"],
                    "source_type": "fit",
                    "source_name": fit_row.get("variant_name", ""),
                    "source_slug": _fit_source_slug(fit_row.get("variant_slug", "")),
                    "run_name": "",
                    "run_slug": "",
                    "statistic_name": statistic_name,
                    "statistic_value": float(statistic_value),
                }
            )
    return rows


def _intervention_rows_by_key(
    summary_path: Path,
) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    rows = read_csv_rows(summary_path)
    return {
        (
            str(row.get("source_type", "")),
            str(row.get("source_name", "")),
            str(row.get("source_slug", "")),
            str(row.get("run_name", "")),
            str(row.get("run_slug", "")),
        ): row
        for row in rows
    }


def _gte_trial_rows(cohort_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cohort_row in cohort_rows:
        experiment_root = Path(str(cohort_row["experiment_root"]))
        summary_root = experiment_root / "intervention_summaries"
        all_rows = _intervention_rows_by_key(summary_root / "all_intervention.csv")
        no_rows = _intervention_rows_by_key(summary_root / "no_intervention.csv")
        if set(all_rows) != set(no_rows):
            missing_from_no = sorted(set(all_rows) - set(no_rows))
            missing_from_all = sorted(set(no_rows) - set(all_rows))
            raise ValueError(
                f"Intervention summary mismatch for {experiment_root}. "
                f"Missing from no_intervention: {missing_from_no}. "
                f"Missing from all_intervention: {missing_from_all}."
            )
        for key in sorted(all_rows):
            all_row = all_rows[key]
            no_row = no_rows[key]
            all_value = _as_float(all_row.get(_GTE_COLUMN))
            no_value = _as_float(no_row.get(_GTE_COLUMN))
            if all_value is None or no_value is None:
                continue
            source_type, source_name, source_slug, run_name, run_slug = key
            rows.append(
                {
                    "cohort_label": cohort_row["cohort_label"],
                    "experiment_name": cohort_row["experiment_name"],
                    "experiment_slug": cohort_row["experiment_slug"],
                    "descriptor": cohort_row["descriptor"],
                    "source_type": source_type,
                    "source_name": source_name,
                    "source_slug": source_slug,
                    "run_name": run_name,
                    "run_slug": run_slug,
                    "statistic_name": "gte_overall_mean_magnetization",
                    "statistic_value": float(all_value - no_value),
                }
            )
    return rows


def collect_trial_statistics(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    *,
    cohort_label: str,
) -> list[dict[str, object]]:
    cohort_rows = _cohort_rows(generation_manifest_path, cohort_label=cohort_label)
    cohort_by_slug = {
        str(row["experiment_slug"]): row
        for row in cohort_rows
    }
    rows = _parameter_trial_rows(fit_manifest_path, cohort_by_slug)
    rows.extend(_gte_trial_rows(cohort_rows))
    rows.sort(
        key=lambda row: (
            str(row.get("cohort_label", "")),
            str(row.get("experiment_slug", "")),
            str(row.get("source_type", "")),
            str(row.get("source_name", "")),
            str(row.get("run_slug", "")),
            str(row.get("statistic_name", "")),
        )
    )
    return rows


def summarize_trial_statistics(
    trial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in trial_rows:
        key = (
            str(row.get("cohort_label", "")),
            str(row.get("source_type", "")),
            str(row.get("source_name", "")),
            str(row.get("source_slug", "")),
            str(row.get("statistic_name", "")),
        )
        grouped[key].append(float(row["statistic_value"]))

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=float)
        cohort_label, source_type, source_name, source_slug, statistic_name = key
        row: dict[str, object] = {
            "cohort_label": cohort_label,
            "source_type": source_type,
            "source_name": source_name,
            "source_slug": source_slug,
            "statistic_name": statistic_name,
            "num_trials": int(values.size),
            "mean": float(np.mean(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q500": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
        }
        if values.size >= 2:
            sample_std = float(np.std(values, ddof=1))
            row["sample_std"] = sample_std
            row["standard_error"] = float(sample_std / np.sqrt(values.size))
        else:
            row["sample_std"] = ""
            row["standard_error"] = ""
        summary_rows.append(row)
    return summary_rows


def build_trial_summary_wide_rows(
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in summary_rows:
        key = (
            str(row.get("cohort_label", "")),
            str(row.get("source_type", "")),
            str(row.get("source_name", "")),
            str(row.get("source_slug", "")),
        )
        wide_row = grouped.setdefault(
            key,
            {
                "cohort_label": key[0],
                "source_type": key[1],
                "source_name": key[2],
                "source_slug": key[3],
            },
        )
        statistic_name = str(row["statistic_name"])
        for metric_name in (
            "num_trials",
            "mean",
            "sample_std",
            "standard_error",
            "q025",
            "q500",
            "q975",
        ):
            wide_row[f"{statistic_name}_{metric_name}"] = row.get(metric_name, "")
    return [grouped[key] for key in sorted(grouped)]


def write_trial_aggregation_reports(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    *,
    cohort_label: str,
    output_dir: str | Path | None = None,
    write_wide: bool = False,
) -> dict[str, object]:
    generation_manifest_root = Path(generation_manifest_path).resolve().parent
    resolved_output_dir = (
        generation_manifest_root / "trial_aggregation"
        if output_dir is None
        else Path(output_dir)
    )
    trial_rows = collect_trial_statistics(
        generation_manifest_path,
        fit_manifest_path,
        cohort_label=cohort_label,
    )
    summary_rows = summarize_trial_statistics(trial_rows)

    trial_statistics_path = resolved_output_dir / "trial_statistics.csv"
    summary_path = resolved_output_dir / f"{cohort_label}_summary.csv"
    write_csv(trial_statistics_path, trial_rows, TRIAL_STATISTICS_COLUMNS)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)

    outputs: dict[str, object] = {
        "trial_statistics_path": str(trial_statistics_path),
        "summary_path": str(summary_path),
        "num_trial_rows": int(len(trial_rows)),
        "num_summary_rows": int(len(summary_rows)),
    }
    if write_wide:
        wide_rows = build_trial_summary_wide_rows(summary_rows)
        wide_path = resolved_output_dir / f"{cohort_label}_wide.csv"
        write_csv_rows(wide_path, wide_rows)
        outputs["wide_path"] = str(wide_path)
        outputs["num_wide_rows"] = int(len(wide_rows))
    return outputs
