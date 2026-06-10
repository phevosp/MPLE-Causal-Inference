"""Cross-trial aggregation for replicated experiment cohorts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.t0_csv_utils import read_csv_rows, write_csv, write_csv_rows
from utils.t0_orcd_path_remap import resolve_orcd_local_path
from utils.t8_output_writers import _as_float
from utils.t8_parameter_recovery_reporting import read_summary_entries, scalar_value


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
WARNING_COLUMNS = [
    "cohort_label",
    "experiment_slug",
    "source_type",
    "source_name",
    "source_slug",
    "run_name",
    "run_slug",
    "statistic_name",
    "warning_kind",
    "artifact_path",
    "message",
]
_GTE_COLUMN = "overall_mean_magnetization_mean"
_FIT_TRIAL_STATISTIC_NAME_MAP = {
    "beta_estimate": "beta",
    "xi_estimate": "xi",
    "eta_estimate": "eta",
    "field_rmse": "field_rmse",
}


def _warning_row(
    *,
    cohort_label: str,
    experiment_slug: str = "",
    source_type: str = "",
    source_name: str = "",
    source_slug: str = "",
    run_name: str = "",
    run_slug: str = "",
    statistic_name: str = "",
    warning_kind: str,
    artifact_path: str | Path | None = None,
    message: str,
) -> dict[str, object]:
    artifact_text = "" if artifact_path in (None, "") else str(artifact_path)
    return {
        "cohort_label": str(cohort_label),
        "experiment_slug": str(experiment_slug),
        "source_type": str(source_type),
        "source_name": str(source_name),
        "source_slug": str(source_slug),
        "run_name": str(run_name),
        "run_slug": str(run_slug),
        "statistic_name": str(statistic_name),
        "warning_kind": str(warning_kind),
        "artifact_path": artifact_text,
        "message": str(message),
    }


def _cohort_rows(
    generation_manifest_path: str | Path,
    *,
    cohort_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for manifest_row in read_csv_rows(generation_manifest_path):
        experiment_path = str(manifest_row.get("experiment_path", "")).strip()
        experiment_root: Path | None = None
        if experiment_path:
            try:
                experiment_root = resolve_orcd_local_path(experiment_path)
            except FileNotFoundError as exc:
                warnings.append(
                    _warning_row(
                        cohort_label=str(cohort_label),
                        experiment_slug=str(manifest_row.get("experiment_slug", "")),
                        warning_kind="missing_experiment_path",
                        artifact_path=experiment_path,
                        message=str(exc),
                    )
                )
        else:
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_label),
                    experiment_slug=str(manifest_row.get("experiment_slug", "")),
                    warning_kind="missing_experiment_path",
                    artifact_path="",
                    message="Generation manifest row is missing experiment_path.",
                )
            )
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
    return rows, warnings


def _fit_source_slug(variant_slug: object) -> str:
    return f"fit_{str(variant_slug)}"


def _parameter_trial_rows(
    fit_manifest_path: str | Path,
    cohort_by_slug: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for manifest_row in read_csv_rows(fit_manifest_path):
        experiment_slug = str(
            manifest_row.get("experiment_slug", manifest_row.get("experiment_name", ""))
        )
        cohort_row = cohort_by_slug.get(experiment_slug)
        if cohort_row is None:
            continue
        source_name = str(manifest_row.get("variant_name", "")).strip()
        source_slug = _fit_source_slug(manifest_row.get("variant_slug", ""))
        fit_path = str(manifest_row.get("fit_path", "")).strip()
        if not fit_path:
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_row["cohort_label"]),
                    experiment_slug=experiment_slug,
                    source_type="fit",
                    source_name=source_name,
                    source_slug=source_slug,
                    warning_kind="missing_fit_path",
                    artifact_path="",
                    message="Fit manifest row is missing fit_path.",
                )
            )
            continue
        try:
            fit_root = resolve_orcd_local_path(fit_path)
        except FileNotFoundError as exc:
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_row["cohort_label"]),
                    experiment_slug=experiment_slug,
                    source_type="fit",
                    source_name=source_name,
                    source_slug=source_slug,
                    warning_kind="missing_fit_path",
                    artifact_path=fit_path,
                    message=str(exc),
                )
            )
            continue

        summary_path = fit_root / "mple_summary.csv"
        if not summary_path.exists():
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_row["cohort_label"]),
                    experiment_slug=experiment_slug,
                    source_type="fit",
                    source_name=source_name,
                    source_slug=source_slug,
                    warning_kind="missing_fit_summary",
                    artifact_path=summary_path,
                    message=f"Missing fit summary at {summary_path}.",
                )
            )
            continue
        try:
            summary_entries = read_summary_entries(summary_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_row["cohort_label"]),
                    experiment_slug=experiment_slug,
                    source_type="fit",
                    source_name=source_name,
                    source_slug=source_slug,
                    warning_kind="unreadable_fit_summary",
                    artifact_path=summary_path,
                    message=str(exc),
                )
            )
            continue

        for statistic_name, summary_key in _FIT_TRIAL_STATISTIC_NAME_MAP.items():
            statistic_value = scalar_value(summary_entries, summary_key, "estimate")
            if statistic_value is None:
                warnings.append(
                    _warning_row(
                        cohort_label=str(cohort_row["cohort_label"]),
                        experiment_slug=experiment_slug,
                        source_type="fit",
                        source_name=source_name,
                        source_slug=source_slug,
                        statistic_name=statistic_name,
                        warning_kind="missing_fit_statistic",
                        artifact_path=summary_path,
                        message=(
                            f"Fit summary at {summary_path} is missing a numeric "
                            f"value for {statistic_name}."
                        ),
                    )
                )
                continue
            rows.append(
                {
                    "cohort_label": cohort_row["cohort_label"],
                    "experiment_name": cohort_row["experiment_name"],
                    "experiment_slug": cohort_row["experiment_slug"],
                    "descriptor": cohort_row["descriptor"],
                    "source_type": "fit",
                    "source_name": source_name,
                    "source_slug": source_slug,
                    "run_name": "",
                    "run_slug": "",
                    "statistic_name": statistic_name,
                    "statistic_value": float(statistic_value),
                }
            )
    return rows, warnings


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


def _read_intervention_rows_by_key(
    summary_path: Path,
    *,
    cohort_label: str,
    experiment_slug: str,
    warning_kind: str,
    unreadable_warning_kind: str,
) -> tuple[dict[tuple[str, str, str, str, str], dict[str, str]] | None, list[dict[str, object]]]:
    if not summary_path.exists():
        return None, [
            _warning_row(
                cohort_label=cohort_label,
                experiment_slug=experiment_slug,
                warning_kind=warning_kind,
                artifact_path=summary_path,
                message=f"Missing intervention summary at {summary_path}.",
            )
        ]
    try:
        return _intervention_rows_by_key(summary_path), []
    except Exception as exc:  # noqa: BLE001
        return None, [
            _warning_row(
                cohort_label=cohort_label,
                experiment_slug=experiment_slug,
                warning_kind=unreadable_warning_kind,
                artifact_path=summary_path,
                message=str(exc),
            )
        ]


def _gte_trial_rows(
    cohort_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for cohort_row in cohort_rows:
        experiment_root_value = cohort_row.get("experiment_root")
        if experiment_root_value in (None, ""):
            continue
        experiment_root = Path(str(experiment_root_value))
        summary_root = experiment_root / "intervention_summaries"
        all_rows, current_warnings = _read_intervention_rows_by_key(
            summary_root / "all_intervention.csv",
            cohort_label=str(cohort_row["cohort_label"]),
            experiment_slug=str(cohort_row["experiment_slug"]),
            warning_kind="missing_all_intervention_summary",
            unreadable_warning_kind="unreadable_all_intervention_summary",
        )
        warnings.extend(current_warnings)
        no_rows, current_warnings = _read_intervention_rows_by_key(
            summary_root / "no_intervention.csv",
            cohort_label=str(cohort_row["cohort_label"]),
            experiment_slug=str(cohort_row["experiment_slug"]),
            warning_kind="missing_no_intervention_summary",
            unreadable_warning_kind="unreadable_no_intervention_summary",
        )
        warnings.extend(current_warnings)
        if all_rows is None or no_rows is None:
            continue
        if set(all_rows) != set(no_rows):
            missing_from_no = sorted(set(all_rows) - set(no_rows))
            missing_from_all = sorted(set(no_rows) - set(all_rows))
            warnings.append(
                _warning_row(
                    cohort_label=str(cohort_row["cohort_label"]),
                    experiment_slug=str(cohort_row["experiment_slug"]),
                    warning_kind="intervention_summary_mismatch",
                    artifact_path=summary_root,
                    message=(
                        f"Intervention summary mismatch for {experiment_root}. "
                        f"Missing from no_intervention: {missing_from_no}. "
                        f"Missing from all_intervention: {missing_from_all}."
                    ),
                )
            )
            continue
        for key in sorted(all_rows):
            all_row = all_rows[key]
            no_row = no_rows[key]
            all_value = _as_float(all_row.get(_GTE_COLUMN))
            no_value = _as_float(no_row.get(_GTE_COLUMN))
            if all_value is None or no_value is None:
                source_type, source_name, source_slug, run_name, run_slug = key
                warnings.append(
                    _warning_row(
                        cohort_label=str(cohort_row["cohort_label"]),
                        experiment_slug=str(cohort_row["experiment_slug"]),
                        source_type=source_type,
                        source_name=source_name,
                        source_slug=source_slug,
                        run_name=run_name,
                        run_slug=run_slug,
                        statistic_name="gte_overall_mean_magnetization",
                        warning_kind="missing_gte_statistic",
                        artifact_path=summary_root,
                        message=(
                            f"Missing numeric {_GTE_COLUMN} value in intervention "
                            f"summaries for source '{source_slug}' run '{run_slug}'."
                        ),
                    )
                )
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
    return rows, warnings


def collect_trial_statistics_with_warnings(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    *,
    cohort_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    cohort_rows, warnings = _cohort_rows(
        generation_manifest_path,
        cohort_label=cohort_label,
    )
    cohort_by_slug = {
        str(row["experiment_slug"]): row
        for row in cohort_rows
    }
    rows, current_warnings = _parameter_trial_rows(fit_manifest_path, cohort_by_slug)
    warnings.extend(current_warnings)
    gte_rows, current_warnings = _gte_trial_rows(cohort_rows)
    warnings.extend(current_warnings)
    rows.extend(gte_rows)
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
    warnings.sort(
        key=lambda row: (
            str(row.get("cohort_label", "")),
            str(row.get("experiment_slug", "")),
            str(row.get("source_type", "")),
            str(row.get("source_name", "")),
            str(row.get("run_slug", "")),
            str(row.get("statistic_name", "")),
            str(row.get("warning_kind", "")),
        )
    )
    return rows, warnings, int(len(cohort_rows))


def _group_trial_rows(
    trial_rows: list[dict[str, object]],
) -> dict[tuple[str, str, str, str, str], list[float]]:
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
    return grouped


def _incomplete_statistic_warnings(
    trial_rows: list[dict[str, object]],
    *,
    expected_num_trials: int,
) -> list[dict[str, object]]:
    if expected_num_trials <= 0:
        return []
    warnings: list[dict[str, object]] = []
    grouped = _group_trial_rows(trial_rows)
    for key in sorted(grouped):
        values = grouped[key]
        if len(values) == int(expected_num_trials):
            continue
        cohort_label, source_type, source_name, source_slug, statistic_name = key
        warnings.append(
            _warning_row(
                cohort_label=cohort_label,
                source_type=source_type,
                source_name=source_name,
                source_slug=source_slug,
                statistic_name=statistic_name,
                warning_kind="incomplete_statistic",
                artifact_path="",
                message=(
                    f"Incomplete statistic '{statistic_name}' for source "
                    f"'{source_slug}': expected {expected_num_trials} trial(s), "
                    f"found {len(values)}."
                ),
            )
        )
    return warnings


def collect_trial_statistics(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    *,
    cohort_label: str,
) -> list[dict[str, object]]:
    rows, _, _ = collect_trial_statistics_with_warnings(
        generation_manifest_path,
        fit_manifest_path,
        cohort_label=cohort_label,
    )
    return rows


def summarize_trial_statistics(
    trial_rows: list[dict[str, object]],
    *,
    expected_num_trials: int | None = None,
) -> list[dict[str, object]]:
    grouped = _group_trial_rows(trial_rows)
    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=float)
        if expected_num_trials is not None and values.size != int(expected_num_trials):
            continue
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
    trial_rows, warnings, expected_num_trials = collect_trial_statistics_with_warnings(
        generation_manifest_path,
        fit_manifest_path,
        cohort_label=cohort_label,
    )
    incomplete_statistic_warnings = _incomplete_statistic_warnings(
        trial_rows,
        expected_num_trials=expected_num_trials,
    )
    warnings.extend(incomplete_statistic_warnings)
    summary_rows = summarize_trial_statistics(
        trial_rows,
        expected_num_trials=expected_num_trials,
    )

    trial_statistics_path = resolved_output_dir / "trial_statistics.csv"
    summary_path = resolved_output_dir / f"{cohort_label}_summary.csv"
    warnings_path = resolved_output_dir / "trial_aggregation_warnings.csv"
    write_csv(trial_statistics_path, trial_rows, TRIAL_STATISTICS_COLUMNS)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(warnings_path, warnings, WARNING_COLUMNS)

    outputs: dict[str, object] = {
        "trial_statistics_path": str(trial_statistics_path),
        "summary_path": str(summary_path),
        "warnings_path": str(warnings_path),
        "num_trial_rows": int(len(trial_rows)),
        "num_summary_rows": int(len(summary_rows)),
        "num_warning_rows": int(len(warnings)),
        "num_complete_summary_rows": int(len(summary_rows)),
        "num_incomplete_summary_groups": int(len(incomplete_statistic_warnings)),
    }
    if write_wide:
        wide_rows = build_trial_summary_wide_rows(summary_rows)
        wide_path = resolved_output_dir / f"{cohort_label}_wide.csv"
        write_csv_rows(wide_path, wide_rows)
        outputs["wide_path"] = str(wide_path)
        outputs["num_wide_rows"] = int(len(wide_rows))
    return outputs
