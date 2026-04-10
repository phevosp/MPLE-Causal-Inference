"""Build binary NFL + COVID county outcomes/interventions and diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CASE_OUTCOME_FAMILY,
    CORE_END_DATE,
    CORE_START_DATE,
    DEATH_OUTCOME_FAMILY,
    PROCESSED_DIR,
    SUPPORTED_LAGS,
    intervention_code_from_threshold,
    intervention_label_from_code,
    lag_code_to_steps,
    outcome_code_from_threshold,
    outcome_label_from_code,
    write_readme,
)


PANEL_PATH = PROCESSED_DIR / "nfl_county_weekly_panel.csv.gz"
BINARY_PANEL_PATH = PROCESSED_DIR / "nfl_covid_binary_panel.csv.gz"
DIAGNOSTICS_CSV_PATH = PROCESSED_DIR / "nfl_covid_binary_threshold_diagnostics.csv"
DIAGNOSTICS_MD_PATH = PROCESSED_DIR / "nfl_covid_binary_threshold_diagnostics.md"
SUPPORT_CSV_PATH = PROCESSED_DIR / "nfl_covid_realized_support_summary.csv"
SUPPORT_MD_PATH = PROCESSED_DIR / "nfl_covid_realized_support_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build binary NFL + COVID county outcome and intervention columns."
    )
    parser.add_argument(
        "--outcome_case_rate_thresholds",
        nargs="*",
        type=float,
        help="Case-rate thresholds for outcome binaries, for example 100 200.",
    )
    parser.add_argument(
        "--outcome_death_rate_thresholds",
        nargs="*",
        type=float,
        help="Death-rate thresholds for outcome binaries, for example 1 2.",
    )
    parser.add_argument(
        "--attendance_share_thresholds",
        nargs="+",
        type=float,
        required=True,
        help="Attendance-share thresholds for intervention binaries, for example 0.01 0.05 0.1.",
    )
    return parser.parse_args()


def binary_from_threshold(series: pd.Series, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.where(values >= threshold, 1, -1), index=series.index, dtype="float")
    out[values.isna()] = np.nan
    return out.astype("Int64")


def transition_rate(panel: pd.DataFrame, column: str) -> float:
    ordered = panel.sort_values(["fips", "WeekStartDate"]).copy()
    ordered["prev_value"] = ordered.groupby("fips", sort=False)[column].shift(1)
    valid = ordered[column].notna() & ordered["prev_value"].notna()
    if not valid.any():
        return float("nan")
    return float((ordered.loc[valid, column] != ordered.loc[valid, "prev_value"]).mean())


def summarize_binary(panel: pd.DataFrame, column: str, definition_type: str, label: str) -> dict[str, object]:
    values = panel[column]
    available = values.notna()
    positive = values.eq(1)
    per_county_variety = (
        panel.loc[available, ["fips", column]].groupby("fips")[column].nunique()
        if available.any()
        else pd.Series(dtype=int)
    )
    available_dates = panel.loc[available, "WeekStartDate"]
    return {
        "definition_type": definition_type,
        "column": column,
        "label": label,
        "eligible_rows": int(available.sum()),
        "eligible_share": float(available.mean()),
        "positive_share": float(positive.loc[available].mean()) if available.any() else float("nan"),
        "transition_rate": transition_rate(panel, column),
        "counties_with_both_states": int((per_county_variety == 2).sum()),
        "counties_constant": int((per_county_variety == 1).sum()),
        "eligible_start_date": available_dates.min() if available.any() else pd.NaT,
        "eligible_end_date": available_dates.max() if available.any() else pd.NaT,
    }


def compute_support_summary(panel: pd.DataFrame, outcome_col: str, intervention_col: str, lag_code: str) -> dict[str, object]:
    lag_steps = lag_code_to_steps(lag_code)
    node_order = sorted(panel["fips"].unique().tolist())
    filtered = panel.loc[panel["WeekStartDate"].between(CORE_START_DATE, CORE_END_DATE)].copy()
    filtered = filtered.sort_values(["fips", "WeekStartDate"]).reset_index(drop=True)
    filtered["Outcome_pm1"] = filtered[outcome_col].astype("Int64")
    filtered["Intervention_pm1_raw"] = filtered[intervention_col].astype("Int64")
    filtered["Intervention_pm1"] = (
        filtered.groupby("fips", sort=False)["Intervention_pm1_raw"].shift(lag_steps).astype("Int64")
    )

    eligible = filtered["Outcome_pm1"].notna() & filtered["Intervention_pm1"].notna()
    eligibility_matrix = (
        filtered.assign(eligible=eligible)
        .pivot(index="WeekStartDate", columns="fips", values="eligible")
        .reindex(columns=node_order)
        .sort_index()
    )
    best_support: tuple[int, int, int, int, pd.Series] | None = None
    for start_index in range(len(eligibility_matrix.index)):
        suffix = eligibility_matrix.iloc[start_index:]
        complete_nodes = suffix.all(axis=0)
        node_count = int(complete_nodes.sum())
        if node_count == 0:
            continue
        week_count = int(suffix.shape[0])
        area = int(node_count * week_count)
        candidate = (area, node_count, week_count, -start_index, complete_nodes)
        if best_support is None or candidate[:4] > best_support[:4]:
            best_support = candidate

    if best_support is None:
        return {
            "requested_node_count": int(len(node_order)),
            "requested_calendar_weeks": int(eligibility_matrix.shape[0]),
            "realized_node_count": 0,
            "realized_calendar_weeks": 0,
            "weeks_dropped_due_to_missing_or_lag": int(eligibility_matrix.shape[0]),
            "dropped_node_count": int(len(node_order)),
            "support_selection_rule": "max_complete_suffix_by_node_week_area",
        }

    _, realized_node_count, realized_week_count, neg_start_index, _ = best_support
    realized_start_index = -neg_start_index
    return {
        "requested_node_count": int(len(node_order)),
        "requested_calendar_weeks": int(eligibility_matrix.shape[0]),
        "realized_node_count": int(realized_node_count),
        "realized_calendar_weeks": int(realized_week_count),
        "weeks_dropped_due_to_missing_or_lag": int(eligibility_matrix.shape[0] - realized_week_count),
        "dropped_node_count": int(len(node_order) - realized_node_count),
        "realized_start_date": eligibility_matrix.index[realized_start_index].date().isoformat()
        if len(eligibility_matrix.index)
        else "",
        "support_selection_rule": "max_complete_suffix_by_node_week_area",
    }


def write_markdown_summary(rows: list[dict[str, object]]) -> None:
    headers = [
        "definition_type",
        "column",
        "label",
        "eligible_rows",
        "eligible_share",
        "positive_share",
        "transition_rate",
        "counties_with_both_states",
        "counties_constant",
        "eligible_start_date",
        "eligible_end_date",
    ]
    with DIAGNOSTICS_MD_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# NFL COVID Binary Definitions\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    rendered.append(f"{value:.6f}")
                else:
                    rendered.append("" if pd.isna(value) else str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def write_support_markdown(rows: list[dict[str, object]]) -> None:
    headers = [
        "outcome_code",
        "intervention_code",
        "lag_code",
        "requested_node_count",
        "realized_node_count",
        "requested_calendar_weeks",
        "realized_calendar_weeks",
        "weeks_dropped_due_to_missing_or_lag",
        "dropped_node_count",
        "realized_start_date",
    ]
    with SUPPORT_MD_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# NFL COVID Realized Support Summary\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    rendered.append(f"{value:.6f}")
                else:
                    rendered.append("" if pd.isna(value) else str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    args = parse_args()
    case_thresholds = sorted(set(args.outcome_case_rate_thresholds or []))
    death_thresholds = sorted(set(args.outcome_death_rate_thresholds or []))
    if not case_thresholds and not death_thresholds:
        raise ValueError("Provide at least one outcome threshold via --outcome_case_rate_thresholds and/or --outcome_death_rate_thresholds.")
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PANEL_PATH.name}. Run prepare_nfl_covid_data.py first."
        )

    panel = pd.read_csv(PANEL_PATH, dtype={"fips": str}, parse_dates=["WeekStartDate", "WeekEndDate"])
    panel = panel.sort_values(["fips", "WeekStartDate"]).reset_index(drop=True)

    binary_panel = panel.copy()
    diagnostic_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []

    outcome_codes: list[str] = []
    intervention_codes: list[str] = []

    for threshold in case_thresholds:
        code = outcome_code_from_threshold(CASE_OUTCOME_FAMILY, float(threshold))
        column = f"x_{code}_pm1"
        binary_panel[column] = binary_from_threshold(binary_panel["case_rate_100k"], float(threshold))
        outcome_codes.append(code)
        diagnostic_rows.append(
            summarize_binary(binary_panel, column, "outcome", outcome_label_from_code(code))
        )

    for threshold in death_thresholds:
        code = outcome_code_from_threshold(DEATH_OUTCOME_FAMILY, float(threshold))
        column = f"x_{code}_pm1"
        binary_panel[column] = binary_from_threshold(binary_panel["death_rate_100k"], float(threshold))
        outcome_codes.append(code)
        diagnostic_rows.append(
            summarize_binary(binary_panel, column, "outcome", outcome_label_from_code(code))
        )

    for threshold in sorted(set(args.attendance_share_thresholds)):
        code = intervention_code_from_threshold(float(threshold))
        column = f"z_{code}_pm1"
        binary_panel[column] = binary_from_threshold(binary_panel["attendance_share_pct"], float(threshold))
        intervention_codes.append(code)
        diagnostic_rows.append(
            summarize_binary(binary_panel, column, "intervention", intervention_label_from_code(code))
        )

    for outcome_code in outcome_codes:
        for intervention_code in intervention_codes:
            outcome_col = f"x_{outcome_code}_pm1"
            intervention_col = f"z_{intervention_code}_pm1"
            for lag_code in SUPPORTED_LAGS:
                summary = compute_support_summary(binary_panel, outcome_col, intervention_col, lag_code)
                summary.update(
                    {
                        "outcome_code": outcome_code,
                        "intervention_code": intervention_code,
                        "lag_code": lag_code,
                    }
                )
                support_rows.append(summary)

    binary_panel.to_csv(BINARY_PANEL_PATH, index=False)
    with DIAGNOSTICS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    support_table = pd.DataFrame(support_rows)
    support_table.to_csv(SUPPORT_CSV_PATH, index=False)
    write_markdown_summary(diagnostic_rows)
    write_support_markdown(support_rows)
    write_readme()


if __name__ == "__main__":
    main()
