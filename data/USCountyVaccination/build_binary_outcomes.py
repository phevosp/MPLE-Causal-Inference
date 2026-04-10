"""Build binary nationwide US county outcomes/interventions and diagnostics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import INTERVENTION_SPECS, OUTCOME_SPECS, PROCESSED_DIR  # noqa: E402


PANEL_PATH = PROCESSED_DIR / "us_county_weekly_panel.csv.gz"
BINARY_PANEL_PATH = PROCESSED_DIR / "us_county_binary_panel.csv.gz"
DIAGNOSTICS_CSV_PATH = PROCESSED_DIR / "us_county_binary_threshold_diagnostics.csv"
DIAGNOSTICS_MD_PATH = PROCESSED_DIR / "us_county_binary_threshold_diagnostics.md"


def binary_from_threshold(series: pd.Series, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.where(values >= threshold, 1, -1), index=series.index, dtype="float")
    out[values.isna()] = np.nan
    return out.astype("Int64")


def fill_pre_reporting_with_negative_ones(
    panel: pd.DataFrame,
    source_column: str,
    binary_values: pd.Series,
) -> pd.Series:
    source = pd.to_numeric(panel[source_column], errors="coerce")
    first_observed = (
        panel.loc[source.notna(), ["fips", "WeekEndDate"]]
        .groupby("fips", sort=False)["WeekEndDate"]
        .min()
    )
    first_by_row = panel["fips"].map(first_observed)
    adjusted = binary_values.copy()
    adjusted.loc[first_by_row.notna() & panel["WeekEndDate"].lt(first_by_row)] = -1
    return adjusted.astype("Int64")


def transition_rate(panel: pd.DataFrame, column: str) -> float:
    ordered = panel.sort_values(["fips", "WeekEndDate"]).copy()
    ordered["prev_value"] = ordered.groupby("fips", sort=False)[column].shift(1)
    valid = ordered[column].notna() & ordered["prev_value"].notna()
    if not valid.any():
        return float("nan")
    return float((ordered.loc[valid, column] != ordered.loc[valid, "prev_value"]).mean())


def summarize_binary(panel: pd.DataFrame, column: str, definition_type: str, label: str, notes: str) -> dict[str, object]:
    values = panel[column]
    available = values.notna()
    positive = values.eq(1)
    per_county_variety = (
        panel.loc[available, ["fips", column]].groupby("fips")[column].nunique()
        if available.any()
        else pd.Series(dtype=int)
    )
    available_dates = panel.loc[available, "WeekEndDate"]
    return {
        "definition_type": definition_type,
        "column": column,
        "label": label,
        "notes": notes,
        "eligible_rows": int(available.sum()),
        "eligible_share": float(available.mean()),
        "positive_share": float(positive.loc[available].mean()) if available.any() else float("nan"),
        "transition_rate": transition_rate(panel, column),
        "counties_with_both_states": int((per_county_variety == 2).sum()),
        "counties_constant": int((per_county_variety == 1).sum()),
        "eligible_start_date": available_dates.min() if available.any() else pd.NaT,
        "eligible_end_date": available_dates.max() if available.any() else pd.NaT,
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
        handle.write("# US County Vaccination Binary Definitions\n\n")
        handle.write(
            "This table summarizes the fixed county-week binary outcome and intervention "
            "definitions used by the nationwide US county vaccination experiments. "
            "Here `+1` denotes the above-threshold state and `-1` denotes the below-threshold state.\n\n"
        )
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            rendered: list[str] = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    rendered.append(f"{value:.4f}")
                else:
                    rendered.append("" if pd.isna(value) else str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PANEL_PATH.name}. Run prepare_us_county_vaccination_data.py first."
        )

    panel = pd.read_csv(PANEL_PATH, dtype={"fips": str}, parse_dates=["WeekStartDate", "WeekEndDate"])
    panel = panel.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)

    binary_panel = panel.copy()
    diagnostic_rows: list[dict[str, object]] = []

    for outcome_code, spec in OUTCOME_SPECS.items():
        column = f"x_{outcome_code}_pm1"
        binary_panel[column] = binary_from_threshold(binary_panel[spec.source_column], float(spec.threshold))
        diagnostic_rows.append(
            summarize_binary(binary_panel, column, "outcome", spec.label, spec.notes)
        )

    for intervention_code, spec in INTERVENTION_SPECS.items():
        column = f"z_{intervention_code}_pm1"
        binary_panel[column] = fill_pre_reporting_with_negative_ones(
            binary_panel,
            spec.source_column,
            binary_from_threshold(binary_panel[spec.source_column], spec.threshold),
        )
        diagnostic_rows.append(
            summarize_binary(binary_panel, column, "intervention", spec.label, spec.notes)
        )

    binary_panel.to_csv(BINARY_PANEL_PATH, index=False)
    with DIAGNOSTICS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    write_markdown_summary(diagnostic_rows)


if __name__ == "__main__":
    main()
