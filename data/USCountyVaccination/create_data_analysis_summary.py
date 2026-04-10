"""Create descriptive analysis outputs for the USCountyVaccination dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "data_analysis"
PANEL_PATH = PROCESSED_DIR / "us_county_binary_panel.csv.gz"

OUTCOME_COLUMN = "x_death_rate_100k_ge_2_pm1"
OUTCOME_CODE = "death_rate_100k_ge_2"
INTERVENTION_COLUMNS = {
    "complete_cov_ge_20": "z_complete_cov_ge_20_pm1",
    "complete_cov_ge_30": "z_complete_cov_ge_30_pm1",
    "complete_cov_ge_40": "z_complete_cov_ge_40_pm1",
}
CORRELATION_LAG_WEEKS = 2


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PANEL_PATH.name}. Run the USCountyVaccination preprocessing first."
        )
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"fips": str},
        parse_dates=["WeekEndDate", "WeekStartDate"],
    )
    return panel.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)


def build_county_population_summary(panel: pd.DataFrame) -> pd.DataFrame:
    county_summary = (
        panel.groupby("fips", as_index=False)
        .agg(
            county=("county", "last"),
            state_name=("state_name", "last"),
            population=("population", "max"),
            land_area_sq_km=("land_area_sq_km", "max"),
        )
        .sort_values(["state_name", "county", "fips"])
        .reset_index(drop=True)
    )
    county_summary["population_rank_desc"] = county_summary["population"].rank(
        method="min", ascending=False
    ).astype("Int64")
    county_summary["population_missing"] = county_summary["population"].isna()
    return county_summary


def summarize_binary_by_week(
    panel: pd.DataFrame,
    variable_code: str,
    column: str,
) -> pd.DataFrame:
    grouped = panel.groupby("WeekEndDate", as_index=False)
    summary = grouped.agg(
        county_count=("fips", "nunique"),
        eligible_count=(column, lambda s: int(s.notna().sum())),
        have_count=(column, lambda s: int(s.eq(1).sum())),
        do_not_have_count=(column, lambda s: int(s.eq(-1).sum())),
    )
    summary["missing_count"] = summary["county_count"] - summary["eligible_count"]
    summary["have_share"] = summary["have_count"] / summary["eligible_count"]
    summary["do_not_have_share"] = summary["do_not_have_count"] / summary["eligible_count"]
    summary["missing_share"] = summary["missing_count"] / summary["county_count"]
    summary.insert(1, "variable_code", variable_code)
    return summary


def build_weekly_binary_shares(panel: pd.DataFrame) -> pd.DataFrame:
    frames = [summarize_binary_by_week(panel, OUTCOME_CODE, OUTCOME_COLUMN)]
    for variable_code, column in INTERVENTION_COLUMNS.items():
        frames.append(summarize_binary_by_week(panel, variable_code, column))
    weekly_summary = pd.concat(frames, ignore_index=True)
    return weekly_summary.sort_values(["variable_code", "WeekEndDate"]).reset_index(drop=True)


def pooled_prevalence(panel: pd.DataFrame, column: str) -> tuple[int, int, float]:
    eligible = panel[column].notna()
    positive = panel[column].eq(1)
    eligible_count = int(eligible.sum())
    positive_count = int(positive.loc[eligible].sum())
    positive_share = positive_count / eligible_count
    return eligible_count, positive_count, positive_share


def build_lagged_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = panel.sort_values(["fips", "WeekEndDate"]).copy()
    for variable_code, column in INTERVENTION_COLUMNS.items():
        lagged_column = f"{column}_lag{CORRELATION_LAG_WEEKS}"
        ordered[lagged_column] = ordered.groupby("fips", sort=False)[column].shift(CORRELATION_LAG_WEEKS)
        valid = ordered[OUTCOME_COLUMN].notna() & ordered[lagged_column].notna()
        aligned = ordered.loc[valid, [OUTCOME_COLUMN, lagged_column]]
        rows.append(
            {
                "outcome_code": OUTCOME_CODE,
                "outcome_column": OUTCOME_COLUMN,
                "intervention_code": variable_code,
                "intervention_column": column,
                "lag_weeks": CORRELATION_LAG_WEEKS,
                "valid_rows": int(valid.sum()),
                "correlation": float(aligned[OUTCOME_COLUMN].corr(aligned[lagged_column])),
            }
        )
    return pd.DataFrame(rows).sort_values("intervention_code").reset_index(drop=True)


def render_population_plot(county_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    population = county_summary["population"].dropna()
    ax.hist(population, bins=40, color="#355070", edgecolor="white", alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("County population (log scale)")
    ax.set_ylabel("Number of counties")
    ax.set_title("Distribution of county population")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_share_plot(
    weekly_summary: pd.DataFrame,
    variable_code: str,
    title: str,
    output_path: Path,
) -> None:
    data = weekly_summary.loc[weekly_summary["variable_code"] == variable_code].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["WeekEndDate"], data["have_share"], label="Have", color="#2a9d8f", linewidth=2.2)
    ax.plot(
        data["WeekEndDate"],
        data["do_not_have_share"],
        label="Do not have",
        color="#e76f51",
        linewidth=2.2,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of eligible counties")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(
    panel: pd.DataFrame,
    county_summary: pd.DataFrame,
    weekly_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> str:
    county_count = county_summary["fips"].nunique()
    week_count = panel["WeekEndDate"].nunique()
    date_min = panel["WeekEndDate"].min().date().isoformat()
    date_max = panel["WeekEndDate"].max().date().isoformat()

    population_stats = county_summary["population"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )

    prevalence_rows = []
    for variable_code, column in [(OUTCOME_CODE, OUTCOME_COLUMN), *INTERVENTION_COLUMNS.items()]:
        eligible_count, positive_count, positive_share = pooled_prevalence(panel, column)
        prevalence_rows.append(
            {
                "variable_code": variable_code,
                "column": column,
                "eligible_count": eligible_count,
                "have_count": positive_count,
                "have_share": positive_share,
                "missing_count": int(panel[column].isna().sum()),
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)

    lines: list[str] = [
        "# USCountyVaccination Data Analysis",
        "",
        "This folder summarizes the nationwide county-week dataset for outcome "
        f"`{OUTCOME_CODE}` and interventions "
        "`complete_cov_ge_20`, `complete_cov_ge_30`, and `complete_cov_ge_40`.",
        "",
        "## Headline facts",
        "",
        f"- Counties in scope: `{format_int(int(county_count))}`",
        f"- Weeks in scope: `{format_int(int(week_count))}`",
        f"- County-week rows: `{format_int(int(len(panel)))}`",
        f"- Date range: `{date_min}` through `{date_max}`",
        "",
        "## Outputs",
        "",
        "- [county_population_summary.csv](county_population_summary.csv)",
        "- [weekly_binary_shares.csv](weekly_binary_shares.csv)",
        "- [lag2_correlations.csv](lag2_correlations.csv)",
        "- [county_population_distribution.png](county_population_distribution.png)",
        "- [intervention_share_complete_cov_ge_20.png](intervention_share_complete_cov_ge_20.png)",
        "- [intervention_share_complete_cov_ge_30.png](intervention_share_complete_cov_ge_30.png)",
        "- [intervention_share_complete_cov_ge_40.png](intervention_share_complete_cov_ge_40.png)",
        "- [outcome_share_death_rate_100k_ge_2.png](outcome_share_death_rate_100k_ge_2.png)",
        "",
        "## Methodology",
        "",
        "- The source of truth is `processed/us_county_binary_panel.csv.gz`.",
        "- The binary variables are the `pm1` columns in that processed panel.",
        "- Intervention binaries inherit the repo's existing semantics: weeks before first observed vaccination reporting are prefilled with `-1`.",
        "- Weekly shares use the full county panel for each week and exclude only truly missing values from the share denominator.",
        f"- The correlation uses outcome at week `t` and intervention at week `t-{CORRELATION_LAG_WEEKS}`.",
        "",
        "## County population summary",
        "",
        f"- Non-missing county populations: `{format_int(int(county_summary['population'].notna().sum()))}`",
        f"- Mean population: `{format_float(float(population_stats['mean']), 1)}`",
        f"- Median population: `{format_float(float(population_stats['50%']), 1)}`",
        f"- 10th percentile: `{format_float(float(population_stats['10%']), 1)}`",
        f"- 90th percentile: `{format_float(float(population_stats['90%']), 1)}`",
        f"- Maximum population: `{format_float(float(population_stats['max']), 1)}`",
        "",
        "## Binary prevalence and missingness",
        "",
        "| variable_code | column | eligible_count | missing_count | have_count | have_share |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for _, row in prevalence.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variable_code"]),
                    str(row["column"]),
                    format_int(int(row["eligible_count"])),
                    format_int(int(row["missing_count"])),
                    format_int(int(row["have_count"])),
                    format_float(float(row["have_share"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Lag-2 correlations",
            "",
            "| intervention_code | intervention_column | lag_weeks | valid_rows | correlation |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in correlation_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["intervention_code"]),
                    str(row["intervention_column"]),
                    str(int(row["lag_weeks"])),
                    format_int(int(row["valid_rows"])),
                    format_float(float(row["correlation"]), 6),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Outcome shares are available for all county-weeks in the panel.",
            "- The selected intervention columns contain only a small number of truly missing rows; those rows remain visible in the CSV outputs via `missing_count` and `missing_share`.",
            "- The intervention time-series reflect the binary threshold definitions already used by the experiment pipeline, so early pre-vaccination weeks appear as counties not yet having the intervention.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    county_summary = build_county_population_summary(panel)
    weekly_summary = build_weekly_binary_shares(panel)
    correlation_summary = build_lagged_correlations(panel)

    county_summary.to_csv(OUTPUT_DIR / "county_population_summary.csv", index=False)
    weekly_summary.to_csv(OUTPUT_DIR / "weekly_binary_shares.csv", index=False)
    correlation_summary.to_csv(OUTPUT_DIR / "lag2_correlations.csv", index=False)

    render_population_plot(county_summary, OUTPUT_DIR / "county_population_distribution.png")
    render_share_plot(
        weekly_summary,
        "complete_cov_ge_20",
        "Share of counties with complete_cov_ge_20 over time",
        OUTPUT_DIR / "intervention_share_complete_cov_ge_20.png",
    )
    render_share_plot(
        weekly_summary,
        "complete_cov_ge_30",
        "Share of counties with complete_cov_ge_30 over time",
        OUTPUT_DIR / "intervention_share_complete_cov_ge_30.png",
    )
    render_share_plot(
        weekly_summary,
        "complete_cov_ge_40",
        "Share of counties with complete_cov_ge_40 over time",
        OUTPUT_DIR / "intervention_share_complete_cov_ge_40.png",
    )
    render_share_plot(
        weekly_summary,
        OUTCOME_CODE,
        "Share of counties with death_rate_100k_ge_2 over time",
        OUTPUT_DIR / "outcome_share_death_rate_100k_ge_2.png",
    )

    report = build_report(panel, county_summary, weekly_summary, correlation_summary)
    (OUTPUT_DIR / "summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
