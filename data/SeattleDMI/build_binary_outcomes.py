from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def add_district_metadata(panel: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Merge neighborhood-district labels onto the panel via the processed block crosswalk."""
    crosswalk = pd.read_csv(
        processed_dir / "seattledmi_block_crosswalk.csv",
        dtype={"GEOID10": str},
        usecols=["GEOID10", "NEIGHBORHOOD_DISTRICT_NAME"],
    )
    crosswalk["NEIGHBORHOOD_DISTRICT_NAME"] = crosswalk[
        "NEIGHBORHOOD_DISTRICT_NAME"
    ].fillna("MissingDistrict")
    return panel.merge(crosswalk, on="GEOID10", how="left")


def add_intervention_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Annotate the panel with treated-ever and treated-pre/post group labels."""
    treated_ever = panel.groupby("GEOID10")["Intervention"].max().rename("treated_ever")
    panel = panel.merge(treated_ever, on="GEOID10", how="left")
    panel["group"] = "untreated"
    panel.loc[
        (panel["treated_ever"] == 1) & (panel["Intervention"] == 0), "group"
    ] = "treated_pre"
    panel.loc[
        (panel["treated_ever"] == 1) & (panel["Intervention"] == 1), "group"
    ] = "treated_post"
    return panel


def binary_pm1(series: pd.Series, threshold: float) -> pd.Series:
    """Convert a count outcome to {-1, +1} using the rule count > threshold."""
    return (series.gt(threshold).astype(int) * 2 - 1).astype("int8")


def transition_rate(panel: pd.DataFrame, column: str) -> float:
    """Compute the share of within-block time steps where the binary state changes."""
    ordered = panel.sort_values(["GEOID10", "time"]).copy()
    ordered["prev"] = ordered.groupby("GEOID10")[column].shift(1)
    valid = ordered["prev"].notna()
    if not valid.any():
        return float("nan")
    return float((ordered.loc[valid, column] != ordered.loc[valid, "prev"]).mean())


def summarize_threshold(panel: pd.DataFrame, source_col: str, threshold: int) -> dict[str, float | int | str]:
    """Summarize one thresholded binary outcome."""
    column = f"{source_col}_gt_{threshold}_pm1"
    y01 = panel[column].eq(1).astype(int)
    per_block_variety = panel.assign(y01=y01).groupby("GEOID10")["y01"].nunique()

    summary = {
        "outcome": source_col,
        "threshold_rule": f"{source_col} > {threshold}",
        "positive_share_overall": float(y01.mean()),
        "positive_share_untreated": float(y01.loc[panel["group"] == "untreated"].mean()),
        "positive_share_treated_pre": float(y01.loc[panel["group"] == "treated_pre"].mean()),
        "positive_share_treated_post": float(y01.loc[panel["group"] == "treated_post"].mean()),
        "transition_rate_overall": transition_rate(panel, column),
        "blocks_with_both_states": int((per_block_variety == 2).sum()),
        "blocks_constant": int((per_block_variety == 1).sum()),
    }
    return summary


def summarize_named_threshold(panel: pd.DataFrame, source_col: str, column: str, label: str) -> dict[str, float | int | str]:
    """Summarize one precomputed thresholded binary outcome."""
    y01 = panel[column].eq(1).astype(int)
    per_block_variety = panel.assign(y01=y01).groupby("GEOID10")["y01"].nunique()
    return {
        "outcome": source_col,
        "threshold_rule": label,
        "positive_share_overall": float(y01.mean()),
        "positive_share_untreated": float(y01.loc[panel["group"] == "untreated"].mean()),
        "positive_share_treated_pre": float(y01.loc[panel["group"] == "treated_pre"].mean()),
        "positive_share_treated_post": float(y01.loc[panel["group"] == "treated_post"].mean()),
        "transition_rate_overall": transition_rate(panel, column),
        "blocks_with_both_states": int((per_block_variety == 2).sum()),
        "blocks_constant": int((per_block_variety == 1).sum()),
    }


def district_mean_threshold(
    panel: pd.DataFrame,
    source_col: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Threshold each observation against the mean level of that outcome in its district."""
    district_thresholds = (
        panel.groupby("NEIGHBORHOOD_DISTRICT_NAME", dropna=False)[source_col]
        .mean()
        .rename(f"{source_col}_district_mean")
        .reset_index()
    )
    panel_with_means = panel.merge(
        district_thresholds,
        on="NEIGHBORHOOD_DISTRICT_NAME",
        how="left",
    )
    binary = binary_pm1(panel_with_means[source_col], panel_with_means[f"{source_col}_district_mean"])
    return binary, district_thresholds


def write_markdown_summary(output_path: Path, threshold_rows: list[dict[str, float | int | str]]) -> None:
    """Write a short Markdown interpretation of the binary outcome thresholds."""
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SeattleDMI Binary Outcome Summary\n\n")
        handle.write(
            "This summary compares candidate binary outcomes built from the SeattleDMI panel. "
            "The requested primary outcomes are the `> 0` rules written to "
            "`seattledmi_binary_outcomes.csv.gz`.\n\n"
        )

        handle.write("## Primary Outcomes Saved\n\n")
        handle.write("- `i_drugs_gt_0_pm1`: `+1` if `i_drugs >= 1`, `-1` otherwise\n")
        handle.write("- `any_crime_gt_0_pm1`: `+1` if `any_crime >= 1`, `-1` otherwise\n\n")
        handle.write("- `i_drugs_gt_district_mean_pm1`: `+1` if `i_drugs` is above the mean `i_drugs` level in that block's neighborhood district, `-1` otherwise\n")
        handle.write("- `any_crime_gt_district_mean_pm1`: `+1` if `any_crime` is above the mean `any_crime` level in that block's neighborhood district, `-1` otherwise\n\n")

        handle.write("## Threshold Comparison\n\n")
        headers = [
            "outcome",
            "threshold_rule",
            "positive_share_overall",
            "positive_share_untreated",
            "positive_share_treated_pre",
            "positive_share_treated_post",
            "transition_rate_overall",
            "blocks_with_both_states",
            "blocks_constant",
        ]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in threshold_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["outcome"]),
                        str(row["threshold_rule"]),
                        f"{float(row['positive_share_overall']):.4f}",
                        f"{float(row['positive_share_untreated']):.4f}",
                        f"{float(row['positive_share_treated_pre']):.4f}",
                        f"{float(row['positive_share_treated_post']):.4f}",
                        f"{float(row['transition_rate_overall']):.4f}",
                        str(int(row["blocks_with_both_states"])),
                        str(int(row["blocks_constant"])),
                    ]
                )
                + " |\n"
            )

        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- `i_drugs > 0` is very sparse. Only about 3% of block-quarters are positive, "
            "so it isolates drug activity sharply but may be too coarse if you want a more balanced binary state.\n"
        )
        handle.write(
            "- `any_crime > 0` is much denser, around 49% positive overall, so it has much better class balance and more variation for binary-state modeling.\n"
        )
        handle.write(
            "- For `i_drugs`, thresholds above 0 become extremely rare and therefore are usually too aggressive.\n"
        )
        handle.write(
            "- For `any_crime`, thresholds like `> 1`, `> 2`, or `> 3` are plausible alternatives if you want a more selective definition of a `bad` block-quarter.\n"
        )
        handle.write(
            "- District-mean thresholding adapts the cutoff to local baseline crime levels, which can be attractive if you want `good` and `bad` to be defined relative to a block's surrounding district rather than globally.\n"
        )
        handle.write(
            "- A reasonable default pair is:\n"
            "  - `i_drugs > 0` if you want a targeted drug-market outcome\n"
            "  - `any_crime > 1` or `any_crime > 2` if you want a broader but less trivial crime outcome\n"
            "  - district-mean thresholding if you specifically want relative-within-district classification\n"
        )


def main() -> None:
    """Create binary outcome files and threshold summaries for SeattleDMI."""
    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "processed"
    panel_path = processed_dir / "seattledmi_panel.csv.gz"
    panel = pd.read_csv(panel_path, dtype={"GEOID10": str})
    panel = add_district_metadata(panel, processed_dir)
    panel = add_intervention_groups(panel)

    threshold_grid = {
        "i_drugs": [0, 1, 2, 3],
        "any_crime": [0, 1, 2, 3, 5],
    }

    binary_output = panel[
        [
            "GEOID10",
            "time",
            "Intervention",
            "treated_ever",
            "group",
            "NEIGHBORHOOD_DISTRICT_NAME",
        ]
    ].copy()
    threshold_rows: list[dict[str, float | int | str]] = []
    district_threshold_tables: list[pd.DataFrame] = []

    for source_col, thresholds in threshold_grid.items():
        for threshold in thresholds:
            out_col = f"{source_col}_gt_{threshold}_pm1"
            panel[out_col] = binary_pm1(panel[source_col], threshold)
            threshold_rows.append(summarize_threshold(panel, source_col, threshold))
            if threshold == 0:
                binary_output[out_col] = panel[out_col]

        district_col = f"{source_col}_gt_district_mean_pm1"
        district_binary, district_thresholds = district_mean_threshold(panel, source_col)
        panel[district_col] = district_binary
        threshold_rows.append(
            summarize_named_threshold(
                panel,
                source_col,
                district_col,
                f"{source_col} > district_mean",
            )
        )
        binary_output[district_col] = panel[district_col]
        district_threshold_tables.append(district_thresholds)

    binary_output.to_csv(processed_dir / "seattledmi_binary_outcomes.csv.gz", index=False)

    district_thresholds = district_threshold_tables[0].merge(
        district_threshold_tables[1],
        on="NEIGHBORHOOD_DISTRICT_NAME",
        how="outer",
    )
    district_thresholds.to_csv(
        processed_dir / "seattledmi_district_mean_thresholds.csv",
        index=False,
    )

    summary_csv = processed_dir / "seattledmi_binary_threshold_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    write_markdown_summary(
        processed_dir / "seattledmi_binary_threshold_summary.md",
        threshold_rows,
    )


if __name__ == "__main__":
    main()
