"""Prepare the Ohio and Massachusetts COVID School Data Hub package."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils import (  # noqa: E402
    build_knn_and_kernel_edges,
    build_touching_edge_list,
    count_connected_components,
    download_if_missing,
    fetch_arcgis_geojson,
    normalize_name,
    parse_numeric_text,
    save_raw_geojson,
    standardize_id,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"


CSDH_URLS = {
    "district_monthly_shares": "https://assets.ctfassets.net/9fbw4onh0qc1/4LRV2nKQOBoCudvyVxLx6w/fdb67c6da6252d520b83d173f3a41237/District_Monthly_Shares_03.08.23.csv",
    "learning_csv_zip": "https://assets.ctfassets.net/9fbw4onh0qc1/3JXV9ahOubLLnh9aHTHgKv/6e3c8a2baf1f2e0517edd9e454ee5c74/CSDH_District_Files_-_CSV.zip",
    "community_case_rate_zip": "https://downloads.ctfassets.net/9fbw4onh0qc1/1FyYF7Qqmn2fXfWYqcqZUB/d2f9ec9d4a78bdedbc93869396393c09/Matched_Districts_and_Case_Rates.zip",
    "community_case_rate_codebook": "https://assets.ctfassets.net/9fbw4onh0qc1/3vad828a7tYRJ2F7Qeqfbh/1c4612a5918ac618e9b05eeae46344ee/Cate_Rate_Codebook.xlsx",
    "nces_district_demographics": "https://assets.ctfassets.net/9fbw4onh0qc1/6lSX82GvL9tPRpSE9VNkB/24cd9aea4dc7af91be9344c3ccd661f0/NCES_2020-2021_District_Demographics.csv",
    "district_merge_code_do": "https://assets.ctfassets.net/9fbw4onh0qc1/1lKC4sytDUc3laSmTxhSFl/72189d83203a5ce045f1cf4739914469/district_code_to_combine_learning_model.do",
}

MASS_OFFICIAL_SERVICE = "https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/AGOL/Public_School_Districts/FeatureServer"
MASS_OFFICIAL_LAYERS = {0: "k12", 1: "secondary", 2: "elementary"}
OHIO_OFFICIAL_LAYER = "https://maps.ohio.gov/arcgis/rest/services/Hosted/Ohio_School_Districts_2025/FeatureServer/0"

EDGE_STANDARDIZED_URL = "https://nces.ed.gov/programs/edge/data/EDGE_SCHOOLDISTRICT_TL21_SY2021.zip"
EDGE_ACS_DASHBOARD_BASE = "https://nces.ed.gov/programs/edge/ACSDashboard"

STATE_NAMES = {"OH": "Ohio", "MA": "Massachusetts"}
STATE_FIPS = {"MA": "25", "OH": "39"}
FIPS_TO_STATE = {value: key for key, value in STATE_FIPS.items()}

SAIPE_URLS = {
    "ohio": "https://www2.census.gov/programs-surveys/saipe/datasets/2021/2021-school-districts/sd21-oh.txt",
    "massachusetts": "https://www2.census.gov/programs-surveys/saipe/datasets/2021/2021-school-districts/sd21-ma.txt",
    "layout": "https://www2.census.gov/programs-surveys/saipe/technical-documentation/file-layouts/school-district/2021-district-layout.txt",
}

EDGE_TEXT_FIELD_MAP = {
    "Total Population": "edge_acsed_total_population",
    "Median Household Income": "edge_acsed_median_household_income",
    "Total Households": "edge_acsed_total_households",
}

EDGE_GAUGE_FIELD_MAP = {
    "White": "edge_acsed_pct_white",
    "Black or African American": "edge_acsed_pct_black",
    "Hispanic or Latino": "edge_acsed_pct_hispanic",
    "Asian": "edge_acsed_pct_asian",
    "American Indian/ Alaska Native": "edge_acsed_pct_american_indian_alaska_native",
    "Native Hawaiian and Other Pacific Islander": "edge_acsed_pct_native_hawaiian_pacific_islander",
    "Some other race alone": "edge_acsed_pct_other_race",
    "Two or more races": "edge_acsed_pct_two_or_more_races",
    "Married-Couple": "edge_acsed_pct_married_couple_households",
    "Cohabitating-Couple": "edge_acsed_pct_cohabitating_couple_households",
    "Female householder, no spouse/partner present": "edge_acsed_pct_female_householder_no_partner",
    "Male householder, no spouse/partner present": "edge_acsed_pct_male_householder_no_partner",
}

EDGE_CHART_FIELD_MAP = {
    "englishornotchart": [
        "edge_acsed_pct_english_only",
        "edge_acsed_pct_english_less_than_well",
        "edge_acsed_pct_english_very_well",
    ],
    "attainmentchart": [
        "edge_acsed_pct_less_than_high_school",
        "edge_acsed_pct_high_school",
        "edge_acsed_pct_some_college",
        "edge_acsed_pct_bachelors_or_higher",
    ],
    "laborchart": [
        "edge_acsed_pct_in_labor_force",
        "edge_acsed_pct_not_in_labor_force",
        "edge_acsed_pct_management_occupations",
        "edge_acsed_pct_service_occupations",
        "edge_acsed_pct_sales_office_occupations",
        "edge_acsed_pct_natural_resources_construction_maintenance_occupations",
        "edge_acsed_pct_production_transportation_material_moving_occupations",
    ],
    "fschart": ["edge_acsed_pct_households_snap"],
    "broadbandchart": ["edge_acsed_pct_households_broadband"],
    "povertychart": ["edge_acsed_pct_below_poverty"],
}


def ensure_directories() -> dict[str, Path]:
    """Create the raw and processed directories used by the COVID School Data Hub package."""
    paths = {
        "base": BASE_DIR,
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "csdh": RAW_DIR / "csdh",
        "geo_ma": RAW_DIR / "geography" / "massachusetts",
        "geo_oh": RAW_DIR / "geography" / "ohio",
        "geo_edge": RAW_DIR / "geography" / "edge",
        "features_edge": RAW_DIR / "features" / "edge",
        "features_edge_dashboards": RAW_DIR / "features" / "edge" / "acs_dashboard_html",
        "features_saipe": RAW_DIR / "features" / "saipe",
        "features_csdh": RAW_DIR / "features" / "csdh",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI options for optionally reusing expensive intermediate outputs."""
    parser = argparse.ArgumentParser(
        description="Prepare the Ohio and Massachusetts COVID School Data Hub package."
    )
    parser.add_argument(
        "--reuse_processed_tables",
        action="store_true",
        help="Reuse the already-written cleaned learning/case/joined tables instead of rebuilding them.",
    )
    parser.add_argument(
        "--reuse_processed_features",
        action="store_true",
        help="Reuse the already-written EDGE ACS-ED and SAIPE feature tables instead of refetching/parsing them.",
    )
    parser.add_argument(
        "--reuse_processed_networks",
        action="store_true",
        help="Reuse the already-written network summary instead of rewriting all centroid and adjacency artifacts.",
    )
    return parser.parse_args()


def load_learning_periods(learning_zip_path: Path) -> pd.DataFrame:
    """Load and standardize the district learning-model files from the CSDH zip archive."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(learning_zip_path) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and "__macosx" not in name.lower()
        ]
        for name in csv_names:
            frame = pd.read_csv(archive.open(name), low_memory=False)
            frame["source_file"] = Path(name).name
            frames.append(frame)

    learning = pd.concat(frames, ignore_index=True, sort=False)
    learning = learning.loc[learning["DataLevel"].eq("District")].copy()
    learning["NCESDistrictID"] = standardize_id(learning["NCESDistrictID"], width=7)
    learning["StateAssignedDistrictID"] = standardize_id(learning["StateAssignedDistrictID"])
    learning["PeriodStartDate"] = pd.to_datetime(
        learning["TimePeriodStart"],
        errors="coerce",
        format="mixed",
    )
    learning["PeriodEndDate"] = pd.to_datetime(
        learning["TimePeriodEnd"],
        errors="coerce",
        format="mixed",
    )
    learning["DistrictNameNormalized"] = learning["DistrictName"].map(normalize_name)
    learning = learning.dropna(subset=["NCESDistrictID", "PeriodStartDate", "PeriodEndDate"])
    learning = learning.sort_values(
        ["StateAbbrev", "NCESDistrictID", "PeriodStartDate", "PeriodEndDate"]
    ).reset_index(drop=True)
    return learning


def load_case_rates(case_zip_path: Path) -> pd.DataFrame:
    """Load and standardize the district-week-ZIP community case-rate file."""
    with zipfile.ZipFile(case_zip_path) as archive:
        csv_name = next(
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and "__macosx" not in name.lower()
        )
        case_rates = pd.read_csv(archive.open(csv_name), sep="|", low_memory=False)

    case_rates["NCESDistrictID"] = standardize_id(case_rates["leaid"], width=7)
    case_rates["StateAssignedDistrictID"] = standardize_id(case_rates["state_leaid"])
    case_rates["StateAbbrev"] = standardize_id(case_rates["state"])
    case_rates["zip"] = standardize_id(case_rates["zip"], width=5)
    case_rates["WeekStartDate"] = pd.to_datetime(case_rates["start_date"], format="%d%b%Y", errors="coerce")
    case_rates["WeekEndDate"] = pd.to_datetime(case_rates["end_date"], format="%d%b%Y", errors="coerce")
    case_rates["DistrictNameNormalized"] = case_rates["lea_name"].map(normalize_name)
    case_rates = case_rates.dropna(subset=["NCESDistrictID", "WeekStartDate", "WeekEndDate"])
    case_rates = case_rates.sort_values(
        ["StateAbbrev", "NCESDistrictID", "WeekStartDate", "zip"]
    ).reset_index(drop=True)
    return case_rates


def load_monthly_shares(monthly_csv_path: Path) -> pd.DataFrame:
    """Load and standardize the CSDH district monthly in-person share file."""
    monthly = pd.read_csv(monthly_csv_path, low_memory=False)
    monthly["NCESDistrictID"] = standardize_id(monthly["NCESDistrictID"], width=7)
    monthly["StateAbbrev"] = standardize_id(monthly["StateAbbrev"])
    monthly["Month"] = monthly["month"].astype(str).str.strip()
    month_parts = monthly["Month"].str.extract(r"(?P<year>\d{4})m(?P<month>\d{1,2})")
    monthly["MonthStartDate"] = pd.to_datetime(
        month_parts["year"] + "-" + month_parts["month"] + "-01",
        errors="coerce",
    )
    monthly["MonthEndDate"] = monthly["MonthStartDate"] + pd.offsets.MonthEnd(0)
    monthly["DistrictNameNormalized"] = monthly["DistrictName"].map(normalize_name)
    for column in ["share_inperson", "share_hybrid", "share_virtual"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    monthly = monthly.dropna(subset=["NCESDistrictID", "MonthStartDate"])
    monthly = monthly.sort_values(["StateAbbrev", "NCESDistrictID", "MonthStartDate"]).reset_index(drop=True)
    return monthly


def load_nces_district_master(nces_csv_path: Path) -> pd.DataFrame:
    """Load the NCES district demographics file as the canonical district master."""
    nces = pd.read_csv(nces_csv_path, low_memory=False)
    nces["NCESDistrictID"] = standardize_id(nces["NCESDistrictID"], width=7)
    nces["state_leaid"] = standardize_id(nces["state_leaid"])
    nces["state"] = standardize_id(nces["fips"]).str[:2]
    nces["district_name_normalized"] = nces["lea_name"].map(normalize_name)
    nces = nces.sort_values(["NCESDistrictID"]).reset_index(drop=True)
    return nces


def save_core_tables(
    learning: pd.DataFrame,
    case_rates: pd.DataFrame,
    nces_master: pd.DataFrame,
) -> None:
    """Write the main cleaned national tables and Ohio/Massachusetts subsets."""
    learning.to_csv(PROCESSED_DIR / "csdh_district_learning_periods.csv.gz", index=False)
    case_rates.to_csv(PROCESSED_DIR / "csdh_case_rates_by_district_zip_week.csv.gz", index=False)
    nces_master.to_csv(PROCESSED_DIR / "csdh_nces_district_master.csv", index=False)

    for state_abbrev, state_name in STATE_NAMES.items():
        slug = state_name.lower()
        learning.loc[learning["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"csdh_district_learning_periods_{slug}.csv.gz",
            index=False,
        )
        case_rates.loc[case_rates["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"csdh_case_rates_by_district_zip_week_{slug}.csv.gz",
            index=False,
        )


def save_monthly_share_tables(monthly_shares: pd.DataFrame) -> None:
    """Write the district monthly share table and state-specific subsets."""
    monthly_shares.to_csv(PROCESSED_DIR / "csdh_district_monthly_shares.csv.gz", index=False)
    for state_abbrev, state_name in STATE_NAMES.items():
        slug = state_name.lower()
        monthly_shares.loc[monthly_shares["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"csdh_district_monthly_shares_{slug}.csv.gz",
            index=False,
        )


def load_processed_core_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously written cleaned learning, case, joined, and NCES master tables."""
    learning = pd.read_csv(
        PROCESSED_DIR / "csdh_district_learning_periods.csv.gz",
        parse_dates=["PeriodStartDate", "PeriodEndDate"],
        low_memory=False,
    )
    case_rates = pd.read_csv(
        PROCESSED_DIR / "csdh_case_rates_by_district_zip_week.csv.gz",
        parse_dates=["WeekStartDate", "WeekEndDate"],
        low_memory=False,
    )
    joined = pd.read_csv(
        PROCESSED_DIR / "csdh_learning_case_joined.csv.gz",
        parse_dates=["WeekStartDate", "WeekEndDate", "PeriodStartDate", "PeriodEndDate"],
        low_memory=False,
    )
    nces_master = pd.read_csv(PROCESSED_DIR / "csdh_nces_district_master.csv", low_memory=False)
    for frame in [learning, case_rates, joined, nces_master]:
        if "NCESDistrictID" in frame.columns:
            frame["NCESDistrictID"] = standardize_id(frame["NCESDistrictID"], width=7)
    for frame in [learning, case_rates, joined]:
        if "StateAssignedDistrictID" in frame.columns:
            frame["StateAssignedDistrictID"] = standardize_id(frame["StateAssignedDistrictID"])
        if "StateAbbrev" in frame.columns:
            frame["StateAbbrev"] = standardize_id(frame["StateAbbrev"])
    return learning, case_rates, joined, nces_master


def join_learning_to_case_rates(
    learning: pd.DataFrame,
    case_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Join district learning periods to district-week-ZIP case rates by overlapping dates."""
    joins: list[pd.DataFrame] = []
    week_calendar = (
        case_rates[["StateAbbrev", "wave_count", "WeekStartDate", "WeekEndDate"]]
        .drop_duplicates()
        .sort_values(["StateAbbrev", "WeekStartDate"])
    )

    for state_abbrev in sorted(case_rates["StateAbbrev"].dropna().unique()):
        learning_state = learning.loc[learning["StateAbbrev"] == state_abbrev].copy()
        case_state = case_rates.loc[case_rates["StateAbbrev"] == state_abbrev].copy()
        weeks_state = week_calendar.loc[week_calendar["StateAbbrev"] == state_abbrev].copy()
        if learning_state.empty or case_state.empty or weeks_state.empty:
            continue

        learning_weeks = learning_state.merge(weeks_state, on="StateAbbrev", how="left")
        overlap = (
            (learning_weeks["WeekStartDate"] <= learning_weeks["PeriodEndDate"])
            & (learning_weeks["WeekEndDate"] >= learning_weeks["PeriodStartDate"])
        )
        learning_weeks = learning_weeks.loc[overlap].copy()
        overlap_start = learning_weeks[["WeekStartDate", "PeriodStartDate"]].max(axis=1)
        overlap_end = learning_weeks[["WeekEndDate", "PeriodEndDate"]].min(axis=1)
        learning_weeks["overlap_days"] = (overlap_end - overlap_start).dt.days + 1
        learning_weeks = learning_weeks.sort_values(
            [
                "NCESDistrictID",
                "wave_count",
                "overlap_days",
                "PeriodStartDate",
                "PeriodEndDate",
            ],
            ascending=[True, True, False, False, False],
        )
        learning_weeks = learning_weeks.drop_duplicates(
            subset=["StateAbbrev", "NCESDistrictID", "wave_count"],
            keep="first",
        )

        joined = learning_weeks.merge(
            case_state,
            on=["StateAbbrev", "NCESDistrictID", "wave_count", "WeekStartDate", "WeekEndDate"],
            how="inner",
            suffixes=("_learning", "_case"),
        )
        joins.append(joined)

    joined = pd.concat(joins, ignore_index=True, sort=False) if joins else pd.DataFrame()
    if not joined.empty:
        joined = joined.sort_values(
            ["StateAbbrev", "NCESDistrictID", "WeekStartDate", "zip"]
        ).reset_index(drop=True)
    return joined


def join_monthly_shares_to_case_rates(
    case_rates: pd.DataFrame,
    monthly_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Join monthly in-person shares to the weekly case-rate panel."""
    panel = case_rates.copy()
    panel["Month"] = panel["WeekStartDate"].dt.year.astype(str) + "m" + panel["WeekStartDate"].dt.month.astype(str)
    panel = panel.merge(
        monthly_shares[
            [
                "StateAbbrev",
                "NCESDistrictID",
                "Month",
                "DistrictName",
                "share_inperson",
                "share_hybrid",
                "share_virtual",
                "MonthStartDate",
                "MonthEndDate",
            ]
        ],
        on=["StateAbbrev", "NCESDistrictID", "Month"],
        how="left",
        suffixes=("", "_monthly"),
    )
    panel["InterventionShare_pm1"] = np.where(
        panel["share_inperson"].notna() & (panel["share_inperson"] >= 0.5),
        1,
        np.where(panel["share_inperson"].notna(), -1, np.nan),
    ).astype("float")
    panel["InterventionShare_binary"] = panel["InterventionShare_pm1"].astype("Int64")
    return panel.sort_values(["StateAbbrev", "NCESDistrictID", "WeekStartDate", "zip"]).reset_index(drop=True)


def aggregate_case_panel_to_district_week(
    panel: pd.DataFrame,
    extra_first_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Aggregate one district-week-ZIP case panel to one row per district-week."""
    working = panel.copy()
    if "DistrictName" in working.columns and "lea_name" in working.columns:
        working["DistrictNameResolved"] = working["DistrictName"].fillna(working["lea_name"])
    elif "DistrictName" in working.columns:
        working["DistrictNameResolved"] = working["DistrictName"]
    elif "lea_name" in working.columns:
        working["DistrictNameResolved"] = working["lea_name"]
    else:
        working["DistrictNameResolved"] = pd.NA

    if "StateAssignedDistrictID" not in working.columns and "state_leaid" in working.columns:
        working["StateAssignedDistrictID"] = standardize_id(working["state_leaid"])

    numeric_sum_columns = ["total_tests", "total_positives", "total_negatives", "tot_zip_week", "tot_zip_pop"]
    for column in numeric_sum_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    weight = (
        pd.to_numeric(working["tot_zip_week"], errors="coerce").fillna(0.0)
        if "tot_zip_week" in working.columns
        else pd.Series(1.0, index=working.index, dtype=float)
    )
    value_columns = ["case_rate_per100k_zip", "case_rate_per100k_state", "positive_rate"]
    for column in value_columns:
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        valid_weight = weight.where(values.notna(), 0.0)
        working[f"{column}__weighted_value"] = values.fillna(0.0) * valid_weight
        working[f"{column}__weight"] = valid_weight
        working[column] = values

    group_cols = ["StateAbbrev", "NCESDistrictID"]
    if "wave_count" in working.columns:
        group_cols.append("wave_count")
    group_cols.extend(["WeekStartDate", "WeekEndDate"])

    agg_spec: dict[str, tuple[str, str]] = {
        "DistrictName": ("DistrictNameResolved", "first"),
    }
    if "StateAssignedDistrictID" in working.columns:
        agg_spec["StateAssignedDistrictID"] = ("StateAssignedDistrictID", "first")
    for column in extra_first_columns:
        if column in working.columns:
            agg_spec[column] = (column, "first")
    if "zip" in working.columns:
        agg_spec["zip_rows"] = ("zip", "nunique")
    for column in numeric_sum_columns:
        if column in working.columns:
            agg_spec[column] = (column, "sum")
    for column in value_columns:
        if column in working.columns:
            agg_spec[f"{column}__weighted_value"] = (f"{column}__weighted_value", "sum")
            agg_spec[f"{column}__weight"] = (f"{column}__weight", "sum")
            agg_spec[f"{column}__mean"] = (column, "mean")

    aggregated = working.groupby(group_cols, sort=True, dropna=False).agg(**agg_spec).reset_index()

    for column in value_columns:
        if f"{column}__weighted_value" not in aggregated.columns:
            continue
        weighted_value = aggregated[f"{column}__weighted_value"]
        valid_weight = aggregated[f"{column}__weight"]
        fallback_mean = aggregated[f"{column}__mean"]
        aggregated[column] = np.where(valid_weight > 1e-12, weighted_value / valid_weight, fallback_mean)
        aggregated = aggregated.drop(
            columns=[f"{column}__weighted_value", f"{column}__weight", f"{column}__mean"]
        )

    sort_columns = [column for column in ["StateAbbrev", "NCESDistrictID", "WeekStartDate"] if column in aggregated.columns]
    return aggregated.sort_values(sort_columns).reset_index(drop=True)


def load_edge_standardized_geometry(zip_path: Path, state_abbrev: str) -> gpd.GeoDataFrame:
    """Load the NCES EDGE composite school-district boundaries for one state."""
    gdf = gpd.read_file(f"zip://{zip_path}")
    gdf = gdf.loc[gdf["STATEFP"] == STATE_FIPS[state_abbrev]].copy()
    gdf["NCESDistrictID"] = standardize_id(gdf["GEOID"], width=7)
    gdf["district_type_edge"] = standardize_id(gdf["SDTYP"])
    gdf["district_name_source"] = gdf["NAME"].astype(str)
    gdf["district_name_normalized"] = gdf["district_name_source"].map(normalize_name)
    gdf["state_abbrev"] = state_abbrev
    gdf["state_name"] = STATE_NAMES[state_abbrev]
    return gdf.set_crs(4326, allow_override=True)


def build_massachusetts_official_geometry(paths: dict[str, Path]) -> gpd.GeoDataFrame:
    """Download and combine the Massachusetts official public-school-district layers."""
    frames: list[gpd.GeoDataFrame] = []
    for layer_id, layer_slug in MASS_OFFICIAL_LAYERS.items():
        raw_path = paths["geo_ma"] / f"massachusetts_public_school_districts_{layer_slug}.geojson"
        if raw_path.exists():
            layer_gdf = gpd.read_file(raw_path)
        else:
            layer_gdf = fetch_arcgis_geojson(f"{MASS_OFFICIAL_SERVICE}/{layer_id}")
            save_raw_geojson(layer_gdf, raw_path)
        layer_gdf["ma_layer_id"] = layer_id
        layer_gdf["ma_layer_name"] = layer_slug
        frames.append(layer_gdf)

    official = pd.concat(frames, ignore_index=True)
    official = gpd.GeoDataFrame(official, geometry="geometry", crs="EPSG:4326")
    official["ORG4CODE"] = standardize_id(official["ORG4CODE"], width=4)
    official["ORG8CODE"] = standardize_id(official["ORG8CODE"], width=8)
    official["district_name_normalized"] = official["DISTRICT_NAME"].map(normalize_name)
    official["state_name"] = "Massachusetts"
    official["state_abbrev"] = "MA"
    return official


def build_ohio_official_geometry(paths: dict[str, Path]) -> gpd.GeoDataFrame:
    """Download and standardize the Ohio official school-district boundary layer."""
    raw_path = paths["geo_oh"] / "ohio_school_districts_2025.geojson"
    if raw_path.exists():
        official = gpd.read_file(raw_path)
    else:
        official = fetch_arcgis_geojson(OHIO_OFFICIAL_LAYER)
        save_raw_geojson(official, raw_path)

    official["irn"] = standardize_id(official["irn"], width=6)
    official["taxid"] = standardize_id(official["taxid"], width=4)
    official["district_name_normalized"] = official["name"].map(normalize_name)
    official["state_name"] = "Ohio"
    official["state_abbrev"] = "OH"
    return official


def build_standardized_geometry(paths: dict[str, Path], state_abbrev: str) -> gpd.GeoDataFrame:
    """Load the EDGE school-district boundary composite for one state."""
    zip_path = paths["geo_edge"] / Path(EDGE_STANDARDIZED_URL).name
    return load_edge_standardized_geometry(zip_path, state_abbrev)


def parse_edge_dashboard(html: str, district_id: str) -> dict[str, object]:
    """Parse one EDGE district dashboard HTML page into a flat feature record."""
    record: dict[str, object] = {
        "NCESDistrictID": district_id,
        "edge_acsed_available": False,
        "edge_acsed_dashboard_url": f"{EDGE_ACS_DASHBOARD_BASE}/{district_id}",
        "edge_acsed_dashboard_years": "2018-22",
    }

    title_match = re.search(r'<div class="districtTtl">(.*?)</div>', html, flags=re.S)
    if title_match is None:
        return record
    district_title = html_lib.unescape(re.sub(r"<[^>]+>", "", title_match.group(1))).strip()
    if not district_title or "District Demographic Dashboard" in district_title:
        return record

    record["edge_acsed_available"] = True
    record["edge_acsed_district_title"] = district_title

    for match in re.finditer(
        r'<span class="dataHdr">(.*?)</span>\s*<br\s*/?>\s*<span class="dataNumber">\s*(.*?)\s*</span>',
        html,
        flags=re.S,
    ):
        header = html_lib.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
        header = header.split("|")[0].strip()
        if header in EDGE_TEXT_FIELD_MAP:
            record[EDGE_TEXT_FIELD_MAP[header]] = parse_numeric_text(match.group(2))

    for match in re.finditer(
        r'class="GaugeMeter".*?data-percent="([^"]+)".*?data-label="([^"]+)"',
        html,
        flags=re.S,
    ):
        percent = parse_numeric_text(match.group(1))
        label = html_lib.unescape(match.group(2)).split("|")[0].strip()
        if label in EDGE_GAUGE_FIELD_MAP:
            record[EDGE_GAUGE_FIELD_MAP[label]] = percent

    for function_name, column_names in EDGE_CHART_FIELD_MAP.items():
        match = re.search(rf"{function_name}\(([^)]*)\)", html)
        if match is None:
            continue
        values = [parse_numeric_text(token) for token in match.group(1).split(",")]
        for column_name, value in zip(column_names, values):
            record[column_name] = value

    return record


def fetch_edge_dashboard_html(
    district_id: str,
    cache_dir: Path,
    session: requests.Session | None = None,
    max_attempts: int = 3,
) -> str:
    """Fetch one EDGE district dashboard page, caching the raw HTML locally."""
    cache_path = cache_dir / f"{district_id}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    local_session = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = local_session.get(
                f"{EDGE_ACS_DASHBOARD_BASE}/{district_id}",
                timeout=90,
            )
            response.raise_for_status()
            cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except Exception as exc:  # pragma: no cover - network-retry path
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download EDGE dashboard for district {district_id}") from last_error


def build_edge_acsed_features(
    districts: pd.DataFrame,
    paths: dict[str, Path],
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch and parse EDGE ACS-ED dashboard features for the supplied districts."""
    source = (
        districts.loc[districts["StateAbbrev"].isin(STATE_NAMES), ["StateAbbrev", "NCESDistrictID"]]
        .drop_duplicates()
        .sort_values(["StateAbbrev", "NCESDistrictID"])
        .reset_index(drop=True)
    )
    cache_dir = paths["features_edge_dashboards"]
    records: list[dict[str, object]] = []

    def worker(row: tuple[str, str]) -> dict[str, object]:
        state_abbrev, district_id = row
        html = fetch_edge_dashboard_html(district_id, cache_dir)
        record = parse_edge_dashboard(html, district_id)
        record["StateAbbrev"] = state_abbrev
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(worker, (row.StateAbbrev, row.NCESDistrictID)): row.NCESDistrictID
            for row in source.itertuples(index=False)
        }
        for future in as_completed(futures):
            records.append(future.result())

    features = pd.DataFrame(records)
    if features.empty:
        return features
    features["NCESDistrictID"] = standardize_id(features["NCESDistrictID"], width=7)
    features = features.sort_values(["StateAbbrev", "NCESDistrictID"]).reset_index(drop=True)
    return features


def load_saipe_features(text_path: Path) -> pd.DataFrame:
    """Load one state SAIPE fixed-width school-district file."""
    saipe = pd.read_fwf(
        text_path,
        colspecs=[(0, 2), (3, 8), (9, 81), (82, 90), (91, 99), (100, 108)],
        names=[
            "fips_state",
            "district_code_5",
            "district_name",
            "saipe_total_population",
            "saipe_population_5_17",
            "saipe_children_5_17_in_poverty",
        ],
        dtype="string",
    )
    saipe["fips_state"] = standardize_id(saipe["fips_state"], width=2)
    saipe["district_code_5"] = standardize_id(saipe["district_code_5"], width=5)
    saipe["NCESDistrictID"] = saipe["fips_state"] + saipe["district_code_5"]
    saipe["StateAbbrev"] = saipe["fips_state"].map(FIPS_TO_STATE)
    for column in [
        "saipe_total_population",
        "saipe_population_5_17",
        "saipe_children_5_17_in_poverty",
    ]:
        saipe[column] = pd.to_numeric(saipe[column], errors="coerce")
    saipe["saipe_child_poverty_rate"] = np.where(
        saipe["saipe_population_5_17"] > 0,
        saipe["saipe_children_5_17_in_poverty"] / saipe["saipe_population_5_17"],
        np.nan,
    )
    return saipe.sort_values(["StateAbbrev", "NCESDistrictID"]).reset_index(drop=True)


def build_feature_basis_table(
    districts: pd.DataFrame,
    edge_features: pd.DataFrame,
    saipe_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create a clean district-level feature table for later external-field construction."""
    base = (
        districts.loc[districts["StateAbbrev"].isin(STATE_NAMES), [
            "StateAbbrev",
            "NCESDistrictID",
            "StateAssignedDistrictID",
            "DistrictName",
            "DistrictNameNormalized",
        ]]
        .drop_duplicates()
        .sort_values(["StateAbbrev", "NCESDistrictID"])
        .reset_index(drop=True)
    )
    merged = base.merge(
        edge_features.drop(columns=["edge_acsed_dashboard_url"], errors="ignore"),
        on=["StateAbbrev", "NCESDistrictID"],
        how="left",
    )
    merged = merged.merge(
        saipe_features.drop(columns=["district_name"], errors="ignore"),
        on=["StateAbbrev", "NCESDistrictID"],
        how="left",
    )
    return merged


def write_feature_tables(
    edge_features: pd.DataFrame,
    saipe_features: pd.DataFrame,
    feature_basis: pd.DataFrame,
) -> None:
    """Write combined and state-specific district feature tables."""
    edge_features.to_csv(PROCESSED_DIR / "edge_acsed_district_features.csv.gz", index=False)
    saipe_features.to_csv(PROCESSED_DIR / "saipe_district_features.csv.gz", index=False)
    feature_basis.to_csv(PROCESSED_DIR / "district_feature_basis.csv.gz", index=False)

    for state_abbrev, state_name in STATE_NAMES.items():
        slug = state_name.lower()
        edge_features.loc[edge_features["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"edge_acsed_district_features_{slug}.csv.gz",
            index=False,
        )
        saipe_features.loc[saipe_features["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"saipe_district_features_{slug}.csv.gz",
            index=False,
        )
        feature_basis.loc[feature_basis["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"district_feature_basis_{slug}.csv.gz",
            index=False,
        )


def load_processed_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously written EDGE, SAIPE, and combined district feature tables."""
    edge_features = pd.read_csv(PROCESSED_DIR / "edge_acsed_district_features.csv.gz", low_memory=False)
    saipe_features = pd.read_csv(PROCESSED_DIR / "saipe_district_features.csv.gz", low_memory=False)
    feature_basis = pd.read_csv(PROCESSED_DIR / "district_feature_basis.csv.gz", low_memory=False)
    for frame in [edge_features, saipe_features, feature_basis]:
        if "NCESDistrictID" in frame.columns:
            frame["NCESDistrictID"] = standardize_id(frame["NCESDistrictID"], width=7)
        if "StateAbbrev" in frame.columns:
            frame["StateAbbrev"] = standardize_id(frame["StateAbbrev"])
    return edge_features, saipe_features, feature_basis


def write_feature_dictionary() -> None:
    """Write a compact dictionary for the district-level external-field feature tables."""
    rows: list[dict[str, str]] = []
    for label, column in EDGE_TEXT_FIELD_MAP.items():
        rows.append(
            {
                "column_name": column,
                "source": "EDGE ACS-ED dashboard",
                "description": label,
                "feature_group": "core community counts",
            }
        )
    for label, column in EDGE_GAUGE_FIELD_MAP.items():
        rows.append(
            {
                "column_name": column,
                "source": "EDGE ACS-ED dashboard",
                "description": f"{label} percent",
                "feature_group": "race or family composition",
            }
        )
    chart_descriptions = {
        "edge_acsed_pct_english_only": "Children who speak only English",
        "edge_acsed_pct_english_less_than_well": "Children who speak English less than well",
        "edge_acsed_pct_english_very_well": "Children who speak English very well",
        "edge_acsed_pct_less_than_high_school": "Adults with less than high school attainment",
        "edge_acsed_pct_high_school": "Adults with high school attainment",
        "edge_acsed_pct_some_college": "Adults with some college attainment",
        "edge_acsed_pct_bachelors_or_higher": "Adults with a bachelor's degree or higher",
        "edge_acsed_pct_in_labor_force": "Adults in the labor force",
        "edge_acsed_pct_not_in_labor_force": "Adults not in the labor force",
        "edge_acsed_pct_management_occupations": "Workers in management occupations",
        "edge_acsed_pct_service_occupations": "Workers in service occupations",
        "edge_acsed_pct_sales_office_occupations": "Workers in sales and office occupations",
        "edge_acsed_pct_natural_resources_construction_maintenance_occupations": "Workers in natural resources, construction, and maintenance occupations",
        "edge_acsed_pct_production_transportation_material_moving_occupations": "Workers in production, transportation, and material moving occupations",
        "edge_acsed_pct_households_snap": "Households receiving SNAP/Food Stamps",
        "edge_acsed_pct_households_broadband": "Households with broadband internet",
        "edge_acsed_pct_below_poverty": "Population below poverty",
    }
    for column, description in chart_descriptions.items():
        rows.append(
            {
                "column_name": column,
                "source": "EDGE ACS-ED dashboard",
                "description": description,
                "feature_group": "education, labor, poverty, or broadband",
            }
        )
    for column, description in {
        "saipe_total_population": "SAIPE total population",
        "saipe_population_5_17": "SAIPE population ages 5 to 17",
        "saipe_children_5_17_in_poverty": "SAIPE children ages 5 to 17 in poverty",
        "saipe_child_poverty_rate": "SAIPE poverty rate among children ages 5 to 17",
    }.items():
        rows.append(
            {
                "column_name": column,
                "source": "Census SAIPE 2021 school district estimates",
                "description": description,
                "feature_group": "supplemental poverty and population",
            }
        )
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "district_feature_dictionary.csv", index=False)


def build_massachusetts_crosswalk(
    districts: pd.DataFrame,
    official: gpd.GeoDataFrame,
    standardized_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Create Massachusetts district crosswalks for official and EDGE geometries."""
    ma = districts.loc[districts["StateAbbrev"] == "MA", [
        "NCESDistrictID",
        "StateAssignedDistrictID",
        "DistrictName",
        "DistrictNameNormalized",
    ]].drop_duplicates().copy()
    ma["official_org8code"] = ma["StateAssignedDistrictID"].str.zfill(8)
    ma["official_org4code"] = ma["official_org8code"].str[:4]

    official_min = official[["ORG4CODE", "ORG8CODE", "DISTRICT_NAME", "DISTRICT_TYPE", "district_name_normalized"]].drop_duplicates()
    crosswalk = ma.merge(
        official_min,
        left_on="official_org8code",
        right_on="ORG8CODE",
        how="left",
    )

    unmatched = crosswalk["ORG8CODE"].isna()
    official_name_lookup = (
        official_min.groupby("district_name_normalized")
        .filter(lambda frame: len(frame) == 1)
        .drop_duplicates("district_name_normalized")
        .set_index("district_name_normalized")
    )
    for idx in crosswalk.index[unmatched]:
        name_key = crosswalk.at[idx, "DistrictNameNormalized"]
        if name_key in official_name_lookup.index:
            matched = official_name_lookup.loc[name_key]
            crosswalk.at[idx, "ORG4CODE"] = matched["ORG4CODE"]
            crosswalk.at[idx, "ORG8CODE"] = matched["ORG8CODE"]
            crosswalk.at[idx, "DISTRICT_NAME"] = matched["DISTRICT_NAME"]
            crosswalk.at[idx, "DISTRICT_TYPE"] = matched["DISTRICT_TYPE"]

    edge_lookup = standardized_geometry[["NCESDistrictID", "district_name_source"]].drop_duplicates()
    crosswalk = crosswalk.merge(edge_lookup, on="NCESDistrictID", how="left")
    crosswalk["official_match"] = crosswalk["ORG8CODE"].notna()
    crosswalk["edge_match"] = crosswalk["district_name_source"].notna()
    crosswalk["standardized_match"] = crosswalk["edge_match"]
    crosswalk["state_abbrev"] = "MA"
    return crosswalk.sort_values(["NCESDistrictID"]).reset_index(drop=True)


def build_ohio_crosswalk(
    districts: pd.DataFrame,
    official: gpd.GeoDataFrame,
    standardized_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Create Ohio district crosswalks for official and EDGE geometries."""
    oh = districts.loc[districts["StateAbbrev"] == "OH", [
        "NCESDistrictID",
        "StateAssignedDistrictID",
        "DistrictName",
        "DistrictNameNormalized",
    ]].drop_duplicates().copy()
    oh["official_irn"] = oh["StateAssignedDistrictID"].str.zfill(6)

    official_min = official[["irn", "taxid", "name", "district_name_normalized"]].drop_duplicates()
    crosswalk = oh.merge(
        official_min,
        left_on="official_irn",
        right_on="irn",
        how="left",
    )

    unmatched = crosswalk["irn"].isna()
    official_name_lookup = (
        official_min.groupby("district_name_normalized")
        .filter(lambda frame: len(frame) == 1)
        .drop_duplicates("district_name_normalized")
        .set_index("district_name_normalized")
    )
    for idx in crosswalk.index[unmatched]:
        name_key = crosswalk.at[idx, "DistrictNameNormalized"]
        if name_key in official_name_lookup.index:
            matched = official_name_lookup.loc[name_key]
            crosswalk.at[idx, "irn"] = matched["irn"]
            crosswalk.at[idx, "taxid"] = matched["taxid"]
            crosswalk.at[idx, "name"] = matched["name"]

    edge_lookup = standardized_geometry[["NCESDistrictID", "district_name_source"]].drop_duplicates()
    crosswalk = crosswalk.merge(edge_lookup, on="NCESDistrictID", how="left")
    crosswalk["official_match"] = crosswalk["irn"].notna()
    crosswalk["edge_match"] = crosswalk["district_name_source"].notna()
    crosswalk["standardized_match"] = crosswalk["edge_match"]
    crosswalk["state_abbrev"] = "OH"
    return crosswalk.sort_values(["NCESDistrictID"]).reset_index(drop=True)


def merge_geometry_with_crosswalk(
    geometry: gpd.GeoDataFrame,
    crosswalk: pd.DataFrame,
    geometry_kind: str,
) -> gpd.GeoDataFrame:
    """Attach NCES district IDs to one geometry source using the prepared crosswalk."""
    if geometry_kind == "official" and "ORG8CODE" in geometry.columns:
        merged = geometry.merge(
            crosswalk[["NCESDistrictID", "ORG8CODE"]].dropna().drop_duplicates(),
            on="ORG8CODE",
            how="inner",
        )
    elif geometry_kind == "official" and "irn" in geometry.columns:
        merged = geometry.merge(
            crosswalk[["NCESDistrictID", "irn"]].dropna().drop_duplicates(),
            on="irn",
            how="inner",
        )
    else:
        merged = geometry.merge(
            crosswalk[["NCESDistrictID"]].dropna().drop_duplicates(),
            on="NCESDistrictID",
            how="inner",
        )
    return merged.drop_duplicates(subset=["NCESDistrictID"]).reset_index(drop=True)


def build_centroid_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute projected centroids for one district geometry file."""
    projected = gdf.to_crs(2163)
    centroids = projected.geometry.centroid
    centroids_ll = gpd.GeoSeries(centroids, crs=projected.crs).to_crs(4326)
    return pd.DataFrame(
        {
            "NCESDistrictID": gdf["NCESDistrictID"].to_numpy(),
            "centroid_x": centroids.x.to_numpy(),
            "centroid_y": centroids.y.to_numpy(),
            "centroid_lon": centroids_ll.x.to_numpy(),
            "centroid_lat": centroids_ll.y.to_numpy(),
        }
    )


def write_network_artifacts(
    state_slug: str,
    source_slug: str,
    gdf: gpd.GeoDataFrame,
) -> list[dict[str, object]]:
    """Write centroid and edge-list files for one state/geometry combination."""
    centroids = build_centroid_table(gdf)
    contiguity = build_touching_edge_list(
        gdf,
        id_column="NCESDistrictID",
        neighbor_column="neighbor_id",
    )
    knn, kernel = build_knn_and_kernel_edges(
        centroids,
        id_column="NCESDistrictID",
        x_column="centroid_x",
        y_column="centroid_y",
        k=8,
    )

    centroids.to_csv(PROCESSED_DIR / f"{state_slug}_{source_slug}_centroids.csv", index=False)
    contiguity.to_csv(
        PROCESSED_DIR / f"{state_slug}_{source_slug}_contiguity_adjacency.csv.gz",
        index=False,
    )
    knn.to_csv(
        PROCESSED_DIR / f"{state_slug}_{source_slug}_knn_8_adjacency.csv.gz",
        index=False,
    )
    kernel.to_csv(
        PROCESSED_DIR / f"{state_slug}_{source_slug}_distance_kernel_8_adjacency.csv.gz",
        index=False,
    )

    nodes = centroids["NCESDistrictID"].tolist()
    summaries = []
    for network_name, edge_frame in [
        ("contiguity", contiguity),
        ("knn_8", knn),
        ("distance_kernel_8", kernel),
    ]:
        summaries.append(
            {
                "state": state_slug,
                "geometry_source": source_slug,
                "network_type": network_name,
                "node_count": len(nodes),
                "edge_count": int(len(edge_frame)),
                "connected_components": int(
                    count_connected_components(nodes, edge_frame, "NCESDistrictID", "neighbor_id")
                ),
            }
        )
    return summaries


def build_state_join_coverage(
    joined: pd.DataFrame,
    crosswalks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize official and EDGE-standardized geometry coverage."""
    rows: list[dict[str, object]] = []
    for state_abbrev, crosswalk in crosswalks.items():
        joined_state = joined.loc[joined["StateAbbrev"] == state_abbrev].copy()
        joined_with_flags = joined_state.merge(
            crosswalk[
                [
                    "NCESDistrictID",
                    "official_match",
                    "standardized_match",
                    "edge_match",
                ]
            ].drop_duplicates(),
            on="NCESDistrictID",
            how="left",
        )
        rows.append(
            {
                "state_abbrev": state_abbrev,
                "state_name": STATE_NAMES[state_abbrev],
                "district_count": int(crosswalk["NCESDistrictID"].nunique()),
                "official_match_districts": int(crosswalk["official_match"].sum()),
                "standardized_match_districts": int(crosswalk["standardized_match"].sum()),
                "edge_match_districts": int(crosswalk["edge_match"].sum()),
                "official_match_pct_districts": float(crosswalk["official_match"].mean()),
                "standardized_match_pct_districts": float(crosswalk["standardized_match"].mean()),
                "edge_match_pct_districts": float(crosswalk["edge_match"].mean()),
                "joined_rows": int(len(joined_state)),
                "official_match_pct_joined_rows": float(joined_with_flags["official_match"].fillna(False).mean()) if len(joined_with_flags) else 0.0,
                "standardized_match_pct_joined_rows": float(joined_with_flags["standardized_match"].fillna(False).mean()) if len(joined_with_flags) else 0.0,
                "edge_match_pct_joined_rows": float(joined_with_flags["edge_match"].fillna(False).mean()) if len(joined_with_flags) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_feature_join_coverage(
    districts: pd.DataFrame,
    edge_features: pd.DataFrame,
    saipe_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize district-level feature availability for EDGE ACS-ED and SAIPE."""
    rows: list[dict[str, object]] = []
    for state_abbrev, state_name in STATE_NAMES.items():
        state_districts = (
            districts.loc[districts["StateAbbrev"] == state_abbrev, ["NCESDistrictID"]]
            .drop_duplicates()
            .copy()
        )
        edge_state = edge_features.loc[edge_features["StateAbbrev"] == state_abbrev].copy()
        saipe_state = saipe_features.loc[saipe_features["StateAbbrev"] == state_abbrev].copy()

        edge_ids = set(
            edge_state.loc[edge_state["edge_acsed_available"].fillna(False), "NCESDistrictID"].dropna()
        )
        district_ids = set(state_districts["NCESDistrictID"].dropna())
        edge_ids &= district_ids
        saipe_ids = set(saipe_state["NCESDistrictID"].dropna()) & district_ids

        rows.append(
            {
                "state_abbrev": state_abbrev,
                "state_name": state_name,
                "district_count": int(len(state_districts)),
                "edge_acsed_match_districts": int(len(edge_ids)),
                "edge_acsed_match_pct_districts": float(len(edge_ids) / len(state_districts)) if len(state_districts) else 0.0,
                "saipe_match_districts": int(len(saipe_ids)),
                "saipe_match_pct_districts": float(len(saipe_ids) / len(state_districts)) if len(state_districts) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_state_dataset_summary(
    learning: pd.DataFrame,
    case_rates: pd.DataFrame,
    joined: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize row counts, district counts, and date ranges for Ohio and Massachusetts."""
    rows: list[dict[str, object]] = []
    for state_abbrev, state_name in STATE_NAMES.items():
        learning_state = learning.loc[learning["StateAbbrev"] == state_abbrev]
        case_state = case_rates.loc[case_rates["StateAbbrev"] == state_abbrev]
        joined_state = joined.loc[joined["StateAbbrev"] == state_abbrev]
        rows.append(
            {
                "state_abbrev": state_abbrev,
                "state_name": state_name,
                "learning_rows": int(len(learning_state)),
                "learning_districts": int(learning_state["NCESDistrictID"].nunique()),
                "learning_start_date": learning_state["PeriodStartDate"].min(),
                "learning_end_date": learning_state["PeriodEndDate"].max(),
                "case_rows": int(len(case_state)),
                "case_districts": int(case_state["NCESDistrictID"].nunique()),
                "case_start_date": case_state["WeekStartDate"].min(),
                "case_end_date": case_state["WeekEndDate"].max(),
                "joined_rows": int(len(joined_state)),
                "joined_districts": int(joined_state["NCESDistrictID"].nunique()),
                "joined_start_date": joined_state["WeekStartDate"].min() if len(joined_state) else pd.NaT,
                "joined_end_date": joined_state["WeekEndDate"].max() if len(joined_state) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def write_summary_json(
    learning: pd.DataFrame,
    case_rates: pd.DataFrame,
    joined: pd.DataFrame,
    monthly_shares: pd.DataFrame,
    monthly_joined: pd.DataFrame,
    mass_crosswalk: pd.DataFrame,
    ohio_crosswalk: pd.DataFrame,
    network_summary: pd.DataFrame,
    edge_features: pd.DataFrame,
    saipe_features: pd.DataFrame,
) -> None:
    """Write a compact JSON summary of the processed package."""
    summary = {
        "learning_rows": int(len(learning)),
        "learning_districts": int(learning["NCESDistrictID"].nunique()),
        "case_rows": int(len(case_rates)),
        "case_districts": int(case_rates["NCESDistrictID"].nunique()),
        "monthly_share_rows": int(len(monthly_shares)),
        "monthly_share_districts": int(monthly_shares["NCESDistrictID"].nunique()) if not monthly_shares.empty else 0,
        "monthly_joined_rows": int(len(monthly_joined)),
        "monthly_joined_districts": int(monthly_joined["NCESDistrictID"].nunique()) if not monthly_joined.empty else 0,
        "joined_rows": int(len(joined)),
        "joined_districts": int(joined["NCESDistrictID"].nunique()) if not joined.empty else 0,
        "states_in_learning": sorted(learning["StateAbbrev"].dropna().unique().tolist()),
        "states_in_case_rates": sorted(case_rates["StateAbbrev"].dropna().unique().tolist()),
        "ohio_official_match_districts": int(ohio_crosswalk["official_match"].sum()),
        "ohio_standardized_match_districts": int(ohio_crosswalk["standardized_match"].sum()),
        "massachusetts_official_match_districts": int(mass_crosswalk["official_match"].sum()),
        "massachusetts_standardized_match_districts": int(mass_crosswalk["standardized_match"].sum()),
        "edge_acsed_rows": int(len(edge_features)),
        "edge_acsed_available_districts": int(edge_features.loc[edge_features["edge_acsed_available"].fillna(False), "NCESDistrictID"].nunique()) if not edge_features.empty else 0,
        "saipe_rows": int(len(saipe_features)),
        "saipe_districts": int(saipe_features["NCESDistrictID"].nunique()) if not saipe_features.empty else 0,
        "network_rows": int(len(network_summary)),
    }
    (PROCESSED_DIR / "processing_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    """Download, preprocess, and summarize the Ohio and Massachusetts CSDH data package."""
    args = parse_args()
    paths = ensure_directories()

    csdh_targets = {
        "district_monthly_shares": paths["csdh"] / "District_Monthly_Shares_03.08.23.csv",
        "learning_csv_zip": paths["csdh"] / "CSDH_District_Files_-_CSV.zip",
        "community_case_rate_zip": paths["csdh"] / "Matched_Districts_and_Case_Rates.zip",
        "community_case_rate_codebook": paths["csdh"] / "Cate_Rate_Codebook.xlsx",
        "nces_district_demographics": paths["csdh"] / "NCES_2020-2021_District_Demographics.csv",
        "district_merge_code_do": paths["csdh"] / "district_code_to_combine_learning_model.do",
    }
    for key, destination in csdh_targets.items():
        download_if_missing(CSDH_URLS[key], destination)
    monthly_shares = load_monthly_shares(csdh_targets["district_monthly_shares"])
    save_monthly_share_tables(monthly_shares)

    download_if_missing(EDGE_STANDARDIZED_URL, paths["geo_edge"] / Path(EDGE_STANDARDIZED_URL).name)
    for key in ["ohio", "massachusetts", "layout"]:
        download_if_missing(SAIPE_URLS[key], paths["features_saipe"] / Path(SAIPE_URLS[key]).name)

    if args.reuse_processed_tables and (PROCESSED_DIR / "csdh_learning_case_joined.csv.gz").exists():
        learning, case_rates, joined, nces_master = load_processed_core_tables()
    else:
        learning = load_learning_periods(csdh_targets["learning_csv_zip"])
        case_rates = load_case_rates(csdh_targets["community_case_rate_zip"])
        nces_master = load_nces_district_master(csdh_targets["nces_district_demographics"])
        save_core_tables(learning, case_rates, nces_master)

        joined = join_learning_to_case_rates(learning, case_rates)
        joined.to_csv(PROCESSED_DIR / "csdh_learning_case_joined.csv.gz", index=False)
        for state_abbrev, state_name in STATE_NAMES.items():
            joined.loc[joined["StateAbbrev"] == state_abbrev].to_csv(
                PROCESSED_DIR / f"csdh_learning_case_joined_{state_name.lower()}.csv.gz",
                index=False,
            )

    aggregate_case_panel_to_district_week(
        joined,
        extra_first_columns=("LearningModel", "PeriodStartDate", "PeriodEndDate"),
    ).to_csv(
        PROCESSED_DIR / "csdh_learning_case_joined_district_week.csv.gz",
        index=False,
    )

    monthly_joined = join_monthly_shares_to_case_rates(case_rates, monthly_shares)
    monthly_joined.to_csv(PROCESSED_DIR / "csdh_learning_case_joined_monthly_shares.csv.gz", index=False)
    monthly_joined_district_week = aggregate_case_panel_to_district_week(
        monthly_joined,
        extra_first_columns=(
            "Month",
            "MonthStartDate",
            "MonthEndDate",
            "share_inperson",
            "share_hybrid",
            "share_virtual",
            "InterventionShare_pm1",
            "InterventionShare_binary",
        ),
    )
    monthly_joined_district_week.to_csv(
        PROCESSED_DIR / "csdh_learning_case_joined_monthly_shares_district_week.csv.gz",
        index=False,
    )
    for state_abbrev, state_name in STATE_NAMES.items():
        monthly_joined.loc[monthly_joined["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"csdh_learning_case_joined_monthly_shares_{state_name.lower()}.csv.gz",
            index=False,
        )
        monthly_joined_district_week.loc[monthly_joined_district_week["StateAbbrev"] == state_abbrev].to_csv(
            PROCESSED_DIR / f"csdh_learning_case_joined_monthly_shares_district_week_{state_name.lower()}.csv.gz",
            index=False,
        )

    mass_official = build_massachusetts_official_geometry(paths)
    ohio_official = build_ohio_official_geometry(paths)
    mass_standardized = build_standardized_geometry(paths, "MA")
    ohio_standardized = build_standardized_geometry(paths, "OH")

    mass_crosswalk = build_massachusetts_crosswalk(
        learning,
        mass_official,
        mass_standardized,
    )
    ohio_crosswalk = build_ohio_crosswalk(
        learning,
        ohio_official,
        ohio_standardized,
    )
    mass_crosswalk.to_csv(PROCESSED_DIR / "massachusetts_district_crosswalk.csv", index=False)
    ohio_crosswalk.to_csv(PROCESSED_DIR / "ohio_district_crosswalk.csv", index=False)

    mass_official_joined = merge_geometry_with_crosswalk(mass_official, mass_crosswalk, "official")
    ohio_official_joined = merge_geometry_with_crosswalk(ohio_official, ohio_crosswalk, "official")
    mass_standardized_joined = merge_geometry_with_crosswalk(mass_standardized, mass_crosswalk, "standardized")
    ohio_standardized_joined = merge_geometry_with_crosswalk(ohio_standardized, ohio_crosswalk, "standardized")

    mass_official_joined.to_file(
        PROCESSED_DIR / "massachusetts_districts_official.gpkg",
        layer="districts",
        driver="GPKG",
    )
    mass_standardized_joined.to_file(
        PROCESSED_DIR / "massachusetts_districts_standardized.gpkg",
        layer="districts",
        driver="GPKG",
    )
    ohio_official_joined.to_file(
        PROCESSED_DIR / "ohio_districts_official.gpkg",
        layer="districts",
        driver="GPKG",
    )
    ohio_standardized_joined.to_file(
        PROCESSED_DIR / "ohio_districts_standardized.gpkg",
        layer="districts",
        driver="GPKG",
    )

    if args.reuse_processed_networks and (PROCESSED_DIR / "state_network_summary.csv").exists():
        network_summary = pd.read_csv(PROCESSED_DIR / "state_network_summary.csv")
    else:
        network_rows: list[dict[str, object]] = []
        network_rows.extend(write_network_artifacts("massachusetts", "official", mass_official_joined))
        network_rows.extend(write_network_artifacts("massachusetts", "standardized", mass_standardized_joined))
        network_rows.extend(write_network_artifacts("ohio", "official", ohio_official_joined))
        network_rows.extend(write_network_artifacts("ohio", "standardized", ohio_standardized_joined))
        network_summary = pd.DataFrame(network_rows)
        network_summary.to_csv(PROCESSED_DIR / "state_network_summary.csv", index=False)

    if args.reuse_processed_features and (PROCESSED_DIR / "edge_acsed_district_features.csv.gz").exists():
        edge_features, saipe_features, feature_basis = load_processed_feature_tables()
    else:
        edge_features = build_edge_acsed_features(learning, paths)
        saipe_features = pd.concat(
            [
                load_saipe_features(paths["features_saipe"] / Path(SAIPE_URLS["ohio"]).name),
                load_saipe_features(paths["features_saipe"] / Path(SAIPE_URLS["massachusetts"]).name),
            ],
            ignore_index=True,
            sort=False,
        )
        feature_basis = build_feature_basis_table(learning, edge_features, saipe_features)
        write_feature_tables(edge_features, saipe_features, feature_basis)
        write_feature_dictionary()

    build_state_join_coverage(
        joined,
        {"MA": mass_crosswalk, "OH": ohio_crosswalk},
    ).to_csv(PROCESSED_DIR / "state_join_coverage.csv", index=False)
    build_feature_join_coverage(
        learning,
        edge_features,
        saipe_features,
    ).to_csv(PROCESSED_DIR / "state_feature_coverage.csv", index=False)

    build_state_dataset_summary(
        learning,
        case_rates,
        joined,
    ).to_csv(PROCESSED_DIR / "state_dataset_summary.csv", index=False)

    write_summary_json(
        learning,
        case_rates,
        joined,
        monthly_shares,
        monthly_joined,
        mass_crosswalk,
        ohio_crosswalk,
        network_summary,
        edge_features,
        saipe_features,
    )


if __name__ == "__main__":
    main()
