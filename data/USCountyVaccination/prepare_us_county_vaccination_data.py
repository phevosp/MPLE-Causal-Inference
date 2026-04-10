"""Prepare the nationwide US county vaccination real-data package."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ACS_2021_COUNTY_ENDPOINTS,
    BANSAL_VACCINATION_URL,
    CDC_SVI_2022_PR_COUNTY_URL,
    CDC_SVI_2022_US_COUNTY_URL,
    CDC_VACCINATION_URL,
    CORE_END_DATE,
    CORE_START_DATE,
    PROCESSED_DIR,
    TIGER_2021_COUNTY_URL,
    TIGER_2022_COUNTY_URL,
    UNIT_LABEL,
    USDA_ERS_RUCC_2023_URL,
    add_iso_week_window,
    dump_json,
    ensure_directories,
    standardize_fips,
)
from data_utils import (  # noqa: E402
    build_knn_and_kernel_edges,
    build_touching_edge_list,
    count_connected_components,
    download_if_missing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the nationwide US county vaccination package."
    )
    parser.add_argument(
        "--vaccination_source",
        choices=["bansal", "cdc"],
        default="cdc",
        help="Vaccination source to normalize into the canonical weekly county table.",
    )
    parser.add_argument(
        "--reuse_processed_tables",
        action="store_true",
        help="Reuse already-written processed daily/weekly/panel CSV tables when present.",
    )
    parser.add_argument(
        "--reuse_processed_networks",
        action="store_true",
        help="Reuse already-written geometry, centroid, and adjacency artifacts when present.",
    )
    parser.add_argument(
        "--reuse_processed_features",
        action="store_true",
        help="Reuse already-written county feature tables when present.",
    )
    return parser.parse_args()


def fetch_json_table(
    url: str,
    params: dict[str, str] | None = None,
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    if response.status_code == 204 or not response.text.strip():
        if empty_columns is None:
            raise ValueError(f"Empty response from {url} for params={params}.")
        return pd.DataFrame(columns=empty_columns)
    payload = response.json()
    return pd.DataFrame(payload[1:], columns=payload[0])


def load_or_download_nyt(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_nyt"] / "us-counties.csv"
    download_if_missing("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv", target)
    nyt = pd.read_csv(target, dtype={"fips": str})
    nyt = nyt.loc[nyt["county"].ne("Unknown") & nyt["fips"].notna()].copy()
    nyt["fips"] = standardize_fips(nyt["fips"])
    nyt["date"] = pd.to_datetime(nyt["date"])
    nyt["cases"] = pd.to_numeric(nyt["cases"], errors="coerce").fillna(0.0)
    nyt["deaths"] = pd.to_numeric(nyt["deaths"], errors="coerce").fillna(0.0)
    nyt = nyt.sort_values(["fips", "date"]).reset_index(drop=True)
    nyt["new_cases"] = nyt.groupby("fips", sort=False)["cases"].diff().fillna(nyt["cases"]).clip(lower=0.0)
    nyt["new_deaths"] = nyt.groupby("fips", sort=False)["deaths"].diff().fillna(nyt["deaths"]).clip(lower=0.0)
    return nyt


def load_or_download_bansal(paths: dict[str, Path]) -> tuple[pd.DataFrame, str]:
    target = paths["raw_vaccination"] / "bansal_data_county_timeseries.csv"
    download_if_missing(BANSAL_VACCINATION_URL, target)
    vacc = pd.read_csv(target)
    vacc["fips"] = standardize_fips(vacc["COUNTY"])
    vacc["DATE"] = pd.to_datetime(vacc["DATE"], errors="coerce")
    vacc["WEEK"] = pd.to_numeric(vacc["WEEK"], errors="coerce").astype("Int64")
    vacc["YEAR"] = pd.to_numeric(vacc["YEAR"], errors="coerce").astype("Int64")
    vacc["CASES"] = pd.to_numeric(vacc["CASES"], errors="coerce")
    vacc["POPN"] = pd.to_numeric(vacc["POPN"], errors="coerce")
    vacc = vacc.loc[vacc["fips"].str.len() == 5].copy()
    vacc = vacc.loc[vacc["GEOFLAG"].eq("County")].copy()
    vacc = vacc.dropna(subset=["DATE", "WEEK", "YEAR"])

    index_cols = ["fips", "STATE_NAME", "COUNTY_NAME", "DATE", "WEEK", "YEAR", "POPN"]
    wide = (
        vacc.pivot_table(index=index_cols, columns="CASE_TYPE", values="CASES", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )
    wide = wide.rename(
        columns={
            "STATE_NAME": "state_name",
            "COUNTY_NAME": "county",
            "DATE": "source_date",
            "WEEK": "iso_week",
            "YEAR": "iso_year",
            "POPN": "population",
            "Complete": "complete_count",
            "Complete Coverage": "complete_cov",
            "Partial": "partial_count",
            "Partial Coverage": "partial_cov",
            "Booster": "booster_count",
            "Booster Coverage": "booster_cov",
        }
    )
    wide = add_iso_week_window(wide)
    wide = (
        wide.sort_values(["fips", "iso_year", "iso_week", "source_date"])
        .groupby(["fips", "iso_year", "iso_week", "WeekStartDate", "WeekEndDate"], as_index=False)
        .agg(
            source_date=("source_date", "max"),
            state_name=("state_name", "last"),
            county=("county", "last"),
            population=("population", "last"),
            complete_count=("complete_count", "last"),
            complete_cov=("complete_cov", "last"),
            partial_count=("partial_count", "last"),
            partial_cov=("partial_cov", "last"),
            booster_count=("booster_count", "last"),
            booster_cov=("booster_cov", "last"),
        )
    )
    wide = wide.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    for metric in ["complete_cov", "partial_cov", "booster_cov"]:
        wide[f"{metric}_delta"] = wide.groupby("fips", sort=False)[metric].diff()
    wide["vaccination_source"] = "bansal"
    wide = wide.loc[
        (wide["WeekEndDate"] >= CORE_START_DATE) & (wide["WeekEndDate"] <= CORE_END_DATE)
    ].copy()
    return wide, "bansal"


def load_or_download_cdc(paths: dict[str, Path]) -> tuple[pd.DataFrame, str]:
    target = paths["raw_vaccination"] / "cdc_county_vaccinations.csv"
    download_if_missing(CDC_VACCINATION_URL, target)
    vacc = pd.read_csv(target, dtype={"fips": str})
    vacc = vacc.loc[vacc["fips"].notna()].copy()
    vacc["fips"] = standardize_fips(vacc["fips"])
    vacc["date"] = pd.to_datetime(vacc["date"], errors="coerce")
    vacc["mmwr_week"] = pd.to_numeric(vacc["mmwr_week"], errors="coerce").astype("Int64")
    vacc["census2019"] = pd.to_numeric(vacc["census2019"], errors="coerce")
    vacc["administered_dose1_pop_pct"] = pd.to_numeric(vacc["administered_dose1_pop_pct"], errors="coerce")
    vacc["administered_dose1_recip"] = pd.to_numeric(vacc["administered_dose1_recip"], errors="coerce")
    vacc["series_complete_pop_pct"] = pd.to_numeric(vacc["series_complete_pop_pct"], errors="coerce")
    vacc["series_complete_yes"] = pd.to_numeric(vacc["series_complete_yes"], errors="coerce")
    vacc["booster_doses"] = pd.to_numeric(vacc["booster_doses"], errors="coerce")
    vacc["complete_count"] = vacc["series_complete_yes"]
    vacc["complete_cov"] = vacc["series_complete_pop_pct"]
    vacc["partial_count"] = vacc["administered_dose1_recip"]
    vacc["partial_cov"] = vacc["administered_dose1_pop_pct"]
    vacc["booster_count"] = vacc["booster_doses"]
    vacc["booster_cov"] = np.where(
        vacc["booster_doses"].notna() & vacc["census2019"].gt(0),
        100.0 * vacc["booster_doses"] / vacc["census2019"],
        np.nan,
    )
    iso_calendar = vacc["date"].dt.isocalendar()
    vacc["iso_year"] = iso_calendar["year"].astype(int)
    vacc["iso_week"] = iso_calendar["week"].astype(int)
    vacc = vacc.sort_values(["fips", "date"]).reset_index(drop=True)
    weekly = (
        vacc.groupby(["fips", "recip_state", "recip_county", "iso_year", "iso_week"], as_index=False)
        .tail(1)
        .rename(
            columns={
                "date": "source_date",
                "recip_state": "state_abbrev",
                "recip_county": "county",
                "census2019": "population",
            }
        )
    )
    weekly = weekly[
        [
            "fips",
            "state_abbrev",
            "county",
            "source_date",
            "iso_year",
            "iso_week",
            "population",
            "complete_count",
            "complete_cov",
            "partial_count",
            "partial_cov",
            "booster_count",
            "booster_cov",
        ]
    ].copy()
    weekly = add_iso_week_window(weekly)
    weekly = weekly.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    for metric in ["complete_cov", "partial_cov", "booster_cov"]:
        weekly[f"{metric}_delta"] = weekly.groupby("fips", sort=False)[metric].diff()
    weekly["vaccination_source"] = "cdc"
    weekly = weekly.loc[
        (weekly["WeekEndDate"] >= CORE_START_DATE) & (weekly["WeekEndDate"] <= CORE_END_DATE)
    ].copy()
    return weekly, "cdc"


def combine_cdc_with_bansal_fill(
    cdc_weekly: pd.DataFrame,
    bansal_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    key_cols = ["fips", "iso_year", "iso_week", "WeekStartDate", "WeekEndDate"]
    merged = cdc_weekly.merge(
        bansal_weekly,
        on=key_cols,
        how="outer",
        suffixes=("_cdc", "_bansal"),
    )
    combined = pd.DataFrame({column: merged[column] for column in key_cols})
    combined["county"] = merged["county_cdc"].fillna(merged["county_bansal"])
    combined["state_name"] = merged["state_abbrev"].fillna(merged["state_name"])
    combined["source_date"] = merged["source_date_cdc"].fillna(merged["source_date_bansal"])
    combined["population"] = merged["population_cdc"].fillna(merged["population_bansal"])
    for metric in [
        "complete_count",
        "complete_cov",
        "partial_count",
        "partial_cov",
        "booster_count",
        "booster_cov",
    ]:
        cdc_col = f"{metric}_cdc"
        bansal_col = f"{metric}_bansal"
        if cdc_col in merged.columns and bansal_col in merged.columns:
            combined[metric] = merged[cdc_col].fillna(merged[bansal_col])
        elif cdc_col in merged.columns:
            combined[metric] = merged[cdc_col]
        elif bansal_col in merged.columns:
            combined[metric] = merged[bansal_col]
    combined["vaccination_source"] = np.where(
        merged["complete_cov_cdc"].notna() | merged["partial_cov_cdc"].notna() | merged["booster_cov_cdc"].notna(),
        "cdc",
        "bansal_fill",
    )
    combined = combined.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    for metric in ["complete_cov", "partial_cov", "booster_cov"]:
        combined[f"{metric}_delta"] = combined.groupby("fips", sort=False)[metric].diff()
    return combined, "cdc_with_bansal_fill"


def build_population_lookup(vaccination_weekly: pd.DataFrame, feature_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    population = (
        vaccination_weekly[["fips", "population"]]
        .dropna(subset=["population"])
        .sort_values(["fips", "population"], ascending=[True, False])
        .drop_duplicates("fips")
    )
    if feature_frame is not None and "total_population" in feature_frame.columns:
        fallback = feature_frame[["fips", "total_population"]].rename(columns={"total_population": "population"})
        population = fallback.merge(population, on="fips", how="left", suffixes=("_acs", "_vacc"))
        population["population"] = population["population_vacc"].fillna(population["population_acs"])
        population = population[["fips", "population"]]
    return population.drop_duplicates("fips")


def aggregate_nyt_weekly(nyt_daily: pd.DataFrame, population_lookup: pd.DataFrame) -> pd.DataFrame:
    iso_calendar = nyt_daily["date"].dt.isocalendar()
    weekly = nyt_daily.assign(iso_year=iso_calendar["year"].astype(int), iso_week=iso_calendar["week"].astype(int))
    weekly = (
        weekly.groupby(["fips", "iso_year", "iso_week"], as_index=False)
        .agg(
            county=("county", "last"),
            state_name=("state", "last"),
            new_cases=("new_cases", "sum"),
            new_deaths=("new_deaths", "sum"),
            cases=("cases", "max"),
            deaths=("deaths", "max"),
            available_daily_rows=("date", "count"),
        )
    )
    weekly = add_iso_week_window(weekly)
    weekly = weekly.merge(population_lookup, on="fips", how="left")
    weekly["case_rate_100k"] = np.where(
        weekly["population"].gt(0),
        100000.0 * weekly["new_cases"] / weekly["population"],
        np.nan,
    )
    weekly["death_rate_100k"] = np.where(
        weekly["population"].gt(0),
        100000.0 * weekly["new_deaths"] / weekly["population"],
        np.nan,
    )
    weekly = weekly.loc[
        (weekly["WeekEndDate"] >= CORE_START_DATE) & (weekly["WeekEndDate"] <= CORE_END_DATE)
    ].copy()
    return weekly.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)


def fetch_acs_features(state_fips_list: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for state_fips in state_fips_list:
        acs = fetch_json_table(
            ACS_2021_COUNTY_ENDPOINTS["acs5"],
            params={
                "get": "NAME,B01003_001E,B19013_001E",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "B01003_001E", "B19013_001E", "state", "county"],
        ).rename(
            columns={
                "NAME": "county_label",
                "B01003_001E": "total_population",
                "B19013_001E": "median_household_income",
            }
        )
        subject = fetch_json_table(
            ACS_2021_COUNTY_ENDPOINTS["subject"],
            params={
                "get": "NAME,S1701_C03_001E",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "S1701_C03_001E", "state", "county"],
        ).rename(columns={"S1701_C03_001E": "poverty_rate"})
        profile = fetch_json_table(
            ACS_2021_COUNTY_ENDPOINTS["profile"],
            params={
                "get": "NAME,DP05_0024PE,DP02_0068PE",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "DP05_0024PE", "DP02_0068PE", "state", "county"],
        ).rename(
            columns={
                "DP05_0024PE": "senior_population",
                "DP02_0068PE": "college_education",
            }
        )
        merged = acs.merge(subject[["state", "county", "poverty_rate"]], on=["state", "county"], how="left")
        merged = merged.merge(
            profile[["state", "county", "senior_population", "college_education"]],
            on=["state", "county"],
            how="left",
        )
        rows.append(merged)

    merged = pd.concat(rows, ignore_index=True)
    merged["fips"] = merged["state"].astype(str).str.zfill(2) + merged["county"].astype(str).str.zfill(3)
    numeric_columns = [
        "total_population",
        "median_household_income",
        "poverty_rate",
        "senior_population",
        "college_education",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged[
        [
            "fips",
            "total_population",
            "median_household_income",
            "poverty_rate",
            "senior_population",
            "college_education",
        ]
    ].sort_values("fips").reset_index(drop=True)


def load_cdc_svi_2022_features(paths: dict[str, Path]) -> pd.DataFrame:
    us_path = paths["raw_features"] / "cdc_svi_2022_us_county.csv"
    pr_path = paths["raw_features"] / "cdc_svi_2022_pr_county.csv"
    download_if_missing(CDC_SVI_2022_US_COUNTY_URL, us_path)
    download_if_missing(CDC_SVI_2022_PR_COUNTY_URL, pr_path)

    frames: list[pd.DataFrame] = []
    for path in [us_path, pr_path]:
        frame = pd.read_csv(
            path,
            usecols=["FIPS", "STCNTY", "RPL_THEMES"],
            dtype={"FIPS": str, "STCNTY": str},
        )
        frame = frame.assign(
            fips=standardize_fips(frame.get("FIPS", frame.get("STCNTY"))),
            cdc_svi_2022_overall=pd.to_numeric(frame["RPL_THEMES"], errors="coerce"),
        )
        frames.append(frame[["fips", "cdc_svi_2022_overall"]].copy())

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["fips"])
        .drop_duplicates("fips", keep="first")
        .sort_values("fips")
        .reset_index(drop=True)
    )


def load_usda_ers_rucc_2023_features(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_features"] / "usda_ers_rucc_2023.csv"
    download_if_missing(USDA_ERS_RUCC_2023_URL, target)
    raw = pd.read_csv(target, dtype={"FIPS": str}, encoding="latin-1")
    rucc = raw.loc[raw["Attribute"].eq("RUCC_2023"), ["FIPS", "Value"]].copy()
    rucc["fips"] = standardize_fips(rucc["FIPS"])
    rucc["usda_ers_rucc_2023"] = pd.to_numeric(rucc["Value"], errors="coerce")
    return (
        rucc[["fips", "usda_ers_rucc_2023"]]
        .dropna(subset=["fips"])
        .drop_duplicates("fips", keep="first")
        .sort_values("fips")
        .reset_index(drop=True)
    )


def _mode_or_nan(series: pd.Series) -> float:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return float("nan")
    mode = non_null.mode()
    if mode.empty:
        return float(non_null.iloc[0])
    return float(mode.iloc[0])


def build_feature_basis(
    counties: pd.DataFrame,
    vaccination_weekly: pd.DataFrame,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, str]:
    try:
        state_fips_list = sorted(counties["STATEFP"].astype(str).unique().tolist())
        acs_features = fetch_acs_features(state_fips_list)
        svi_features = load_cdc_svi_2022_features(paths)
        rucc_features = load_usda_ers_rucc_2023_features(paths)
        features = counties.merge(acs_features, on="fips", how="left")
        features = features.merge(svi_features, on="fips", how="left")
        features = features.merge(rucc_features, on="fips", how="left")
        for column in [
            "total_population",
            "median_household_income",
            "poverty_rate",
            "senior_population",
            "college_education",
            "cdc_svi_2022_overall",
        ]:
            median_value = pd.to_numeric(features[column], errors="coerce").median()
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(median_value)
        rucc_fill = _mode_or_nan(features["usda_ers_rucc_2023"])
        features["usda_ers_rucc_2023"] = pd.to_numeric(
            features["usda_ers_rucc_2023"],
            errors="coerce",
        ).fillna(rucc_fill)
        basis_mode = "acs_2021"
    except Exception as exc:
        warnings.warn(
            f"County feature retrieval failed; falling back to population-only basis. {exc}",
            stacklevel=2,
        )
        population = build_population_lookup(vaccination_weekly)
        features = counties.merge(population, on="fips", how="left")
        if features["population"].notna().any():
            features["total_population"] = features["population"].fillna(features["population"].median())
            features["median_household_income"] = np.nan
            features["poverty_rate"] = np.nan
            features["senior_population"] = np.nan
            features["college_education"] = np.nan
            features["cdc_svi_2022_overall"] = np.nan
            features["usda_ers_rucc_2023"] = np.nan
            basis_mode = "population_only"
        else:
            features["total_population"] = np.nan
            features["median_household_income"] = np.nan
            features["poverty_rate"] = np.nan
            features["senior_population"] = np.nan
            features["college_education"] = np.nan
            features["cdc_svi_2022_overall"] = np.nan
            features["usda_ers_rucc_2023"] = np.nan
            basis_mode = "zero"
    features["population_density"] = np.where(
        pd.to_numeric(features["land_area_sq_km"], errors="coerce").gt(0),
        pd.to_numeric(features["total_population"], errors="coerce")
        / pd.to_numeric(features["land_area_sq_km"], errors="coerce"),
        np.nan,
    )
    if features["population_density"].notna().any():
        density_median = pd.to_numeric(features["population_density"], errors="coerce").median()
        features["population_density"] = pd.to_numeric(features["population_density"], errors="coerce").fillna(
            density_median
        )
    features["log_population"] = np.log1p(pd.to_numeric(features["total_population"], errors="coerce"))
    return features.sort_values("fips").reset_index(drop=True), basis_mode


def feature_dictionary(feature_basis_mode: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "population_density",
                "description": "County population density using total population divided by TIGER land area in square kilometers.",
                "available_in_basis_mode": feature_basis_mode != "zero",
            },
            {
                "column": "senior_population",
                "description": "ACS 2021 share of the population age 65 and older.",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
            {
                "column": "college_education",
                "description": "ACS 2021 share of adults age 25 and older with a bachelor's degree or higher.",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
            {
                "column": "poverty_rate",
                "description": "ACS 2021 poverty rate percentage.",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
            {
                "column": "log_population",
                "description": "Natural log of one plus total county population.",
                "available_in_basis_mode": feature_basis_mode != "zero",
            },
            {
                "column": "median_household_income",
                "description": "ACS 2021 median household income.",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
            {
                "column": "cdc_svi_2022_overall",
                "description": "CDC/ATSDR SVI 2022 overall county percentile ranking (RPL_THEMES).",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
            {
                "column": "usda_ers_rucc_2023",
                "description": "USDA ERS 2023 Rural-Urban Continuum Code as an ordinal county urbanicity feature.",
                "available_in_basis_mode": feature_basis_mode == "acs_2021",
            },
        ]
    )


def load_or_download_geometry(paths: dict[str, Path]) -> tuple[gpd.GeoDataFrame, str]:
    primary_zip = paths["raw_geography"] / "tl_2021_us_county.zip"
    fallback_zip = paths["raw_geography"] / "tl_2022_us_county.zip"
    geometry_source = "TIGER2021"
    try:
        download_if_missing(TIGER_2021_COUNTY_URL, primary_zip)
        geometry_path = primary_zip
    except Exception:
        download_if_missing(TIGER_2022_COUNTY_URL, fallback_zip)
        geometry_path = fallback_zip
        geometry_source = "TIGER2022"

    counties = gpd.read_file(f"zip://{geometry_path.resolve()}")
    counties["fips"] = counties["GEOID"].astype(str).str.zfill(5)
    counties["county"] = counties["NAME"].astype(str)
    counties["state_name"] = counties["STATE_NAME"].astype(str) if "STATE_NAME" in counties.columns else counties["STATEFP"].astype(str)
    counties["land_area_sq_km"] = pd.to_numeric(counties.get("ALAND"), errors="coerce") / 1_000_000.0
    counties = counties.loc[counties["fips"].str.len() == 5].copy()
    counties = counties[["fips", "county", "state_name", "STATEFP", "COUNTYFP", "land_area_sq_km", "geometry"]].copy()
    counties = counties.sort_values("fips").reset_index(drop=True)
    return counties, geometry_source


def build_geometry_network_artifacts(counties_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gpkg_path = PROCESSED_DIR / "us_counties.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    counties_gdf.to_file(gpkg_path, layer="counties", driver="GPKG")

    projected = counties_gdf.to_crs(5070).copy()
    centroids = projected.geometry.centroid
    centroids_ll = gpd.GeoSeries(centroids, crs=projected.crs).to_crs(4326)
    centroid_table = pd.DataFrame(
        {
            "fips": projected["fips"],
            "county": projected["county"],
            "state_name": projected["state_name"],
            "centroid_x": centroids.x,
            "centroid_y": centroids.y,
            "centroid_lon": centroids_ll.x,
            "centroid_lat": centroids_ll.y,
        }
    ).sort_values("fips")
    centroid_table.to_csv(PROCESSED_DIR / "us_county_centroids.csv", index=False)

    contiguity = build_touching_edge_list(
        projected[["fips", "geometry"]].copy(),
        id_column="fips",
        neighbor_column="neighbor_fips",
    )
    contiguity.to_csv(PROCESSED_DIR / "us_county_contiguity_adjacency.csv.gz", index=False)

    knn_8, kernel_8 = build_knn_and_kernel_edges(
        centroid_table,
        id_column="fips",
        x_column="centroid_x",
        y_column="centroid_y",
        k=8,
    )
    knn_8 = knn_8.rename(columns={"neighbor_id": "neighbor_fips"})
    kernel_8 = kernel_8.rename(columns={"neighbor_id": "neighbor_fips"})
    knn_8.to_csv(PROCESSED_DIR / "us_county_knn_8_adjacency.csv.gz", index=False)
    kernel_8.to_csv(PROCESSED_DIR / "us_county_distance_kernel_8_adjacency.csv.gz", index=False)

    node_ids = centroid_table["fips"].tolist()
    summary_rows: list[dict[str, object]] = []
    for network_name, edge_frame in [
        ("contiguity", contiguity),
        ("knn_8", knn_8),
        ("distance_kernel_8", kernel_8),
    ]:
        summary_rows.append(
            {
                "network_name": network_name,
                "node_count": len(node_ids),
                "edge_count": int(len(edge_frame)),
                "connected_components": int(
                    count_connected_components(node_ids, edge_frame, "fips", "neighbor_fips")
                ),
            }
        )
    network_summary = pd.DataFrame(summary_rows)
    network_summary.to_csv(PROCESSED_DIR / "us_county_network_summary.csv", index=False)
    return network_summary


def build_week_calendar() -> pd.DataFrame:
    week_end = pd.date_range(CORE_START_DATE, CORE_END_DATE, freq="W-SUN")
    calendar = pd.DataFrame({"WeekEndDate": week_end})
    iso = calendar["WeekEndDate"].dt.isocalendar()
    calendar["WeekStartDate"] = calendar["WeekEndDate"] - pd.Timedelta(days=6)
    calendar["iso_year"] = iso["year"].astype(int)
    calendar["iso_week"] = iso["week"].astype(int)
    return calendar


def build_county_master(
    geometry: gpd.GeoDataFrame,
    nyt_daily: pd.DataFrame,
    vaccination_weekly: pd.DataFrame,
) -> pd.DataFrame:
    overlap = set(geometry["fips"]) & set(nyt_daily["fips"]) & set(vaccination_weekly["fips"])
    counties = pd.DataFrame(geometry.loc[geometry["fips"].isin(overlap)].drop(columns="geometry"))
    counties = counties.sort_values("fips").reset_index(drop=True)
    return counties


def build_joined_panel(
    counties: pd.DataFrame,
    weekly_nyt: pd.DataFrame,
    weekly_vaccination: pd.DataFrame,
) -> pd.DataFrame:
    calendar = build_week_calendar()
    county_calendar = counties.assign(_key=1).merge(calendar.assign(_key=1), on="_key").drop(columns="_key")
    panel = county_calendar.merge(
        weekly_nyt.drop(columns=["county", "state_name"], errors="ignore"),
        on=["fips", "iso_year", "iso_week", "WeekStartDate", "WeekEndDate"],
        how="left",
    ).merge(
        weekly_vaccination.drop(columns=["county", "state_name", "state_abbrev"], errors="ignore"),
        on=["fips", "iso_year", "iso_week", "WeekStartDate", "WeekEndDate"],
        how="left",
        suffixes=("_nyt", ""),
    )
    panel["population"] = panel["population"].fillna(panel["population_nyt"])
    panel = panel.drop(columns=["population_nyt"], errors="ignore")
    panel = panel.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    for column in [
        "new_cases",
        "new_deaths",
        "cases",
        "deaths",
        "available_daily_rows",
        "case_rate_100k",
        "death_rate_100k",
    ]:
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    panel["prev_case_rate_100k"] = panel.groupby("fips", sort=False)["case_rate_100k"].shift(1)
    panel["case_growth_ratio"] = np.where(
        panel["prev_case_rate_100k"].gt(0),
        panel["case_rate_100k"] / panel["prev_case_rate_100k"],
        np.nan,
    )
    panel["vaccination_observed"] = panel["complete_cov"].notna()
    panel["booster_observed"] = panel["booster_cov"].notna()
    return panel


def build_processed_outputs(args: argparse.Namespace) -> None:
    paths = ensure_directories()
    geometry_gdf, geometry_source = load_or_download_geometry(paths)

    bansal_weekly: pd.DataFrame | None = None
    cdc_weekly: pd.DataFrame | None = None
    if args.vaccination_source == "cdc":
        cdc_weekly, _ = load_or_download_cdc(paths)
        bansal_weekly, _ = load_or_download_bansal(paths)
        vaccination_weekly, resolved_vaccination_source = combine_cdc_with_bansal_fill(
            cdc_weekly,
            bansal_weekly,
        )
    else:
        vaccination_weekly, resolved_vaccination_source = load_or_download_bansal(paths)
    nyt_daily = load_or_download_nyt(paths)
    counties = build_county_master(geometry_gdf, nyt_daily, vaccination_weekly)
    geometry_gdf = geometry_gdf.loc[geometry_gdf["fips"].isin(counties["fips"])].copy()
    vaccination_weekly = vaccination_weekly.loc[vaccination_weekly["fips"].isin(counties["fips"])].copy()
    nyt_daily = nyt_daily.loc[nyt_daily["fips"].isin(counties["fips"])].copy()

    if args.reuse_processed_features and (PROCESSED_DIR / "us_county_feature_basis.csv.gz").exists():
        features = pd.read_csv(PROCESSED_DIR / "us_county_feature_basis.csv.gz", dtype={"fips": str})
        feature_basis_mode = str(features.get("feature_basis_mode", pd.Series(["unknown"])).iloc[0])
    else:
        features, feature_basis_mode = build_feature_basis(counties, vaccination_weekly, paths)
        features["feature_basis_mode"] = feature_basis_mode
        features.to_csv(PROCESSED_DIR / "us_county_feature_basis.csv.gz", index=False)
        feature_dictionary(feature_basis_mode).to_csv(
            PROCESSED_DIR / "us_county_feature_dictionary.csv",
            index=False,
        )

    population_lookup = build_population_lookup(vaccination_weekly, features)
    weekly_nyt = aggregate_nyt_weekly(nyt_daily, population_lookup)
    weekly_nyt = weekly_nyt.loc[weekly_nyt["fips"].isin(counties["fips"])].copy()
    panel = build_joined_panel(counties, weekly_nyt, vaccination_weekly)

    if not args.reuse_processed_tables or not (PROCESSED_DIR / "us_county_weekly_panel.csv.gz").exists():
        nyt_daily.to_csv(PROCESSED_DIR / "us_county_daily_nyt.csv.gz", index=False)
        weekly_nyt.to_csv(PROCESSED_DIR / "us_county_weekly_nyt.csv.gz", index=False)
        vaccination_weekly.to_csv(PROCESSED_DIR / "us_county_weekly_vaccination.csv.gz", index=False)
        panel.to_csv(PROCESSED_DIR / "us_county_weekly_panel.csv.gz", index=False)

    if args.reuse_processed_networks and (PROCESSED_DIR / "us_county_network_summary.csv").exists():
        network_summary = pd.read_csv(PROCESSED_DIR / "us_county_network_summary.csv")
    else:
        network_summary = build_geometry_network_artifacts(geometry_gdf)

    dump_json(
        PROCESSED_DIR / "processing_summary.json",
        {
            "unit_label": UNIT_LABEL,
            "vaccination_source_requested": args.vaccination_source,
            "vaccination_source_resolved": resolved_vaccination_source,
            "geometry_source": geometry_source,
            "feature_basis_mode": feature_basis_mode,
            "county_count_geometry": int(geometry_gdf["fips"].nunique()),
            "county_count_weekly_panel": int(panel["fips"].nunique()),
            "state_count": int(counties["STATEFP"].nunique()),
            "core_start_date": CORE_START_DATE,
            "core_end_date": CORE_END_DATE,
            "daily_nyt_rows": int(len(nyt_daily)),
            "weekly_nyt_rows": int(len(weekly_nyt)),
            "vaccination_weekly_rows": int(len(vaccination_weekly)),
            "cdc_weekly_rows": int(len(cdc_weekly)) if cdc_weekly is not None else None,
            "bansal_weekly_rows": int(len(bansal_weekly)) if bansal_weekly is not None else None,
            "panel_rows": int(len(panel)),
            "vaccination_observed_rows": int(panel["vaccination_observed"].sum()),
            "booster_observed_rows": int(panel["booster_observed"].sum()),
            "networks": network_summary.to_dict(orient="records"),
        },
    )


def main() -> None:
    args = parse_args()
    build_processed_outputs(args)


if __name__ == "__main__":
    main()
