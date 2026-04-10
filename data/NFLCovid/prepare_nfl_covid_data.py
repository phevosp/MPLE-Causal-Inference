"""Prepare the nationwide NFL + COVID county dataset package."""

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
    ABBR_TO_TEAM,
    ACS_2019_COUNTY_ENDPOINTS,
    ALLOWED_GAME_TYPES,
    ATTENDANCE_SHEET_CSV_URL,
    CORE_END_DATE,
    CORE_START_DATE,
    ESPN_SUMMARY_API_TEMPLATE,
    OVERRIDE_PATH,
    PROCESSED_DIR,
    RUCC_2013_URL,
    SOURCE_LABEL,
    STATE_ABBREV_TO_FIPS,
    STATE_FIPS_TO_NAME,
    SVI_2020_US_COUNTY_URL,
    TEAM_TO_ABBREV,
    TIGER_2021_COUNTY_URL,
    TIGER_2022_COUNTY_URL,
    NFLVERSE_GAMES_URL,
    NYT_COUNTIES_URL,
    add_sunday_week_window,
    build_sunday_week_calendar,
    dump_json,
    ensure_directories,
    normalize_county_name,
    standardize_fips,
    write_readme,
)
from data_utils import (  # noqa: E402
    build_knn_and_kernel_edges,
    build_touching_edge_list,
    count_connected_components,
    download_if_missing,
)


ANALYSIS_END_DATE = CORE_END_DATE + pd.Timedelta(days=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the nationwide NFL + COVID county package."
    )
    parser.add_argument(
        "--reuse_cached_game_attendance",
        action="store_true",
        help="Reuse any cached ESPN attendance table and fetch only missing event ids.",
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


def download_attendance_sheet(paths: dict[str, Path]) -> Path:
    target = paths["raw_attendance"] / "nfl_2020_attendance_geography.csv"
    download_if_missing(ATTENDANCE_SHEET_CSV_URL, target)
    return target


def parse_attendance_geography(sheet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(sheet_path)
    raw = raw.rename(
        columns={
            raw.columns[0]: "team_cell",
            raw.columns[1]: "open_date_cell",
            raw.columns[2]: "stadium_county",
            raw.columns[3]: "stadium_state",
            raw.columns[4]: "stadium_pct",
            raw.columns[5]: "neutral_county",
            raw.columns[6]: "neutral_state",
            raw.columns[7]: "neutral_pct",
            raw.columns[8]: "notes_cell",
        }
    )

    current_team: str | None = None
    team_meta: dict[str, dict[str, list[str]]] = {}
    rows: list[dict[str, object]] = []

    def valid_county_name(value: object) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if "COUNTY DATA NOT AVAILABLE" in upper:
            return False
        if "FIRST HOME GAME" in upper:
            return False
        return True

    def append_row(
        team_name: str,
        role: str,
        county_name_raw: object,
        state_abbrev: object,
        pct: object,
    ) -> None:
        if not valid_county_name(county_name_raw):
            return
        pct_value = pd.to_numeric(pd.Series([pct]), errors="coerce").iloc[0]
        if pd.isna(pct_value):
            return
        rows.append(
            {
                "team_name": team_name,
                "team_abbrev": TEAM_TO_ABBREV[team_name],
                "fan_share_role": role,
                "county_name_raw": str(county_name_raw).strip(),
                "state_abbrev": str(state_abbrev).strip(),
                "fan_share_pct": float(pct_value),
                "county_name_normalized": normalize_county_name(county_name_raw),
            }
        )

    for _, row in raw.iterrows():
        if pd.notna(row["team_cell"]):
            candidate = str(row["team_cell"]).strip()
            if candidate and candidate != "Team":
                current_team = candidate
                team_meta.setdefault(current_team, {"open_date_tokens": [], "notes_tokens": []})
        if current_team is None:
            continue
        if pd.notna(row["open_date_cell"]):
            token = str(row["open_date_cell"]).strip()
            if token:
                team_meta[current_team]["open_date_tokens"].append(token)
        if pd.notna(row["notes_cell"]):
            token = str(row["notes_cell"]).strip()
            if token and token.lower() != "nan":
                team_meta[current_team]["notes_tokens"].append(token)

        append_row(
            current_team,
            "stadium",
            row["stadium_county"],
            row["stadium_state"],
            row["stadium_pct"],
        )
        append_row(
            current_team,
            "neutral",
            row["neutral_county"],
            row["neutral_state"],
            row["neutral_pct"],
        )

    geography = pd.DataFrame(rows).sort_values(
        ["team_name", "fan_share_role", "state_abbrev", "county_name_raw"]
    ).reset_index(drop=True)
    meta_rows = []
    for team_name, payload in team_meta.items():
        meta_rows.append(
            {
                "team_name": team_name,
                "team_abbrev": TEAM_TO_ABBREV.get(team_name, ""),
                "open_date_raw": " | ".join(pd.unique(pd.Series(payload["open_date_tokens"], dtype="string").dropna())),
                "notes_raw": " | ".join(pd.unique(pd.Series(payload["notes_tokens"], dtype="string").dropna())),
            }
        )
    team_metadata = pd.DataFrame(meta_rows).sort_values("team_name").reset_index(drop=True)
    geography = geography.merge(team_metadata, on=["team_name", "team_abbrev"], how="left")
    return geography, team_metadata


def load_county_overrides() -> pd.DataFrame:
    overrides = pd.read_csv(OVERRIDE_PATH, dtype={"fips": str})
    overrides["county_name_raw"] = overrides["county_name_raw"].astype(str).str.strip()
    overrides["state_abbrev"] = overrides["state_abbrev"].astype(str).str.strip()
    overrides["supported_in_county_panel"] = overrides["supported_in_county_panel"].astype(bool)
    overrides = overrides.rename(
        columns={
            "fips": "fips_override",
            "override_reason": "override_reason_override",
            "supported_in_county_panel": "supported_in_county_panel_override",
        }
    )
    return overrides


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
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties["COUNTYFP"] = counties["COUNTYFP"].astype(str).str.zfill(3)
    counties["state_name"] = counties["STATEFP"].map(STATE_FIPS_TO_NAME)
    counties["land_area_sq_km"] = pd.to_numeric(counties.get("ALAND"), errors="coerce") / 1_000_000.0
    counties = counties.loc[counties["fips"].str.len() == 5].copy()
    counties["county_name_normalized"] = counties["county"].map(normalize_county_name)
    counties = counties[
        [
            "fips",
            "county",
            "county_name_normalized",
            "state_name",
            "STATEFP",
            "COUNTYFP",
            "land_area_sq_km",
            "geometry",
        ]
    ].sort_values("fips")
    return counties.reset_index(drop=True), geometry_source


def load_or_download_nyt(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_nyt"] / "us-counties.csv"
    download_if_missing(NYT_COUNTIES_URL, target)
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


def fetch_acs_features(state_fips_list: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for state_fips in state_fips_list:
        acs = fetch_json_table(
            ACS_2019_COUNTY_ENDPOINTS["acs5"],
            params={
                "get": "NAME,B01001_001E",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "B01001_001E", "state", "county"],
        ).rename(columns={"B01001_001E": "total_population"})
        subject = fetch_json_table(
            ACS_2019_COUNTY_ENDPOINTS["subject"],
            params={
                "get": "NAME,S0101_C01_030E",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "S0101_C01_030E", "state", "county"],
        ).rename(columns={"S0101_C01_030E": "senior_population"})
        profile = fetch_json_table(
            ACS_2019_COUNTY_ENDPOINTS["profile"],
            params={
                "get": "NAME,DP02_0068PE",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
            empty_columns=["NAME", "DP02_0068PE", "state", "county"],
        ).rename(columns={"DP02_0068PE": "college_education"})
        merged = acs.merge(
            subject[["state", "county", "senior_population"]],
            on=["state", "county"],
            how="left",
        ).merge(
            profile[["state", "county", "college_education"]],
            on=["state", "county"],
            how="left",
        )
        rows.append(merged)

    merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if merged.empty:
        return pd.DataFrame(columns=["fips", "total_population", "senior_population", "college_education"])
    merged["fips"] = merged["state"].astype(str).str.zfill(2) + merged["county"].astype(str).str.zfill(3)
    for column in ["total_population", "senior_population", "college_education"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged[["fips", "total_population", "senior_population", "college_education"]].sort_values("fips")


def load_or_download_svi(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_features"] / "SVI_2020_US_county.csv"
    download_if_missing(SVI_2020_US_COUNTY_URL, target)
    svi = pd.read_csv(target, dtype={"FIPS": str})
    svi["fips"] = standardize_fips(svi["FIPS"])
    svi["svi_overall"] = pd.to_numeric(svi["RPL_THEMES"], errors="coerce")
    svi["svi_total_population"] = pd.to_numeric(svi["E_TOTPOP"], errors="coerce")
    return svi[["fips", "svi_overall", "svi_total_population"]].sort_values("fips")


def load_or_download_rucc(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_features"] / "2013-rural-urban-continuum-codes.xls"
    download_if_missing(RUCC_2013_URL, target)
    rucc = pd.read_excel(target, sheet_name="Rural-urban Continuum Code 2013", dtype={"FIPS": str})
    rucc["fips"] = standardize_fips(rucc["FIPS"])
    rucc["rucc_2013"] = pd.to_numeric(rucc["RUCC_2013"], errors="coerce")
    return rucc[["fips", "rucc_2013"]].sort_values("fips")


def build_full_feature_table(
    counties: gpd.GeoDataFrame,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, str]:
    state_fips_list = sorted(counties["STATEFP"].dropna().astype(str).unique().tolist())
    acs = fetch_acs_features(state_fips_list)
    svi = load_or_download_svi(paths)
    rucc = load_or_download_rucc(paths)

    features = counties.drop(columns="geometry").merge(acs, on="fips", how="left")
    features = features.merge(svi, on="fips", how="left").merge(rucc, on="fips", how="left")
    features["total_population"] = pd.to_numeric(features["total_population"], errors="coerce")
    features["svi_total_population"] = pd.to_numeric(features["svi_total_population"], errors="coerce")
    features["total_population"] = features["total_population"].fillna(features["svi_total_population"])
    for column in ["senior_population", "college_education", "svi_overall", "rucc_2013"]:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if features["total_population"].notna().any():
        population_median = float(features["total_population"].median())
        features["total_population"] = features["total_population"].fillna(population_median)
    else:
        features["total_population"] = 0.0

    features["population_density"] = np.where(
        pd.to_numeric(features["land_area_sq_km"], errors="coerce").gt(0),
        pd.to_numeric(features["total_population"], errors="coerce")
        / pd.to_numeric(features["land_area_sq_km"], errors="coerce"),
        np.nan,
    )
    density_values = pd.to_numeric(features["population_density"], errors="coerce")
    density_median = float(density_values.median()) if density_values.notna().any() else 0.0
    features["population_density"] = density_values.fillna(density_median)
    features["log_population"] = np.log1p(pd.to_numeric(features["total_population"], errors="coerce"))
    features["feature_basis_mode"] = "static_county_covariates"
    return features.sort_values("fips").reset_index(drop=True), "static_county_covariates"


def build_county_master(counties: gpd.GeoDataFrame, nyt_daily: pd.DataFrame) -> pd.DataFrame:
    nyt_window = nyt_daily.loc[nyt_daily["date"].between(CORE_START_DATE, ANALYSIS_END_DATE)].copy()
    total_cases = (
        nyt_window.groupby("fips", as_index=False)["new_cases"]
        .sum()
        .rename(columns={"new_cases": "total_cases_window"})
    )
    nyt_overlap = set(nyt_daily["fips"].dropna().unique().tolist())
    master = counties.loc[counties["fips"].isin(nyt_overlap)].drop(columns="geometry").copy()
    master = master.merge(total_cases, on="fips", how="left")
    master["total_cases_window"] = pd.to_numeric(master["total_cases_window"], errors="coerce").fillna(0.0)
    master["meets_case_threshold"] = master["total_cases_window"].ge(200.0)
    retained = master.loc[master["meets_case_threshold"]].copy()
    return retained.sort_values("fips").reset_index(drop=True)


def augment_attendance_geography(
    geography: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    county_master: pd.DataFrame,
) -> pd.DataFrame:
    overrides = load_county_overrides()
    augmented = geography.copy()
    augmented["STATEFP"] = augmented["state_abbrev"].map(STATE_ABBREV_TO_FIPS)

    lookup = counties[["fips", "STATEFP", "county_name_normalized"]].drop_duplicates()
    candidate_matches = augmented.merge(
        lookup,
        on=["STATEFP", "county_name_normalized"],
        how="left",
    )
    auto_candidates = (
        candidate_matches.groupby(list(augmented.columns), dropna=False)["fips"]
        .agg(lambda values: sorted(set(value for value in values if pd.notna(value))))
        .reset_index()
    )
    auto_candidates["auto_match_count"] = auto_candidates["fips"].apply(len)
    auto_candidates["auto_fips"] = auto_candidates["fips"].apply(lambda values: values[0] if len(values) == 1 else pd.NA)
    auto_candidates = auto_candidates.drop(columns=["fips"])

    augmented = augmented.merge(
        auto_candidates,
        on=list(augmented.columns),
        how="left",
    )
    augmented = augmented.merge(
        overrides,
        on=["county_name_raw", "state_abbrev"],
        how="left",
    )

    override_present = augmented["fips_override"].notna()
    exact_present = augmented["auto_match_count"].eq(1)
    ambiguous_present = augmented["auto_match_count"].gt(1)
    augmented["fips"] = np.where(
        override_present,
        augmented["fips_override"],
        np.where(exact_present, augmented["auto_fips"], pd.NA),
    )
    augmented["match_status"] = np.select(
        [override_present, exact_present, ambiguous_present],
        ["override_match", "exact_match", "ambiguous_match"],
        default="unresolved",
    )
    augmented["override_reason"] = augmented["override_reason_override"].fillna("")
    augmented["base_supported_in_county_panel"] = np.where(
        override_present,
        augmented["supported_in_county_panel_override"].fillna(True),
        True,
    )
    modeled_fips = set(county_master["fips"].tolist())
    augmented["supported_in_county_panel"] = (
        augmented["fips"].notna()
        & augmented["base_supported_in_county_panel"].astype(bool)
        & augmented["fips"].isin(modeled_fips)
    )
    return augmented[
        [
            "team_name",
            "team_abbrev",
            "fan_share_role",
            "county_name_raw",
            "state_abbrev",
            "STATEFP",
            "fan_share_pct",
            "county_name_normalized",
            "open_date_raw",
            "notes_raw",
            "fips",
            "match_status",
            "override_reason",
            "supported_in_county_panel",
        ]
    ].sort_values(["team_name", "fan_share_role", "state_abbrev", "county_name_raw"]).reset_index(drop=True)


def load_or_download_nfl_games(paths: dict[str, Path]) -> pd.DataFrame:
    target = paths["raw_nfl"] / "nflverse_games.csv"
    download_if_missing(NFLVERSE_GAMES_URL, target)
    games = pd.read_csv(target)
    games = games.loc[
        games["season"].eq(2020) & games["game_type"].isin(ALLOWED_GAME_TYPES)
    ].copy()
    games["gameday"] = pd.to_datetime(games["gameday"])
    games["espn"] = pd.to_numeric(games["espn"], errors="coerce").astype("Int64")
    games = games.sort_values(["gameday", "game_type", "week", "home_team", "away_team"]).reset_index(drop=True)
    return games[
        [
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "espn",
            "stadium",
        ]
    ].copy()


def fetch_game_attendance_table(
    paths: dict[str, Path],
    games: pd.DataFrame,
    reuse_cached: bool,
) -> pd.DataFrame:
    cache_path = paths["raw_nfl"] / "espn_game_attendance.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame(columns=["espn", "attendance"])

    cached["espn"] = pd.to_numeric(cached.get("espn"), errors="coerce").astype("Int64")
    cached["attendance"] = pd.to_numeric(cached.get("attendance"), errors="coerce")
    cached = cached.dropna(subset=["espn"]).drop_duplicates("espn")

    if not reuse_cached and cache_path.exists():
        needed = set(games["espn"].dropna().astype(int).tolist())
        cached = cached.loc[cached["espn"].astype(int).isin(needed)].copy()

    needed = set(games["espn"].dropna().astype(int).tolist())
    existing = set(cached["espn"].dropna().astype(int).tolist())
    missing = sorted(needed - existing)

    if missing:
        session = requests.Session()
        rows = []
        for event_id in missing:
            response = session.get(ESPN_SUMMARY_API_TEMPLATE.format(event_id=event_id), timeout=60)
            response.raise_for_status()
            payload = response.json()
            attendance = payload.get("gameInfo", {}).get("attendance")
            rows.append({"espn": int(event_id), "attendance": attendance})
        fetched = pd.DataFrame(rows)
        cached = pd.concat([cached, fetched], ignore_index=True, sort=False).drop_duplicates("espn", keep="last")
        cached = cached.sort_values("espn").reset_index(drop=True)
        cached.to_csv(cache_path, index=False)
    elif not cache_path.exists():
        cached.to_csv(cache_path, index=False)

    return cached


def build_game_attendance_table(
    games: pd.DataFrame,
    attendance_cache: pd.DataFrame,
    geography: pd.DataFrame,
) -> pd.DataFrame:
    listed_share = (
        geography.groupby(["team_name", "team_abbrev"], as_index=False)["fan_share_pct"]
        .sum()
        .rename(columns={"fan_share_pct": "team_listed_share_pct"})
    )
    game_table = games.merge(attendance_cache, on="espn", how="left")
    game_table["attendance"] = pd.to_numeric(game_table["attendance"], errors="coerce")
    game_table["home_team_name"] = game_table["home_team"].map(ABBR_TO_TEAM)
    game_table = game_table.merge(
        listed_share[["team_abbrev", "team_listed_share_pct"]],
        left_on="home_team",
        right_on="team_abbrev",
        how="left",
    ).drop(columns=["team_abbrev"], errors="ignore")
    game_table["team_listed_share_pct"] = pd.to_numeric(
        game_table["team_listed_share_pct"], errors="coerce"
    ).fillna(0.0)
    game_table["excluded_arizona_home_game"] = game_table["home_team"].eq("ARI")
    game_table["zeroed_unsupported_home_game"] = game_table["home_team"].isin(["NO", "WAS"]) & game_table["attendance"].gt(0)
    game_table["has_sheet_geography"] = game_table["team_listed_share_pct"].gt(0)
    game_table["included_in_county_expansion"] = (
        ~game_table["excluded_arizona_home_game"] & game_table["has_sheet_geography"]
    )
    game_table["unassigned_attendance_count"] = np.where(
        game_table["included_in_county_expansion"],
        game_table["attendance"].fillna(0.0)
        * np.clip(1.0 - game_table["team_listed_share_pct"] / 100.0, 0.0, None),
        0.0,
    )
    return game_table.sort_values(["gameday", "home_team", "away_team"]).reset_index(drop=True)


def build_county_game_exposure(
    game_table: pd.DataFrame,
    geography_augmented: pd.DataFrame,
    full_features: pd.DataFrame,
) -> pd.DataFrame:
    resolved_geography = geography_augmented.loc[geography_augmented["fips"].notna()].copy()
    exposure = game_table.loc[game_table["included_in_county_expansion"]].merge(
        resolved_geography,
        left_on="home_team",
        right_on="team_abbrev",
        how="inner",
    )
    population_lookup = full_features[["fips", "total_population"]].rename(
        columns={"total_population": "county_population"}
    )
    exposure = exposure.merge(population_lookup, on="fips", how="left")
    exposure["county_attendance_count"] = exposure["attendance"].fillna(0.0) * exposure["fan_share_pct"] / 100.0
    exposure["county_attendance_share_pct"] = np.where(
        pd.to_numeric(exposure["county_population"], errors="coerce").gt(0),
        100.0 * exposure["county_attendance_count"] / pd.to_numeric(exposure["county_population"], errors="coerce"),
        np.nan,
    )
    exposure = add_sunday_week_window(exposure, date_column="gameday")
    return exposure[
        [
            "game_id",
            "game_type",
            "week",
            "gameday",
            "WeekStartDate",
            "WeekEndDate",
            "week_year",
            "week_index",
            "home_team",
            "home_team_name",
            "away_team",
            "espn",
            "attendance",
            "team_listed_share_pct",
            "county_name_raw",
            "state_abbrev",
            "fan_share_role",
            "fan_share_pct",
            "fips",
            "county_population",
            "county_attendance_count",
            "county_attendance_share_pct",
            "match_status",
            "override_reason",
            "supported_in_county_panel",
        ]
    ].sort_values(["gameday", "home_team", "fips"]).reset_index(drop=True)


def aggregate_weekly_exposure(
    county_game_exposure: pd.DataFrame,
    modeled_features: pd.DataFrame,
) -> pd.DataFrame:
    supported = county_game_exposure.loc[county_game_exposure["supported_in_county_panel"]].copy()
    if supported.empty:
        return pd.DataFrame(
            columns=[
                "fips",
                "WeekStartDate",
                "WeekEndDate",
                "week_year",
                "week_index",
                "attendance_count",
                "attendance_share_pct",
                "games_with_exposure",
            ]
        )
    weekly = (
        supported.groupby(["fips", "WeekStartDate", "WeekEndDate", "week_year", "week_index"], as_index=False)
        .agg(
            attendance_count=("county_attendance_count", "sum"),
            games_with_exposure=("game_id", "nunique"),
        )
        .sort_values(["fips", "WeekStartDate"])
        .reset_index(drop=True)
    )
    population_lookup = modeled_features[["fips", "total_population"]]
    weekly = weekly.merge(population_lookup, on="fips", how="left")
    weekly["attendance_share_pct"] = np.where(
        pd.to_numeric(weekly["total_population"], errors="coerce").gt(0),
        100.0 * weekly["attendance_count"] / pd.to_numeric(weekly["total_population"], errors="coerce"),
        np.nan,
    )
    return weekly.drop(columns=["total_population"])


def aggregate_nyt_weekly(nyt_daily: pd.DataFrame, population_lookup: pd.DataFrame) -> pd.DataFrame:
    working = add_sunday_week_window(nyt_daily, date_column="date")
    weekly = (
        working.groupby(["fips", "WeekStartDate", "WeekEndDate", "week_year", "week_index"], as_index=False)
        .agg(
            county=("county", "last"),
            state_name=("state", "last"),
            new_cases=("new_cases", "sum"),
            new_deaths=("new_deaths", "sum"),
            cases=("cases", "max"),
            deaths=("deaths", "max"),
            available_daily_rows=("date", "count"),
        )
        .sort_values(["fips", "WeekStartDate"])
        .reset_index(drop=True)
    )
    weekly = weekly.merge(population_lookup, on="fips", how="left")
    weekly["case_rate_100k"] = np.where(
        pd.to_numeric(weekly["total_population"], errors="coerce").gt(0),
        100000.0 * weekly["new_cases"] / pd.to_numeric(weekly["total_population"], errors="coerce"),
        np.nan,
    )
    weekly["death_rate_100k"] = np.where(
        pd.to_numeric(weekly["total_population"], errors="coerce").gt(0),
        100000.0 * weekly["new_deaths"] / pd.to_numeric(weekly["total_population"], errors="coerce"),
        np.nan,
    )
    return weekly


def build_feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "population_density",
                "description": "County population density using ACS 2019 total population divided by TIGER land area in square kilometers.",
            },
            {
                "column": "log_population",
                "description": "Natural log of one plus county total population.",
            },
            {
                "column": "svi_overall",
                "description": "CDC/ATSDR 2020 Social Vulnerability Index overall percentile ranking (RPL_THEMES).",
            },
            {
                "column": "rucc_2013",
                "description": "USDA ERS 2013 Rural-Urban Continuum Code.",
            },
            {
                "column": "senior_population",
                "description": "ACS 2019 share of the population age 65 and older.",
            },
            {
                "column": "college_education",
                "description": "ACS 2019 share of adults age 25 and older with a bachelor's degree or higher.",
            },
        ]
    )


def compute_missingness_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows = []
    for column in columns:
        values = frame[column]
        rows.append(
            {
                "column": column,
                "missing_count": int(values.isna().sum()),
                "missing_share": float(values.isna().mean()),
            }
        )
    return rows


def build_weekly_panel(
    county_master: pd.DataFrame,
    weekly_nyt: pd.DataFrame,
    weekly_exposure: pd.DataFrame,
    modeled_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    calendar = build_sunday_week_calendar()
    county_calendar = county_master.assign(_key=1).merge(calendar.assign(_key=1), on="_key").drop(columns="_key")
    panel = county_calendar.merge(
        weekly_nyt.drop(columns=["county", "state_name", "total_population"], errors="ignore"),
        on=["fips", "WeekStartDate", "WeekEndDate", "week_year", "week_index"],
        how="left",
    )
    panel = panel.merge(
        weekly_exposure,
        on=["fips", "WeekStartDate", "WeekEndDate", "week_year", "week_index"],
        how="left",
    )
    panel = panel.merge(
        modeled_features[
            [
                "fips",
                "total_population",
                "population_density",
                "log_population",
                "svi_overall",
                "rucc_2013",
                "senior_population",
                "college_education",
                "feature_basis_mode",
            ]
        ],
        on="fips",
        how="left",
    )

    before_rows = compute_missingness_rows(
        panel,
        [
            "new_cases",
            "new_deaths",
            "case_rate_100k",
            "death_rate_100k",
            "attendance_count",
            "attendance_share_pct",
        ],
    )
    panel = panel.sort_values(["fips", "WeekStartDate"]).reset_index(drop=True)
    panel["new_cases"] = pd.to_numeric(panel["new_cases"], errors="coerce").fillna(0.0)
    panel["new_deaths"] = pd.to_numeric(panel["new_deaths"], errors="coerce").fillna(0.0)
    panel["cases"] = panel.groupby("fips", sort=False)["new_cases"].cumsum()
    panel["deaths"] = panel.groupby("fips", sort=False)["new_deaths"].cumsum()
    panel["available_daily_rows"] = pd.to_numeric(panel["available_daily_rows"], errors="coerce").fillna(0.0)
    panel["attendance_count"] = pd.to_numeric(panel["attendance_count"], errors="coerce").fillna(0.0)
    panel["games_with_exposure"] = pd.to_numeric(panel["games_with_exposure"], errors="coerce").fillna(0.0)
    panel["case_rate_100k"] = np.where(
        pd.to_numeric(panel["total_population"], errors="coerce").gt(0),
        100000.0 * panel["new_cases"] / pd.to_numeric(panel["total_population"], errors="coerce"),
        np.nan,
    )
    panel["death_rate_100k"] = np.where(
        pd.to_numeric(panel["total_population"], errors="coerce").gt(0),
        100000.0 * panel["new_deaths"] / pd.to_numeric(panel["total_population"], errors="coerce"),
        np.nan,
    )
    panel["attendance_share_pct"] = np.where(
        pd.to_numeric(panel["total_population"], errors="coerce").gt(0),
        100.0 * panel["attendance_count"] / pd.to_numeric(panel["total_population"], errors="coerce"),
        np.nan,
    )
    after_rows = compute_missingness_rows(
        panel,
        [
            "new_cases",
            "new_deaths",
            "case_rate_100k",
            "death_rate_100k",
            "attendance_count",
            "attendance_share_pct",
        ],
    )
    return panel, before_rows, after_rows


def build_geometry_network_artifacts(counties_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gpkg_path = PROCESSED_DIR / "nfl_covid_counties.gpkg"
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
    centroid_table.to_csv(PROCESSED_DIR / "nfl_covid_county_centroids.csv", index=False)

    contiguity = build_touching_edge_list(
        projected[["fips", "geometry"]].copy(),
        id_column="fips",
        neighbor_column="neighbor_fips",
    )
    contiguity.to_csv(PROCESSED_DIR / "nfl_covid_county_contiguity_adjacency.csv.gz", index=False)

    knn_8, kernel_8 = build_knn_and_kernel_edges(
        centroid_table,
        id_column="fips",
        x_column="centroid_x",
        y_column="centroid_y",
        k=8,
    )
    knn_8 = knn_8.rename(columns={"neighbor_id": "neighbor_fips"})
    kernel_8 = kernel_8.rename(columns={"neighbor_id": "neighbor_fips"})
    knn_8.to_csv(PROCESSED_DIR / "nfl_covid_county_knn_8_adjacency.csv.gz", index=False)
    kernel_8.to_csv(PROCESSED_DIR / "nfl_covid_county_distance_kernel_8_adjacency.csv.gz", index=False)

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
    network_summary.to_csv(PROCESSED_DIR / "nfl_covid_county_network_summary.csv", index=False)
    return network_summary


def main() -> None:
    args = parse_args()
    paths = ensure_directories()

    sheet_path = download_attendance_sheet(paths)
    raw_geography, _team_metadata = parse_attendance_geography(sheet_path)
    counties, geometry_source = load_or_download_geometry(paths)
    nyt_daily = load_or_download_nyt(paths)
    county_master = build_county_master(counties, nyt_daily)

    full_features, feature_basis_mode = build_full_feature_table(counties, paths)
    full_features_lookup = full_features.drop(columns=["svi_total_population"], errors="ignore").copy()
    modeled_features = full_features_lookup.loc[full_features_lookup["fips"].isin(county_master["fips"])].copy()

    feature_missing_before = compute_missingness_rows(
        modeled_features,
        [
            "population_density",
            "log_population",
            "svi_overall",
            "rucc_2013",
            "senior_population",
            "college_education",
        ],
    )
    for column in [
        "population_density",
        "log_population",
        "svi_overall",
        "rucc_2013",
        "senior_population",
        "college_education",
    ]:
        values = pd.to_numeric(modeled_features[column], errors="coerce")
        median_value = float(values.median()) if values.notna().any() else 0.0
        modeled_features[column] = values.fillna(median_value)
    feature_missing_after = compute_missingness_rows(
        modeled_features,
        [
            "population_density",
            "log_population",
            "svi_overall",
            "rucc_2013",
            "senior_population",
            "college_education",
        ],
    )

    geography_augmented = augment_attendance_geography(raw_geography, counties, county_master)
    geography_augmented.to_csv(PROCESSED_DIR / "nfl_team_county_fan_shares.csv.gz", index=False)

    games = load_or_download_nfl_games(paths)
    attendance_cache = fetch_game_attendance_table(
        paths,
        games,
        reuse_cached=args.reuse_cached_game_attendance,
    )
    game_table = build_game_attendance_table(games, attendance_cache, geography_augmented)
    game_table.to_csv(PROCESSED_DIR / "nfl_game_attendance.csv.gz", index=False)

    county_game_exposure = build_county_game_exposure(game_table, geography_augmented, full_features_lookup)
    county_game_exposure.to_csv(PROCESSED_DIR / "nfl_county_game_exposure.csv.gz", index=False)

    population_lookup = full_features_lookup[["fips", "total_population"]]
    weekly_nyt = aggregate_nyt_weekly(nyt_daily, population_lookup)
    weekly_nyt = weekly_nyt.loc[
        weekly_nyt["WeekStartDate"].between(CORE_START_DATE, CORE_END_DATE)
    ].copy()
    weekly_nyt = weekly_nyt.loc[weekly_nyt["fips"].isin(county_master["fips"])].copy()
    weekly_nyt.to_csv(PROCESSED_DIR / "nfl_county_weekly_nyt.csv.gz", index=False)

    weekly_exposure = aggregate_weekly_exposure(county_game_exposure, modeled_features)
    panel, panel_missing_before, panel_missing_after = build_weekly_panel(
        county_master,
        weekly_nyt,
        weekly_exposure,
        modeled_features,
    )
    panel.to_csv(PROCESSED_DIR / "nfl_county_weekly_panel.csv.gz", index=False)

    modeled_features.to_csv(PROCESSED_DIR / "nfl_county_feature_basis.csv.gz", index=False)
    build_feature_dictionary().to_csv(PROCESSED_DIR / "nfl_county_feature_dictionary.csv", index=False)

    counties_modeled = counties.loc[counties["fips"].isin(county_master["fips"])].copy()
    network_summary = build_geometry_network_artifacts(counties_modeled)

    team_share_sums = (
        raw_geography.groupby("team_name", as_index=False)["fan_share_pct"]
        .sum()
        .rename(columns={"fan_share_pct": "listed_share_pct"})
        .sort_values("listed_share_pct")
        .reset_index(drop=True)
    )
    crosswalk_counts = geography_augmented["match_status"].value_counts(dropna=False).to_dict()
    unsupported_share_rows = int((~geography_augmented["supported_in_county_panel"]).sum())
    teams_with_geography = sorted(raw_geography["team_name"].dropna().unique().tolist())
    teams_without_geography = sorted(set(TEAM_TO_ABBREV) - set(teams_with_geography))
    unsupported_outcome_exposure = county_game_exposure.loc[
        ~county_game_exposure["supported_in_county_panel"], "county_attendance_count"
    ].sum()

    processed_outputs = [
        "nfl_team_county_fan_shares.csv.gz",
        "nfl_game_attendance.csv.gz",
        "nfl_county_game_exposure.csv.gz",
        "nfl_county_weekly_nyt.csv.gz",
        "nfl_county_weekly_panel.csv.gz",
        "nfl_county_feature_basis.csv.gz",
        "nfl_county_feature_dictionary.csv",
        "nfl_covid_counties.gpkg",
        "nfl_covid_county_centroids.csv",
        "nfl_covid_county_contiguity_adjacency.csv.gz",
        "nfl_covid_county_knn_8_adjacency.csv.gz",
        "nfl_covid_county_distance_kernel_8_adjacency.csv.gz",
        "nfl_covid_county_network_summary.csv",
        "processing_summary.json",
    ]

    summary = {
        "source": SOURCE_LABEL,
        "analysis_start_date": CORE_START_DATE.date().isoformat(),
        "analysis_end_date": CORE_END_DATE.date().isoformat(),
        "analysis_end_observation_date": ANALYSIS_END_DATE.date().isoformat(),
        "geometry_source": geometry_source,
        "feature_basis_mode": feature_basis_mode,
        "attendance_geography": {
            "attendance_row_count": int(len(raw_geography)),
            "sheet_team_count_total": int(len(TEAM_TO_ABBREV)),
            "sheet_team_count_with_geography": int(len(teams_with_geography)),
            "sheet_team_count_without_geography": int(len(teams_without_geography)),
            "unique_county_state_pairs": int(len(raw_geography[["county_name_raw", "state_abbrev"]].drop_duplicates())),
            "listed_share_pct_min": float(team_share_sums["listed_share_pct"].min()),
            "listed_share_pct_max": float(team_share_sums["listed_share_pct"].max()),
            "exact_match_count": int(crosswalk_counts.get("exact_match", 0)),
            "override_match_count": int(crosswalk_counts.get("override_match", 0)),
            "ambiguous_match_count": int(crosswalk_counts.get("ambiguous_match", 0)),
            "unresolved_count": int(crosswalk_counts.get("unresolved", 0)),
            "unsupported_in_county_panel_count": unsupported_share_rows,
            "teams_without_geography": ", ".join(teams_without_geography),
        },
        "team_share_sums": team_share_sums.to_dict(orient="records"),
        "game_attendance": {
            "total_games": int(len(game_table)),
            "positive_attendance_games": int(game_table["attendance"].gt(0).sum()),
            "zero_attendance_games": int(game_table["attendance"].eq(0).sum()),
            "missing_attendance_games": int(game_table["attendance"].isna().sum()),
            "date_range_start": game_table["gameday"].min().date().isoformat(),
            "date_range_end": game_table["gameday"].max().date().isoformat(),
            "excluded_arizona_games": int(game_table["excluded_arizona_home_game"].sum()),
            "excluded_arizona_attendance": float(
                game_table.loc[game_table["excluded_arizona_home_game"], "attendance"].fillna(0.0).sum()
            ),
            "zeroed_unsupported_games": int(game_table["zeroed_unsupported_home_game"].sum()),
            "zeroed_unsupported_attendance": float(
                game_table.loc[game_table["zeroed_unsupported_home_game"], "attendance"].fillna(0.0).sum()
            ),
        },
        "game_type_counts": (
            game_table.groupby("game_type", as_index=False)
            .size()
            .rename(columns={"size": "game_count"})
            .to_dict(orient="records")
        ),
        "county_game_exposure": {
            "county_game_row_count": int(len(county_game_exposure)),
            "county_game_modeled_row_count": int(county_game_exposure["supported_in_county_panel"].sum()),
            "assigned_attendance_total": float(county_game_exposure["county_attendance_count"].sum()),
            "assigned_attendance_modeled_total": float(
                county_game_exposure.loc[county_game_exposure["supported_in_county_panel"], "county_attendance_count"].sum()
            ),
            "unassigned_attendance_total": float(game_table["unassigned_attendance_count"].sum()),
            "unsupported_outcome_attendance_total": float(unsupported_outcome_exposure),
        },
        "county_week_panel": {
            "county_count": int(panel["fips"].nunique()),
            "week_count": int(panel["WeekStartDate"].nunique()),
            "row_count": int(len(panel)),
            "case_threshold_for_retention": 200,
            "counties_removed_below_case_threshold": int(len(counties) - len(county_master)),
        },
        "panel_missingness_before_fill": panel_missing_before,
        "panel_missingness_after_fill": panel_missing_after,
        "feature_missingness_before_imputation": feature_missing_before,
        "feature_missingness_after_imputation": feature_missing_after,
        "network_summary": network_summary.to_dict(orient="records"),
        "processed_outputs": processed_outputs,
    }
    dump_json(PROCESSED_DIR / "processing_summary.json", summary)
    write_readme()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        warnings.warn(f"NFL COVID data preparation failed: {exc}", stacklevel=2)
        raise
