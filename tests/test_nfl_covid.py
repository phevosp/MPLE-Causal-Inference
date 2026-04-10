from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NFL_COVID_DIR = REPO_ROOT / "data" / "NFLCovid"
WORKSPACE_TEMP_DIR = REPO_ROOT / ".tmp_tests"
WORKSPACE_TEMP_DIR.mkdir(exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NFL_COVID_DIR) not in sys.path:
    sys.path.insert(0, str(NFL_COVID_DIR))

import build_binary_outcomes as binary_builder  # noqa: E402
import prepare_nfl_covid_data as prep  # noqa: E402
from common import (  # noqa: E402
    CASE_OUTCOME_FAMILY,
    CORE_START_DATE,
    DEATH_OUTCOME_FAMILY,
    discover_outcome_codes,
    normalize_county_name,
    outcome_code_from_threshold,
    outcome_label_from_code,
)


class NFLCovidTests(unittest.TestCase):
    def test_parse_attendance_geography_extracts_rows_and_metadata(self) -> None:
        frame = pd.DataFrame(
            [
                ["Buffalo", "9/13/2020", "Erie", "NY", 30.5, "Monroe", "NY", 10.0, "first note"],
                [pd.NA, "9/13/2020", "COUNTY DATA NOT AVAILABLE", "NY", pd.NA, "Albany", "NY", 5.0, "first note"],
            ],
            columns=["team", "open", "stadium_county", "stadium_state", "stadium_pct", "neutral_county", "neutral_state", "neutral_pct", "notes"],
        )
        sheet_path = WORKSPACE_TEMP_DIR / "attendance_sheet_fixture.csv"
        try:
            frame.to_csv(sheet_path, index=False)
            geography, metadata = prep.parse_attendance_geography(sheet_path)
        finally:
            if sheet_path.exists():
                sheet_path.unlink()

        self.assertEqual(len(geography), 3)
        self.assertEqual(sorted(geography["county_name_raw"].tolist()), ["Albany", "Erie", "Monroe"])
        self.assertEqual(sorted(geography["fan_share_role"].unique().tolist()), ["neutral", "stadium"])
        self.assertEqual(metadata.loc[0, "team_name"], "Buffalo")
        self.assertEqual(metadata.loc[0, "team_abbrev"], "BUF")
        self.assertEqual(metadata.loc[0, "open_date_raw"], "9/13/2020")
        self.assertEqual(metadata.loc[0, "notes_raw"], "first note")

    def test_augment_attendance_geography_applies_overrides(self) -> None:
        geography = pd.DataFrame(
            [
                {
                    "team_name": "Baltimore",
                    "team_abbrev": "BAL",
                    "fan_share_role": "stadium",
                    "county_name_raw": "Baltimore City",
                    "state_abbrev": "MD",
                    "fan_share_pct": 40.0,
                    "county_name_normalized": normalize_county_name("Baltimore City"),
                    "open_date_raw": "9/13/2020",
                    "notes_raw": "",
                },
                {
                    "team_name": "NY Jets",
                    "team_abbrev": "NYJ",
                    "fan_share_role": "stadium",
                    "county_name_raw": "New York",
                    "state_abbrev": "NY",
                    "fan_share_pct": 12.0,
                    "county_name_normalized": normalize_county_name("New York"),
                    "open_date_raw": "9/13/2020",
                    "notes_raw": "",
                },
            ]
        )
        counties = pd.DataFrame(
            [
                {"fips": "24510", "STATEFP": "24", "county_name_normalized": normalize_county_name("Baltimore City")},
                {"fips": "36061", "STATEFP": "36", "county_name_normalized": normalize_county_name("New York")},
            ]
        )
        county_master = pd.DataFrame([{"fips": "24510"}])

        augmented = prep.augment_attendance_geography(geography, counties, county_master)

        baltimore = augmented.loc[augmented["county_name_raw"].eq("Baltimore City")].iloc[0]
        new_york = augmented.loc[augmented["county_name_raw"].eq("New York")].iloc[0]
        self.assertEqual(baltimore["fips"], "24510")
        self.assertEqual(baltimore["match_status"], "override_match")
        self.assertTrue(bool(baltimore["supported_in_county_panel"]))
        self.assertEqual(new_york["fips"], "36061")
        self.assertEqual(new_york["match_status"], "override_match")
        self.assertFalse(bool(new_york["supported_in_county_panel"]))

    def test_aggregate_weekly_exposure_computes_counts_and_share(self) -> None:
        county_game_exposure = pd.DataFrame(
            [
                {
                    "fips": "01001",
                    "WeekStartDate": CORE_START_DATE,
                    "WeekEndDate": CORE_START_DATE + pd.Timedelta(days=6),
                    "week_year": 2020,
                    "week_index": 0,
                    "county_attendance_count": 20.0,
                    "game_id": "A",
                    "supported_in_county_panel": True,
                },
                {
                    "fips": "01001",
                    "WeekStartDate": CORE_START_DATE,
                    "WeekEndDate": CORE_START_DATE + pd.Timedelta(days=6),
                    "week_year": 2020,
                    "week_index": 0,
                    "county_attendance_count": 30.0,
                    "game_id": "B",
                    "supported_in_county_panel": True,
                },
                {
                    "fips": "01001",
                    "WeekStartDate": CORE_START_DATE,
                    "WeekEndDate": CORE_START_DATE + pd.Timedelta(days=6),
                    "week_year": 2020,
                    "week_index": 0,
                    "county_attendance_count": 99.0,
                    "game_id": "C",
                    "supported_in_county_panel": False,
                },
            ]
        )
        modeled_features = pd.DataFrame([{"fips": "01001", "total_population": 10000.0}])

        weekly = prep.aggregate_weekly_exposure(county_game_exposure, modeled_features)

        self.assertEqual(len(weekly), 1)
        self.assertAlmostEqual(float(weekly.loc[0, "attendance_count"]), 50.0)
        self.assertEqual(int(weekly.loc[0, "games_with_exposure"]), 2)
        self.assertAlmostEqual(float(weekly.loc[0, "attendance_share_pct"]), 0.5)

    def test_outcome_helpers_support_case_and_death_rates(self) -> None:
        case_code = outcome_code_from_threshold(CASE_OUTCOME_FAMILY, 200)
        death_code = outcome_code_from_threshold(DEATH_OUTCOME_FAMILY, 2)

        self.assertEqual(case_code, "case_rate_100k_ge_200")
        self.assertEqual(death_code, "death_rate_100k_ge_2")
        self.assertEqual(outcome_label_from_code(case_code), "case_rate_100k >= 200")
        self.assertEqual(outcome_label_from_code(death_code), "death_rate_100k >= 2")
        discovered = discover_outcome_codes(
            [
                "x_death_rate_100k_ge_2_pm1",
                "x_case_rate_100k_ge_200_pm1",
                "x_case_rate_100k_ge_100_pm1",
            ]
        )
        self.assertEqual(
            discovered,
            (
                "case_rate_100k_ge_100",
                "case_rate_100k_ge_200",
                "death_rate_100k_ge_2",
            ),
        )

    def test_build_weekly_panel_keeps_population_and_fills_missing(self) -> None:
        county_master = pd.DataFrame(
            [
                {
                    "fips": "01001",
                    "county": "Autauga",
                    "county_name_normalized": normalize_county_name("Autauga"),
                    "state_name": "Alabama",
                    "STATEFP": "01",
                    "COUNTYFP": "001",
                    "land_area_sq_km": 1000.0,
                    "total_cases_window": 250.0,
                    "meets_case_threshold": True,
                }
            ]
        )
        weekly_nyt = pd.DataFrame(
            [
                {
                    "fips": "01001",
                    "WeekStartDate": CORE_START_DATE,
                    "WeekEndDate": CORE_START_DATE + pd.Timedelta(days=6),
                    "week_year": 2020,
                    "week_index": 0,
                    "new_cases": 10.0,
                    "new_deaths": 1.0,
                    "cases": 10.0,
                    "deaths": 1.0,
                    "available_daily_rows": 7,
                    "case_rate_100k": 100.0,
                    "death_rate_100k": 10.0,
                    "total_population": 10000.0,
                }
            ]
        )
        weekly_exposure = pd.DataFrame(
            [
                {
                    "fips": "01001",
                    "WeekStartDate": CORE_START_DATE,
                    "WeekEndDate": CORE_START_DATE + pd.Timedelta(days=6),
                    "week_year": 2020,
                    "week_index": 0,
                    "attendance_count": 5.0,
                    "attendance_share_pct": 0.05,
                    "games_with_exposure": 1,
                }
            ]
        )
        modeled_features = pd.DataFrame(
            [
                {
                    "fips": "01001",
                    "total_population": 10000.0,
                    "population_density": 10.0,
                    "log_population": 9.0,
                    "svi_overall": 0.2,
                    "rucc_2013": 2.0,
                    "senior_population": 15.0,
                    "college_education": 25.0,
                    "feature_basis_mode": "static_county_covariates",
                }
            ]
        )

        panel, before_rows, after_rows = prep.build_weekly_panel(
            county_master,
            weekly_nyt,
            weekly_exposure,
            modeled_features,
        )

        self.assertIn("total_population", panel.columns)
        self.assertEqual(int(panel["fips"].nunique()), 1)
        self.assertEqual(int(panel["WeekStartDate"].nunique()), 47)
        first_week = panel.loc[panel["week_index"].eq(0)].iloc[0]
        second_week = panel.loc[panel["week_index"].eq(1)].iloc[0]
        self.assertAlmostEqual(float(first_week["case_rate_100k"]), 100.0)
        self.assertAlmostEqual(float(first_week["death_rate_100k"]), 10.0)
        self.assertAlmostEqual(float(first_week["attendance_share_pct"]), 0.05)
        self.assertAlmostEqual(float(second_week["new_cases"]), 0.0)
        self.assertAlmostEqual(float(second_week["new_deaths"]), 0.0)
        self.assertAlmostEqual(float(second_week["attendance_count"]), 0.0)
        after_lookup = {row["column"]: row["missing_count"] for row in after_rows}
        self.assertEqual(after_lookup["new_cases"], 0)
        self.assertEqual(after_lookup["new_deaths"], 0)
        self.assertEqual(after_lookup["death_rate_100k"], 0)
        self.assertEqual(after_lookup["attendance_share_pct"], 0)
        before_lookup = {row["column"]: row["missing_count"] for row in before_rows}
        self.assertGreater(before_lookup["attendance_count"], 0)

    def test_compute_support_summary_respects_two_week_lag(self) -> None:
        rows = []
        for week_index in range(4):
            week_start = CORE_START_DATE + pd.Timedelta(days=7 * week_index)
            week_end = week_start + pd.Timedelta(days=6)
            for fips in ["01001", "01003"]:
                rows.append(
                    {
                        "fips": fips,
                        "WeekStartDate": week_start,
                        "WeekEndDate": week_end,
                        "x_case_rate_100k_ge_200_pm1": 1 if week_index >= 2 else -1,
                        "z_attendance_share_pct_ge_0p01_pm1": 1 if week_index >= 1 else -1,
                    }
                )
        panel = pd.DataFrame(rows)

        summary = binary_builder.compute_support_summary(
            panel,
            "x_case_rate_100k_ge_200_pm1",
            "z_attendance_share_pct_ge_0p01_pm1",
            "2w",
        )

        self.assertEqual(summary["requested_node_count"], 2)
        self.assertEqual(summary["requested_calendar_weeks"], 4)
        self.assertEqual(summary["realized_node_count"], 2)
        self.assertEqual(summary["realized_calendar_weeks"], 2)
        self.assertEqual(summary["weeks_dropped_due_to_missing_or_lag"], 2)
        self.assertEqual(summary["realized_start_date"], "2020-04-19")


if __name__ == "__main__":
    unittest.main()
