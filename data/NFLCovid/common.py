from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import normalize_sparse_matrix_infinity  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
OVERRIDE_PATH = BASE_DIR / "county_fips_overrides.csv"
README_PATH = BASE_DIR / "README.md"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "NFLCovid_US"

CORE_START_DATE = pd.Timestamp("2020-04-05")
CORE_END_DATE = pd.Timestamp("2021-02-21")

SOURCE_LABEL = "NFLCovid"
STATE_SCOPE_LABEL = "United States"
UNIT_LABEL = "US county"

ATTENDANCE_SHEET_SHARE_URL = "https://tinyurl.com/bdemmhx8"
ATTENDANCE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1DRMB5FLC3tdngeurDwps1CS8-6smEZvmq5R-ghjEv5k/export?format=csv&gid=0"
)
NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
ESPN_SUMMARY_API_TEMPLATE = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
)
NYT_COUNTIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
TIGER_2021_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"
TIGER_2022_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
SVI_2020_US_COUNTY_URL = "https://svi.cdc.gov/Documents/Data/2020/CSV/states_counties/SVI_2020_US_county.csv"
RUCC_2013_URL = "https://www.ers.usda.gov/media/5769/2013-rural-urban-continuum-codes.xls?v=15208"

ACS_2019_COUNTY_ENDPOINTS = {
    "acs5": "https://api.census.gov/data/2019/acs/acs5",
    "subject": "https://api.census.gov/data/2019/acs/acs5/subject",
    "profile": "https://api.census.gov/data/2019/acs/acs5/profile",
}

ALLOWED_GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")
SUPPORTED_LAGS = ("0w", "1w", "2w", "3w", "4w")
DEFAULT_LAGS = ("2w",)
DEFAULT_NETWORKS = ("distance_kernel_8", "contiguity", "knn_8")

OUTCOME_PREFIX = "x_"
INTERVENTION_PREFIX = "z_"
OUTCOME_COLUMN_PREFIXES = ("x_case_rate_100k_ge_", "x_death_rate_100k_ge_")
INTERVENTION_COLUMN_PREFIX = "z_attendance_share_pct_ge_"
CASE_OUTCOME_FAMILY = "case_rate_100k"
DEATH_OUTCOME_FAMILY = "death_rate_100k"

TEAM_TO_ABBREV = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Green Bay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "Las Vegas": "LV",
    "LA Chargers": "LAC",
    "LA Rams": "LA",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "New England": "NE",
    "New Orleans": "NO",
    "NY Giants": "NYG",
    "NY Jets": "NYJ",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "San Francisco": "SF",
    "Seattle": "SEA",
    "Tampa Bay": "TB",
    "Tennessee": "TEN",
    "Washington": "WAS",
}

ABBR_TO_TEAM = {value: key for key, value in TEAM_TO_ABBREV.items()}

STATE_ABBREV_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
    "PR": "72",
    "VI": "78",
}

STATE_FIPS_TO_NAME = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "60": "American Samoa",
    "66": "Guam",
    "69": "Northern Mariana Islands",
    "72": "Puerto Rico",
    "78": "Virgin Islands",
}


def ensure_directories() -> dict[str, Path]:
    paths = {
        "base": BASE_DIR,
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "raw_attendance": RAW_DIR / "attendance",
        "raw_nfl": RAW_DIR / "nfl",
        "raw_nyt": RAW_DIR / "nyt",
        "raw_geography": RAW_DIR / "geography",
        "raw_features": RAW_DIR / "features",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def standardize_fips(values: pd.Series | np.ndarray | list[object]) -> pd.Series:
    return (
        pd.Series(values, copy=False)
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .str.zfill(5)
    )


def normalize_county_name(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("saint ", "st ")
    text = text.replace("st. ", "st ")
    text = re.sub(r"\bcity and borough\b", "", text)
    text = re.sub(r"\bcity\b", "", text)
    text = re.sub(r"\bborough\b", "", text)
    text = re.sub(r"\bcensus area\b", "", text)
    text = re.sub(r"\bcounty\b", "", text)
    text = re.sub(r"\bparish\b", "", text)
    text = re.sub(r"\bmunicipality\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sunday_week_start(date_values: pd.Series | pd.DatetimeIndex | pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(date_values)
    offsets = (dates.dt.dayofweek + 1) % 7
    return dates - pd.to_timedelta(offsets, unit="D")


def add_sunday_week_window(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result["WeekStartDate"] = sunday_week_start(pd.to_datetime(result[date_column]))
    result["WeekEndDate"] = result["WeekStartDate"] + pd.Timedelta(days=6)
    result["week_year"] = result["WeekStartDate"].dt.year.astype(int)
    result["week_index"] = (
        ((result["WeekStartDate"] - CORE_START_DATE) / pd.Timedelta(days=7)).round().astype(int)
    )
    return result


def build_sunday_week_calendar(
    start_date: pd.Timestamp = CORE_START_DATE,
    end_date: pd.Timestamp = CORE_END_DATE,
) -> pd.DataFrame:
    week_start = pd.date_range(start_date, end_date, freq="W-SUN")
    calendar = pd.DataFrame({"WeekStartDate": week_start})
    calendar["WeekEndDate"] = calendar["WeekStartDate"] + pd.Timedelta(days=6)
    calendar["week_year"] = calendar["WeekStartDate"].dt.year.astype(int)
    calendar["week_index"] = np.arange(len(calendar), dtype=int)
    return calendar


def lag_code_to_steps(lag_code: str) -> int:
    if not lag_code.endswith("w"):
        raise ValueError(f"Unsupported lag code '{lag_code}'. Expected codes like 0w or 2w.")
    return int(lag_code[:-1])


def threshold_to_code(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    token = f"{float(threshold):.10f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def code_to_threshold(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def outcome_code_from_threshold(outcome_family: str, threshold: float) -> str:
    if outcome_family not in {CASE_OUTCOME_FAMILY, DEATH_OUTCOME_FAMILY}:
        raise ValueError(
            f"Unsupported NFL COVID outcome family '{outcome_family}'. "
            f"Expected one of {CASE_OUTCOME_FAMILY!r} or {DEATH_OUTCOME_FAMILY!r}."
        )
    return f"{outcome_family}_ge_{threshold_to_code(threshold)}"


def intervention_code_from_threshold(threshold: float) -> str:
    return f"attendance_share_pct_ge_{threshold_to_code(threshold)}"


def outcome_label_from_code(code: str) -> str:
    if code.startswith(f"{CASE_OUTCOME_FAMILY}_ge_"):
        threshold_token = code.removeprefix(f"{CASE_OUTCOME_FAMILY}_ge_")
        return f"{CASE_OUTCOME_FAMILY} >= {code_to_threshold(threshold_token):g}"
    if code.startswith(f"{DEATH_OUTCOME_FAMILY}_ge_"):
        threshold_token = code.removeprefix(f"{DEATH_OUTCOME_FAMILY}_ge_")
        return f"{DEATH_OUTCOME_FAMILY} >= {code_to_threshold(threshold_token):g}"
    return code


def intervention_label_from_code(code: str) -> str:
    threshold_token = code.removeprefix("attendance_share_pct_ge_")
    return f"attendance_share_pct >= {code_to_threshold(threshold_token):g}"


def outcome_sort_key(code: str) -> tuple[int, float | str]:
    if code.startswith(f"{CASE_OUTCOME_FAMILY}_ge_"):
        return (0, code_to_threshold(code.removeprefix(f"{CASE_OUTCOME_FAMILY}_ge_")))
    if code.startswith(f"{DEATH_OUTCOME_FAMILY}_ge_"):
        return (1, code_to_threshold(code.removeprefix(f"{DEATH_OUTCOME_FAMILY}_ge_")))
    return (99, code)


def discover_outcome_codes(columns: list[str] | pd.Index) -> tuple[str, ...]:
    discovered = [
        column.removeprefix(OUTCOME_PREFIX).removesuffix("_pm1")
        for column in columns
        if str(column).startswith(OUTCOME_PREFIX)
        and str(column).endswith("_pm1")
        and any(str(column).startswith(prefix) for prefix in OUTCOME_COLUMN_PREFIXES)
    ]
    return tuple(sorted(discovered, key=outcome_sort_key))


def discover_intervention_codes(columns: list[str] | pd.Index) -> tuple[str, ...]:
    discovered = [
        column.removeprefix(INTERVENTION_PREFIX).removesuffix("_pm1")
        for column in columns
        if str(column).startswith(INTERVENTION_COLUMN_PREFIX) and str(column).endswith("_pm1")
    ]
    return tuple(sorted(discovered, key=lambda code: code_to_threshold(code.rsplit("_", 1)[-1])))


def experiment_name(outcome_code: str, intervention_code: str, lag_code: str, network_name: str) -> str:
    return (
        f"outcome_{outcome_code}"
        f"__intervention_{intervention_code}"
        f"__lag_{lag_code}"
        f"__{network_name}"
    )


def parse_experiment_name(experiment_name_value: str) -> dict[str, str]:
    parts = experiment_name_value.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"Experiment folder '{experiment_name_value}' does not match the expected four-part pattern."
        )
    outcome_part, intervention_part, lag_part, network_name = parts
    return {
        "outcome_code": outcome_part.removeprefix("outcome_"),
        "intervention_code": intervention_part.removeprefix("intervention_"),
        "lag_code": lag_part.removeprefix("lag_"),
        "network_name": network_name,
    }


def build_sparse_network_from_edges(
    edges: pd.DataFrame,
    node_order: list[str],
    source_column: str,
    target_column: str,
    weight_column: str | None = None,
) -> sparse.csr_matrix:
    lookup = {node_id: idx for idx, node_id in enumerate(node_order)}
    rows = edges[source_column].map(lookup).to_numpy()
    cols = edges[target_column].map(lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    if weight_column is None:
        data = np.ones(len(rows), dtype=float)
    else:
        data = pd.to_numeric(edges.loc[valid, weight_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(node_order), len(node_order))).tocsr()
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return normalize_sparse_matrix_infinity(matrix)


def sparse_matrix_stats(matrix: sparse.csr_matrix) -> dict[str, float | int]:
    row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    return {
        "nnz": int(matrix.nnz),
        "undirected_edges": int(matrix.nnz // 2),
        "avg_degree": float(row_sums.mean()) if row_sums.size else 0.0,
        "max_degree": float(row_sums.max()) if row_sums.size else 0.0,
        "gamma_inf_norm": float(row_sums.max()) if row_sums.size else 0.0,
        "gamma_fro_norm": float(np.sqrt(matrix.multiply(matrix).sum())) if matrix.nnz else 0.0,
    }


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_number(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value) - round(float(value))) < 1e-10:
            return f"{int(round(float(value))):,}"
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append(
            "| " + " | ".join(_format_number(row.get(key, "")) for key, _ in columns) + " |"
        )
    return "\n".join([header, rule, *body_lines])


def _rows_from_dict(mapping: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in mapping.items():
        rows.append({"metric": key, "value": value})
    return rows


def write_readme(
    processing_summary_path: Path = PROCESSED_DIR / "processing_summary.json",
    diagnostics_path: Path = PROCESSED_DIR / "nfl_covid_binary_threshold_diagnostics.csv",
    support_summary_path: Path = PROCESSED_DIR / "nfl_covid_realized_support_summary.csv",
) -> None:
    if not processing_summary_path.exists():
        return

    summary = load_json(processing_summary_path)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    support = pd.read_csv(support_summary_path) if support_summary_path.exists() else pd.DataFrame()

    lines: list[str] = [
        "# NFLCovid",
        "",
        "Nationwide county-week NFL attendance and COVID case/death dataset for MPLE experiments.",
        "",
        "## Sources",
        "",
        f"- Attendance geography Google Sheet: `{ATTENDANCE_SHEET_SHARE_URL}`",
        f"- NFL schedule/calendar: `{NFLVERSE_GAMES_URL}`",
        f"- Game attendance: ESPN summary API `gameInfo.attendance` via `{ESPN_SUMMARY_API_TEMPLATE}`",
        f"- COVID outcomes: `{NYT_COUNTIES_URL}`",
        f"- County geometry: `{TIGER_2021_COUNTY_URL}` with `{TIGER_2022_COUNTY_URL}` fallback",
        f"- ACS 2019 county covariates: `{ACS_2019_COUNTY_ENDPOINTS['acs5']}` and profile/subject companions",
        f"- CDC/ATSDR SVI 2020 county data: `{SVI_2020_US_COUNTY_URL}`",
        f"- USDA ERS RUCC 2013: `{RUCC_2013_URL}`",
        "",
        "## Dataset Snapshot",
        "",
    ]

    attendance_summary = summary.get("attendance_geography", {})
    lines.extend(
        [
            "### Raw attendance geography",
            "",
            markdown_table(
                _rows_from_dict(attendance_summary),
                [("metric", "Metric"), ("value", "Value")],
            ),
            "",
        ]
    )
    share_rows = summary.get("team_share_sums", [])
    if share_rows:
        lines.extend(
            [
                "County-share totals by team:",
                "",
                markdown_table(
                    share_rows,
                    [("team_name", "Team"), ("listed_share_pct", "Listed share pct")],
                ),
                "",
            ]
        )

    game_summary = summary.get("game_attendance", {})
    lines.extend(
        [
            "### Game attendance",
            "",
            markdown_table(
                _rows_from_dict(game_summary),
                [("metric", "Metric"), ("value", "Value")],
            ),
            "",
        ]
    )
    game_type_rows = summary.get("game_type_counts", [])
    if game_type_rows:
        lines.extend(
            [
                "Game counts by type:",
                "",
                markdown_table(game_type_rows, [("game_type", "Type"), ("game_count", "Games")]),
                "",
            ]
        )

    county_game_summary = summary.get("county_game_exposure", {})
    lines.extend(
        [
            "### County-game exposure",
            "",
            markdown_table(
                _rows_from_dict(county_game_summary),
                [("metric", "Metric"), ("value", "Value")],
            ),
            "",
        ]
    )

    panel_summary = summary.get("county_week_panel", {})
    lines.extend(
        [
            "### County-week panel",
            "",
            markdown_table(
                _rows_from_dict(panel_summary),
                [("metric", "Metric"), ("value", "Value")],
            ),
            "",
        ]
    )
    panel_missing_before_rows = summary.get("panel_missingness_before_fill", [])
    if panel_missing_before_rows:
        lines.extend(
            [
                "Panel missingness before fills:",
                "",
                markdown_table(
                    panel_missing_before_rows,
                    [
                        ("column", "Column"),
                        ("missing_count", "Missing count"),
                        ("missing_share", "Missing share"),
                    ],
                ),
                "",
            ]
        )
    panel_missing_after_rows = summary.get("panel_missingness_after_fill", [])
    if panel_missing_after_rows:
        lines.extend(
            [
                "Panel missingness after fills:",
                "",
                markdown_table(
                    panel_missing_after_rows,
                    [
                        ("column", "Column"),
                        ("missing_count", "Missing count"),
                        ("missing_share", "Missing share"),
                    ],
                ),
                "",
            ]
        )

    before_rows = summary.get("feature_missingness_before_imputation", [])
    if before_rows:
        lines.extend(
            [
                "Feature missingness before imputation:",
                "",
                markdown_table(
                    before_rows,
                    [
                        ("column", "Column"),
                        ("missing_count", "Missing count"),
                        ("missing_share", "Missing share"),
                    ],
                ),
                "",
            ]
        )
    after_rows = summary.get("feature_missingness_after_imputation", [])
    if after_rows:
        lines.extend(
            [
                "Feature missingness after imputation:",
                "",
                markdown_table(
                    after_rows,
                    [
                        ("column", "Column"),
                        ("missing_count", "Missing count"),
                        ("missing_share", "Missing share"),
                    ],
                ),
                "",
            ]
        )

    network_rows = summary.get("network_summary", [])
    if network_rows:
        lines.extend(
            [
                "### Network and support",
                "",
                markdown_table(
                    network_rows,
                    [
                        ("network_name", "Network"),
                        ("node_count", "Nodes"),
                        ("edge_count", "Edges"),
                        ("connected_components", "Components"),
                    ],
                ),
                "",
            ]
        )
    if not diagnostics.empty:
        lines.extend(
            [
                "Binary threshold diagnostics:",
                "",
                markdown_table(
                    diagnostics.to_dict(orient="records"),
                    [
                        ("definition_type", "Type"),
                        ("column", "Column"),
                        ("eligible_rows", "Eligible rows"),
                        ("eligible_share", "Eligible share"),
                        ("positive_share", "Positive share"),
                        ("transition_rate", "Transition rate"),
                    ],
                ),
                "",
            ]
        )
    if not support.empty:
        lines.extend(
            [
                "Realized dense-support summary:",
                "",
                markdown_table(
                    support.to_dict(orient="records"),
                    [
                        ("outcome_code", "Outcome"),
                        ("intervention_code", "Intervention"),
                        ("lag_code", "Lag"),
                        ("requested_node_count", "Requested nodes"),
                        ("realized_node_count", "Realized nodes"),
                        ("requested_calendar_weeks", "Requested weeks"),
                        ("realized_calendar_weeks", "Realized weeks"),
                    ],
                ),
                "",
            ]
        )

    processed_outputs = summary.get("processed_outputs", [])
    if processed_outputs:
        lines.extend(
            [
                "## Processed Outputs",
                "",
                *(f"- `{name}`" for name in processed_outputs),
                "",
            ]
        )

    README_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
