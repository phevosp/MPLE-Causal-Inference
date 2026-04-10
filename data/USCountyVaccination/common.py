from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "USCountyVaccination_US"

CORE_START_DATE = pd.Timestamp("2020-01-26")
CORE_END_DATE = pd.Timestamp("2022-05-15")
BOOSTER_START_DATE = pd.Timestamp("2021-12-19")

UNIT_LABEL = "US county"
SOURCE_LABEL = "USCountyVaccination"
STATE_SCOPE_LABEL = "United States"

NYT_COUNTIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
BANSAL_VACCINATION_URL = (
    "https://media.githubusercontent.com/media/"
    "bansallab/vaccinetracking/main/vacc_data/data_county_timeseries.csv"
)
CDC_VACCINATION_URL = (
    "https://data.cdc.gov/resource/8xkx-amqh.csv"
    "?$select=date,fips,mmwr_week,recip_county,recip_state,"
    "administered_dose1_recip,administered_dose1_pop_pct,"
    "series_complete_yes,series_complete_pop_pct,booster_doses,census2019"
    "&$limit=5000000"
)
TIGER_2021_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"
TIGER_2022_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
CDC_SVI_2022_US_COUNTY_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/SVI_2022_US_county.csv"
CDC_SVI_2022_PR_COUNTY_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/PuertoRico_county.csv"
USDA_ERS_RUCC_2023_URL = "https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv?v=33808"

ACS_2021_COUNTY_ENDPOINTS = {
    "acs5": "https://api.census.gov/data/2021/acs/acs5",
    "subject": "https://api.census.gov/data/2021/acs/acs5/subject",
    "profile": "https://api.census.gov/data/2021/acs/acs5/profile",
}

DEFAULT_NETWORKS = ("contiguity", "knn_8", "distance_kernel_8")


@dataclass(frozen=True)
class OutcomeSpec:
    code: str
    label: str
    source_column: str
    threshold: float | None
    notes: str
    guarded: bool = False


@dataclass(frozen=True)
class InterventionSpec:
    code: str
    label: str
    source_column: str
    threshold: float
    family: str
    notes: str


OUTCOME_SPECS = {
    "case_rate_100k_ge_100": OutcomeSpec(
        code="case_rate_100k_ge_100",
        label="case_rate_100k >= 100",
        source_column="case_rate_100k",
        threshold=100.0,
        notes="+1 when weekly case rate per 100k is at least 100.",
    ),
    "case_rate_100k_ge_200": OutcomeSpec(
        code="case_rate_100k_ge_200",
        label="case_rate_100k >= 200",
        source_column="case_rate_100k",
        threshold=200.0,
        notes="+1 when weekly case rate per 100k is at least 200.",
    ),
    "death_rate_100k_ge_2": OutcomeSpec(
        code="death_rate_100k_ge_2",
        label="death_rate_100k >= 2",
        source_column="death_rate_100k",
        threshold=2.0,
        notes="+1 when weekly death rate per 100k is at least 2.",
    ),
}

INTERVENTION_SPECS = {
    "complete_cov_ge_10": InterventionSpec(
        code="complete_cov_ge_10",
        label="complete_cov >= 10",
        source_column="complete_cov",
        threshold=10.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 10 percentage points.",
    ),
    "complete_cov_ge_20": InterventionSpec(
        code="complete_cov_ge_20",
        label="complete_cov >= 20",
        source_column="complete_cov",
        threshold=20.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 20 percentage points.",
    ),
    "complete_cov_ge_30": InterventionSpec(
        code="complete_cov_ge_30",
        label="complete_cov >= 30",
        source_column="complete_cov",
        threshold=30.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 30 percentage points.",
    ),
    "complete_cov_ge_40": InterventionSpec(
        code="complete_cov_ge_40",
        label="complete_cov >= 40",
        source_column="complete_cov",
        threshold=40.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 40 percentage points.",
    ),
    "complete_cov_ge_50": InterventionSpec(
        code="complete_cov_ge_50",
        label="complete_cov >= 50",
        source_column="complete_cov",
        threshold=50.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 50 percentage points.",
    ),
    "complete_cov_ge_60": InterventionSpec(
        code="complete_cov_ge_60",
        label="complete_cov >= 60",
        source_column="complete_cov",
        threshold=60.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 60 percentage points.",
    ),
    "complete_cov_ge_70": InterventionSpec(
        code="complete_cov_ge_70",
        label="complete_cov >= 70",
        source_column="complete_cov",
        threshold=70.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 70 percentage points.",
    ),
    "complete_cov_ge_80": InterventionSpec(
        code="complete_cov_ge_80",
        label="complete_cov >= 80",
        source_column="complete_cov",
        threshold=80.0,
        family="core",
        notes="+1 when complete vaccination coverage is at least 80 percentage points.",
    ),
    "partial_cov_ge_10": InterventionSpec(
        code="partial_cov_ge_10",
        label="partial_cov >= 10",
        source_column="partial_cov",
        threshold=10.0,
        family="core",
        notes="+1 when first-dose coverage is at least 10 percentage points.",
    ),
    "partial_cov_ge_20": InterventionSpec(
        code="partial_cov_ge_20",
        label="partial_cov >= 20",
        source_column="partial_cov",
        threshold=20.0,
        family="core",
        notes="+1 when first-dose coverage is at least 20 percentage points.",
    ),
    "partial_cov_ge_30": InterventionSpec(
        code="partial_cov_ge_30",
        label="partial_cov >= 30",
        source_column="partial_cov",
        threshold=30.0,
        family="core",
        notes="+1 when first-dose coverage is at least 30 percentage points.",
    ),
    "partial_cov_ge_40": InterventionSpec(
        code="partial_cov_ge_40",
        label="partial_cov >= 40",
        source_column="partial_cov",
        threshold=40.0,
        family="core",
        notes="+1 when first-dose coverage is at least 40 percentage points.",
    ),
    "partial_cov_ge_50": InterventionSpec(
        code="partial_cov_ge_50",
        label="partial_cov >= 50",
        source_column="partial_cov",
        threshold=50.0,
        family="core",
        notes="+1 when first-dose coverage is at least 50 percentage points.",
    ),
    "partial_cov_ge_60": InterventionSpec(
        code="partial_cov_ge_60",
        label="partial_cov >= 60",
        source_column="partial_cov",
        threshold=60.0,
        family="core",
        notes="+1 when first-dose coverage is at least 60 percentage points.",
    ),
    "partial_cov_ge_70": InterventionSpec(
        code="partial_cov_ge_70",
        label="partial_cov >= 70",
        source_column="partial_cov",
        threshold=70.0,
        family="core",
        notes="+1 when first-dose coverage is at least 70 percentage points.",
    ),
    "partial_cov_ge_80": InterventionSpec(
        code="partial_cov_ge_80",
        label="partial_cov >= 80",
        source_column="partial_cov",
        threshold=80.0,
        family="core",
        notes="+1 when first-dose coverage is at least 80 percentage points.",
    ),
}

DEFAULT_CORE_OUTCOMES = (
    "case_rate_100k_ge_100",
    "case_rate_100k_ge_200",
    "death_rate_100k_ge_2",
)
DEFAULT_CORE_INTERVENTIONS = (
    "complete_cov_ge_10",
    "complete_cov_ge_20",
    "complete_cov_ge_30",
    "complete_cov_ge_40",
    "complete_cov_ge_50",
    "complete_cov_ge_60",
    "complete_cov_ge_70",
    "complete_cov_ge_80",
    "partial_cov_ge_10",
    "partial_cov_ge_20",
    "partial_cov_ge_30",
    "partial_cov_ge_40",
    "partial_cov_ge_50",
    "partial_cov_ge_60",
    "partial_cov_ge_70",
    "partial_cov_ge_80",
)
DEFAULT_BOOSTER_INTERVENTIONS = ()

DEFAULT_CORE_LAGS = ("0w", "1w", "2w", "3w", "4w")
DEFAULT_BOOSTER_LAGS = ("0w", "1w", "2w")

if any(
    spec.source_column == "partial_cov" and float(spec.threshold) < 10.0
    for spec in INTERVENTION_SPECS.values()
):
    raise ValueError("partial_cov intervention thresholds below 10% are not allowed.")


def ensure_directories() -> dict[str, Path]:
    paths = {
        "base": BASE_DIR,
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "raw_nyt": RAW_DIR / "nyt",
        "raw_vaccination": RAW_DIR / "vaccination",
        "raw_geography": RAW_DIR / "geography",
        "raw_features": RAW_DIR / "features",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def standardize_fips(values: pd.Series | np.ndarray) -> pd.Series:
    return (
        pd.Series(values, copy=False)
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def week_end_from_iso(iso_year: int, iso_week: int) -> pd.Timestamp:
    return pd.Timestamp.fromisocalendar(int(iso_year), int(iso_week), 7)


def add_iso_week_window(
    frame: pd.DataFrame,
    iso_year_col: str = "iso_year",
    iso_week_col: str = "iso_week",
) -> pd.DataFrame:
    result = frame.copy()
    result["WeekEndDate"] = pd.to_datetime(
        [week_end_from_iso(year_value, week_value) for year_value, week_value in zip(result[iso_year_col], result[iso_week_col])]
    )
    result["WeekStartDate"] = result["WeekEndDate"] - pd.Timedelta(days=6)
    return result


def lag_code_to_steps(lag_code: str) -> int:
    if not lag_code.endswith("w"):
        raise ValueError(f"Unsupported lag code '{lag_code}'. Expected codes like 0w or 2w.")
    return int(lag_code[:-1])


def load_sparse_network_from_edge_list(
    edge_path: Path,
    node_order: list[str],
    source_column: str,
    target_column: str,
    weight_column: str | None = None,
) -> sparse.csr_matrix:
    edges = pd.read_csv(edge_path, dtype={source_column: str, target_column: str})
    return build_sparse_network_from_edges(edges, node_order, source_column, target_column, weight_column)


def build_sparse_network_from_edges(
    edges: pd.DataFrame,
    node_order: list[str],
    source_column: str,
    target_column: str,
    weight_column: str | None = None,
) -> sparse.csr_matrix:
    """Build a normalized sparse matrix from an in-memory edge list."""
    lookup = {node_id: idx for idx, node_id in enumerate(node_order)}
    rows = edges[source_column].map(lookup).to_numpy()
    cols = edges[target_column].map(lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    if weight_column is None:
        data = np.ones(len(rows), dtype=float)
    else:
        data = (
            pd.to_numeric(edges.loc[valid, weight_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
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
    if not outcome_part.startswith("outcome_"):
        raise ValueError(f"Missing outcome_ prefix in '{experiment_name_value}'.")
    if not intervention_part.startswith("intervention_"):
        raise ValueError(f"Missing intervention_ prefix in '{experiment_name_value}'.")
    if not lag_part.startswith("lag_"):
        raise ValueError(f"Missing lag_ prefix in '{experiment_name_value}'.")
    return {
        "outcome_code": outcome_part.removeprefix("outcome_"),
        "intervention_code": intervention_part.removeprefix("intervention_"),
        "lag_code": lag_part.removeprefix("lag_"),
        "network_name": network_name,
    }


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
