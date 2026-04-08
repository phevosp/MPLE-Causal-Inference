"""Materialize and optionally fit Ohio COVID School Data Hub MPLE experiments.

The runner mirrors the SeattleDMI and USCountyVaccination experiment scripts:

- load processed Ohio district-week panels, centroids, and adjacency once
- cache the reusable weekly and monthly Ohio panels
- materialize one requested experiment or the default Ohio grid
- optionally fit MPLE after writing each folder
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (  # noqa: E402
    center_and_normalize_vector_infinity,
    normalize_sparse_matrix_infinity,
    standardize_id,
)
from model_utils import validate_basis_infinity_norms  # noqa: E402


STATE_ABBREV = "OH"
THRESHOLD = 12.5
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "COVIDSchoolDataHub_OH"
PROCESSED_DIR = REPO_ROOT / "data" / "COVIDSchoolDataHub" / "processed"
DEFAULT_SHARE_THRESHOLD = 0.5
DEFAULT_GRID_SPECS: tuple[dict[str, object], ...] = (
    {"panel_frequency": "weekly", "intervention_source": "learning_model", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": None},
    {"panel_frequency": "weekly", "intervention_source": "learning_model", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "2w"},
    {"panel_frequency": "weekly", "intervention_source": "learning_model", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "3w"},
    {"panel_frequency": "weekly", "intervention_source": "learning_model", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "4w"},
    {"panel_frequency": "weekly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": None},
    {"panel_frequency": "weekly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "2w"},
    {"panel_frequency": "weekly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "3w"},
    {"panel_frequency": "weekly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "4w"},
    {"panel_frequency": "monthly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": None},
    {"panel_frequency": "monthly", "intervention_source": "monthly_share", "share_threshold": DEFAULT_SHARE_THRESHOLD, "lag_period": "1m"},
)


def make_experiment_name(
    intervention_source: str,
    share_threshold: float,
    lag_period: str | None = None,
) -> str:
    """Build a stable experiment folder name from the intervention source."""
    if intervention_source == "monthly_share":
        suffix = f"inpersonshare{str(share_threshold).replace('.', 'p')}"
    else:
        suffix = "inperson"
    suffix += lag_suffix(lag_period)
    return f"ohio_case_gt_12p5_{suffix}__contiguity"


def make_monthly_experiment_name(
    intervention_source: str,
    share_threshold: float,
    lag_period: str | None = None,
) -> str:
    """Build a stable experiment folder name for monthly-aggregated Ohio experiments."""
    if intervention_source == "monthly_share":
        suffix = f"monthlyshare{str(share_threshold).replace('.', 'p')}"
    else:
        suffix = "monthlyinperson"
    suffix += lag_suffix(lag_period)
    return f"ohio_monthly_case_gt_12p5_{suffix}__contiguity"


def lag_suffix(lag_period: str | None) -> str:
    """Return a filename-safe suffix for the selected lag period."""
    if lag_period is None:
        return ""
    return f"_lag{lag_period}"


def make_experiment_name_for_spec(spec: dict[str, object]) -> str:
    """Return the canonical experiment folder name for one Ohio spec."""
    panel_frequency = str(spec["panel_frequency"])
    intervention_source = str(spec["intervention_source"])
    share_threshold = float(spec["share_threshold"])
    lag_period = spec.get("lag_period")
    if panel_frequency == "monthly":
        return make_monthly_experiment_name(intervention_source, share_threshold, lag_period=lag_period)
    return make_experiment_name(intervention_source, share_threshold, lag_period=lag_period)


def default_grid_specs() -> list[dict[str, object]]:
    """Return a writable copy of the default Ohio experiment grid."""
    return [dict(spec) for spec in DEFAULT_GRID_SPECS]


def resolve_lag_steps(panel_frequency: str, lag_period: str | None) -> int:
    """Map a requested lag period to the number of rows to shift within each district."""
    if lag_period is None:
        return 0
    if panel_frequency == "weekly":
        if lag_period not in {"2w", "3w", "4w"}:
            raise ValueError("Weekly experiments only support lag periods 2w, 3w, or 4w.")
        return int(lag_period[0])
    if panel_frequency == "monthly":
        if lag_period != "1m":
            raise ValueError("Monthly experiments only support a single-month lag (1m).")
        return 1
    raise ValueError(f"Unknown panel_frequency '{panel_frequency}'.")


def lag_panel_columns(
    panel: pd.DataFrame,
    columns: list[str],
    lag_steps: int,
) -> pd.DataFrame:
    """Lag selected columns within each district by the requested number of rows."""
    if lag_steps <= 0:
        return panel.copy()
    lagged = panel.sort_values(["StateAbbrev", "NCESDistrictID", "WeekStartDate"]).copy()
    lagged[columns] = lagged.groupby(["StateAbbrev", "NCESDistrictID"], sort=False)[columns].shift(lag_steps)
    return lagged


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Ohio district-week panels, district features, monthly shares, centroids, and adjacency."""
    panel = pd.read_csv(
        PROCESSED_DIR / "csdh_learning_case_joined_district_week.csv.gz",
        parse_dates=["WeekStartDate", "WeekEndDate", "PeriodStartDate", "PeriodEndDate"],
        low_memory=False,
    )
    panel = panel.loc[panel["StateAbbrev"] == STATE_ABBREV].copy()
    panel["NCESDistrictID"] = standardize_id(panel["NCESDistrictID"], width=7)

    monthly_share_weekly_path = PROCESSED_DIR / "csdh_learning_case_joined_monthly_shares_district_week_ohio.csv.gz"
    if monthly_share_weekly_path.exists():
        monthly_share_weekly = pd.read_csv(
            monthly_share_weekly_path,
            parse_dates=["WeekStartDate", "WeekEndDate", "MonthStartDate", "MonthEndDate"],
            low_memory=False,
        )
        monthly_share_weekly["NCESDistrictID"] = standardize_id(monthly_share_weekly["NCESDistrictID"], width=7)
        monthly_share_weekly["StateAbbrev"] = standardize_id(monthly_share_weekly["StateAbbrev"])
    else:
        monthly_share_weekly = pd.DataFrame()

    features = pd.read_csv(PROCESSED_DIR / "district_feature_basis_ohio.csv.gz", low_memory=False)
    features["NCESDistrictID"] = standardize_id(features["NCESDistrictID"], width=7)
    features["StateAbbrev"] = standardize_id(features["StateAbbrev"])

    monthly_shares = pd.read_csv(
        PROCESSED_DIR / "csdh_district_monthly_shares_ohio.csv.gz",
        parse_dates=["MonthStartDate", "MonthEndDate"],
        low_memory=False,
    )
    monthly_shares["NCESDistrictID"] = standardize_id(monthly_shares["NCESDistrictID"], width=7)
    monthly_shares["StateAbbrev"] = standardize_id(monthly_shares["StateAbbrev"])

    centroids = pd.read_csv(PROCESSED_DIR / "ohio_standardized_centroids.csv", low_memory=False)
    centroids["NCESDistrictID"] = standardize_id(centroids["NCESDistrictID"], width=7)
    adjacency_edges = pd.read_csv(PROCESSED_DIR / "ohio_standardized_contiguity_adjacency.csv.gz", low_memory=False)
    adjacency_edges["NCESDistrictID"] = standardize_id(adjacency_edges["NCESDistrictID"], width=7)
    adjacency_edges["neighbor_id"] = standardize_id(adjacency_edges["neighbor_id"], width=7)
    adjacency_edges["weight"] = pd.to_numeric(adjacency_edges["weight"], errors="coerce").fillna(1.0)

    return panel, monthly_share_weekly, features, monthly_shares, centroids, adjacency_edges


def load_raw_case_rates() -> pd.DataFrame:
    """Load the raw Ohio district-week-ZIP case-rate rows."""
    case_rates = pd.read_csv(
        PROCESSED_DIR / "csdh_case_rates_by_district_zip_week_ohio.csv.gz",
        parse_dates=["WeekStartDate", "WeekEndDate"],
        low_memory=False,
    )
    case_rates["NCESDistrictID"] = standardize_id(case_rates["NCESDistrictID"], width=7)
    case_rates["StateAbbrev"] = standardize_id(case_rates["StateAbbrev"])
    case_rates["tot_zip_week"] = pd.to_numeric(case_rates["tot_zip_week"], errors="coerce").fillna(0.0)
    return case_rates


def aggregate_case_rates_weekly(case_rates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw district-week-ZIP case rates to one row per district-week."""
    group_cols = ["StateAbbrev", "NCESDistrictID", "WeekStartDate", "WeekEndDate"]

    working = case_rates.copy()
    weight = pd.to_numeric(working["tot_zip_week"], errors="coerce").fillna(0.0)
    working["total_tests"] = pd.to_numeric(working["total_tests"], errors="coerce").fillna(0.0)
    working["total_positives"] = pd.to_numeric(working["total_positives"], errors="coerce").fillna(0.0)
    working["total_negatives"] = pd.to_numeric(working["total_negatives"], errors="coerce").fillna(0.0)
    working["tot_zip_week"] = weight
    district_name_column = "lea_name" if "lea_name" in working.columns else "DistrictName"
    value_columns = ["case_rate_per100k_zip", "case_rate_per100k_state", "positive_rate"]
    for column in value_columns:
        values = pd.to_numeric(working[column], errors="coerce")
        valid_weight = weight.where(values.notna(), 0.0)
        working[f"{column}__weighted_value"] = values.fillna(0.0) * valid_weight
        working[f"{column}__weight"] = valid_weight
        working[column] = values

    aggregated = (
        working.groupby(group_cols, sort=True, dropna=False)
        .agg(
            DistrictName=(district_name_column, "first"),
            StateAssignedDistrictID=("StateAssignedDistrictID", "first"),
            total_tests=("total_tests", "sum"),
            total_positives=("total_positives", "sum"),
            total_negatives=("total_negatives", "sum"),
            tot_zip_week=("tot_zip_week", "sum"),
            zip_count=("zip", "nunique") if "zip" in working.columns else ("NCESDistrictID", "size"),
            case_rate_per100k_zip__weighted_value=("case_rate_per100k_zip__weighted_value", "sum"),
            case_rate_per100k_zip__weight=("case_rate_per100k_zip__weight", "sum"),
            case_rate_per100k_zip__mean=("case_rate_per100k_zip", "mean"),
            case_rate_per100k_state__weighted_value=("case_rate_per100k_state__weighted_value", "sum"),
            case_rate_per100k_state__weight=("case_rate_per100k_state__weight", "sum"),
            case_rate_per100k_state__mean=("case_rate_per100k_state", "mean"),
            positive_rate__weighted_value=("positive_rate__weighted_value", "sum"),
            positive_rate__weight=("positive_rate__weight", "sum"),
            positive_rate__mean=("positive_rate", "mean"),
        )
        .reset_index()
    )
    weekly = aggregated.rename(columns={"zip_count": "zip_count"}).copy()
    for column in value_columns:
        weighted_value = weekly[f"{column}__weighted_value"]
        valid_weight = weekly[f"{column}__weight"]
        fallback_mean = weekly[f"{column}__mean"]
        weekly[column] = np.where(valid_weight > 1e-12, weighted_value / valid_weight, fallback_mean)
        weekly = weekly.drop(
            columns=[
                f"{column}__weighted_value",
                f"{column}__weight",
                f"{column}__mean",
            ]
        )
    weekly["Month"] = weekly["WeekStartDate"].dt.year.astype(str) + "m" + weekly["WeekStartDate"].dt.month.astype(str)
    return weekly.sort_values(["StateAbbrev", "NCESDistrictID", "WeekStartDate"]).reset_index(drop=True)


def attach_monthly_shares(panel: pd.DataFrame, monthly_shares: pd.DataFrame) -> pd.DataFrame:
    """Attach district-month share information to a weekly or monthly panel."""
    merged = panel.copy()
    for column in ["share_inperson", "share_hybrid", "share_virtual"]:
        if column in merged.columns:
            merged = merged.drop(columns=[column])
    if "Month" not in merged.columns:
        merged["Month"] = merged["WeekStartDate"].dt.year.astype(str) + "m" + merged["WeekStartDate"].dt.month.astype(str)
    merged = merged.merge(
        monthly_shares[
            [
                "StateAbbrev",
                "NCESDistrictID",
                "Month",
                "share_inperson",
                "share_hybrid",
                "share_virtual",
            ]
        ],
        on=["StateAbbrev", "NCESDistrictID", "Month"],
        how="left",
    )
    for column in ["share_inperson", "share_hybrid", "share_virtual"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def aggregate_weekly_panel_to_monthly(weekly_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a district-week panel to district-month."""
    monthly = weekly_panel.copy()
    monthly["MonthStartDate"] = pd.to_datetime(monthly["WeekStartDate"].dt.year.astype(str) + "-" + monthly["WeekStartDate"].dt.month.astype(str) + "-01")
    monthly["MonthEndDate"] = monthly["MonthStartDate"] + pd.offsets.MonthEnd(0)
    numeric_mean_columns = [
        "case_rate_per100k_zip",
        "case_rate_per100k_state",
        "positive_rate",
        "share_inperson",
        "share_hybrid",
        "share_virtual",
    ]
    numeric_sum_columns = ["total_tests", "total_positives", "total_negatives", "tot_zip_week"]
    for column in numeric_mean_columns + numeric_sum_columns:
        if column in monthly.columns:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    agg_spec: dict[str, tuple[str, str]] = {
        "DistrictName": ("DistrictName", "first"),
    }
    if "StateAssignedDistrictID" in monthly.columns:
        agg_spec["StateAssignedDistrictID"] = ("StateAssignedDistrictID", "first")
    for column in numeric_mean_columns:
        if column in monthly.columns:
            agg_spec[column] = (column, "mean")
    for column in numeric_sum_columns:
        if column in monthly.columns:
            agg_spec[column] = (column, "sum")

    monthly_panel = (
        monthly.groupby(["StateAbbrev", "NCESDistrictID", "MonthStartDate", "MonthEndDate"], sort=True, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    monthly_panel["WeekStartDate"] = monthly_panel["MonthStartDate"]
    monthly_panel["WeekEndDate"] = monthly_panel["MonthEndDate"]
    monthly_panel["Month"] = monthly_panel["WeekStartDate"].dt.year.astype(str) + "m" + monthly_panel["WeekStartDate"].dt.month.astype(str)
    return monthly_panel.sort_values(["StateAbbrev", "NCESDistrictID", "WeekStartDate"]).reset_index(drop=True)


def build_cached_base_panels(
    learning_panel: pd.DataFrame,
    monthly_share_weekly: pd.DataFrame,
    monthly_shares: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Build the reusable Ohio base panels once per invocation."""
    cached: dict[tuple[str, str], pd.DataFrame] = {
        ("weekly", "learning_model"): learning_panel.copy(),
    }
    if monthly_share_weekly.empty:
        case_rates_raw = load_raw_case_rates()
        monthly_share_weekly = attach_monthly_shares(aggregate_case_rates_weekly(case_rates_raw), monthly_shares)
    cached[("weekly", "monthly_share")] = monthly_share_weekly
    cached[("monthly", "monthly_share")] = attach_monthly_shares(
        aggregate_weekly_panel_to_monthly(monthly_share_weekly),
        monthly_shares,
    )
    return cached


def build_outcome_and_intervention(
    panel: pd.DataFrame,
    intervention_source: str,
    share_threshold: float,
    lag_period: str | None = None,
    panel_frequency: str = "weekly",
) -> pd.DataFrame:
    """Create the binary outcome and intervention columns used in the experiment.

    The outcome is always contemporaneous. If a lag is requested, only the
    intervention series is shifted backward within each district.
    """
    lag_steps = resolve_lag_steps(panel_frequency, lag_period)
    panel = panel.copy()
    panel["Outcome_pm1"] = np.where(panel["case_rate_per100k_zip"] > THRESHOLD, 1, -1).astype(np.int8)

    if intervention_source == "learning_model":
        if lag_steps > 0:
            panel = lag_panel_columns(panel, ["LearningModel"], lag_steps)
        panel["Intervention_pm1"] = np.where(
            panel["LearningModel"].eq("In-person"),
            1,
            -1,
        ).astype(np.int8)
        panel["InterventionShare"] = np.where(panel["LearningModel"].eq("In-person"), 1.0, 0.0)
        panel["intervention_rule"] = (
            "lagged LearningModel == In-person" if lag_steps > 0 else "LearningModel == In-person"
        )
    elif intervention_source == "monthly_share":
        if "share_inperson" not in panel.columns:
            raise KeyError("share_inperson is missing from the panel after merging monthly shares.")
        if lag_steps > 0:
            panel = lag_panel_columns(panel, ["share_inperson"], lag_steps)
        panel["InterventionShare"] = pd.to_numeric(panel["share_inperson"], errors="coerce")
        panel["Intervention_pm1"] = np.where(
            panel["InterventionShare"].fillna(-1) >= share_threshold,
            1,
            -1,
        ).astype(np.int8)
        panel["intervention_rule"] = (
            f"lagged share_inperson >= {share_threshold}"
            if lag_steps > 0
            else f"share_inperson >= {share_threshold}"
        )
    else:
        raise ValueError(f"Unknown intervention_source: {intervention_source}")
    return panel


def build_monthly_panel(
    panel: pd.DataFrame,
    intervention_source: str,
    share_threshold: float,
    lag_period: str | None = None,
    panel_frequency: str = "monthly",
) -> pd.DataFrame:
    """Aggregate the Ohio panel to district-month and build binary x/z there.

    The outcome remains contemporaneous at the monthly level. If a lag is
    requested, only the intervention series is shifted backward within each
    district before thresholding.
    """
    monthly = panel.copy()
    monthly["Month"] = monthly["WeekStartDate"].dt.year.astype(str) + "m" + monthly["WeekStartDate"].dt.month.astype(str)
    group_cols = [
        "StateAbbrev",
        "NCESDistrictID",
        "Month",
        "share_inperson",
        "share_hybrid",
        "share_virtual",
    ]
    agg = (
        monthly.groupby(group_cols, dropna=False)
        .agg(
            DistrictName=("DistrictName", "first"),
            WeekStartDate=("WeekStartDate", "min"),
            WeekEndDate=("WeekEndDate", "max"),
            PeriodStartDate=("PeriodStartDate", "first"),
            PeriodEndDate=("PeriodEndDate", "first"),
            case_rate_per100k_zip=("case_rate_per100k_zip", "mean"),
            case_rate_per100k_state=("case_rate_per100k_state", "mean"),
            positive_rate=("positive_rate", "mean"),
            total_tests=("total_tests", "sum"),
            total_positives=("total_positives", "sum"),
            total_negatives=("total_negatives", "sum"),
            tot_zip_week=("tot_zip_week", "sum"),
        )
        .reset_index()
    )
    lag_steps = resolve_lag_steps(panel_frequency, lag_period)
    agg["Outcome_pm1"] = np.where(agg["case_rate_per100k_zip"] > THRESHOLD, 1, -1).astype(np.int8)
    if intervention_source == "learning_model":
        if lag_steps > 0:
            agg = lag_panel_columns(agg, ["LearningModel"], lag_steps)
        agg["Intervention_pm1"] = np.where(agg["LearningModel"].eq("In-person"), 1, -1).astype(np.int8)
        agg["InterventionShare"] = np.where(agg["LearningModel"].eq("In-person"), 1.0, 0.0)
        agg["intervention_rule"] = (
            "lagged LearningModel == In-person" if lag_steps > 0 else "LearningModel == In-person"
        )
    elif intervention_source == "monthly_share":
        if lag_steps > 0:
            agg = lag_panel_columns(agg, ["share_inperson"], lag_steps)
        agg["InterventionShare"] = pd.to_numeric(agg["share_inperson"], errors="coerce")
        agg["Intervention_pm1"] = np.where(
            agg["InterventionShare"].fillna(-1) >= share_threshold,
            1,
            -1,
        ).astype(np.int8)
        agg["intervention_rule"] = (
            f"lagged share_inperson >= {share_threshold}"
            if lag_steps > 0
            else f"share_inperson >= {share_threshold}"
        )
    else:
        raise ValueError(f"Unknown intervention_source: {intervention_source}")
    return agg


def build_node_table(
    centroids: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the node ordering and compact district feature table."""
    node_table = centroids[["NCESDistrictID", "centroid_x", "centroid_y"]].copy()
    node_table["state_abbrev"] = STATE_ABBREV
    node_table = node_table.sort_values("NCESDistrictID").reset_index(drop=True)
    node_table["node_index"] = np.arange(len(node_table), dtype=int)
    feature_columns = [
        "edge_acsed_total_population",
        "edge_acsed_median_household_income",
        "edge_acsed_pct_below_poverty",
        "edge_acsed_pct_black",
        "edge_acsed_pct_hispanic",
        "edge_acsed_pct_in_labor_force",
    ]
    merged = node_table.merge(
        features[["NCESDistrictID", "DistrictName", "StateAssignedDistrictID"] + feature_columns],
        on="NCESDistrictID",
        how="left",
    )
    return merged.sort_values("NCESDistrictID").reset_index(drop=True)


def build_field_basis(node_table: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct a compact, infinity-normalized field basis."""
    feature_specs = [
        (
            "log_total_population",
            np.log1p(node_table["edge_acsed_total_population"].to_numpy(dtype=float)),
        ),
        (
            "median_household_income",
            node_table["edge_acsed_median_household_income"].to_numpy(dtype=float),
        ),
        (
            "pct_below_poverty",
            node_table["edge_acsed_pct_below_poverty"].to_numpy(dtype=float),
        ),
        ("pct_black", node_table["edge_acsed_pct_black"].to_numpy(dtype=float)),
        ("pct_hispanic", node_table["edge_acsed_pct_hispanic"].to_numpy(dtype=float)),
        (
            "pct_in_labor_force",
            node_table["edge_acsed_pct_in_labor_force"].to_numpy(dtype=float),
        ),
    ]

    basis_vectors: list[np.ndarray] = []
    basis_names: list[str] = []
    for name, raw in feature_specs:
        centered = center_and_normalize_vector_infinity(raw)
        if np.linalg.norm(centered, ord=np.inf) < 1e-12:
            continue
        basis_vectors.append(centered)
        basis_names.append(name)

    if not basis_vectors:
        return np.empty((0, len(node_table)), dtype=float), ()
    field_basis = np.vstack(basis_vectors)
    return field_basis, tuple(basis_names)


def build_panel_arrays(
    panel: pd.DataFrame,
    node_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp], int]:
    """Build x, z, x_0, z_0 arrays for the MPLE fit."""
    ordered = panel.copy()
    ordered = ordered.sort_values(["WeekStartDate", "NCESDistrictID"]).reset_index(drop=True)

    x_pivot = (
        ordered.pivot(index="WeekStartDate", columns="NCESDistrictID", values="Outcome_pm1")
        .reindex(columns=node_order)
        .sort_index()
    )
    z_pivot = (
        ordered.pivot(index="WeekStartDate", columns="NCESDistrictID", values="Intervention_pm1")
        .reindex(columns=node_order)
        .sort_index()
    )
    if x_pivot.isna().any().any():
        missing = int(x_pivot.isna().sum().sum())
        raise ValueError(f"Outcome panel has {missing} missing entries.")
    if z_pivot.isna().any().any():
        missing = int(z_pivot.isna().sum().sum())
        raise ValueError(f"Intervention panel has {missing} missing entries.")

    times = list(x_pivot.index.to_list())
    x_all = x_pivot.to_numpy(dtype=np.int8)
    z_all = z_pivot.to_numpy(dtype=np.int8)
    x_0 = x_all[0].astype(np.int8)
    z_0 = z_all[0].astype(np.int8)
    x = x_all[1:].astype(np.int8)
    z = z_all[1:].astype(np.int8)
    treated_rows = np.any(z == 1, axis=1)
    s = int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])
    return x, z, x_0, z_0, times, s


def build_contiguity_matrix(edge_list: pd.DataFrame, node_order: list[str]) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build a normalized contiguity adjacency matrix from the saved adjacency list."""
    edge_list = edge_list.copy()
    node_lookup = {district_id: idx for idx, district_id in enumerate(node_order)}
    rows = edge_list["NCESDistrictID"].map(node_lookup).to_numpy()
    cols = edge_list["neighbor_id"].map(node_lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    data = pd.to_numeric(edge_list.loc[valid, "weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(node_order), len(node_order))).tocsr()
    matrix = matrix + matrix.T
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    matrix = normalize_sparse_matrix_infinity(matrix)
    return matrix, edge_list


def save_experiment(
    experiment_dir: Path,
    config,
    metadata: dict[str, object],
    panel: pd.DataFrame,
    node_table: pd.DataFrame,
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    gamma_matrix: sparse.csr_matrix,
    adjacency_edges: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    times: list[pd.Timestamp],
) -> None:
    """Persist all artifacts needed by mple.py for a real-data experiment."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, experiment_dir / "realized_config.yaml")
    OmegaConf.save(OmegaConf.create(metadata), experiment_dir / "experiment_metadata.yaml")
    np.savez(experiment_dir / "panel_data.npz", x=x, z=z)
    np.save(experiment_dir / "x_0.npy", x_0)
    np.save(experiment_dir / "z_0.npy", z_0)
    np.save(experiment_dir / "field_basis.npy", field_basis)
    np.save(experiment_dir / "field_basis_names.npy", np.asarray(field_basis_names, dtype="<U128"))
    np.save(experiment_dir / "shared_features.npy", np.empty((0, field_basis.shape[1]), dtype=float))
    np.save(experiment_dir / "shared_feature_names.npy", np.asarray([], dtype="<U1"))
    sparse.save_npz(experiment_dir / "gamma_matrix_sparse.npz", gamma_matrix)
    sparse.save_npz(experiment_dir / "interaction_basis_sparse.npz", gamma_matrix)
    np.save(
        experiment_dir / "interaction_basis_names.npy",
        np.asarray(["adjacency"], dtype="<U128"),
    )
    node_table.to_csv(experiment_dir / "node_index.csv", index=False)
    time_index = pd.DataFrame(
        {
            "model_index": np.arange(len(times) - 1, dtype=int),
            "WeekStartDate": times[1:],
        }
    )
    time_index.to_csv(experiment_dir / "time_index.csv", index=False)
    adjacency_edges.to_csv(experiment_dir / "adjacency_edge_list.csv.gz", index=False)
    panel.to_csv(experiment_dir / "panel_data.csv.gz", index=False)


def create_config(
    n_nodes: int,
    t_steps: int,
    s: int,
    intervention_rule: str,
    outcome_rule: str,
    field_basis_names: tuple[str, ...],
    tau_zero_mean: bool = False,
    tau_smoothness_lambda: float = 0.0,
) -> OmegaConf:
    """Build the minimal config consumed by mple.py for this experiment."""
    return OmegaConf.create(
        {
            "global_params": {
                "N": int(n_nodes),
                "T": int(t_steps),
                "s": int(s),
                "gamma_matrix_generator": "real_data",
                "x_0_generator": "observed",
            },
            "estimation_params": {
                "fit_intervention_model": True,
                "beta": 0.0,
                "eta": 0.0,
                "zeta": 0.0,
                "psi": 0.0,
                "tau_params": None,
                "tau_zero_mean": bool(tau_zero_mean),
                "tau_smoothness_lambda": float(tau_smoothness_lambda),
            },
            "real_data_params": {
                "source": "COVIDSchoolDataHub",
                "state": "Ohio",
                "outcome_column": (
                    "monthly_case_rate_per100k_zip_gt_12p5_pm1"
                    if outcome_rule.startswith("monthly mean")
                    else f"case_rate_per100k_zip_gt_{str(THRESHOLD).replace('.', 'p')}_pm1"
                ),
                "outcome_threshold": THRESHOLD,
                "outcome_rule": outcome_rule,
                "intervention_rule": intervention_rule,
                "network_name": "contiguity",
                "network_source": "ohio_standardized_contiguity",
                "field_basis_names": list(field_basis_names),
            },
        }
    )


def compute_binary_summary(panel: pd.DataFrame, intervention_rule: str, outcome_rule: str) -> pd.DataFrame:
    """Compute a short threshold summary for the final Ohio experiment."""
    panel = panel.sort_values(["NCESDistrictID", "WeekStartDate"]).copy()
    outcome = panel["Outcome_pm1"].eq(1).astype(int)
    intervention = panel["Intervention_pm1"].eq(1).astype(int)
    panel["outcome_good"] = outcome
    panel["intervention_good"] = intervention
    panel["outcome_prev"] = panel.groupby("NCESDistrictID")["outcome_good"].shift(1)
    panel["intervention_prev"] = panel.groupby("NCESDistrictID")["intervention_good"].shift(1)
    outcome_valid = panel["outcome_prev"].notna()
    intervention_valid = panel["intervention_prev"].notna()

    return pd.DataFrame(
        [
            {
                "variable": "outcome",
                "rule": outcome_rule,
                "positive_share": float(outcome.mean()),
                "variance": float(panel["Outcome_pm1"].var(ddof=0)),
                "transition_rate": float(
                    (panel.loc[outcome_valid, "outcome_good"] != panel.loc[outcome_valid, "outcome_prev"]).mean()
                ),
            },
            {
                "variable": "intervention",
                "rule": intervention_rule,
                "positive_share": float(intervention.mean()),
                "variance": float(panel["Intervention_pm1"].var(ddof=0)),
                "transition_rate": float(
                    (panel.loc[intervention_valid, "intervention_good"] != panel.loc[intervention_valid, "intervention_prev"]).mean()
                ),
            },
        ]
    )


def outcome_rule_for_frequency(panel_frequency: str) -> str:
    """Return the outcome threshold rule string for the requested panel frequency."""
    if panel_frequency == "monthly":
        return "monthly mean case_rate_per100k_zip > 12.5"
    return "case_rate_per100k_zip > 12.5"


def lag_application_for_period(lag_period: str | None) -> str:
    """Return the metadata label describing how the lag is applied."""
    return "intervention_only" if lag_period else "none"


def intervention_rule_for_spec(
    intervention_source: str,
    share_threshold: float,
    lag_period: str | None,
) -> str:
    """Return the human-readable intervention rule for one spec."""
    prefix = "lagged " if lag_period else ""
    if intervention_source == "learning_model":
        return prefix + "LearningModel == In-person"
    return prefix + f"share_inperson >= {share_threshold}"


def build_panel_for_spec(
    cached_base_panels: dict[tuple[str, str], pd.DataFrame],
    learning_panel: pd.DataFrame,
    spec: dict[str, object],
) -> pd.DataFrame:
    """Return a binary x/z panel for one requested Ohio experiment spec."""
    panel_frequency = str(spec["panel_frequency"])
    intervention_source = str(spec["intervention_source"])
    share_threshold = float(spec["share_threshold"])
    lag_period = spec.get("lag_period")
    if panel_frequency == "monthly" and intervention_source == "learning_model":
        raise ValueError("Monthly Ohio CSDH experiments currently support only intervention_source=monthly_share.")
    base_key = (panel_frequency, intervention_source)
    if base_key in cached_base_panels:
        base_panel = cached_base_panels[base_key]
        return build_outcome_and_intervention(
            base_panel,
            intervention_source,
            share_threshold,
            lag_period=lag_period,
            panel_frequency=panel_frequency,
        )

    raise ValueError(
        f"Unsupported Ohio experiment spec: panel_frequency={panel_frequency}, "
        f"intervention_source={intervention_source}."
    )


def write_manifest(experiment_root: Path, manifest_rows: list[dict[str, object]]) -> None:
    """Persist a simple manifest for one Ohio batch run."""
    if not manifest_rows:
        return
    manifest_path = experiment_root / "manifest.csv"
    new_manifest = pd.DataFrame(manifest_rows)
    if manifest_path.exists():
        existing_manifest = pd.read_csv(manifest_path)
        if "experiment_name" in existing_manifest.columns:
            existing_manifest = existing_manifest.loc[
                ~existing_manifest["experiment_name"].isin(new_manifest["experiment_name"])
            ].copy()
        manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True, sort=False)
    else:
        manifest = new_manifest
    manifest.to_csv(manifest_path, index=False)


def experiment_has_panel_artifacts(experiment_dir: Path) -> bool:
    """Return whether the saved Ohio experiment folder has the required panel artifacts."""
    required = [
        experiment_dir / "panel_data.npz",
        experiment_dir / "x_0.npy",
        experiment_dir / "z_0.npy",
        experiment_dir / "gamma_matrix_sparse.npz",
        experiment_dir / "field_basis.npy",
    ]
    return all(path.exists() for path in required)


def experiment_has_fit_outputs(experiment_dir: Path) -> bool:
    """Return whether the Ohio experiment folder already has fitted MPLE outputs."""
    return (experiment_dir / "mple_summary.csv").exists()


def fit_mple(experiment_dir: Path, steps: int, tol: float, seed: int, outcome_only: bool) -> None:
    """Run mple.py on one prepared experiment folder."""
    command = [
        sys.executable,
        str(REPO_ROOT / "mple.py"),
        "--data_folder",
        str(experiment_dir),
        "--steps",
        str(steps),
        "--tol",
        str(tol),
        "--seed",
        str(seed),
    ]
    if outcome_only:
        command.append("--outcome_only")
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Materialize and optionally fit Ohio CSDH MPLE experiments.")
    parser.add_argument(
        "--run_full_grid",
        action="store_true",
        help="Materialize the full cached Ohio experiment grid in one process instead of a single spec.",
    )
    parser.add_argument(
        "--intervention_source",
        choices=["learning_model", "monthly_share"],
        default="monthly_share",
        help="Use either the categorical learning model or the monthly in-person share file for z.",
    )
    parser.add_argument(
        "--panel_frequency",
        choices=["weekly", "monthly"],
        default="monthly",
        help="Aggregate the outcome/intervention panel at the weekly or monthly level.",
    )
    parser.add_argument(
        "--share_threshold",
        type=float,
        default=0.5,
        help="Threshold used when intervention_source=monthly_share.",
    )
    parser.add_argument(
        "--lag_period",
        type=str,
        choices=["2w", "3w", "4w", "1m"],
        default=None,
        help="Optional lag applied to the intervention series only. Weekly panels allow 2w/3w/4w; monthly panels allow 1m.",
    )
    parser.add_argument(
        "--experiment_root",
        type=Path,
        default=Path("experiments/COVIDSchoolDataHub_OH"),
        help="Root directory where the experiment folder will be written.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment subfolder name. If omitted, a name is derived from the intervention source.",
    )
    parser.add_argument(
        "--max_experiments",
        type=int,
        default=None,
        help="Optional cap on the number of specs written when --run_full_grid is enabled.",
    )
    parser.add_argument("--steps", type=int, default=1500, help="Maximum L-BFGS iterations.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Optimizer tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--run_mple",
        action="store_true",
        help="Run mple.py after writing each Ohio experiment folder.",
    )
    parser.add_argument(
        "--tau_zero_mean",
        action="store_true",
        help="Constrain the temporal tau block to have zero mean during optimization.",
    )
    parser.add_argument(
        "--tau_smoothness_lambda",
        type=float,
        default=0.0,
        help="Quadratic first-difference penalty weight for the tau time series.",
    )
    parser.add_argument(
        "--outcome_only",
        action="store_true",
        help="Fit only the outcome model and skip the intervention process likelihood.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the experiment folder before rebuilding it.",
    )
    parser.add_argument(
        "--skip_fit",
        action="store_true",
        help="Deprecated legacy alias; the runner now materializes only unless --run_mple is passed.",
    )
    return parser.parse_args()


def materialize_ohio_experiment(
    *,
    spec: dict[str, object],
    args: argparse.Namespace,
    experiment_root: Path,
    learning_panel: pd.DataFrame,
    cached_base_panels: dict[tuple[str, str], pd.DataFrame],
    node_table: pd.DataFrame,
    node_order: list[str],
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    gamma_matrix: sparse.csr_matrix,
    adjacency_edges: pd.DataFrame,
    custom_experiment_name: str | None = None,
) -> dict[str, object]:
    """Write and optionally fit one Ohio CSDH experiment using cached inputs."""
    panel_frequency = str(spec["panel_frequency"])
    intervention_source = str(spec["intervention_source"])
    share_threshold = float(spec["share_threshold"])
    lag_period = spec.get("lag_period")
    outcome_rule = outcome_rule_for_frequency(panel_frequency)
    lag_application = lag_application_for_period(lag_period)
    intervention_rule = intervention_rule_for_spec(intervention_source, share_threshold, lag_period)
    panel = build_panel_for_spec(cached_base_panels, learning_panel, spec)
    x, z, x_0, z_0, times, s = build_panel_arrays(panel, node_order)

    experiment_name = custom_experiment_name or make_experiment_name_for_spec(spec)
    experiment_dir = experiment_root / experiment_name
    if experiment_dir.exists() and args.overwrite:
        shutil.rmtree(experiment_dir)

    config = create_config(
        n_nodes=x.shape[1],
        t_steps=x.shape[0],
        s=s,
        intervention_rule=intervention_rule,
        outcome_rule=outcome_rule,
        field_basis_names=field_basis_names,
        tau_zero_mean=args.tau_zero_mean,
        tau_smoothness_lambda=args.tau_smoothness_lambda,
    )
    metadata = {
        "source": "COVIDSchoolDataHub",
        "state": "Ohio",
        "has_truth": False,
        "x_sign_convention": "+1_above_threshold_-1_below_threshold",
        "z_sign_convention": "+1_in_person_or_high_share_-1_not_in_person_or_low_share",
        "lag_application": lag_application,
        "panel_frequency": panel_frequency,
        "outcome_rule": outcome_rule,
        "intervention_source": intervention_source,
        "intervention_rule": intervention_rule,
        "lag_period": lag_period,
        "lag_steps": resolve_lag_steps(panel_frequency, lag_period),
        "outcome_column": (
            "monthly_case_rate_per100k_zip_gt_12p5_pm1"
            if panel_frequency == "monthly"
            else "case_rate_per100k_zip_gt_12p5_pm1"
        ),
        "fit_intervention_model": bool(not args.outcome_only),
        "tau_zero_mean": bool(args.tau_zero_mean),
        "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
        "network_name": "contiguity",
        "network_source": "ohio_standardized_contiguity",
        "field_basis_names": list(field_basis_names),
        "node_count": int(x.shape[1]),
        "time_steps": int(x.shape[0]),
        "pre_intervention_steps": int(s),
        "threshold": THRESHOLD,
    }

    if experiment_dir.exists() and not args.overwrite:
        if not experiment_has_panel_artifacts(experiment_dir):
            raise FileExistsError(
                f"{experiment_dir} exists but is missing required panel artifacts. "
                "Re-run with --overwrite to rebuild it cleanly."
            )
    else:
        save_experiment(
            experiment_dir=experiment_dir,
            config=config,
            metadata=metadata,
            panel=panel,
            node_table=node_table,
            field_basis=field_basis,
            field_basis_names=field_basis_names,
            gamma_matrix=gamma_matrix,
            adjacency_edges=adjacency_edges,
            x=x,
            z=z,
            x_0=x_0,
            z_0=z_0,
            times=times,
        )

        binary_summary = compute_binary_summary(panel, intervention_rule, outcome_rule)
        binary_summary.to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
        binary_lookup = binary_summary.set_index("variable")
        (experiment_dir / "binary_definition_summary.md").write_text(
            "# Ohio CSDH Binary Experiment Summary\n\n"
            f"- Outcome rule: `{outcome_rule}`\n"
            f"- Outcome positive share: `{binary_lookup.loc['outcome', 'positive_share']:.6f}`\n"
            f"- Outcome transition rate: `{binary_lookup.loc['outcome', 'transition_rate']:.6f}`\n"
            f"- Intervention rule: `{intervention_rule}`\n"
            f"- Intervention positive share: `{binary_lookup.loc['intervention', 'positive_share']:.6f}`\n"
            f"- Intervention transition rate: `{binary_lookup.loc['intervention', 'transition_rate']:.6f}`\n"
            f"- Lag application: `{lag_application}`\n"
            f"- Lag period: `{lag_period or 'none'}`\n"
            "- Network: Ohio standardized contiguity adjacency\n",
            encoding="utf-8",
        )

    binary_summary = compute_binary_summary(panel, intervention_rule, outcome_rule)
    if args.run_mple and not args.skip_fit and (args.overwrite or not experiment_has_fit_outputs(experiment_dir)):
        fit_mple(
            experiment_dir=experiment_dir,
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            outcome_only=args.outcome_only,
        )

    summary_lookup = binary_summary.set_index("variable")
    return {
        "experiment_name": experiment_name,
        "panel_frequency": panel_frequency,
        "intervention_source": intervention_source,
        "share_threshold": share_threshold,
        "lag_period": lag_period or "",
        "lag_steps": resolve_lag_steps(panel_frequency, lag_period),
        "node_count": int(x.shape[1]),
        "time_steps": int(x.shape[0]),
        "pre_intervention_steps": int(s),
        "fit_intervention_model": bool(not args.outcome_only),
        "tau_zero_mean": bool(args.tau_zero_mean),
        "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
        "outcome_positive_share": float(summary_lookup.loc["outcome", "positive_share"]),
        "outcome_transition_rate": float(summary_lookup.loc["outcome", "transition_rate"]),
        "intervention_positive_share": float(summary_lookup.loc["intervention", "positive_share"]),
        "intervention_transition_rate": float(summary_lookup.loc["intervention", "transition_rate"]),
    }


def main() -> None:
    """Build one or many Ohio experiments and optionally fit MPLE."""
    args = parse_args()
    if args.run_full_grid and args.experiment_name is not None:
        raise ValueError("--experiment_name can only be used for a single-spec run.")

    learning_panel, monthly_share_weekly, features, monthly_shares, centroids, adjacency_edges = load_inputs()
    cached_base_panels = build_cached_base_panels(learning_panel, monthly_share_weekly, monthly_shares)
    node_table = build_node_table(centroids, features)
    node_order = node_table["NCESDistrictID"].tolist()
    field_basis, field_basis_names = build_field_basis(node_table)
    gamma_matrix, adjacency_edges = build_contiguity_matrix(adjacency_edges, node_order)
    validate_basis_infinity_norms(field_basis, gamma_matrix)

    experiment_root = (REPO_ROOT / args.experiment_root).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    if args.run_full_grid:
        specs = default_grid_specs()
        if args.max_experiments is not None:
            specs = specs[: args.max_experiments]
    else:
        specs = [
            {
                "panel_frequency": args.panel_frequency,
                "intervention_source": args.intervention_source,
                "share_threshold": args.share_threshold,
                "lag_period": args.lag_period,
            }
        ]

    manifest_rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        custom_name = args.experiment_name if index == 0 and not args.run_full_grid else None
        manifest_rows.append(
            materialize_ohio_experiment(
                spec=spec,
                args=args,
                experiment_root=experiment_root,
                learning_panel=learning_panel,
                cached_base_panels=cached_base_panels,
                node_table=node_table,
                node_order=node_order,
                field_basis=field_basis,
                field_basis_names=field_basis_names,
                gamma_matrix=gamma_matrix,
                adjacency_edges=adjacency_edges,
                custom_experiment_name=custom_name,
            )
        )

    write_manifest(experiment_root, manifest_rows)


if __name__ == "__main__":
    main()
