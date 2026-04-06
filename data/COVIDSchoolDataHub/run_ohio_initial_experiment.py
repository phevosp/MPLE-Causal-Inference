"""Build and fit the first Ohio COVID School Data Hub MPLE experiment.

This script creates one real-data experiment folder with:
- outcome: district-week case rate thresholded at 12.5
- intervention: either learning mode or monthly in-person share
- network: Ohio standardized contiguity adjacency
- field basis: a compact set of Ohio district-level EDGE ACS-ED features

The resulting folder is written in the same format used by ``mple.py`` for
real-data experiments, then fit immediately unless ``--skip_fit`` is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (  # noqa: E402
    build_touching_edge_list,
    center_and_normalize_vector_infinity,
    normalize_sparse_matrix_infinity,
    standardize_id,
)
from model_utils import validate_basis_infinity_norms  # noqa: E402


STATE_ABBREV = "OH"
THRESHOLD = 12.5
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "COVIDSchoolDataHub_OH"
PROCESSED_DIR = REPO_ROOT / "data" / "COVIDSchoolDataHub" / "processed"


def make_experiment_name(intervention_source: str, share_threshold: float) -> str:
    """Build a stable experiment folder name from the intervention source."""
    if intervention_source == "monthly_share":
        suffix = f"inpersonshare{str(share_threshold).replace('.', 'p')}"
    else:
        suffix = "inperson"
    return f"ohio_case_gt_12p5_{suffix}__contiguity"


def make_monthly_experiment_name(intervention_source: str, share_threshold: float) -> str:
    """Build a stable experiment folder name for monthly-aggregated Ohio experiments."""
    if intervention_source == "monthly_share":
        suffix = f"monthlyshare{str(share_threshold).replace('.', 'p')}"
    else:
        suffix = "monthlyinperson"
    return f"ohio_monthly_case_gt_12p5_{suffix}__contiguity"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Load Ohio district-week data, district features, monthly shares, and geometry."""
    panel = pd.read_csv(
        PROCESSED_DIR / "csdh_learning_case_joined_district_week.csv.gz",
        parse_dates=["WeekStartDate", "WeekEndDate", "PeriodStartDate", "PeriodEndDate"],
        low_memory=False,
    )
    panel = panel.loc[panel["StateAbbrev"] == STATE_ABBREV].copy()
    panel["NCESDistrictID"] = standardize_id(panel["NCESDistrictID"], width=7)

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

    geometry = gpd.read_file(PROCESSED_DIR / "ohio_districts_standardized.gpkg", layer="districts")
    geometry["NCESDistrictID"] = standardize_id(geometry["NCESDistrictID"], width=7)
    geometry["state_abbrev"] = standardize_id(geometry["state_abbrev"])
    geometry = geometry.loc[geometry["NCESDistrictID"].notna()].copy()
    geometry = geometry.sort_values("NCESDistrictID").reset_index(drop=True)

    return panel, features, monthly_shares, geometry


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

    def weighted_mean(group: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(group[column], errors="coerce")
        weights = pd.to_numeric(group["tot_zip_week"], errors="coerce").fillna(0.0)
        mask = values.notna() & weights.notna()
        if not mask.any() or float(weights.loc[mask].sum()) <= 1e-12:
            return float(values.mean())
        return float(np.average(values.loc[mask], weights=weights.loc[mask]))

    rows: list[dict[str, object]] = []
    for keys, group in case_rates.groupby(group_cols, sort=True):
        state_abbrev, district_id, week_start, week_end = keys
        rows.append(
            {
                "StateAbbrev": state_abbrev,
                "NCESDistrictID": district_id,
                "WeekStartDate": week_start,
                "WeekEndDate": week_end,
                "DistrictName": group["lea_name"].iloc[0] if "lea_name" in group.columns else group["DistrictName"].iloc[0],
                "StateAssignedDistrictID": group["StateAssignedDistrictID"].iloc[0] if "StateAssignedDistrictID" in group.columns else group["StateAssignedDistrictID"].iloc[0],
                "case_rate_per100k_zip": weighted_mean(group, "case_rate_per100k_zip"),
                "case_rate_per100k_state": weighted_mean(group, "case_rate_per100k_state"),
                "positive_rate": weighted_mean(group, "positive_rate"),
                "total_tests": float(pd.to_numeric(group["total_tests"], errors="coerce").fillna(0.0).sum()),
                "total_positives": float(pd.to_numeric(group["total_positives"], errors="coerce").fillna(0.0).sum()),
                "total_negatives": float(pd.to_numeric(group["total_negatives"], errors="coerce").fillna(0.0).sum()),
                "tot_zip_week": float(pd.to_numeric(group["tot_zip_week"], errors="coerce").fillna(0.0).sum()),
                "zip_count": int(group["zip"].nunique()) if "zip" in group.columns else 0,
            }
        )
    weekly = pd.DataFrame(rows)
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
    group_cols = ["StateAbbrev", "NCESDistrictID", "MonthStartDate", "MonthEndDate"]
    rows: list[dict[str, object]] = []
    for keys, group in monthly.groupby(group_cols, sort=True):
        state_abbrev, district_id, month_start, month_end = keys
        rows.append(
            {
                "StateAbbrev": state_abbrev,
                "NCESDistrictID": district_id,
                "WeekStartDate": month_start,
                "WeekEndDate": month_end,
                "DistrictName": group["DistrictName"].iloc[0],
                "StateAssignedDistrictID": group["StateAssignedDistrictID"].iloc[0],
                "case_rate_per100k_zip": float(pd.to_numeric(group["case_rate_per100k_zip"], errors="coerce").mean()),
                "case_rate_per100k_state": float(pd.to_numeric(group["case_rate_per100k_state"], errors="coerce").mean()),
                "positive_rate": float(pd.to_numeric(group["positive_rate"], errors="coerce").mean()),
                "total_tests": float(pd.to_numeric(group["total_tests"], errors="coerce").fillna(0.0).sum()),
                "total_positives": float(pd.to_numeric(group["total_positives"], errors="coerce").fillna(0.0).sum()),
                "total_negatives": float(pd.to_numeric(group["total_negatives"], errors="coerce").fillna(0.0).sum()),
                "tot_zip_week": float(pd.to_numeric(group["tot_zip_week"], errors="coerce").fillna(0.0).sum()),
                "share_inperson": float(pd.to_numeric(group["share_inperson"], errors="coerce").mean()),
                "share_hybrid": float(pd.to_numeric(group["share_hybrid"], errors="coerce").mean()),
                "share_virtual": float(pd.to_numeric(group["share_virtual"], errors="coerce").mean()),
            }
        )
    monthly_panel = pd.DataFrame(rows)
    monthly_panel["Month"] = monthly_panel["WeekStartDate"].dt.year.astype(str) + "m" + monthly_panel["WeekStartDate"].dt.month.astype(str)
    return monthly_panel.sort_values(["StateAbbrev", "NCESDistrictID", "WeekStartDate"]).reset_index(drop=True)


def build_outcome_and_intervention(
    panel: pd.DataFrame,
    intervention_source: str,
    share_threshold: float,
) -> pd.DataFrame:
    """Create the binary outcome and intervention columns used in the experiment."""
    panel = panel.copy()
    panel["Outcome_pm1"] = np.where(panel["case_rate_per100k_zip"] > THRESHOLD, 1, -1).astype(
        np.int8
    )

    if intervention_source == "learning_model":
        panel["Intervention_pm1"] = np.where(
            panel["LearningModel"].eq("In-person"),
            1,
            -1,
        ).astype(np.int8)
        panel["InterventionShare"] = np.where(panel["LearningModel"].eq("In-person"), 1.0, 0.0)
        panel["intervention_rule"] = "LearningModel == In-person"
    elif intervention_source == "monthly_share":
        if "share_inperson" not in panel.columns:
            raise KeyError("share_inperson is missing from the panel after merging monthly shares.")
        panel["InterventionShare"] = pd.to_numeric(panel["share_inperson"], errors="coerce")
        panel["Intervention_pm1"] = np.where(
            panel["InterventionShare"].fillna(-1) >= share_threshold,
            1,
            -1,
        ).astype(np.int8)
        panel["intervention_rule"] = f"share_inperson >= {share_threshold}"
    else:
        raise ValueError(f"Unknown intervention_source: {intervention_source}")
    return panel


def build_monthly_panel(
    panel: pd.DataFrame,
    intervention_source: str,
    share_threshold: float,
) -> pd.DataFrame:
    """Aggregate the Ohio panel to district-month and build binary x/z there."""
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
    agg["Outcome_pm1"] = np.where(agg["case_rate_per100k_zip"] > THRESHOLD, 1, -1).astype(np.int8)
    if intervention_source == "learning_model":
        agg["Intervention_pm1"] = np.where(agg["LearningModel"].eq("In-person"), 1, -1).astype(np.int8)
        agg["InterventionShare"] = np.where(agg["LearningModel"].eq("In-person"), 1.0, 0.0)
        agg["intervention_rule"] = "LearningModel == In-person"
    elif intervention_source == "monthly_share":
        agg["InterventionShare"] = pd.to_numeric(agg["share_inperson"], errors="coerce")
        agg["Intervention_pm1"] = np.where(
            agg["InterventionShare"].fillna(-1) >= share_threshold,
            1,
            -1,
        ).astype(np.int8)
        agg["intervention_rule"] = f"share_inperson >= {share_threshold}"
    else:
        raise ValueError(f"Unknown intervention_source: {intervention_source}")
    return agg


def build_node_table(
    geometry: gpd.GeoDataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the node ordering and compact district feature table."""
    projected = geometry.to_crs(5070).copy()
    centroids = projected.geometry.centroid
    node_table = projected[["NCESDistrictID", "state_abbrev"]].copy()
    node_table["node_index"] = np.arange(len(node_table), dtype=int)
    node_table["centroid_x"] = centroids.x
    node_table["centroid_y"] = centroids.y
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
        ("intercept", np.ones(len(node_table), dtype=float)),
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
        if name == "intercept":
            basis_vectors.append(np.ones(len(node_table), dtype=float))
            basis_names.append(name)
            continue
        centered = center_and_normalize_vector_infinity(raw)
        if np.linalg.norm(centered, ord=np.inf) < 1e-12:
            continue
        basis_vectors.append(centered)
        basis_names.append(name)

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


def build_contiguity_matrix(geometry: gpd.GeoDataFrame, node_order: list[str]) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build a normalized contiguity adjacency matrix from standardized geometry."""
    edge_list = build_touching_edge_list(
        geometry[["NCESDistrictID", "geometry"]].copy(),
        id_column="NCESDistrictID",
        neighbor_column="neighbor_id",
    )
    node_lookup = {district_id: idx for idx, district_id in enumerate(node_order)}
    rows = edge_list["NCESDistrictID"].map(node_lookup).to_numpy()
    cols = edge_list["neighbor_id"].map(node_lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    data = np.ones(len(rows), dtype=float)
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
                "feature_basis": [
                    "log_total_population",
                    "median_household_income",
                    "pct_below_poverty",
                    "pct_black",
                    "pct_hispanic",
                    "pct_in_labor_force",
                ],
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
    parser = argparse.ArgumentParser(description="Build and fit the first Ohio CSDH MPLE experiment.")
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
    parser.add_argument("--steps", type=int, default=1500, help="Maximum L-BFGS iterations.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Optimizer tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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
        help="Write the experiment folder but do not run mple.py.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the Ohio experiment folder and fit MPLE."""
    args = parse_args()
    learning_panel, features, monthly_shares, geometry = load_inputs()
    if args.intervention_source == "monthly_share":
        case_rates_raw = load_raw_case_rates()
        weekly_panel = aggregate_case_rates_weekly(case_rates_raw)
        weekly_panel = attach_monthly_shares(weekly_panel, monthly_shares)
        if args.panel_frequency == "monthly":
            panel = aggregate_weekly_panel_to_monthly(weekly_panel)
            panel = attach_monthly_shares(panel, monthly_shares)
        else:
            panel = weekly_panel
        panel = build_outcome_and_intervention(panel, args.intervention_source, args.share_threshold)
    else:
        panel = learning_panel.copy()
        if args.panel_frequency == "monthly":
            panel = build_monthly_panel(panel, args.intervention_source, args.share_threshold)
        else:
            panel = build_outcome_and_intervention(panel, args.intervention_source, args.share_threshold)

    node_table = build_node_table(geometry, features)
    node_order = node_table["NCESDistrictID"].tolist()
    field_basis, field_basis_names = build_field_basis(node_table)
    gamma_matrix, adjacency_edges = build_contiguity_matrix(geometry, node_order)
    validate_basis_infinity_norms(field_basis, gamma_matrix)

    x, z, x_0, z_0, times, s = build_panel_arrays(panel, node_order)
    experiment_root = (REPO_ROOT / args.experiment_root).resolve()
    if args.panel_frequency == "monthly":
        experiment_name = args.experiment_name or make_monthly_experiment_name(
            args.intervention_source,
            args.share_threshold,
        )
    else:
        experiment_name = args.experiment_name or make_experiment_name(args.intervention_source, args.share_threshold)
    experiment_dir = experiment_root / experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    if experiment_dir.exists():
        if args.overwrite:
            shutil.rmtree(experiment_dir)
        else:
            raise FileExistsError(
                f"{experiment_dir} already exists. Re-run with --overwrite to rebuild it."
            )

    intervention_rule = (
        "LearningModel == In-person"
        if args.intervention_source == "learning_model"
        else f"share_inperson >= {args.share_threshold}"
    )
    outcome_rule = f"monthly mean case_rate_per100k_zip > {THRESHOLD}" if args.panel_frequency == "monthly" else "case_rate_per100k_zip > 12.5"
    config = create_config(
        n_nodes=x.shape[1],
        t_steps=x.shape[0],
        s=s,
        intervention_rule=intervention_rule,
        outcome_rule=outcome_rule,
    )
    metadata = {
            "source": "COVIDSchoolDataHub",
            "state": "Ohio",
            "has_truth": False,
            "x_sign_convention": "+1_above_threshold_-1_below_threshold",
            "z_sign_convention": "+1_in_person_or_high_share_-1_not_in_person_or_low_share",
        "panel_frequency": args.panel_frequency,
        "outcome_rule": outcome_rule,
        "intervention_source": args.intervention_source,
        "intervention_rule": intervention_rule,
            "outcome_column": (
                "monthly_case_rate_per100k_zip_gt_12p5_pm1"
                if args.panel_frequency == "monthly"
                else "case_rate_per100k_zip_gt_12p5_pm1"
            ),
        "fit_intervention_model": bool(not args.outcome_only),
        "network_name": "contiguity",
        "network_source": "ohio_standardized_contiguity",
        "field_basis_names": list(field_basis_names),
        "node_count": int(x.shape[1]),
        "time_steps": int(x.shape[0]),
        "pre_intervention_steps": int(s),
        "threshold": THRESHOLD,
    }

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

    compute_binary_summary(panel, intervention_rule, outcome_rule).to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
    (experiment_dir / "binary_definition_summary.md").write_text(
        "# Ohio CSDH Binary Experiment Summary\n\n"
        f"- Outcome rule: `{outcome_rule}`\n"
        f"- Intervention rule: `{intervention_rule}`\n"
        "- Network: Ohio standardized contiguity adjacency\n",
        encoding="utf-8",
    )

    if not args.skip_fit:
        fit_mple(
            experiment_dir=experiment_dir,
            steps=args.steps,
            tol=args.tol,
            seed=args.seed,
            outcome_only=args.outcome_only,
        )


if __name__ == "__main__":
    main()
