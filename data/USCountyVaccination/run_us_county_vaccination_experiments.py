"""Materialize and optionally fit nationwide US county vaccination MPLE experiments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BOOSTER_START_DATE,
    CORE_END_DATE,
    CORE_START_DATE,
    DEFAULT_BOOSTER_INTERVENTIONS,
    DEFAULT_BOOSTER_LAGS,
    DEFAULT_CORE_INTERVENTIONS,
    DEFAULT_CORE_LAGS,
    DEFAULT_CORE_OUTCOMES,
    EXPERIMENT_ROOT,
    INTERVENTION_SPECS,
    OUTCOME_SPECS,
    PROCESSED_DIR,
    SOURCE_LABEL,
    STATE_SCOPE_LABEL,
    build_sparse_network_from_edges,
    experiment_name,
    lag_code_to_steps,
    sparse_matrix_stats,
)
from data_utils import center_and_normalize_vector_infinity  # noqa: E402
from model_utils import ModelArtifacts, save_model_artifacts, validate_basis_infinity_norms  # noqa: E402


NON_MAINLAND_STATEFPS = frozenset({"02", "15", "60", "66", "69", "72", "78"})
TRIM_RULE_LABEL = "mainland_us_and_total_population_ge_2000"
TRIMMED_STATE_SCOPE_LABEL = "Mainland US counties with total_population >= 2000"


@dataclass(frozen=True)
class RealizedBinaryArtifact:
    code: str
    panel_key: str
    values: np.ndarray
    initial_values: np.ndarray
    observed_mask: np.ndarray
    initial_observed_mask: np.ndarray
    node_order: list[str]
    time_index: pd.DataFrame
    artifact_dir: Path
    metadata: dict[str, object]


@dataclass(frozen=True)
class RealizedNetworkArtifact:
    network_name: str
    gamma_matrix: sparse.csr_matrix
    adjacency_edges: pd.DataFrame
    node_order: list[str]
    artifact_dir: Path
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize and optionally fit nationwide US county vaccination experiments."
    )
    parser.add_argument("--run_mple", action="store_true", help="Run mple.py after writing each experiment folder.")
    parser.add_argument("--overwrite", action="store_true", help="Allow rewriting existing experiment folders.")
    parser.add_argument("--steps", type=int, default=1500, help="Maximum L-BFGS iterations when fitting.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Optimizer tolerance passed to mple.py.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to mple.py.")
    parser.add_argument(
        "--tau_zero_mean",
        action="store_true",
        help="Constrain the fitted tau block to have zero mean.",
    )
    parser.add_argument(
        "--tau_smoothness_lambda",
        type=float,
        default=0.0,
        help="Quadratic penalty weight on first differences of tau to discourage sudden jumps.",
    )
    parser.add_argument(
        "--beta_mask_pre_intervention",
        action="store_true",
        help="Mask the beta*z term on pre-intervention rows (t < s) when fitting MPLE.",
    )
    parser.add_argument(
        "--beta_mask_rescale",
        action="store_true",
        help=(
            "When beta masking is enabled, rescale the masked beta feature by "
            "total_cells / active_cells to preserve comparable magnitude."
        ),
    )
    parser.add_argument("--max_experiments", type=int, default=None, help="Optional cap on the number of experiments.")
    parser.add_argument("--lags", nargs="*", default=None, help="Lag codes to include, for example 0w 1w 2w.")
    parser.add_argument("--outcomes", nargs="*", default=None, help="Outcome codes to include.")
    parser.add_argument("--interventions", nargs="*", default=None, help="Intervention codes to include.")
    parser.add_argument(
        "--networks",
        nargs="*",
        default=["contiguity"],
        choices=["contiguity", "knn_8", "distance_kernel_8"],
        help="Known-network variants to materialize. Defaults to contiguity only.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("experiments/USCountyVaccination_US"),
        help="Root directory where experiment folders will be written.",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Restrict support to mainland US counties with total population at least 2,000.",
    )
    parser.add_argument(
        "--field_mode",
        choices=["additive", "latent_feature_matrix"],
        default="additive",
        help=(
            "Field parameterization for the outcome process. "
            "'additive' uses the county feature basis; 'latent_feature_matrix' fits a low-rank latent field."
        ),
    )
    parser.add_argument(
        "--latent_rank",
        type=int,
        default=10,
        help="Latent rank used when --field_mode latent_feature_matrix.",
    )
    parser.add_argument(
        "--latent_B",
        type=float,
        default=1.0,
        help=(
            "Infinity-norm bound B used to constrain the realized latent field in MPLE "
            "when --field_mode latent_feature_matrix."
        ),
    )
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary_panel_path = PROCESSED_DIR / "us_county_binary_panel.csv.gz"
    feature_path = PROCESSED_DIR / "us_county_feature_basis.csv.gz"
    centroid_path = PROCESSED_DIR / "us_county_centroids.csv"
    required = [binary_panel_path, feature_path, centroid_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed US county files: "
            + ", ".join(missing)
            + ". Run prepare_us_county_vaccination_data.py and build_binary_outcomes.py first."
        )
    panel = pd.read_csv(binary_panel_path, dtype={"fips": str}, parse_dates=["WeekStartDate", "WeekEndDate"])
    features = pd.read_csv(feature_path, dtype={"fips": str})
    centroids = pd.read_csv(centroid_path, dtype={"fips": str})
    return panel, features, centroids


def build_node_table(features: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    node_table = features.merge(
        centroids,
        on=["fips", "county", "state_name"],
        how="left",
    ).sort_values("fips").reset_index(drop=True)
    node_table["node_index"] = np.arange(len(node_table), dtype=int)
    return node_table


def apply_optional_trim(
    node_table: pd.DataFrame,
    panel: pd.DataFrame,
    trim_requested: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pre_trim_node_count = int(len(node_table))
    if not trim_requested:
        return (
            node_table.copy(),
            panel.copy(),
            {
                "trim_applied": False,
                "trim_rule": "none",
                "trim_scope_label": STATE_SCOPE_LABEL,
                "pre_trim_node_count": pre_trim_node_count,
                "trimmed_node_count": pre_trim_node_count,
                "trim_excluded_node_count": 0,
                "trim_excluded_non_mainland_count": 0,
                "trim_excluded_population_below_2000_count": 0,
                "trim_excluded_missing_population_count": 0,
                "trim_population_min": None,
            },
        )

    statefp = pd.Series(node_table["STATEFP"], copy=False).astype("string").str.zfill(2)
    total_population = pd.to_numeric(node_table["total_population"], errors="coerce")
    mainland_mask = ~statefp.isin(NON_MAINLAND_STATEFPS)
    population_present_mask = total_population.notna()
    population_threshold_mask = total_population.ge(2000.0).fillna(False)
    keep_mask = mainland_mask & population_threshold_mask

    trimmed_node_table = node_table.loc[keep_mask].copy().sort_values("fips").reset_index(drop=True)
    if trimmed_node_table.empty:
        raise ValueError("The requested mainland/population trim removed every county.")

    keep_fips = set(trimmed_node_table["fips"])
    trimmed_panel = panel.loc[panel["fips"].isin(keep_fips)].copy()
    return (
        trimmed_node_table,
        trimmed_panel,
        {
            "trim_applied": True,
            "trim_rule": TRIM_RULE_LABEL,
            "trim_scope_label": TRIMMED_STATE_SCOPE_LABEL,
            "pre_trim_node_count": pre_trim_node_count,
            "trimmed_node_count": int(len(trimmed_node_table)),
            "trim_excluded_node_count": int((~keep_mask).sum()),
            "trim_excluded_non_mainland_count": int((~mainland_mask).sum()),
            "trim_excluded_population_below_2000_count": int(
                (mainland_mask & population_present_mask & ~population_threshold_mask).sum()
            ),
            "trim_excluded_missing_population_count": int((mainland_mask & ~population_present_mask).sum()),
            "trim_population_min": 2000,
        },
    )


def build_field_basis(node_table: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], str]:
    basis_mode = str(node_table["feature_basis_mode"].iloc[0]) if "feature_basis_mode" in node_table.columns else "unknown"
    if basis_mode == "zero":
        return np.empty((0, len(node_table)), dtype=float), (), basis_mode

    required_columns = ["population_density", "log_population"]
    if basis_mode == "acs_2021":
        required_columns.extend(
            [
                "senior_population",
                "college_education",
                "poverty_rate",
                "median_household_income",
                "cdc_svi_2022_overall",
                "usda_ers_rucc_2023",
            ]
        )
    missing_columns = [column for column in required_columns if column not in node_table.columns]
    if missing_columns:
        raise KeyError(
            "Missing required county feature columns: "
            + ", ".join(missing_columns)
            + ". Re-run prepare_us_county_vaccination_data.py to rebuild us_county_feature_basis.csv.gz."
        )

    feature_specs: list[tuple[str, np.ndarray]] = [
        ("population_density", pd.to_numeric(node_table["population_density"], errors="coerce").to_numpy(dtype=float)),
    ]
    if basis_mode == "acs_2021":
        feature_specs.extend(
            [
                ("senior_population", pd.to_numeric(node_table["senior_population"], errors="coerce").to_numpy(dtype=float)),
                ("college_education", pd.to_numeric(node_table["college_education"], errors="coerce").to_numpy(dtype=float)),
                ("poverty_rate", pd.to_numeric(node_table["poverty_rate"], errors="coerce").to_numpy(dtype=float)),
                ("log_population", pd.to_numeric(node_table["log_population"], errors="coerce").to_numpy(dtype=float)),
                ("median_household_income", pd.to_numeric(node_table["median_household_income"], errors="coerce").to_numpy(dtype=float)),
                ("cdc_svi_2022_overall", pd.to_numeric(node_table["cdc_svi_2022_overall"], errors="coerce").to_numpy(dtype=float)),
                ("usda_ers_rucc_2023", pd.to_numeric(node_table["usda_ers_rucc_2023"], errors="coerce").to_numpy(dtype=float)),
            ]
        )
    else:
        feature_specs.extend(
            [
                ("log_population", pd.to_numeric(node_table["log_population"], errors="coerce").to_numpy(dtype=float)),
            ]
        )

    basis_vectors: list[np.ndarray] = []
    basis_names: list[str] = []
    for name, raw_values in feature_specs:
        vector = center_and_normalize_vector_infinity(raw_values)
        if np.linalg.norm(vector, ord=np.inf) < 1e-12:
            continue
        basis_vectors.append(vector)
        basis_names.append(name)
    if not basis_vectors:
        return np.empty((0, len(node_table)), dtype=float), (), "zero"
    return np.vstack(basis_vectors), tuple(basis_names), basis_mode


def load_network_edge_tables(network_names: list[str]) -> dict[str, pd.DataFrame]:
    """Load the requested county edge tables once so the runner can reuse them across the grid."""
    edge_paths = {
        "contiguity": (PROCESSED_DIR / "us_county_contiguity_adjacency.csv.gz", {"fips": str, "neighbor_fips": str}),
        "knn_8": (PROCESSED_DIR / "us_county_knn_8_adjacency.csv.gz", {"fips": str, "neighbor_fips": str}),
        "distance_kernel_8": (PROCESSED_DIR / "us_county_distance_kernel_8_adjacency.csv.gz", {"fips": str, "neighbor_fips": str}),
    }
    edge_tables: dict[str, pd.DataFrame] = {}
    for network_name in network_names:
        if network_name not in edge_paths:
            raise ValueError(f"Unknown network '{network_name}'.")
        edge_path, dtype_map = edge_paths[network_name]
        edge_tables[network_name] = pd.read_csv(edge_path, dtype=dtype_map)
    return edge_tables


def load_network_matrix(
    network_name: str,
    node_order: list[str],
    edge_tables: dict[str, pd.DataFrame],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    keep = set(node_order)
    if network_name == "contiguity":
        edges = edge_tables["contiguity"]
        edges = edges.loc[edges["fips"].isin(keep) & edges["neighbor_fips"].isin(keep)].copy()
        matrix = build_sparse_network_from_edges(edges, node_order, "fips", "neighbor_fips")
        return matrix, edges
    if network_name == "knn_8":
        edges = edge_tables["knn_8"]
        edges = edges.loc[edges["fips"].isin(keep) & edges["neighbor_fips"].isin(keep)].copy()
        matrix = build_sparse_network_from_edges(edges, node_order, "fips", "neighbor_fips", weight_column="weight")
        return matrix, edges
    if network_name == "distance_kernel_8":
        edges = edge_tables["distance_kernel_8"]
        edges = edges.loc[edges["fips"].isin(keep) & edges["neighbor_fips"].isin(keep)].copy()
        matrix = build_sparse_network_from_edges(edges, node_order, "fips", "neighbor_fips", weight_column="weight")
        return matrix, edges
    raise ValueError(f"Unknown network '{network_name}'.")


def build_experiment_grid(args: argparse.Namespace) -> list[dict[str, str]]:
    requested_outcomes = tuple(args.outcomes) if args.outcomes else DEFAULT_CORE_OUTCOMES
    requested_interventions = tuple(args.interventions) if args.interventions else (DEFAULT_CORE_INTERVENTIONS + DEFAULT_BOOSTER_INTERVENTIONS)
    requested_lags = tuple(args.lags) if args.lags else ()

    for outcome_code in requested_outcomes:
        if outcome_code not in OUTCOME_SPECS:
            raise ValueError(f"Unknown outcome code '{outcome_code}'.")
    for intervention_code in requested_interventions:
        if intervention_code not in INTERVENTION_SPECS:
            raise ValueError(f"Unknown intervention code '{intervention_code}'.")

    grid: list[dict[str, str]] = []
    for intervention_code in requested_interventions:
        spec = INTERVENTION_SPECS[intervention_code]
        allowed_lags = DEFAULT_BOOSTER_LAGS if spec.family == "booster" else DEFAULT_CORE_LAGS
        if requested_lags:
            selected_lags = tuple(lag for lag in requested_lags if lag in allowed_lags)
            invalid = set(requested_lags) - set(allowed_lags)
            if invalid:
                raise ValueError(
                    f"Intervention '{intervention_code}' does not support lag codes: {', '.join(sorted(invalid))}."
                )
        else:
            selected_lags = allowed_lags
        for outcome_code in requested_outcomes:
            for lag_code in selected_lags:
                for network_name in args.networks:
                    grid.append(
                        {
                            "outcome_code": outcome_code,
                            "intervention_code": intervention_code,
                            "lag_code": lag_code,
                            "network_name": network_name,
                        }
                    )
    return grid


def requested_window_for_intervention(intervention_code: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return CORE_START_DATE, CORE_END_DATE


def realized_outcome_name(outcome_code: str, trim_applied: bool) -> str:
    trim_label = "trimmed" if trim_applied else "full"
    return f"outcome_{outcome_code}__scope_{trim_label}"


def realized_intervention_name(
    intervention_code: str, lag_code: str, trim_applied: bool
) -> str:
    trim_label = "trimmed" if trim_applied else "full"
    return f"intervention_{intervention_code}__lag_{lag_code}__scope_{trim_label}"


def realized_network_name(network_name: str, trim_applied: bool) -> str:
    trim_label = "trimmed" if trim_applied else "full"
    return f"network_{network_name}__scope_{trim_label}"


def canonical_time_index(panel: pd.DataFrame) -> pd.DataFrame:
    time_index = (
        panel.loc[panel["WeekEndDate"].between(CORE_START_DATE, CORE_END_DATE)][
            ["WeekStartDate", "WeekEndDate", "iso_year", "iso_week"]
        ]
        .drop_duplicates()
        .sort_values("WeekEndDate")
        .reset_index(drop=True)
    )
    time_index["model_index"] = np.arange(len(time_index), dtype=int)
    return time_index


def _panel_grid(
    panel: pd.DataFrame,
    node_order: list[str],
    time_index: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    pivot = (
        panel.pivot(index="WeekEndDate", columns="fips", values=value_column)
        .reindex(index=time_index["WeekEndDate"], columns=node_order)
        .sort_index()
    )
    observed_mask = pivot.notna().to_numpy(dtype=bool)
    values = pivot.to_numpy(dtype=float)
    return values, observed_mask


def write_realized_binary_artifact(
    artifact_dir: Path,
    artifact: RealizedBinaryArtifact,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact_dir / "panel_data.npz",
        **{
            artifact.panel_key: np.asarray(artifact.values, dtype=np.float32),
            "observed_mask": np.asarray(artifact.observed_mask, dtype=bool),
        },
    )
    np.save(
        artifact_dir / f"{artifact.panel_key}_0.npy",
        np.asarray(artifact.initial_values, dtype=np.float32),
    )
    np.save(
        artifact_dir / f"{artifact.panel_key}0_observed_mask.npy",
        np.asarray(artifact.initial_observed_mask, dtype=bool),
    )
    node_frame = pd.DataFrame({"fips": artifact.node_order, "node_index": np.arange(len(artifact.node_order), dtype=int)})
    node_frame.to_csv(artifact_dir / "node_index.csv", index=False)
    artifact.time_index.to_csv(artifact_dir / "time_index.csv", index=False)
    OmegaConf.save(OmegaConf.create(artifact.metadata), artifact_dir / "panel_metadata.yaml")


def write_realized_network_artifact(
    artifact_dir: Path,
    artifact: RealizedNetworkArtifact,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(artifact_dir / "gamma_matrix_sparse.npz", artifact.gamma_matrix)
    artifact.adjacency_edges.to_csv(artifact_dir / "adjacency_edge_list.csv.gz", index=False)
    pd.DataFrame(
        {"fips": artifact.node_order, "node_index": np.arange(len(artifact.node_order), dtype=int)}
    ).to_csv(artifact_dir / "node_index.csv", index=False)
    OmegaConf.save(
        OmegaConf.create(artifact.metadata),
        artifact_dir / "network_metadata.yaml",
    )


def build_realized_outcome_artifact(
    panel: pd.DataFrame,
    node_order: list[str],
    time_index: pd.DataFrame,
    outcome_code: str,
    trim_applied: bool,
    artifact_dir: Path,
) -> RealizedBinaryArtifact:
    x_col = f"x_{outcome_code}_pm1"
    filtered = panel.loc[
        panel["WeekEndDate"].between(CORE_START_DATE, CORE_END_DATE)
        & panel["fips"].isin(node_order),
        ["fips", "WeekEndDate", x_col],
    ].copy()
    values_all, observed_all = _panel_grid(filtered, node_order, time_index, x_col)
    if values_all.shape[0] < 2:
        raise ValueError(f"Outcome artifact for {outcome_code} does not contain at least two weeks.")
    metadata = {
        "artifact_type": "realized_outcome",
        "source": SOURCE_LABEL,
        "outcome_code": outcome_code,
        "outcome_label": OUTCOME_SPECS[outcome_code].label,
        "trim_applied": bool(trim_applied),
        "requested_core_start_date": CORE_START_DATE.date().isoformat(),
        "requested_core_end_date": CORE_END_DATE.date().isoformat(),
        "node_count": int(len(node_order)),
        "calendar_weeks": int(len(time_index)),
    }
    return RealizedBinaryArtifact(
        code=outcome_code,
        panel_key="x",
        values=values_all[1:],
        initial_values=values_all[0],
        observed_mask=observed_all[1:],
        initial_observed_mask=observed_all[0],
        node_order=list(node_order),
        time_index=time_index.copy(),
        artifact_dir=artifact_dir,
        metadata=metadata,
    )


def build_realized_intervention_artifact(
    panel: pd.DataFrame,
    node_order: list[str],
    time_index: pd.DataFrame,
    intervention_code: str,
    lag_code: str,
    trim_applied: bool,
    artifact_dir: Path,
) -> RealizedBinaryArtifact:
    z_col = f"z_{intervention_code}_pm1"
    lag_steps = lag_code_to_steps(lag_code)
    filtered = panel.loc[
        panel["WeekEndDate"].between(CORE_START_DATE, CORE_END_DATE)
        & panel["fips"].isin(node_order),
        ["fips", "WeekEndDate", z_col],
    ].copy()
    filtered = filtered.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    filtered["Intervention_pm1_raw"] = filtered[z_col].astype("Int64")
    filtered["Intervention_pm1"] = (
        filtered.groupby("fips", sort=False)["Intervention_pm1_raw"]
        .shift(lag_steps)
        .astype("Int64")
    )
    leading_lag_mask = filtered.groupby("fips", sort=False).cumcount() < lag_steps
    filtered.loc[
        leading_lag_mask & filtered["Intervention_pm1"].isna(),
        "Intervention_pm1",
    ] = -1
    values_all, observed_all = _panel_grid(
        filtered, node_order, time_index, "Intervention_pm1"
    )
    if values_all.shape[0] < 2:
        raise ValueError(
            f"Intervention artifact for {intervention_code} at lag {lag_code} does not contain at least two weeks."
        )
    metadata = {
        "artifact_type": "realized_intervention",
        "source": SOURCE_LABEL,
        "intervention_code": intervention_code,
        "intervention_label": INTERVENTION_SPECS[intervention_code].label,
        "intervention_family": INTERVENTION_SPECS[intervention_code].family,
        "lag_code": lag_code,
        "lag_steps": lag_steps,
        "trim_applied": bool(trim_applied),
        "requested_core_start_date": CORE_START_DATE.date().isoformat(),
        "requested_core_end_date": CORE_END_DATE.date().isoformat(),
        "node_count": int(len(node_order)),
        "calendar_weeks": int(len(time_index)),
    }
    return RealizedBinaryArtifact(
        code=intervention_code,
        panel_key="z",
        values=values_all[1:],
        initial_values=values_all[0],
        observed_mask=observed_all[1:],
        initial_observed_mask=observed_all[0],
        node_order=list(node_order),
        time_index=time_index.copy(),
        artifact_dir=artifact_dir,
        metadata=metadata,
    )


def build_realized_network_artifact(
    network_name: str,
    node_order: list[str],
    gamma_matrix: sparse.csr_matrix,
    adjacency_edges: pd.DataFrame,
    trim_applied: bool,
    artifact_dir: Path,
) -> RealizedNetworkArtifact:
    metadata = {
        "artifact_type": "realized_network",
        "source": SOURCE_LABEL,
        "network_name": network_name,
        "trim_applied": bool(trim_applied),
        "node_count": int(len(node_order)),
        **sparse_matrix_stats(gamma_matrix),
    }
    return RealizedNetworkArtifact(
        network_name=network_name,
        gamma_matrix=gamma_matrix,
        adjacency_edges=adjacency_edges.copy(),
        node_order=list(node_order),
        artifact_dir=artifact_dir,
        metadata=metadata,
    )


def reconstruct_full_binary_panel(
    artifact: RealizedBinaryArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([artifact.initial_values[None, :], artifact.values])
    observed = np.vstack(
        [artifact.initial_observed_mask[None, :], artifact.observed_mask]
    )
    return values, observed


def select_dense_suffix_support(
    outcome_artifact: RealizedBinaryArtifact,
    intervention_artifact: RealizedBinaryArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, object]]:
    if outcome_artifact.node_order != intervention_artifact.node_order:
        raise ValueError("Outcome and intervention artifacts do not share the same node ordering.")
    if not outcome_artifact.time_index["WeekEndDate"].equals(
        intervention_artifact.time_index["WeekEndDate"]
    ):
        raise ValueError("Outcome and intervention artifacts do not share the same weekly index.")

    x_all, x_obs = reconstruct_full_binary_panel(outcome_artifact)
    z_all, z_obs = reconstruct_full_binary_panel(intervention_artifact)
    eligibility_matrix = x_obs & z_obs
    best_support: tuple[int, int, int, int, np.ndarray] | None = None
    for start_index in range(eligibility_matrix.shape[0]):
        suffix = eligibility_matrix[start_index:, :]
        complete_nodes = suffix.all(axis=0)
        node_count = int(complete_nodes.sum())
        if node_count == 0:
            continue
        week_count = int(suffix.shape[0])
        area = int(node_count * week_count)
        candidate = (area, node_count, week_count, -start_index, complete_nodes)
        if best_support is None or candidate[:4] > best_support[:4]:
            best_support = candidate
    if best_support is None:
        raise ValueError("No dense realized panel remains after combining outcome and intervention artifacts.")

    _, realized_node_count, realized_week_count, neg_start_index, complete_nodes = best_support
    realized_start_index = -neg_start_index
    node_indices = np.flatnonzero(complete_nodes)
    realized_node_order = [outcome_artifact.node_order[idx] for idx in node_indices]
    realized_time_index = (
        outcome_artifact.time_index.iloc[realized_start_index:]
        .reset_index(drop=True)
        .copy()
    )
    realized_time_index["model_index"] = np.arange(len(realized_time_index), dtype=int)
    x_realized = x_all[realized_start_index:, :][:, node_indices]
    z_realized = z_all[realized_start_index:, :][:, node_indices]
    x = x_realized[1:].astype(np.int8)
    z = z_realized[1:].astype(np.int8)
    x_0 = x_realized[0].astype(np.int8)
    z_0 = z_realized[0].astype(np.int8)
    support_metadata = {
        "requested_node_count": int(len(outcome_artifact.node_order)),
        "requested_calendar_weeks": int(len(outcome_artifact.time_index)),
        "requested_start_date": CORE_START_DATE.date().isoformat(),
        "requested_end_date": CORE_END_DATE.date().isoformat(),
        "support_selection_rule": "max_complete_suffix_by_node_week_area",
        "realized_node_count": int(realized_node_count),
        "realized_calendar_weeks": int(realized_week_count),
        "weeks_dropped_due_to_missing_or_lag": int(len(outcome_artifact.time_index) - realized_week_count),
        "dropped_node_count": int(len(outcome_artifact.node_order) - realized_node_count),
    }
    return x, z, x_0, z_0, realized_time_index, realized_node_order, support_metadata


def assembled_panel_from_arrays(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    time_index: pd.DataFrame,
    node_order: list[str],
    outcome_code: str,
    intervention_code: str,
) -> pd.DataFrame:
    x_all = np.vstack([x_0[None, :], x])
    z_all = np.vstack([z_0[None, :], z])
    week_count = x_all.shape[0]
    node_count = len(node_order)
    return pd.DataFrame(
        {
            "WeekStartDate": np.repeat(time_index["WeekStartDate"].to_numpy(), node_count),
            "WeekEndDate": np.repeat(time_index["WeekEndDate"].to_numpy(), node_count),
            "iso_year": np.repeat(time_index["iso_year"].to_numpy(), node_count),
            "iso_week": np.repeat(time_index["iso_week"].to_numpy(), node_count),
            "fips": np.tile(np.asarray(node_order, dtype=object), week_count),
            "Outcome_pm1": x_all.reshape(-1).astype(np.int8),
            "Intervention_pm1": z_all.reshape(-1).astype(np.int8),
            "outcome_code": outcome_code,
            "intervention_code": intervention_code,
            "outcome_label": OUTCOME_SPECS[outcome_code].label,
            "intervention_label": INTERVENTION_SPECS[intervention_code].label,
        }
    )


def subset_network_artifact(
    artifact: RealizedNetworkArtifact, realized_node_order: list[str]
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    lookup = {node_id: idx for idx, node_id in enumerate(artifact.node_order)}
    indices = np.asarray([lookup[node_id] for node_id in realized_node_order], dtype=int)
    gamma_matrix = artifact.gamma_matrix[indices][:, indices].tocsr()
    edge_columns = list(artifact.adjacency_edges.columns)
    source_column = edge_columns[0]
    target_column = edge_columns[1]
    keep = set(realized_node_order)
    adjacency_edges = artifact.adjacency_edges.loc[
        artifact.adjacency_edges[source_column].isin(keep)
        & artifact.adjacency_edges[target_column].isin(keep)
    ].copy()
    return gamma_matrix, adjacency_edges


def prepare_panel_for_experiment(
    panel: pd.DataFrame,
    node_order: list[str],
    outcome_code: str,
    intervention_code: str,
    lag_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    x_col = f"x_{outcome_code}_pm1"
    z_col = f"z_{intervention_code}_pm1"
    lag_steps = lag_code_to_steps(lag_code)
    requested_start, requested_end = requested_window_for_intervention(intervention_code)

    filtered = panel.loc[
        panel["WeekEndDate"].between(requested_start, requested_end) & panel["fips"].isin(node_order)
    ].copy()

    filtered = filtered.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    filtered["Outcome_pm1"] = filtered[x_col].astype("Int64")
    filtered["Intervention_pm1_raw"] = filtered[z_col].astype("Int64")
    filtered["Intervention_pm1"] = (
        filtered.groupby("fips", sort=False)["Intervention_pm1_raw"].shift(lag_steps).astype("Int64")
    )
    leading_lag_mask = filtered.groupby("fips", sort=False).cumcount() < lag_steps
    filtered.loc[leading_lag_mask & filtered["Intervention_pm1"].isna(), "Intervention_pm1"] = -1

    eligible = filtered["Outcome_pm1"].notna() & filtered["Intervention_pm1"].notna()
    eligibility_matrix = (
        filtered.assign(eligible=eligible)
        .pivot(index="WeekEndDate", columns="fips", values="eligible")
        .reindex(columns=node_order)
        .sort_index()
    )
    if eligibility_matrix.empty:
        raise ValueError(
            f"No county-week rows remain for outcome={outcome_code}, intervention={intervention_code}, lag={lag_code}."
        )

    best_support: tuple[int, int, int, int, pd.Series] | None = None
    for start_index in range(len(eligibility_matrix.index)):
        suffix = eligibility_matrix.iloc[start_index:]
        complete_nodes = suffix.all(axis=0)
        node_count = int(complete_nodes.sum())
        if node_count == 0:
            continue
        week_count = int(suffix.shape[0])
        area = int(node_count * week_count)
        candidate = (area, node_count, week_count, -start_index, complete_nodes)
        if best_support is None or candidate[:4] > best_support[:4]:
            best_support = candidate

    if best_support is None:
        raise ValueError(
            f"No dense realized panel remains for outcome={outcome_code}, intervention={intervention_code}, lag={lag_code}."
        )

    _, realized_node_count, realized_week_count, neg_start_index, complete_nodes = best_support
    realized_start_index = -neg_start_index
    realized_week_dates = eligibility_matrix.index[realized_start_index:]
    realized_node_order = [node_id for node_id in node_order if bool(complete_nodes.get(node_id, False))]

    filtered = filtered.loc[
        filtered["WeekEndDate"].isin(realized_week_dates) & filtered["fips"].isin(realized_node_order)
    ].copy()
    filtered = filtered.sort_values(["WeekEndDate", "fips"]).reset_index(drop=True)

    week_counts = filtered.groupby("WeekEndDate")["fips"].nunique()
    if not week_counts.eq(realized_node_count).all():
        raise ValueError("Filtered experiment panel does not contain every realized county at every retained week.")

    time_index = (
        filtered[["WeekStartDate", "WeekEndDate", "iso_year", "iso_week"]]
        .drop_duplicates()
        .sort_values("WeekEndDate")
        .reset_index(drop=True)
    )
    time_index["model_index"] = np.arange(len(time_index), dtype=int)

    support_metadata = {
        "requested_node_count": int(len(node_order)),
        "requested_calendar_weeks": int(eligibility_matrix.shape[0]),
        "requested_start_date": requested_start.date().isoformat(),
        "requested_end_date": requested_end.date().isoformat(),
        "support_selection_rule": "max_complete_suffix_by_node_week_area",
        "realized_node_count": int(realized_node_count),
        "realized_calendar_weeks": int(realized_week_count),
        "weeks_dropped_due_to_missing_or_lag": int(eligibility_matrix.shape[0] - realized_week_count),
        "dropped_node_count": int(len(node_order) - realized_node_count),
    }
    return filtered, time_index, realized_node_order, support_metadata


def build_panel_arrays(panel: pd.DataFrame, node_order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    ordered = panel.sort_values(["WeekEndDate", "fips"]).reset_index(drop=True)
    x_pivot = ordered.pivot(index="WeekEndDate", columns="fips", values="Outcome_pm1").reindex(columns=node_order).sort_index()
    z_pivot = ordered.pivot(index="WeekEndDate", columns="fips", values="Intervention_pm1").reindex(columns=node_order).sort_index()
    if x_pivot.isna().any().any():
        raise ValueError("Outcome panel contains missing values after experiment alignment.")
    if z_pivot.isna().any().any():
        raise ValueError("Intervention panel contains missing values after experiment alignment.")

    x_all = x_pivot.to_numpy(dtype=np.int8)
    z_all = z_pivot.to_numpy(dtype=np.int8)
    x_0 = x_all[0].astype(np.int8)
    z_0 = z_all[0].astype(np.int8)
    x = x_all[1:].astype(np.int8)
    z = z_all[1:].astype(np.int8)
    treated_rows = np.any(z == 1, axis=1)
    s = int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])
    return x, z, x_0, z_0, s


def compute_binary_summary(panel: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for variable, column, rule in [
        ("outcome", "Outcome_pm1", panel["outcome_label"].iloc[0]),
        ("intervention", "Intervention_pm1", panel["intervention_label"].iloc[0]),
    ]:
        ordered = panel.sort_values(["fips", "WeekEndDate"]).copy()
        ordered["prev"] = ordered.groupby("fips", sort=False)[column].shift(1)
        valid = ordered[column].notna() & ordered["prev"].notna()
        summary_rows.append(
            {
                "variable": variable,
                "rule": rule,
                "positive_share": float(ordered[column].eq(1).mean()),
                "variance": float(ordered[column].var(ddof=0)),
                "transition_rate": float((ordered.loc[valid, column] != ordered.loc[valid, "prev"]).mean()) if valid.any() else float("nan"),
                "weeks": int(ordered["WeekEndDate"].nunique()),
                "counties": int(ordered["fips"].nunique()),
            }
        )
    return pd.DataFrame(summary_rows)


def create_config(
    n_nodes: int,
    t_steps: int,
    s: int,
    outcome_code: str,
    intervention_code: str,
    lag_code: str,
    network_name: str,
    field_basis_mode: str,
    field_basis_names: tuple[str, ...],
    model_field_mode: str,
    latent_rank: int,
    latent_B: float,
    state_scope_label: str,
    tau_zero_mean: bool,
    tau_smoothness_lambda: float,
    beta_mask_pre_intervention: bool,
    beta_mask_rescale: bool,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "global_params": {
                "N": int(n_nodes),
                "T": int(t_steps),
                "s": int(s),
                "B": float(latent_B),
                "basis_params": {
                    "field_mode": str(model_field_mode),
                    "latent_rank": int(latent_rank),
                },
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
                "beta_mask_pre_intervention": bool(beta_mask_pre_intervention),
                "beta_mask_rescale": bool(beta_mask_rescale),
            },
            "real_data_params": {
                "source": SOURCE_LABEL,
                "state": state_scope_label,
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "lag_application": "intervention_only",
                "network_name": network_name,
                "field_basis_mode": field_basis_mode,
                "field_basis_names": list(field_basis_names),
                "model_field_mode": model_field_mode,
                "latent_rank": int(latent_rank),
                "latent_B": float(latent_B),
            },
        }
    )


def shared_panel_name(
    outcome_code: str,
    intervention_code: str,
    lag_code: str,
    trim_applied: bool,
) -> str:
    trim_label = "trimmed" if trim_applied else "full"
    return (
        f"outcome_{outcome_code}__intervention_{intervention_code}"
        f"__lag_{lag_code}__scope_{trim_label}"
    )


def write_shared_panel_artifacts(
    shared_panel_dir: Path,
    panel: pd.DataFrame,
    node_table: pd.DataFrame,
    time_index: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    metadata: dict[str, object],
) -> None:
    shared_panel_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shared_panel_dir / "panel_data.npz", x=x, z=z)
    np.save(shared_panel_dir / "x_0.npy", x_0)
    np.save(shared_panel_dir / "z_0.npy", z_0)
    node_table.to_csv(shared_panel_dir / "node_index.csv", index=False)
    time_index.to_csv(shared_panel_dir / "time_index.csv", index=False)
    panel.to_csv(shared_panel_dir / "panel_data.csv.gz", index=False)
    OmegaConf.save(OmegaConf.create(metadata), shared_panel_dir / "panel_metadata.yaml")


def save_experiment(
    experiment_dir: Path,
    config,
    metadata: dict[str, object],
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    model_field_mode: str,
    latent_rank: int,
    gamma_matrix: sparse.csr_matrix,
    adjacency_edges: pd.DataFrame,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, experiment_dir / "realized_config.yaml")
    OmegaConf.save(OmegaConf.create(metadata), experiment_dir / "experiment_metadata.yaml")
    if model_field_mode == "latent_feature_matrix":
        artifacts = ModelArtifacts(
            field_mode="latent_feature_matrix",
            gamma_matrix=gamma_matrix,
            latent_rank=int(latent_rank),
        )
    else:
        field_mode = "uniform" if field_basis.shape[0] == 0 else "shared_feature_field"
        artifacts = ModelArtifacts(
            field_mode=field_mode,
            gamma_matrix=gamma_matrix,
            field_basis=field_basis,
            field_names=field_basis_names,
        )
    save_model_artifacts(
        experiment_dir,
        artifacts,
    )
    adjacency_edges.to_csv(experiment_dir / "adjacency_edge_list.csv.gz", index=False)


def run_mple(
    experiment_dir: Path,
    panel_path: Path,
    x0_path: Path,
    z0_path: Path,
    steps: int,
    tol: float,
    seed: int,
    outcome_only: bool = False,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "mple.py"),
        "--data_folder",
        str(experiment_dir),
        "--panel_path",
        str(panel_path),
        "--x0_path",
        str(x0_path),
        "--z0_path",
        str(z0_path),
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


def experiment_has_panel_artifacts(experiment_dir: Path) -> bool:
    metadata_path = experiment_dir / "experiment_metadata.yaml"
    if not metadata_path.exists():
        return False
    metadata = OmegaConf.to_container(OmegaConf.load(metadata_path), resolve=True)
    if not isinstance(metadata, dict):
        return False
    panel_path = Path(str(metadata.get("shared_panel_path", "")))
    x0_path = Path(str(metadata.get("shared_x0_path", "")))
    z0_path = Path(str(metadata.get("shared_z0_path", "")))
    required = [
        panel_path,
        x0_path,
        z0_path,
        experiment_dir / "gamma_matrix_sparse.npz",
        experiment_dir / "field_artifacts.npz",
    ]
    return all(path.exists() for path in required)


def existing_experiment_trim_setting(experiment_dir: Path) -> bool | None:
    metadata_path = experiment_dir / "experiment_metadata.yaml"
    if not metadata_path.exists():
        return None
    metadata = OmegaConf.to_container(OmegaConf.load(metadata_path), resolve=True)
    if not isinstance(metadata, dict):
        return None
    if "trim_applied" not in metadata:
        return False
    return bool(metadata["trim_applied"])


def main() -> None:
    args = parse_args()
    if args.field_mode == "latent_feature_matrix":
        if args.latent_rank <= 0:
            raise ValueError("--latent_rank must be positive when --field_mode latent_feature_matrix.")
        if args.latent_B <= 0.0:
            raise ValueError("--latent_B must be positive when --field_mode latent_feature_matrix.")
    panel, features, centroids = load_inputs()
    full_node_table = build_node_table(features, centroids)
    untrimmed_node_order = full_node_table["fips"].tolist()
    if set(untrimmed_node_order) != set(centroids["fips"]):
        raise ValueError("Feature basis and centroid county coverage do not match.")
    full_node_table, panel, trim_metadata = apply_optional_trim(full_node_table, panel, args.trim)
    full_node_order = full_node_table["fips"].tolist()
    state_scope_label = str(trim_metadata["trim_scope_label"])
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    realized_outcome_root = output_root / "realized_outcomes"
    realized_intervention_root = output_root / "realized_interventions"
    realized_network_root = output_root / "realized_networks"
    shared_panel_root = output_root / "shared_panels"
    realized_outcome_root.mkdir(parents=True, exist_ok=True)
    realized_intervention_root.mkdir(parents=True, exist_ok=True)
    realized_network_root.mkdir(parents=True, exist_ok=True)
    shared_panel_root.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(args)
    network_edge_tables = load_network_edge_tables(args.networks)
    core_time_index = canonical_time_index(panel)
    requested_outcomes = sorted({item["outcome_code"] for item in grid})
    requested_interventions = sorted(
        {(item["intervention_code"], item["lag_code"]) for item in grid}
    )
    requested_networks = sorted({item["network_name"] for item in grid})

    realized_outcome_artifacts: dict[str, RealizedBinaryArtifact] = {}
    for outcome_code in requested_outcomes:
        artifact_dir = realized_outcome_root / realized_outcome_name(
            outcome_code, bool(trim_metadata["trim_applied"])
        )
        artifact = build_realized_outcome_artifact(
            panel=panel,
            node_order=full_node_order,
            time_index=core_time_index,
            outcome_code=outcome_code,
            trim_applied=bool(trim_metadata["trim_applied"]),
            artifact_dir=artifact_dir,
        )
        if args.overwrite or not (artifact_dir / "panel_data.npz").exists():
            write_realized_binary_artifact(artifact_dir, artifact)
        realized_outcome_artifacts[outcome_code] = artifact

    realized_intervention_artifacts: dict[tuple[str, str], RealizedBinaryArtifact] = {}
    for intervention_code, lag_code in requested_interventions:
        artifact_dir = realized_intervention_root / realized_intervention_name(
            intervention_code,
            lag_code,
            bool(trim_metadata["trim_applied"]),
        )
        artifact = build_realized_intervention_artifact(
            panel=panel,
            node_order=full_node_order,
            time_index=core_time_index,
            intervention_code=intervention_code,
            lag_code=lag_code,
            trim_applied=bool(trim_metadata["trim_applied"]),
            artifact_dir=artifact_dir,
        )
        if args.overwrite or not (artifact_dir / "panel_data.npz").exists():
            write_realized_binary_artifact(artifact_dir, artifact)
        realized_intervention_artifacts[(intervention_code, lag_code)] = artifact

    realized_network_artifacts: dict[str, RealizedNetworkArtifact] = {}
    for network_name in requested_networks:
        gamma_matrix, adjacency_edges = load_network_matrix(
            network_name, full_node_order, network_edge_tables
        )
        artifact_dir = realized_network_root / realized_network_name(
            network_name, bool(trim_metadata["trim_applied"])
        )
        artifact = build_realized_network_artifact(
            network_name=network_name,
            node_order=full_node_order,
            gamma_matrix=gamma_matrix,
            adjacency_edges=adjacency_edges,
            trim_applied=bool(trim_metadata["trim_applied"]),
            artifact_dir=artifact_dir,
        )
        if args.overwrite or not (artifact_dir / "gamma_matrix_sparse.npz").exists():
            write_realized_network_artifact(artifact_dir, artifact)
        realized_network_artifacts[network_name] = artifact

    manifest_rows: list[dict[str, object]] = []
    experiment_counter = 0

    for item in grid:
        outcome_code = item["outcome_code"]
        intervention_code = item["intervention_code"]
        lag_code = item["lag_code"]
        network_name = item["network_name"]
        experiment_dir = output_root / experiment_name(outcome_code, intervention_code, lag_code, network_name)

        if experiment_dir.exists() and args.overwrite:
            shutil.rmtree(experiment_dir)

        outcome_artifact = realized_outcome_artifacts[outcome_code]
        intervention_artifact = realized_intervention_artifacts[
            (intervention_code, lag_code)
        ]
        x, z, x_0, z_0, time_index, realized_node_order, support_metadata = select_dense_suffix_support(
            outcome_artifact,
            intervention_artifact,
        )
        aligned_panel = assembled_panel_from_arrays(
            x=x,
            z=z,
            x_0=x_0,
            z_0=z_0,
            time_index=time_index,
            node_order=realized_node_order,
            outcome_code=outcome_code,
            intervention_code=intervention_code,
        )

        node_table = (
            full_node_table.loc[full_node_table["fips"].isin(realized_node_order)]
            .sort_values("fips")
            .reset_index(drop=True)
        )
        excluded_node_table = (
            full_node_table.loc[~full_node_table["fips"].isin(realized_node_order)]
            .sort_values("fips")
            .reset_index(drop=True)
        )
        if args.field_mode == "latent_feature_matrix":
            field_basis = np.empty((0, len(node_table)), dtype=float)
            field_basis_names = ()
            field_basis_mode = "latent_feature_matrix"
            model_field_mode = "latent_feature_matrix"
        else:
            field_basis, field_basis_names, field_basis_mode = build_field_basis(node_table)
            model_field_mode = "shared_feature_field" if field_basis.shape[0] > 0 else "uniform"
        gamma_matrix, adjacency_edges = subset_network_artifact(
            realized_network_artifacts[network_name], realized_node_order
        )
        validate_basis_infinity_norms(
            None if model_field_mode == "latent_feature_matrix" else field_basis,
            gamma_matrix,
        )
        treated_rows = np.any(z == 1, axis=1)
        s = int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])
        shared_panel_dir = shared_panel_root / shared_panel_name(
            outcome_code=outcome_code,
            intervention_code=intervention_code,
            lag_code=lag_code,
            trim_applied=bool(trim_metadata["trim_applied"]),
        )
        shared_panel_path = shared_panel_dir / "panel_data.npz"
        shared_x0_path = shared_panel_dir / "x_0.npy"
        shared_z0_path = shared_panel_dir / "z_0.npy"
        stats = sparse_matrix_stats(gamma_matrix)
        config = create_config(
            n_nodes=len(realized_node_order),
            t_steps=x.shape[0],
            s=s,
            outcome_code=outcome_code,
            intervention_code=intervention_code,
            lag_code=lag_code,
            network_name=network_name,
            field_basis_mode=field_basis_mode,
            field_basis_names=field_basis_names,
            model_field_mode=model_field_mode,
            latent_rank=int(args.latent_rank),
            latent_B=float(args.latent_B),
            state_scope_label=state_scope_label,
            tau_zero_mean=args.tau_zero_mean,
            tau_smoothness_lambda=args.tau_smoothness_lambda,
            beta_mask_pre_intervention=bool(args.beta_mask_pre_intervention),
            beta_mask_rescale=bool(args.beta_mask_rescale),
        )
        metadata = {
            "source": SOURCE_LABEL,
            "state": state_scope_label,
            "has_truth": False,
            "x_sign_convention": "+1_above_threshold_-1_below_threshold",
            "z_sign_convention": "+1_above_threshold_-1_below_threshold",
            "lag_application": "intervention_only",
            "outcome_code": outcome_code,
            "outcome_label": OUTCOME_SPECS[outcome_code].label,
            "outcome_notes": OUTCOME_SPECS[outcome_code].notes,
            "intervention_code": intervention_code,
            "intervention_label": INTERVENTION_SPECS[intervention_code].label,
            "intervention_notes": INTERVENTION_SPECS[intervention_code].notes,
            "intervention_family": INTERVENTION_SPECS[intervention_code].family,
            "lag_code": lag_code,
            "lag_steps": lag_code_to_steps(lag_code),
            "network_name": network_name,
            "field_basis_mode": field_basis_mode,
            "field_basis_names": list(field_basis_names),
            "model_field_mode": model_field_mode,
            "latent_rank": int(args.latent_rank) if model_field_mode == "latent_feature_matrix" else None,
            "latent_B": float(args.latent_B) if model_field_mode == "latent_feature_matrix" else None,
            "trim_applied": bool(trim_metadata["trim_applied"]),
            "trim_rule": trim_metadata["trim_rule"],
            "pre_trim_node_count": int(trim_metadata["pre_trim_node_count"]),
            "trimmed_node_count": int(trim_metadata["trimmed_node_count"]),
            "trim_excluded_node_count": int(trim_metadata["trim_excluded_node_count"]),
            "trim_excluded_non_mainland_count": int(trim_metadata["trim_excluded_non_mainland_count"]),
            "trim_excluded_population_below_2000_count": int(trim_metadata["trim_excluded_population_below_2000_count"]),
            "trim_excluded_missing_population_count": int(trim_metadata["trim_excluded_missing_population_count"]),
            "trim_population_min": trim_metadata["trim_population_min"],
            "tau_zero_mean": bool(args.tau_zero_mean),
            "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
            "beta_mask_pre_intervention": bool(args.beta_mask_pre_intervention),
            "beta_mask_rescale": bool(args.beta_mask_rescale),
            "requested_node_count": int(support_metadata["requested_node_count"]),
            "node_count": int(len(realized_node_order)),
            "dropped_node_count": int(support_metadata["dropped_node_count"]),
            "time_steps": int(x.shape[0]),
            "pre_intervention_steps": int(s),
            "requested_core_start_date": support_metadata["requested_start_date"],
            "requested_core_end_date": support_metadata["requested_end_date"],
            "realized_week_start_date": time_index["WeekStartDate"].min().date().isoformat(),
            "realized_week_end_date": time_index["WeekEndDate"].max().date().isoformat(),
            "realized_calendar_weeks": int(len(time_index)),
            "requested_calendar_weeks": int(support_metadata["requested_calendar_weeks"]),
            "weeks_dropped_due_to_missing_or_lag": int(support_metadata["weeks_dropped_due_to_missing_or_lag"]),
            "support_selection_rule": support_metadata["support_selection_rule"],
            "shared_panel_dir": str(shared_panel_dir),
            "shared_panel_path": str(shared_panel_path),
            "shared_x0_path": str(shared_x0_path),
            "shared_z0_path": str(shared_z0_path),
            "shared_node_index_path": str(shared_panel_dir / "node_index.csv"),
            "shared_time_index_path": str(shared_panel_dir / "time_index.csv"),
            "realized_outcome_dir": str(outcome_artifact.artifact_dir),
            "realized_intervention_dir": str(intervention_artifact.artifact_dir),
            "realized_network_dir": str(realized_network_artifacts[network_name].artifact_dir),
            **stats,
        }

        if experiment_dir.exists() and not args.overwrite:
            existing_trim_applied = existing_experiment_trim_setting(experiment_dir)
            if existing_trim_applied is not None and existing_trim_applied != bool(args.trim):
                raise FileExistsError(
                    f"{experiment_dir} exists with trim_applied={existing_trim_applied}. "
                    "Use --overwrite or choose a different --output_root for the other sample scope."
                )
            if not experiment_has_panel_artifacts(experiment_dir):
                raise FileExistsError(
                    f"{experiment_dir} exists but is missing panel artifacts. Re-run with --overwrite to rebuild it."
                )
        else:
            shared_panel_metadata = {
                key: metadata[key]
                for key in [
                    "source",
                    "state",
                    "outcome_code",
                    "outcome_label",
                    "intervention_code",
                    "intervention_label",
                    "intervention_family",
                    "lag_code",
                    "lag_steps",
                    "trim_applied",
                    "trim_rule",
                    "requested_node_count",
                    "node_count",
                    "dropped_node_count",
                    "time_steps",
                    "pre_intervention_steps",
                    "requested_core_start_date",
                    "requested_core_end_date",
                    "realized_week_start_date",
                    "realized_week_end_date",
                    "realized_calendar_weeks",
                    "requested_calendar_weeks",
                    "weeks_dropped_due_to_missing_or_lag",
                    "support_selection_rule",
                ]
                if key in metadata
            }
            if args.overwrite or not shared_panel_path.exists():
                write_shared_panel_artifacts(
                    shared_panel_dir=shared_panel_dir,
                    panel=aligned_panel,
                    node_table=node_table,
                    time_index=time_index,
                    x=x,
                    z=z,
                    x_0=x_0,
                    z_0=z_0,
                    metadata=shared_panel_metadata,
                )
            save_experiment(
                experiment_dir,
                config,
                metadata,
                field_basis,
                field_basis_names,
                model_field_mode,
                int(args.latent_rank),
                gamma_matrix,
                adjacency_edges,
            )
            binary_summary = compute_binary_summary(aligned_panel)
            binary_summary.to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
            binary_lookup = binary_summary.set_index("variable")
            excluded_node_table.to_csv(experiment_dir / "excluded_node_index.csv", index=False)
            (experiment_dir / "binary_definition_summary.md").write_text(
                "# US County Vaccination Binary Experiment Summary\n\n"
                f"- Outcome: `{OUTCOME_SPECS[outcome_code].label}`\n"
                f"- Outcome positive share: `{binary_lookup.loc['outcome', 'positive_share']:.6f}`\n"
                f"- Outcome variance: `{binary_lookup.loc['outcome', 'variance']:.6f}`\n"
                f"- Outcome transition rate: `{binary_lookup.loc['outcome', 'transition_rate']:.6f}`\n"
                f"- Intervention: `{INTERVENTION_SPECS[intervention_code].label}`\n"
                f"- Intervention positive share: `{binary_lookup.loc['intervention', 'positive_share']:.6f}`\n"
                f"- Intervention variance: `{binary_lookup.loc['intervention', 'variance']:.6f}`\n"
                f"- Intervention transition rate: `{binary_lookup.loc['intervention', 'transition_rate']:.6f}`\n"
                f"- Lag: `{lag_code}` applied to the intervention only\n"
                f"- Network: `{network_name}`\n"
                f"- Trim: `{trim_metadata['trim_rule']}`\n"
                f"- Requested counties: `{support_metadata['requested_node_count']}`\n"
                f"- Realized counties: `{len(realized_node_order)}`\n"
                f"- Realized weeks: `{time_index['WeekEndDate'].min().date()}` through `{time_index['WeekEndDate'].max().date()}`\n",
                encoding="utf-8",
            )

        existing_summary = (experiment_dir / "mple_summary.csv").exists()
        full_fit_status = "completed_existing" if existing_summary and not args.run_mple else "not_run"
        outcome_only_fit_status = "not_run"
        fallback_run = False
        if args.run_mple:
            try:
                run_mple(
                    experiment_dir,
                    panel_path=shared_panel_path,
                    x0_path=shared_x0_path,
                    z0_path=shared_z0_path,
                    steps=args.steps,
                    tol=args.tol,
                    seed=args.seed,
                    outcome_only=False,
                )
                full_fit_status = "completed"
            except subprocess.CalledProcessError:
                full_fit_status = "failed"
                try:
                    run_mple(
                        experiment_dir,
                        panel_path=shared_panel_path,
                        x0_path=shared_x0_path,
                        z0_path=shared_z0_path,
                        steps=args.steps,
                        tol=args.tol,
                        seed=args.seed,
                        outcome_only=True,
                    )
                    outcome_only_fit_status = "completed"
                    fallback_run = True
                except subprocess.CalledProcessError:
                    outcome_only_fit_status = "failed"

        manifest_rows.append(
            {
                "experiment_name": experiment_dir.name,
                "path": str(experiment_dir),
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "network_name": network_name,
                "intervention_family": INTERVENTION_SPECS[intervention_code].family,
                "field_basis_mode": field_basis_mode,
                "model_field_mode": model_field_mode,
                "latent_rank": int(args.latent_rank) if model_field_mode == "latent_feature_matrix" else None,
                "latent_B": float(args.latent_B) if model_field_mode == "latent_feature_matrix" else None,
                "trim_applied": bool(trim_metadata["trim_applied"]),
                "trim_rule": trim_metadata["trim_rule"],
                "pre_trim_node_count": int(trim_metadata["pre_trim_node_count"]),
                "trimmed_node_count": int(trim_metadata["trimmed_node_count"]),
                "trim_excluded_node_count": int(trim_metadata["trim_excluded_node_count"]),
                "requested_node_count": int(support_metadata["requested_node_count"]),
                "node_count": int(len(realized_node_order)),
                "dropped_node_count": int(support_metadata["dropped_node_count"]),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
                "realized_start_date": time_index["WeekEndDate"].min().date().isoformat(),
                "realized_end_date": time_index["WeekEndDate"].max().date().isoformat(),
                "requested_calendar_weeks": int(support_metadata["requested_calendar_weeks"]),
                "realized_calendar_weeks": int(support_metadata["realized_calendar_weeks"]),
                "weeks_dropped_due_to_missing_or_lag": int(support_metadata["weeks_dropped_due_to_missing_or_lag"]),
                "support_selection_rule": support_metadata["support_selection_rule"],
                "shared_panel_dir": str(shared_panel_dir),
                "full_fit_status": full_fit_status,
                "outcome_only_fit_status": outcome_only_fit_status,
                "fallback_run": fallback_run,
                **stats,
            }
        )

        experiment_counter += 1
        if args.max_experiments is not None and experiment_counter >= args.max_experiments:
            break

    if manifest_rows:
        manifest_path = output_root / "manifest.csv"
        new_manifest = pd.DataFrame(manifest_rows)
        if manifest_path.exists() and not args.overwrite:
            existing_manifest = pd.read_csv(manifest_path)
            if "experiment_name" in existing_manifest.columns:
                existing_manifest = existing_manifest.loc[
                    ~existing_manifest["experiment_name"].isin(new_manifest["experiment_name"])
                ].copy()
            combined_manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True, sort=False)
        else:
            combined_manifest = new_manifest
        combined_manifest.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()
