"""Helpers for USCountyVaccination realized artifacts and experiment folders."""

from __future__ import annotations

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
    CORE_END_DATE,
    CORE_START_DATE,
    DEFAULT_BOOSTER_INTERVENTIONS,
    DEFAULT_BOOSTER_LAGS,
    DEFAULT_CORE_INTERVENTIONS,
    DEFAULT_CORE_LAGS,
    INTERVENTION_SPECS,
    OUTCOME_SPECS,
    PROCESSED_DIR,
    SOURCE_LABEL,
    STATE_SCOPE_LABEL,
    experiment_name,
    lag_code_to_steps,
)
from .data_utils import normalize_sparse_matrix_infinity  # noqa: E402
from utils.model_utils import ModelArtifacts, save_model_artifacts  # noqa: E402


NON_MAINLAND_STATEFPS = frozenset({"02", "15", "60", "66", "69", "72", "78"})
TRIM_RULE_LABEL = "mainland_us_and_total_population_ge_2000"
TRIMMED_STATE_SCOPE_LABEL = "Mainland US counties with total_population >= 2000"


def build_sparse_network_from_edges(
    edges: pd.DataFrame,
    node_order: list[str],
    source_column: str,
    target_column: str,
    weight_column: str | None = None,
) -> sparse.csr_matrix:
    """Build a normalized sparse graph matrix for realized network artifacts."""
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary_panel_path = PROCESSED_DIR / "us_county_binary_panel.csv.gz"
    node_geography_path = PROCESSED_DIR / "us_county_node_geography.csv.gz"
    centroid_path = PROCESSED_DIR / "us_county_centroids.csv"
    required = [binary_panel_path, node_geography_path, centroid_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed US county files: "
            + ", ".join(missing)
            + ". Run preprocess_us_county_vaccination_data.py first."
        )
    panel = pd.read_csv(binary_panel_path, dtype={"fips": str}, parse_dates=["WeekStartDate", "WeekEndDate"])
    node_geography = pd.read_csv(node_geography_path, dtype={"fips": str})
    centroids = pd.read_csv(centroid_path, dtype={"fips": str})
    return panel, node_geography, centroids


def build_node_table(node_geography: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    node_table = node_geography.merge(
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


def build_experiment_grid(args) -> list[dict[str, str]]:
    requested_outcomes = tuple(args.outcomes) if args.outcomes else ("death_rate_100k_ge_2",)
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
    outcome_code: str,
    intervention_code: str,
    lag_code: str,
    network_name: str,
    state_scope_label: str,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "global_params": {
                "N": int(n_nodes),
                "T": int(t_steps),
                "gamma_matrix_generator": "real_data",
                "x_0_generator": "observed",
            },
            "estimation_params": {
                "beta": 0.0,
                "eta": 0.0,
                "tau_params": None,
                "fixed_scalar_params": {},
            },
            "real_data_params": {
                "source": SOURCE_LABEL,
                "state": state_scope_label,
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "lag_application": "intervention_only",
                "network_name": network_name,
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


def load_realized_binary_artifact(
    artifact_dir: Path,
    panel_key: str,
) -> RealizedBinaryArtifact:
    metadata_path = artifact_dir / "panel_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing realized artifact metadata: {metadata_path}")
    metadata = OmegaConf.to_container(OmegaConf.load(metadata_path), resolve=True)
    if not isinstance(metadata, dict):
        metadata = {}
    panel_path = artifact_dir / "panel_data.npz"
    initial_path = artifact_dir / f"{panel_key}_0.npy"
    initial_mask_path = artifact_dir / f"{panel_key}0_observed_mask.npy"
    node_path = artifact_dir / "node_index.csv"
    time_path = artifact_dir / "time_index.csv"
    required = [panel_path, initial_path, initial_mask_path, node_path, time_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing realized binary artifact file(s): " + ", ".join(missing)
        )
    with np.load(panel_path, allow_pickle=False) as payload:
        values = np.asarray(payload[panel_key])
        observed_mask = np.asarray(payload["observed_mask"], dtype=bool)
    node_index = pd.read_csv(node_path, dtype={"fips": str})
    time_index = pd.read_csv(time_path, parse_dates=["WeekStartDate", "WeekEndDate"])
    code = str(
        metadata.get(
            "outcome_code" if panel_key == "x" else "intervention_code",
            artifact_dir.name,
        )
    )
    return RealizedBinaryArtifact(
        code=code,
        panel_key=panel_key,
        values=values,
        initial_values=np.asarray(np.load(initial_path, allow_pickle=False)),
        observed_mask=observed_mask,
        initial_observed_mask=np.asarray(
            np.load(initial_mask_path, allow_pickle=False), dtype=bool
        ),
        node_order=node_index.sort_values("node_index")["fips"].astype(str).tolist(),
        time_index=time_index,
        artifact_dir=artifact_dir,
        metadata=dict(metadata),
    )


def load_realized_network_artifact(artifact_dir: Path) -> RealizedNetworkArtifact:
    metadata_path = artifact_dir / "network_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing realized network metadata: {metadata_path}")
    metadata = OmegaConf.to_container(OmegaConf.load(metadata_path), resolve=True)
    if not isinstance(metadata, dict):
        metadata = {}
    gamma_path = artifact_dir / "gamma_matrix_sparse.npz"
    edge_path = artifact_dir / "adjacency_edge_list.csv.gz"
    node_path = artifact_dir / "node_index.csv"
    required = [gamma_path, edge_path, node_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing realized network artifact file(s): " + ", ".join(missing)
        )
    node_index = pd.read_csv(node_path, dtype={"fips": str})
    return RealizedNetworkArtifact(
        network_name=str(metadata.get("network_name", artifact_dir.name)),
        gamma_matrix=sparse.load_npz(gamma_path).tocsr(),
        adjacency_edges=pd.read_csv(edge_path, dtype={"fips": str, "neighbor_fips": str}),
        node_order=node_index.sort_values("node_index")["fips"].astype(str).tolist(),
        artifact_dir=artifact_dir,
        metadata=dict(metadata),
    )


def load_shared_panel_artifacts(shared_panel_dir: Path) -> dict[str, object]:
    panel_path = shared_panel_dir / "panel_data.npz"
    metadata_path = shared_panel_dir / "panel_metadata.yaml"
    required = [
        panel_path,
        shared_panel_dir / "x_0.npy",
        shared_panel_dir / "z_0.npy",
        shared_panel_dir / "node_index.csv",
        shared_panel_dir / "time_index.csv",
        shared_panel_dir / "panel_data.csv.gz",
        metadata_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing shared panel artifact file(s): " + ", ".join(missing)
        )
    with np.load(panel_path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=np.int8)
        z = np.asarray(payload["z"], dtype=np.int8)
    metadata = OmegaConf.to_container(OmegaConf.load(metadata_path), resolve=True)
    if not isinstance(metadata, dict):
        metadata = {}
    node_index = pd.read_csv(shared_panel_dir / "node_index.csv", dtype={"fips": str})
    return {
        "x": x,
        "z": z,
        "x_0": np.asarray(np.load(shared_panel_dir / "x_0.npy", allow_pickle=False), dtype=np.int8),
        "z_0": np.asarray(np.load(shared_panel_dir / "z_0.npy", allow_pickle=False), dtype=np.int8),
        "node_order": node_index.sort_values("node_index")["fips"].astype(str).tolist(),
        "node_index": node_index,
        "time_index": pd.read_csv(
            shared_panel_dir / "time_index.csv",
            parse_dates=["WeekStartDate", "WeekEndDate"],
        ),
        "panel": pd.read_csv(
            shared_panel_dir / "panel_data.csv.gz",
            dtype={"fips": str},
            parse_dates=["WeekStartDate", "WeekEndDate"],
        ),
        "metadata": dict(metadata),
    }


def save_experiment(
    experiment_dir: Path,
    config,
    metadata: dict[str, object],
    gamma_matrix: sparse.csr_matrix,
    adjacency_edges: pd.DataFrame,
    panel: pd.DataFrame,
    node_table: pd.DataFrame,
    time_index: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, experiment_dir / "realized_config.yaml")
    OmegaConf.save(OmegaConf.create(metadata), experiment_dir / "experiment_metadata.yaml")
    np.savez(experiment_dir / "panel_data.npz", x=x, z=z)
    np.save(experiment_dir / "x_0.npy", x_0)
    np.save(experiment_dir / "z_0.npy", z_0)
    node_table.to_csv(experiment_dir / "node_index.csv", index=False)
    time_index.to_csv(experiment_dir / "time_index.csv", index=False)
    panel.to_csv(experiment_dir / "panel_data.csv.gz", index=False)
    artifacts = ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(np.asarray(x).shape[0]),
        latent_rank=0,
        field_matrix=np.zeros_like(np.asarray(x, dtype=float)),
    )
    save_model_artifacts(
        experiment_dir,
        artifacts,
    )
    adjacency_edges.to_csv(experiment_dir / "adjacency_edge_list.csv.gz", index=False)


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
        experiment_dir / "panel_data.npz",
        experiment_dir / "x_0.npy",
        experiment_dir / "z_0.npy",
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

