"""Materialize and optionally fit nationwide US county vaccination MPLE experiments."""

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
from model_utils import validate_basis_infinity_norms  # noqa: E402


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


def build_field_basis(node_table: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], str]:
    basis_mode = str(node_table["feature_basis_mode"].iloc[0]) if "feature_basis_mode" in node_table.columns else "unknown"
    if basis_mode == "zero":
        return np.empty((0, len(node_table)), dtype=float), (), basis_mode

    feature_specs: list[tuple[str, np.ndarray]] = [
        ("log_population", pd.to_numeric(node_table["log_population"], errors="coerce").to_numpy(dtype=float)),
    ]
    if basis_mode == "acs_2021":
        feature_specs.extend(
            [
                ("median_household_income", pd.to_numeric(node_table["median_household_income"], errors="coerce").to_numpy(dtype=float)),
                ("poverty_rate", pd.to_numeric(node_table["poverty_rate"], errors="coerce").to_numpy(dtype=float)),
                ("pct_black", pd.to_numeric(node_table["pct_black"], errors="coerce").to_numpy(dtype=float)),
                ("pct_hispanic", pd.to_numeric(node_table["pct_hispanic"], errors="coerce").to_numpy(dtype=float)),
                ("pct_in_labor_force", pd.to_numeric(node_table["pct_in_labor_force"], errors="coerce").to_numpy(dtype=float)),
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
    if INTERVENTION_SPECS[intervention_code].family == "booster":
        return BOOSTER_START_DATE, CORE_END_DATE
    return CORE_START_DATE, CORE_END_DATE


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

    filtered = panel.loc[panel["WeekEndDate"].between(requested_start, requested_end)].copy()

    filtered = filtered.sort_values(["fips", "WeekEndDate"]).reset_index(drop=True)
    filtered["Outcome_pm1"] = filtered[x_col].astype("Int64")
    filtered["Intervention_pm1_raw"] = filtered[z_col].astype("Int64")
    filtered["Intervention_pm1"] = (
        filtered.groupby("fips", sort=False)["Intervention_pm1_raw"].shift(lag_steps).astype("Int64")
    )

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
    tau_zero_mean: bool,
    tau_smoothness_lambda: float,
) -> OmegaConf:
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
                "source": SOURCE_LABEL,
                "state": STATE_SCOPE_LABEL,
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "lag_application": "intervention_only",
                "network_name": network_name,
                "field_basis_mode": field_basis_mode,
                "field_basis_names": list(field_basis_names),
            },
        }
    )


def save_experiment(
    experiment_dir: Path,
    config,
    metadata: dict[str, object],
    panel: pd.DataFrame,
    node_table: pd.DataFrame,
    time_index: pd.DataFrame,
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    gamma_matrix: sparse.csr_matrix,
    interaction_name: str,
    adjacency_edges: pd.DataFrame,
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
    np.save(experiment_dir / "field_basis.npy", field_basis)
    np.save(experiment_dir / "field_basis_names.npy", np.asarray(field_basis_names, dtype="<U128"))
    np.save(experiment_dir / "shared_features.npy", np.empty((0, field_basis.shape[1]), dtype=float))
    np.save(experiment_dir / "shared_feature_names.npy", np.asarray([], dtype="<U1"))
    sparse.save_npz(experiment_dir / "gamma_matrix_sparse.npz", gamma_matrix)
    sparse.save_npz(experiment_dir / "interaction_basis_sparse.npz", gamma_matrix)
    np.save(experiment_dir / "interaction_basis_names.npy", np.asarray([interaction_name], dtype="<U128"))
    node_table.to_csv(experiment_dir / "node_index.csv", index=False)
    time_index.to_csv(experiment_dir / "time_index.csv", index=False)
    adjacency_edges.to_csv(experiment_dir / "adjacency_edge_list.csv.gz", index=False)
    panel.to_csv(experiment_dir / "panel_data.csv.gz", index=False)


def run_mple(experiment_dir: Path, steps: int, tol: float, seed: int, outcome_only: bool = False) -> None:
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


def experiment_has_panel_artifacts(experiment_dir: Path) -> bool:
    required = [
        experiment_dir / "panel_data.npz",
        experiment_dir / "x_0.npy",
        experiment_dir / "z_0.npy",
        experiment_dir / "gamma_matrix_sparse.npz",
        experiment_dir / "field_basis.npy",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    panel, features, centroids = load_inputs()
    full_node_table = build_node_table(features, centroids)
    full_node_order = full_node_table["fips"].tolist()
    if set(full_node_order) != set(centroids["fips"]):
        raise ValueError("Feature basis and centroid county coverage do not match.")
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(args)
    network_edge_tables = load_network_edge_tables(args.networks)
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

        aligned_panel, time_index, realized_node_order, support_metadata = prepare_panel_for_experiment(
            panel,
            full_node_order,
            outcome_code,
            intervention_code,
            lag_code,
        )
        aligned_panel["outcome_code"] = outcome_code
        aligned_panel["intervention_code"] = intervention_code
        aligned_panel["outcome_label"] = OUTCOME_SPECS[outcome_code].label
        aligned_panel["intervention_label"] = INTERVENTION_SPECS[intervention_code].label

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
        field_basis, field_basis_names, field_basis_mode = build_field_basis(node_table)
        gamma_matrix, adjacency_edges = load_network_matrix(network_name, realized_node_order, network_edge_tables)
        validate_basis_infinity_norms(field_basis, gamma_matrix)
        x, z, x_0, z_0, s = build_panel_arrays(aligned_panel, realized_node_order)
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
            tau_zero_mean=args.tau_zero_mean,
            tau_smoothness_lambda=args.tau_smoothness_lambda,
        )
        metadata = {
            "source": SOURCE_LABEL,
            "state": STATE_SCOPE_LABEL,
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
            "tau_zero_mean": bool(args.tau_zero_mean),
            "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
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
            **stats,
        }

        if experiment_dir.exists() and not args.overwrite:
            if not experiment_has_panel_artifacts(experiment_dir):
                raise FileExistsError(
                    f"{experiment_dir} exists but is missing panel artifacts. Re-run with --overwrite to rebuild it."
                )
        else:
            save_experiment(
                experiment_dir,
                config,
                metadata,
                aligned_panel,
                node_table,
                time_index,
                field_basis,
                field_basis_names,
                gamma_matrix,
                network_name,
                adjacency_edges,
                x,
                z,
                x_0,
                z_0,
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
                run_mple(experiment_dir, steps=args.steps, tol=args.tol, seed=args.seed, outcome_only=False)
                full_fit_status = "completed"
            except subprocess.CalledProcessError:
                full_fit_status = "failed"
                try:
                    run_mple(experiment_dir, steps=args.steps, tol=args.tol, seed=args.seed, outcome_only=True)
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
