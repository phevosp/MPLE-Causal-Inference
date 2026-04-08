from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import validate_basis_infinity_norms
from data_utils import (
    normalize_sparse_matrix_infinity,
    center_and_normalize_vector_infinity,
    safe_ratio,
)


DEFAULT_OUTCOMES = (
    "i_drugs_gt_0_pm1",
    "any_crime_gt_0_pm1",
    "any_crime_gt_1_pm1",
    "any_crime_gt_2_pm1",
    "any_crime_gt_3_pm1",
    "any_crime_gt_district_mean_pm1",
    "any_crime_gt_block_mean_pm1",
)

DEFAULT_NETWORKS = (
    "contiguity",
    "knn_8",
    "knn_16",
    "centroid_distance_kernel_8",
    "centroid_distance_kernel_16",
)


def experiment_group(field_basis_mode: str, outcome_only: bool) -> str:
    """Return the SeattleDMI experiment subfolder for the requested fit mode."""
    parts: list[str] = []
    if outcome_only:
        parts.append("outcome_only")
    if field_basis_mode == "zero":
        parts.append("zero_basis")
    if not parts:
        parts.append("static")
    return "_".join(parts)


def outcome_base_name(outcome_column: str) -> str:
    """Map a binary outcome column name back to its underlying crime count variable."""
    if outcome_column.startswith("i_drugs"):
        return "i_drugs"
    if outcome_column.startswith("any_crime"):
        return "any_crime"
    return outcome_column.split("_gt_", 1)[0]


def load_inputs(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed SeattleDMI tables needed for real-data MPLE experiments."""
    binary_outcomes_path = processed_dir / "seattledmi_binary_outcomes.csv.gz"
    block_features_path = processed_dir / "seattledmi_block_features.csv"
    crosswalk_path = processed_dir / "seattledmi_block_crosswalk.csv"
    centroids_path = processed_dir / "seattledmi_block_centroids.csv"
    required = [binary_outcomes_path, block_features_path, crosswalk_path, centroids_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Could not find {', '.join(missing)} in {processed_dir}.")
    binary_outcomes = pd.read_csv(binary_outcomes_path, dtype={"GEOID10": str})
    block_features = pd.read_csv(block_features_path, dtype={"GEOID10": str})
    crosswalk = pd.read_csv(crosswalk_path, dtype={"GEOID10": str})
    centroids = pd.read_csv(centroids_path, dtype={"GEOID10": str})
    return binary_outcomes, block_features, crosswalk, centroids


def build_node_table(
    block_features: pd.DataFrame,
    crosswalk: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Create the canonical node ordering and projected centroid coordinates."""
    node_table = block_features.merge(
        crosswalk[["GEOID10", "NEIGHBORHOOD_DISTRICT_NAME"]],
        on="GEOID10",
        how="left",
        validate="one_to_one",
    ).merge(
        centroids,
        on="GEOID10",
        how="left",
        validate="one_to_one",
    )
    node_table = node_table.sort_values("GEOID10").reset_index(drop=True)
    node_table["node_index"] = np.arange(len(node_table), dtype=int)
    return node_table


def build_panel_arrays(
    binary_outcomes: pd.DataFrame,
    outcome_column: str,
    node_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    """Build x, z, x_0, z_0 arrays in the format expected by mple.py."""
    ordered = binary_outcomes.copy()
    ordered["Intervention_pm1"] = ordered["Intervention"].map({0: -1, 1: 1}).astype(np.int8)
    ordered = ordered.sort_values(["time", "GEOID10"]).reset_index(drop=True)

    outcome_pivot = (
        ordered.pivot(index="time", columns="GEOID10", values=outcome_column)
        .reindex(columns=node_order)
        .sort_index()
    )
    intervention_pivot = (
        ordered.pivot(index="time", columns="GEOID10", values="Intervention_pm1")
        .reindex(columns=node_order)
        .sort_index()
    )
    if outcome_pivot.isna().any().any():
        raise ValueError(f"Outcome panel for '{outcome_column}' has missing values.")
    if intervention_pivot.isna().any().any():
        raise ValueError("Intervention panel has missing values.")

    times = [int(value) for value in outcome_pivot.index.tolist()]
    outcome_array = outcome_pivot.to_numpy(dtype=np.int8)
    intervention_array = intervention_pivot.to_numpy(dtype=np.int8)
    x_0 = outcome_array[0].astype(np.int8)
    z_0 = intervention_array[0].astype(np.int8)
    x = outcome_array[1:].astype(np.int8)
    z = intervention_array[1:].astype(np.int8)

    treated_rows = np.any(z == 1, axis=1)
    s = int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])
    return x, z, x_0, z_0, times, s


def build_saved_panel(binary_outcomes: pd.DataFrame, outcome_column: str) -> pd.DataFrame:
    """Build the human-readable Seattle panel table saved alongside each experiment."""
    panel = binary_outcomes[["GEOID10", "time", outcome_column, "Intervention"]].copy()
    panel["Outcome_pm1"] = panel[outcome_column].astype(np.int8)
    panel["Intervention_pm1"] = panel["Intervention"].map({0: -1, 1: 1}).astype(np.int8)
    return panel.sort_values(["time", "GEOID10"]).reset_index(drop=True)


def compute_binary_summary(binary_outcomes: pd.DataFrame, outcome_column: str) -> pd.DataFrame:
    """Compute binary share and transition diagnostics for one Seattle outcome/intervention pair."""
    ordered = binary_outcomes.sort_values(["GEOID10", "time"]).copy()
    ordered["Outcome_pm1"] = ordered[outcome_column].astype(np.int8)
    ordered["Intervention_pm1"] = ordered["Intervention"].map({0: -1, 1: 1}).astype(np.int8)
    summary_rows: list[dict[str, object]] = []
    for variable, column, rule in [
        ("outcome", "Outcome_pm1", outcome_column),
        ("intervention", "Intervention_pm1", "Intervention == 1"),
    ]:
        ordered["prev"] = ordered.groupby("GEOID10", sort=False)[column].shift(1)
        valid = ordered[column].notna() & ordered["prev"].notna()
        summary_rows.append(
            {
                "variable": variable,
                "rule": rule,
                "positive_share": float(ordered[column].eq(1).mean()),
                "variance": float(ordered[column].var(ddof=0)),
                "transition_rate": float((ordered.loc[valid, column] != ordered.loc[valid, "prev"]).mean()) if valid.any() else float("nan"),
                "time_periods": int(ordered["time"].nunique()),
                "blocks": int(ordered["GEOID10"].nunique()),
            }
        )
    return pd.DataFrame(summary_rows)


def build_network_from_edge_table(
    edges: pd.DataFrame,
    node_lookup: dict[str, int],
    n_nodes: int,
) -> sparse.csr_matrix:
    """Convert one saved Seattle edge list into a normalized sparse matrix."""
    rows = edges["GEOID10"].map(node_lookup).to_numpy()
    cols = edges["neighbor_GEOID10"].map(node_lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    if "weight" in edges.columns:
        data = pd.to_numeric(edges.loc[valid, "weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        data = np.ones(len(rows), dtype=float)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return normalize_sparse_matrix_infinity(matrix)


def build_networks(
    node_table: pd.DataFrame,
    edge_tables: dict[str, pd.DataFrame],
) -> dict[str, sparse.csr_matrix]:
    """Build the requested known-network variants for SeattleDMI from cached edge tables."""
    node_lookup = dict(zip(node_table["GEOID10"], node_table["node_index"]))
    networks: dict[str, sparse.csr_matrix] = {}

    for network_name, edge_table in edge_tables.items():
        network = build_network_from_edge_table(edge_table, node_lookup, len(node_table))
        networks[network_name] = network
    return networks


def load_network_edge_tables(
    processed_dir: Path,
    network_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load the requested Seattle edge lists once so they can be reused across the grid."""
    edge_tables: dict[str, pd.DataFrame] = {}
    edge_paths = {
        "contiguity": processed_dir / "seattledmi_block_adjacency.csv.gz",
        "knn_8": processed_dir / "seattledmi_block_knn_8_adjacency.csv.gz",
        "knn_16": processed_dir / "seattledmi_block_knn_16_adjacency.csv.gz",
        "centroid_distance_kernel_8": processed_dir / "seattledmi_block_distance_kernel_8_adjacency.csv.gz",
        "centroid_distance_kernel_16": processed_dir / "seattledmi_block_distance_kernel_16_adjacency.csv.gz",
    }
    for network_name in network_names:
        edge_path = edge_paths.get(network_name)
        if edge_path is None:
            raise ValueError(f"Unknown network '{network_name}'.")
        edge_tables[network_name] = pd.read_csv(
            edge_path,
            dtype={"GEOID10": str, "neighbor_GEOID10": str},
        )
    return edge_tables


def build_field_basis(
    node_table: pd.DataFrame,
    field_basis_mode: str = "static",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct an infinity-normalized field basis for a SeattleDMI experiment."""
    if field_basis_mode == "zero":
        return np.empty((0, len(node_table)), dtype=float), ()
    elif field_basis_mode == "static":
        candidate_features = [
            (
                "total_pop",
                np.nan_to_num(node_table["TotalPop"].to_numpy(dtype=float), nan=0.0),
            ),
            ("black_share", safe_ratio(node_table["BLACK"], node_table["TotalPop"])),
            ("hispanic_share", safe_ratio(node_table["HISPANIC"], node_table["TotalPop"])),
            (
                "male_1521_share",
                safe_ratio(node_table["Males_1521"], node_table["TotalPop"]),
            ),
            (
                "family_household_share",
                safe_ratio(node_table["FAMILYHOUS"], node_table["HOUSEHOLDS"]),
            ),
            (
                "female_household_share",
                safe_ratio(node_table["FEMALE_HOU"], node_table["HOUSEHOLDS"]),
            ),
            ("renter_share", safe_ratio(node_table["RENTER_HOU"], node_table["HOUSEHOLDS"])),
            ("vacant_share", safe_ratio(node_table["VACANT_HOU"], node_table["HOUSEHOLDS"])),
        ]
    else:
        raise ValueError(f"Unknown field_basis_mode '{field_basis_mode}'.")

    basis_vectors: list[np.ndarray] = []
    basis_names: list[str] = []
    for name, raw_feature in candidate_features:
        normalized = center_and_normalize_vector_infinity(np.asarray(raw_feature, dtype=float))
        if np.linalg.norm(normalized, ord=np.inf) < 1e-12:
            continue
        basis_vectors.append(normalized)
        basis_names.append(name)

    if not basis_vectors:
        return np.empty((0, len(node_table)), dtype=float), ()
    field_basis = np.vstack(basis_vectors)
    return field_basis, tuple(basis_names)


def sparse_matrix_stats(matrix: sparse.csr_matrix) -> dict[str, float | int]:
    """Compute basic diagnostics for one sparse known network."""
    row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    return {
        "nnz": int(matrix.nnz),
        "undirected_edges": int(matrix.nnz // 2),
        "avg_degree": float(row_sums.mean()),
        "max_degree": float(row_sums.max()) if row_sums.size else 0.0,
        "gamma_inf_norm": float(row_sums.max()) if row_sums.size else 0.0,
        "gamma_fro_norm": float(np.sqrt(matrix.multiply(matrix).sum())),
    }


def create_config(
    n_nodes: int,
    t_steps: int,
    s: int,
    field_basis_names: tuple[str, ...],
    field_basis_mode: str,
    network_name: str,
    outcome_column: str,
    tau_zero_mean: bool,
    tau_smoothness_lambda: float,
    fit_intervention_model: bool = True,
) -> OmegaConf:
    """Build the minimal realized config needed by mple.py for a real-data experiment."""
    config = OmegaConf.create(
        {
            "global_params": {
                "N": int(n_nodes),
                "T": int(t_steps),
                "s": int(s),
                "gamma_matrix_generator": "real_data",
                "x_0_generator": "observed",
            },
            "estimation_params": {
                "fit_intervention_model": bool(fit_intervention_model),
                "beta": 0.0,
                "eta": 0.0,
                "zeta": 0.0,
                "psi": 0.0,
                "tau_params": None,
                "tau_zero_mean": bool(tau_zero_mean),
                "tau_smoothness_lambda": float(tau_smoothness_lambda),
            },
            "real_data_params": {
                "source": "SeattleDMI",
                "outcome_column": outcome_column,
                "network_name": network_name,
                "field_basis_mode": field_basis_mode,
                "field_basis_names": list(field_basis_names),
            },
        }
    )
    return config


def write_index_tables(
    experiment_dir: Path,
    node_table: pd.DataFrame,
    times: list[int],
) -> None:
    """Write node and time lookup tables for downstream interpretation."""
    keep_columns = [
        "node_index",
        "GEOID10",
        "treated_ever",
        "intervention_start_time",
        "NEIGHBORHOOD_DISTRICT_NAME",
        "centroid_x",
        "centroid_y",
    ]
    node_table[keep_columns].to_csv(experiment_dir / "node_index.csv", index=False)
    time_index = pd.DataFrame(
        {
            "model_index": np.arange(len(times) - 1, dtype=int),
            "original_time": times[1:],
        }
    )
    time_index.to_csv(experiment_dir / "time_index.csv", index=False)


def save_experiment(
    experiment_dir: Path,
    config,
    metadata: dict[str, object],
    panel: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    interaction_matrix: sparse.csr_matrix,
    interaction_name: str,
    adjacency_edges: pd.DataFrame,
    node_table: pd.DataFrame,
    times: list[int],
) -> None:
    """Write one SeattleDMI experiment folder in the format expected by mple.py."""
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
    sparse.save_npz(experiment_dir / "gamma_matrix_sparse.npz", interaction_matrix)
    sparse.save_npz(experiment_dir / "interaction_basis_sparse.npz", interaction_matrix)
    np.save(
        experiment_dir / "interaction_basis_names.npy",
        np.asarray([interaction_name], dtype="<U128"),
    )
    write_index_tables(experiment_dir, node_table, times)
    adjacency_edges.to_csv(experiment_dir / "adjacency_edge_list.csv.gz", index=False)
    panel.to_csv(experiment_dir / "panel_data.csv.gz", index=False)


def write_manifest(manifest_path: Path, manifest_rows: list[dict[str, object]], overwrite: bool) -> None:
    """Merge this Seattle run's manifest rows into the output-root manifest."""
    if not manifest_rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new_manifest = pd.DataFrame(manifest_rows)
    if manifest_path.exists() and not overwrite:
        existing_manifest = pd.read_csv(manifest_path)
        if "experiment_name" in existing_manifest.columns:
            existing_manifest = existing_manifest.loc[
                ~existing_manifest["experiment_name"].isin(new_manifest["experiment_name"])
            ].copy()
        combined_manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True, sort=False)
    else:
        combined_manifest = new_manifest
    combined_manifest.to_csv(manifest_path, index=False)


def experiment_has_fit_outputs(experiment_dir: Path) -> bool:
    """Return whether one experiment folder already has fitted MPLE outputs."""
    return (experiment_dir / "mple_summary.csv").exists()


def experiment_has_panel_artifacts(experiment_dir: Path) -> bool:
    """Return whether one experiment folder already has the saved panel inputs."""
    required = [
        experiment_dir / "panel_data.npz",
        experiment_dir / "x_0.npy",
        experiment_dir / "z_0.npy",
        experiment_dir / "gamma_matrix_sparse.npz",
        experiment_dir / "field_basis.npy",
    ]
    return all(path.exists() for path in required)


def run_mple(experiment_dir: Path, steps: int, tol: float, seed: int) -> None:
    """Launch mple.py on one prepared SeattleDMI experiment folder."""
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
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def run_mple_with_mode(
    experiment_dir: Path,
    steps: int,
    tol: float,
    seed: int,
    outcome_only: bool,
) -> None:
    """Launch mple.py on one prepared SeattleDMI experiment folder, optionally in outcome-only mode."""
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


def main() -> None:
    """Build SeattleDMI real-data MPLE experiments and optionally run the fits."""
    parser = argparse.ArgumentParser(
        description="Prepare and optionally fit SeattleDMI MPLE experiments."
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/SeattleDMI/processed"),
        help="Directory containing the processed SeattleDMI tables.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("experiments/SeattleDMI"),
        help="Root directory where experiment subfolders will be written.",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("experiments/SeattleDMI/manifest.csv"),
        help="CSV manifest that records each written experiment folder.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="*",
        default=list(DEFAULT_OUTCOMES),
        help="Binary outcome columns from seattledmi_binary_outcomes.csv.gz.",
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        default=list(DEFAULT_NETWORKS),
        help="Known-network variants to realize.",
    )
    parser.add_argument(
        "--run_mple",
        action="store_true",
        help="Run mple.py after writing each experiment folder.",
    )
    parser.add_argument(
        "--outcome_only",
        action="store_true",
        help="Pass --outcome_only through to mple.py so only the x-outcome model is fit.",
    )
    parser.add_argument(
        "--field_basis_mode",
        choices=["static", "zero"],
        default="static",
        help="Choose a static covariate basis or a zero basis for the external field.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum L-BFGS iterations when --run_mple is enabled.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Optimizer tolerance passed to mple.py when --run_mple is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to mple.py when --run_mple is enabled.",
    )
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
        "--overwrite",
        action="store_true",
        help="Allow rewriting existing experiment folders.",
    )
    parser.add_argument(
        "--max_experiments",
        type=int,
        default=None,
        help="Optional cap on the number of experiments to materialize.",
    )
    args = parser.parse_args()

    processed_dir = (REPO_ROOT / args.processed_dir).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    manifest_path = (REPO_ROOT / args.manifest_path).resolve()
    default_manifest_path = (REPO_ROOT / Path("experiments/SeattleDMI/manifest.csv")).resolve()
    experiment_group_name = experiment_group(args.field_basis_mode, args.outcome_only)
    output_root = output_root / experiment_group_name
    if manifest_path == default_manifest_path:
        manifest_path = output_root / "manifest.csv"
    if args.overwrite:
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        if manifest_path.exists():
            manifest_path.unlink()

    binary_outcomes, block_features, crosswalk, centroids = load_inputs(processed_dir)
    node_table = build_node_table(block_features, crosswalk, centroids)
    node_order = node_table["GEOID10"].tolist()

    network_edge_tables = load_network_edge_tables(processed_dir, tuple(args.networks))
    networks = build_networks(node_table, network_edge_tables)
    field_basis, field_basis_names = build_field_basis(node_table, args.field_basis_mode)
    experiment_count = 0
    manifest_rows: list[dict[str, object]] = []

    for outcome_column in args.outcomes:
        if outcome_column not in binary_outcomes.columns:
            raise ValueError(f"Unknown outcome column '{outcome_column}'.")

        x, z, x_0, z_0, times, s = build_panel_arrays(binary_outcomes, outcome_column, node_order)
        saved_panel = build_saved_panel(binary_outcomes, outcome_column)
        binary_summary = compute_binary_summary(binary_outcomes, outcome_column)

        for network_name, gamma_matrix in networks.items():
            adjacency_edges = network_edge_tables[network_name]
            validate_basis_infinity_norms(field_basis, gamma_matrix)
            experiment_name = f"{outcome_column}__{network_name}"
            experiment_dir = output_root / experiment_name
            stats = sparse_matrix_stats(gamma_matrix)
            config = create_config(
                n_nodes=x.shape[1],
                t_steps=x.shape[0],
                s=s,
                field_basis_names=field_basis_names,
                field_basis_mode=args.field_basis_mode,
                network_name=network_name,
                outcome_column=outcome_column,
                tau_zero_mean=args.tau_zero_mean,
                tau_smoothness_lambda=args.tau_smoothness_lambda,
                fit_intervention_model=not args.outcome_only,
            )
            metadata = {
                "source": "SeattleDMI",
                "has_truth": False,
                "outcome_column": outcome_column,
                "base_outcome": outcome_base_name(outcome_column),
                "x_sign_convention": "+1_above_threshold_-1_below_threshold",
                "z_sign_convention": "+1_intervention_-1_no_intervention",
                "fit_intervention_model": bool(not args.outcome_only),
                "experiment_group": experiment_group_name,
                "network_name": network_name,
                "field_basis_mode": args.field_basis_mode,
                "field_basis_names": list(field_basis_names),
                "tau_zero_mean": bool(args.tau_zero_mean),
                "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
                "node_count": int(x.shape[1]),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
                **stats,
            }

            manifest_row = {
                "experiment_name": experiment_name,
                "experiment_group": experiment_group_name,
                "path": str(experiment_dir),
                "outcome_column": outcome_column,
                "network_name": network_name,
                "node_count": int(x.shape[1]),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
                "fit_intervention_model": bool(not args.outcome_only),
                "tau_zero_mean": bool(args.tau_zero_mean),
                "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
                **stats,
            }

            if experiment_dir.exists() and not args.overwrite:
                if not experiment_has_panel_artifacts(experiment_dir):
                    raise FileExistsError(
                        f"{experiment_dir} exists but is missing required panel artifacts. "
                        "Re-run with --overwrite to rebuild it cleanly."
                    )
                if args.run_mple and not experiment_has_fit_outputs(experiment_dir):
                    run_mple_with_mode(
                        experiment_dir,
                        steps=args.steps,
                        tol=args.tol,
                        seed=args.seed,
                        outcome_only=args.outcome_only,
                    )
                manifest_rows.append(manifest_row)
                experiment_count += 1
                if args.max_experiments is not None and experiment_count >= args.max_experiments:
                    write_manifest(manifest_path, manifest_rows, overwrite=args.overwrite)
                    return
                continue

            save_experiment(
                experiment_dir=experiment_dir,
                config=config,
                metadata=metadata,
                panel=saved_panel,
                x=x,
                z=z,
                x_0=x_0,
                z_0=z_0,
                field_basis=field_basis,
                field_basis_names=field_basis_names,
                interaction_matrix=gamma_matrix,
                interaction_name=network_name,
                adjacency_edges=adjacency_edges,
                node_table=node_table,
                times=times,
            )
            binary_summary.to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
            binary_lookup = binary_summary.set_index("variable")
            (experiment_dir / "binary_definition_summary.md").write_text(
                "# SeattleDMI Binary Experiment Summary\n\n"
                f"- Outcome: `{outcome_column}`\n"
                f"- Outcome positive share: `{binary_lookup.loc['outcome', 'positive_share']:.6f}`\n"
                f"- Outcome variance: `{binary_lookup.loc['outcome', 'variance']:.6f}`\n"
                f"- Outcome transition rate: `{binary_lookup.loc['outcome', 'transition_rate']:.6f}`\n"
                "- Intervention: `Intervention == 1`\n"
                f"- Intervention positive share: `{binary_lookup.loc['intervention', 'positive_share']:.6f}`\n"
                f"- Intervention variance: `{binary_lookup.loc['intervention', 'variance']:.6f}`\n"
                f"- Intervention transition rate: `{binary_lookup.loc['intervention', 'transition_rate']:.6f}`\n"
                f"- Network: `{network_name}`\n",
                encoding="utf-8",
            )
            if args.run_mple:
                run_mple_with_mode(
                    experiment_dir,
                    steps=args.steps,
                    tol=args.tol,
                    seed=args.seed,
                    outcome_only=args.outcome_only,
                )
            manifest_rows.append(manifest_row)

            experiment_count += 1
            if args.max_experiments is not None and experiment_count >= args.max_experiments:
                write_manifest(manifest_path, manifest_rows, overwrite=args.overwrite)
                return

    write_manifest(manifest_path, manifest_rows, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
