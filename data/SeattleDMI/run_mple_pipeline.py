from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import validate_basis_infinity_norms


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


def normalize_vector_infinity(vector: np.ndarray) -> np.ndarray:
    """Scale one field template so its infinity norm is one."""
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array, ord=np.inf)
    if norm < 1e-12:
        return np.zeros_like(array)
    return array / norm


def normalize_sparse_matrix_infinity(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Scale one sparse interaction matrix so its infinity norm is one."""
    csr = matrix.tocsr().astype(float)
    row_sums = np.asarray(np.abs(csr).sum(axis=1)).ravel()
    norm = float(row_sums.max()) if row_sums.size else 0.0
    if norm < 1e-12:
        return csr
    return (csr / norm).tocsr()


def outcome_base_name(outcome_column: str) -> str:
    """Map a binary outcome column name back to its underlying crime count variable."""
    if outcome_column.startswith("i_drugs"):
        return "i_drugs"
    if outcome_column.startswith("any_crime"):
        return "any_crime"
    return outcome_column.split("_gt_", 1)[0]


def load_inputs(processed_dir: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load the processed SeattleDMI tables needed for real-data MPLE experiments."""
    binary_outcomes_path = processed_dir / "seattledmi_binary_outcomes.csv.gz"
    if not binary_outcomes_path.exists():
        fallback_path = processed_dir / "seattledmi_binary_outcomes.csv"
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"Could not find {binary_outcomes_path.name} or {fallback_path.name} in {processed_dir}."
            )
        binary_outcomes_path = fallback_path
    binary_outcomes = pd.read_csv(binary_outcomes_path, dtype={"GEOID10": str})
    blocks = gpd.read_file(processed_dir / "seattledmi_blocks.gpkg", layer="blocks")
    blocks["GEOID10"] = blocks["GEOID10"].astype(str).str.zfill(15)
    return binary_outcomes, blocks


def build_node_table(blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create the canonical node ordering and projected centroid coordinates."""
    projected = blocks.to_crs(2285).copy()
    centroids = projected.geometry.centroid
    node_table = projected.drop(columns="geometry").copy()
    node_table["centroid_x"] = centroids.x
    node_table["centroid_y"] = centroids.y
    node_table = node_table.sort_values("GEOID10").reset_index(drop=True)
    node_table["node_index"] = np.arange(len(node_table), dtype=int)
    return pd.DataFrame(node_table)


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


def build_contiguity_network(
    processed_dir: Path,
    node_lookup: dict[str, int],
    n_nodes: int,
) -> sparse.csr_matrix:
    """Load the saved queen-contiguity edge list into a normalized sparse matrix."""
    edges = pd.read_csv(
        processed_dir / "seattledmi_block_adjacency.csv.gz",
        dtype={"GEOID10": str, "neighbor_GEOID10": str},
    )
    rows = edges["GEOID10"].map(node_lookup).to_numpy()
    cols = edges["neighbor_GEOID10"].map(node_lookup).to_numpy()
    data = np.ones(len(edges), dtype=float)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    matrix = matrix + matrix.T
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return normalize_sparse_matrix_infinity(matrix)


def build_knn_network(
    coords: np.ndarray,
    k: int,
) -> sparse.csr_matrix:
    """Construct a symmetric k-nearest-neighbor graph from projected centroids."""
    tree = cKDTree(coords)
    _, neighbor_idx = tree.query(coords, k=min(k + 1, len(coords)))
    rows = np.repeat(np.arange(len(coords)), neighbor_idx.shape[1] - 1)
    cols = neighbor_idx[:, 1:].reshape(-1)
    data = np.ones(len(rows), dtype=float)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(coords), len(coords))).tocsr()
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return normalize_sparse_matrix_infinity(matrix)


def build_distance_kernel_network(
    coords: np.ndarray,
    k: int,
) -> sparse.csr_matrix:
    """Construct a sparse centroid-distance kernel restricted to k nearest neighbors."""
    tree = cKDTree(coords)
    distances, neighbor_idx = tree.query(coords, k=min(k + 1, len(coords)))
    neighbor_distances = distances[:, 1:].reshape(-1)
    positive = neighbor_distances[neighbor_distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-neighbor_distances / scale)
    rows = np.repeat(np.arange(len(coords)), neighbor_idx.shape[1] - 1)
    cols = neighbor_idx[:, 1:].reshape(-1)
    matrix = sparse.coo_matrix((weights, (rows, cols)), shape=(len(coords), len(coords))).tocsr()
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return normalize_sparse_matrix_infinity(matrix)


def build_networks(
    processed_dir: Path,
    node_table: pd.DataFrame,
    network_names: tuple[str, ...],
) -> dict[str, sparse.csr_matrix]:
    """Build the requested known-network variants for SeattleDMI."""
    coords = node_table[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    node_lookup = dict(zip(node_table["GEOID10"], node_table["node_index"]))
    networks: dict[str, sparse.csr_matrix] = {}

    for network_name in network_names:
        if network_name == "contiguity":
            network = build_contiguity_network(processed_dir, node_lookup, len(node_table))
        elif network_name.startswith("knn_"):
            k = int(network_name.split("_", 1)[1])
            network = build_knn_network(coords, k)
        elif network_name.startswith("centroid_distance_kernel_"):
            k = int(network_name.rsplit("_", 1)[1])
            network = build_distance_kernel_network(coords, k)
        else:
            raise ValueError(f"Unknown network '{network_name}'.")
        networks[network_name] = network
    return networks


def ratio_feature(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Build a safe ratio feature, defaulting to zero when the denominator vanishes."""
    num = np.nan_to_num(numerator.to_numpy(dtype=float), nan=0.0)
    den = np.nan_to_num(denominator.to_numpy(dtype=float), nan=0.0)
    out = np.zeros_like(num, dtype=float)
    valid = np.abs(den) > 1e-12
    out[valid] = num[valid] / den[valid]
    return out


def build_field_basis(node_table: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct an infinity-normalized field basis from static block covariates only."""
    candidate_features = [
        ("intercept", np.ones(len(node_table), dtype=float), False),
        ("total_pop", np.nan_to_num(node_table["TotalPop"].to_numpy(dtype=float), nan=0.0), True),
        ("black_share", ratio_feature(node_table["BLACK"], node_table["TotalPop"]), True),
        ("hispanic_share", ratio_feature(node_table["HISPANIC"], node_table["TotalPop"]), True),
        ("male_1521_share", ratio_feature(node_table["Males_1521"], node_table["TotalPop"]), True),
        ("family_household_share", ratio_feature(node_table["FAMILYHOUS"], node_table["HOUSEHOLDS"]), True),
        ("female_household_share", ratio_feature(node_table["FEMALE_HOU"], node_table["HOUSEHOLDS"]), True),
        ("renter_share", ratio_feature(node_table["RENTER_HOU"], node_table["HOUSEHOLDS"]), True),
        ("vacant_share", ratio_feature(node_table["VACANT_HOU"], node_table["HOUSEHOLDS"]), True),
    ]

    basis_vectors: list[np.ndarray] = []
    basis_names: list[str] = []
    for name, raw_feature, center in candidate_features:
        feature = np.asarray(raw_feature, dtype=float)
        if center:
            feature = feature - feature.mean()
        normalized = normalize_vector_infinity(feature)
        if np.linalg.norm(normalized, ord=np.inf) < 1e-12:
            continue
        basis_vectors.append(normalized)
        basis_names.append(name)

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
    network_name: str,
    outcome_column: str,
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
            },
            "real_data_params": {
                "source": "SeattleDMI",
                "outcome_column": outcome_column,
                "network_name": network_name,
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
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    field_basis: np.ndarray,
    field_basis_names: tuple[str, ...],
    interaction_matrix: sparse.csr_matrix,
    interaction_name: str,
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


def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    """Append one experiment entry to the manifest CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    if args.overwrite:
        if output_root.exists():
            for child in output_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        if manifest_path.exists():
            manifest_path.unlink()

    binary_outcomes, blocks = load_inputs(processed_dir)
    node_table = build_node_table(blocks)
    node_order = node_table["GEOID10"].tolist()

    networks = build_networks(processed_dir, node_table, tuple(args.networks))
    experiment_count = 0

    for outcome_column in args.outcomes:
        if outcome_column not in binary_outcomes.columns:
            raise ValueError(f"Unknown outcome column '{outcome_column}'.")

        x, z, x_0, z_0, times, s = build_panel_arrays(binary_outcomes, outcome_column, node_order)
        field_basis, field_basis_names = build_field_basis(node_table)

        for network_name, gamma_matrix in networks.items():
            validate_basis_infinity_norms(field_basis, gamma_matrix)
            experiment_name = f"{outcome_column}__{network_name}"
            experiment_dir = output_root / experiment_name
            stats = sparse_matrix_stats(gamma_matrix)
            config = create_config(
                n_nodes=x.shape[1],
                t_steps=x.shape[0],
                s=s,
                field_basis_names=field_basis_names,
                network_name=network_name,
                outcome_column=outcome_column,
                fit_intervention_model=not args.outcome_only,
            )
            metadata = {
                "source": "SeattleDMI",
                "has_truth": False,
                "outcome_column": outcome_column,
                "base_outcome": outcome_base_name(outcome_column),
                "x_sign_convention": "+1_good_-1_bad",
                "z_sign_convention": "+1_intervention_-1_no_intervention",
                "fit_intervention_model": bool(not args.outcome_only),
                "network_name": network_name,
                "field_basis_names": list(field_basis_names),
                "node_count": int(x.shape[1]),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
                **stats,
            }

            manifest_row = {
                "experiment_name": experiment_name,
                "path": str(experiment_dir),
                "outcome_column": outcome_column,
                "network_name": network_name,
                "node_count": int(x.shape[1]),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
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
                experiment_count += 1
                if args.max_experiments is not None and experiment_count >= args.max_experiments:
                    return
                continue

            save_experiment(
                experiment_dir=experiment_dir,
                config=config,
                metadata=metadata,
                x=x,
                z=z,
                x_0=x_0,
                z_0=z_0,
                field_basis=field_basis,
                field_basis_names=field_basis_names,
                interaction_matrix=gamma_matrix,
                interaction_name=network_name,
                node_table=node_table,
                times=times,
            )
            append_manifest_row(manifest_path, manifest_row)
            if args.run_mple:
                run_mple_with_mode(
                    experiment_dir,
                    steps=args.steps,
                    tol=args.tol,
                    seed=args.seed,
                    outcome_only=args.outcome_only,
                )

            experiment_count += 1
            if args.max_experiments is not None and experiment_count >= args.max_experiments:
                return


if __name__ == "__main__":
    main()
