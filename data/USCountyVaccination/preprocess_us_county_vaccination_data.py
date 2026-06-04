"""Preprocess USCountyVaccination data and materialize reusable realized artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # noqa: E402
    from .processed_data import build_binary_panel, build_processed_outputs
    from .experiment_artifacts import (
        INTERVENTION_SPECS,
        OUTCOME_SPECS,
        REPO_ROOT as WORKFLOW_REPO_ROOT,
        SOURCE_LABEL,
        apply_optional_trim,
        assembled_panel_from_arrays,
        build_experiment_grid,
        build_node_table,
        build_realized_intervention_artifact,
        build_realized_network_artifact,
        build_realized_outcome_artifact,
        canonical_time_index,
        lag_code_to_steps,
        load_inputs,
        load_network_edge_tables,
        load_network_matrix,
        realized_intervention_name,
        realized_network_name,
        realized_outcome_name,
        select_dense_suffix_support,
        shared_panel_name,
        write_realized_binary_artifact,
        write_realized_network_artifact,
        write_shared_panel_artifacts,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from processed_data import build_binary_panel, build_processed_outputs
    from experiment_artifacts import (
        INTERVENTION_SPECS,
        OUTCOME_SPECS,
        REPO_ROOT as WORKFLOW_REPO_ROOT,
        SOURCE_LABEL,
        apply_optional_trim,
        assembled_panel_from_arrays,
        build_experiment_grid,
        build_node_table,
        build_realized_intervention_artifact,
        build_realized_network_artifact,
        build_realized_outcome_artifact,
        canonical_time_index,
        lag_code_to_steps,
        load_inputs,
        load_network_edge_tables,
        load_network_matrix,
        realized_intervention_name,
        realized_network_name,
        realized_outcome_name,
        select_dense_suffix_support,
        shared_panel_name,
        write_realized_binary_artifact,
        write_realized_network_artifact,
        write_shared_panel_artifacts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build processed US county data, binary panels, realized artifacts, "
            "and shared panels. This does not create experiment folders."
        )
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite realized artifacts and shared panels.")
    parser.add_argument("--max_experiments", type=int, default=None, help="Optional cap on shared panels to build.")
    parser.add_argument("--lags", nargs="*", default=None, help="Lag codes to include, for example 0w 1w 2w.")
    parser.add_argument("--outcomes", nargs="*", default=None, help="Outcome codes to include.")
    parser.add_argument("--interventions", nargs="*", default=None, help="Intervention codes to include.")
    parser.add_argument(
        "--networks",
        nargs="*",
        default=["contiguity"],
        choices=["contiguity", "knn_8", "distance_kernel_8"],
        help="Known-network variants to realize. Defaults to contiguity only.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("experiments/USCountyVaccination_US_trimmed"),
        help="Root directory where realized artifacts and shared panels will be written.",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict support to mainland US counties with total population at least 2,000.",
    )
    parser.add_argument(
        "--vaccination_source",
        choices=["bansal", "cdc"],
        default="cdc",
        help="Vaccination source passed to the processed-data builder.",
    )
    parser.add_argument("--reuse_processed_tables", action="store_true")
    parser.add_argument("--reuse_processed_networks", action="store_true")
    return parser.parse_args()


def _processed_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        vaccination_source=args.vaccination_source,
        reuse_processed_tables=args.reuse_processed_tables,
        reuse_processed_networks=args.reuse_processed_networks,
    )


def materialize_realized_artifacts(args: argparse.Namespace) -> None:
    build_processed_outputs(_processed_args(args))
    build_binary_panel()

    panel, node_geography, centroids = load_inputs()
    full_node_table = build_node_table(node_geography, centroids)
    if set(full_node_table["fips"].astype(str)) != set(centroids["fips"].astype(str)):
        raise ValueError("Node geography and centroid county coverage do not match.")
    full_node_table, panel, trim_metadata = apply_optional_trim(full_node_table, panel, args.trim)
    full_node_order = full_node_table["fips"].astype(str).tolist()

    output_root = (WORKFLOW_REPO_ROOT / args.output_root).resolve()
    realized_outcome_root = output_root / "realized_outcomes"
    realized_intervention_root = output_root / "realized_interventions"
    realized_network_root = output_root / "realized_networks"
    shared_panel_root = output_root / "shared_panels"
    for root in [realized_outcome_root, realized_intervention_root, realized_network_root, shared_panel_root]:
        root.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(args)
    core_time_index = canonical_time_index(panel)
    requested_outcomes = sorted({item["outcome_code"] for item in grid})
    requested_interventions = sorted(
        {(item["intervention_code"], item["lag_code"]) for item in grid}
    )
    requested_networks = sorted({item["network_name"] for item in grid})

    realized_outcomes = {}
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
        realized_outcomes[outcome_code] = artifact

    realized_interventions = {}
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
        realized_interventions[(intervention_code, lag_code)] = artifact

    network_edge_tables = load_network_edge_tables(args.networks)
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

    shared_counter = 0
    for outcome_code, intervention_code, lag_code in sorted(
        {
            (item["outcome_code"], item["intervention_code"], item["lag_code"])
            for item in grid
        }
    ):
        outcome_artifact = realized_outcomes[outcome_code]
        intervention_artifact = realized_interventions[(intervention_code, lag_code)]
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
        shared_panel_dir = shared_panel_root / shared_panel_name(
            outcome_code=outcome_code,
            intervention_code=intervention_code,
            lag_code=lag_code,
            trim_applied=bool(trim_metadata["trim_applied"]),
        )
        metadata = {
            "source": SOURCE_LABEL,
            "state": str(trim_metadata["trim_scope_label"]),
            "outcome_code": outcome_code,
            "outcome_label": OUTCOME_SPECS[outcome_code].label,
            "intervention_code": intervention_code,
            "intervention_label": INTERVENTION_SPECS[intervention_code].label,
            "intervention_family": INTERVENTION_SPECS[intervention_code].family,
            "lag_code": lag_code,
            "lag_steps": lag_code_to_steps(lag_code),
            **trim_metadata,
            **support_metadata,
            "node_count": int(len(realized_node_order)),
            "time_steps": int(x.shape[0]),
            "realized_week_start_date": time_index["WeekStartDate"].min().date().isoformat(),
            "realized_week_end_date": time_index["WeekEndDate"].max().date().isoformat(),
            "realized_outcome_dir": str(outcome_artifact.artifact_dir),
            "realized_intervention_dir": str(intervention_artifact.artifact_dir),
        }
        if args.overwrite or not (shared_panel_dir / "panel_data.npz").exists():
            write_shared_panel_artifacts(
                shared_panel_dir=shared_panel_dir,
                panel=aligned_panel,
                node_table=node_table,
                time_index=time_index,
                x=x,
                z=z,
                x_0=x_0,
                z_0=z_0,
                metadata=metadata,
            )
        shared_counter += 1
        if args.max_experiments is not None and shared_counter >= args.max_experiments:
            break


def main() -> None:
    materialize_realized_artifacts(parse_args())


if __name__ == "__main__":
    main()
