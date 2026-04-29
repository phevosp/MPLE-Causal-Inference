"""Create USCountyVaccination experiment folders from realized artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_artifacts import (  # noqa: E402
    INTERVENTION_SPECS,
    OUTCOME_SPECS,
    REPO_ROOT as WORKFLOW_REPO_ROOT,
    SOURCE_LABEL,
    STATE_SCOPE_LABEL,
    assembled_panel_from_arrays,
    build_experiment_grid,
    build_node_table,
    compute_binary_summary,
    create_config,
    experiment_has_panel_artifacts,
    experiment_name,
    existing_experiment_trim_setting,
    lag_code_to_steps,
    load_inputs,
    load_realized_binary_artifact,
    load_realized_network_artifact,
    load_shared_panel_artifacts,
    realized_intervention_name,
    realized_network_name,
    realized_outcome_name,
    save_experiment,
    shared_panel_name,
    sparse_matrix_stats,
    subset_network_artifact,
)
from pipeline_specs import slugify  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared-pipeline experiment folders from saved USCounty realized artifacts."
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow rewriting existing experiment folders.")
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
        default=Path("experiments/USCountyVaccination_US_trimmed"),
        help="Root directory containing realized artifacts and receiving experiment folders.",
    )
    parser.add_argument(
        "--start_dates",
        nargs="*",
        default=None,
        help=(
            "Optional ISO dates (YYYY-MM-DD). Each requested date is resolved to the "
            "first available modeled WeekEndDate on or after it, and a separate "
            "suffixed experiment is materialized for each resolved start."
        ),
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trimmed or full-scope realized artifact names.",
    )
    return parser.parse_args()


def _metadata_value(metadata: dict[str, object], primary: str, fallback: str | None = None):
    if primary in metadata:
        return metadata[primary]
    if fallback is not None and fallback in metadata:
        return metadata[fallback]
    raise KeyError(primary)


def _write_manifest(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    if not rows:
        return
    new_manifest = pd.DataFrame(rows)
    if path.exists() and not overwrite:
        existing = pd.read_csv(path)
        if "experiment_name" in existing.columns:
            existing = existing.loc[
                ~existing["experiment_name"].isin(new_manifest["experiment_name"])
            ].copy()
        new_manifest = pd.concat([existing, new_manifest], ignore_index=True, sort=False)
    new_manifest.to_csv(path, index=False)


def _resolve_start_index(
    time_index: pd.DataFrame,
    start_date: str,
) -> tuple[int, str]:
    requested = pd.Timestamp(start_date).normalize()
    week_ends = pd.to_datetime(time_index["WeekEndDate"]).dt.normalize()
    matches = np.flatnonzero(week_ends.ge(requested).to_numpy())
    if matches.size == 0:
        raise ValueError(
            f"Start date {start_date} is after the last available week "
            f"{week_ends.max().date().isoformat()}."
        )
    start_index = int(matches[0])
    if start_index >= len(time_index) - 1:
        raise ValueError(f"Start date {start_date} leaves no transition weeks to fit.")
    return start_index, week_ends.iloc[start_index].date().isoformat()


def _slice_panel_for_start(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    time_index: pd.DataFrame,
    start_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    if len(time_index) != x.shape[0] + 1:
        raise ValueError(
            f"time_index has {len(time_index)} rows but panel x has {x.shape[0]} transition rows."
        )
    if z.shape != x.shape:
        raise ValueError("x and z must have matching shapes before start-date slicing.")

    start_index, resolved_week_end = _resolve_start_index(time_index, start_date)
    x_all = np.vstack([x_0[None, :], x])
    z_all = np.vstack([z_0[None, :], z])
    sliced_time_index = time_index.iloc[start_index:].reset_index(drop=True).copy()
    sliced_time_index["model_index"] = np.arange(len(sliced_time_index), dtype=int)
    sliced_x = x_all[start_index + 1 :].astype(np.int8)
    sliced_z = z_all[start_index + 1 :].astype(np.int8)
    sliced_x_0 = x_all[start_index].astype(np.int8)
    sliced_z_0 = z_all[start_index].astype(np.int8)
    metadata = {
        "requested_start_date": pd.Timestamp(start_date).date().isoformat(),
        "resolved_start_week_end_date": resolved_week_end,
        "start_index": int(start_index),
        "dropped_transition_weeks_for_start": int(start_index),
    }
    return sliced_x, sliced_z, sliced_x_0, sliced_z_0, sliced_time_index, metadata


def _sliced_experiment_name(base_name: str, resolved_start_date: str) -> str:
    return f"{base_name}__start_{slugify(resolved_start_date)}"


def create_experiment_folders(args: argparse.Namespace) -> None:
    _, node_geography, centroids = load_inputs()
    full_node_table = build_node_table(node_geography, centroids)
    output_root = (WORKFLOW_REPO_ROOT / args.output_root).resolve()
    realized_outcome_root = output_root / "realized_outcomes"
    realized_intervention_root = output_root / "realized_interventions"
    realized_network_root = output_root / "realized_networks"
    shared_panel_root = output_root / "shared_panels"
    for root in [realized_outcome_root, realized_intervention_root, realized_network_root, shared_panel_root]:
        if not root.exists():
            raise FileNotFoundError(
                f"Missing {root}. Run preprocess_us_county_vaccination_data.py first."
            )

    manifest_rows: list[dict[str, object]] = []
    created_experiment_count = 0
    stop_requested = False
    requested_start_dates = [None] if not args.start_dates else list(args.start_dates)
    for item in build_experiment_grid(args):
        if stop_requested:
            break
        outcome_code = item["outcome_code"]
        intervention_code = item["intervention_code"]
        lag_code = item["lag_code"]
        network_name = item["network_name"]
        base_experiment_name = experiment_name(
            outcome_code, intervention_code, lag_code, network_name
        )

        outcome_artifact = load_realized_binary_artifact(
            realized_outcome_root / realized_outcome_name(outcome_code, bool(args.trim)),
            "x",
        )
        intervention_artifact = load_realized_binary_artifact(
            realized_intervention_root / realized_intervention_name(intervention_code, lag_code, bool(args.trim)),
            "z",
        )
        network_artifact = load_realized_network_artifact(
            realized_network_root / realized_network_name(network_name, bool(args.trim))
        )
        shared_panel_dir = shared_panel_root / shared_panel_name(
            outcome_code,
            intervention_code,
            lag_code,
            bool(args.trim),
        )
        shared_panel = load_shared_panel_artifacts(shared_panel_dir)
        shared_metadata = shared_panel["metadata"]
        x = shared_panel["x"]
        z = shared_panel["z"]
        x_0 = shared_panel["x_0"]
        z_0 = shared_panel["z_0"]
        time_index = shared_panel["time_index"]
        realized_node_order = shared_panel["node_order"]
        aligned_panel = shared_panel["panel"]

        if outcome_artifact.node_order != intervention_artifact.node_order:
            raise ValueError("Outcome and intervention realized artifacts have different node ordering.")

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

        gamma_matrix, adjacency_edges = subset_network_artifact(network_artifact, realized_node_order)
        stats = sparse_matrix_stats(gamma_matrix)
        state_scope_label = str(shared_metadata.get("state", STATE_SCOPE_LABEL))
        resolved_start_dates: dict[str, str] = {}
        for requested_start_date in requested_start_dates:
            slice_metadata: dict[str, object] = {}
            experiment_x = x
            experiment_z = z
            experiment_x_0 = x_0
            experiment_z_0 = z_0
            experiment_time_index = time_index
            experiment_panel = aligned_panel
            experiment_name_value = base_experiment_name

            if requested_start_date is not None:
                (
                    experiment_x,
                    experiment_z,
                    experiment_x_0,
                    experiment_z_0,
                    experiment_time_index,
                    slice_metadata,
                ) = _slice_panel_for_start(
                    x,
                    z,
                    x_0,
                    z_0,
                    time_index,
                    requested_start_date,
                )
                resolved_start_date = str(slice_metadata["resolved_start_week_end_date"])
                existing_request = resolved_start_dates.get(resolved_start_date)
                if existing_request is not None:
                    raise ValueError(
                        "Requested start dates resolve to duplicate experiment names for "
                        f"'{base_experiment_name}': {existing_request} and {requested_start_date} "
                        f"both resolve to {resolved_start_date}."
                    )
                resolved_start_dates[resolved_start_date] = requested_start_date
                experiment_name_value = _sliced_experiment_name(
                    base_experiment_name, resolved_start_date
                )
                experiment_panel = assembled_panel_from_arrays(
                    x=experiment_x,
                    z=experiment_z,
                    x_0=experiment_x_0,
                    z_0=experiment_z_0,
                    time_index=experiment_time_index,
                    node_order=realized_node_order,
                    outcome_code=outcome_code,
                    intervention_code=intervention_code,
                )

            experiment_dir = output_root / experiment_name_value
            if experiment_dir.exists() and args.overwrite:
                shutil.rmtree(experiment_dir)

            config = create_config(
                n_nodes=len(realized_node_order),
                t_steps=experiment_x.shape[0],
                outcome_code=outcome_code,
                intervention_code=intervention_code,
                lag_code=lag_code,
                network_name=network_name,
                state_scope_label=state_scope_label,
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
                "shared_panel_dir": str(shared_panel_dir),
                "shared_panel_path": str(shared_panel_dir / "panel_data.npz"),
                "shared_x0_path": str(shared_panel_dir / "x_0.npy"),
                "shared_z0_path": str(shared_panel_dir / "z_0.npy"),
                "shared_node_index_path": str(shared_panel_dir / "node_index.csv"),
                "shared_time_index_path": str(shared_panel_dir / "time_index.csv"),
                "realized_outcome_dir": str(outcome_artifact.artifact_dir),
                "realized_intervention_dir": str(intervention_artifact.artifact_dir),
                "realized_network_dir": str(network_artifact.artifact_dir),
                **shared_metadata,
                **stats,
            }
            metadata["time_steps"] = int(experiment_x.shape[0])
            if requested_start_date is not None:
                metadata.update(
                    {
                        "requested_start_date": str(slice_metadata["requested_start_date"]),
                        "resolved_start_week_end_date": str(
                            slice_metadata["resolved_start_week_end_date"]
                        ),
                        "start_index": int(slice_metadata["start_index"]),
                        "dropped_transition_weeks_for_start": int(
                            slice_metadata["dropped_transition_weeks_for_start"]
                        ),
                        "support_selection_rule": "materialized_start_week_slice_from_shared_panel",
                        "realized_week_start_date": experiment_time_index["WeekStartDate"]
                        .min()
                        .date()
                        .isoformat(),
                        "realized_week_end_date": experiment_time_index["WeekEndDate"]
                        .max()
                        .date()
                        .isoformat(),
                    }
                )

            if experiment_dir.exists() and not args.overwrite:
                existing_trim_applied = existing_experiment_trim_setting(experiment_dir)
                if existing_trim_applied is not None and existing_trim_applied != bool(args.trim):
                    raise FileExistsError(
                        f"{experiment_dir} exists with trim_applied={existing_trim_applied}. "
                        "Use --overwrite or choose a different --output_root for the other sample scope."
                    )
                if not experiment_has_panel_artifacts(experiment_dir):
                    raise FileExistsError(
                        f"{experiment_dir} exists but is missing panel artifacts. Re-run with --overwrite."
                    )
            else:
                save_experiment(
                    experiment_dir,
                    config,
                    metadata,
                    gamma_matrix,
                    adjacency_edges,
                    experiment_panel,
                    node_table,
                    experiment_time_index,
                    experiment_x,
                    experiment_z,
                    experiment_x_0,
                    experiment_z_0,
                )
                binary_summary = compute_binary_summary(experiment_panel)
                binary_summary.to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
                excluded_node_table.to_csv(experiment_dir / "excluded_node_index.csv", index=False)
                binary_lookup = binary_summary.set_index("variable")
                requested_start_summary = ""
                if requested_start_date is not None:
                    requested_start_summary = (
                        f"- Requested start date: `{slice_metadata['requested_start_date']}`\n"
                        f"- Resolved start week: `{slice_metadata['resolved_start_week_end_date']}`\n"
                    )
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
                    f"- Trim: `{shared_metadata.get('trim_rule', '')}`\n"
                    f"- Requested counties: `{shared_metadata.get('requested_node_count', '')}`\n"
                    f"- Realized counties: `{len(realized_node_order)}`\n"
                    f"{requested_start_summary}"
                    f"- Realized weeks: `{experiment_time_index['WeekEndDate'].min().date()}` through `{experiment_time_index['WeekEndDate'].max().date()}`\n",
                    encoding="utf-8",
                )

            manifest_row = {
                "experiment_name": experiment_dir.name,
                "experiment_slug": slugify(experiment_dir.name),
                "descriptor": experiment_dir.name,
                "experiment_path": str(experiment_dir.resolve()),
                "intervention_source": "real_data",
                "graph_source": network_name,
                "N": int(len(realized_node_order)),
                "T": int(experiment_x.shape[0]),
                "has_truth": False,
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "network_name": network_name,
                "intervention_family": INTERVENTION_SPECS[intervention_code].family,
                "trim_applied": bool(shared_metadata.get("trim_applied", args.trim)),
                "trim_rule": shared_metadata.get("trim_rule", ""),
                "requested_node_count": int(_metadata_value(shared_metadata, "requested_node_count")),
                "node_count": int(len(realized_node_order)),
                "dropped_node_count": int(_metadata_value(shared_metadata, "dropped_node_count")),
                "time_steps": int(experiment_x.shape[0]),
                "realized_start_date": experiment_time_index["WeekEndDate"].min().date().isoformat(),
                "realized_end_date": experiment_time_index["WeekEndDate"].max().date().isoformat(),
                "requested_calendar_weeks": int(_metadata_value(shared_metadata, "requested_calendar_weeks")),
                "realized_calendar_weeks": int(len(experiment_time_index)),
                "weeks_dropped_due_to_missing_or_lag": int(_metadata_value(shared_metadata, "weeks_dropped_due_to_missing_or_lag")),
                "support_selection_rule": metadata.get("support_selection_rule", ""),
                "shared_panel_dir": str(shared_panel_dir),
                **stats,
            }
            if requested_start_date is not None:
                manifest_row.update(
                    {
                        "requested_start_date": str(slice_metadata["requested_start_date"]),
                        "resolved_start_week_end_date": str(
                            slice_metadata["resolved_start_week_end_date"]
                        ),
                        "start_index": int(slice_metadata["start_index"]),
                        "dropped_transition_weeks_for_start": int(
                            slice_metadata["dropped_transition_weeks_for_start"]
                        ),
                    }
                )
            manifest_rows.append(manifest_row)
            created_experiment_count += 1
            if (
                args.max_experiments is not None
                and created_experiment_count >= args.max_experiments
            ):
                stop_requested = True
                break

    _write_manifest(output_root / "generation_manifest.csv", manifest_rows, args.overwrite)


def main() -> None:
    create_experiment_folders(parse_args())


if __name__ == "__main__":
    main()
