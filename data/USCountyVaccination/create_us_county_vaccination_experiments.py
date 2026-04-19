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
    build_experiment_grid,
    build_field_basis,
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
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trimmed or full-scope realized artifact names.",
    )
    parser.add_argument(
        "--field_mode",
        choices=["additive", "latent_feature_matrix"],
        default="additive",
        help="Field parameterization recorded in realized configs.",
    )
    parser.add_argument("--latent_rank", type=int, default=10)
    parser.add_argument("--latent_B", type=float, default=1.0)
    parser.add_argument("--tau_zero_mean", action="store_true")
    parser.add_argument("--tau_smoothness_lambda", type=float, default=0.0)
    parser.add_argument("--beta_mask_pre_intervention", action="store_true")
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


def create_experiment_folders(args: argparse.Namespace) -> None:
    if args.field_mode == "latent_feature_matrix":
        if args.latent_rank <= 0:
            raise ValueError("--latent_rank must be positive when --field_mode latent_feature_matrix.")
        if args.latent_B <= 0.0:
            raise ValueError("--latent_B must be positive when --field_mode latent_feature_matrix.")

    _, features, centroids = load_inputs()
    full_node_table = build_node_table(features, centroids)
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
    for idx, item in enumerate(build_experiment_grid(args)):
        if args.max_experiments is not None and idx >= args.max_experiments:
            break
        outcome_code = item["outcome_code"]
        intervention_code = item["intervention_code"]
        lag_code = item["lag_code"]
        network_name = item["network_name"]
        experiment_dir = output_root / experiment_name(outcome_code, intervention_code, lag_code, network_name)
        if experiment_dir.exists() and args.overwrite:
            shutil.rmtree(experiment_dir)

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
        if args.field_mode == "latent_feature_matrix":
            field_basis_names = ()
            field_basis_mode = "latent_feature_matrix"
            model_field_mode = "latent_feature_matrix"
        else:
            field_basis, field_basis_names, field_basis_mode = build_field_basis(node_table)
            model_field_mode = "shared_feature_field" if field_basis.shape[0] > 0 else "uniform"

        gamma_matrix, adjacency_edges = subset_network_artifact(network_artifact, realized_node_order)
        treated_rows = np.any(z == 1, axis=1)
        s = int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])
        stats = sparse_matrix_stats(gamma_matrix)
        state_scope_label = str(shared_metadata.get("state", STATE_SCOPE_LABEL))
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
            "tau_zero_mean": bool(args.tau_zero_mean),
            "tau_smoothness_lambda": float(args.tau_smoothness_lambda),
            "beta_mask_pre_intervention": bool(args.beta_mask_pre_intervention),
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
        metadata["time_steps"] = int(x.shape[0])
        metadata["pre_intervention_steps"] = int(s)

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
                aligned_panel,
                node_table,
                time_index,
                x,
                z,
                x_0,
                z_0,
            )
            binary_summary = compute_binary_summary(aligned_panel)
            binary_summary.to_csv(experiment_dir / "binary_definition_summary.csv", index=False)
            excluded_node_table.to_csv(experiment_dir / "excluded_node_index.csv", index=False)
            binary_lookup = binary_summary.set_index("variable")
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
                f"- Realized weeks: `{time_index['WeekEndDate'].min().date()}` through `{time_index['WeekEndDate'].max().date()}`\n",
                encoding="utf-8",
            )

        manifest_rows.append(
            {
                "experiment_name": experiment_dir.name,
                "experiment_slug": slugify(experiment_dir.name),
                "descriptor": experiment_dir.name,
                "experiment_path": str(experiment_dir.resolve()),
                "intervention_source": "real_data",
                "graph_source": network_name,
                "N": int(len(realized_node_order)),
                "T": int(x.shape[0]),
                "s": int(s),
                "has_truth": False,
                "outcome_code": outcome_code,
                "intervention_code": intervention_code,
                "lag_code": lag_code,
                "network_name": network_name,
                "intervention_family": INTERVENTION_SPECS[intervention_code].family,
                "field_basis_mode": field_basis_mode,
                "model_field_mode": model_field_mode,
                "latent_rank": int(args.latent_rank) if model_field_mode == "latent_feature_matrix" else None,
                "latent_B": float(args.latent_B) if model_field_mode == "latent_feature_matrix" else None,
                "trim_applied": bool(shared_metadata.get("trim_applied", args.trim)),
                "trim_rule": shared_metadata.get("trim_rule", ""),
                "requested_node_count": int(_metadata_value(shared_metadata, "requested_node_count")),
                "node_count": int(len(realized_node_order)),
                "dropped_node_count": int(_metadata_value(shared_metadata, "dropped_node_count")),
                "time_steps": int(x.shape[0]),
                "pre_intervention_steps": int(s),
                "realized_start_date": time_index["WeekEndDate"].min().date().isoformat(),
                "realized_end_date": time_index["WeekEndDate"].max().date().isoformat(),
                "requested_calendar_weeks": int(_metadata_value(shared_metadata, "requested_calendar_weeks")),
                "realized_calendar_weeks": int(_metadata_value(shared_metadata, "realized_calendar_weeks")),
                "weeks_dropped_due_to_missing_or_lag": int(_metadata_value(shared_metadata, "weeks_dropped_due_to_missing_or_lag")),
                "support_selection_rule": shared_metadata.get("support_selection_rule", ""),
                "shared_panel_dir": str(shared_panel_dir),
                **stats,
            }
        )

    _write_manifest(output_root / "generation_manifest.csv", manifest_rows, args.overwrite)


def main() -> None:
    create_experiment_folders(parse_args())


if __name__ == "__main__":
    main()
