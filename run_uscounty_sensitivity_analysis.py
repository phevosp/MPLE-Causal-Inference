from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

from data.synthetic_data_generation import derive_pre_intervention_steps
from data.USCountyVaccination.experiment_artifacts import (
    assembled_panel_from_arrays,
    save_experiment,
)
from pipeline_specs import read_csv_manifest, slugify, write_csv_manifest
from report_parameter_recovery_detailed import (
    latent_diagnostics,
    parse_optimizer_status,
    read_summary_entries,
)
from run_fit_pipeline import run_fits


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_START_DATES = (
    "2020-01-26",
    "2020-03-01",
    "2020-06-07",
    "2020-09-06",
    "2021-01-03",
)
DEFAULT_LATENT_RANKS = (0, 10, 20, 40)
DEFAULT_LAMBDA_NUCLEAR_VALUES: tuple[float, ...] = ()
SUMMARY_COLUMNS = [
    "rank_in_sensitivity",
    "experiment_name",
    "parent_experiment_name",
    "sensitivity_start_date",
    "sensitivity_start_week_end_date",
    "sensitivity_start_index",
    "variant_name",
    "optimizer_mode",
    "latent_rank",
    "lambda_nuclear",
    "lambda_uv_ridge",
    "T",
    "s",
    "final_loss",
    "optimizer_status",
    "beta",
    "xi",
    "eta",
    "estimated_field_inf_norm",
    "estimated_field_rank",
    "fit_path",
]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return dict(payload) if isinstance(payload, dict) else {}


def _load_gamma_matrix(experiment_root: Path):
    sparse_path = experiment_root / "gamma_matrix_sparse.npz"
    dense_path = experiment_root / "gamma_matrix.npy"
    if sparse_path.exists():
        return sparse.load_npz(sparse_path).tocsr()
    if dense_path.exists():
        return np.load(dense_path)
    raise FileNotFoundError(f"Missing gamma matrix in {experiment_root}.")


def _load_adjacency_edges(experiment_root: Path) -> pd.DataFrame:
    edge_path = experiment_root / "adjacency_edge_list.csv.gz"
    if edge_path.exists():
        return pd.read_csv(edge_path, dtype={"fips": str, "neighbor_fips": str})
    return pd.DataFrame(columns=["fips", "neighbor_fips"])


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


def _slice_panel(
    experiment_root: Path,
    start_date: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]
]:
    with np.load(experiment_root / "panel_data.npz", allow_pickle=False) as panel:
        x = np.asarray(panel["x"], dtype=np.int8)
        z = np.asarray(panel["z"], dtype=np.int8)
    x_0 = np.asarray(
        np.load(experiment_root / "x_0.npy", allow_pickle=False), dtype=np.int8
    )
    z_0 = np.asarray(
        np.load(experiment_root / "z_0.npy", allow_pickle=False), dtype=np.int8
    )
    time_index = pd.read_csv(
        experiment_root / "time_index.csv",
        parse_dates=["WeekStartDate", "WeekEndDate"],
    )

    if len(time_index) != x.shape[0] + 1:
        raise ValueError(
            f"{experiment_root} has {len(time_index)} time-index rows but "
            f"{x.shape[0]} transition rows."
        )
    if z.shape != x.shape:
        raise ValueError(f"x/z panel shapes differ in {experiment_root}.")

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


def _derived_experiment_name(parent_name: str, resolved_start_date: str) -> str:
    return f"{parent_name}__start_{slugify(resolved_start_date)}"


def _save_sliced_experiment(
    source_row: dict[str, str],
    output_root: Path,
    start_date: str,
    overwrite: bool,
) -> dict[str, Any]:
    source_root = _repo_path(source_row["experiment_path"])
    x, z, x_0, z_0, time_index, slice_metadata = _slice_panel(source_root, start_date)
    resolved_start = str(slice_metadata["resolved_start_week_end_date"])
    parent_name = source_row.get("experiment_name", source_root.name)
    experiment_name = _derived_experiment_name(parent_name, resolved_start)
    experiment_root = output_root / experiment_name
    if experiment_root.exists():
        if overwrite:
            shutil.rmtree(experiment_root)
        else:
            raise FileExistsError(
                f"{experiment_root} already exists. Re-run with --overwrite to rebuild it."
            )

    source_metadata = _load_yaml_mapping(source_root / "experiment_metadata.yaml")
    source_config = OmegaConf.load(source_root / "realized_config.yaml")
    source_config.global_params.N = int(x.shape[1])
    source_config.global_params.T = int(x.shape[0])
    source_config.global_params.s = int(derive_pre_intervention_steps(z))
    if "real_data_params" in source_config:
        source_config.real_data_params.sensitivity_start_date = resolved_start
        source_config.real_data_params.parent_experiment_name = parent_name

    node_table = pd.read_csv(source_root / "node_index.csv", dtype={"fips": str})
    node_order = node_table.sort_values("node_index")["fips"].astype(str).tolist()
    panel = assembled_panel_from_arrays(
        x=x,
        z=z,
        x_0=x_0,
        z_0=z_0,
        time_index=time_index,
        node_order=node_order,
        outcome_code=str(
            source_row.get("outcome_code", source_metadata.get("outcome_code", ""))
        ),
        intervention_code=str(
            source_row.get(
                "intervention_code", source_metadata.get("intervention_code", "")
            )
        ),
    )
    gamma_matrix = _load_gamma_matrix(source_root)
    adjacency_edges = _load_adjacency_edges(source_root)
    metadata = {
        **source_metadata,
        "has_truth": False,
        "sensitivity_parent_experiment_name": parent_name,
        "sensitivity_parent_experiment_path": str(source_root.resolve()),
        "sensitivity_requested_start_date": slice_metadata["requested_start_date"],
        "sensitivity_start_week_end_date": resolved_start,
        "sensitivity_start_index": int(slice_metadata["start_index"]),
        "sensitivity_dropped_transition_weeks_for_start": int(
            slice_metadata["dropped_transition_weeks_for_start"]
        ),
        "support_selection_rule": "sensitivity_start_week_slice_from_existing_panel",
        "shared_panel_dir": "",
        "shared_panel_path": "",
        "shared_x0_path": "",
        "shared_z0_path": "",
        "shared_time_index_path": "",
        "time_steps": int(x.shape[0]),
        "pre_intervention_steps": int(source_config.global_params.s),
        "realized_week_start_date": time_index["WeekStartDate"]
        .min()
        .date()
        .isoformat(),
        "realized_week_end_date": time_index["WeekEndDate"].max().date().isoformat(),
    }
    save_experiment(
        experiment_dir=experiment_root,
        config=source_config,
        metadata=metadata,
        gamma_matrix=gamma_matrix,
        adjacency_edges=adjacency_edges,
        panel=panel,
        node_table=node_table,
        time_index=time_index,
        x=x,
        z=z,
        x_0=x_0,
        z_0=z_0,
    )
    manifest_row: dict[str, Any] = {
        **source_row,
        "experiment_name": experiment_name,
        "experiment_slug": slugify(experiment_name),
        "descriptor": experiment_name,
        "experiment_path": str(experiment_root.resolve()),
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "s": int(source_config.global_params.s),
        "time_steps": int(x.shape[0]),
        "pre_intervention_steps": int(source_config.global_params.s),
        "realized_start_date": resolved_start,
        "realized_end_date": time_index["WeekEndDate"].max().date().isoformat(),
        "sensitivity_parent_experiment_name": parent_name,
        "sensitivity_parent_experiment_path": str(source_root.resolve()),
        "sensitivity_requested_start_date": slice_metadata["requested_start_date"],
        "sensitivity_start_week_end_date": resolved_start,
        "sensitivity_start_index": int(slice_metadata["start_index"]),
    }
    return manifest_row


def materialize_sensitivity_experiments(
    source_manifest_path: str | Path,
    output_root: str | Path,
    start_dates: list[str],
    experiment_names: set[str] | None = None,
    overwrite: bool = False,
) -> Path:
    source_rows = read_csv_manifest(_repo_path(source_manifest_path))
    output_root_path = _repo_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    selected_rows = [
        row
        for row in source_rows
        if experiment_names is None or row.get("experiment_name") in experiment_names
    ]
    if not selected_rows:
        raise ValueError("No source experiments matched the requested filters.")

    manifest_rows: list[dict[str, Any]] = []
    for source_row in selected_rows:
        for start_date in start_dates:
            manifest_rows.append(
                _save_sliced_experiment(
                    source_row=source_row,
                    output_root=output_root_path,
                    start_date=start_date,
                    overwrite=overwrite,
                )
            )
    manifest_path = output_root_path / "generation_manifest.csv"
    write_csv_manifest(manifest_path, manifest_rows)
    return manifest_path


def write_sensitivity_fit_spec(
    output_root: str | Path,
    latent_ranks: list[int],
    steps: int,
    tol: float,
    seed: int,
    n_starts: int = 1,
    lambda_nuclear_values: list[float] | None = None,
) -> Path:
    output_root_path = _repo_path(output_root)
    if not latent_ranks:
        raise ValueError("At least one latent rank is required.")
    if any(rank < 0 for rank in latent_ranks):
        raise ValueError("Latent ranks must be nonnegative.")
    lambda_nuclear_values = list(lambda_nuclear_values or [])
    if any(value < 0.0 for value in lambda_nuclear_values):
        raise ValueError("lambda_nuclear values must be nonnegative.")
    variants: list[dict[str, Any]] = []
    for latent_rank in latent_ranks:
        variants.append(
            {
                "name": f"rank_{int(latent_rank)}",
                "optimizer_mode": "manifold",
                "latent_rank": int(latent_rank),
                "lambda_nuclear": 0.0,
                "lambda_uv_ridge": 0.0,
            }
        )
    for lambda_value in lambda_nuclear_values:
        lambda_label = ("%g" % float(lambda_value)).replace(".", "p").replace("-", "m")
        variants.append(
            {
                "name": f"nuclear_lambda_{lambda_label}",
                "optimizer_mode": "nuclear_norm",
                "latent_rank": 0,
                "lambda_nuclear": float(lambda_value),
                "lambda_uv_ridge": 0.0,
            }
        )
    spec = OmegaConf.create(
        {
            "base": {
                "fit_root_name": "fits",
                "fit_manifest_path": str(
                    (output_root_path / "fit_manifest.csv").resolve()
                ),
                "optimizer": {
                    "steps": int(steps),
                    "tol": float(tol),
                    "seed": int(seed),
                    "n_starts": int(n_starts),
                },
                "latent_rank": int(latent_ranks[0]),
                "optimizer_mode": "manifold",
                "lambda_nuclear": 0.0,
                "lambda_uv_ridge": 0.0,
                "estimation": {
                    "fixed_scalar_params": {},
                },
            },
            "variants": variants,
        }
    )
    spec_path = output_root_path / "fits_sensitivity_spec.yaml"
    OmegaConf.save(spec, spec_path)
    return spec_path


def _summary_value(
    entries: dict[str, dict[str, float | None]], name: str
) -> float | None:
    return entries.get(name, {}).get("estimate")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6f}"
    return str(value)


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# USCountyVaccination Sensitivity Summary\n\n")
        handle.write(
            "Rows are ranked by lowest MPLE final loss because these real-data "
            "experiments do not have truth parameters.\n\n"
        )
        handle.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(_fmt(row.get(column, "")) for column in SUMMARY_COLUMNS)
                + " |\n"
            )


def write_sensitivity_summary(fit_manifest_path: str | Path) -> Path:
    fit_manifest = _repo_path(fit_manifest_path)
    rows: list[dict[str, Any]] = []
    for fit_row in read_csv_manifest(fit_manifest):
        fit_root = _repo_path(fit_row["fit_path"])
        summary_path = fit_root / "mple_summary.csv"
        if not summary_path.exists():
            continue
        entries = read_summary_entries(summary_path)
        metadata = _load_yaml_mapping(
            _repo_path(fit_row["experiment_path"]) / "experiment_metadata.yaml"
        )
        row: dict[str, Any] = {
            "experiment_name": fit_row.get("experiment_name", ""),
            "parent_experiment_name": metadata.get(
                "sensitivity_parent_experiment_name", ""
            ),
            "sensitivity_start_date": metadata.get(
                "sensitivity_requested_start_date", ""
            ),
            "sensitivity_start_week_end_date": metadata.get(
                "sensitivity_start_week_end_date", ""
            ),
            "sensitivity_start_index": metadata.get("sensitivity_start_index", ""),
            "variant_name": fit_row.get("variant_name", ""),
            "optimizer_mode": fit_row.get("optimizer_mode", "manifold"),
            "latent_rank": fit_row.get("latent_rank", ""),
            "lambda_nuclear": fit_row.get("lambda_nuclear", ""),
            "lambda_uv_ridge": fit_row.get("lambda_uv_ridge", ""),
            "T": fit_row.get("T", ""),
            "s": fit_row.get("s", ""),
            "final_loss": _summary_value(entries, "final_loss"),
            "optimizer_status": parse_optimizer_status(fit_root / "mple.log"),
            "beta": _summary_value(entries, "beta"),
            "xi": _summary_value(entries, "xi"),
            "eta": _summary_value(entries, "eta"),
            "fit_path": str(fit_root.resolve()),
        }
        row.update(latent_diagnostics(fit_root))
        rows.append(row)

    if not rows:
        raise ValueError(f"No finished fits were found in {fit_manifest}.")
    rows.sort(
        key=lambda row: (
            math.inf if row.get("final_loss") is None else float(row["final_loss"]),
            str(row.get("sensitivity_start_week_end_date", "")),
            str(row.get("optimizer_mode", "manifold")),
            int(row.get("latent_rank") or 0),
            float(row.get("lambda_nuclear") or 0.0),
            float(row.get("lambda_uv_ridge") or 0.0),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank_in_sensitivity"] = index

    csv_path = fit_manifest.parent / "sensitivity_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [
                {column: row.get(column, "") for column in SUMMARY_COLUMNS}
                for row in rows
            ]
        )
    write_markdown_summary(fit_manifest.parent / "sensitivity_summary.md", rows)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create USCountyVaccination start-week sensitivity experiments and "
            "optionally fit a latent-rank grid."
        )
    )
    parser.add_argument(
        "--source_manifest_path",
        default="experiments/USCountyVaccination_US_trimmed/generation_manifest.csv",
        help="Existing USCountyVaccination generation manifest to slice.",
    )
    parser.add_argument(
        "--output_root",
        default="experiments/USCountyVaccination_US_sensitivity",
        help="Output root for derived experiments, generated spec, and summaries.",
    )
    parser.add_argument(
        "--experiment_names",
        nargs="*",
        default=None,
        help="Optional source experiment_name values to include. Defaults to all rows.",
    )
    parser.add_argument("--start_dates", nargs="+", default=list(DEFAULT_START_DATES))
    parser.add_argument(
        "--latent_ranks",
        nargs="+",
        type=int,
        default=list(DEFAULT_LATENT_RANKS),
    )
    parser.add_argument(
        "--lambda_nuclear_values",
        nargs="*",
        type=float,
        default=list(DEFAULT_LAMBDA_NUCLEAR_VALUES),
        help="Optional nuclear-norm penalty values to add as convex-relaxation fit variants.",
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_starts",
        type=int,
        default=5,
        help="Number of random starts for each fit variant.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run_fits",
        action="store_true",
        help="Run MPLE fits after materializing the sensitivity experiments.",
    )
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Only rebuild sensitivity_summary.csv/md from an existing fit manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _repo_path(args.output_root)
    if args.summarize_only:
        summary_path = write_sensitivity_summary(output_root / "fit_manifest.csv")
        print(f"Sensitivity summary: {summary_path}")
        return

    manifest_path = materialize_sensitivity_experiments(
        source_manifest_path=args.source_manifest_path,
        output_root=output_root,
        start_dates=list(args.start_dates),
        experiment_names=set(args.experiment_names) if args.experiment_names else None,
        overwrite=bool(args.overwrite),
    )
    spec_path = write_sensitivity_fit_spec(
        output_root=output_root,
        latent_ranks=list(args.latent_ranks),
        steps=int(args.steps),
        tol=float(args.tol),
        seed=int(args.seed),
        n_starts=int(args.n_starts),
        lambda_nuclear_values=list(args.lambda_nuclear_values),
    )
    print(f"Sensitivity generation manifest: {manifest_path}")
    print(f"Sensitivity fit spec: {spec_path}")
    if args.run_fits:
        fit_manifest = run_fits(
            manifest_path, spec_path, overwrite=bool(args.overwrite)
        )
        summary_path = write_sensitivity_summary(fit_manifest)
        print(f"Sensitivity fit manifest: {fit_manifest}")
        print(f"Sensitivity summary: {summary_path}")
    else:
        print("Fits were not run. Add --run_fits when you are ready to launch MPLE.")


if __name__ == "__main__":
    main()
