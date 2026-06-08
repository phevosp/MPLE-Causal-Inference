"""Aggregate MPLE fit results into per-experiment and cross-experiment summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.t0_csv_utils import read_csv_rows, write_csv
from utils.t3_field_operations import latent_field_bound_norm
from utils.t8_output_writers import _as_float, _metric_or_inf


SCALAR_NAMES = ("beta", "xi", "eta")
METRIC_NAMES = ("final_loss", "field_rmse", "interaction_fro_error")
SINGULAR_VALUE_TOP_K = 5


def _sv_columns(prefix: str) -> list[str]:
    return [f"{prefix}_sv_{index}" for index in range(1, SINGULAR_VALUE_TOP_K + 1)]


def _field_diagnostic_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_field_max_abs_entry",
        f"{prefix}_field_rank",
        f"{prefix}_field_frobenius_norm",
        f"{prefix}_field_nuclear_norm",
        f"{prefix}_singular_value_count",
        f"{prefix}_u_frobenius_norm",
        f"{prefix}_v_frobenius_norm",
        *_sv_columns(prefix),
    ]


PER_EXPERIMENT_COLUMNS = [
    "experiment_name",
    "descriptor",
    "variant_name",
    "variant_slug",
    "optimizer_mode",
    "field_mode",
    "latent_rank",
    "lambda_nuclear",
    "lambda_frobenius",
    "lambda_uv_ridge",
    "fixed_scalar_params",
    "ranking_mode",
    "rank_in_experiment",
    "is_best",
    "total_recovery_rmse",
    "final_loss",
    "field_rmse",
    "interaction_fro_error",
    "optimizer_status",
    "beta_abs_error",
    "xi_abs_error",
    "eta_abs_error",
    *_field_diagnostic_columns("estimated"),
    *_field_diagnostic_columns("true"),
]
_PER_EXPERIMENT_COLUMNS_NO_TRUTH = [
    "experiment_name",
    "descriptor",
    "variant_name",
    "variant_slug",
    "optimizer_mode",
    "field_mode",
    "latent_rank",
    "lambda_nuclear",
    "lambda_frobenius",
    "lambda_uv_ridge",
    "fixed_scalar_params",
    "ranking_mode",
    "rank_in_experiment",
    "is_best",
    "total_recovery_rmse",
    "final_loss",
    "field_rmse",
    "interaction_fro_error",
    "optimizer_status",
    "beta_estimate",
    "xi_estimate",
    "eta_estimate",
    *_field_diagnostic_columns("estimated"),
]
WINNER_COLUMNS = [
    "experiment_name",
    "descriptor",
    "intervention_source",
    "graph_source",
    "field_mode",
    "N",
    "T",
    "s",
    "variant_name",
    "variant_slug",
    "optimizer_mode",
    "latent_rank",
    "lambda_nuclear",
    "lambda_frobenius",
    "lambda_uv_ridge",
    "fixed_scalar_params",
    "ranking_mode",
    "total_recovery_rmse",
    "final_loss",
    "field_rmse",
    "interaction_fro_error",
    "optimizer_status",
    *_field_diagnostic_columns("estimated"),
    *_field_diagnostic_columns("true"),
]


def _per_experiment_columns(has_truth: bool) -> list[str]:
    return PER_EXPERIMENT_COLUMNS if has_truth else _PER_EXPERIMENT_COLUMNS_NO_TRUTH


def total_recovery_rmse(row: dict[str, object]) -> float | None:
    field_rmse = _as_float(row.get("field_rmse"))
    if field_rmse is None:
        return None
    total = field_rmse
    for scalar_name in SCALAR_NAMES:
        scalar_error = _as_float(row.get(f"{scalar_name}_abs_error"))
        if scalar_error is not None:
            total += scalar_error
    return total


def read_summary_entries(summary_path: Path) -> dict[str, dict[str, float | None]]:
    entries: dict[str, dict[str, float | None]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entries[row["name"]] = {
                "estimate": _as_float(row["estimate"]),
                "true": _as_float(row["true"]),
                "squared_error": _as_float(row["squared_error"]),
            }
    return entries


def load_field_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "field_matrix" not in data:
            return None
        return np.asarray(data["field_matrix"], dtype=float)


def parse_optimizer_status(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    status = ""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "Optimizer status:" in line:
            status = line.split("Optimizer status:", 1)[1].strip()
    return status


def scalar_value(
    summary_entries: dict[str, dict[str, float | None]],
    name: str,
    key: str,
) -> float | None:
    return summary_entries.get(name, {}).get(key)


def svd_field_diagnostics(prefix: str, field_matrix: np.ndarray) -> dict[str, object]:
    singular_values = np.linalg.svd(field_matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(field_matrix))
    row: dict[str, object] = {
        f"{prefix}_field_max_abs_entry": latent_field_bound_norm(field_matrix),
        f"{prefix}_field_rank": rank,
        f"{prefix}_field_frobenius_norm": float(np.linalg.norm(field_matrix, ord="fro")),
        f"{prefix}_field_nuclear_norm": float(np.sum(singular_values)),
        f"{prefix}_singular_value_count": rank,
        f"{prefix}_u_frobenius_norm": float(np.sqrt(rank)),
        f"{prefix}_v_frobenius_norm": float(np.sqrt(rank)),
    }
    for index, column in enumerate(_sv_columns(prefix)):
        if index < rank:
            row[column] = float(singular_values[index])
    return row


def latent_diagnostics(folder: Path) -> dict[str, object]:
    estimated_field = load_field_matrix(folder / "estimated_field_artifacts.npz")
    true_field = load_field_matrix(folder / "true_field_artifacts.npz")
    if estimated_field is None:
        summary_entries = read_summary_entries(folder / "mple_summary.csv")
        row: dict[str, object] = {}
        estimated_value = scalar_value(
            summary_entries, "estimated_field_max_abs_entry", "estimate"
        )
        if estimated_value is None:
            estimated_value = scalar_value(
                summary_entries, "estimated_field_inf_norm", "estimate"
            )
        true_value = scalar_value(
            summary_entries, "true_field_max_abs_entry", "estimate"
        )
        if true_value is None:
            true_value = scalar_value(summary_entries, "true_field_inf_norm", "estimate")
        if estimated_value is not None:
            row["estimated_field_max_abs_entry"] = estimated_value
        if true_value is not None:
            row["true_field_max_abs_entry"] = true_value
        return row
    row = svd_field_diagnostics("estimated", estimated_field)
    if true_field is not None:
        row.update(svd_field_diagnostics("true", true_field))
    return row


def _fit_row_from_manifest(manifest_row: dict[str, str]) -> dict[str, object] | None:
    fit_root = Path(manifest_row["fit_path"])
    summary_path = fit_root / "mple_summary.csv"
    if not summary_path.exists():
        return None

    summary_entries = read_summary_entries(summary_path)
    row: dict[str, object] = {
        "experiment_name": manifest_row.get("experiment_name", ""),
        "experiment_path": manifest_row.get("experiment_path", ""),
        "descriptor": manifest_row.get("descriptor", ""),
        "intervention_source": manifest_row.get("intervention_source", ""),
        "graph_source": manifest_row.get("graph_source", ""),
        "field_mode": manifest_row.get("field_mode", ""),
        "variant_name": manifest_row.get("variant_name", ""),
        "variant_slug": manifest_row.get("variant_slug", ""),
        "fit_path": str(fit_root.resolve()),
        "N": manifest_row.get("N", ""),
        "T": manifest_row.get("T", ""),
        "s": manifest_row.get("s", ""),
        "optimizer_mode": manifest_row.get("optimizer_mode", "no_external_field"),
        "latent_rank": (
            int(manifest_row["latent_rank"])
            if manifest_row.get("latent_rank") not in (None, "")
            else ""
        ),
        "lambda_nuclear": _as_float(manifest_row.get("lambda_nuclear")),
        "lambda_frobenius": _as_float(manifest_row.get("lambda_frobenius")),
        "lambda_uv_ridge": _as_float(manifest_row.get("lambda_uv_ridge")),
        "fixed_scalar_params": manifest_row.get("fixed_scalar_params", ""),
        "optimizer_status": parse_optimizer_status(fit_root / "mple.log"),
    }
    for metric_name in METRIC_NAMES:
        row[metric_name] = scalar_value(summary_entries, metric_name, "estimate")
    for scalar_name in SCALAR_NAMES:
        est = scalar_value(summary_entries, scalar_name, "estimate")
        true = scalar_value(summary_entries, scalar_name, "true")
        row[f"{scalar_name}_estimate"] = est
        row[f"{scalar_name}_abs_error"] = (
            abs(est - true) if est is not None and true is not None else None
        )
    row.update(latent_diagnostics(fit_root))
    return row


def collect_fit_rows(manifest_path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in read_csv_rows(manifest_path):
        fit_row = _fit_row_from_manifest(manifest_row)
        if fit_row is not None:
            rows.append(fit_row)
    rows.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("variant_name", "")),
        )
    )
    return rows


def _group_has_truth(rows: list[dict[str, object]]) -> bool:
    return any(
        row.get("field_rmse") is not None
        or any(row.get(f"{s}_abs_error") is not None for s in SCALAR_NAMES)
        for row in rows
    )


def ranking_key(row: dict[str, object], use_truth_metrics: bool) -> tuple[float, ...]:
    if use_truth_metrics:
        return (
            _metric_or_inf(total_recovery_rmse(row)),
            _metric_or_inf(row.get("field_rmse")),
            _metric_or_inf(row.get("interaction_fro_error")),
            _metric_or_inf(row.get("final_loss")),
        )
    return (_metric_or_inf(row.get("final_loss")),)


def rank_rows_within_experiment(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    use_truth_metrics = _group_has_truth(rows)
    ranking_mode = "total_recovery_rmse" if use_truth_metrics else "final_loss_only"
    ordered = sorted(
        rows,
        key=lambda row: (
            ranking_key(row, use_truth_metrics),
            str(row.get("variant_name", "")),
        ),
    )
    ranked_rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        ranked = dict(row)
        ranked["ranking_mode"] = ranking_mode
        ranked["rank_in_experiment"] = index
        ranked["is_best"] = index == 1
        ranked["total_recovery_rmse"] = total_recovery_rmse(ranked)
        ranked_rows.append(ranked)
    return ranked_rows


def group_and_rank_fit_rows(
    rows: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_path"])].append(row)

    ranked_groups: dict[str, list[dict[str, object]]] = {}
    winners: list[dict[str, object]] = []
    for experiment_path, group_rows in grouped.items():
        ranked = rank_rows_within_experiment(group_rows)
        ranked_groups[experiment_path] = ranked
        winners.append(dict(ranked[0]))
    winners.sort(
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("variant_name", "")),
        )
    )
    return ranked_groups, winners


def write_per_experiment_summary(
    experiment_path: str | Path,
    rows: list[dict[str, object]],
    *,
    filename: str = "fit_summary.csv",
) -> Path:
    has_truth = _group_has_truth(rows)
    columns = _per_experiment_columns(has_truth)
    experiment_root = Path(experiment_path)
    csv_path = experiment_root / str(filename)
    write_csv(csv_path, rows, columns)
    return csv_path


def write_cross_experiment_summary(
    manifest_path: str | Path,
    winner_rows: list[dict[str, object]],
    *,
    filename: str = "best_fit_by_experiment.csv",
) -> Path:
    manifest_root = Path(manifest_path).resolve().parent
    csv_path = manifest_root / str(filename)
    write_csv(csv_path, winner_rows, WINNER_COLUMNS)
    return csv_path


def write_fit_reports(
    manifest_path: str | Path,
    *,
    per_experiment_filename: str = "fit_summary.csv",
    winners_filename: str = "best_fit_by_experiment.csv",
) -> dict[str, object]:
    rows = collect_fit_rows(manifest_path)
    if not rows:
        raise ValueError(f"No finished fits were found in manifest {manifest_path}.")

    ranked_groups, winners = group_and_rank_fit_rows(rows)
    per_experiment_outputs: dict[str, dict[str, str]] = {}
    for experiment_path, ranked_rows in ranked_groups.items():
        csv_path = write_per_experiment_summary(
            experiment_path,
            ranked_rows,
            filename=per_experiment_filename,
        )
        per_experiment_outputs[experiment_path] = {
            "csv": str(csv_path),
        }
    winners_csv = write_cross_experiment_summary(
        manifest_path,
        winners,
        filename=winners_filename,
    )
    return {
        "per_experiment": per_experiment_outputs,
        "winners_csv": str(winners_csv),
    }
