"""Summarize synthetic MPLE runs with concise experiment-level diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


SCALAR_NAMES = ("beta", "xi", "eta", "zeta", "psi")
METRIC_NAMES = (
    "final_loss",
    "field_rmse",
    "static_field_rmse",
    "interaction_fro_error",
    "tau_rmse",
)


def read_manifest(manifest_path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_summary_entries(summary_path: Path) -> dict[str, dict[str, float | None]]:
    entries: dict[str, dict[str, float | None]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entries[row["name"]] = {
                "estimate": float(row["estimate"]) if row["estimate"] else None,
                "true": float(row["true"]) if row["true"] else None,
                "squared_error": (
                    float(row["squared_error"]) if row["squared_error"] else None
                ),
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


def field_diagnostics(
    folder: Path, field_mode: str, bound: float | None, latent_rank: int | None
) -> dict[str, object]:
    estimated_field = load_field_matrix(folder / "estimated_field_artifacts.npz")
    true_field = load_field_matrix(folder / "true_field_artifacts.npz")
    row: dict[str, object] = {}

    if field_mode == "latent_feature_matrix" and estimated_field is not None:
        est_norm = float(np.linalg.norm(estimated_field, ord=np.inf))
        row["estimated_field_inf_norm"] = est_norm
        if bound is not None:
            row["bound_B"] = bound
        if true_field is not None:
            row["true_field_inf_norm"] = float(np.linalg.norm(true_field, ord=np.inf))
        est_rank = int(np.linalg.matrix_rank(estimated_field))
        row["estimated_field_rank"] = est_rank
        row["latent_rank_cap"] = latent_rank if latent_rank is not None else ""
        if true_field is not None:
            row["true_field_rank"] = int(np.linalg.matrix_rank(true_field))
    return row


def collect_rows(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for folder in read_manifest(manifest_path):
        summary_path = folder / "mple_summary.csv"
        if not summary_path.exists():
            continue

        summary_entries = read_summary_entries(summary_path)
        metadata = (
            OmegaConf.load(folder / "experiment_metadata.yaml")
            if (folder / "experiment_metadata.yaml").exists()
            else OmegaConf.create({})
        )
        config = (
            OmegaConf.load(folder / "realized_config.yaml")
            if (folder / "realized_config.yaml").exists()
            else OmegaConf.create({})
        )

        field_mode = str(metadata.get("field_mode", ""))
        latent_rank = (
            int(metadata.get("latent_rank"))
            if metadata.get("latent_rank") not in (None, "")
            else None
        )
        bound = (
            float(config.global_params.B)
            if "global_params" in config and "B" in config.global_params
            else None
        )
        row: dict[str, object] = {
            "descriptor": str(metadata.get("descriptor", folder.name)),
            "field_mode": field_mode,
            "N": (
                int(config.global_params.N)
                if "global_params" in config and "N" in config.global_params
                else ""
            ),
            "T": (
                int(config.global_params.T)
                if "global_params" in config and "T" in config.global_params
                else ""
            ),
            "s": (
                int(config.global_params.s)
                if "global_params" in config and "s" in config.global_params
                else ""
            ),
            "optimizer_status": parse_optimizer_status(folder / "mple.log"),
        }

        for metric_name in METRIC_NAMES:
            row[metric_name] = scalar_value(summary_entries, metric_name, "estimate")

        for scalar_name in SCALAR_NAMES:
            est = scalar_value(summary_entries, scalar_name, "estimate")
            true = scalar_value(summary_entries, scalar_name, "true")
            row[f"{scalar_name}_abs_error"] = (
                abs(est - true) if est is not None and true is not None else None
            )

        row.update(field_diagnostics(folder, field_mode, bound, latent_rank))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row.get("field_mode", "")),
            int(row.get("N") or 0),
            int(row.get("T") or 0),
            str(row.get("descriptor", "")),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished experiments were found in the manifest.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def aggregate_by_field_mode(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("field_mode", "")), []).append(row)

    aggregates: list[dict[str, object]] = []
    for field_mode, mode_rows in sorted(by_mode.items()):
        field_rmses = [
            float(row["field_rmse"])
            for row in mode_rows
            if isinstance(row.get("field_rmse"), float)
        ]
        interaction_errors = [
            float(row["interaction_fro_error"])
            for row in mode_rows
            if isinstance(row.get("interaction_fro_error"), float)
        ]
        xi_errors = [
            float(row["xi_abs_error"])
            for row in mode_rows
            if isinstance(row.get("xi_abs_error"), float)
        ]
        beta_errors = [
            float(row["beta_abs_error"])
            for row in mode_rows
            if isinstance(row.get("beta_abs_error"), float)
        ]
        aggregates.append(
            {
                "field_mode": field_mode,
                "count": len(mode_rows),
                "median_field_rmse": median(field_rmses),
                "median_interaction_fro_error": median(interaction_errors),
                "median_beta_abs_error": median(beta_errors),
                "median_xi_abs_error": median(xi_errors),
            }
        )
    return aggregates


def write_table(handle, rows: list[dict[str, object]], headers: list[str]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(fmt(row.get(key, "")) for key in headers) + " |\n")
    handle.write("\n")


def select_columns(rows: list[dict[str, object]], headers: list[str]) -> list[dict[str, object]]:
    return [{key: row.get(key, "") for key in headers} for row in rows]


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No finished experiments were found in the manifest.")

    overview = aggregate_by_field_mode(rows)
    scalar_headers = [
        "descriptor",
        "field_mode",
        "N",
        "T",
        "final_loss",
        "beta_abs_error",
        "xi_abs_error",
        "eta_abs_error",
        "zeta_abs_error",
        "psi_abs_error",
        "optimizer_status",
    ]
    recovery_headers = [
        "descriptor",
        "field_mode",
        "N",
        "T",
        "field_rmse",
        "static_field_rmse",
        "interaction_fro_error",
        "tau_rmse",
    ]
    latent_headers = [
        "descriptor",
        "N",
        "T",
        "estimated_field_inf_norm",
        "bound_B",
        "estimated_field_rank",
        "latent_rank_cap",
        "true_field_rank",
    ]
    overview_headers = [
        "field_mode",
        "count",
        "median_field_rmse",
        "median_interaction_fro_error",
        "median_beta_abs_error",
        "median_xi_abs_error",
    ]
    latent_rows = [row for row in rows if row.get("field_mode") == "latent_feature_matrix"]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic Experiment Summary\n\n")
        handle.write(
            "This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, "
            "interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.\n\n"
        )
        handle.write(f"- Completed experiments: {len(rows)}\n")
        handle.write(
            f"- Field modes covered: {', '.join(sorted({str(row['field_mode']) for row in rows}))}\n\n"
        )
        handle.write("## By Field Mode\n\n")
        write_table(handle, overview, overview_headers)
        handle.write("## Scalar Recovery\n\n")
        write_table(handle, select_columns(rows, scalar_headers), scalar_headers)
        handle.write("## Field And Interaction Recovery\n\n")
        write_table(handle, select_columns(rows, recovery_headers), recovery_headers)
        if latent_rows:
            handle.write("## Latent Diagnostics\n\n")
            write_table(handle, select_columns(latent_rows, latent_headers), latent_headers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize synthetic MPLE runs from a manifest."
    )
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--report_stem", required=True, type=str)
    args = parser.parse_args()

    rows = collect_rows(Path(args.manifest))
    report_stem = Path(args.report_stem)
    report_stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(f"{args.report_stem}.csv"), rows)
    write_markdown(Path(f"{args.report_stem}.md"), rows)


if __name__ == "__main__":
    main()
