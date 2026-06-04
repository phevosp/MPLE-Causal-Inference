"""Report latent-field recovery diagnostics for synthetic MPLE experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from utils.t1_matrix_io import load_gamma_matrix
from utils.t0_path_utils import first_existing_path
from utils.t3_interaction_matrices import interaction_term
from utils.t3_model_artifacts import load_model_artifacts

_DEGENERACY_THRESHOLD = 1e-12  # denominators below this are treated as degenerate


DIAGNOSTIC_FIELDNAMES = [
    "experiment_path",
    "experiment_name",
    "truth_beta",
    "truth_xi",
    "truth_eta",
    "feature_rms_field",
    "feature_rms_beta_z",
    "feature_rms_eta_prev_x",
    "feature_rms_xi_gamma_x",
    "oracle_loss_true_field",
    "oracle_loss_zero_field",
    "true_field_rms",
    "true_field_max_abs",
    "true_field_rank",
    "true_field_fro_norm",
    "true_field_nuclear_norm",
    "true_field_spectral_norm",
    "true_field_top_singulars",
    "fit_path",
    "fit_name",
    "fit_summary_exists",
    "fit_has_estimated_field",
    "fit_final_loss",
    "estimated_field_rms",
    "estimated_field_max_abs",
    "estimated_field_rank",
    "estimated_field_fro_norm",
    "estimated_field_nuclear_norm",
    "estimated_field_spectral_norm",
    "estimated_field_top_singulars",
    "field_rmse",
    "relative_fro_error",
    "field_correlation",
    "field_cosine_alignment",
    "oracle_loss_estimated_field",
]


def _load_field(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "field_matrix" not in data:
            return None
        return np.asarray(data["field_matrix"], dtype=float)


def _field_stats(prefix: str, field: np.ndarray | None) -> dict[str, object]:
    if field is None:
        return {}
    singular_values = np.linalg.svd(field, compute_uv=False)
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(field * field))),
        f"{prefix}_max_abs": float(np.max(np.abs(field))),
        f"{prefix}_rank": int(np.linalg.matrix_rank(field)),
        f"{prefix}_fro_norm": float(np.linalg.norm(field, ord="fro")),
        f"{prefix}_nuclear_norm": float(singular_values.sum()),
        f"{prefix}_spectral_norm": float(singular_values[0])
        if singular_values.size
        else 0.0,
        f"{prefix}_top_singulars": ";".join(
            f"{value:.12g}" for value in singular_values[:10]
        ),
    }


def _field_alignment(true_field: np.ndarray, estimated_field: np.ndarray) -> dict[str, float]:
    """Compute RMSE, relative Frobenius error, Pearson correlation, and cosine similarity between two fields."""
    error = estimated_field - true_field
    true_flat = true_field.reshape(-1)
    est_flat = estimated_field.reshape(-1)
    centered_true = true_flat - true_flat.mean()
    centered_est = est_flat - est_flat.mean()
    corr_denom = np.linalg.norm(centered_true) * np.linalg.norm(centered_est)
    cosine_denom = np.linalg.norm(true_flat) * np.linalg.norm(est_flat)
    return {
        "field_rmse": float(np.sqrt(np.mean(error * error))),
        "relative_fro_error": float(
            np.linalg.norm(error, ord="fro") / np.linalg.norm(true_field, ord="fro")
        ),
        "field_correlation": float(np.dot(centered_true, centered_est) / corr_denom)
        if corr_denom
        else np.nan,
        "field_cosine_alignment": float(np.dot(true_flat, est_flat) / cosine_denom)
        if cosine_denom
        else np.nan,
    }


def _outcome_loss(
    field: np.ndarray,
    *,
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    beta: float,
    xi: float,
    eta: float,
    interaction_term_x: np.ndarray,
) -> float:
    prev_x = np.vstack([x_0, x[:-1, :]])
    h_x = field + beta * z + eta * prev_x + interaction_term_x
    return float(np.mean(np.logaddexp(h_x, -h_x) - x * h_x))


def _read_summary_value(summary_path: Path, name: str) -> float | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("name") == name and row.get("estimate"):
                return float(row["estimate"])
    return None


def _fit_dirs(experiment_root: Path, explicit_fit_path: Path | None) -> list[Path | None]:
    if explicit_fit_path is not None:
        return [explicit_fit_path]
    fits_root = experiment_root / "fits"
    if not fits_root.exists():
        return [None]
    return [None, *sorted(path for path in fits_root.iterdir() if path.is_dir())]


def build_diagnostic_rows(
    experiment_path: str | Path, fit_path: str | Path | None = None
) -> list[dict[str, object]]:
    experiment_root = Path(experiment_path)
    config = OmegaConf.load(
        first_existing_path(
            experiment_root / "generation_realized_config.yaml",
            experiment_root / "realized_config.yaml",
        )
    )
    artifacts = load_model_artifacts(experiment_root)
    true_field = np.asarray(artifacts.field_matrix, dtype=float)
    gamma_matrix = load_gamma_matrix(experiment_root)
    with np.load(experiment_root / "panel_data.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    x_0 = np.asarray(np.load(experiment_root / "x_0.npy"), dtype=float)
    interaction_term_x = interaction_term(x, float(config.estimation_params.xi), gamma_matrix)
    beta = float(config.estimation_params.beta)
    xi = float(config.estimation_params.xi)
    eta = float(config.estimation_params.eta)
    prev_x = np.vstack([x_0, x[:-1, :]])
    zero_field = np.zeros_like(true_field)

    base: dict[str, object] = {
        "experiment_path": str(experiment_root.resolve()),
        "experiment_name": experiment_root.name,
        "truth_beta": beta,
        "truth_xi": xi,
        "truth_eta": eta,
        "feature_rms_field": float(np.sqrt(np.mean(true_field * true_field))),
        "feature_rms_beta_z": float(np.sqrt(np.mean((beta * z) ** 2))),
        "feature_rms_eta_prev_x": float(np.sqrt(np.mean((eta * prev_x) ** 2))),
        "feature_rms_xi_gamma_x": float(np.sqrt(np.mean(interaction_term_x**2))),
        "oracle_loss_true_field": _outcome_loss(
            true_field,
            x=x,
            z=z,
            x_0=x_0,
            beta=beta,
            xi=xi,
            eta=eta,
            interaction_term_x=interaction_term_x,
        ),
        "oracle_loss_zero_field": _outcome_loss(
            zero_field,
            x=x,
            z=z,
            x_0=x_0,
            beta=beta,
            xi=xi,
            eta=eta,
            interaction_term_x=interaction_term_x,
        ),
        **_field_stats("true_field", true_field),
    }

    rows: list[dict[str, object]] = []
    for current_fit_path in _fit_dirs(
        experiment_root, None if fit_path is None else Path(fit_path)
    ):
        row = dict(base)
        if current_fit_path is None:
            row["fit_path"] = ""
            row["fit_name"] = "truth_only"
            rows.append(row)
            continue
        estimated_field = _load_field(current_fit_path / "estimated_field_artifacts.npz")
        summary_path = current_fit_path / "mple_summary.csv"
        row["fit_path"] = str(current_fit_path.resolve())
        row["fit_name"] = current_fit_path.name
        row["fit_summary_exists"] = bool(summary_path.exists())
        row["fit_has_estimated_field"] = estimated_field is not None
        row["fit_final_loss"] = _read_summary_value(summary_path, "final_loss")
        row.update(_field_stats("estimated_field", estimated_field))
        if estimated_field is not None:
            row.update(_field_alignment(true_field, estimated_field))
            row["oracle_loss_estimated_field"] = _outcome_loss(
                estimated_field,
                x=x,
                z=z,
                x_0=x_0,
                beta=beta,
                xi=xi,
                eta=eta,
                interaction_term_x=interaction_term_x,
            )
        rows.append(row)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(DIAGNOSTIC_FIELDNAMES)
    seen: set[str] = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write latent-field recovery diagnostics for one experiment."
    )
    parser.add_argument("--experiment_path", required=True)
    parser.add_argument("--fit_path", default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    rows = build_diagnostic_rows(args.experiment_path, fit_path=args.fit_path)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.experiment_path) / "latent_recovery_diagnostics.csv"
    )
    write_rows(output_path, rows)
    print(f"Wrote latent recovery diagnostics to {output_path}")


if __name__ == "__main__":
    main()

