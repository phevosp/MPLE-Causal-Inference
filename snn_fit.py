"""Run one standard SNN fit and write SNN-specific artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from utils.t0_csv_utils import write_csv_rows
from utils.t0_path_utils import io_path
from utils.t8_snn_core import SyntheticNearestNeighbors


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path.resolve()))
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(io_path(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _snn_params_from_config(config: object) -> dict[str, Any]:
    raw = getattr(config, "snn_params", None)
    params = (
        {}
        if raw in (None, "")
        else OmegaConf.to_container(raw, resolve=True)  # type: ignore[arg-type]
    )
    if not isinstance(params, dict):
        raise ValueError("fit_realized_config.yaml snn_params must be a mapping.")
    return {
        "n_neighbors": int(params.get("n_neighbors", 1)),
        "weights": str(params.get("weights", "uniform")),
        "random_splits": bool(params.get("random_splits", False)),
        "max_rank": (
            None if params.get("max_rank", None) is None else int(params["max_rank"])
        ),
        "spectral_t": (
            None
            if params.get("spectral_t", None) is None
            else float(params["spectral_t"])
        ),
        "linear_span_eps": float(params.get("linear_span_eps", 0.1)),
        "subspace_eps": float(params.get("subspace_eps", 0.1)),
        "min_value": (
            None if params.get("min_value", None) is None else float(params["min_value"])
        ),
        "max_value": (
            None if params.get("max_value", None) is None else float(params["max_value"])
        ),
    }


def _build_treatment_split_matrices(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    treated_input = np.where(np.asarray(z, dtype=float) > 0.0, np.asarray(x, dtype=float), np.nan)
    untreated_input = np.where(
        np.asarray(z, dtype=float) < 0.0,
        np.asarray(x, dtype=float),
        np.nan,
    )
    return treated_input, untreated_input


def _run_completion(
    matrix: np.ndarray,
    *,
    snn_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    model = SyntheticNearestNeighbors(verbose=False, **snn_params)
    completed = np.asarray(model.fit_transform(matrix), dtype=float)
    feasible = np.asarray(model.feasible, dtype=float)
    return completed, feasible


def _optional_scalar(value: Any) -> float:
    return np.nan if value is None else float(value)


def _completion_stats(
    input_matrix: np.ndarray,
    completed_matrix: np.ndarray,
) -> dict[str, int]:
    target_missing_mask = np.isnan(np.asarray(input_matrix, dtype=float))
    completed_mask = np.isfinite(np.asarray(completed_matrix, dtype=float)) & target_missing_mask
    return {
        "num_observed": int(np.count_nonzero(~target_missing_mask)),
        "num_target_missing": int(np.count_nonzero(target_missing_mask)),
        "num_completed": int(np.count_nonzero(completed_mask)),
        "num_failed": int(np.count_nonzero(target_missing_mask) - np.count_nonzero(completed_mask)),
    }


def _completion_rate(num_completed: int, num_target_missing: int) -> float:
    if num_target_missing <= 0:
        return np.nan
    return float(num_completed) / float(num_target_missing)


def run_snn_fit(data_folder: str | Path) -> dict[str, Any]:
    fit_root = Path(data_folder)
    config = OmegaConf.load(io_path(fit_root / "fit_realized_config.yaml"))
    logger = _setup_logger(fit_root / "snn.log")
    snn_params = _snn_params_from_config(config)

    logger.info("Starting SNN fit in %s", fit_root)
    logger.info("Resolved SNN parameters: %s", snn_params)

    panel_path = Path(str(config.input_artifacts.panel_path))
    if not panel_path.exists():
        raise FileNotFoundError(f"panel_data.npz does not exist at {panel_path}.")
    with np.load(io_path(panel_path), allow_pickle=False) as panel:
        x = np.asarray(panel["x"], dtype=float)
        z = np.asarray(panel["z"], dtype=float)
    if x.shape != z.shape:
        raise ValueError(f"x shape {x.shape} does not match z shape {z.shape}.")

    logger.info("Loaded panel with shape T=%s, N=%s", x.shape[0], x.shape[1])

    treated_input, untreated_input = _build_treatment_split_matrices(x, z)
    treated_completed, treated_feasible = _run_completion(
        treated_input,
        snn_params=snn_params,
    )
    untreated_completed, untreated_feasible = _run_completion(
        untreated_input,
        snn_params=snn_params,
    )

    treated_stats = _completion_stats(treated_input, treated_completed)
    untreated_stats = _completion_stats(untreated_input, untreated_completed)
    total_target_missing = (
        treated_stats["num_target_missing"] + untreated_stats["num_target_missing"]
    )
    total_completed = treated_stats["num_completed"] + untreated_stats["num_completed"]

    logger.info(
        "Treated stats: observed=%s target_missing=%s completed=%s failed=%s",
        treated_stats["num_observed"],
        treated_stats["num_target_missing"],
        treated_stats["num_completed"],
        treated_stats["num_failed"],
    )
    logger.info(
        "Untreated stats: observed=%s target_missing=%s completed=%s failed=%s",
        untreated_stats["num_observed"],
        untreated_stats["num_target_missing"],
        untreated_stats["num_completed"],
        untreated_stats["num_failed"],
    )

    np.savez(
        io_path(fit_root / "estimated_snn_artifacts.npz"),
        treated_input_matrix=np.asarray(treated_input, dtype=float),
        untreated_input_matrix=np.asarray(untreated_input, dtype=float),
        treated_completed_matrix=np.asarray(treated_completed, dtype=float),
        untreated_completed_matrix=np.asarray(untreated_completed, dtype=float),
        treated_finite_mask=np.asarray(np.isfinite(treated_completed), dtype=bool),
        untreated_finite_mask=np.asarray(np.isfinite(untreated_completed), dtype=bool),
        treated_feasible_mask=np.asarray(treated_feasible, dtype=float),
        untreated_feasible_mask=np.asarray(untreated_feasible, dtype=float),
        n_neighbors=np.asarray(int(snn_params["n_neighbors"]), dtype=int),
        weights=np.asarray(str(snn_params["weights"])),
        random_splits=np.asarray(bool(snn_params["random_splits"]), dtype=bool),
        max_rank=np.asarray(_optional_scalar(snn_params["max_rank"]), dtype=float),
        spectral_t=np.asarray(_optional_scalar(snn_params["spectral_t"]), dtype=float),
        linear_span_eps=np.asarray(float(snn_params["linear_span_eps"]), dtype=float),
        subspace_eps=np.asarray(float(snn_params["subspace_eps"]), dtype=float),
        min_value=np.asarray(_optional_scalar(snn_params["min_value"]), dtype=float),
        max_value=np.asarray(_optional_scalar(snn_params["max_value"]), dtype=float),
        treated_num_observed=np.asarray(treated_stats["num_observed"], dtype=int),
        treated_num_target_missing=np.asarray(treated_stats["num_target_missing"], dtype=int),
        treated_num_completed=np.asarray(treated_stats["num_completed"], dtype=int),
        treated_num_failed=np.asarray(treated_stats["num_failed"], dtype=int),
        untreated_num_observed=np.asarray(untreated_stats["num_observed"], dtype=int),
        untreated_num_target_missing=np.asarray(
            untreated_stats["num_target_missing"], dtype=int
        ),
        untreated_num_completed=np.asarray(untreated_stats["num_completed"], dtype=int),
        untreated_num_failed=np.asarray(untreated_stats["num_failed"], dtype=int),
    )

    summary_rows = [
        {"name": "n_neighbors", "value": int(snn_params["n_neighbors"])},
        {"name": "weights", "value": str(snn_params["weights"])},
        {"name": "random_splits", "value": bool(snn_params["random_splits"])},
        {"name": "max_rank", "value": "" if snn_params["max_rank"] is None else int(snn_params["max_rank"])},
        {"name": "spectral_t", "value": "" if snn_params["spectral_t"] is None else float(snn_params["spectral_t"])},
        {"name": "linear_span_eps", "value": float(snn_params["linear_span_eps"])},
        {"name": "subspace_eps", "value": float(snn_params["subspace_eps"])},
        {"name": "min_value", "value": "" if snn_params["min_value"] is None else float(snn_params["min_value"])},
        {"name": "max_value", "value": "" if snn_params["max_value"] is None else float(snn_params["max_value"])},
        {"name": "treated_num_observed", "value": treated_stats["num_observed"]},
        {"name": "treated_num_target_missing", "value": treated_stats["num_target_missing"]},
        {"name": "treated_num_completed", "value": treated_stats["num_completed"]},
        {"name": "treated_num_failed", "value": treated_stats["num_failed"]},
        {"name": "untreated_num_observed", "value": untreated_stats["num_observed"]},
        {"name": "untreated_num_target_missing", "value": untreated_stats["num_target_missing"]},
        {"name": "untreated_num_completed", "value": untreated_stats["num_completed"]},
        {"name": "untreated_num_failed", "value": untreated_stats["num_failed"]},
        {
            "name": "treated_completion_rate",
            "value": _completion_rate(
                treated_stats["num_completed"], treated_stats["num_target_missing"]
            ),
        },
        {
            "name": "untreated_completion_rate",
            "value": _completion_rate(
                untreated_stats["num_completed"], untreated_stats["num_target_missing"]
            ),
        },
        {
            "name": "overall_completion_rate",
            "value": _completion_rate(total_completed, total_target_missing),
        },
        {"name": "status", "value": "completed"},
    ]
    write_csv_rows(fit_root / "snn_summary.csv", summary_rows, columns=["name", "value"])
    logger.info("Completed SNN fit successfully.")

    return {
        "treated_stats": treated_stats,
        "untreated_stats": untreated_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SNN fit root.")
    parser.add_argument("--data_folder", required=True, help="Fit root containing fit_realized_config.yaml.")
    args = parser.parse_args()
    try:
        run_snn_fit(args.data_folder)
    except Exception as exc:  # noqa: BLE001
        log_path = Path(args.data_folder) / "snn.log"
        logger = _setup_logger(log_path)
        logger.exception("SNN fit failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
