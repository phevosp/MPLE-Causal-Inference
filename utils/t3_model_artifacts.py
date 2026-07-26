"""Model artifact definitions and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from utils.t0_path_utils import io_path
from utils.t1_matrix_io import load_gamma_matrix


DEFAULT_LATENT_RANK = 0
_DEGENERACY_THRESHOLD = 1e-12   # norms below this are treated as zero/degenerate
_DEFAULT_FIELD_RMS_FRACTION = 0.4
SCALAR_PARAMETER_ORDER = ("beta", "xi", "eta")
SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK = "random_low_rank"
SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK = "confounded_low_rank"
VALID_SYNTHETIC_FIELD_MODES = {
    SYNTHETIC_FIELD_MODE_RANDOM_LOW_RANK,
    SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK,
}
OPTIMIZER_MODE_NO_EXTERNAL_FIELD = "no_external_field"
OPTIMIZER_MODE_NUCLEAR_NORM = "nuclear_norm"
OPTIMIZER_MODE_EXACT_RANK_MANIFOLD = "exact_rank_manifold"
OPTIMIZER_MODE_ALTERNATING_LATENT_RANK = "alternating_latent_rank"
OPTIMIZER_MODE_CONCURRENT_LATENT_RANK = "concurrent_latent_rank"
OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK = (
    "alternating_treatment_split_latent_rank"
)
OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK = (
    "alternating_treatment_shared_unit_latent_rank"
)
VALID_OPTIMIZER_MODES = {
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
}
FULL_FIELD_SERIALIZER_OPTIMIZER_MODES = {
    OPTIMIZER_MODE_NUCLEAR_NORM,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
}
LOW_RANK_OPTIMIZER_MODES = {
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SPLIT_LATENT_RANK,
    OPTIMIZER_MODE_ALTERNATING_TREATMENT_SHARED_UNIT_LATENT_RANK,
}


@dataclass(frozen=True)
class ModelArtifacts:
    """Latent-field artifacts for one experiment."""

    gamma_matrix: object
    t_steps: int
    latent_rank: int = 0
    optimizer_mode: str = OPTIMIZER_MODE_EXACT_RANK_MANIFOLD
    field_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class SpectralLowRankStructure:
    """Low-rank matrix structure with canonical panel orientation (T, N)."""

    node_factors: np.ndarray
    time_factors: np.ndarray
    singular_values: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class SyntheticFieldSpec:
    """Parsed synthetic-field configuration for generation."""

    mode: str
    singular_values: np.ndarray
    target_rms_fraction: float
    shared_rank: int | None
    B: float
    seed: int
    n_nodes: int | None
    t_steps: int | None


@dataclass(frozen=True)
class ConfoundedFieldLayout:
    """Resolved shared/nonshared rank split for confounded field generation."""

    total_rank: int
    available_shared_rank: int
    shared_rank: int
    nonshared_rank: int


@dataclass(frozen=True)
class SyntheticFieldBuildResult:
    """Generation-only synthetic-field build output with optional confounding layout."""

    artifacts: ModelArtifacts
    confounded_layout: ConfoundedFieldLayout | None = None


@dataclass(frozen=True)
class TreatmentFieldArtifacts:
    """Supplemental treatment-specific fit artifacts for treatment-split baselines."""

    optimizer_mode: str
    latent_rank: int
    control_field_matrix: np.ndarray
    treated_field_matrix: np.ndarray
    realized_field_matrix: np.ndarray
    lambda_uv_ridge: float
    best_start: int
    n_starts: int
    final_mple_loss: float
    final_penalized_objective: float
    control_node_factors: np.ndarray | None = None
    control_time_factors: np.ndarray | None = None
    treated_node_factors: np.ndarray | None = None
    treated_time_factors: np.ndarray | None = None
    shared_node_factors: np.ndarray | None = None


def build_fit_model_artifacts(config, gamma_matrix) -> ModelArtifacts:
    from utils.t2_normalization import (
        normalize_known_graph,
        validate_graph_infinity_norm,
    )
    from utils.t4_scalar_parameters import get_latent_rank, get_optimizer_mode
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    optimizer_mode = get_optimizer_mode(config)
    latent_rank = get_latent_rank(config)
    if optimizer_mode in {
        OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        OPTIMIZER_MODE_NUCLEAR_NORM,
    }:
        latent_rank = 0
    elif optimizer_mode in LOW_RANK_OPTIMIZER_MODES and latent_rank <= 0:
        raise ValueError(
            "global_params.latent_rank must be positive for optimizer_mode="
            f"'{optimizer_mode}'."
        )
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(config.global_params.T),
        latent_rank=latent_rank,
        optimizer_mode=optimizer_mode,
    )


def save_field_artifacts(path: str | Path, artifacts: ModelArtifacts) -> None:
    payload: dict[str, np.ndarray] = {
        "latent_rank": np.asarray(int(artifacts.latent_rank), dtype=int),
        "t_steps": np.asarray(int(artifacts.t_steps), dtype=int),
    }
    if artifacts.field_matrix is not None:
        payload["field_matrix"] = np.asarray(artifacts.field_matrix, dtype=float)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(io_path(target), **payload)


def load_field_artifacts(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as data:
        result: dict[str, object] = {
            "latent_rank": int(data["latent_rank"]),
            "t_steps": int(data["t_steps"]),
        }
        for key in ["field_matrix"]:
            if key in data:
                result[key] = data[key]
    return result


def save_treatment_field_artifacts(
    path: str | Path,
    artifacts: TreatmentFieldArtifacts,
) -> None:
    payload: dict[str, np.ndarray] = {
        "optimizer_mode": np.asarray(str(artifacts.optimizer_mode)),
        "latent_rank": np.asarray(int(artifacts.latent_rank), dtype=int),
        "control_field_matrix": np.asarray(artifacts.control_field_matrix, dtype=float),
        "treated_field_matrix": np.asarray(artifacts.treated_field_matrix, dtype=float),
        "realized_field_matrix": np.asarray(
            artifacts.realized_field_matrix,
            dtype=float,
        ),
        "lambda_uv_ridge": np.asarray(float(artifacts.lambda_uv_ridge), dtype=float),
        "best_start": np.asarray(int(artifacts.best_start), dtype=int),
        "n_starts": np.asarray(int(artifacts.n_starts), dtype=int),
        "final_mple_loss": np.asarray(float(artifacts.final_mple_loss), dtype=float),
        "final_penalized_objective": np.asarray(
            float(artifacts.final_penalized_objective),
            dtype=float,
        ),
    }
    optional_arrays = {
        "control_node_factors": artifacts.control_node_factors,
        "control_time_factors": artifacts.control_time_factors,
        "treated_node_factors": artifacts.treated_node_factors,
        "treated_time_factors": artifacts.treated_time_factors,
        "shared_node_factors": artifacts.shared_node_factors,
    }
    for key, value in optional_arrays.items():
        if value is not None:
            payload[key] = np.asarray(value, dtype=float)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(io_path(target), **payload)


def load_treatment_field_artifacts(
    path: str | Path,
) -> TreatmentFieldArtifacts:
    with np.load(Path(path), allow_pickle=False) as data:
        def _optional_array(key: str) -> np.ndarray | None:
            return np.asarray(data[key], dtype=float) if key in data else None

        return TreatmentFieldArtifacts(
            optimizer_mode=str(np.asarray(data["optimizer_mode"]).item()),
            latent_rank=int(np.asarray(data["latent_rank"]).item()),
            control_field_matrix=np.asarray(data["control_field_matrix"], dtype=float),
            treated_field_matrix=np.asarray(data["treated_field_matrix"], dtype=float),
            realized_field_matrix=np.asarray(data["realized_field_matrix"], dtype=float),
            lambda_uv_ridge=float(np.asarray(data["lambda_uv_ridge"]).item()),
            best_start=int(np.asarray(data["best_start"]).item()),
            n_starts=int(np.asarray(data["n_starts"]).item()),
            final_mple_loss=float(np.asarray(data["final_mple_loss"]).item()),
            final_penalized_objective=float(
                np.asarray(data["final_penalized_objective"]).item()
            ),
            control_node_factors=_optional_array("control_node_factors"),
            control_time_factors=_optional_array("control_time_factors"),
            treated_node_factors=_optional_array("treated_node_factors"),
            treated_time_factors=_optional_array("treated_time_factors"),
            shared_node_factors=_optional_array("shared_node_factors"),
        )


def save_model_artifacts(data_folder: str | Path, artifacts: ModelArtifacts) -> None:
    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)
    if sparse.issparse(artifacts.gamma_matrix):
        sparse.save_npz(
            data_path / "gamma_matrix_sparse.npz",
            sparse.csr_matrix(artifacts.gamma_matrix),
        )
    else:
        np.save(
            data_path / "gamma_matrix.npy",
            np.asarray(artifacts.gamma_matrix, dtype=float),
        )
    save_field_artifacts(data_path / "field_artifacts.npz", artifacts)


def load_model_artifacts(data_folder: str | Path) -> ModelArtifacts:
    data_path = Path(data_folder)
    field_path = data_path / "field_artifacts.npz"
    if not field_path.exists():
        raise FileNotFoundError(f"Missing field_artifacts.npz in {data_path}.")
    gamma_matrix = load_gamma_matrix(data_path)
    payload = load_field_artifacts(field_path)
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(payload["t_steps"]),
        latent_rank=int(payload.get("latent_rank", 0)),
        optimizer_mode=OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        field_matrix=payload.get("field_matrix"),
    )
