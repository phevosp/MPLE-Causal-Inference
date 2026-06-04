"""Model artifact definitions and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from utils.t0_path_utils import io_path


DEFAULT_LATENT_RANK = 0
_DEGENERACY_THRESHOLD = 1e-12   # norms below this are treated as zero/degenerate
_DEFAULT_FIELD_RMS_FRACTION = 0.4
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
VALID_OPTIMIZER_MODES = {
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    OPTIMIZER_MODE_NUCLEAR_NORM,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
    OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
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


def build_fit_model_artifacts(config, gamma_matrix) -> ModelArtifacts:
    from utils.t2_normalization import (
        normalize_known_graph,
        validate_graph_infinity_norm,
    )
    gamma_matrix = normalize_known_graph(gamma_matrix)
    validate_graph_infinity_norm(gamma_matrix)
    optimizer_mode = get_optimizer_mode(config)
    latent_rank = get_latent_rank(config)
    if optimizer_mode in {
        OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        OPTIMIZER_MODE_NUCLEAR_NORM,
    }:
        latent_rank = 0
    elif (
        optimizer_mode in {
            OPTIMIZER_MODE_ALTERNATING_LATENT_RANK,
            OPTIMIZER_MODE_CONCURRENT_LATENT_RANK,
        }
        and latent_rank <= 0
    ):
        raise ValueError(
            "global_params.latent_rank must be positive for optimizer_mode="
            f"'{optimizer_mode}'."
        )
    elif optimizer_mode == OPTIMIZER_MODE_EXACT_RANK_MANIFOLD and latent_rank <= 0:
        raise ValueError(
            "global_params.latent_rank must be positive for optimizer_mode='exact_rank_manifold'."
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
    gamma_sparse = data_path / "gamma_matrix_sparse.npz"
    gamma_dense = data_path / "gamma_matrix.npy"
    if gamma_sparse.exists():
        gamma_matrix = sparse.load_npz(gamma_sparse).tocsr()
    elif gamma_dense.exists():
        gamma_matrix = np.load(gamma_dense)
    else:
        raise FileNotFoundError(f"Missing gamma matrix artifact in {data_path}.")
    payload = load_field_artifacts(field_path)
    return ModelArtifacts(
        gamma_matrix=gamma_matrix,
        t_steps=int(payload["t_steps"]),
        latent_rank=int(payload.get("latent_rank", 0)),
        optimizer_mode=OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
        field_matrix=payload.get("field_matrix"),
    )


def get_latent_rank(config) -> int:
    if "latent_rank" not in config.global_params:
        return DEFAULT_LATENT_RANK
    rank = int(config.global_params.latent_rank)
    if rank < 0:
        raise ValueError("global_params.latent_rank must be nonnegative.")
    return rank


def get_optimizer_mode(config) -> str:
    global_params = getattr(config, "global_params", None)
    if global_params is None or "optimizer_mode" not in global_params:
        if global_params is not None and int(global_params.get("latent_rank", 0)) > 0:
            return OPTIMIZER_MODE_EXACT_RANK_MANIFOLD
        return OPTIMIZER_MODE_NO_EXTERNAL_FIELD
    optimizer_mode = str(global_params.optimizer_mode)
    if optimizer_mode not in VALID_OPTIMIZER_MODES:
        raise ValueError(
            "global_params.optimizer_mode must be one of: "
            + ", ".join(sorted(VALID_OPTIMIZER_MODES))
        )
    return optimizer_mode
