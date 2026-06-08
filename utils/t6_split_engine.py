"""Shared split-construction engine for train_cv and test_train_cv bundles."""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

from utils.t0_csv_utils import read_csv_rows, write_csv
from utils.t0_path_utils import io_path
from utils.t1_matrix_io import load_gamma_matrix
from utils.t5_experiment_context import load_experiment_panel_context


DEFAULT_GAMMA_TOLERANCE = 1.0e-12
DEFAULT_NUM_FOLDS = 5
SPLIT_KIND_TRAIN_CV = "train_cv"
SPLIT_KIND_TEST_TRAIN_CV = "test_train_cv"
VALID_SPLIT_KINDS = frozenset({SPLIT_KIND_TRAIN_CV, SPLIT_KIND_TEST_TRAIN_CV})
DEFAULT_OUTER_NUM_FOLDS = 5
DEFAULT_TEST_FOLD_ID = 1
ROLE_CODE_TRAINING = 0
ROLE_CODE_SEPARATOR = 1
ROLE_CODE_VALIDATION = 2
ROLE_CODE_OUTER_TEST = 3
ROLE_NAME_BY_CODE = {
    ROLE_CODE_TRAINING: "training",
    ROLE_CODE_SEPARATOR: "separator",
    ROLE_CODE_VALIDATION: "validation",
    ROLE_CODE_OUTER_TEST: "outer_test",
}


def train_cv_split_output_root(
    experiment_root: str | Path,
    *,
    num_folds: int,
) -> Path:
    return Path(experiment_root).resolve() / "splits" / SPLIT_KIND_TRAIN_CV / f"folds_{int(num_folds)}"


def test_train_cv_split_output_root(
    experiment_root: str | Path,
    *,
    outer_num_folds: int,
    test_fold_id: int,
    inner_num_folds: int,
) -> Path:
    return (
        Path(experiment_root).resolve()
        / "splits"
        / SPLIT_KIND_TEST_TRAIN_CV
        / f"outer_{int(outer_num_folds)}__test_{int(test_fold_id)}__inner_{int(inner_num_folds)}"
    )


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), io_path(path))


def _min_time_block_sizes_for_folds(num_folds: int) -> tuple[int, ...]:
    if int(num_folds) < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    if int(num_folds) == 1:
        return (1,)
    return (1,) + (2,) * (int(num_folds) - 1)


def _load_pymetis():
    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError(
            "pymetis is required for split graph partitioning. Install or refresh the "
            "Pixi environment so it includes the conda-forge 'pymetis' package."
        ) from exc
    return pymetis


def _max_abs_sparse_data(matrix: sparse.spmatrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    return float(np.max(np.abs(matrix.data)))


def validate_gamma_matrix(
    gamma_matrix: sparse.spmatrix | np.ndarray,
    *,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, float]:
    shape = tuple(int(dim) for dim in gamma_matrix.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Gamma matrix must be square; received shape {shape}.")

    if sparse.issparse(gamma_matrix):
        gamma_csr = gamma_matrix.tocsr()
        symmetry_diff = gamma_csr - gamma_csr.transpose()
        max_symmetry_violation = _max_abs_sparse_data(symmetry_diff)
        diagonal = np.asarray(gamma_csr.diagonal(), dtype=float)
    else:
        gamma_dense = np.asarray(gamma_matrix, dtype=float)
        symmetry_diff = gamma_dense - gamma_dense.T
        max_symmetry_violation = float(np.max(np.abs(symmetry_diff)))
        diagonal = np.asarray(np.diag(gamma_dense), dtype=float)

    max_abs_diagonal = float(np.max(np.abs(diagonal))) if diagonal.size else 0.0
    violations: list[str] = []
    if max_symmetry_violation > tolerance:
        violations.append(
            "Gamma matrix must be symmetric"
            f" (max symmetry violation {max_symmetry_violation:.3e} > {tolerance:.3e})."
        )
    if max_abs_diagonal > tolerance:
        violations.append(
            "Gamma matrix must have zero diagonal"
            f" (max absolute diagonal entry {max_abs_diagonal:.3e} > {tolerance:.3e})."
        )
    if violations:
        raise ValueError(" ".join(violations))

    return {
        "tolerance": float(tolerance),
        "max_symmetry_violation": max_symmetry_violation,
        "max_abs_diagonal": max_abs_diagonal,
    }


def _support_adjacency_from_gamma(
    gamma_matrix: sparse.spmatrix | np.ndarray,
    *,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> tuple[list[list[int]], int]:
    n_vertices = int(gamma_matrix.shape[0])
    adjacency: list[list[int]] = []

    if sparse.issparse(gamma_matrix):
        gamma_csr = gamma_matrix.tocsr()
        for vertex_index in range(n_vertices):
            start = int(gamma_csr.indptr[vertex_index])
            stop = int(gamma_csr.indptr[vertex_index + 1])
            neighbors = gamma_csr.indices[start:stop]
            weights = gamma_csr.data[start:stop]
            mask = (neighbors != vertex_index) & (np.abs(weights) > tolerance)
            adjacency.append([int(value) for value in neighbors[mask]])
    else:
        gamma_dense = np.asarray(gamma_matrix, dtype=float)
        for vertex_index in range(n_vertices):
            row = gamma_dense[vertex_index]
            mask = np.abs(row) > tolerance
            mask[vertex_index] = False
            adjacency.append(np.flatnonzero(mask).astype(int).tolist())

    num_edges = int(sum(len(neighbors) for neighbors in adjacency) // 2)
    return adjacency, num_edges


def _compute_partition_metrics(
    adjacency: list[list[int]],
    membership: np.ndarray,
    *,
    num_folds: int,
) -> dict[str, Any]:
    num_vertices = len(adjacency)
    separator_sets: list[set[int]] = [set() for _ in range(num_folds)]
    cut_edge_count = 0

    for vertex_index, neighbors in enumerate(adjacency):
        vertex_partition = int(membership[vertex_index])
        for neighbor_index in neighbors:
            neighbor_partition = int(membership[neighbor_index])
            if neighbor_partition == vertex_partition:
                continue
            separator_sets[vertex_partition].add(int(neighbor_index))
            if neighbor_index > vertex_index:
                cut_edge_count += 1

    partition_sizes = np.bincount(membership, minlength=num_folds).astype(int)
    separator_union: set[int] = set()
    for separator_set in separator_sets:
        separator_union.update(separator_set)
    separator_union_vertex_count = int(len(separator_union))
    mean_partition_size = float(num_vertices) / float(num_folds)

    return {
        "separator_sets": separator_sets,
        "partition_sizes": [int(value) for value in partition_sizes.tolist()],
        "separator_sizes": [int(len(values)) for values in separator_sets],
        "separator_union_vertex_count": separator_union_vertex_count,
        "separator_union_vertex_fraction": (
            float(separator_union_vertex_count) / float(num_vertices) if num_vertices > 0 else 0.0
        ),
        "total_separator_memberships": int(sum(len(values) for values in separator_sets)),
        "cut_edge_count": int(cut_edge_count),
        "partition_size_mean": mean_partition_size,
        "partition_size_min": int(np.min(partition_sizes)) if partition_sizes.size else 0,
        "partition_size_max": int(np.max(partition_sizes)) if partition_sizes.size else 0,
        "partition_size_std": float(np.std(partition_sizes, ddof=0)) if partition_sizes.size else 0.0,
        "max_relative_deviation_from_mean": (
            float(np.max(np.abs(partition_sizes - mean_partition_size) / mean_partition_size))
            if mean_partition_size > 0.0
            else 0.0
        ),
    }


def _infer_t_steps_from_panel(experiment_root: str | Path) -> int:
    panel_path = Path(experiment_root) / "panel_data.npz"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Could not infer time horizon because {panel_path} does not exist."
        )
    with np.load(panel_path, allow_pickle=False) as data:
        if "x" in data:
            return int(np.asarray(data["x"]).shape[0])
        if "z" in data:
            return int(np.asarray(data["z"]).shape[0])
    raise KeyError(f"Expected 'x' or 'z' in {panel_path}.")


def _load_time_index(
    experiment_root: str | Path,
) -> tuple[int, list[dict[str, object]], list[str], str]:
    experiment_path = Path(experiment_root)
    time_index_path = experiment_path / "time_index.csv"
    inferred_t_steps = _infer_t_steps_from_panel(experiment_path)

    if not time_index_path.exists():
        rows = [{"time_index": int(time_index)} for time_index in range(inferred_t_steps)]
        return inferred_t_steps, rows, [], "inferred_from_panel_data"

    with time_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if len(raw_rows) == inferred_t_steps + 1:
        raw_rows = raw_rows[1:]
    if len(raw_rows) != inferred_t_steps:
        raise ValueError(
            f"time_index.csv row count {len(raw_rows)} does not match panel horizon "
            f"{inferred_t_steps} in {experiment_root}."
        )
    passthrough_columns = [
        column
        for column in fieldnames
        if column not in {"time_index", "time_block_id", "block_position", "is_transition_step"}
    ]
    rows: list[dict[str, object]] = []
    for time_index, raw_row in enumerate(raw_rows):
        row = {"time_index": int(time_index)}
        for column in passthrough_columns:
            row[column] = raw_row.get(column, "")
        rows.append(row)
    return inferred_t_steps, rows, passthrough_columns, "time_index_csv"


def _build_time_block_plan(
    t_steps: int,
    *,
    num_folds: int,
) -> dict[str, Any]:
    min_block_sizes = _min_time_block_sizes_for_folds(num_folds)
    minimum_steps = int(sum(min_block_sizes))
    if int(t_steps) < minimum_steps:
        raise ValueError(
            f"Spatiotemporal CV with {num_folds} folds requires at least {minimum_steps} "
            f"time steps to support the 1-step transition separators (received T={t_steps})."
        )

    block_sizes = list(int(value) for value in min_block_sizes)
    remaining = int(t_steps) - minimum_steps
    while remaining > 0:
        smallest_size = min(block_sizes)
        for block_index, block_size in enumerate(block_sizes):
            if remaining <= 0:
                break
            if block_size == smallest_size:
                block_sizes[block_index] += 1
                remaining -= 1

    time_block_ids = np.zeros(int(t_steps), dtype=int)
    block_positions = np.zeros(int(t_steps), dtype=int)
    is_transition_step = np.zeros(int(t_steps), dtype=bool)
    block_start_indices: list[int] = []
    block_end_indices: list[int] = []
    cursor = 0
    for block_index, block_size in enumerate(block_sizes, start=1):
        block_start_indices.append(int(cursor))
        for offset in range(block_size):
            time_index = cursor + offset
            time_block_ids[time_index] = int(block_index)
            block_positions[time_index] = int(offset + 1)
        cursor += block_size
        block_end_indices.append(int(cursor - 1))
    for start_index in block_start_indices[1:]:
        is_transition_step[start_index] = True

    return {
        "block_sizes": [int(value) for value in block_sizes],
        "time_block_ids": time_block_ids,
        "block_positions": block_positions,
        "is_transition_step": is_transition_step,
        "block_start_indices": [int(value) for value in block_start_indices],
        "block_end_indices": [int(value) for value in block_end_indices],
        "transition_time_indices": [int(value) for value in block_start_indices[1:]],
    }


def _build_validation_schedule(*, num_folds: int) -> np.ndarray:
    schedule = np.zeros((num_folds, num_folds), dtype=int)
    for fold_index in range(num_folds):
        for block_index in range(num_folds):
            schedule[fold_index, block_index] = ((block_index - fold_index) % num_folds) + 1
    return schedule


def _build_partition_vertex_sets(
    membership: np.ndarray,
    *,
    num_folds: int,
) -> list[set[int]]:
    partition_sets: list[set[int]] = [set() for _ in range(num_folds)]
    for vertex_index, partition_id_zero_based in enumerate(np.asarray(membership, dtype=int)):
        partition_sets[int(partition_id_zero_based)].add(int(vertex_index))
    return partition_sets


def _build_role_tensor(
    *,
    partition_sets: list[set[int]],
    separator_sets: list[set[int]],
    validation_partition_ids_by_fold_block: np.ndarray,
    time_block_ids: np.ndarray,
    is_transition_step: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    num_folds = int(validation_partition_ids_by_fold_block.shape[0])
    t_steps = int(time_block_ids.shape[0])
    role_codes = np.full((num_folds, t_steps, num_vertices), ROLE_CODE_TRAINING, dtype=np.int8)
    for fold_index in range(num_folds):
        for time_index in range(t_steps):
            block_zero_based = int(time_block_ids[time_index]) - 1
            current_partition_id = int(validation_partition_ids_by_fold_block[fold_index, block_zero_based])
            current_partition_set = partition_sets[current_partition_id - 1]
            current_separator_set = separator_sets[current_partition_id - 1]
            if bool(is_transition_step[time_index]):
                previous_partition_id = int(validation_partition_ids_by_fold_block[fold_index, block_zero_based - 1])
                previous_partition_set = partition_sets[previous_partition_id - 1]
                transition_separator = set(current_separator_set) | set(previous_partition_set) | set(current_partition_set)
                if transition_separator:
                    role_codes[fold_index, time_index, np.asarray(sorted(transition_separator), dtype=int)] = ROLE_CODE_SEPARATOR
                continue
            if current_separator_set:
                role_codes[fold_index, time_index, np.asarray(sorted(current_separator_set), dtype=int)] = ROLE_CODE_SEPARATOR
            if current_partition_set:
                role_codes[fold_index, time_index, np.asarray(sorted(current_partition_set), dtype=int)] = ROLE_CODE_VALIDATION
    return role_codes


def _summarize_markov_blanket_validation(
    adjacency: list[list[int]],
    role_codes: np.ndarray,
) -> dict[str, Any]:
    role_array = np.asarray(role_codes, dtype=int)
    num_folds = int(role_array.shape[0])
    num_time_steps = int(role_array.shape[1])
    num_vertices = int(role_array.shape[2])
    spatial_violations_by_fold = [0 for _ in range(num_folds)]
    temporal_violations_by_fold = [0 for _ in range(num_folds)]

    for fold_index in range(num_folds):
        for time_index in range(num_time_steps):
            roles_at_time = role_array[fold_index, time_index, :]
            for vertex_index, neighbors in enumerate(adjacency):
                vertex_role = int(roles_at_time[vertex_index])
                for neighbor_index in neighbors:
                    if neighbor_index <= vertex_index:
                        continue
                    neighbor_role = int(roles_at_time[neighbor_index])
                    if {vertex_role, neighbor_role} == {ROLE_CODE_VALIDATION, ROLE_CODE_TRAINING}:
                        spatial_violations_by_fold[fold_index] += 1
                    if {vertex_role, neighbor_role} == {ROLE_CODE_OUTER_TEST, ROLE_CODE_TRAINING}:
                        spatial_violations_by_fold[fold_index] += 1

    for fold_index in range(num_folds):
        for time_index in range(1, num_time_steps):
            previous_roles = role_array[fold_index, time_index - 1, :]
            current_roles = role_array[fold_index, time_index, :]
            temporal_violations_by_fold[fold_index] += int(
                np.count_nonzero(
                    ((previous_roles == ROLE_CODE_VALIDATION) & (current_roles == ROLE_CODE_TRAINING))
                    | ((previous_roles == ROLE_CODE_TRAINING) & (current_roles == ROLE_CODE_VALIDATION))
                )
            )

    spatial_violation_edge_count = int(sum(spatial_violations_by_fold))
    temporal_violation_edge_count = int(sum(temporal_violations_by_fold))
    total_violation_count = int(spatial_violation_edge_count + temporal_violation_edge_count)
    violations_by_fold = [
        int(spatial_count + temporal_count)
        for spatial_count, temporal_count in zip(spatial_violations_by_fold, temporal_violations_by_fold)
    ]
    return {
        "blanket_validation_passed": bool(total_violation_count == 0),
        "num_folds": int(num_folds),
        "num_vertices": int(num_vertices),
        "num_time_steps": int(num_time_steps),
        "spatial_violation_edge_count": spatial_violation_edge_count,
        "temporal_violation_edge_count": temporal_violation_edge_count,
        "total_violation_count": total_violation_count,
        "num_folds_with_any_violation": int(sum(count > 0 for count in violations_by_fold)),
        "violations_by_fold": [int(value) for value in violations_by_fold],
        "spatial_violations_by_fold": [int(value) for value in spatial_violations_by_fold],
        "temporal_violations_by_fold": [int(value) for value in temporal_violations_by_fold],
        "validation_rule": {
            "graph_scope": "full_spatiotemporal",
            "spatial_rule": "no_same_time_gamma_edge_between_validation_and_training",
            "temporal_rule": "no_adjacent_time_self_edge_between_validation_and_training",
        },
        "role_code_map": {
            "training": int(ROLE_CODE_TRAINING),
            "separator": int(ROLE_CODE_SEPARATOR),
            "validation": int(ROLE_CODE_VALIDATION),
        },
    }


def _panel_shape(experiment_root: str | Path) -> tuple[int, int]:
    panel_context = load_experiment_panel_context(experiment_root)
    return int(panel_context["T"]), int(panel_context["N"])


def _stable_partition_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    stable = dict(metadata)
    stable.pop("runtime_seconds", None)
    return stable


def _build_full_panel_cv_artifacts(
    experiment_root: str | Path,
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, Any]:
    if int(num_folds) < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    experiment_path = Path(experiment_root).resolve()
    gamma_matrix = load_gamma_matrix(experiment_path)
    gamma_validation = validate_gamma_matrix(gamma_matrix, tolerance=tolerance)
    num_vertices = int(gamma_matrix.shape[0])
    if int(num_folds) > int(num_vertices):
        raise ValueError(
            f"num_folds={int(num_folds)} exceeds the number of vertices {int(num_vertices)}."
        )
    adjacency, num_edges = _support_adjacency_from_gamma(gamma_matrix, tolerance=tolerance)
    t_steps, _, _, time_source = _load_time_index(experiment_path)
    time_plan = _build_time_block_plan(t_steps, num_folds=int(num_folds))
    pymetis = _load_pymetis()
    start_time = time.perf_counter()
    metis_cutcount, membership = pymetis.part_graph(
        int(num_folds),
        adjacency=adjacency,
        recursive=bool(recursive),
        contiguous=bool(contiguous),
    )
    runtime_seconds = float(time.perf_counter() - start_time)
    membership_array = np.asarray(membership, dtype=int)
    if membership_array.shape != (num_vertices,):
        raise ValueError(
            "pymetis returned an invalid membership vector with shape "
            f"{membership_array.shape}; expected ({num_vertices},)."
        )
    if np.any(membership_array < 0) or np.any(membership_array >= int(num_folds)):
        raise ValueError("pymetis returned partition ids outside the requested range.")
    partition_sets = _build_partition_vertex_sets(membership_array, num_folds=int(num_folds))
    partition_metrics = _compute_partition_metrics(adjacency, membership_array, num_folds=int(num_folds))
    separator_sets = partition_metrics["separator_sets"]
    validation_schedule = _build_validation_schedule(num_folds=int(num_folds))
    role_codes = _build_role_tensor(
        partition_sets=partition_sets,
        separator_sets=separator_sets,
        validation_partition_ids_by_fold_block=validation_schedule,
        time_block_ids=time_plan["time_block_ids"],
        is_transition_step=time_plan["is_transition_step"],
        num_vertices=num_vertices,
    )
    blanket_summary = _summarize_markov_blanket_validation(adjacency, role_codes)
    if not bool(blanket_summary["blanket_validation_passed"]):
        raise ValueError(
            "Constructed full-panel folds failed the spatiotemporal Markov-blanket validation."
        )
    return {
        "role_codes": role_codes,
        "spatial_partition_metadata": {
            "experiment_root": str(experiment_path),
            "num_folds": int(num_folds),
            "seed": int(seed),
            "partitioner": "pymetis",
            "requested_seed": int(seed),
            "runtime_seconds": runtime_seconds,
            "gamma_validation": {"passed": True, **gamma_validation},
            "metrics": {
                "num_vertices": int(num_vertices),
                "num_edges": int(num_edges),
                "partition_sizes": partition_metrics["partition_sizes"],
                "cut_edge_count": int(partition_metrics["cut_edge_count"]),
                "separator_sizes": partition_metrics["separator_sizes"],
            },
        },
        "spatiotemporal_metadata": {
            "num_cv_folds": int(num_folds),
            "num_vertices": int(num_vertices),
            "num_time_steps": int(t_steps),
            "time_source": str(time_source),
            "time_block_sizes": time_plan["block_sizes"],
            "transition_time_indices": time_plan["transition_time_indices"],
            "validation_schedule": validation_schedule.tolist(),
            "role_code_map": {name: int(code) for code, name in ROLE_NAME_BY_CODE.items()},
        },
        "markov_blanket_summary": blanket_summary,
    }


def _role_masks_for_fold(role_codes: np.ndarray, fold_id: int) -> dict[str, np.ndarray]:
    fold_index = int(fold_id) - 1
    if fold_index < 0 or fold_index >= int(role_codes.shape[0]):
        raise ValueError(
            f"fold_id must be between 1 and {int(role_codes.shape[0])}, got {fold_id}."
        )
    fold_roles = np.asarray(role_codes[fold_index], dtype=np.int8)
    return {
        "training_mask": np.asarray(fold_roles == ROLE_CODE_TRAINING, dtype=bool),
        "separator_mask": np.asarray(fold_roles == ROLE_CODE_SEPARATOR, dtype=bool),
        "validation_mask": np.asarray(fold_roles == ROLE_CODE_VALIDATION, dtype=bool),
    }


def build_outer_layer_for_train_cv(
    experiment_root: str | Path,
) -> dict[str, Any]:
    t_steps, n_nodes = _panel_shape(experiment_root)
    outer_active_mask = np.ones((t_steps, n_nodes), dtype=bool)
    outer_separator_mask = np.zeros((t_steps, n_nodes), dtype=bool)
    outer_test_mask = np.zeros((t_steps, n_nodes), dtype=bool)
    return {
        "outer_active_mask": outer_active_mask,
        "outer_separator_mask": outer_separator_mask,
        "outer_test_mask": outer_test_mask,
        "metadata": {
            "split_kind": SPLIT_KIND_TRAIN_CV,
            "panel_shape": {"T": int(t_steps), "N": int(n_nodes)},
            "num_outer_active_slots": int(np.count_nonzero(outer_active_mask)),
            "num_outer_separator_slots": 0,
            "num_outer_test_slots": 0,
        },
        "blanket_summary": {
            "blanket_validation_passed": True,
            "reason": "trivial_outer_layer",
        },
    }


def build_outer_layer_for_test_train_cv(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, Any]:
    if int(outer_num_folds) < 1:
        raise ValueError(f"outer_num_folds must be >= 1, got {outer_num_folds}.")
    if int(test_fold_id) < 1 or int(test_fold_id) > int(outer_num_folds):
        raise ValueError(
            f"test_fold_id must be between 1 and {int(outer_num_folds)}, got {test_fold_id}."
        )
    outer_artifacts = _build_full_panel_cv_artifacts(
        experiment_root,
        num_folds=int(outer_num_folds),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )
    masks = _role_masks_for_fold(np.asarray(outer_artifacts["role_codes"], dtype=np.int8), int(test_fold_id))
    return {
        "outer_active_mask": np.asarray(masks["training_mask"], dtype=bool),
        "outer_separator_mask": np.asarray(masks["separator_mask"], dtype=bool),
        "outer_test_mask": np.asarray(masks["validation_mask"], dtype=bool),
        "metadata": {
            "split_kind": SPLIT_KIND_TEST_TRAIN_CV,
            "outer_num_folds": int(outer_num_folds),
            "test_fold_id": int(test_fold_id),
            "seed": int(seed),
            "recursive": bool(recursive),
            "contiguous": bool(contiguous),
            "tolerance": float(tolerance),
            "outer_spatial_partition_metadata": _stable_partition_metadata(
                dict(outer_artifacts["spatial_partition_metadata"])
            ),
            "outer_spatiotemporal_metadata": dict(outer_artifacts["spatiotemporal_metadata"]),
            "num_outer_active_slots": int(np.count_nonzero(masks["training_mask"])),
            "num_outer_separator_slots": int(np.count_nonzero(masks["separator_mask"])),
            "num_outer_test_slots": int(np.count_nonzero(masks["validation_mask"])),
        },
        "blanket_summary": dict(outer_artifacts["markov_blanket_summary"]),
    }


def _validate_outer_layer_masks(
    *,
    outer_active_mask: np.ndarray,
    outer_separator_mask: np.ndarray,
    outer_test_mask: np.ndarray,
) -> None:
    active = np.asarray(outer_active_mask, dtype=bool)
    separator = np.asarray(outer_separator_mask, dtype=bool)
    test_mask = np.asarray(outer_test_mask, dtype=bool)
    if active.shape != separator.shape or active.shape != test_mask.shape:
        raise ValueError("Outer-layer masks must all have the same shape.")
    overlap_count = int(
        np.count_nonzero(active & separator)
        + np.count_nonzero(active & test_mask)
        + np.count_nonzero(separator & test_mask)
    )
    if overlap_count > 0:
        raise ValueError("Outer-layer masks must be pairwise disjoint.")


def _active_pattern_key(mask_row: np.ndarray) -> bytes:
    return np.asarray(mask_row, dtype=np.uint8).tobytes()


def _active_pattern_signature(mask_row: np.ndarray) -> str:
    indices = np.flatnonzero(np.asarray(mask_row, dtype=bool)).astype(int).tolist()
    return ",".join(str(index) for index in indices)


def _build_induced_adjacency(
    full_adjacency: list[list[int]],
    active_nodes: np.ndarray,
) -> tuple[list[list[int]], dict[int, int]]:
    active_list = [int(value) for value in np.asarray(active_nodes, dtype=int).tolist()]
    index_by_node = {node: index for index, node in enumerate(active_list)}
    active_set = set(active_list)
    induced: list[list[int]] = []
    for node in active_list:
        induced.append(
            [index_by_node[int(neighbor)] for neighbor in full_adjacency[node] if int(neighbor) in active_set]
        )
    return induced, index_by_node


def _build_partition_for_active_pattern(
    *,
    active_nodes: np.ndarray,
    full_adjacency: list[list[int]],
    num_folds: int,
    pymetis,
    recursive: bool,
    contiguous: bool,
) -> dict[str, Any]:
    active_array = np.asarray(active_nodes, dtype=int)
    if active_array.size < int(num_folds):
        raise ValueError(
            f"Active region has only {int(active_array.size)} nodes, which is fewer than "
            f"num_folds={int(num_folds)}. Cannot build inner folds."
        )
    induced_adjacency, _ = _build_induced_adjacency(full_adjacency, active_array)
    metis_cutcount, membership = pymetis.part_graph(
        int(num_folds),
        adjacency=induced_adjacency,
        recursive=bool(recursive),
        contiguous=bool(contiguous),
    )
    membership_array = np.asarray(membership, dtype=int)
    if membership_array.shape != (int(active_array.size),):
        raise ValueError(
            "pymetis returned an invalid membership vector for the active induced graph "
            f"with shape {membership_array.shape}; expected ({int(active_array.size)},)."
        )
    if np.any(membership_array < 0) or np.any(membership_array >= int(num_folds)):
        raise ValueError("pymetis returned partition ids outside the requested active-region range.")
    partition_sets_local = _build_partition_vertex_sets(
        membership_array,
        num_folds=int(num_folds),
    )
    partition_metrics = _compute_partition_metrics(
        induced_adjacency,
        membership_array,
        num_folds=int(num_folds),
    )
    separator_sets_local = partition_metrics["separator_sets"]
    partition_sets_global = [
        {int(active_array[local_index]) for local_index in sorted(local_partition)}
        for local_partition in partition_sets_local
    ]
    separator_sets_global = [
        {int(active_array[local_index]) for local_index in sorted(local_separator)}
        for local_separator in separator_sets_local
    ]
    return {
        "num_active_nodes": int(active_array.size),
        "partition_sets": partition_sets_global,
        "separator_sets": separator_sets_global,
        "reported_cutcount": int(metis_cutcount),
        "partition_sizes": partition_metrics["partition_sizes"],
        "separator_sizes": partition_metrics["separator_sizes"],
        "cut_edge_count": int(partition_metrics["cut_edge_count"]),
        "all_separator": False,
    }


def _build_fold_summary_rows(
    *,
    training_masks: np.ndarray,
    separator_masks: np.ndarray,
    validation_masks: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_index in range(int(training_masks.shape[0])):
        rows.append(
            {
                "fold_id": int(fold_index + 1),
                "num_training_slots": int(np.count_nonzero(training_masks[fold_index])),
                "num_separator_slots": int(np.count_nonzero(separator_masks[fold_index])),
                "num_validation_slots": int(np.count_nonzero(validation_masks[fold_index])),
            }
        )
    return rows


def build_model_selection_folds(
    experiment_root: str | Path,
    *,
    num_folds: int,
    outer_active_mask: np.ndarray,
    outer_separator_mask: np.ndarray,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, Any]:
    if int(num_folds) < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    experiment_path = Path(experiment_root).resolve()
    gamma_matrix = load_gamma_matrix(experiment_path)
    gamma_validation = validate_gamma_matrix(gamma_matrix, tolerance=float(tolerance))
    full_adjacency, num_edges = _support_adjacency_from_gamma(gamma_matrix, tolerance=float(tolerance))
    t_steps, _, _, time_source = _load_time_index(experiment_path)
    active_mask = np.asarray(outer_active_mask, dtype=bool)
    outer_separator = np.asarray(outer_separator_mask, dtype=bool)
    partitionable_mask = np.asarray(active_mask | outer_separator, dtype=bool)
    if active_mask.shape[0] != int(t_steps):
        raise ValueError(
            f"outer_active_mask has T={active_mask.shape[0]}, expected {int(t_steps)} for {experiment_path}."
        )
    if active_mask.shape != outer_separator.shape:
        raise ValueError("outer_active_mask and outer_separator_mask must share the same shape.")
    time_plan = _build_time_block_plan(int(t_steps), num_folds=int(num_folds))
    validation_schedule = _build_validation_schedule(num_folds=int(num_folds))
    num_vertices = int(active_mask.shape[1])
    if int(num_folds) > int(num_vertices):
        raise ValueError(
            f"num_folds={int(num_folds)} exceeds the number of vertices {int(num_vertices)}."
        )

    pattern_cache: dict[bytes, dict[str, Any]] = {}
    pattern_occurrences: dict[bytes, int] = {}
    pattern_signatures: dict[bytes, str] = {}
    pymetis = _load_pymetis()

    training_masks = np.zeros((int(num_folds), int(t_steps), int(num_vertices)), dtype=bool)
    separator_masks = np.zeros_like(training_masks)
    validation_masks = np.zeros_like(training_masks)
    role_codes = np.full(training_masks.shape, ROLE_CODE_SEPARATOR, dtype=np.int8)

    for time_index in range(int(t_steps)):
        current_training_candidates = np.flatnonzero(active_mask[time_index]).astype(int)
        current_partitionable_nodes = np.flatnonzero(partitionable_mask[time_index]).astype(int)
        fixed_outer_separator_nodes = np.flatnonzero(outer_separator[time_index]).astype(int)
        pattern_key = _active_pattern_key(partitionable_mask[time_index])
        pattern_occurrences[pattern_key] = int(pattern_occurrences.get(pattern_key, 0) + 1)
        pattern_signatures.setdefault(pattern_key, _active_pattern_signature(partitionable_mask[time_index]))
        if current_training_candidates.size == 0:
            continue
        if current_partitionable_nodes.size == 0:
            continue
        if pattern_key not in pattern_cache:
            pattern_cache[pattern_key] = _build_partition_for_active_pattern(
                active_nodes=current_partitionable_nodes,
                full_adjacency=full_adjacency,
                num_folds=int(num_folds),
                pymetis=pymetis,
                recursive=bool(recursive),
                contiguous=bool(contiguous),
            )
        pattern = pattern_cache[pattern_key]
        block_zero_based = int(time_plan["time_block_ids"][time_index]) - 1
        previous_pattern = None
        if time_index > 0:
            previous_partitionable_nodes = np.flatnonzero(partitionable_mask[time_index - 1]).astype(int)
            previous_key = _active_pattern_key(partitionable_mask[time_index - 1])
            previous_pattern = pattern_cache.get(previous_key)
            if previous_pattern is None:
                previous_pattern = _build_partition_for_active_pattern(
                    active_nodes=previous_partitionable_nodes,
                    full_adjacency=full_adjacency,
                    num_folds=int(num_folds),
                    pymetis=pymetis,
                    recursive=bool(recursive),
                    contiguous=bool(contiguous),
                )
                pattern_cache[previous_key] = previous_pattern

        for fold_index in range(int(num_folds)):
            role_codes[fold_index, time_index, current_training_candidates] = ROLE_CODE_TRAINING
            current_partition_id = int(validation_schedule[fold_index, block_zero_based])
            current_partition = set(pattern["partition_sets"][current_partition_id - 1])
            current_separator = set(pattern["separator_sets"][current_partition_id - 1])
            current_validation = current_partition & set(current_training_candidates.tolist())
            visible_separator = current_separator | set(fixed_outer_separator_nodes.tolist())

            if bool(time_plan["is_transition_step"][time_index]):
                previous_partition_id = int(validation_schedule[fold_index, block_zero_based - 1])
                previous_partition = set()
                if previous_pattern is not None:
                    previous_partition = set(previous_pattern["partition_sets"][previous_partition_id - 1])
                transition_separator = visible_separator | current_validation | (
                    previous_partition & set(current_training_candidates.tolist())
                )
                if transition_separator:
                    separator_indices = np.asarray(sorted(transition_separator), dtype=int)
                    separator_masks[fold_index, time_index, separator_indices] = True
                    role_codes[fold_index, time_index, separator_indices] = ROLE_CODE_SEPARATOR
                continue

            if visible_separator:
                separator_indices = np.asarray(sorted(visible_separator), dtype=int)
                separator_masks[fold_index, time_index, separator_indices] = True
                role_codes[fold_index, time_index, separator_indices] = ROLE_CODE_SEPARATOR
            if current_validation:
                validation_indices = np.asarray(sorted(current_validation), dtype=int)
                validation_masks[fold_index, time_index, validation_indices] = True
                role_codes[fold_index, time_index, validation_indices] = ROLE_CODE_VALIDATION

    outer_test_mask = ~(active_mask | outer_separator)
    for fold_index in range(int(num_folds)):
        for time_index in range(int(t_steps)):
            outer_test_nodes = np.flatnonzero(outer_test_mask[time_index]).astype(int)
            if outer_test_nodes.size > 0:
                role_codes[fold_index, time_index, outer_test_nodes] = ROLE_CODE_OUTER_TEST

    training_masks = np.asarray(
        active_mask[None, :, :] & ~separator_masks & ~validation_masks,
        dtype=bool,
    )
    if np.any(training_masks & separator_masks) or np.any(training_masks & validation_masks) or np.any(separator_masks & validation_masks):
        raise ValueError("Model-selection fold masks must be pairwise disjoint.")

    blanket_summary = _summarize_markov_blanket_validation(full_adjacency, role_codes)
    fold_summary_rows = _build_fold_summary_rows(
        training_masks=training_masks,
        separator_masks=separator_masks,
        validation_masks=validation_masks,
    )
    supported_fold_ids = [
        int(row["fold_id"])
        for row in fold_summary_rows
        if int(row["num_training_slots"]) > 0 and int(row["num_validation_slots"]) > 0
    ]
    if not bool(blanket_summary["blanket_validation_passed"]):
        raise ValueError("Inner model-selection folds failed the spatiotemporal Markov-blanket validation.")
    if not supported_fold_ids:
        raise ValueError("Inner model-selection folds produce no folds with non-empty training and validation support.")

    metadata = {
        "num_folds": int(num_folds),
        "seed": int(seed),
        "recursive": bool(recursive),
        "contiguous": bool(contiguous),
        "tolerance": float(tolerance),
        "gamma_validation": {"passed": True, **gamma_validation},
        "num_vertices": int(num_vertices),
        "num_time_steps": int(t_steps),
        "num_edges": int(num_edges),
        "time_source": str(time_source),
        "num_active_slots": int(np.count_nonzero(active_mask)),
        "num_partitionable_slots": int(np.count_nonzero(partitionable_mask)),
        "num_outer_separator_slots_visible_as_context": int(np.count_nonzero(outer_separator)),
        "supported_fold_ids": supported_fold_ids,
        "num_supported_folds": int(len(supported_fold_ids)),
        "fold_summary_rows": fold_summary_rows,
        "pattern_summaries": [
            {
                "active_node_signature": pattern_signatures[key],
                "num_occurrences": int(pattern_occurrences[key]),
                "num_active_nodes": int(pattern_cache[key]["num_active_nodes"]),
                "partition_sizes": list(pattern_cache[key]["partition_sizes"]),
                "separator_sizes": list(pattern_cache[key]["separator_sizes"]),
                "cut_edge_count": int(pattern_cache[key]["cut_edge_count"]),
            }
            for key in pattern_cache
        ],
        "blanket_summary": dict(blanket_summary),
    }
    return {
        "training_masks": training_masks,
        "separator_masks": separator_masks,
        "validation_masks": validation_masks,
        "metadata": metadata,
        "fold_summary_rows": fold_summary_rows,
    }


def build_train_cv_bundle(
    experiment_root: str | Path,
    *,
    num_folds: int,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, Any]:
    outer_layer = build_outer_layer_for_train_cv(experiment_root)
    model_selection = build_model_selection_folds(
        experiment_root,
        num_folds=int(num_folds),
        outer_active_mask=np.asarray(outer_layer["outer_active_mask"], dtype=bool),
        outer_separator_mask=np.asarray(outer_layer["outer_separator_mask"], dtype=bool),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )
    t_steps, n_nodes = _panel_shape(experiment_root)
    return {
        "split_kind": SPLIT_KIND_TRAIN_CV,
        "outer_layer": outer_layer,
        "model_selection": model_selection,
        "metadata": {
            "split_kind": SPLIT_KIND_TRAIN_CV,
            "experiment_root": str(Path(experiment_root).resolve()),
            "panel_shape": {"T": int(t_steps), "N": int(n_nodes)},
            "num_folds": int(num_folds),
            "seed": int(seed),
            "recursive": bool(recursive),
            "contiguous": bool(contiguous),
            "tolerance": float(tolerance),
            "outer_layer": dict(outer_layer["metadata"]),
            "model_selection": dict(model_selection["metadata"]),
        },
    }


def build_test_train_cv_bundle(
    experiment_root: str | Path,
    *,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    inner_num_folds: int,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> dict[str, Any]:
    outer_layer = build_outer_layer_for_test_train_cv(
        experiment_root,
        outer_num_folds=int(outer_num_folds),
        test_fold_id=int(test_fold_id),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )
    _validate_outer_layer_masks(
        outer_active_mask=np.asarray(outer_layer["outer_active_mask"], dtype=bool),
        outer_separator_mask=np.asarray(outer_layer["outer_separator_mask"], dtype=bool),
        outer_test_mask=np.asarray(outer_layer["outer_test_mask"], dtype=bool),
    )
    model_selection = build_model_selection_folds(
        experiment_root,
        num_folds=int(inner_num_folds),
        outer_active_mask=np.asarray(outer_layer["outer_active_mask"], dtype=bool),
        outer_separator_mask=np.asarray(outer_layer["outer_separator_mask"], dtype=bool),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
    )
    t_steps, n_nodes = _panel_shape(experiment_root)
    return {
        "split_kind": SPLIT_KIND_TEST_TRAIN_CV,
        "outer_layer": outer_layer,
        "model_selection": model_selection,
        "metadata": {
            "split_kind": SPLIT_KIND_TEST_TRAIN_CV,
            "experiment_root": str(Path(experiment_root).resolve()),
            "panel_shape": {"T": int(t_steps), "N": int(n_nodes)},
            "outer_num_folds": int(outer_num_folds),
            "test_fold_id": int(test_fold_id),
            "inner_num_folds": int(inner_num_folds),
            "seed": int(seed),
            "recursive": bool(recursive),
            "contiguous": bool(contiguous),
            "tolerance": float(tolerance),
            "outer_layer": dict(outer_layer["metadata"]),
            "outer_blanket_summary": dict(outer_layer["blanket_summary"]),
            "model_selection": dict(model_selection["metadata"]),
        },
    }


def write_split_bundle(
    output_root: str | Path,
    bundle: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    root = Path(output_root).resolve()
    if root.exists() and bool(overwrite):
        shutil.rmtree(io_path(root))
    root.mkdir(parents=True, exist_ok=True)
    outer_layer = bundle["outer_layer"]
    model_selection = bundle["model_selection"]
    np.savez(
        io_path(root / "outer_layer.npz"),
        outer_active_mask=np.asarray(outer_layer["outer_active_mask"], dtype=bool),
        outer_separator_mask=np.asarray(outer_layer["outer_separator_mask"], dtype=bool),
        outer_test_mask=np.asarray(outer_layer["outer_test_mask"], dtype=bool),
    )
    np.savez(
        io_path(root / "model_selection_folds.npz"),
        training_masks=np.asarray(model_selection["training_masks"], dtype=bool),
        separator_masks=np.asarray(model_selection["separator_masks"], dtype=bool),
        validation_masks=np.asarray(model_selection["validation_masks"], dtype=bool),
    )
    write_csv(
        root / "fold_summary.csv",
        list(model_selection["fold_summary_rows"]),
        [
            "fold_id",
            "num_training_slots",
            "num_separator_slots",
            "num_validation_slots",
        ],
    )
    _save_yaml(root / "bundle_metadata.yaml", dict(bundle["metadata"]))
    return root


def _build_split_for_experiment(
    experiment_root: str | Path,
    *,
    split_kind: str,
    num_folds: int,
    outer_num_folds: int = DEFAULT_OUTER_NUM_FOLDS,
    test_fold_id: int = DEFAULT_TEST_FOLD_ID,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
    overwrite: bool = False,
) -> Path:
    if str(split_kind) == SPLIT_KIND_TRAIN_CV:
        bundle = build_train_cv_bundle(
            experiment_root,
            num_folds=int(num_folds),
            seed=int(seed),
            recursive=bool(recursive),
            contiguous=bool(contiguous),
            tolerance=float(tolerance),
        )
        output_root = train_cv_split_output_root(experiment_root, num_folds=int(num_folds))
    else:
        bundle = build_test_train_cv_bundle(
            experiment_root,
            outer_num_folds=int(outer_num_folds),
            test_fold_id=int(test_fold_id),
            inner_num_folds=int(num_folds),
            seed=int(seed),
            recursive=bool(recursive),
            contiguous=bool(contiguous),
            tolerance=float(tolerance),
        )
        output_root = test_train_cv_split_output_root(
            experiment_root,
            outer_num_folds=int(outer_num_folds),
            test_fold_id=int(test_fold_id),
            inner_num_folds=int(num_folds),
        )
    return write_split_bundle(output_root, bundle, overwrite=bool(overwrite))


def _run_build_cv_folds_for_experiment(
    experiment_root: str | Path,
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
    overwrite: bool = False,
) -> Path:
    return _build_split_for_experiment(
        experiment_root,
        split_kind=SPLIT_KIND_TRAIN_CV,
        num_folds=int(num_folds),
        seed=int(seed),
        recursive=bool(recursive),
        contiguous=bool(contiguous),
        tolerance=float(tolerance),
        overwrite=bool(overwrite),
    )


def run_build_cv_folds(
    generation_manifest_path: str | Path,
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
    overwrite: bool = False,
) -> list[Path]:
    output_paths: list[Path] = []
    for row in read_csv_rows(generation_manifest_path):
        experiment_path = str(row.get("experiment_path", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        output_paths.append(
            _run_build_cv_folds_for_experiment(
                experiment_path,
                num_folds=int(num_folds),
                seed=int(seed),
                recursive=bool(recursive),
                contiguous=bool(contiguous),
                tolerance=float(tolerance),
                overwrite=bool(overwrite),
            )
        )
    return output_paths
