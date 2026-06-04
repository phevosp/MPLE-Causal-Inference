from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

from utils.t0_csv_utils import write_csv
from utils.t1_matrix_io import load_gamma_matrix
from utils.t0_path_utils import io_path
from pipeline_specs import read_csv_manifest


DEFAULT_GAMMA_TOLERANCE = 1.0e-12
DEFAULT_NUM_FOLDS = 5
ROLE_CODE_TRAINING = 0
ROLE_CODE_SEPARATOR = 1
ROLE_CODE_VALIDATION = 2
ROLE_NAME_BY_CODE = {
    ROLE_CODE_TRAINING: "training",
    ROLE_CODE_SEPARATOR: "separator",
    ROLE_CODE_VALIDATION: "validation",
}


def _min_time_block_sizes_for_folds(num_folds: int) -> tuple[int, ...]:
    """Generate minimum time block sizes for k-fold CV.

    First block gets 1 step, remaining blocks get 2 steps each (for transition support).
    """
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
            "pymetis is required for CV graph partitioning. Install or refresh the "
            "Pixi environment so it includes the conda-forge 'pymetis' package."
        ) from exc
    return pymetis




def _gamma_artifact_kind(experiment_root: str | Path) -> str:
    root = Path(experiment_root)
    sparse_path = root / "gamma_matrix_sparse.npz"
    dense_path = root / "gamma_matrix.npy"
    if sparse_path.exists():
        return "sparse_npz"
    if dense_path.exists():
        return "dense_npy"
    raise FileNotFoundError(f"Missing gamma matrix artifact in {root}.")


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
    adjacency_to_other_partition = np.zeros(num_vertices, dtype=bool)
    cut_edge_count = 0

    for vertex_index, neighbors in enumerate(adjacency):
        vertex_partition = int(membership[vertex_index])
        for neighbor_index in neighbors:
            neighbor_partition = int(membership[neighbor_index])
            if neighbor_partition == vertex_partition:
                continue
            adjacency_to_other_partition[vertex_index] = True
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
        "adjacency_to_other_partition": adjacency_to_other_partition,
        "separator_sets": separator_sets,
        "partition_sizes": [int(value) for value in partition_sizes.tolist()],
        "separator_sizes": [int(len(values)) for values in separator_sets],
        "separator_union_vertex_count": separator_union_vertex_count,
        "separator_union_vertex_fraction": (
            float(separator_union_vertex_count) / float(num_vertices)
            if num_vertices > 0
            else 0.0
        ),
        "total_separator_memberships": int(sum(len(values) for values in separator_sets)),
        "cut_edge_count": int(cut_edge_count),
        "partition_size_mean": mean_partition_size,
        "partition_size_min": int(np.min(partition_sizes)) if partition_sizes.size else 0,
        "partition_size_max": int(np.max(partition_sizes)) if partition_sizes.size else 0,
        "partition_size_std": (
            float(np.std(partition_sizes, ddof=0)) if partition_sizes.size else 0.0
        ),
        "max_relative_deviation_from_mean": (
            float(np.max(np.abs(partition_sizes - mean_partition_size) / mean_partition_size))
            if mean_partition_size > 0.0
            else 0.0
        ),
    }


def _load_node_index(
    experiment_root: str | Path,
    *,
    expected_vertices: int,
) -> tuple[list[dict[str, str]], list[str]]:
    node_index_path = Path(experiment_root) / "node_index.csv"
    if not node_index_path.exists():
        return [], []

    with node_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    if len(rows) != expected_vertices:
        raise ValueError(
            f"node_index.csv row count {len(rows)} does not match Gamma dimension "
            f"{expected_vertices} in {experiment_root}."
        )
    passthrough_columns = [
        column
        for column in columns
        if column
        not in {
            "vertex_index",
            "partition_id",
            "separator_set_id",
            "vertex_partition_id",
            "time_index",
            "time_block_id",
            "block_position",
            "is_transition_step",
            "cv_fold_id",
            "validation_partition_id",
            "spatial_separator_set_id",
            "transition_from_partition_id",
            "transition_to_partition_id",
            "transition_time_index",
        }
    ]
    return rows, passthrough_columns


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
        # Real-data experiment roots may include the initial-state calendar row in time_index.csv
        # while panel_data.npz only stores the transition panel x[0:T]. Align the index to the
        # modeled transition horizon by dropping the initial row.
        raw_rows = raw_rows[1:]
    if len(raw_rows) != inferred_t_steps:
        raise ValueError(
            f"time_index.csv row count {len(raw_rows)} does not match panel horizon "
            f"{inferred_t_steps} in {experiment_root}."
        )
    passthrough_columns = [
        column
        for column in fieldnames
        if column
        not in {
            "time_index",
            "time_block_id",
            "block_position",
            "is_transition_step",
        }
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
    role_codes = np.full(
        (num_folds, t_steps, num_vertices),
        ROLE_CODE_TRAINING,
        dtype=np.int8,
    )

    for fold_index in range(num_folds):
        for time_index in range(t_steps):
            block_zero_based = int(time_block_ids[time_index]) - 1
            current_partition_id = int(
                validation_partition_ids_by_fold_block[fold_index, block_zero_based]
            )
            current_partition_set = partition_sets[current_partition_id - 1]
            current_separator_set = separator_sets[current_partition_id - 1]

            if bool(is_transition_step[time_index]):
                previous_partition_id = int(
                    validation_partition_ids_by_fold_block[fold_index, block_zero_based - 1]
                )
                previous_partition_set = partition_sets[previous_partition_id - 1]
                transition_separator = (
                    set(current_separator_set)
                    | set(previous_partition_set)
                    | set(current_partition_set)
                )
                if transition_separator:
                    role_codes[
                        fold_index,
                        time_index,
                        np.asarray(sorted(transition_separator), dtype=int),
                    ] = ROLE_CODE_SEPARATOR
                continue

            if current_separator_set & current_partition_set:
                raise ValueError(
                    f"Separator set S_{current_partition_id} overlaps its partition "
                    f"C_{current_partition_id}, which violates the artifact contract."
                )
            if current_separator_set:
                role_codes[
                    fold_index,
                    time_index,
                    np.asarray(sorted(current_separator_set), dtype=int),
                ] = ROLE_CODE_SEPARATOR
            if current_partition_set:
                role_codes[
                    fold_index,
                    time_index,
                    np.asarray(sorted(current_partition_set), dtype=int),
                ] = ROLE_CODE_VALIDATION

    return role_codes


def _build_time_block_rows(
    time_rows: list[dict[str, object]],
    time_columns: list[str],
    *,
    time_block_ids: np.ndarray,
    block_positions: np.ndarray,
    is_transition_step: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for time_index, base_row in enumerate(time_rows):
        row = {
            "time_index": int(time_index),
            "time_block_id": int(time_block_ids[time_index]),
            "block_position": int(block_positions[time_index]),
            "is_transition_step": bool(is_transition_step[time_index]),
        }
        for column in time_columns:
            row[column] = base_row.get(column, "")
        rows.append(row)
    return rows


def _build_fold_schedule_rows(
    validation_partition_ids_by_fold_block: np.ndarray,
    block_start_indices: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_folds = int(validation_partition_ids_by_fold_block.shape[0])
    for fold_index in range(num_folds):
        for block_index in range(num_folds):
            validation_partition_id = int(
                validation_partition_ids_by_fold_block[fold_index, block_index]
            )
            row = {
                "cv_fold_id": int(fold_index + 1),
                "time_block_id": int(block_index + 1),
                "validation_partition_id": validation_partition_id,
                "spatial_separator_set_id": validation_partition_id,
                "transition_from_partition_id": "",
                "transition_to_partition_id": "",
                "transition_time_index": "",
            }
            if block_index > 0:
                row["transition_from_partition_id"] = int(
                    validation_partition_ids_by_fold_block[fold_index, block_index - 1]
                )
                row["transition_to_partition_id"] = validation_partition_id
                row["transition_time_index"] = int(block_start_indices[block_index])
            rows.append(row)
    return rows


def _build_fold_role_count_rows(
    role_codes: np.ndarray,
    time_block_ids: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_folds = int(role_codes.shape[0])
    num_blocks = int(np.max(time_block_ids))
    for fold_index in range(num_folds):
        for block_id in range(1, num_blocks + 1):
            block_mask = time_block_ids == block_id
            block_slice = role_codes[fold_index, block_mask, :]
            num_time_indices = int(np.count_nonzero(block_mask))
            for role_code, role_name in ROLE_NAME_BY_CODE.items():
                total_assignments = int(np.count_nonzero(block_slice == role_code))
                rows.append(
                    {
                        "cv_fold_id": int(fold_index + 1),
                        "time_block_id": int(block_id),
                        "role_code": int(role_code),
                        "role_name": role_name,
                        "num_time_indices": int(num_time_indices),
                        "total_vertex_assignments": int(total_assignments),
                        "mean_vertices_per_time_index": (
                            float(total_assignments) / float(num_time_indices)
                            if num_time_indices > 0
                            else 0.0
                        ),
                    }
                )
    return rows


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
                    if {
                        vertex_role,
                        neighbor_role,
                    } == {
                        ROLE_CODE_VALIDATION,
                        ROLE_CODE_TRAINING,
                    }:
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
    total_violation_count = int(
        spatial_violation_edge_count + temporal_violation_edge_count
    )
    violations_by_fold = [
        int(spatial_count + temporal_count)
        for spatial_count, temporal_count in zip(
            spatial_violations_by_fold,
            temporal_violations_by_fold,
        )
    ]
    return {
        "blanket_validation_passed": bool(total_violation_count == 0),
        "num_folds": int(num_folds),
        "num_vertices": int(num_vertices),
        "num_time_steps": int(num_time_steps),
        "spatial_violation_edge_count": spatial_violation_edge_count,
        "temporal_violation_edge_count": temporal_violation_edge_count,
        "total_violation_count": total_violation_count,
        "num_folds_with_any_violation": int(
            sum(count > 0 for count in violations_by_fold)
        ),
        "violations_by_fold": [int(value) for value in violations_by_fold],
        "spatial_violations_by_fold": [
            int(value) for value in spatial_violations_by_fold
        ],
        "temporal_violations_by_fold": [
            int(value) for value in temporal_violations_by_fold
        ],
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


def _summarize_role_coverage_counts(role_codes: np.ndarray) -> dict[str, Any]:
    role_array = np.asarray(role_codes, dtype=int)
    num_folds = int(role_array.shape[0])
    num_time_steps = int(role_array.shape[1])
    num_vertices = int(role_array.shape[2])
    total_assignment_slots_per_vertex = int(num_folds * num_time_steps)

    validation_counts = np.count_nonzero(
        role_array == ROLE_CODE_VALIDATION,
        axis=(0, 1),
    ).astype(int)
    separator_counts = np.count_nonzero(
        role_array == ROLE_CODE_SEPARATOR,
        axis=(0, 1),
    ).astype(int)
    training_counts = np.count_nonzero(
        role_array == ROLE_CODE_TRAINING,
        axis=(0, 1),
    ).astype(int)

    return {
        "num_folds": int(num_folds),
        "num_vertices": int(num_vertices),
        "num_time_steps": int(num_time_steps),
        "total_assignment_slots_per_vertex": total_assignment_slots_per_vertex,
        "validation_count_min": int(np.min(validation_counts)) if validation_counts.size else 0,
        "validation_count_mean": (
            float(np.mean(validation_counts)) if validation_counts.size else 0.0
        ),
        "validation_count_max": int(np.max(validation_counts)) if validation_counts.size else 0,
        "separator_count_min": int(np.min(separator_counts)) if separator_counts.size else 0,
        "separator_count_mean": (
            float(np.mean(separator_counts)) if separator_counts.size else 0.0
        ),
        "separator_count_max": int(np.max(separator_counts)) if separator_counts.size else 0,
        "training_count_min": int(np.min(training_counts)) if training_counts.size else 0,
        "training_count_mean": (
            float(np.mean(training_counts)) if training_counts.size else 0.0
        ),
        "training_count_max": int(np.max(training_counts)) if training_counts.size else 0,
        "num_vertices_with_zero_validation_count": int(
            np.count_nonzero(validation_counts == 0)
        ),
        "num_vertices_with_zero_training_count": int(
            np.count_nonzero(training_counts == 0)
        ),
        "num_vertices_with_validation_count_lt_2": int(
            np.count_nonzero(validation_counts < 2)
        ),
        "num_vertices_with_training_count_lt_2": int(
            np.count_nonzero(training_counts < 2)
        ),
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    os.makedirs(io_path(path.parent), exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), io_path(path))


def _build_cv_fold_artifacts(
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
    gamma_artifact_kind = _gamma_artifact_kind(experiment_path)
    gamma_matrix = load_gamma_matrix(experiment_path)
    gamma_validation = validate_gamma_matrix(gamma_matrix, tolerance=tolerance)

    num_vertices = int(gamma_matrix.shape[0])
    if num_folds > num_vertices:
        raise ValueError(
            f"num_folds={num_folds} exceeds the number of vertices {num_vertices}."
        )

    adjacency, num_edges = _support_adjacency_from_gamma(
        gamma_matrix,
        tolerance=tolerance,
    )
    t_steps, time_rows, time_columns, time_source = _load_time_index(experiment_path)
    time_plan = _build_time_block_plan(t_steps, num_folds=num_folds)

    start_time = time.perf_counter()
    pymetis = _load_pymetis()
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
    if np.any(membership_array < 0) or np.any(membership_array >= num_folds):
        raise ValueError("pymetis returned partition ids outside the requested range.")

    partition_sets = _build_partition_vertex_sets(membership_array, num_folds=num_folds)
    partition_metrics = _compute_partition_metrics(
        adjacency,
        membership_array,
        num_folds=num_folds,
    )
    separator_sets = partition_metrics["separator_sets"]
    validation_schedule = _build_validation_schedule(num_folds=num_folds)
    role_codes = _build_role_tensor(
        partition_sets=partition_sets,
        separator_sets=separator_sets,
        validation_partition_ids_by_fold_block=validation_schedule,
        time_block_ids=time_plan["time_block_ids"],
        is_transition_step=time_plan["is_transition_step"],
        num_vertices=num_vertices,
    )

    if not np.all(np.isin(role_codes, tuple(ROLE_NAME_BY_CODE))):
        raise ValueError("Role tensor contains unknown role codes.")

    node_rows, node_columns = _load_node_index(
        experiment_path,
        expected_vertices=num_vertices,
    )
    assignment_columns = ["vertex_index", "partition_id", *node_columns]
    assignment_rows: list[dict[str, object]] = []
    for vertex_index in range(num_vertices):
        row = {
            "vertex_index": int(vertex_index),
            "partition_id": int(membership_array[vertex_index] + 1),
        }
        if node_rows:
            for column in node_columns:
                row[column] = node_rows[vertex_index].get(column, "")
        assignment_rows.append(row)

    separator_columns = [
        "separator_set_id",
        "vertex_index",
        "vertex_partition_id",
        *node_columns,
    ]
    separator_rows: list[dict[str, object]] = []
    for separator_index, separator_vertices in enumerate(separator_sets, start=1):
        for vertex_index in sorted(separator_vertices):
            row = {
                "separator_set_id": int(separator_index),
                "vertex_index": int(vertex_index),
                "vertex_partition_id": int(membership_array[vertex_index] + 1),
            }
            if node_rows:
                for column in node_columns:
                    row[column] = node_rows[vertex_index].get(column, "")
            separator_rows.append(row)

    time_block_rows = _build_time_block_rows(
        time_rows,
        time_columns,
        time_block_ids=time_plan["time_block_ids"],
        block_positions=time_plan["block_positions"],
        is_transition_step=time_plan["is_transition_step"],
    )
    fold_schedule_rows = _build_fold_schedule_rows(
        validation_schedule,
        time_plan["block_start_indices"],
    )
    fold_role_count_rows = _build_fold_role_count_rows(
        role_codes,
        time_plan["time_block_ids"],
    )

    spatial_partition_metadata = {
        "experiment_root": str(experiment_path),
        "num_folds": int(num_folds),
        "seed": int(seed),
        "partitioner": "pymetis",
        "pymetis_seed_supported": False,
        "requested_seed": int(seed),
        "runtime_seconds": runtime_seconds,
        "gamma_artifact_kind": gamma_artifact_kind,
        "gamma_validation": {"passed": True, **gamma_validation},
        "graph_preprocessing": {
            "support_threshold": float(tolerance),
            "interpreted_as_unweighted_support_graph": True,
            "symmetry_enforced_via_validation_only": True,
            "zero_diagonal_enforced_via_validation_only": True,
        },
        "metis": {
            "mode": "recursive" if recursive else "kway",
            "contiguous": bool(contiguous),
            "reported_cutcount": int(metis_cutcount),
        },
        "metrics": {
            "num_vertices": int(num_vertices),
            "num_edges": int(num_edges),
            "partition_sizes": partition_metrics["partition_sizes"],
            "cut_edge_count": int(partition_metrics["cut_edge_count"]),
            "separator_sizes": partition_metrics["separator_sizes"],
            "separator_union_vertex_count": int(
                partition_metrics["separator_union_vertex_count"]
            ),
            "separator_union_vertex_fraction": float(
                partition_metrics["separator_union_vertex_fraction"]
            ),
            "total_separator_memberships": int(
                partition_metrics["total_separator_memberships"]
            ),
            "partition_size_mean": float(partition_metrics["partition_size_mean"]),
            "partition_size_min": int(partition_metrics["partition_size_min"]),
            "partition_size_max": int(partition_metrics["partition_size_max"]),
            "partition_size_std": float(partition_metrics["partition_size_std"]),
            "max_relative_deviation_from_mean": float(
                partition_metrics["max_relative_deviation_from_mean"]
            ),
        },
    }
    coverage_count_summary = _summarize_role_coverage_counts(role_codes)
    min_block_sizes = _min_time_block_sizes_for_folds(num_folds)
    spatiotemporal_metadata = {
        "experiment_root": str(experiment_path),
        "num_cv_folds": int(num_folds),
        "num_vertices": int(num_vertices),
        "num_time_steps": int(t_steps),
        "time_block_sizes": time_plan["block_sizes"],
        "transition_time_indices": time_plan["transition_time_indices"],
        "validation_partition_ids_by_fold_block": validation_schedule.tolist(),
        "time_source": str(time_source),
        "minimum_supported_time_steps": int(sum(min_block_sizes)),
        "time_block_rule": {
            "minimum_block_sizes": [int(value) for value in min_block_sizes],
            "transition_step_location": "first_step_of_next_block",
            "wraparound_transition": False,
        },
        "validation_schedule": validation_schedule.tolist(),
        "time_blocks": {
            "sizes": time_plan["block_sizes"],
            "start_indices": time_plan["block_start_indices"],
            "end_indices": time_plan["block_end_indices"],
            "transition_time_indices": time_plan["transition_time_indices"],
        },
        "role_codes": {
            "training": int(ROLE_CODE_TRAINING),
            "separator": int(ROLE_CODE_SEPARATOR),
            "validation": int(ROLE_CODE_VALIDATION),
        },
        "tensor_shape": [int(value) for value in role_codes.shape],
        "validation_assignments_by_fold": [
            int(np.count_nonzero(role_codes[fold_index] == ROLE_CODE_VALIDATION))
            for fold_index in range(num_folds)
        ],
        "separator_assignments_by_fold": [
            int(np.count_nonzero(role_codes[fold_index] == ROLE_CODE_SEPARATOR))
            for fold_index in range(num_folds)
        ],
        "training_assignments_by_fold": [
            int(np.count_nonzero(role_codes[fold_index] == ROLE_CODE_TRAINING))
            for fold_index in range(num_folds)
        ],
        "validation_empty_on_transition_steps": True,
        "role_code_map": {name: int(code) for code, name in ROLE_NAME_BY_CODE.items()},
        "coverage_counts": coverage_count_summary,
    }

    markov_blanket_summary = _summarize_markov_blanket_validation(
        adjacency,
        role_codes,
    )
    if not bool(markov_blanket_summary["blanket_validation_passed"]):
        raise ValueError(
            "Constructed CV folds failed the spatiotemporal Markov-blanket validation."
        )

    return {
        "experiment_root": experiment_path,
        "num_folds": int(num_folds),
        "assignment_columns": assignment_columns,
        "assignment_rows": assignment_rows,
        "separator_columns": separator_columns,
        "separator_rows": separator_rows,
        "time_columns": time_columns,
        "time_block_rows": time_block_rows,
        "fold_schedule_rows": fold_schedule_rows,
        "fold_role_count_rows": fold_role_count_rows,
        "time_plan": time_plan,
        "validation_schedule": validation_schedule,
        "role_codes": role_codes,
        "spatial_partition_metadata": spatial_partition_metadata,
        "spatiotemporal_metadata": spatiotemporal_metadata,
        "markov_blanket_summary": markov_blanket_summary,
    }


def _write_cv_fold_artifacts(output_path: str | Path, artifacts: dict[str, Any]) -> Path:
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.mkdir(parents=True, exist_ok=True)

    time_plan = artifacts["time_plan"]
    validation_schedule = np.asarray(artifacts["validation_schedule"], dtype=np.int16)
    role_codes = np.asarray(artifacts["role_codes"], dtype=np.int8)

    np.savez(
        io_path(resolved_output_path / "fold_roles.npz"),
        role_codes=role_codes,
        time_block_ids=np.asarray(time_plan["time_block_ids"], dtype=np.int16),
        is_transition_step=np.asarray(time_plan["is_transition_step"], dtype=bool),
        validation_partition_ids_by_fold_block=validation_schedule,
    )
    write_csv(
        resolved_output_path / "vertex_assignments.csv",
        artifacts["assignment_rows"],
        artifacts["assignment_columns"],
    )
    write_csv(
        resolved_output_path / "separator_vertices.csv",
        artifacts["separator_rows"],
        artifacts["separator_columns"],
    )
    write_csv(
        resolved_output_path / "time_blocks.csv",
        artifacts["time_block_rows"],
        [
            "time_index",
            "time_block_id",
            "block_position",
            "is_transition_step",
            *artifacts["time_columns"],
        ],
    )
    write_csv(
        resolved_output_path / "fold_schedule.csv",
        artifacts["fold_schedule_rows"],
        [
            "cv_fold_id",
            "time_block_id",
            "validation_partition_id",
            "spatial_separator_set_id",
            "transition_from_partition_id",
            "transition_to_partition_id",
            "transition_time_index",
        ],
    )
    write_csv(
        resolved_output_path / "fold_role_counts.csv",
        artifacts["fold_role_count_rows"],
        [
            "cv_fold_id",
            "time_block_id",
            "role_code",
            "role_name",
            "num_time_indices",
            "total_vertex_assignments",
            "mean_vertices_per_time_index",
        ],
    )
    _write_yaml(
        resolved_output_path / "spatial_partition_metadata.yaml",
        artifacts["spatial_partition_metadata"],
    )
    _write_yaml(
        resolved_output_path / "spatiotemporal_cv_metadata.yaml",
        artifacts["spatiotemporal_metadata"],
    )
    _write_yaml(
        resolved_output_path / "markov_blanket_summary.yaml",
        artifacts["markov_blanket_summary"],
    )
    return resolved_output_path


def _run_build_cv_folds_for_experiment(
    experiment_root: str | Path,
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 0,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> Path:
    experiment_path = Path(experiment_root).resolve()
    artifacts = _build_cv_fold_artifacts(
        experiment_path,
        num_folds=num_folds,
        seed=seed,
        recursive=recursive,
        contiguous=contiguous,
        tolerance=tolerance,
    )
    output_path = (
        Path(output_dir).resolve()
        if output_dir is not None
        else experiment_path / "cv_folds" / f"folds_{num_folds}"
    )
    return _write_cv_fold_artifacts(output_path, artifacts)


def _generation_manifest_rows(
    generation_manifest_path: str | Path,
) -> list[dict[str, str]]:
    rows = read_csv_manifest(generation_manifest_path)
    if not rows:
        raise ValueError(f"No rows found in generation manifest {generation_manifest_path}.")
    return rows


def run_build_cv_folds(
    generation_manifest_path: str | Path,
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 0,
    recursive: bool = False,
    contiguous: bool = False,
    tolerance: float = DEFAULT_GAMMA_TOLERANCE,
) -> list[Path]:
    if int(num_folds) < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    output_paths: list[Path] = []
    for row in _generation_manifest_rows(generation_manifest_path):
        experiment_path = str(row.get("experiment_path", "")).strip()
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        output_paths.append(
            _run_build_cv_folds_for_experiment(
                experiment_path,
                num_folds=num_folds,
                seed=seed,
                recursive=recursive,
                contiguous=contiguous,
                tolerance=tolerance,
            )
        )
        if experiment_name:
            print(
                f"Built CV folds for {experiment_name}: {output_paths[-1]}",
                flush=True,
            )
    return output_paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build unified spatial and spatiotemporal CV fold artifacts for every experiment in a generation manifest."
        )
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_GAMMA_TOLERANCE)
    args = parser.parse_args(argv)

    output_paths = run_build_cv_folds(
        args.generation_manifest_path,
        num_folds=args.num_folds,
        seed=args.seed,
        recursive=args.recursive,
        contiguous=args.contiguous,
        tolerance=args.tolerance,
    )
    print(f"Built CV fold outputs for {len(output_paths)} experiments.")


if __name__ == "__main__":
    main(sys.argv[1:])

