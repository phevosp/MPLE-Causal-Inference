"""Parameter packing/unpacking for theta vectors."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from utils.t0_path_utils import io_path
from utils.t0_config_utils import load_yaml_config
from utils.t3_model_artifacts import (
    ModelArtifacts,
    SpectralLowRankStructure,
    SCALAR_PARAMETER_ORDER,
    OPTIMIZER_MODE_NUCLEAR_NORM,
)
from utils.t3_field_operations import compose_field_matrix_from_theta
from utils.t4_scalar_parameters import (
    scalar_parameter_names,
    validate_fixed_scalar_params,
    free_scalar_parameter_names,
)


def parameter_names(
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    n_nodes = artifacts.gamma_matrix.shape[0]
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
        field_keys = [
            f"F::time_{time_idx}::node_{node_idx}"
            for time_idx in range(artifacts.t_steps)
            for node_idx in range(n_nodes)
        ]
    else:
        field_keys = [
            f"U::node_{node_idx}::r_{rank_idx}"
            for node_idx in range(n_nodes)
            for rank_idx in range(artifacts.latent_rank)
        ]
        field_keys.extend(
            f"V::time_{time_idx}::r_{rank_idx}"
            for time_idx in range(artifacts.t_steps)
            for rank_idx in range(artifacts.latent_rank)
        )
    return field_keys + free_scalar_parameter_names(fixed_scalar_params)


def summarize_theta_for_logging(param_names: list[str], theta: np.ndarray) -> str:
    scalar_names = set(SCALAR_PARAMETER_ORDER)
    scalar_parts = [
        f"{key}: {value:+.4f}"
        for key, value in zip(param_names, theta)
        if key in scalar_names
    ]
    if scalar_parts:
        return "  " + ",  ".join(scalar_parts)
    return "  no free scalar parameters"


def unpack_theta(
    theta: np.ndarray,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> dict[str, object]:
    theta = np.asarray(theta, dtype=float)
    n_nodes = artifacts.gamma_matrix.shape[0]
    t_steps = artifacts.t_steps
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
        n_field = t_steps * n_nodes
        n_u = 0
        n_v = 0
    else:
        n_field = 0
        n_u = n_nodes * artifacts.latent_rank
        n_v = t_steps * artifacts.latent_rank
    tail = len(free_scalar_parameter_names(fixed_scalar_params))
    expected_length = n_field + n_u + n_v + tail
    if len(theta) != expected_length:
        raise ValueError(
            f"Theta length {len(theta)} does not match expected length {expected_length}."
        )
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
        field_matrix = theta[:n_field].reshape(t_steps, n_nodes)
        node_factors = np.zeros((n_nodes, 0), dtype=float)
        time_factors = np.zeros((t_steps, 0), dtype=float)
        cursor = n_field
    else:
        field_matrix = None
        node_factors = theta[:n_u].reshape(n_nodes, artifacts.latent_rank)
        time_factors = theta[n_u : n_u + n_v].reshape(t_steps, artifacts.latent_rank)
        cursor = n_u + n_v
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    scalar_values: dict[str, float] = {}
    for name in scalar_parameter_names():
        if name in fixed:
            scalar_values[name] = float(fixed[name])
        else:
            scalar_values[name] = float(theta[cursor])
            cursor += 1
    return {
        "node_factors": node_factors,
        "time_factors": time_factors,
        "field_matrix": field_matrix,
        "beta": scalar_values["beta"],
        "xi": scalar_values["xi"],
        "eta": scalar_values["eta"],
    }


def pack_theta(
    theta_parts: dict[str, object],
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if artifacts.optimizer_mode == OPTIMIZER_MODE_NUCLEAR_NORM:
        if theta_parts.get("field_matrix") is None:
            raise ValueError("Full-matrix parameterization requires field_matrix.")
        field_block = np.asarray(theta_parts["field_matrix"], dtype=float).reshape(-1)
    else:
        if theta_parts["node_factors"] is None or theta_parts["time_factors"] is None:
            raise ValueError(
                "Factorized parameterization requires node_factors and time_factors."
            )
        field_block = np.concatenate(
            [
                np.asarray(theta_parts["node_factors"], dtype=float).reshape(-1),
                np.asarray(theta_parts["time_factors"], dtype=float).reshape(-1),
            ]
        )
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    tail = [
        np.array([float(theta_parts[name])], dtype=float)
        for name in scalar_parameter_names()
        if name not in fixed
    ]
    return np.concatenate([field_block, *tail])


def load_true_parameters(
    config,
    artifacts: ModelArtifacts,
    fixed_scalar_params: dict[str, float] | None = None,
) -> np.ndarray:
    if artifacts.field_matrix is None:
        raise ValueError("Missing latent truth field in field_artifacts.")
    truth_artifacts = ModelArtifacts(
        gamma_matrix=artifacts.gamma_matrix,
        t_steps=artifacts.t_steps,
        latent_rank=0,
        optimizer_mode=OPTIMIZER_MODE_NUCLEAR_NORM,
        field_matrix=artifacts.field_matrix,
    )
    theta_parts = {
        "field_matrix": artifacts.field_matrix,
        "beta": float(config.estimation_params.beta),
        "xi": float(config.estimation_params.xi),
        "eta": float(config.estimation_params.eta),
    }
    return pack_theta(
        theta_parts,
        truth_artifacts,
        fixed_scalar_params=fixed_scalar_params,
    )
