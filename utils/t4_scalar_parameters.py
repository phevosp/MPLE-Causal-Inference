"""Scalar parameter configuration and validation."""

from __future__ import annotations

from utils.t3_model_artifacts import (
    VALID_OPTIMIZER_MODES,
    DEFAULT_LATENT_RANK,
    OPTIMIZER_MODE_EXACT_RANK_MANIFOLD,
    OPTIMIZER_MODE_NO_EXTERNAL_FIELD,
    ModelArtifacts,
    SCALAR_PARAMETER_ORDER,
)


def scalar_parameter_names() -> list[str]:
    return list(SCALAR_PARAMETER_ORDER)


def validate_fixed_scalar_params(
    fixed_scalar_params: dict[str, float] | None,
) -> dict[str, float]:
    fixed_scalar_params = {
        str(key): float(value) for key, value in (fixed_scalar_params or {}).items()
    }
    invalid = sorted(set(fixed_scalar_params) - set(SCALAR_PARAMETER_ORDER))
    if invalid:
        raise ValueError(f"Unknown fixed scalar parameter(s): {', '.join(invalid)}.")
    return fixed_scalar_params


def free_scalar_parameter_names(
    fixed_scalar_params: dict[str, float] | None = None,
) -> list[str]:
    fixed = validate_fixed_scalar_params(fixed_scalar_params)
    return [name for name in scalar_parameter_names() if name not in fixed]




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


def uses_full_matrix_parameterization(
    artifacts: ModelArtifacts,
) -> bool:
    return artifacts.optimizer_mode == "nuclear_norm"


def get_B(config) -> float:
    if "B" not in config.global_params:
        raise KeyError("global_params.B is required.")
    return float(config.global_params.B)


def get_xi(config) -> float:
    if "xi" not in config.estimation_params:
        raise KeyError("estimation_params.xi is required.")
    return float(config.estimation_params.xi)
