"""Configuration file loading and normalization utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from utils.t0_path_utils import io_path


def load_yaml_config(path: str | Path):
    """Load a YAML configuration file using OmegaConf."""
    return OmegaConf.load(io_path(path))


def to_plain_mapping(value: Any) -> dict[str, Any]:
    """Resolve a mapping-like config object into a plain Python dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Expected a mapping-like YAML object.")
    return deepcopy(container)


def deep_merge_mappings(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge two plain Python mappings without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_mappings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a resolved plain Python mapping."""
    container = OmegaConf.to_container(load_yaml_config(path), resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"Spec at {path} must be a mapping.")
    return deepcopy(container)


def assign_nested_value(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Assign a value to a nested dict using a tuple path."""
    cursor = target
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value
