"""Configuration file loading utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from utils.t0_path_utils import io_path


def load_yaml_config(path: str | Path):
    """Load a YAML configuration file using OmegaConf."""
    return OmegaConf.load(io_path(path))
