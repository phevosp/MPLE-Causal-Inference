"""Backward-compat re-exports from restructured I/O modules."""

from __future__ import annotations

# Tier 1: Matrix I/O
from utils.t1_matrix_io import load_gamma_matrix, save_loss_mask  # noqa: F401

# Tier 0: Infrastructure
from utils.t0_path_utils import io_path  # noqa: F401

# Tier 8: Output writers and formatting helpers
from utils.t8_output_writers import (  # noqa: F401
    _as_float,
    _metric_or_inf,
    write_counterfactual_summary_tables,
    write_observed_predictive_summary_tables,
    write_predictive_stats_tables,
)
