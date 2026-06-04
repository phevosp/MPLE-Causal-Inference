"""Utility modules for the MPLE causal inference pipeline.

Module organization:

INFRASTRUCTURE (pure utilities with no domain dependencies):
  - path_utils: Path resolution and file existence checks (Windows-safe I/O).
  - config_utils: YAML configuration file loading.
  - csv_utils: Generic CSV writing and value formatting for reports.

NUMERICAL OPERATIONS:
  - normalization: Matrix and vector normalization (infinity norm, graph normalization).
  - summary_statistics: Statistical summaries (quantiles, means over masks, finite-value filtering).

DOMAIN DATA LOADING:
  - io_utils: Domain-specific I/O (gamma matrix loading, loss mask saving, predictive output writers).
  - loading_utils: Experiment panel context and parameter bundle loading/saving.

MODEL MANAGEMENT:
  - model_utils: Model artifacts, field generation, synthetic field specs, parameter packing/unpacking.
  - split_artifact_utils: CV and validation/test split mask management.

WORKFLOWS:
  - intervention_utils: Intervention construction and artifact management.
  - posterior_predictive_job_utils: Posterior-predictive target resolution and manifest building.
  - validation_metric_utils: Fold metrics, test metrics, CV result aggregation.
  - posterior_predictive_utils: Outcome simulation and statistical reporting.
"""
