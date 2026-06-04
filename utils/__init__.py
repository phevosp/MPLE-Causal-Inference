"""Utility modules for the MPLE causal inference pipeline.

Organization: 9-tier hierarchical structure for clean dependency management.

TIER 0: INFRASTRUCTURE (no domain dependencies)
  t0_path_utils: Path resolution, file existence checks (Windows-safe I/O)
  t0_config_utils: YAML configuration file loading
  t0_csv_utils: Generic CSV writing, value formatting for reports

TIER 1: MATRIX I/O
  t1_matrix_io: Gamma matrix loading, loss mask saving

TIER 2: NUMERICAL OPERATIONS (depends on Tier 0)
  t2_normalization: Matrix/vector normalization, graph normalization
  t2_summary_statistics: Statistical summaries (quantiles, means on masks, finite values)

TIER 3: MODEL DEFINITIONS (depends on Tier 0-2)
  t3_model_artifacts: ModelArtifacts dataclass, build/save/load artifacts
  t3_interaction_matrices: Interaction matrix composition and application
  t3_field_operations: Field composition, scaling, projection, truncation
  t3_field_generation: Synthetic field generation and spec parsing

TIER 4: PARAMETER MANAGEMENT (depends on Tier 0-3)
  t4_scalar_parameters: Scalar parameter validation and retrieval
  t4_parameter_packing: Theta vector packing/unpacking, parameter loading

TIER 5: DOMAIN DATA LOADING (depends on Tier 0-4)
  t5_experiment_context: Experiment panel context loading
  t5_parameter_bundles: Parameter bundle dataclass and I/O

TIER 6: WORKFLOW UTILITIES (depends on Tier 0-5)
  t6_fit_materialization: Shared fit config/materialization/execution helpers
  t6_split_management: CV fold and validation/test split mask management
  t6_intervention_utils: Intervention construction and artifact management
  t6_posterior_predictive_manifest: Target resolution and fit lookup
  t6_posterior_predictive_summary: Manifest row building and metadata

TIER 7: VALIDATION & EVALUATION (depends on Tier 0-6)
  t7_validation_metrics: Fold/test metric evaluation, Brier score, ECE
  t7_cv_aggregation: CV fold aggregation, candidate scoring

TIER 8: OUTPUT GENERATION (depends on Tier 0-7)
  t8_posterior_predictive_sim: Outcome simulation, panel statistics
  t8_posterior_predictive_reporting: Summary statistics reporting
  t8_fit_outputs: MPLE fit diagnostics, summary tables, saved output artifacts
  t8_output_writers: Output table writers, formatting helpers

TYPICAL IMPORT PATTERNS
========================

For model fitting (mple.py, run_fit_pipeline.py):
  from utils.t3_model_artifacts import ModelArtifacts, load_model_artifacts
  from utils.t4_parameter_packing import pack_theta, unpack_theta, parameter_names
  from utils.t5_experiment_context import load_experiment_panel_context
  from utils.t6_fit_materialization import materialize_fit_root, execute_fit_root

For validation (run_cv_folds.py, run_test_evaluation.py):
  from utils.t6_split_management import load_model_selection_split_masks
  from utils.t7_validation_metrics import evaluate_fold_metrics
  from utils.t7_cv_aggregation import build_candidate_score_row

For posterior predictive (run_posterior_predictive.py):
  from utils.t5_parameter_bundles import load_fit_parameter_bundle
  from utils.t6_posterior_predictive_manifest import resolve_target_pairs
  from utils.t8_posterior_predictive_sim import simulate_outcomes_for_bundle
  from utils.t8_output_writers import write_predictive_stats_tables

For data generation (data/synthetic_data_generation.py):
  from utils.t3_field_generation import build_synthetic_field_with_layout, parse_synthetic_field_spec
  from utils.t3_model_artifacts import save_model_artifacts
  from utils.t4_scalar_parameters import get_xi
"""
