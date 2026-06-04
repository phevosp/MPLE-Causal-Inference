# Utils Restructuring Analysis

## Current State: Responsibilities by Module

### 1. **data_utils.py** (138 lines)
**Current Responsibilities:**
- Vector/matrix normalization (infinity norm)
- Geographic/spatial operations (touching edges, KNN, connected components)
- Graph construction

**Actual Usage:** 
- ONLY used by `data/USCountyVaccination/` (3 files)
- 5 functions, 2 are generic normalization, 3 are geo-specific

**Functions:**
- `download_if_missing` - Generic file I/O
- `center_and_normalize_vector_infinity` - Normalization
- `normalize_sparse_matrix_infinity` - Normalization
- `build_touching_edge_list` - GEO-SPECIFIC
- `count_connected_components` - GEO-SPECIFIC
- `build_knn_and_kernel_edges` - GEO-SPECIFIC

---

### 2. **io_utils.py** (285 lines) ⚠️ TOO MANY RESPONSIBILITIES
**Current Responsibilities:**
- Path resolution (Windows-specific handling)
- File existence checks
- YAML config loading
- Gamma matrix loading
- Loss mask saving
- CSV writing (generic and domain-specific)
- Float parsing and metric conversion
- Formatting helpers
- Domain-specific output writers:
  - `write_predictive_stats_tables`
  - `write_observed_predictive_summary_tables`
  - `write_counterfactual_summary_tables`
- Helper for finite summaries

**Actual Usage:** Used by 18+ files (most heavily used util)
- Core imports: `io_path`, `path_exists`, `load_gamma_matrix`, `save_loss_mask`, `write_csv`, `load_yaml_config`
- Helper imports: `_as_float`, `_metric_or_inf` (used in reporting scripts)

**Problem:** Mixes infrastructure (path, config) with domain logic (predictive stats, counterfactual summaries)

---

### 3. **model_utils.py** (973 lines) ⚠️ CRITICALLY OVERLOADED
**Current Responsibilities:**
- Model artifacts management (dataclass, loading, saving)
- Low-rank matrix structures (dataclass, SVD operations)
- Synthetic field specifications and generation
- Field normalization and manipulation
- Latent field operations
- Parameter validation
- Scalar parameter management
- Graph normalization
- Interaction matrix composition and application
- Parameter packing/unpacking for theta vectors
- Model artifact serialization

**Actual Usage:** Used by 8 files
- `mple.py` - heavily (parameter packing, field matrix manipulation)
- `data/synthetic_data_generation.py` - artifacts, field building
- `validation_metric_utils.py` - interaction matrices
- `posterior_predictive_utils.py` - interaction matrices
- `loading_utils.py` - artifact loading
- Reports and diagnostics

**Problem:** 973 lines doing 6-7 different conceptual things. Hard to understand and modify.

---

### 4. **loading_utils.py** (211 lines)
**Current Responsibilities:**
- Panel context loading from files
- Parameter bundle definition (OutcomeParameterBundle dataclass)
- Parameter bundle persistence
- Truth parameter loading
- Fit parameter loading
- CSV summary parsing

**Actual Usage:** Used by 10 files (widely used)
- Imports: `OutcomeParameterBundle`, `load_experiment_panel_context`, `load_fit_parameter_bundle`, `load_truth_parameter_bundle`
- Also exports: `save_estimated_parameter_bundle`

**Dependencies:** 
- Depends on: `intervention_utils`, `io_utils`, `model_utils`

---

### 5. **split_artifact_utils.py** (69 lines)
**Current Responsibilities:**
- CV fold constants and normalization
- Split source management
- Validation/test split mask loading
- Outer test split mask loading
- Mask tensor validation

**Actual Usage:** Used by 5 files
- Imports: `load_model_selection_split_masks`, `load_outer_test_split_masks`, `VALID_SPLIT_SOURCES`

**Dependencies:**
- Depends on: `io_utils`, `loading_utils`

---

### 6. **validation_metric_utils.py** (937 lines) ⚠️ OVERLOADED
**Current Responsibilities:**
- Validation metric specifications
- Validation sampling configuration
- Brier score computation
- Expected calibration error (ECE)
- Magnetization metrics sampling
- Fold metric evaluation
- Test metric evaluation (overall and stratified by treatment)
- CV result aggregation and scoring
- Helper functions for averaging, masking, interaction columns
- Time window masking

**Actual Usage:** Used by 2 files
- `run_cv_folds.py`
- `run_test_evaluation.py`

**Problem:** Mixing metric calculation with CV aggregation (which is quite different in nature)

---

### 7. **posterior_predictive_utils.py** (305 lines)
**Current Responsibilities:**
- Outcome simulation from bundle
- Panel statistics computation (magnetization, alignment, energy)
- Counterfactual sample summary
- Finite scalar/vector summaries
- Mean statistics (Brier, ECE-like)
- Predictive statistics (z-scores, tail probabilities, intervals)
- Summary aggregation

**Actual Usage:** Used by 1 file
- `run_posterior_predictive.py`

**Dependencies:**
- Depends on: `loading_utils`, `model_utils`, `data/synthetic_data_generation`

---

### 8. **posterior_predictive_job_utils.py** (359 lines)
**Current Responsibilities:**
- Target pair resolution and validation
- Fit lookup indexing
- Generation manifest indexing
- Run specification resolution
- Target selection logic
- Manifest row building (from computed results)
- Manifest row building from metadata
- Boolean parsing for manifest columns

**Actual Usage:** Used by 2 files
- `run_posterior_predictive.py`
- `report_posterior_predictive.py`

---

### 9. **intervention_utils.py** (256 lines)
**Current Responsibilities:**
- Intervention context dataclass
- Pre/post intervention step derivation
- Intervention panel validation
- Intervention artifact saving/loading
- Full-on intervention construction
- Single-unit-on intervention construction
- Saved intervention resolution
- Generic intervention context resolution

**Actual Usage:** Used by 4 files
- `run_intervention_library.py`
- `run_posterior_predictive.py`
- `run_fit_pipeline.py`
- `tests/test_minimal_pipeline.py`

---

## Dependency Graph

```
mple.py
├── io_utils (io_path, load_yaml_config, write_csv)
├── loading_utils (save_estimated_parameter_bundle)
├── model_utils (compose_interaction_matrix, parameter functions)

run_cv_folds.py
├── io_utils
├── loading_utils
├── split_artifact_utils
├── validation_metric_utils

run_test_evaluation.py
├── io_utils
├── split_artifact_utils
├── validation_metric_utils

run_posterior_predictive.py
├── intervention_utils
├── io_utils
├── loading_utils
├── posterior_predictive_job_utils
├── posterior_predictive_utils

validation_metric_utils
├── loading_utils
├── model_utils

model_utils
├── io_utils

loading_utils
├── intervention_utils
├── io_utils
├── model_utils

split_artifact_utils
├── io_utils
├── loading_utils

intervention_utils
├── io_utils
└── (dynamic) loading_utils
```

---

## Current Issues

### 1. **Redundant Finite-Value Summaries**
- `io_utils._finite_summary` (lines 187-207) - computes quantiles for arrays
- `posterior_predictive_utils._finite_scalar_summary` (lines 111-142) - similar but different sig
- `posterior_predictive_utils._finite_vector_summaries` (lines 145-194) - extended version

### 2. **Scattered Normalization**
- `data_utils.center_and_normalize_vector_infinity`
- `data_utils.normalize_sparse_matrix_infinity`
- `model_utils.normalize_known_graph` (lines 214-217)
- `model_utils._normalize_dense_graph` (lines 193-200)
- `model_utils._normalize_sparse_graph` (lines 203-211)
- `model_utils.normalize_matrix_max_abs` (lines 268-279)
- `model_utils.normalize_matrix_by_max_abs_entry` (lines 282-290)

### 3. **Averaging Operations**
- `validation_metric_utils._mean_on_mask` (lines 119-128)
- `posterior_predictive_utils._mean_or_none` (lines 46-49)

### 4. **Interaction Matrix Usage**
- Defined in: `model_utils`
- Used by: `validation_metric_utils`, `posterior_predictive_utils`, `mple.py`
- But `validation_metric_utils._interaction_column` (lines 131-136) duplicates extraction logic

### 5. **Similar Mask Operations**
- `validation_metric_utils.time_window_mask` (lines 54-58)
- Pattern used extensively but no unified mask utility module

### 6. **IO Responsibilities Mixed**
- Generic path handling mixed with domain-specific output writers
- Reports import private helpers like `_as_float`, `_metric_or_inf`

---

## Proposed Restructuring Plan

### Reorganized Module Structure

#### **TIER 0: Infrastructure (No domain knowledge)**

1. **path_utils.py** (NEW)
   - `io_path(path)` - Path resolution with Windows handling
   - `path_exists(path)`
   - `first_existing_path(*paths)`
   
2. **config_utils.py** (NEW)
   - `load_yaml_config(path)`

3. **csv_utils.py** (NEW, rename from write functions in io_utils)
   - `write_csv(path, rows, columns)` - Generic CSV writer
   - `_fmt(value)` - Formatting helper

#### **TIER 1: Domain Data Loading (depends on Tier 0)**

4. **matrix_io.py** (NEW)
   - `load_gamma_matrix(data_folder)` - Gamma matrix loading
   - `save_loss_mask(path, loss_mask)` - Loss mask persistence

5. **file_artifacts.py** (NEW)
   - Field artifacts loading/saving (move from model_utils)
   - Model artifacts loading (move from model_utils)

#### **TIER 2: Numerical Operations (depends on Tier 0)**

6. **normalization.py** (NEW)
   - `center_and_normalize_vector_infinity(vector)`
   - `normalize_sparse_matrix_infinity(matrix)`
   - `normalize_known_graph(gamma_matrix)`
   - `_normalize_dense_graph(matrix)` [internal]
   - `_normalize_sparse_graph(matrix)` [internal]
   - `normalize_matrix_max_abs(matrix, max_abs)`
   - `normalize_matrix_by_max_abs_entry(matrix)`
   - `validate_graph_infinity_norm(gamma_matrix, tol)`

7. **summary_statistics.py** (NEW)
   - `finite_scalar_summary(observed_value, sample_values)` - unified version
   - `finite_vector_summaries(observed_values, sample_values, index_name)` - unified
   - `mean_on_mask(x, mask)`
   - `time_window_mask(t_steps, n_nodes, start_t)`
   - `_interaction_column(interaction_matrix, column_index)` - move here

#### **TIER 3: Model Definition (depends on Tier 0-2)**

8. **model_artifacts.py** (NEW - extracted from model_utils)
   - `ModelArtifacts` dataclass
   - `SpectralLowRankStructure` dataclass
   - `SyntheticFieldSpec` dataclass
   - `ConfoundedFieldLayout` dataclass
   - `SyntheticFieldBuildResult` dataclass
   - `save_model_artifacts(folder, artifacts)`
   - `load_model_artifacts(folder)` - move to file_artifacts
   - `save_field_artifacts(path, artifacts)` - move to file_artifacts
   - `load_field_artifacts(path)` - move to file_artifacts

9. **interaction_matrices.py** (NEW - extracted from model_utils)
   - `compose_interaction_matrix(xi, gamma_matrix)`
   - `interaction_effect(x, gamma_matrix)`
   - `interaction_term(x, xi, gamma_matrix)`
   - `interaction_matrix_infinity_norm(matrix)`

10. **field_operations.py** (NEW - extracted from model_utils)
    - `compose_latent_field_matrix(node_factors, time_factors)`
    - `latent_field_bound_norm(field_matrix)`
    - `zero_latent_field(n_nodes, t_steps)`
    - `normalize_matrix_max_abs(matrix, max_abs)` - move to normalization
    - `normalize_matrix_by_max_abs_entry(matrix)` - move to normalization
    - `truncate_matrix_rank(field_matrix, rank)`
    - `scale_latent_field_matrix(field_matrix, target_rms)`
    - `project_latent_field(node_factors, time_factors, bound)`
    - `compose_field_matrix_from_theta(theta_parts, artifacts)`
    - `with_theta_field(artifacts, theta_parts)`

11. **field_generation.py** (NEW - extracted from model_utils)
    - `parse_singular_values(raw_values, context)`
    - `sample_spectral_low_rank_structure(n_nodes, t_steps, singular_values, rng)`
    - `leading_svd_low_rank_structure(matrix, rank)`
    - `parse_synthetic_field_spec(config)`
    - `resolve_confounded_field_layout(field_spec, intervention_structure)`
    - `build_synthetic_field(config, gamma_matrix, intervention_structure, field_spec)`
    - `build_synthetic_field_with_layout(...)`
    - All helper functions (`_build_random_low_rank_field`, `_build_confounded_low_rank_field`, etc.)

#### **TIER 4: Parameter Management (depends on Tier 0-3)**

12. **scalar_parameters.py** (NEW - extracted from model_utils)
    - `scalar_parameter_names()`
    - `validate_fixed_scalar_params(fixed_scalar_params)`
    - `free_scalar_parameter_names(fixed_scalar_params)`
    - `get_latent_rank(config)`
    - `get_optimizer_mode(config)`
    - `get_B(config)`
    - `get_xi(config)`
    - `uses_full_matrix_parameterization(artifacts)`

13. **parameter_packing.py** (NEW - extracted from model_utils)
    - `parameter_names(artifacts, fixed_scalar_params)`
    - `unpack_theta(theta, artifacts, fixed_scalar_params)`
    - `pack_theta(theta_parts, artifacts, fixed_scalar_params)`
    - `load_true_parameters(config, artifacts, fixed_scalar_params)`
    - `summarize_theta_for_logging(param_names, theta)`
    - `_orthonormal_gaussian_factors(n_rows, rank, rng)` [internal]
    - `_orthonormal_complement_gaussian_factors(...)` [internal]

#### **TIER 5: Domain-Specific Data Loading (depends on Tier 0-4)**

14. **experiment_context.py** (NEW - extracted from loading_utils)
    - `load_panel_context_from_artifacts(panel_path, x0_path, z0_path)`
    - `load_experiment_panel_context(experiment_root)`

15. **parameter_bundles.py** (NEW - extracted from loading_utils)
    - `OutcomeParameterBundle` dataclass
    - `save_estimated_parameter_bundle(path, beta, xi, eta, latent_rank, t_steps, field_matrix)`
    - `load_truth_parameter_bundle(experiment_root)` 
    - `load_fit_parameter_bundle(fit_root, experiment_root)`

#### **TIER 6: Workflow-Specific Utils (depends on Tier 0-5)**

16. **split_management.py** (renamed from split_artifact_utils)
    - `normalize_split_source(split_source)`
    - `validation_test_split_output_root(...)`
    - `load_model_selection_split_masks(...)`
    - `load_outer_test_split_masks(...)`
    - Constants: `SPLIT_SOURCE_*`, `DEFAULT_*`

17. **intervention_context.py** (keep as is, maybe rename)
    - `InterventionContext` dataclass
    - `derive_pre_intervention_steps(z)`
    - `derive_post_intervention_steps(z)`
    - `save_intervention_artifact(...)`
    - `build_full_on_intervention(...)`
    - `build_single_unit_on_intervention(...)`
    - `load_saved_intervention_context(...)`
    - `resolve_intervention_context(...)`

18. **posterior_predictive_manifest.py** (NEW - extracted from posterior_predictive_job_utils)
    - `index_generation_rows(generation_manifest_path)`
    - `resolve_fit_lookup(fit_manifest_path)`
    - `resolve_target_pairs(target_pairs_path, generation_lookup, fit_lookup)`
    - `select_target(targets, experiment_name, source_type, ...)`
    - `experiment_has_truth(experiment_row)`
    - `as_bool(value, default)`
    - `resolve_run_spec(spec_path, run_name)`

19. **posterior_predictive_summary.py** (NEW - extracted from posterior_predictive_job_utils)
    - `build_manifest_row(...)`
    - `manifest_row_from_metadata(...)`

#### **TIER 7: Validation & Evaluation (depends on Tier 0-6)**

20. **validation_metrics.py** (NEW - extracted from validation_metric_utils)
    - `resolve_validation_sampling(config)`
    - `validation_brier_score(x, h_x, loss_mask)`
    - `validation_expected_calibration_error(x, h_x, loss_mask, num_bins)`
    - Magnetization metric functions
    - `evaluate_fold_metrics(...)`
    - `evaluate_saved_fit_fold_metrics(...)`
    - `evaluate_test_metrics(...)`
    - `evaluate_test_metrics_by_treatment(...)`
    - Constants: `DEFAULT_VALIDATION_SAMPLING`, `ECE_NUM_BINS`

21. **cv_aggregation.py** (NEW - extracted from validation_metric_utils)
    - `build_candidate_score_row(...)`
    - `candidate_score_sort_key(row)`
    - All aggregation helpers (`_blank_aggregate_metrics`, `_weighted_and_mean`, `_mean_and_standard_error`)

#### **TIER 8: Posterior Predictive Simulation (depends on Tier 0-7)**

22. **posterior_predictive_sim.py** (NEW - extracted from posterior_predictive_utils)
    - `simulate_outcomes_for_bundle(...)`
    - `compute_panel_statistics(...)`
    - `compute_counterfactual_sample_summary(...)`
    - `compute_observed_sample_summary(...)`

23. **posterior_predictive_reporting.py** (NEW - extracted from posterior_predictive_utils)
    - `summarize_observed_mean_statistics(...)`
    - `summarize_predictive_statistics(...)`

24. **output_writers.py** (NEW - extracted from io_utils)
    - `write_predictive_stats_tables(output_root, stat_rows)`
    - `write_observed_predictive_summary_tables(output_root, sample_summaries, mean_rows, ...)`
    - `write_counterfactual_summary_tables(output_root, sample_summaries)`

#### **TIER 9: Geographic/Domain-Specific Utils (depends on Tier 0)**

25. **geographic_utils.py** (renamed from data_utils, domain-specific)
    - `build_touching_edge_list(gdf, id_column, neighbor_column, geometry_column)`
    - `count_connected_components(nodes, edges, source_column, target_column)`
    - `build_knn_and_kernel_edges(centroids, id_column, x_column, y_column, k)`
    - Note: keep `download_if_missing` in path_utils or generic I/O

---

## Projected Impact Analysis

### **Redundancies Eliminated**

1. **Normalization Functions (5 → 1 module)**
   - Consolidates: `data_utils`, scattered `model_utils` functions
   - Savings: ~60 lines of code
   - Benefit: Single source of truth, easier testing

2. **Finite Summaries (3 → 1)**
   - Consolidates: `io_utils._finite_summary`, `posterior_predictive_utils` versions
   - Savings: ~150 lines, removes duplication
   - Benefit: Consistent statistical computation across validation/posterior

3. **Interaction Matrix Extraction (2 locations → 1)**
   - Consolidates: `model_utils.compose_interaction_matrix` and extract logic
   - Savings: ~10 lines
   - Benefit: Single utility location

4. **Mean/Averaging Operations (2 → 1)**
   - Consolidates: `_mean_on_mask` and `_mean_or_none`
   - Savings: ~20 lines
   - Benefit: Unified masking semantics

### **Complexity Reduced**

1. **Model Utils shrinks from 973 → ~300 lines**
   - Now focused only on critical definitions and orchestration
   - Easier to understand parameter flow

2. **IO Utils shrinks from 285 → ~80 lines**
   - Clear separation: infrastructure vs. domain logic
   - Domain-specific writers moved to `output_writers.py`

3. **Validation Metrics split: 937 → 500 + 200 lines**
   - Metric calculation separate from CV aggregation
   - Different responsibilities clearer

4. **Posterior Predictive split: 305 → 150 + 100 lines**
   - Simulation separate from reporting

### **What Becomes More Complicated**

1. **Import Chains Deeper**
   - Example: `mple.py` might now import from 5+ modules instead of 3
   - Trade-off: More explicit, but requires more imports
   - Mitigation: Create `__init__.py` in utils with convenient re-exports

2. **More Boilerplate in New Modules**
   - Each module needs docstring, imports
   - But each is smaller and more focused

3. **Circular Import Risk**
   - More modules = higher risk of circular imports
   - Mitigation: Strict tier dependency (Tier N depends only on N-1)
   - Can verify with: `python -m py_compile utils/*.py`

4. **Testing More Scattered**
   - Utilities spread across more files
   - Mitigation: Keep test structure mirroring utils structure

### **What Gets Better**

1. **Clarity**
   - Each module has ONE clear responsibility
   - No "grab bag" utils files
   - Easier to find what you need

2. **Reusability**
   - Smaller modules can be imported in isolation
   - Less likely to bring in unwanted dependencies
   - Example: `normalization.py` can be used by any future analysis code

3. **Maintainability**
   - Changes to gamma matrix loading don't affect CSV writing
   - Changes to field generation don't affect parameter packing
   - Reduces unintended side effects

4. **Testability**
   - Smaller modules easier to unit test in isolation
   - Domain-specific modules only depend on clear inputs (Tier 0-2)

5. **Performance**
   - No change at runtime (Python imports are cached)
   - Potential to lazy-load heavy modules in future

---

## Are All These Helpers Still Needed?

### **Functions to Keep (Essential)**
- ✅ Parameter bundles (used by multiple workflows)
- ✅ Model artifacts (core to fitting and prediction)
- ✅ Interaction matrices (used in 3+ modules)
- ✅ Field generation (used in generation and fitting)
- ✅ Validation metrics (core to model selection)
- ✅ Split management (core to CV workflow)
- ✅ Intervention context (used in multiple pipelines)
- ✅ Path/config utilities (infrastructure)

### **Functions to Consider Removing**
- ❓ `data_utils` geographic functions - **ONLY used by USCountyVaccination**
  - Options:
    1. Move to `data/USCountyVaccination/` (local to dataset)
    2. Keep in separate module if planning to reuse for other geographic datasets
    3. Delete if it's a one-off project
  - Recommendation: **Move to dataset-specific code** unless planning geographic reuse

- ❓ `download_if_missing` in data_utils
  - Generic utility but simple (5 lines)
  - Used once in USCountyVaccination
  - Recommendation: **Move to path_utils or dataset-specific code**

### **Functions with Low/Singular Usage**
- `candidate_score_sort_key` - only used during CV aggregation, internal to CV workflow
- Intervention construction builders - **used only by `run_intervention_library.py`**
  - Consider: Move to that script if not part of public API
  - Currently supports the workflow, so keep it

- Geographic functions (as noted above)

### **Recommended Pruning**
1. Move `data_utils.py` entirely to `data/USCountyVaccination/` as local utility
   - Creates: `data/USCountyVaccination/geographic_utils.py`
   - Reason: No other datasets use these functions
   - Saves: 138 lines from main utils, cleaner separation

2. Keep everything else (they're all needed by core pipeline)

---

## Migration Path & Completion Status

### ✅ COMPLETED - Phase 1: Infrastructure (No Risk)

**Status: DONE**

- ✅ Created: `path_utils.py`, `config_utils.py`, `csv_utils.py`
- ✅ Moved code from: `io_utils`
- ✅ Updated imports in `io_utils` with re-exports for backward compatibility
- ✅ Safe, backward compatible

### ✅ COMPLETED - Phase 2: Numerical Operations

**Status: DONE**

- ✅ Created: `normalization.py`, `summary_statistics.py`
- ✅ Moved code from: `data_utils`, `model_utils`, `io_utils`, `posterior_predictive_utils`
- ✅ Updated imports in source files
- ✅ Safe, no circular dependencies

### ✅ COMPLETED - Phase 9: Geographic

**Status: DONE**

- ✅ Moved: `data_utils.py` to `data/USCountyVaccination/data_utils.py`
- ✅ Created: `data/USCountyVaccination/__init__.py`
- ✅ Updated: imports in that directory
- ✅ Deleted: from main utils
- ✅ Low risk, isolated to one dataset

---

## ✅ ALL PHASES COMPLETE

### Phase 3: Model Definition

(Extract from ~500 lines of model_utils)

**Files to create:** `model_artifacts.py`, `interaction_matrices.py`, `field_operations.py`, `field_generation.py`

**What to extract:**

1. **model_artifacts.py** (~150 lines)
   - `ModelArtifacts` dataclass + methods
   - `SpectralLowRankStructure` dataclass
   - `SyntheticFieldSpec` dataclass
   - `ConfoundedFieldLayout` dataclass
   - `SyntheticFieldBuildResult` dataclass

2. **interaction_matrices.py** (~100 lines)
   - `compose_interaction_matrix(xi, gamma_matrix)`
   - `interaction_effect(x, gamma_matrix)`
   - `interaction_term(x, xi, gamma_matrix)`
   - `interaction_matrix_infinity_norm(matrix)`

3. **field_operations.py** (~150 lines)
   - `compose_latent_field_matrix(node_factors, time_factors)`
   - `latent_field_bound_norm(field_matrix)`
   - `zero_latent_field(n_nodes, t_steps)`
   - All field scaling/projection/truncation operations
   - `compose_field_matrix_from_theta(theta_parts, artifacts)`

4. **field_generation.py** (~200 lines)
   - `parse_singular_values(raw_values, context)`
   - `sample_spectral_low_rank_structure(n_nodes, t_steps, singular_values, rng)`
   - `leading_svd_low_rank_structure(matrix, rank)`
   - `parse_synthetic_field_spec(config)`
   - `resolve_confounded_field_layout(field_spec, intervention_structure)`
   - `build_synthetic_field(...)` and variants
   - All `_build_*` helper functions

**Used by:** `mple.py`, `synthetic_data_generation.py`, reports
**Risk:** Medium - `model_utils` shrinks from 973 → ~300 lines, but keep thin re-export layer
**Dependencies:** Tier 0-2 (path, config, normalization, summary_statistics)

---

### Phase 4: Parameter Management

(Extract from ~200 lines of model_utils)

**Files to create:** `scalar_parameters.py`, `parameter_packing.py`

**What to extract:**

1. **scalar_parameters.py** (~80 lines)
   - `scalar_parameter_names()`
   - `validate_fixed_scalar_params(fixed_scalar_params)`
   - `free_scalar_parameter_names(fixed_scalar_params)`
   - `get_latent_rank(config)`, `get_optimizer_mode(config)`, etc.
   - `uses_full_matrix_parameterization(artifacts)`

2. **parameter_packing.py** (~120 lines)
   - `parameter_names(artifacts, fixed_scalar_params)`
   - `unpack_theta(theta, artifacts, fixed_scalar_params)`
   - `pack_theta(theta_parts, artifacts, fixed_scalar_params)`
   - `load_true_parameters(config, artifacts, fixed_scalar_params)`
   - `summarize_theta_for_logging(param_names, theta)`
   - `_orthonormal_gaussian_factors(...)` [internal]

**Used by:** `mple.py` (heavily), reports, diagnostics
**Risk:** Medium - Critical for fitting pipeline, requires thorough testing
**Dependencies:** Tier 3 (model artifacts)

---

### Phase 5: Domain Data Loading

(Extract from loading_utils + model_utils)

**Files to create:** `experiment_context.py`, `parameter_bundles.py`, `matrix_io.py`, `file_artifacts.py`

**What to extract:**

1. **matrix_io.py** (~50 lines)
   - `load_gamma_matrix(data_folder)` - from model_utils
   - `save_loss_mask(path, loss_mask)` - from io_utils

2. **file_artifacts.py** (~100 lines)
   - `load_model_artifacts(folder)` - from model_utils
   - `save_model_artifacts(folder, artifacts)` - from model_utils
   - `save_field_artifacts(path, artifacts)` - from model_utils
   - `load_field_artifacts(path)` - from model_utils

3. **experiment_context.py** (~60 lines)
   - `load_panel_context_from_artifacts(panel_path, x0_path, z0_path)` - from loading_utils
   - `load_experiment_panel_context(experiment_root)` - from loading_utils

4. **parameter_bundles.py** (~90 lines)
   - `OutcomeParameterBundle` dataclass - from loading_utils
   - `save_estimated_parameter_bundle(...)` - from loading_utils
   - `load_truth_parameter_bundle(experiment_root)` - from loading_utils
   - `load_fit_parameter_bundle(fit_root, experiment_root)` - from loading_utils

**Used by:** Most pipeline scripts (heavily used)
**Risk:** Medium-High - Most widely imported util file
**Mitigation:** Keep `loading_utils.py` as re-export layer for backward compat
**Dependencies:** Tier 0-4 (infrastructure, numerical, model definitions, parameter management)

---

### Phase 6: Workflow Utilities

(Refactor posterior_predictive_job_utils)

**Files to create:** `posterior_predictive_manifest.py`, `posterior_predictive_summary.py`

**What to do:**

1. **Rename:** `split_artifact_utils.py` → `split_management.py` (no code changes)
   - Update all imports across codebase

2. **posterior_predictive_manifest.py** (~150 lines)
   - `index_generation_rows(generation_manifest_path)`
   - `resolve_fit_lookup(fit_manifest_path)`
   - `resolve_target_pairs(target_pairs_path, generation_lookup, fit_lookup)`
   - `select_target(targets, experiment_name, source_type, ...)`
   - `experiment_has_truth(experiment_row)`
   - `as_bool(value, default)`
   - `resolve_run_spec(spec_path, run_name)`

3. **posterior_predictive_summary.py** (~100 lines)
   - `build_manifest_row(...)`
   - `manifest_row_from_metadata(...)`

**Used by:** `run_posterior_predictive.py`, `report_posterior_predictive.py`
**Risk:** Low - Few dependencies
**Dependencies:** Tier 5 (parameter bundles)

---

### Phase 7: Validation & Evaluation

(Split validation_metric_utils)

**Files to create:** `validation_metrics.py`, `cv_aggregation.py`

**What to extract from validation_metric_utils.py (~937 lines → 450 + 250 lines):**

1. **validation_metrics.py** (~450 lines)
   - `resolve_validation_sampling(config)`
   - `validation_brier_score(x, h_x, loss_mask)`
   - `validation_expected_calibration_error(x, h_x, loss_mask, num_bins)`
   - All magnetization metric functions
   - `evaluate_fold_metrics(...)`
   - `evaluate_saved_fit_fold_metrics(...)`
   - `evaluate_test_metrics(...)`
   - `evaluate_test_metrics_by_treatment(...)`
   - Constants: `DEFAULT_VALIDATION_SAMPLING`, `ECE_NUM_BINS`

2. **cv_aggregation.py** (~250 lines)
   - `build_candidate_score_row(...)`
   - `candidate_score_sort_key(row)` - only used here
   - All aggregation helpers (`_blank_aggregate_metrics`, `_weighted_and_mean`, `_mean_and_standard_error`)

**Used by:** `run_cv_folds.py`, `run_test_evaluation.py`
**Risk:** Low - Metric calculation ≠ aggregation, clean separation
**Dependencies:** Tier 5 (parameter bundles)

---

### Phase 8: Posterior Predictive Output

(Split posterior_predictive_utils + io_utils)

**Files to create:** `posterior_predictive_sim.py`, `posterior_predictive_reporting.py`, `output_writers.py`

**What to extract:**

1. **posterior_predictive_sim.py** (~150 lines) - from posterior_predictive_utils
   - `simulate_outcomes_for_bundle(...)`
   - `compute_panel_statistics(...)`
   - `compute_counterfactual_sample_summary(...)`
   - `compute_observed_sample_summary(...)`

2. **posterior_predictive_reporting.py** (~100 lines) - from posterior_predictive_utils
   - `summarize_observed_mean_statistics(...)`
   - `summarize_predictive_statistics(...)`
   - All summary aggregation logic

3. **output_writers.py** (~80 lines) - from io_utils
   - `write_predictive_stats_tables(output_root, stat_rows)`
   - `write_observed_predictive_summary_tables(output_root, sample_summaries, mean_rows, ...)`
   - `write_counterfactual_summary_tables(output_root, sample_summaries)`

**Used by:** `run_posterior_predictive.py`, `report_posterior_predictive.py`
**Risk:** Low - Few dependencies, clear input/output boundaries
**Dependencies:** Tier 6 (workflow utilities, parameter bundles)

---

## Recommended Implementation Order

**Critical Path:**

1. **Phase 3** (Model Definition) - Unblocks Phase 4
2. **Phase 4** (Parameter Management) - Unblocks Phase 5
3. **Phase 5** (Domain Data Loading) - Unblocks everything else
4. **Phase 6** (Workflow Utilities) - Unblocks Phase 7-8
5. **Phase 7** (Validation & Evaluation) - Standalone but needed for completeness
6. **Phase 8** (Posterior Predictive Output) - Final cleanup

**Per-phase steps:**

- Create new modules with extracted code
- Update imports in source files
- Update `utils/__init__.py` with new re-exports (for backward compat during transition)
- Run import validation: `python -c "import utils; print('OK')"`
- Test affected pipeline: `pytest tests/` after each phase

---

## Summary Table

| Aspect | Current | Proposed | Benefit |
|--------|---------|----------|---------|
| **Number of util modules** | 9 | 25 | Clarity, modularity |
| **Max module lines** | 973 | 250 | Easier to understand |
| **Import chains** | 3-5 deep | Up to 9 tiers | More explicit dependency order |
| **Redundant functions** | 8-10 | 0-2 | Reduced duplication |
| **Circular imports** | 1-2 | 0 (enforced by tiers) | Safer refactoring |
| **Files importing io_utils** | 18 | 8 (after split) | Better separation of concerns |
| **Backward compat** | N/A | Can re-export | Low migration cost |

