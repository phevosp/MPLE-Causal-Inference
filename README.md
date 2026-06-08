# MPLE-Causal-Inference

Spec-driven conditional MPLE experiments for binary outcome/intervention panel data with:

- a fixed known graph `Gamma`
- a scalar interaction temperature `xi`
- an optional low-rank external field

The repository currently supports two main workflows:

1. Synthetic and hybrid experiment generation, fitting, and posterior-predictive evaluation
2. Nationwide US county vaccination data preparation and experiment materialization

## Repo Guide

- Root workflow and commands: this file
- Manifest and artifact reference: [PIPELINE_REFERENCE.md](PIPELINE_REFERENCE.md)
- US county workflow details: [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md)

## Quickstart

Verify your install works end-to-end in under 2 minutes using the toy experiment (N=30, T=10, rank=2, 500 optimizer steps):

```bash
pixi install
pixi run python -u run_generation_pipeline.py --spec_path data/configs/quickstart_generation_spec.yaml
pixi run python -u run_fit_pipeline.py --manifest_path experiments/Quickstart/generation_manifest.csv --fits_spec_path data/configs/quickstart_fits_spec.yaml
```

Outputs land in `experiments/Quickstart/`. Once this works, proceed to the full synthetic or real-data workflows below.

## Configuration Guide

All pipeline YAML specs use a `base + named entries` pattern: every named entry is deep-merged with `base`, inheriting all fields it doesn't override. The spec is then expanded into one config per entry by `utils.t6_pipeline_spec_utils.expand_named_entries()`.

**There are two separate config directories — do not mix them:**

| Directory | Used by |
| --- | --- |
| `data/configs/` | Synthetic/hybrid pipeline (`run_generation_pipeline.py`, `run_fit_pipeline.py`, `run_posterior_predictive.py`, `report_posterior_predictive.py`) |
| `data/USCountyVaccination/experiment_configs/` | Real-data pipeline |

Both directories contain their own workflow specs. In `data/configs/`, the canonical example specs are `quickstart_generation_spec.yaml` and `quickstart_fits_spec.yaml`; the real-data pipeline keeps its own separate fit specs under `data/USCountyVaccination/experiment_configs/`.

**Key `quickstart_fits_spec.yaml` fields:**

- `optimizer_mode`: one of `no_external_field`, `nuclear_norm`, `exact_rank_manifold`, `alternating_latent_rank`, or `concurrent_latent_rank`.
- `latent_rank`: must be ≥ 1 for `exact_rank_manifold`, `alternating_latent_rank`, and `concurrent_latent_rank`; ignored for `no_external_field` and `nuclear_norm`.
- `estimation.fixed_scalar_params`: scalars held **fixed** at these values (not initial guesses). Leave as `{}` to estimate all scalars freely.
- `estimation.beta_mask_pre_s`: if `true`, the fit objective masks the `beta * z` term for `t < s`. This is a parameter-estimation choice only; posterior predictive sampling, Brier/ECE metrics, and manifest-driven data generation still use the realized `z`.
- `estimation.beta_mask_post_e`: if `true`, the fit objective masks the `beta * z` term for `t >= e`. This is also fit-only and does not alter predictive sampling, Brier/ECE evaluation, or data generation.
- `lambda_nuclear`: only active for `nuclear_norm`.
- `lambda_frobenius`: only active for `exact_rank_manifold`.
- `lambda_uv_ridge`: only active for `alternating_latent_rank` and `concurrent_latent_rank`.
- `v_column_l2_max`: optional per-column `||v_k||_2 <= c` constraint for `alternating_latent_rank`; ignored by `concurrent_latent_rank`.

**Key `quickstart_generation_spec.yaml` fields:**

- `intervention.generator`: use `low_rank_probability` to sample `z` independently from a spectral low-rank probability matrix
- `intervention.params.singular_values`: explicit singular values for the intervention low-rank generator
- `intervention.params.probability_amplitude`: maps normalized low-rank scores to probabilities via `p = 0.5 + amplitude * score`
- `truth.field_mode`: `random_low_rank` or `confounded_low_rank`
- `truth.field_params.singular_values`: explicit singular values for spectral low-rank field generation
- `truth.field_params.shared_rank`: optional for `confounded_low_rank`; the first `k` field components reuse the intervention basis and the remaining field components are sampled in an orthogonal complement. If omitted, `confounded_low_rank` keeps the original full-confounding behavior.
- `truth.field_params.target_rms_fraction`: optional RMS target for the field as a fraction of `B`; defaults to `0.4`

## Environment

The project is configured with `pixi.toml` and currently targets `win-64`.

Typical setup:

```bash
pixi install
```

Run commands through Pixi so the pinned environment is used:

```bash
pixi run python -u <script>.py ...
```

The shell wrappers in the repo are `bash` scripts. On Windows they are intended for Git Bash, WSL, or another compatible shell.

## Top-Level Workflows

| Workflow | Entry points | Main inputs | Main outputs |
| --- | --- | --- | --- |
| Synthetic and hybrid generation | `run_generation_pipeline.py`, `bash_scripts/submit_generation_jobs.sh` | `data/configs/quickstart_generation_spec.yaml` | `generation_requests.csv`, `generation_manifest.csv`, experiment folders |
| MPLE variant fitting | `run_fit_pipeline.py`, `bash_scripts/submit_fit_jobs.sh` | `generation_manifest.csv`, `data/configs/quickstart_fits_spec.yaml` | `fit_requests.csv`, `fit_manifest.csv`, `fits/<variant>/...`, fit summaries |
| Intervention library generation | `run_intervention_library.py` | generation manifest, `data/configs/intervention_library_spec.yaml` | `intervention_library_manifest.csv`, saved intervention panels |
| Posterior predictive and counterfactual simulation | `run_posterior_predictive.py`, `report_posterior_predictive.py`, `bash_scripts/submit_posterior_predictive_jobs.sh` | generation manifest, fit manifest, `posterior_predictive_spec.yaml`, `posterior_predictive_target_pairs.csv` | `posterior_predictive_manifest.csv`, predictive or counterfactual summaries |
| CV fold construction for `U,V` regularizer tuning | `build_cv_folds.py` | `generation_manifest.csv` for experiments with `Gamma`, `panel_data.npz`, and optional `node_index.csv` / `time_index.csv` | `cv_folds/folds_5/` spatial partitions plus spatiotemporal fold artifacts |
| Cross-validated MPLE hyperparameter search | `run_cv_folds.py` | `generation_manifest.csv`, prebuilt `cv_folds/folds_5/`, `data/configs/cv_spec.yaml` | `cv_requests.csv`, `cv_manifest.csv`, per-search candidate scores and fold fits |
| Real-data raw load | `data/USCountyVaccination/load_raw_data.py` | remote NYT, CDC, Bansal, Census geography sources | cached raw inputs |
| Real-data preprocessing and realization | `data/USCountyVaccination/preprocess_us_county_vaccination_data.py` | cached raw inputs | processed panels, `realized_*`, `shared_panels` |
| Real-data experiment materialization | `data/USCountyVaccination/create_us_county_vaccination_experiments.py` | `realized_*`, `shared_panels` | shared-compatible experiment folders, `generation_manifest.csv` |

## Synthetic And Hybrid Pipeline

### 1. Generation

`run_generation_pipeline.py` expands a YAML spec with `base + experiments` into concrete experiment folders. It now also supports a staged request workflow: write `generation_requests.csv`, materialize one experiment by `experiment_slug`, or refresh `generation_manifest.csv` from completed outputs.

Default config:

- `data/configs/quickstart_generation_spec.yaml`

Default command:

```bash
pixi run python -u run_generation_pipeline.py --spec_path data/configs/quickstart_generation_spec.yaml
```

What generation resolves:

- `dimensions.N` and `dimensions.T`
- outcome truth scalars `beta`, `xi`, `eta`
- `truth.field_mode`
- graph source
- intervention source
- initial state generator
- generation seed and Gibbs sweeps

Supported experiment styles:

- fully synthetic: generated graph and generated intervention panel
- hybrid with fixed intervention artifacts
- hybrid with fixed graph artifacts
- hybrid with both fixed intervention and fixed graph artifacts

If graph or intervention artifacts are fixed, generation materialization can resolve missing `N` and `T` directly from those artifacts.

Outputs:

- `experiments/SyntheticHybridExperiments/generation_requests.csv`
- `experiments/SyntheticHybridExperiments/generation_manifest.csv`
- one folder per generated experiment under `experiments/SyntheticHybridExperiments/<experiment_slug>/`

Batch submission command:

```bash
GENERATION_SPEC_PATH=data/configs/quickstart_generation_spec.yaml \
GENERATION_OVERWRITE=true \
bash bash_scripts/submit_generation_jobs.sh
```

### 2. Fit Variants

`run_fit_pipeline.py` reads a generation manifest plus a fit spec with `base + variants`, creates a fit folder for every `(experiment, variant)` pair, and runs `mple.py`. It now also supports a staged request workflow: write `fit_requests.csv`, run one `(experiment_slug, variant_slug)` request, or refresh `fit_manifest.csv` and grouped fit reports from completed outputs.

Default config:

- `data/configs/quickstart_fits_spec.yaml`

Default command:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/quickstart_fits_spec.yaml
```

Each fit variant controls:

- `B`
- `optimizer_mode`
- `latent_rank`
- `lambda_nuclear` for `nuclear_norm`
- `lambda_frobenius` for `exact_rank_manifold`
- `lambda_uv_ridge` for `alternating_latent_rank` and `concurrent_latent_rank`
- `v_column_l2_max` for alternating-only `V`-column L2-ball constraints
- `estimation.fixed_scalar_params`
- `estimation.beta_mask_pre_s`
- optimizer `steps`, `tol`, `seed`, `n_starts`, and `proximal_lr`

Outputs:

- `experiments/SyntheticHybridExperiments/fit_requests.csv`
- `experiments/SyntheticHybridExperiments/fit_manifest.csv`
- `fits/<variant_slug>/` under each experiment root
- per-experiment `fit_summary.csv`
- cross-experiment `best_fit_by_experiment.csv`

Batch submission command:

```bash
GENERATION_MANIFEST_PATH=experiments/SyntheticHybridExperiments/generation_manifest.csv \
FITS_SPEC_PATH=data/configs/quickstart_fits_spec.yaml \
FIT_OVERWRITE=true \
bash bash_scripts/submit_fit_jobs.sh
```

### 3. Intervention Library

`run_intervention_library.py` creates reusable intervention panels under each generated experiment root listed in a generation manifest. These panels are useful for counterfactual posterior-predictive runs where the outcomes are simulated under an intervention different from the one observed in the original experiment.

Default config:

- `data/configs/intervention_library_spec.yaml`

Default command:

```bash
pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --spec_path data/configs/intervention_library_spec.yaml \
  --overwrite
```

Supported intervention entries:

- `observed_experiment`: copies the realized `z` and `z_0` from the experiment into the library
- `full_on`: creates a population-wide intervention with `activation_scope: all_time | no_time | from_s`
- `single_unit_on`: creates a one-unit intervention with `activation_scope: all_time | no_time | from_s | from_step`

Saved intervention panels use the model's `-1/+1` coding for `z`. The library also accepts the repo's legacy `z_0 = 0` convention when copying observed experiment artifacts.

Examples:

- `full_on` with `activation_scope: all_time` is all `+1`
- `full_on` with `activation_scope: no_time` is all `-1`
- `full_on` with `activation_scope: from_s` is `-1` before the experiment's `s` and `+1` from `s` onward

Outputs:

- `experiments/SyntheticHybridExperiments/intervention_library_manifest.csv`
- `intervention_library/<intervention_slug>/intervention_panel.npz` under each experiment root
- `intervention_library/<intervention_slug>/z_0.npy`
- `intervention_library/<intervention_slug>/intervention_metadata.yaml`

### 4. Posterior Predictive And Counterfactual Simulation

`run_posterior_predictive.py` runs one explicit posterior-predictive or counterfactual target while keeping the intervention panel fixed. `report_posterior_predictive.py` then scans completed outputs, rebuilds the unified manifest, and writes grouped posterior-predictive reports. Simulations can draw from either:

- the experiment truth parameters
- one or more saved MPLE fit bundles

The same runner also supports counterfactual simulations under saved interventions from the intervention library. Observed-intervention runs are used for posterior-predictive goodness-of-fit diagnostics. Saved-intervention runs are treated as counterfactual scenarios and write compact causal summaries instead of z-score diagnostics.

Targeting is explicit. `data/configs/posterior_predictive_target_pairs.csv` must contain:

- `experiment_name`
- `source_type` as `truth` or `fit`
- `variant_name`, blank for truth rows

It may also contain:

- `intervention_source`, either `observed_experiment` or `saved_intervention`
- `intervention_name`, required when `intervention_source` is `saved_intervention`

If the intervention columns are omitted, the runner defaults to `observed_experiment` for backward compatibility.

Run settings live in `data/configs/posterior_predictive_spec.yaml` and currently use `base + runs`.

Single-target command:

```bash
pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fit_manifest_path experiments/SyntheticHybridExperiments/fit_manifest.csv \
  --target_pairs_path data/configs/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name synthetic_rank_40_B1 \
  --source_type fit \
  --variant_name rank_40_B1 \
  --intervention_source observed_experiment \
  --intervention_name observed_experiment \
  --run_name default
```

Batch submission command:

```bash
GEN_MANIFEST=experiments/SyntheticHybridExperiments/generation_manifest.csv \
FIT_MANIFEST=experiments/SyntheticHybridExperiments/fit_manifest.csv \
TARGET_PAIRS_PATH=data/configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/configs/posterior_predictive_spec.yaml \
bash bash_scripts/submit_posterior_predictive_jobs.sh
```

Manifest/report refresh command:

```bash
pixi run python -u report_posterior_predictive.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv
```

Unified outputs:

- `experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv`
- `posterior_predictive/<source_slug>/<run_slug>/...` under each experiment root
- `counterfactual/<source_slug>/<intervention_slug>/<run_slug>/...` under each experiment root
- per-experiment `posterior_predictive_summary.csv`
- cross-experiment `best_posterior_predictive_by_experiment.csv`
- `counterfactual_sample_summaries.npz`
- `counterfactual_summary.csv`
- `counterfactual_unit_summary.csv`
- `counterfactual_time_summary.csv`
- `counterfactual_metadata.yaml`

Counterfactual rows are included in the unified `posterior_predictive_manifest.csv`, but they do not write `posterior_predictive_stats.csv` and are excluded from posterior-predictive ranking.

Saved-intervention `intervention_summaries/<intervention_slug>.csv` files now include truth-referenced counterfactual comparison columns when a matching `truth` row exists for the same run. These reports compare overall, post-intervention, unit-level, and time-level mean magnetization summaries against the saved truth row.

Example counterfactual target-pairs file:

```csv
experiment_name,source_type,variant_name,intervention_source,intervention_name
synthetic_rank_40_B1,truth,,saved_intervention,all_minus_ones
synthetic_rank_40_B1,fit,rank_40_B1,saved_intervention,all_minus_ones
```

The `seed` in `posterior_predictive_spec.yaml` is the base seed for a run. Individual samples use a deterministic hash of:

- the base seed
- the run slug
- the experiment / source / intervention target identity
- `sample_index`

So repeated runs are reproducible, samples within a run are distinct, and different targets do not accidentally reuse the same random stream.

## CV Fold Construction

`build_cv_folds.py` is the canonical downstream entry point for building the spatial and spatiotemporal folds used to tune `lambda_uv_ridge` or related `U,V` regularization choices. It now consumes a generation manifest and builds folds for every experiment listed there.

Run it with:

```bash
pixi run python -u build_cv_folds.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv
```

Required experiment artifacts:

- `gamma_matrix.npy` or `gamma_matrix_sparse.npz`
- `panel_data.npz`
- optional `node_index.csv`
- optional `time_index.csv`

`pymetis` is required and is pinned in `pixi.toml`. The script validates that the loaded `Gamma` artifact is:

- square
- symmetric within the configured tolerance
- zero-diagonal within the configured tolerance

The graph partitioning stage then computes:

- `C_1, ..., C_5`: a 5-way METIS partition of the vertex set
- `S_i = {v in V \ C_i : v has at least one neighbor in C_i}` for each `i`

METIS optimizes a balanced edge-cut objective. The saved separator sets `S_i` are derived diagnostics used to construct the CV folds; they are not the optimization target of METIS itself.

The time horizon is split into 5 contiguous ordered blocks `T_1, ..., T_5` with:

- minimum block sizes `[1, 2, 2, 2, 2]`
- the first time index of `T_2, ..., T_5` marked as a transition step
- a minimum supported horizon of `T = 9`

The 5 composite folds use cyclic validation schedules:

- fold 1 validates `C_1, C_2, C_3, C_4, C_5` on `T_1, ..., T_5`
- fold 2 validates `C_5, C_1, C_2, C_3, C_4`
- fold 3 validates `C_4, C_5, C_1, C_2, C_3`
- fold 4 validates `C_3, C_4, C_5, C_1, C_2`
- fold 5 validates `C_2, C_3, C_4, C_5, C_1`

Per-time role assignment uses three roles:

- `training`
- `separator`
- `validation`

On non-transition times in block `T_b`:

- validation = `C_active`
- separator = `S_active`
- training = `V \ (C_active ∪ S_active)`

On the first time index of each of `T_2, ..., T_5`, if the active validation partition changes from `C_prev` to `C_curr`:

- separator = `S_curr ∪ C_prev ∪ C_curr`
- validation = empty
- training = the complement

This guarantees a 1-step temporal separator whenever a partition switches between training and validation. There is no wraparound transition after `T_5`.

During construction, `build_cv_folds.py` also validates that the separator acts as a Markov blanket in the **full spatiotemporal dependency graph** implied by the model:

- same-time spatial dependencies come from `Gamma`
- adjacent-time self-dependencies come from the `eta * prev_x` term

So the construction succeeds only if:

- there is no same-time `Gamma` edge between validation and training vertices
- there is no adjacent-time self-transition where the same vertex is validation at one time and training at the next

The script always writes the blanket and coverage diagnostics, then raises if the Markov-blanket check fails.

Artifacts are written under each experiment root:

- `<experiment_root>/cv_folds/folds_5/`

Files:

- `vertex_assignments.csv`: one row per vertex with its `C_i` assignment
- `separator_vertices.csv`: one row per `(S_i, vertex)` membership
- `fold_roles.npz`: compact `(cv_fold, time, vertex)` role tensor
- `time_blocks.csv`: block assignment and transition-step metadata for each time index
- `fold_schedule.csv`: the 5 cyclic validation schedules
- `fold_role_counts.csv`: aggregated training/separator/validation counts by fold and block
- `spatial_partition_metadata.yaml`
- `spatiotemporal_cv_metadata.yaml`
- `markov_blanket_summary.yaml`

`fold_roles.npz` contains:

- `role_codes` with shape `(5, T, N)`
- `time_block_ids` with shape `(T,)`
- `is_transition_step` with shape `(T,)`
- `validation_partition_ids_by_fold_block` with shape `(5, 5)`

Role codes are:

- `0 = training`
- `1 = separator`
- `2 = validation`

`markov_blanket_summary.yaml` records whether the constructed folds pass the full spatiotemporal blanket test, along with counts of spatial and temporal violations by fold.

`spatiotemporal_cv_metadata.yaml` now also includes the aggregate coverage-count diagnostics describing how often vertices appear in validation, separator, and training across the full 5-fold family. These coverage counts are descriptive diagnostics only; they are not enforced as hard acceptance thresholds.

Example on a US county experiment root:

```bash
pixi run python -u build_cv_folds.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv
```

## Cross-Validated MPLE Search

`run_cv_folds.py` consumes the prebuilt fold artifacts from `build_cv_folds.py` and runs a 5-fold hyperparameter grid search without duplicating the core MPLE fitting code.

Run it with:

```bash
pixi run python -u run_cv_folds.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --cv_spec_path data/configs/cv_spec.yaml
```

To recompute CV fold and candidate metrics from existing fold fit artifacts without
rerunning any fits, refresh scores from a saved request file:

```bash
pixi run python -u run_cv_folds.py \
  --refresh_scores \
  --cv_requests_path experiments/SyntheticHybridExperiments/cv_requests.csv
```

The CV spec mirrors the fit-spec layout but adds a required `grid:` section inside each named search. Fixed values outside `grid` apply to every candidate, and list-valued leaves inside `grid` are expanded as a Cartesian product.

Example:

```yaml
base:
  cv_root_name: cv_runs
  cv_manifest_path: experiments/SyntheticHybridExperiments/cv_manifest.csv
  optimizer:
    steps: 20000
    tol: 1.0e-9
    seed: 0
    n_starts: 1
    proximal_lr: 1.0
  optimizer_mode: alternating_latent_rank
  latent_rank: 3
  lambda_uv_ridge: 0.0
  estimation:
    fixed_scalar_params: {}

searches:
  - name: alternating_uv_grid
    optimizer_mode: alternating_latent_rank
    grid:
      latent_rank: [3, 5, 7]
      lambda_uv_ridge: [0.001, 0.01, 0.1]
      estimation:
        beta_mask_pre_s: [false, true]
```

Each CV fold uses the saved role tensor from `fold_roles.npz`:

- training loss mask = entries with role `training`
- validation loss mask = entries with role `validation`
- separator entries stay visible in `x`, `z`, `prev_x`, and `Gamma x`, but contribute zero loss during fitting and zero loss during validation scoring

So the fitted MPLE parameters are learned by conditioning on separator unit-times while optimizing only over training unit-times. Validation uses the same fitted parameters and evaluates masked statistics only on validation unit-times. Separator entries remain visible to the model through `x`, `z`, `x_{t-1}`, and `Gamma x`, but they are never scored as part of validation.

Selection is not based on pooled validation Brier score alone. The current winner rule is:

- keep the existing `validation_*` columns as full-horizon validation metrics over all validation slots
- additionally save `post_s_validation_loss`, `post_s_validation_brier_score`, `post_s_validation_ece`, and post-`s` magnetization diagnostics over validation slots with `t >= s`
- for each candidate and fold, save those metrics plus the numbers of active training, validation, and post-`s` validation slots
- aggregate across the folds using slot-weighted means for both the full-horizon and post-`s` validation metrics
- also report mean-per-fold summaries and standard errors for the full-horizon and post-`s` validation metrics
- rank candidates by post-`s` weighted magnetization error first, then post-`s` weighted Brier score, then post-`s` weighted loss; if post-`s` metrics are unavailable, fall back to the full-horizon versions
- choose the final winner with a 1-standard-error rule on mean-per-fold post-`s` magnetization error, then mean-per-fold post-`s` Brier score, and finally prefer the less regularized candidate among those still eligible

The reported validation Brier score and ECE use the model-implied probability of a positive spin:

- `h_it = M_it + beta z_it + xi (Gamma x_t)_i + eta x_{i,t-1}`
- `P(x_it = 1 | h_it) = (1 + tanh(h_it)) / 2`
- observed outcome on the probability scale is `(x_it + 1) / 2`
- fit-only beta masking does not enter this predictive `h_it`; Brier/ECE always use the realized intervention panel
- ECE uses 10 equal-width bins on `[0, 1]`, skips empty bins, and is computed only on validation unit-times
- post-`s` validation metrics further restrict scoring to validation unit-times with `t >= s`

Artifacts are written under:

- `<experiment_root>/<cv_root_name>/<search_slug>/`

Files:

- `candidate_grid.csv`: resolved hyperparameter combinations
- `fold_scores.csv`: one row per `(candidate, cv_fold)`
- `candidate_scores.csv`: aggregated 5-fold metrics and ranks
- `best_candidate.yaml`: winning hyperparameters and summary metrics
- `candidates/<candidate_slug>/fold_<i>/...`: ordinary MPLE fit artifacts for each fold

Top-level outputs:

- `cv_requests.csv`: one row per `(experiment, search, candidate, fold)`
- `cv_manifest.csv`: one row per `(experiment, search)` with the selected winner

The refresh workflow reads `cv_requests.csv`, reloads each saved fold fit, recomputes
`fit_loss`, the full-horizon validation metrics, and the post-`s` validation metrics, and
then rewrites `fold_scores.csv`, `candidate_scores.csv`, `best_candidate.yaml`, and
`cv_manifest.csv`.

The runner requires:

- prebuilt `cv_folds/folds_5/fold_roles.npz`
- `markov_blanket_summary.yaml` with `blanket_validation_passed: true`
- exactly 5 folds in v1

## Core Artifact Contract

At experiment scope, the shared panel/model artifacts are:

- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`
- `field_artifacts.npz`
- `gamma_matrix.npy` or `gamma_matrix_sparse.npz`

`field_artifacts.npz` stores:

- `latent_rank`
- `t_steps`
- `optimizer_mode`
- `field_matrix`

`latent_rank = 0` means the realized field is exactly zero.

Each fit folder additionally writes:

- `fit_realized_config.yaml`
- `fit_metadata.yaml`
- `mple.log`
- `mple_summary.csv`
- `estimated_field_artifacts.npz`
- `estimated_parameter_bundle.npz`
- `estimated_interaction_matrix.npy` or `estimated_interaction_matrix_sparse.npz`

When truth is available, fit folders also write:

- `true_field_artifacts.npz`
- `true_interaction_matrix.npy` or `true_interaction_matrix_sparse.npz`

Each posterior-predictive run writes:

- `posterior_predictive_stats.csv`
- `posterior_predictive_metadata.yaml`

Each saved intervention writes:

- `intervention_panel.npz`
- `z_0.npy`
- `intervention_metadata.yaml`

Each counterfactual run writes:

- `counterfactual_sample_summaries.npz`
- `counterfactual_summary.csv`
- `counterfactual_unit_summary.csv`
- `counterfactual_time_summary.csv`
- `counterfactual_metadata.yaml`

`PIPELINE_REFERENCE.md` has the full directory and manifest layout.

## Ranking Rules

Fit reporting:

- If truth metrics exist, variants are ranked by `total_recovery_rmse = field_rmse + sum(abs scalar errors)`, then `field_rmse`, then `interaction_fro_error`, then `final_loss`
- If truth metrics do not exist, ranking falls back to `final_loss`

Posterior-predictive reporting:

- rows are ranked by lowest `mean_abs_zscore`
- ties break on lowest `max_abs_zscore`

## Main Commands

Generate synthetic and hybrid experiments:

```bash
pixi run python -u run_generation_pipeline.py --spec_path data/configs/quickstart_generation_spec.yaml
```

Fit all configured variants:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/quickstart_fits_spec.yaml
```

Run one posterior predictive target:

```bash
pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fit_manifest_path experiments/SyntheticHybridExperiments/fit_manifest.csv \
  --target_pairs_path data/configs/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name synthetic_rank_40_B1 \
  --source_type fit \
  --variant_name rank_40_B1 \
  --intervention_source observed_experiment \
  --intervention_name observed_experiment \
  --run_name default
```

Build saved intervention panels:

```bash
pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --spec_path data/configs/intervention_library_spec.yaml \
  --overwrite
```

Build unified spatial + spatiotemporal CV folds:

```bash
pixi run python -u build_cv_folds.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv
```

Regenerate grouped fit reports from an existing fit manifest:

```bash
pixi run python -u report_parameter_recovery_detailed.py \
  --manifest experiments/SyntheticHybridExperiments/fit_manifest.csv
```

Refresh the unified posterior-predictive manifest and grouped reports:

```bash
pixi run python -u report_posterior_predictive.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv
```

Run the minimal regression suite:

```bash
pixi run python tests/test_minimal_pipeline.py
```

Run the staged shell orchestrator:

```bash
GENERATION_SPEC_PATH=data/configs/quickstart_generation_spec.yaml \
FITS_SPEC_PATH=data/configs/quickstart_fits_spec.yaml \
INTERVENTION_LIBRARY_SPEC_PATH=data/configs/intervention_library_spec.yaml \
TARGET_PAIRS_PATH=data/configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/configs/posterior_predictive_spec.yaml \
bash bash_scripts/run_and_submit_full_pipeline.sh
```

## Manual MPLE Invocation

`mple.py` can fit directly from a generated experiment root or from an explicit fit folder.

Typical manual fit with an explicit fit config:

```bash
pixi run python -u mple.py \
  --data_folder experiments/<experiment>/fits/<variant>
```

Useful flags:

- optimizer settings and artifact paths are read from `fit_realized_config.yaml` in the fit folder
- `--log_file` redirects the MPLE log

When `n_starts > 1`, MPLE runs independent random starts and keeps the fit with the lowest final pseudo-negative log likelihood. Per-start diagnostics are saved to `optimizer_start_summary.csv`.

For `optimizer_mode: nuclear_norm`, MPLE optimizes the full latent field directly with a nuclear-norm penalty, using proximal singular-value thresholding instead of the low-rank factorization. The usual `mple_summary.csv` includes the unpenalized MPLE loss, penalized objective, nuclear norm, and effective rank.

For `optimizer_mode: concurrent_latent_rank`, MPLE uses SciPy `L-BFGS-B` to optimize the same factorized `U, V` formulation and `lambda_uv_ridge * (||U||_F^2 + ||V||_F^2) / outcome_size` penalty used by `alternating_latent_rank`, but updates all packed parameters jointly rather than alternating block steps.

For `optimizer_mode: alternating_latent_rank`, setting `global_params.v_column_l2_max` projects each column of `V` (`node_factors`) onto the Euclidean ball `||v_k||_2 <= v_column_l2_max` after initialization and after each `V`-update step. The current `concurrent_latent_rank` solver ignores this setting.

`global_params.B` is the active fit-time bound:

- scalar parameters are clipped to `[-B, B]`
- `xi` is also constrained so that `||xi * Gamma||_inf <= B`
- the latent field is projected so its maximum absolute entry respects the same bound

## Shell Wrappers

The repo ships lightweight wrappers:

- `bash_scripts/run_generation_job.sh`
- `bash_scripts/submit_generation_jobs.sh`
- `bash_scripts/run_fit_job.sh`
- `bash_scripts/submit_fit_jobs.sh`
- `bash_scripts/run_posterior_predictive_job.sh`
- `bash_scripts/submit_posterior_predictive_jobs.sh`
- `bash_scripts/run_and_submit_full_pipeline.sh`

They call the same Python entry points and accept environment-variable overrides:

- `GENERATION_SPEC_PATH`, `GENERATION_OVERWRITE`
- `GENERATION_MANIFEST_PATH`, `FITS_SPEC_PATH`, `FIT_OVERWRITE`
- `GEN_MANIFEST`, `FIT_MANIFEST`, `TARGET_PAIRS_PATH`, `POSTERIOR_PREDICTIVE_SPEC_PATH`, `POSTERIOR_PREDICTIVE_OVERWRITE`
- `INTERVENTION_LIBRARY_SPEC_PATH`

## Repository Map

- `data/synthetic_data_generation.py`: synthetic and hybrid artifact materialization
- `run_generation_pipeline.py`: generation request planning, single-request execution, and manifest refresh
- `bash_scripts/submit_generation_jobs.sh`: SLURM fan-out for generation requests plus manifest refresh barrier
- `run_fit_pipeline.py`: fit request planning, single-fit execution, and manifest refresh/report rebuild
- `bash_scripts/submit_fit_jobs.sh`: SLURM fan-out for fit requests plus manifest/report refresh barrier
- `run_intervention_library.py`: reusable intervention-panel materialization
- `build_cv_folds.py`: manifest-driven spatial partition plus spatiotemporal CV-fold construction
- `run_posterior_predictive.py`: single-target posterior-predictive/counterfactual execution
- `report_posterior_predictive.py`: manifest refresh plus grouped posterior-predictive reporting
- `bash_scripts/run_and_submit_full_pipeline.sh`: staged generation → fit → intervention → posterior-predictive shell orchestrator
- `mple.py`: conditional MPLE optimizer and artifact writer
- `utils/`: tiered utility modules (`t0_*` through `t8_*`) for config/path I/O, model artifacts, parameter packing, experiment loading, intervention handling, validation metrics, and posterior-predictive reporting
- `utils/t3_field_generation.py`: synthetic-field specification parsing and field construction
- `utils/t5_experiment_context.py`: experiment/panel artifact loading and experiment-context assembly
- `utils/t6_intervention_utils.py`: intervention construction, saved-intervention artifacts, and intervention timing helpers
- `utils/t8_posterior_predictive_sim.py`: predictive simulation and posterior-predictive panel statistics
- `utils/t6_pipeline_spec_utils.py`: pipeline-spec expansion and validation helpers
- `tests/test_minimal_pipeline.py`: regression coverage for generation, fitting, summaries, and predictive ranking

## Real-Data Workflow

USCountyVaccination experiments are real-data experiments with `has_truth: false`. The materializer writes the same root-level artifact contract as synthetic/hybrid experiments, so the shared fit, intervention-library, and counterfactual posterior-predictive runners can consume them directly. Fit-based posterior predictive is supported; `source_type=truth` targets are intentionally rejected for these experiments.

Default trimmed death-rate/vaccine-rate materialization:

```bash
pixi run python data/USCountyVaccination/load_raw_data.py
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

To materialize trimmed experiments starting from one or more later modeled weeks, pass `--start_dates`. Each requested date rounds forward to the first available `WeekEndDate >= requested date`, and each resolved slice gets its own `__start_YYYY_MM_DD` experiment root:

```bash
pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --start_dates 2020-09-06 2021-01-03 \
  --overwrite
```

Then run the shared workflow:

```bash
GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash bash_scripts/submit_fit_jobs.sh

pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

bash bash_scripts/submit_posterior_predictive_jobs.sh
```

The full US county workflow is documented in [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).
