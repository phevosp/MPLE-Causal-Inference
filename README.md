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
pixi run quickstart-generate
pixi run quickstart-fit
```

Outputs land in `experiments/Quickstart/`. Once this works, proceed to the full synthetic or real-data workflows below.

## Configuration Guide

All pipeline YAML specs use a `base + named entries` pattern: every named entry is deep-merged with `base`, inheriting all fields it doesn't override. The spec is then expanded into one config per entry by `pipeline_specs.expand_named_entries()`.

**There are two separate config directories — do not mix them:**

| Directory | Used by |
| --- | --- |
| `data/configs/` | Synthetic/hybrid pipeline (`run_generation_pipeline.py`, `run_fit_pipeline.py`, `run_posterior_predictive.py`, `report_posterior_predictive.py`) |
| `data/USCountyVaccination/experiment_configs/` | Real-data pipeline |

Both directories contain identically named files (`fits_spec.yaml`, etc.) for their respective workflows. Updating fitting behavior requires editing both files independently.

**Key `fits_spec.yaml` fields:**

- `optimizer_mode`: one of `no_external_field`, `nuclear_norm`, `exact_rank_manifold`, `alternating_latent_rank`, or `concurrent_latent_rank`.
- `latent_rank`: must be ≥ 1 for `exact_rank_manifold`, `alternating_latent_rank`, and `concurrent_latent_rank`; ignored for `no_external_field` and `nuclear_norm`.
- `estimation.fixed_scalar_params`: scalars held **fixed** at these values (not initial guesses). Leave as `{}` to estimate all scalars freely.
- `estimation.beta_mask_pre_s`: if `true`, the fit model masks the `beta * z` term for `t < s`, and fit-based posterior predictive uses the same masked beta effect.
- `lambda_nuclear`: only active for `nuclear_norm`.
- `lambda_frobenius`: only active for `exact_rank_manifold`.
- `lambda_uv_ridge`: only active for `alternating_latent_rank` and `concurrent_latent_rank`.

**Key `generation_spec.yaml` truth fields:**

- `truth.field_mode`: `random_low_rank`, `node_bias_plus_smooth_time_drift`, or `low_rank_plus_early_treatment_confounding`
- `truth.field_params`: optional tuning knobs for structured fields such as `node_bias_scale`, `drift_scale`, `time_trend_sharpness`, `confounding_bias_scale`, and `untreated_score_value`

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
| Synthetic and hybrid generation | `run_generation_pipeline.py`, `submit_generation_jobs.sh` | `data/configs/generation_spec.yaml` | `generation_requests.csv`, `generation_manifest.csv`, experiment folders |
| MPLE variant fitting | `run_fit_pipeline.py`, `submit_fit_jobs.sh` | `generation_manifest.csv`, `data/configs/fits_spec.yaml` | `fit_requests.csv`, `fit_manifest.csv`, `fits/<variant>/...`, fit summaries |
| Intervention library generation | `run_intervention_library.py` | generation manifest, `data/configs/intervention_library_spec.yaml` | `intervention_library_manifest.csv`, saved intervention panels |
| Posterior predictive and counterfactual simulation | `run_posterior_predictive.py`, `report_posterior_predictive.py`, `submit_posterior_predictive_jobs.sh` | generation manifest, fit manifest, `posterior_predictive_spec.yaml`, `posterior_predictive_target_pairs.csv` | `posterior_predictive_manifest.csv`, predictive or counterfactual summaries |
| Real-data raw load | `data/USCountyVaccination/load_raw_data.py` | remote NYT, CDC, Bansal, Census, CDC SVI, USDA ERS sources | cached raw inputs |
| Real-data preprocessing and realization | `data/USCountyVaccination/preprocess_us_county_vaccination_data.py` | cached raw inputs | processed panels, `realized_*`, `shared_panels` |
| Real-data experiment materialization | `data/USCountyVaccination/create_us_county_vaccination_experiments.py` | `realized_*`, `shared_panels` | shared-compatible experiment folders, `generation_manifest.csv` |

## Synthetic And Hybrid Pipeline

### 1. Generation

`run_generation_pipeline.py` expands a YAML spec with `base + experiments` into concrete experiment folders. It now also supports a staged request workflow: write `generation_requests.csv`, materialize one experiment by `experiment_slug`, or refresh `generation_manifest.csv` from completed outputs.

Default config:

- `data/configs/generation_spec.yaml`

Default command:

```bash
pixi run python -u run_generation_pipeline.py --spec_path data/configs/generation_spec.yaml
```

What generation resolves:

- `dimensions.N`, `dimensions.T`, and `dimensions.s`
- outcome truth scalars `beta`, `xi`, `eta`
- generation-only intervention scalars `zeta`, `psi`
- `truth.latent_rank`
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

If graph or intervention artifacts are fixed, the pipeline can infer `N`, `T`, and `s` directly from those artifacts.

Outputs:

- `experiments/SyntheticHybridExperiments/generation_requests.csv`
- `experiments/SyntheticHybridExperiments/generation_manifest.csv`
- one folder per generated experiment under `experiments/SyntheticHybridExperiments/<experiment_slug>/`

Batch submission command:

```bash
GENERATION_SPEC_PATH=data/configs/generation_spec.yaml \
GENERATION_OVERWRITE=true \
bash submit_generation_jobs.sh
```

### 2. Fit Variants

`run_fit_pipeline.py` reads a generation manifest plus a fit spec with `base + variants`, creates a fit folder for every `(experiment, variant)` pair, and runs `mple.py`. It now also supports a staged request workflow: write `fit_requests.csv`, run one `(experiment_slug, variant_slug)` request, or refresh `fit_manifest.csv` and grouped fit reports from completed outputs.

Default config:

- `data/configs/fits_spec.yaml`

Default command:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml
```

Each fit variant controls:

- `B`
- `optimizer_mode`
- `latent_rank`
- `lambda_nuclear` for `nuclear_norm`
- `lambda_frobenius` for `exact_rank_manifold`
- `lambda_uv_ridge` for `alternating_latent_rank` and `concurrent_latent_rank`
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
FITS_SPEC_PATH=data/configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

### 3. Intervention Library

`run_intervention_library.py` creates reusable intervention panels under a generated experiment root. These panels are useful for counterfactual posterior-predictive runs where the outcomes are simulated under an intervention different from the one observed in the original experiment.

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
bash submit_posterior_predictive_jobs.sh
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
- `counterfactual_metadata.yaml`

Counterfactual rows are included in the unified `posterior_predictive_manifest.csv`, but they do not write `posterior_predictive_stats.csv` and are excluded from posterior-predictive ranking.

Saved-intervention `intervention_summaries/<intervention_slug>.csv` files now include truth-referenced counterfactual comparison columns when a matching `truth` row exists for the same run. These reports compare overall, post-intervention, and unit-level mean magnetization summaries against the saved truth row, but they do not report time-level errors because current counterfactual artifacts do not store per-time summaries.

Example counterfactual target-pairs file:

```csv
experiment_name,source_type,variant_name,intervention_source,intervention_name
synthetic_rank_40_B1,truth,,saved_intervention,all_minus_ones
synthetic_rank_40_B1,fit,rank_40_B1,saved_intervention,all_minus_ones
```

The `seed` in `posterior_predictive_spec.yaml` is the starting seed for a run. Individual samples use `seed + sample_index`, so `num_samples` produces reproducible but distinct draws.

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
pixi run python -u run_generation_pipeline.py --spec_path data/configs/generation_spec.yaml
```

Fit all configured variants:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml
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
GENERATION_SPEC_PATH=data/configs/generation_spec.yaml \
FITS_SPEC_PATH=data/configs/fits_spec.yaml \
INTERVENTION_LIBRARY_SPEC_PATH=data/configs/intervention_library_spec.yaml \
TARGET_PAIRS_PATH=data/configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/configs/posterior_predictive_spec.yaml \
bash run_tests.sh
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

`global_params.B` is the active fit-time bound:

- scalar parameters are clipped to `[-B, B]`
- `xi` is also constrained so that `||xi * Gamma||_inf <= B`
- the latent field is projected so its maximum absolute entry respects the same bound

## Shell Wrappers

The repo ships lightweight wrappers:

- `run_generation_job.sh`
- `submit_generation_jobs.sh`
- `run_fit_job.sh`
- `submit_fit_jobs.sh`
- `run_posterior_predictive_job.sh`
- `submit_posterior_predictive_jobs.sh`
- `run_tests.sh`

They call the same Python entry points and accept environment-variable overrides:

- `GENERATION_SPEC_PATH`, `GENERATION_OVERWRITE`
- `GENERATION_MANIFEST_PATH`, `FITS_SPEC_PATH`, `FIT_OVERWRITE`
- `GEN_MANIFEST`, `FIT_MANIFEST`, `TARGET_PAIRS_PATH`, `POSTERIOR_PREDICTIVE_SPEC_PATH`, `POSTERIOR_PREDICTIVE_OVERWRITE`
- `INTERVENTION_LIBRARY_SPEC_PATH`

## Repository Map

- `data/synthetic_data_generation.py`: synthetic and hybrid artifact materialization
- `run_generation_pipeline.py`: generation request planning, single-request execution, and manifest refresh
- `submit_generation_jobs.sh`: SLURM fan-out for generation requests plus manifest refresh barrier
- `run_fit_pipeline.py`: fit request planning, single-fit execution, and manifest refresh/report rebuild
- `submit_fit_jobs.sh`: SLURM fan-out for fit requests plus manifest/report refresh barrier
- `run_intervention_library.py`: reusable intervention-panel materialization
- `run_posterior_predictive.py`: single-target posterior-predictive/counterfactual execution
- `report_posterior_predictive.py`: manifest refresh plus grouped posterior-predictive reporting
- `run_tests.sh`: staged generation → fit → intervention → posterior-predictive shell orchestrator
- `mple.py`: conditional MPLE optimizer and artifact writer
- `model_utils.py`: model artifact loading, parameter packing, and field utilities
- `loading_utils.py`: experiment/panel artifact loading plus fit/truth parameter-bundle loading
- `intervention_utils.py`: intervention construction and saved-intervention artifact helpers
- `posterior_predictive_utils.py`: predictive simulation and posterior-predictive summary utilities
- `pipeline_specs.py`: YAML deep-merge, slugging, and manifest helpers
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
bash submit_fit_jobs.sh

pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

bash submit_posterior_predictive_jobs.sh
```

The full US county workflow is documented in [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).
