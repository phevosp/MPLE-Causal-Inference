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
| Synthetic and hybrid generation | `run_generation_pipeline.py` | `data/configs/generation_spec.yaml` | `generation_manifest.csv`, experiment folders |
| MPLE variant fitting | `run_fit_pipeline.py` | `generation_manifest.csv`, `data/configs/fits_spec.yaml` | `fit_manifest.csv`, `fits/<variant>/...`, fit summaries |
| Intervention library generation | `run_intervention_library.py` | generation manifest, `data/configs/intervention_library_spec.yaml` | `intervention_library_manifest.csv`, saved intervention panels |
| Posterior predictive and counterfactual simulation | `run_posterior_predictive_pipeline.py` | generation manifest, fit manifest, `posterior_predictive_spec.yaml`, `posterior_predictive_target_pairs.csv` | `posterior_predictive_manifest.csv`, `counterfactual_manifest.csv`, predictive or counterfactual summaries |
| Real-data raw load | `data/USCountyVaccination/load_raw_data.py` | remote NYT, CDC, Bansal, Census, CDC SVI, USDA ERS sources | cached raw inputs |
| Real-data preprocessing and realization | `data/USCountyVaccination/preprocess_us_county_vaccination_data.py` | cached raw inputs | processed panels, `realized_*`, `shared_panels` |
| Real-data experiment materialization | `data/USCountyVaccination/create_us_county_vaccination_experiments.py` | `realized_*`, `shared_panels` | shared-compatible experiment folders, `generation_manifest.csv` |
| Real-data sensitivity sweep | `run_uscounty_sensitivity_analysis.py` | USCounty generation manifest, start dates, latent ranks, `B` values | sliced experiment folders, sensitivity fit spec, fit manifest, sensitivity summary |

## Synthetic And Hybrid Pipeline

### 1. Generation

`run_generation_pipeline.py` expands a YAML spec with `base + experiments` into concrete experiment folders.

Default config:

- `data/configs/generation_spec.yaml`

Default command:

```bash
pixi run python -u run_generation_pipeline.py --spec_path data/configs/generation_spec.yaml
```

What generation resolves:

- `dimensions.N`, `dimensions.T`, and `dimensions.s`
- truth scalars `beta`, `xi`, `eta`, `zeta`, `psi`
- `truth.latent_rank`
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

- `experiments/SyntheticHybridExperiments/generation_manifest.csv`
- one folder per generated experiment under `experiments/SyntheticHybridExperiments/<experiment_slug>/`

### 2. Fit Variants

`run_fit_pipeline.py` reads a generation manifest plus a fit spec with `base + variants`, creates a fit folder for every `(experiment, variant)` pair, and runs `mple.py`.

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
- `field_mode`: `low_rank` or `nuclear_norm`
- `latent_rank`
- `lambda_nuclear` for `field_mode: nuclear_norm`
- `estimation.fit_intervention_model`
- `estimation.beta_mask_pre_intervention`
- `estimation.fixed_scalar_params`
- optimizer `steps`, `tol`, `seed`, `n_starts`, `adam_steps`, `adam_lr`, `adam_device`, and `proximal_lr`

Outputs:

- `experiments/SyntheticHybridExperiments/fit_manifest.csv`
- `fits/<variant_slug>/` under each experiment root
- per-experiment `fit_summary.csv` and `fit_summary.md`
- cross-experiment `best_fit_by_experiment.csv` and `best_fit_by_experiment.md`

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

`run_posterior_predictive_pipeline.py` keeps the observed intervention panel fixed and simulates alternative outcome panels from either:

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

Default command:

```bash
pixi run python -u run_posterior_predictive_pipeline.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fit_manifest_path experiments/SyntheticHybridExperiments/fit_manifest.csv \
  --target_pairs_path data/configs/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml
```

For an observed-intervention posterior-predictive run, outputs are:

- `experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv`
- `posterior_predictive/<source_slug>/<run_slug>/...` under each experiment root
- per-experiment `posterior_predictive_summary.csv` and `posterior_predictive_summary.md`
- cross-experiment `best_posterior_predictive_by_experiment.csv` and `best_posterior_predictive_by_experiment.md`

For a saved-intervention counterfactual run, outputs are:

- `experiments/SyntheticHybridExperiments/counterfactual_manifest.csv`
- `counterfactual/<source_slug>/<intervention_slug>/<run_slug>/...` under each experiment root
- `counterfactual_sample_summaries.npz`
- `counterfactual_summary.csv`
- `counterfactual_unit_summary.csv`
- `counterfactual_metadata.yaml`

Counterfactual runs do not write `posterior_predictive_stats.csv` and do not participate in posterior-predictive ranking.

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
- `node_factors`
- `time_factors`
- `field_matrix`

`latent_rank = 0` means the realized field is exactly zero and the latent factor arrays are empty.

Each fit folder additionally writes:

- `fit_realized_config.yaml`
- `fit_metadata.yaml`
- `mple.log`
- `mple_summary.csv`
- `mple_summary.md`
- `estimated_field_artifacts.npz`
- `estimated_parameter_bundle.npz`
- `estimated_interaction_matrix.npy` or `estimated_interaction_matrix_sparse.npz`

When truth is available, fit folders also write:

- `true_field_artifacts.npz`
- `true_interaction_matrix.npy` or `true_interaction_matrix_sparse.npz`

Each posterior-predictive run writes:

- `posterior_predictive_stats.csv`
- `posterior_predictive_stats.md`
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

Run posterior predictive:

```bash
pixi run python -u run_posterior_predictive_pipeline.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fit_manifest_path experiments/SyntheticHybridExperiments/fit_manifest.csv \
  --target_pairs_path data/configs/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml
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

Regenerate grouped posterior-predictive reports from an existing predictive manifest:

```bash
pixi run python -u report_posterior_predictive.py \
  --manifest experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv
```

Run the minimal regression suite:

```bash
pixi run python tests/test_minimal_pipeline.py
```

## Manual MPLE Invocation

`mple.py` can fit directly from a generated experiment root or from an explicit fit folder.

Typical manual fit with an explicit fit config:

```bash
pixi run python -u mple.py \
  --data_folder experiments/<experiment>/fits/<variant> \
  --config_path experiments/<experiment>/fits/<variant>/fit_realized_config.yaml \
  --model_artifact_dir experiments/<experiment> \
  --truth_artifact_dir experiments/<experiment> \
  --panel_path experiments/<experiment>/panel_data.npz \
  --x0_path experiments/<experiment>/x_0.npy \
  --z0_path experiments/<experiment>/z_0.npy
```

Useful flags:

- `--steps`, `--tol`, `--seed`, `--n_starts`, `--adam_steps`, `--adam_lr`, `--adam_device`, `--lambda_nuclear`, and `--proximal_lr` override optimizer settings
- `--outcome_only` disables fitting the intervention process
- `--log_file` redirects the MPLE log

When `n_starts > 1`, MPLE runs independent random starts and keeps the fit with the lowest final pseudo-negative log likelihood. When `adam_steps > 0`, each start first runs a PyTorch Adam basin-search stage and then uses L-BFGS-B for the final polish. Per-start diagnostics are saved to `optimizer_start_summary.csv`.

For `field_mode: nuclear_norm`, MPLE optimizes the full latent field directly with a nuclear-norm penalty, using proximal singular-value thresholding instead of the low-rank factorization. The usual `mple_summary.csv` includes the unpenalized MPLE loss, penalized objective, nuclear norm, and effective rank.

`global_params.B` is the active fit-time bound:

- scalar parameters are clipped to `[-B, B]`
- `xi` is also constrained so that `||xi * Gamma||_inf <= B`
- the latent field is projected so its maximum absolute entry respects the same bound

## Shell Wrappers

The repo ships lightweight wrappers:

- `generate_data.sh`
- `run_experiments.sh`
- `run_posterior_predictive.sh`

They call the same Python entry points and accept environment-variable overrides:

- `GENERATION_SPEC_PATH`, `GENERATION_OVERWRITE`
- `GENERATION_MANIFEST_PATH`, `FITS_SPEC_PATH`, `FIT_OVERWRITE`
- `FIT_MANIFEST_PATH`, `TARGET_PAIRS_PATH`, `POSTERIOR_PREDICTIVE_SPEC_PATH`, `POSTERIOR_PREDICTIVE_OVERWRITE`

## Repository Map

- `data/synthetic_data_generation.py`: synthetic and hybrid artifact materialization
- `run_generation_pipeline.py`: spec expansion for generation experiments
- `run_fit_pipeline.py`: spec expansion for MPLE fit variants
- `run_intervention_library.py`: reusable intervention-panel materialization
- `run_posterior_predictive_pipeline.py`: posterior-predictive orchestration
- `mple.py`: conditional MPLE optimizer and artifact writer
- `model_utils.py`: model artifact loading, parameter packing, and field utilities
- `posterior_predictive_utils.py`: predictive simulation, intervention loading, and summary utilities
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

Then run the shared workflow:

```bash
pixi run python run_fit_pipeline.py \
  --manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --fits_spec_path data/USCountyVaccination/experiment_configs/fits_spec.yaml \
  --overwrite

pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

pixi run python run_posterior_predictive_pipeline.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --fit_manifest_path experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
  --target_pairs_path data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
  --spec_path data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
  --overwrite
```

The full US county workflow is documented in [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).

Start-week, latent-rank, and `B` sensitivity for USCountyVaccination is handled by:

```bash
pixi run python run_uscounty_sensitivity_analysis.py \
  --source_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --output_root experiments/USCountyVaccination_US_sensitivity \
  --experiment_names outcome_death_rate_100k_ge_2__intervention_complete_cov_ge_40__lag_2w__contiguity \
  --start_dates 2020-01-26 2020-03-01 2020-06-07 2020-09-06 2021-01-03 \
  --latent_ranks 0 10 20 40 \
  --B_values 0.5 1 2 5 \
  --lambda_nuclear_values 0.0001 0.0003 0.001 0.003 0.01 \
  --n_starts 5 \
  --adam_steps 1000 \
  --overwrite \
  --run_fits
```

This writes `sensitivity_summary.csv` and `sensitivity_summary.md` ranked by MPLE final loss.
