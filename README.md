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
| Posterior predictive evaluation | `run_posterior_predictive_pipeline.py` | generation manifest, fit manifest, `posterior_predictive_spec.yaml`, `posterior_predictive_target_pairs.csv` | `posterior_predictive_manifest.csv`, predictive summaries |
| Real-data preprocessing | `data/USCountyVaccination/prepare_us_county_vaccination_data.py` | remote NYT, CDC, Bansal, Census, CDC SVI, USDA ERS sources | processed county-week tables and network artifacts |
| Real-data experiment materialization | `data/USCountyVaccination/run_us_county_vaccination_experiments.py` | processed US county artifacts | realized panels, network subsets, experiment folders, optional MPLE fits |

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
- `latent_rank`
- `estimation.fit_intervention_model`
- `estimation.beta_mask_pre_intervention`
- `estimation.beta_mask_rescale`
- `estimation.fixed_scalar_params`
- optimizer `steps`, `tol`, and `seed`

Outputs:

- `experiments/SyntheticHybridExperiments/fit_manifest.csv`
- `fits/<variant_slug>/` under each experiment root
- per-experiment `fit_summary.csv` and `fit_summary.md`
- cross-experiment `best_fit_by_experiment.csv` and `best_fit_by_experiment.md`

### 3. Posterior Predictive

`run_posterior_predictive_pipeline.py` keeps the observed intervention panel fixed and simulates alternative outcome panels from either:

- the experiment truth parameters
- one or more saved MPLE fit bundles

Targeting is explicit. `data/configs/posterior_predictive_target_pairs.csv` must contain:

- `experiment_name`
- `source_type` as `truth` or `fit`
- `variant_name`, blank for truth rows

Run settings live in `data/configs/posterior_predictive_spec.yaml` and currently use `base + runs`.

Default command:

```bash
pixi run python -u run_posterior_predictive_pipeline.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fit_manifest_path experiments/SyntheticHybridExperiments/fit_manifest.csv \
  --target_pairs_path data/configs/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml
```

Outputs:

- `experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv`
- `posterior_predictive/<source_slug>/<run_slug>/...` under each experiment root
- per-experiment `posterior_predictive_summary.csv` and `posterior_predictive_summary.md`
- cross-experiment `best_posterior_predictive_by_experiment.csv` and `best_posterior_predictive_by_experiment.md`

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

- `--steps`, `--tol`, `--seed` override optimizer settings
- `--outcome_only` disables fitting the intervention process
- `--log_file` redirects the MPLE log

`global_params.B` is the active fit-time bound:

- scalar parameters are clipped to `[-B, B]`
- `xi` is also constrained so that `||xi * Gamma||_inf <= B`
- the latent field is projected so its realized infinity norm respects the same bound

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
- `run_posterior_predictive_pipeline.py`: posterior-predictive orchestration
- `mple.py`: conditional MPLE optimizer and artifact writer
- `model_utils.py`: model artifact loading, parameter packing, and field utilities
- `posterior_predictive_utils.py`: predictive simulation and predictive-statistic utilities
- `pipeline_specs.py`: YAML deep-merge, slugging, and manifest helpers
- `tests/test_minimal_pipeline.py`: regression coverage for generation, fitting, summaries, and predictive ranking

## Real-Data Workflow

The nationwide US county workflow is documented separately in [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).
