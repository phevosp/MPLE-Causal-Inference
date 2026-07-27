# MPLE-Causal-Inference

Spec-driven conditional MPLE experiments for binary outcome/intervention panel data with:

- a fixed known graph `Gamma`
- a scalar interaction temperature `xi`
- an optional low-rank external field

The repository currently supports two main workflows:

1. Synthetic and hybrid experiment generation, fitting, and posterior-predictive evaluation
2. Nationwide US county vaccination data preparation and experiment materialization (see [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md))

## Overview

This repo is organized around a small set of shared pipeline entrypoints that work across synthetic, hybrid, and real-data experiments. At the top level, `run_generation_pipeline.py`, `run_fit_pipeline.py`, `run_intervention_library.py`, `run_posterior_predictive.py`, `build_splits.py`, `run_cv_folds.py`, and `run_test_evaluation.py` cover the full lifecycle from data materialization through model selection and held-out evaluation.

In practice, the codebase supports:

- synthetic data generation from YAML specs in `data/configs/`
- hybrid data generation that mixes generated and fixed graph or intervention artifacts
- fixed-hyperparameter fits from a generation manifest plus a fit spec, including standard MPLE variants, treatment-conditioned low-rank baselines, and SNN treatment-split completion
- posterior-predictive and counterfactual simulation, including saved intervention scenarios
- split-bundle construction and cross-validation for hyperparameter selection
- outer-masked retraining and evaluation on unseen test sets
- real-world experiment construction from `data/USCountyVaccination/`

Committed experiment-family configs live under `data/configs/REVISIONS/*/`, while the `data/USCountyVaccination/` subtree contains the staged real-data materialization pipeline. For the applied-data workflow, experiment scope, and example commands, see [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).

## Quickstart

Verify your install works end-to-end in under 2 minutes using the toy experiment (`N=30`, `T=10`, rank `2`, `500` optimizer steps):

```bash
pixi install
pixi run python -u run_generation_pipeline.py --spec_path data/configs/quickstart_generation_spec.yaml
pixi run python -u run_fit_pipeline.py --manifest_path experiments/Quickstart/generation_manifest.csv --fits_spec_path data/configs/quickstart_fits_spec.yaml
pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/Quickstart/generation_manifest.csv \
  --fit_manifest_path experiments/Quickstart/fit_manifest.csv \
  --target_pairs_path data/configs/quickstart_posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name Quickstart \
  --source_type fit \
  --variant_name alternating_rank \
  --intervention_source observed_experiment \
  --run_name default
```

Outputs land in `experiments/Quickstart/`. Once this works, move to the larger synthetic, hybrid, or US-county experiment families below.

### SNN Quickstart

For a minimal SNN example, use the short spec files `data/configs/quickstart_snn_fits_spec.yaml` and `data/configs/quickstart_snn_posterior_predictive_spec.yaml`. SNN fits currently support only `saved_intervention` counterfactual runs, so build the intervention library first and then choose a saved intervention target from `data/configs/quickstart_snn_posterior_predictive_target_pairs.csv`.

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/Quickstart/generation_manifest.csv \
  --fits_spec_path data/configs/quickstart_snn_fits_spec.yaml

pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/Quickstart/generation_manifest.csv \
  --spec_path data/configs/intervention_library_spec.yaml \
  --overwrite

pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/Quickstart/generation_manifest.csv \
  --fit_manifest_path experiments/Quickstart/fit_manifest_snn.csv \
  --target_pairs_path data/configs/quickstart_snn_posterior_predictive_target_pairs.csv \
  --spec_path data/configs/quickstart_snn_posterior_predictive_spec.yaml \
  --experiment_name Quickstart \
  --source_type fit \
  --variant_name snn_1nn \
  --intervention_source saved_intervention \
  --intervention_name no_intervention \
  --run_name default
```

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

## Workflows

| Workflow | Entry Point (Python / Bash) | Description |
| --- | --- | --- |
| Generation | `run_generation_pipeline.py` / `submit_generation_jobs.sh` | Materialize synthetic or hybrid experiments from YAML specs. |
| Real-World Data | `data/USCountyVaccination/load_raw_data.py`, `data/USCountyVaccination/preprocess_us_county_vaccination_data.py`, `data/USCountyVaccination/create_us_county_vaccination_experiments.py` | Stage raw US county inputs into shared-pipeline experiment roots and a compatible generation manifest. |
| Fits | `run_fit_pipeline.py` / `submit_fit_jobs.sh` | Run standard MPLE fits, treatment-conditioned low-rank baselines, or SNN treatment-split fits for every `(experiment, variant)` pair. |
| Posterior Predictive | `run_intervention_library.py`, `run_posterior_predictive.py` / `submit_posterior_predictive_jobs.sh` | Build reusable intervention scenarios and run observed-intervention posterior predictive or saved-intervention counterfactual simulations. |
| Split Bundle Construction | `build_splits.py` | Build the `splits/train_cv/...` and `splits/test_train_cv/...` bundles consumed by CV and held-out test evaluation. |
| CV | `run_cv_folds.py` / `submit_cv_jobs.sh` | Score hyperparameter candidates over saved split bundles and write `best_candidate.yaml` selections for each search. |
| Test Set Evaluation | `run_fit_pipeline.py --fit_mode outer_masked`, `run_test_evaluation.py` / `submit_fit_jobs.sh`, `submit_test_evaluation_jobs.sh` | Refit the best `test_train_cv` candidates on training support only, then evaluate the resulting train fits on unseen test support alongside simple prediction baselines. |

## Workflow Details

### Generation

`run_generation_pipeline.py` expands `base + experiments` specs into concrete experiment folders and a shared `generation_manifest.csv`. The same experiment contract is used for:

- fully synthetic experiments with generated graphs and generated interventions
- hybrid experiments with fixed graph artifacts, fixed intervention artifacts, or both

Hybrid experiments read from saved graph or intervention artifacts from real world data (see [Real-World Data](#real-world-data)).

### Real-World Data

The real-world data workflow lives under `data/USCountyVaccination/` and stages public county-level COVID and vaccination inputs into the same shared experiment contract used downstream by synthetic and hybrid runs. The three main stages are:

1. `load_raw_data.py` to download or refresh raw source files
2. `preprocess_us_county_vaccination_data.py` to build processed county-week tables, threshold panels, and realized artifacts
3. `create_us_county_vaccination_experiments.py` to write experiment roots plus a shared `generation_manifest.csv`

Once materialized, those experiment roots can flow through the same fit, intervention-library, posterior-predictive, split-construction, CV, and test-evaluation runners as any other experiment family. The applied-data details and examples live in [data/USCountyVaccination/README.md](data/USCountyVaccination/README.md).

### Fits

`run_fit_pipeline.py` in standard mode reads a generation manifest plus a fit spec and runs one MPLE fit per `(experiment, variant)` pair. Fit specs control the optimizer mode, latent rank, scalar constraints, masking settings, and any low-rank regularization. Standard fits write `fit_manifest.csv` plus per-experiment and cross-experiment summaries.

This is the right path when you already know which hyperparameters or MPLE variants you want to compare and do not need model selection first.

The current fit families are:

- Standard MPLE variants: `no_external_field`, `nuclear_norm`, `exact_rank_manifold`, `alternating_latent_rank`, and `concurrent_latent_rank`.
- Treatment-conditioned low-rank baselines: `alternating_treatment_split_latent_rank` and `alternating_treatment_shared_unit_latent_rank`. These require `estimation.fixed_scalar_params` to fix `beta`, `xi`, and `eta`, and the committed revision examples live in `data/configs/REVISIONS/synth/*baselines*` and `data/configs/REVISIONS/synth_no_interference/*baselines*`.
- SNN treatment-split completion: `snn_treatment_split`. This uses an `snn:` mapping instead of MPLE optimizer parameters, dispatches to `snn_fit.py`, and writes `estimated_snn_artifacts.npz`, `snn_summary.csv`, and `snn.log`.

Two important limitations are easy to miss:

- `run_fit_pipeline.py --fit_mode outer_masked` does not support `snn_treatment_split`.
- SNN fits are excluded from the standard parameter-recovery fit reports because they do not produce MPLE parameter bundles.

There is also a direct held-out retraining mode for fit-spec-driven experiments:

- `run_fit_pipeline.py --fit_mode masked_standard` reads explicit variants from `--fits_spec_path`, applies a saved `test_train_cv` outer training mask, and writes train-fit outputs under `train_fits/<scope_slug>/...`.
- `--scope_slug` is required in `masked_standard` mode and namespaces the request CSV, refreshed train-fit manifest, and output directories.
- This path does not use `best_candidate.yaml`, `--cv_spec_path`, or `--search_slug`.

### Posterior Predictive

Posterior-predictive work uses two pieces:

- `run_intervention_library.py` to save reusable intervention panels such as no intervention or all interventions
- `run_posterior_predictive.py` to simulate outcomes under truth parameters or saved fit bundles

The `run_posterior_predictive.py` command supports two intervention sources: `observed_experiment` to use the realized interventions from the experiment, and `saved_intervention` to use the saved panels from the intervention library. The former is for goodness-of-fit checks against observed outcomes, while the latter is for counterfactual simulations under hypothetical scenarios generated by `run_intervention_library.py`. The target list comes from a `posterior_predictive_target_pairs.csv` file, while shared simulation settings come from `data/configs/posterior_predictive_spec.yaml`.

Support depends on the parameter source:

- `source_type=truth` and ordinary MPLE `source_type=fit` targets support both `observed_experiment` and `saved_intervention`.
- Treatment-conditioned low-rank baseline fits also run through `source_type=fit`, but `saved_intervention` is the natural path because the code recomposes control and treated field surfaces against the requested intervention panel.
- SNN fits only support `saved_intervention`. They do not run Gibbs sampling, they emit a deterministic counterfactual panel built from saved treated and untreated completed matrices, and `observed_experiment` requests are rejected.

### Split Bundle Construction

`build_splits.py` constructs the saved split bundles used by both CV and held-out test evaluation. It reads a generation manifest plus a CV spec and writes one bundle per required split configuration under:

- `splits/train_cv/folds_<k>/`
- `splits/test_train_cv/outer_<outer>__test_<fold>__inner_<inner>/`

The two supported split kinds are:

- `train_cv` for inner model-selection folds only
- `test_train_cv` for an outer held-out test split plus inner model-selection folds

These bundles capture the outer active/test masks and the inner training/separator/validation masks. They are the prerequisite for `run_cv_folds.py` and for held-out test evaluation later in the pipeline.

### CV

`run_cv_folds.py` consumes the saved split bundles and expands the cv spec into a search over a grid of candidate hyperparameters. It scores candidates fold by fold, writes `candidate_scores.csv`, and selects a winner in `best_candidate.yaml` for each `(experiment, search)` pair. A single CV spec can contain multiple searches, and each search can have multiple candidates. 

The `best_candidate.yaml` file is consumed directly by the held-out test evaluation workflow (see below). For the standard pipeline, the hyperparameters of the best candidate are manually copied into a fit spec that is then used to run the final fits for posterior-predictive simulation. For a more automated workflow, the fit pipeline could be extended to read the CV search results directly and run the winning candidates without an intermediate fit spec.

The July 2026 baseline work added separate committed CV specs for the synthetic families:

- `data/configs/REVISIONS/synth/cv_spec_baselines.yaml`
- `data/configs/REVISIONS/synth_no_interference/cv_spec_baselines.yaml`

### Test Set Evaluation

Held-out testing is a four-step chain:

1. Build `test_train_cv` split bundles with `build_splits.py`.
2. Run `run_cv_folds.py` so each `test_train_cv` search writes its `best_candidate.yaml`.
3. Run `run_fit_pipeline.py --fit_mode outer_masked` to refit the winning candidates on training support only, producing `train_fits/...` outputs and a `train_fit_manifest__<search_slug>.csv`.
4. Run `run_test_evaluation.py` on that train-fit manifest to score the saved fits on the unseen test support.

This separates hyperparameter selection from final test scoring and keeps the held-out test units out of both candidate selection and retraining.

Each `test_metrics.yaml` report now contains:

- the saved fit's deterministic point-prediction losses and calibration metrics on full, post-`s`, treated, and untreated test support
- sampled magnetization diagnostics for the fit
- full-panel regeneration diagnostics bucketed by training, separator, test, treated-test, and untreated-test support
- a `baselines` block with the same test-style metrics for the simple `time_step_mean`, `unit_mean`, and `persistence` predictors

If you only want to refresh the baseline comparison block for existing reports, `run_test_evaluation.py --baselines_only` recomputes just that section and merges it into the saved YAML.

Direct held-out evaluation without CV is also supported:

1. Build the `test_train_cv` split bundle with `build_splits.py`. This still uses the existing CV-spec schema as a split driver, but no CV scoring is required.
2. Run `run_fit_pipeline.py --fit_mode masked_standard` with a fit spec and `--scope_slug ...` to refit explicit variants on the outer training support only.
3. Run `run_test_evaluation.py` on the resulting `train_fit_manifest__<scope_slug>.csv`.

For SNN fits in this direct path, held-out evaluation is deterministic: the evaluator recomposes the saved treated and untreated completed surfaces against the realized intervention panel and reports deterministic loss, calibration, and magnetization summaries instead of Gibbs-sampled diagnostics.

## Example Experiments

### 1. Synthetic Experiment Family With CV, Fixed Fits, And Posterior Predictive

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/REVISIONS/synth/generation_spec.yaml

pixi run python -u build_splits.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec.yaml \
  --overwrite

pixi run python -u run_cv_folds.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec.yaml

pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --fits_spec_path data/configs/REVISIONS/synth/fits_spec.yaml

pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --spec_path data/configs/REVISIONS/synth/intervention_library_spec.yaml \
  --overwrite

pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --fit_manifest_path experiments/Synthetic/fit_manifest_x10.csv \
  --target_pairs_path data/configs/REVISIONS/synth/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name confounding_strong_1 \
  --source_type fit \
  --variant_name alternating_rank_3_uv_5_e2 \
  --intervention_source saved_intervention \
  --intervention_name all_intervention \
  --run_name default
```

### 2. Real-World USCounty Experiment With Fits And Posterior Predictive

The committed US-county revision configs assume the shared workflow outputs live under `experiments/USCounty`, so this example uses that root directly.

```bash
pixi run python data/USCountyVaccination/load_raw_data.py

pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_30 \
  --lags 2w \
  --networks distance_kernel_8 \
  --overwrite

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_30 \
  --lags 2w \
  --networks distance_kernel_8 \
  --start_dates 2020-03-01 \
  --overwrite

pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/USCounty/generation_manifest.csv \
  --fits_spec_path data/configs/REVISIONS/uscounty/fits_spec.yaml

pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/USCounty/generation_manifest.csv \
  --spec_path data/configs/REVISIONS/uscounty/intervention_library_spec.yaml \
  --overwrite

pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/USCounty/generation_manifest.csv \
  --fit_manifest_path experiments/USCounty/fit_manifest.csv \
  --target_pairs_path data/configs/REVISIONS/uscounty/posterior_predictive_target_pairs.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name outcome_death_rate_100k_ge_2__intervention_complete_cov_ge_30__lag_2w__distance_kernel_8__start_2020_03_01 \
  --source_type fit \
  --variant_name alternating_rank_5_uv_5_e2 \
  --intervention_source observed_experiment \
  --run_name default
```

### 3. Synthetic Held-Out Test Evaluation With `default_test_train`

The CV command below runs every committed search in the synth CV spec, including `default_test_train`. The train-fit and test-evaluation steps then narrow to that held-out-test search.

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/REVISIONS/synth/generation_spec.yaml

pixi run python -u build_splits.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec.yaml \
  --overwrite

pixi run python -u run_cv_folds.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec.yaml

pixi run python -u run_fit_pipeline.py \
  --fit_mode outer_masked \
  --manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec.yaml \
  --search_slug default_test_train

pixi run python -u run_test_evaluation.py \
  --fit_manifest_path experiments/Synthetic/train_fit_manifest__default_test_train.csv
```

### 4. Synthetic Baseline Fits And Counterfactuals

The committed synthetic baseline configs run the treatment-conditioned low-rank baselines introduced in late July 2026.

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/REVISIONS/synth/generation_spec.yaml

pixi run python -u build_splits.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec_baselines.yaml \
  --overwrite

pixi run python -u run_cv_folds.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --cv_spec_path data/configs/REVISIONS/synth/cv_spec_baselines.yaml

pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --fits_spec_path data/configs/REVISIONS/synth/fits_spec_baselines.yaml

pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --spec_path data/configs/REVISIONS/synth/intervention_library_spec.yaml \
  --overwrite

pixi run python -u run_posterior_predictive.py \
  --generation_manifest_path experiments/Synthetic/generation_manifest_x10.csv \
  --fit_manifest_path experiments/Synthetic/fit_manifest_baselines_x10.csv \
  --target_pairs_path data/configs/REVISIONS/synth/posterior_predictive_target_pairs_baselines.csv \
  --spec_path data/configs/posterior_predictive_spec.yaml \
  --experiment_name confounding_strong_1 \
  --source_type fit \
  --variant_name treatment_split_rank_1_uv_1_e2 \
  --intervention_source saved_intervention \
  --intervention_name no_intervention \
  --run_name default
```

### 5. Small Synthetic Direct Train/Test Evaluation For MPLE And SNN

This workflow uses committed specs under `data/configs/REVISIONS/synth_direct_eval/` and does not run CV. The split builder still reads a CV-style split-driver spec, but only to materialize the saved `test_train_cv` masks.

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/REVISIONS/synth_direct_eval/generation_spec.yaml

pixi run python -u build_splits.py \
  --generation_manifest_path experiments/SyntheticDirectEval/generation_manifest.csv \
  --cv_spec_path data/configs/REVISIONS/synth_direct_eval/split_cv_spec.yaml \
  --overwrite

pixi run python -u run_fit_pipeline.py \
  --fit_mode masked_standard \
  --manifest_path experiments/SyntheticDirectEval/generation_manifest.csv \
  --fits_spec_path data/configs/REVISIONS/synth_direct_eval/mple_fits_spec.yaml \
  --scope_slug small_mple \
  --split_kind test_train_cv \
  --num_folds 2 \
  --outer_num_folds 2 \
  --test_fold_id 1

pixi run python -u run_test_evaluation.py \
  --fit_manifest_path experiments/SyntheticDirectEval/train_fit_manifest__small_mple.csv

pixi run python -u run_fit_pipeline.py \
  --fit_mode masked_standard \
  --manifest_path experiments/SyntheticDirectEval/generation_manifest.csv \
  --fits_spec_path data/configs/REVISIONS/synth_direct_eval/snn_fits_spec.yaml \
  --scope_slug small_snn \
  --split_kind test_train_cv \
  --num_folds 2 \
  --outer_num_folds 2 \
  --test_fold_id 1

pixi run python -u run_test_evaluation.py \
  --fit_manifest_path experiments/SyntheticDirectEval/train_fit_manifest__small_snn.csv
```
