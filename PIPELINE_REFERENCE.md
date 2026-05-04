# Pipeline Reference

This file documents the stable manifest, spec, and artifact layout used by the synthetic and hybrid pipeline.

## Specs

### Generation Spec

File:

- `data/configs/generation_spec.yaml`

Shape:

- `base`: defaults shared by every experiment
- `experiments`: list of named experiment overrides

Important keys:

- `experiment_root`
- `manifest_path`
- `dimensions.N`, `dimensions.T`
- `generation.seed`, `generation.gibbs_sweeps`
- `x0.generator`, `x0.params`
- `graph.source`
- `graph.generator` and `graph.params` for generated graphs
- `graph.artifact.*` for fixed graph artifacts
- `intervention.source`
- `intervention.generator` and `intervention.params` for generated interventions
- `intervention.artifact.*` for fixed intervention artifacts
- `truth.B`
- `truth.field_mode`
- `truth.field_params`
- `truth.scalars.beta`, `xi`, `eta`
- generation-only intervention scalars `truth.scalars.zeta`, `psi`

Supported source modes:

- `graph.source: generated | fixed_artifact`
- `intervention.source: generated | fixed_artifact`
- `intervention.generator: low_rank_probability`
- `truth.field_mode: random_low_rank | confounded_low_rank`

Dimension resolution:

- fixed graph artifacts can determine `N`
- fixed intervention artifacts can determine `N` and `T`

### Generation Requests

Written by:

- `run_generation_pipeline.py --write_requests`

Default path:

- `experiments/SyntheticHybridExperiments/generation_requests.csv`

Columns:

- `generation_spec_path`
- `experiment_name`
- `experiment_slug`
- `experiment_path`

### Fit Spec

File:

- `data/configs/fits_spec.yaml`

Shape:

- `base`: defaults shared by every fit variant
- `variants`: list of named variant overrides

Important keys:

- `fit_root_name`
- `fit_manifest_path`
- `optimizer.steps`, `optimizer.tol`, `optimizer.seed`
- `optimizer.n_starts`, `optimizer.proximal_lr`
- `B`
- `optimizer_mode`
- `latent_rank`
- `lambda_nuclear`
- `lambda_frobenius`
- `lambda_uv_ridge`
- `v_column_l2_max`
- `estimation.fixed_scalar_params`
- `estimation.beta_mask_pre_s`

### Fit Requests

Written by:

- `run_fit_pipeline.py --write_requests`

Default path:

- `experiments/SyntheticHybridExperiments/fit_requests.csv`

Columns:

- `generation_manifest_path`
- `fits_spec_path`
- `experiment_name`
- `experiment_slug`
- `variant_name`
- `variant_slug`
- `fit_path`

### Posterior Predictive Spec

Files:

- `data/configs/posterior_predictive_spec.yaml`
- `data/configs/posterior_predictive_target_pairs.csv`

`posterior_predictive_spec.yaml` shape:

- `base`
- `runs`

Important run keys:

- `num_samples`
- `gibbs_sweeps`
- `seed`

`posterior_predictive_target_pairs.csv` columns:

- `experiment_name`
- `source_type`
- `variant_name`
- `intervention_source`, optional
- `intervention_name`, optional

Rules:

- `source_type` must be `truth` or `fit`
- truth rows must leave `variant_name` blank
- fit rows must match a `(experiment_name, variant_name)` pair in the fit manifest
- `source_type=truth` requires experiment metadata or manifest `has_truth=true`
- if `intervention_source` is omitted or blank, it defaults to `observed_experiment`
- `intervention_source` may be `observed_experiment` or `saved_intervention`
- `saved_intervention` rows must provide `intervention_name`
- `saved_intervention` rows are routed to the counterfactual output tree rather than the posterior-predictive goodness-of-fit reports

Example observed-intervention target rows:

```csv
experiment_name,source_type,variant_name
synthetic_rank_40_B1,truth,
synthetic_rank_40_B1,fit,rank_40_B1
```

Example saved-intervention counterfactual target rows:

```csv
experiment_name,source_type,variant_name,intervention_source,intervention_name
synthetic_rank_40_B1,truth,,saved_intervention,all_minus_ones
synthetic_rank_40_B1,fit,rank_40_B1,saved_intervention,all_minus_ones
```

The run `seed` is a base seed. Sample `k` uses a deterministic hash of the base seed, run slug, target identity, and `k`, so repeated runs are reproducible while samples within a run are distinct and different targets do not reuse the same random stream.

### Intervention Library Spec

File:

- `data/configs/intervention_library_spec.yaml`

Shape:

- `base`: optional defaults shared by every saved intervention entry
- `interventions`: list of named intervention entries

Each intervention entry is materialized for every experiment listed in the supplied generation manifest.

Important keys:

- `manifest_path`, optional
- `source_kind`
- `activation_scope`
- `unit_index`
- `start_step`

Supported source kinds:

- `observed_experiment`
- `full_on`
- `single_unit_on`

`observed_experiment` copies the realized intervention panel and initial intervention state from the generated experiment.

`full_on` supports:

- `activation_scope: all_time`
- `activation_scope: no_time`
- `activation_scope: from_s`

`single_unit_on` supports:

- `unit_index`
- `activation_scope: all_time | no_time | from_s | from_step`
- `start_step` when `activation_scope: from_step`

Saved intervention panels use `-1/+1` coding for `z`. The saved `z_0` vector may use either `-1/+1` coding or the repo's legacy `0` initial-state convention.

## Manifests

### Generation Manifest

Written by:

- `run_generation_pipeline.py`
- `data/USCountyVaccination/create_us_county_vaccination_experiments.py` for real-data US county experiments

Default path:

- `experiments/SyntheticHybridExperiments/generation_manifest.csv`
- `experiments/USCountyVaccination_US_trimmed/generation_manifest.csv`

Typical columns:

- `experiment_name`
- `experiment_slug`
- `descriptor`
- `experiment_path`
- `intervention_source`
- `graph_source`
- `N`
- `T`
- `s`
- `has_truth`
- `latent_rank`

USCountyVaccination rows also include real-data metadata such as:

- `outcome_code`
- `intervention_code`
- `lag_code`
- `network_name`
- date range and support counts

USCountyVaccination rows set `has_truth=false`.

The synthetic/hybrid generation manifest is refreshed from completed experiment outputs by:

- `run_generation_pipeline.py --refresh_manifest`

### Fit Manifest

Written by:

- `run_fit_pipeline.py`

Default path:

- `experiments/SyntheticHybridExperiments/fit_manifest.csv`

Typical columns:

- `experiment_name`
- `experiment_slug`
- `descriptor`
- `experiment_path`
- `intervention_source`
- `graph_source`
- `variant_name`
- `variant_slug`
- `fit_path`
- `N`
- `T`
- `s`
- `B`
- `latent_rank`
- `fixed_scalar_params`
- `status`

The fit manifest is refreshed from completed fit outputs, and grouped fit reports are rebuilt, by:

- `run_fit_pipeline.py --refresh_manifest`

### Posterior Predictive Manifest

Written by:

- `report_posterior_predictive.py`

Default path:

- `experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv`

Typical columns:

- `experiment_name`
- `experiment_slug`
- `descriptor`
- `experiment_path`
- `intervention_source`
- `graph_source`
- `N`
- `T`
- `s`
- `run_name`
- `run_slug`
- `source_type`
- `source_name`
- `source_slug`
- `target_intervention_source`
- `target_intervention_name`
- `target_intervention_slug`
- `latent_rank`
- `num_samples`
- `gibbs_sweeps`
- `seed`
- `mean_abs_zscore`
- `max_abs_zscore`
- `coverage_rate`
- `num_statistics`
- `output_path`

This unified manifest is rebuilt by scanning completed posterior-predictive and counterfactual output roots. Rows for saved-intervention counterfactual runs leave posterior-predictive ranking metrics blank.

### Intervention Library Manifest

Written by:

- `run_intervention_library.py`

Default path:

- `experiments/SyntheticHybridExperiments/intervention_library_manifest.csv`

Typical columns:

- `experiment_name`
- `experiment_path`
- `intervention_name`
- `intervention_slug`
- `source_kind`
- `N`
- `T`
- `s`
- `output_path`
- `activation_scope`
- `unit_index`
- `start_step`

## Directory Layout

### Generated Experiment Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/
  experiment_metadata.yaml
  generation_realized_config.yaml
  panel_data.npz
  x_0.npy
  z_0.npy
  field_artifacts.npz
  gamma_matrix.npy | gamma_matrix_sparse.npz
  fit_summary.csv
  posterior_predictive_summary.csv
  fits/
  intervention_library/
  posterior_predictive/
  counterfactual/
```

USCountyVaccination experiment roots follow the same shared contract:

```
experiments/USCountyVaccination_US_trimmed/<experiment_slug>/
  experiment_metadata.yaml
  realized_config.yaml
  panel_data.npz
  x_0.npy
  z_0.npy
  node_index.csv
  time_index.csv
  panel_data.csv.gz
  field_artifacts.npz
  gamma_matrix_sparse.npz
  adjacency_edge_list.csv.gz
  binary_definition_summary.csv
  binary_definition_summary.md
  fits/
  intervention_library/
  counterfactual/
```

For USCountyVaccination, `field_artifacts.npz` is a zero-field compatibility artifact and `has_truth=false`; fit variants provide the estimated latent field used downstream.

### Fit Variant Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/fits/<variant_slug>/
  fit_realized_config.yaml
  fit_metadata.yaml
  mple.log
  mple_summary.csv
  estimated_field_artifacts.npz
  estimated_parameter_bundle.npz
  estimated_interaction_matrix.npy | estimated_interaction_matrix_sparse.npz
  true_field_artifacts.npz
  true_interaction_matrix.npy | true_interaction_matrix_sparse.npz
```

### Posterior Predictive Output Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/posterior_predictive/<source_slug>/<run_slug>/
  posterior_predictive_metadata.yaml
  posterior_predictive_stats.csv
```

`source_slug` is:

- `truth`
- `fit_<variant_slug>`

### Intervention Library Artifact Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/intervention_library/<intervention_slug>/
  intervention_metadata.yaml
  intervention_panel.npz
  z_0.npy
```

`intervention_panel.npz` contains:

- `z`

### Counterfactual Output Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/counterfactual/<source_slug>/<intervention_slug>/<run_slug>/
  counterfactual_metadata.yaml
  counterfactual_sample_summaries.npz
  counterfactual_summary.csv
  counterfactual_unit_summary.csv
  counterfactual_time_summary.csv
```

`counterfactual_sample_summaries.npz` contains:

- `overall_mean_magnetization`
- `post_intervention_mean_magnetization`
- `unit_mean_magnetization`
- `time_mean_magnetization`

Full simulated `x` panels are not saved for counterfactual runs.

## Report Outputs

Fit reports:

- per experiment: `fit_summary.csv`
- cross experiment: `best_fit_by_experiment.csv`

Posterior-predictive reports:

- per experiment: `posterior_predictive_summary.csv`
- cross experiment: `best_posterior_predictive_by_experiment.csv`

Counterfactual runs write scenario-specific summary CSVs under each counterfactual output root. They do not participate in posterior-predictive ranking reports.

Saved-intervention `intervention_summaries/<intervention_slug>.csv` reports can compare fit rows against the matching truth row for the same run using overall, post-intervention, unit-level, and time-level mean magnetization summaries.

## Ranking Rules

Fit ranking:

- use `total_recovery_rmse = field_rmse + sum(abs scalar errors)` when truth metrics exist
- break ties by `field_rmse`, then `interaction_fro_error`, then `final_loss`
- otherwise rank by `final_loss`

Posterior-predictive ranking:

- lower `mean_abs_zscore` wins
- ties break on lower `max_abs_zscore`

Counterfactual runs are not ranked by the posterior-predictive report generator because they are not compared to observed outcomes.

## Parameter Bundles

`estimated_parameter_bundle.npz` is the machine-readable bridge from fitting into posterior predictive simulation.

It stores:

- `beta`
- `xi`
- `eta`
- `latent_rank`
- `t_steps`
- `field_matrix`

Truth bundles are loaded from:

- `generation_realized_config.yaml` or `realized_config.yaml`
- `field_artifacts.npz`
- the saved graph artifact in the experiment root

## Report Regeneration

Plan generation requests without materializing experiments:

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/generation_spec.yaml \
  --write_requests
```

Run one planned generation request:

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/generation_spec.yaml \
  --run_request \
  --experiment_slug synthetic_rank_40_b1 \
  --overwrite
```

Refresh the generation manifest from completed outputs:

```bash
pixi run python -u run_generation_pipeline.py \
  --spec_path data/configs/generation_spec.yaml \
  --refresh_manifest
```

Submit all generation requests through SLURM and refresh the generation manifest afterward:

```bash
GENERATION_SPEC_PATH=data/configs/generation_spec.yaml \
GENERATION_OVERWRITE=true \
bash submit_generation_jobs.sh
```

Plan fit requests without launching MPLE:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml \
  --write_requests
```

Run one planned fit request:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml \
  --run_request \
  --experiment_slug synthetic_rank_40_b1 \
  --variant_slug rank_40_b1 \
  --overwrite
```

Refresh the fit manifest and grouped fit reports from completed outputs:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml \
  --refresh_manifest
```

Submit all fit requests through SLURM and refresh the fit manifest/reports afterward:

```bash
GENERATION_MANIFEST_PATH=experiments/SyntheticHybridExperiments/generation_manifest.csv \
FITS_SPEC_PATH=data/configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

Regenerate fit summaries from an existing fit manifest:

```bash
pixi run python -u report_parameter_recovery_detailed.py \
  --manifest experiments/SyntheticHybridExperiments/fit_manifest.csv
```

Refresh the unified posterior-predictive manifest and grouped summaries:

```bash
pixi run python -u report_posterior_predictive.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv
```

Build saved interventions:

```bash
pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --spec_path data/configs/intervention_library_spec.yaml \
  --overwrite
```

Run one posterior predictive or counterfactual target:

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
  --run_name default \
  --overwrite
```

Submit all target/run combinations through SLURM and refresh reports afterward:

```bash
GEN_MANIFEST=experiments/SyntheticHybridExperiments/generation_manifest.csv \
FIT_MANIFEST=experiments/SyntheticHybridExperiments/fit_manifest.csv \
TARGET_PAIRS_PATH=data/configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```

Materialize USCountyVaccination experiments and run the same shared fit/counterfactual path:

```bash
pixi run python -u data/USCountyVaccination/load_raw_data.py
pixi run python -u data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
pixi run python -u data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite

GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh

pixi run python -u run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

GEN_MANIFEST=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FIT_MANIFEST=experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
TARGET_PAIRS_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```
