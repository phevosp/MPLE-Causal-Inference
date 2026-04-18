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
- `dimensions.N`, `dimensions.T`, `dimensions.s`
- `generation.seed`, `generation.gibbs_sweeps`
- `x0.generator`, `x0.params`
- `graph.source`
- `graph.generator` and `graph.params` for generated graphs
- `graph.artifact.*` for fixed graph artifacts
- `intervention.source`
- `intervention.artifact.*` for fixed intervention artifacts
- `truth.B`
- `truth.latent_rank`
- `truth.scalars.beta`, `xi`, `eta`, `zeta`, `psi`

Supported source modes:

- `graph.source: generated | fixed_artifact`
- `intervention.source: generated | fixed_artifact`

Dimension resolution:

- fixed graph artifacts can determine `N`
- fixed intervention artifacts can determine `N`, `T`, and `s`
- generated interventions require an explicit `dimensions.s`

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
- `B`
- `latent_rank`
- `estimation.fit_intervention_model`
- `estimation.beta_mask_pre_intervention`
- `estimation.beta_mask_rescale`
- `estimation.fixed_scalar_params`

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

Rules:

- `source_type` must be `truth` or `fit`
- truth rows must leave `variant_name` blank
- fit rows must match a `(experiment_name, variant_name)` pair in the fit manifest

## Manifests

### Generation Manifest

Written by:

- `run_generation_pipeline.py`

Default path:

- `experiments/SyntheticHybridExperiments/generation_manifest.csv`

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
- `fit_intervention_model`
- `fixed_scalar_params`
- `status`

### Posterior Predictive Manifest

Written by:

- `run_posterior_predictive_pipeline.py`

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
- `latent_rank`
- `B`
- `fit_intervention_model`
- `num_samples`
- `gibbs_sweeps`
- `seed`
- `mean_abs_zscore`
- `max_abs_zscore`
- `coverage_rate`
- `num_statistics`
- `output_path`

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
  fit_summary.md
  posterior_predictive_summary.csv
  posterior_predictive_summary.md
  fits/
  posterior_predictive/
```

### Fit Variant Root

```
experiments/SyntheticHybridExperiments/<experiment_slug>/fits/<variant_slug>/
  fit_realized_config.yaml
  fit_metadata.yaml
  mple.log
  mple_summary.csv
  mple_summary.md
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
  posterior_predictive_stats.md
```

`source_slug` is:

- `truth`
- `fit_<variant_slug>`

## Report Outputs

Fit reports:

- per experiment: `fit_summary.csv`, `fit_summary.md`
- cross experiment: `best_fit_by_experiment.csv`, `best_fit_by_experiment.md`

Posterior-predictive reports:

- per experiment: `posterior_predictive_summary.csv`, `posterior_predictive_summary.md`
- cross experiment: `best_posterior_predictive_by_experiment.csv`, `best_posterior_predictive_by_experiment.md`

## Ranking Rules

Fit ranking:

- use `total_recovery_rmse = field_rmse + sum(abs scalar errors)` when truth metrics exist
- break ties by `field_rmse`, then `interaction_fro_error`, then `final_loss`
- otherwise rank by `final_loss`

Posterior-predictive ranking:

- lower `mean_abs_zscore` wins
- ties break on lower `max_abs_zscore`

## Parameter Bundles

`estimated_parameter_bundle.npz` is the machine-readable bridge from fitting into posterior predictive simulation.

It stores:

- `beta`
- `xi`
- `eta`
- `zeta`
- `psi`
- `fit_intervention_model`
- `latent_rank`
- `t_steps`
- `field_matrix`

Truth bundles are loaded from:

- `generation_realized_config.yaml` or `realized_config.yaml`
- `field_artifacts.npz`
- the saved graph artifact in the experiment root

## Report Regeneration

Regenerate fit summaries from an existing fit manifest:

```bash
pixi run python -u report_parameter_recovery_detailed.py \
  --manifest experiments/SyntheticHybridExperiments/fit_manifest.csv
```

Regenerate posterior-predictive summaries from an existing predictive manifest:

```bash
pixi run python -u report_posterior_predictive.py \
  --manifest experiments/SyntheticHybridExperiments/posterior_predictive_manifest.csv
```
