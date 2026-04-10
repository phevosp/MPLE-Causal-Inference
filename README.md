# MPLE-Causal-Inference

Minimal conditional MPLE pipeline for binary outcome/intervention panel data.

## Active Scope

The active repository is centered on three workflows:

- synthetic conditional data generation
- conditional-model MPLE fitting
- USCountyVaccination preprocessing and experiment materialization

The active conditional model evolves:

- outcomes `x_t in {-1,+1}^N`
- interventions `z_t in {-1,+1}^N`

with

- intervention process: `z^(t)` depends on `x^(t-1)` and `z^(t-1)`
- outcome process: `x^(t)` depends on a node field, `z^(t)`, `x^(t-1)`, and a fixed known graph

## Active Model Interface

The active pipeline keeps one interaction template only:

- a fixed known graph `Gamma`
- one scalar interaction coefficient in `estimation_params.interaction_coefs`

The field side supports:

- `field_mode=uniform`
- `field_mode=shared_feature_field`

There is no active support for richer synthetic interaction-template families in the core pipeline.

## Core Files

- [data/synthetic_data_generation.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/synthetic_data_generation.py): synthetic conditional generator
- [mple.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/mple.py): MPLE objective, optimizer, logging, and saved summaries
- [model_utils.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/model_utils.py): minimal basis, parameter, and summary helpers
- [data/configs/base_config.yaml](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/configs/base_config.yaml): default synthetic config
- [data/USCountyVaccination/run_us_county_vaccination_experiments.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/USCountyVaccination/run_us_county_vaccination_experiments.py): USCountyVaccination panel materialization and fitting entrypoint
- [data/USCountyVaccination/create_data_analysis_summary.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/USCountyVaccination/create_data_analysis_summary.py): descriptive county-vaccination reporting bundle

## Saved Artifacts

Synthetic and real-data runs use the same core panel-state contract:

- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`

MPLE also reads experiment-local basis/network artifacts such as:

- `field_basis.npy`
- `interaction_basis.npy` or `interaction_basis_sparse.npz`
- `gamma_matrix_sparse.npz` for real-data known graphs

`mple.py` can load the panel-state files either from `--data_folder` or from explicit `--panel_path`, `--x0_path`, and `--z0_path` arguments.

For USCountyVaccination experiments, those panel-state files are now shared across network variants and written once per `(outcome, intervention, lag, trim scope)` under a shared panel directory.

## How To Run

Generate one synthetic dataset:

```bash
pixi run python -u data/synthetic_data_generation.py --config_name base_config.yaml
```

Generate one synthetic dataset with fixed interventions loaded from a shared USCountyVaccination panel:

```bash
pixi run python -u data/synthetic_data_generation.py \
  --config_name base_config.yaml \
  --config_override generation_params.intervention_mode='\"fixed_z\"' \
  --config_override generation_params.fixed_z_source.panel_path='\"<shared_panel_dir>/panel_data.npz\"' \
  --config_override generation_params.fixed_z_source.z0_path='\"<shared_panel_dir>/z_0.npy\"'
```

Fit MPLE on one experiment folder:

```bash
pixi run python -u mple.py --data_folder experiments/<folder>
```

Fit MPLE with explicit panel-state artifacts:

```bash
pixi run python -u mple.py \
  --data_folder experiments/<folder> \
  --panel_path <shared_panel_dir>/panel_data.npz \
  --x0_path <shared_panel_dir>/x_0.npy \
  --z0_path <shared_panel_dir>/z_0.npy
```

Build shared USCountyVaccination experiment artifacts:

```bash
pixi run python -u data/USCountyVaccination/run_us_county_vaccination_experiments.py --trim
```

## Parameterization

The active optimizer vector is:

`[field coefficients, tau coefficients, beta, interaction coefficient, eta, zeta, psi]`

When `--outcome_only` is used, the intervention-process parameters `zeta` and `psi` are omitted from the fit while MPLE still conditions on observed `z`.
