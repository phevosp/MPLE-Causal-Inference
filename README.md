# MPLE-Causal-Inference

Minimal conditional MPLE pipeline for binary outcome/intervention panel data.

## Active Runtime

The active repository is built around two workflows:

- synthetic conditional experiments
- USCountyVaccination materialization plus MPLE fitting

Both workflows use the same core artifact contract:

- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`
- `field_artifacts.npz`
- `gamma_matrix.npy` or `gamma_matrix_sparse.npz`

`field_artifacts.npz` is the single field bundle. It stores the active field mode and the mode-relevant field objects:

- additive runs store `field_basis`, `field_names`, and synthetic truth such as `field_coeffs`, `tau`, and `field_matrix`
- latent runs store `latent_rank`, `node_factors`, `time_factors`, and `field_matrix`

The interaction side is hardcoded:

- one fixed known graph `Gamma`
- one scalar temperature parameter `xi`

## Core Files

- [data/synthetic_data_generation.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/synthetic_data_generation.py)
- [mple.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/mple.py)
- [model_utils.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/model_utils.py)
- [data/configs/base_config.yaml](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/configs/base_config.yaml)
- [data/USCountyVaccination/run_us_county_vaccination_experiments.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/USCountyVaccination/run_us_county_vaccination_experiments.py)

## Field Modes

- `uniform`
- `shared_feature_field`
- `latent_feature_matrix`

## Commands

Generate one synthetic experiment:

```bash
pixi run python -u data/synthetic_data_generation.py --config_name base_config.yaml
```

Generate one synthetic fixed-`z` experiment:

```bash
pixi run python -u data/synthetic_data_generation.py \
  --config_name base_config.yaml \
  --config_override generation_params.intervention_mode=fixed_z \
  --config_override generation_params.fixed_z_source.panel_path=<panel_dir>/panel_data.npz \
  --config_override generation_params.fixed_z_source.z0_path=<panel_dir>/z_0.npy
```

Fit MPLE:

```bash
pixi run python -u mple.py --data_folder experiments/<folder>
```

Fit MPLE with shared panel artifacts:

```bash
pixi run python -u mple.py \
  --data_folder experiments/<folder> \
  --panel_path <panel_dir>/panel_data.npz \
  --x0_path <panel_dir>/x_0.npy \
  --z0_path <panel_dir>/z_0.npy
```

Materialize USCountyVaccination experiments:

```bash
pixi run python -u data/USCountyVaccination/run_us_county_vaccination_experiments.py --trim
```

Materialize and fit one USCounty latent-field experiment with explicit rank and `B`:

```bash
pixi run python -u data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --trim \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_20 \
  --lags 2w \
  --max_experiments 1 \
  --field_mode latent_feature_matrix \
  --latent_rank 6 \
  --latent_B 1.5 \
  --beta_mask_pre_intervention \
  --beta_mask_rescale \
  --run_mple
```

`B` is a global MPLE bound: scalar temperature parameters are clipped to `[-B, B]`, the interaction block satisfies `||xi*Gamma||_inf <= B`, and latent field runs also enforce `||field||_inf <= B`.

Run the minimal regression tests:

```bash
pixi run python tests/test_minimal_pipeline.py
```
