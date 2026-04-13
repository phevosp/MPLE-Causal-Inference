# MPLE-Causal-Inference

Minimal conditional MPLE pipeline for binary outcome/intervention panel data.

## Active Runtime

The active repository is built around two workflows:

- synthetic and hybrid generation experiments
- USCountyVaccination materialization plus MPLE fitting

Both workflows use the same core artifact contract:

- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`
- `field_artifacts.npz`
- `gamma_matrix.npy` or `gamma_matrix_sparse.npz`

`field_artifacts.npz` stores only:

- `latent_rank`
- `t_steps`
- `node_factors`
- `time_factors`
- `field_matrix`

`latent_rank=0` means there is no external field. In that case the latent factors are empty and `field_matrix` is identically zero.

The interaction side is hardcoded:

- one fixed known graph `Gamma`
- one scalar temperature parameter `xi`

## Core Files

- [data/synthetic_data_generation.py](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/synthetic_data_generation.py>)
- [run_generation_pipeline.py](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/run_generation_pipeline.py>)
- [run_fit_pipeline.py](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/run_fit_pipeline.py>)
- [mple.py](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/mple.py>)
- [model_utils.py](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/model_utils.py>)
- [data/configs/generation_spec.yaml](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/configs/generation_spec.yaml>)
- [data/configs/fits_spec.yaml](</c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/configs/fits_spec.yaml>)

## Generation And Fits

The synthetic/hybrid pipeline is split into two stages:

1. `run_generation_pipeline.py` expands `generation_spec.yaml` into generated experiment folders and a generation manifest.
2. `run_fit_pipeline.py` expands `fits_spec.yaml` into MPLE variants over that manifest.

The generation spec is split into `base` plus `experiments`, and the fit spec is split into `base` plus `variants`. Generation artifacts live at the experiment root, while each MPLE variant writes to `fits/<variant_slug>/`.

All generation defaults live in `generation_spec.yaml`, and all fit defaults live in `fits_spec.yaml`. `data/synthetic_data_generation.py` is an internal helper used by `run_generation_pipeline.py`.

## Commands

Materialize the spec-driven synthetic/hybrid generation manifest:

```bash
pixi run python -u run_generation_pipeline.py --spec_path data/configs/generation_spec.yaml
```

Run MPLE variants over the generated experiment manifest:

```bash
pixi run python -u run_fit_pipeline.py \
  --manifest_path experiments/SyntheticHybridExperiments/generation_manifest.csv \
  --fits_spec_path data/configs/fits_spec.yaml
```

Shell wrappers:

```bash
bash generate_data.sh
bash run_experiments.sh
```

Fit one experiment manually with an explicit fit config:

```bash
pixi run python -u mple.py \
  --data_folder experiments/<folder>/fits/<variant> \
  --config_path experiments/<folder>/fits/<variant>/fit_realized_config.yaml \
  --model_artifact_dir experiments/<folder> \
  --truth_artifact_dir experiments/<folder> \
  --panel_path experiments/<folder>/panel_data.npz \
  --x0_path experiments/<folder>/x_0.npy \
  --z0_path experiments/<folder>/z_0.npy
```

In `fits_spec.yaml`, `B` is a fit-time parameter. It controls:

- scalar clipping to `[-B, B]`
- the interaction constraint `||xi * Gamma||_inf <= B`
- the latent-field constraint `||H||_inf <= B` with `H = U @ V.T`

Run the minimal regression tests:

```bash
pixi run python tests/test_minimal_pipeline.py
```
