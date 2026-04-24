# CLAUDE.md — MPLE-Causal-Inference

## Project Purpose

Implements MPLE (Majorized Proximal Likelihood Estimation) for causal inference on binary panel data with network effects and low-rank latent fields. Supports both synthetic/hybrid experiments (for parameter recovery studies) and a real-world US County COVID vaccination dataset.

## Pipeline Stages (in order)

```
1. run_generation_pipeline.py   — materialize synthetic experiment artifacts
2. run_fit_pipeline.py          — run MPLE optimizer on each experiment × variant
3. run_intervention_library.py  — (optional) pre-compute intervention panels
4. run_posterior_predictive_pipeline.py — Gibbs-sample outcome trajectories
5. report_parameter_recovery_detailed.py  — aggregate fit summaries
6. report_posterior_predictive.py         — aggregate predictive summaries
```

For real-world data the first stage is instead:
```
data/USCountyVaccination/load_raw_data.py
data/USCountyVaccination/preprocess_us_county_vaccination_data.py
data/USCountyVaccination/create_us_county_vaccination_experiments.py
```
Sensitivity sweeps use `run_uscounty_sensitivity_analysis.py`, which wraps stages 2 and 5.

## Config System

All YAML specs use a **base + named entries** pattern processed by `pipeline_specs.expand_named_entries()`:
```yaml
base:
  key: default_value
  ...
variants:          # key name varies: experiments / variants / interventions / runs
  - name: my_entry
    key: override_value
```
Each entry is deep-merged with `base`; entries inherit all base fields they don't override.

**Two separate config directories** — never mix them:
| Directory | Used by |
|---|---|
| `data/configs/` | Synthetic/hybrid pipeline (`run_generation_pipeline.py`, `run_fit_pipeline.py`, etc.) |
| `data/USCountyVaccination/experiment_configs/` | Real-data pipeline and `run_uscounty_sensitivity_analysis.py` |

## Optimizer Modes (`optimizer_mode` in fits_spec.yaml)

| Mode | Description |
|---|---|
| `exact_rank_manifold` | Riemannian conjugate gradient on the FixedRankEmbedded manifold (pymanopt). Use when rank is known. |
| `no_external_field` | Scalar-only optimization with the latent field fixed to zero. |
| `nuclear_norm` | Proximal gradient descent with nuclear-norm regularization. Promotes low rank without fixing it. |
| `alternating_latent_rank` | Alternating optimization over U and V factor matrices. Requires `latent_rank >= 1`. |
| `concurrent_latent_rank` | Joint L-BFGS-B optimization over U and V factor matrices with the same UV ridge penalty as the alternating mode. Requires `latent_rank >= 1`. |

## Key Source Files

| File | Role |
|---|---|
| `mple.py` | Core MPLE optimizer (all three modes). Also the subprocess entry point called by `run_fit_pipeline.py`. |
| `model_utils.py` | ModelArtifacts dataclass, parameter packing/unpacking, field matrix operations, named constants. |
| `loading_utils.py` | Experiment/panel artifact loading plus fit/truth parameter-bundle loading and persistence. |
| `intervention_utils.py` | Intervention-panel construction plus saved-intervention artifact loading and resolution. |
| `posterior_predictive_utils.py` | Gibbs-sampling outcome simulation and posterior-predictive summary statistics. |
| `pipeline_specs.py` | YAML spec loading, deep-merge, `expand_named_entries`, `validate_fits_spec`, manifest I/O. |
| `io_utils.py` | Shared I/O helpers: `load_yaml_config`, `first_existing_path`, `load_gamma_matrix`, path handling, CSV/Markdown writers. |
| `latent_recovery_diagnostics.py` | Diagnostic stats (RMSE, correlation, cosine alignment) between true and estimated fields. |

## Important Config Semantics

- `estimation.fixed_scalar_params` — scalars **fixed at these values**, not initial guesses. Empty dict `{}` means all scalars are estimated freely.
- `lambda_frobenius` — only active for `exact_rank_manifold` mode.
- `lambda_uv_ridge` — only active for `alternating_latent_rank` and `concurrent_latent_rank` modes.
- `optimizer_mode: no_external_field` → scalar-only model (no latent field).

## Manifest Flow

Each stage writes a CSV manifest that feeds into the next:
```
generation_manifest.csv  →  run_fit_pipeline.py  →  fit_manifest.csv
                                                  ↘
                             run_posterior_predictive_pipeline.py  →  posterior_predictive_manifest.csv
```
Pass these paths explicitly via `--manifest_path`, `--generation_manifest_path`, `--fit_manifest_path`.

## Running the Pipeline

```bash
# Quickstart (toy N=30, T=10 experiment)
pixi run quickstart-generate
pixi run quickstart-fit

# Full synthetic pipeline
pixi run generate
pixi run fit

# Tests
pixi run test

# Dry-run to preview planned work without executing
pixi run python -u run_fit_pipeline.py --manifest_path ... --fits_spec_path ... --dry_run
```

## Model Constants (defined in `model_utils.py`)

- `_RMS_SCALE_FACTOR = 0.4` — initial field RMS target = 0.4 × B
- `_DEGENERACY_THRESHOLD = 1e-12` — norms below this are treated as zero
- `_TAIL_STRENGTH = 0.5` — `tail_strength` for `sklearn.make_low_rank_matrix`
- `_RANDOM_INIT_SCALE = 0.05` (in `mple.py`) — std-dev for random initializations
