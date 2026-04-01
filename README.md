# MPLE-Causal-Inference

Synthetic experiments for maximum pseudo-likelihood estimation in networked causal Ising-style models.

## What This Repository Does

The repo studies binary outcomes `x_t in {-1,+1}^N` and binary interventions `z_t in {-1,+1}^N` evolving on a network over time. It contains:

- synthetic data generation for two data-generating processes (conditional; Ising-style joint)
- pseudo-likelihood estimators for both the conditional model and the joint Ising-style model
- experiment folders that save realized configs, generated data, learned summaries, and logs

## Model Variants

### Conditional Generator

The conditional process samples

- `z^(t)` from a logistic model depending on `x^(t-1)` and `z^(t-1)`
- `x^(t)` from a Gibbs sampler with node-wise field
  `h_x^(t) = h + beta z^(t) + eta x^(t-1) + J x^(t)`

This is implemented in `data/synthetic_data_generation.py` through:

- `sample_z_t`
- `sample_x_t`
- `generate_conditional_model`

### Joint Ising Generator

The joint model performs Gibbs sweeps over the full space-time configuration, using both past and future neighbors in the pseudo-conditional updates. This is implemented in:

- `generate_ising_model`

## New Low-Dimensional Basis Parameterization

The shared basis logic lives in `model_utils.py`.

Default field basis:

- `intercept`
- `linear`
- `quadratic`

Default interaction basis:

- `adjacency`
- `distance_kernel`
- `cross_similarity`

These templates are orthonormalized before use so the learned coefficients are less confounded. The generator saves the realized basis and the effective objects:

- `field_basis.npy`
- `interaction_basis.npy`
- `field_vector.npy`
- `interaction_matrix.npy`

Legacy experiment folders without these files still work: the estimator falls back to the old `alpha * 1` and `xi * Gamma` parameterization.

## Estimation Code

`mple.py` contains three key pieces:

- `conditional_model_pseudo_nll`: conditional-model negative pseudo-log-likelihood and gradient
- `pseudo_nll`: joint Ising pseudo-log-likelihood and gradient, with both unconditioned and conditioned variants
- `mple_gradient_descent`: L-BFGS-B wrapper used by both estimators

The optimization variables are:

`[field coefficients, beta, interaction coefficients, eta, zeta, psi]`

For the two-stage Ising estimator, the final combined estimate keeps the stage-1 parameters except for the field coefficients, which are replaced by the conditioned stage-2 field estimate.

## Repository Map

- `data/synthetic_data_generation.py`: graph realization, basis realization, and synthetic data generation
- `data/configs/base_config.yaml`: default experiment configuration
- `model_utils.py`: basis construction, parameter packing/unpacking, and summary metrics
- `mple.py`: pseudo-likelihood objectives, optimizer, logging, and summary-table export
- `generate_data.sh`: helper script for generating multiple datasets
- `experiments/`: realized experiment folders with configs, arrays, logs, and summary tables

## How To Run

Generate a conditional dataset:

```bash
pixi run python -u data/synthetic_data_generation.py --config_name base_config.yaml
```

Fit the conditional pseudo-likelihood on one experiment folder:

```bash
pixi run python -u mple.py --data_folder experiments/<folder> --use_conditional_npll
```

Fit the joint Ising pseudo-likelihood:

```bash
pixi run python -u mple.py --data_folder experiments/<folder>
```

Each fit writes logs plus machine-readable and markdown summaries such as:

- `mple_conditional_summary.csv`
- `mple_conditional_summary.md`
- `mple_stage1_summary.csv`
- `mple_stage2_summary.csv`
- `mple_combined_summary.csv`

## Latest Conditional Demo

Checked run:

- folder: `experiments/synthetic_data_20260329_122136`
- process: conditional
- `N=200`, `T=80`, `s=8`, Erdos-Renyi `p=0.05`, seed `19`
- true field coefficients: `[0.35, -0.20, 0.08]`
- true interaction coefficients: `[0.14, 0.08, -0.06]`

Parameter summary from `mple_conditional_summary.md`:

| Parameter | True | Estimate | Squared Error |
| --- | ---: | ---: | ---: |
| `field::intercept` | 0.3500 | 0.3946 | 0.0020 |
| `field::linear` | -0.2000 | -0.2074 | 0.0001 |
| `field::quadratic` | 0.0800 | 0.1120 | 0.0010 |
| `beta` | 0.2500 | 0.2449 | 0.0000 |
| `interaction::adjacency` | 0.1400 | 0.0923 | 0.0023 |
| `interaction::distance_kernel` | 0.0800 | 0.0116 | 0.0047 |
| `interaction::cross_similarity` | -0.0600 | -0.1167 | 0.0032 |
| `eta` | 0.0100 | 0.0072 | 0.0000 |
| `zeta` | -0.0300 | -0.0207 | 0.0001 |
| `psi` | 0.1000 | 0.1045 | 0.0000 |

Aggregate metrics:

| Metric | Value |
| --- | ---: |
| final conditional pseudo-NLL | 0.6753 |
| field RMSE | 0.0039 |
| interaction Frobenius error | 0.1009 |
| parameter RMSE | 0.0366 |

The main takeaway from this run is that the conditional estimator recovers the treatment and temporal coefficients very closely, reconstructs the effective field accurately, and captures the interaction structure reasonably well even when the interaction matrix is no longer assumed known up to a single scalar.
