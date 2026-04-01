# MPLE-Causal-Inference

Synthetic experiments for maximum pseudo-likelihood estimation in a conditional networked causal model with binary outcomes and binary interventions.

## What This Repository Does

The repo studies trajectories of

- outcomes `x_t in {-1,+1}^N`
- interventions `z_t in {-1,+1}^N`

over a network and through time. It now supports a single model family throughout the codebase:

- conditional data generation
- conditional-model MPLE estimation

The code no longer includes the older joint Ising generator or the two-stage Ising MPLE routine.

## Model

At each time step:

- `z^(t)` is sampled from a logistic model depending on `x^(t-1)` and `z^(t-1)`
- `x^(t)` is sampled from a Gibbs kernel with local field
  `h_x^(t) = h + beta z^(t) + eta x^(t-1) + J x^(t)`

The external field `h` and interaction matrix `J` are not restricted to the old scalar form `alpha * 1` and `xi * Gamma`. Instead, they are built from low-dimensional bases:

- `h = sum_k a_k b_k`
- `J = sum_l w_l G_l`

where `b_k` are known field templates, `G_l` are known symmetric interaction templates, and the coefficient vectors `a` and `w` are estimated.

## Basis Construction

The shared basis logic lives in [model_utils.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/model_utils.py).

Default field templates:

- `intercept`
- `linear`
- `quadratic`

Default interaction templates:

- `adjacency`
- `distance_kernel`
- `cross_similarity`

Both field templates and interaction templates are normalized by infinity norm before orthonormalization. The generator saves:

- `field_basis.npy`
- `interaction_basis.npy`
- `field_vector.npy`
- `interaction_matrix.npy`

Legacy experiment folders without saved basis artifacts still load correctly through the scalar fallback in `load_or_build_basis(...)`.

## Main Files

- [data/synthetic_data_generation.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/synthetic_data_generation.py): graph realization and conditional synthetic data generation
- [data/configs/base_config.yaml](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/data/configs/base_config.yaml): default configuration
- [model_utils.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/model_utils.py): basis construction, parameter packing, and summary metrics
- [mple.py](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/mple.py): conditional pseudo-NLL, optimizer, logging, and summary export
- [generate_data.sh](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/generate_data.sh): helper script for generating several conditional datasets
- [run_experiments.sh](c:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference/run_experiments.sh): helper script for generating datasets and fitting MPLE across experiment folders

## How To Run

Generate one dataset:

```bash
pixi run python -u data/synthetic_data_generation.py --config_name base_config.yaml
```

Fit MPLE on one experiment folder:

```bash
pixi run python -u mple.py --data_folder experiments/<folder>
```

The estimator writes:

- `mple.log`
- `mple_summary.csv`
- `mple_summary.md`

## Current Optimizer Parameterization

The flattened optimization vector is

`[field coefficients, beta, interaction coefficients, eta, zeta, psi]`

The summary tables report:

- coefficient-wise estimate vs. truth
- field RMSE
- interaction Frobenius error
- overall parameter RMSE
