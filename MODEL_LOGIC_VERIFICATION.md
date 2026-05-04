# Model Logic Consistency Verification

## Mathematical Model

The model implements:
```
p(x^{(t)} | z^{(t)}, x^{(t-1)}) ∝ exp(h_x)

where for unit i:
h_i^{(t)} = α_i^{(t)} + β·z_i^{(t)} + ξ·(γ @ x^{(t)})_i + η·x_i^{(t-1)}
```

Components:
- **α_i^{(t)}**: Latent field (field_matrix[t, i])
- **β·z_i^{(t)}**: Treatment effect (beta * beta_feature_masked[t, i])
- **ξ·(γ @ x)_i**: Network interaction (xi * interaction_effect_x[t, i])
- **η·x_i^{(t-1)}**: Temporal autocorrelation (eta * prev_x[t, i])

---

## Code Verification

### 1. h_x Computation in mple.py (_compute_h_x)

**File**: `mple.py:161-171`

```python
def _compute_h_x(field_matrix, scalar_values, context):
    return (
        np.asarray(field_matrix, dtype=float)                           # α_i^{(t)}
        + float(scalar_values["beta"]) * context.beta_feature_masked    # β·z_i^{(t)}
        + float(scalar_values["eta"]) * context.prev_x                  # η·x_i^{(t-1)}
        + float(scalar_values["xi"]) * context.interaction_effect_x     # ξ·(γ @ x)_i
    )
```

**Components created in _build_fit_eval_context:**
- `beta_feature_masked`: Shape [T, N], masked version of z with zeros before s and after e
- `prev_x`: Shape [T, N], x_0 at t=0, then x[:-1] for t>0
- `interaction_effect_x`: Shape [T, N], computed as `x @ gamma_matrix.T`

✅ **CORRECT**: Matches model specification

---

### 2. Loss Computation

**File**: `mple.py:206`

```python
loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
```

This is the negative log-likelihood for the Ising model with x ∈ {-1, 1}:
- Log-partition: `log(exp(h) + exp(-h)) = log(2·cosh(h))`
- NLL: `log(2·cosh(h)) - h·x`

✅ **CORRECT**: Standard Ising model NLL

---

### 3. Posterior Predictive Sampling

**File**: `data/synthetic_data_generation.py:273-306` (sample_x_t_with_parameters)

```python
interaction_x_t = interaction_matrix @ x_t  # (xi * gamma) @ x_t

for each Gibbs sweep:
    for node i:
        h_x = (
            field_t[i]                      # α_i^{(t)}
            + float(beta) * beta_feature[i] # β·z_i^{(t)}
            + float(eta) * x_prev[i]        # η·x_i^{(t-1)}
            + interaction_x_t[i]            # ξ·(γ @ x)_i
        )
        x_t[i] = spin_sample_from_field(h_x, rng)
```

**composition_interaction_matrix** (model_utils.py:701-712):
```python
def compose_interaction_matrix(xi, gamma_matrix):
    interaction_matrix = xi * gamma_matrix  # Pre-multiply by ξ
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0  # Symmetrize
    np.fill_diagonal(interaction_matrix, 0.0)  # Zero diagonal
    return interaction_matrix
```

So: `interaction_x_t[i] = (ξ·γ @ x_t)[i] = ξ·(γ @ x_t)[i]` ✅

✅ **CORRECT**: Sampling uses the same generative h_x components as predictive metrics

---

### 4. Gamma Matrix Properties

**Files**: `model_utils.py:171-195`, `synthetic_data_generation.py:427-428`

Gamma matrix transformations:
1. **Symmetrization**: `gamma = (gamma + gamma.T) / 2`
2. **Zero diagonal**: `diag(gamma) = 0`
3. **Normalization**: Infinity norm = 1.0
4. **Consistent use**: Pre-composed as `ξ·γ` before sampling/evaluation

✅ **CORRECT**: Symmetric network structure maintained throughout

---

### 5. Beta Masking

**Files**: `mple.py:96-100`

Beta masking is a fit-only choice used inside the MPLE objective:
- Before time s: the fit loss can set `z_i^{(t)} = 0` for t < s
- After time e: the fit loss can set `z_i^{(t)} = 0` for t ≥ e

Posterior predictive sampling and predictive metrics should not use those masks;
they should always use the realized intervention panel `z`.

✅ **CORRECT**: Masking is restricted to fit-loss evaluation rather than changing the generative model

---

### 6. Temporal Autocorrelation

**Files**: `mple.py:114`, `synthetic_data_generation.py:284`

Previous x is constructed as:
```python
prev_x = [x_0, x[:-1, :]]  # x_0 at t=1, then lag-1 of x
```

So: `prev_x[t, i] = x_0[i]` if t=0, else `x[t-1, i]`

✅ **CORRECT**: Proper lag-1 structure for η·x_i^{(t-1)}

---

### 7. Validation Metrics

**Files**: `validation_metric_utils.py`, `run_test_evaluation.py`

Predictive metrics use the same unmasked h_x computation:
- Brier score: `(predicted_prob - observed)²` where predicted = sigmoid(h)
- ECE: Calibration error based on sigmoid(h) predictions
- Magnetization sampling: posterior predictive draws conditional on observed non-held-out entries

Loss metrics are slightly different by design:
- Loss: `logaddexp(h, -h) - x·h`
- When beta masking is enabled, the fit-loss path applies it so reported fit/validation loss mirrors the training objective

✅ **CORRECT**: Predictive metrics use the generative h_x, while loss metrics mirror the fit objective

---

## Summary

| Component | File | Check | Status |
|-----------|------|-------|--------|
| h_x formula | mple.py | α + β·z + ξ·(γ@x) + η·x_prev | ✅ |
| Loss computation | mple.py | NLL for Ising model | ✅ |
| Sampling | synthetic_data_generation.py | Same h_x formula | ✅ |
| Gamma symmetry | model_utils.py | (γ + γ^T)/2, diag=0 | ✅ |
| Beta masking | mple.py | Fit-loss only before s / after e | ✅ |
| prev_x construction | mple.py & synthetic_data_generation.py | x_0 then lag-1 | ✅ |
| Metrics | validation_metric_utils.py | Use h_x consistently | ✅ |

## Conclusion

✅ **The model logic is consistent across the entire codebase and matches the mathematical specification.**

The predictive h_x computation is identical in:
- Posterior predictive sampling (`synthetic_data_generation.py`)
- Predictive validation metrics (`validation_metric_utils.py`)

The fit-objective h_x computation is identical in:
- Loss evaluation (`mple.py`)
- Gradient computation (`mple.py`)

All parameters (α, β, ξ, η) are applied consistently, and fit-only masking is isolated to the MPLE objective.
