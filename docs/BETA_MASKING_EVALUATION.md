# Beta-Masking Evaluation: Documentation vs. Implementation

## Overview

This document provides a rigorous evaluation of how beta-masking works in the MPLE codebase, comparing documented behavior with actual implementation details.

## Documentation Claims (from README.md)

The README makes the following claims about beta-masking (lines 50-51):

1. **`beta_mask_pre_s`**: "if `true`, beta-gradient updates ignore observations with `t < s` while the forward model still uses the realized `z` everywhere. This masked-beta workflow is supported for `no_external_field` and `alternating_latent_rank`."

2. **`beta_mask_post_e`**: "if `true`, beta-gradient updates ignore observations with `t >= e` while the forward model still uses the realized `z` everywhere. This is also supported for `no_external_field` and `alternating_latent_rank`."

The README also mentions (lines 501-505) that CV validation ranking uses "post-`s` validation metrics" when beta masking is configured.

## Implementation Analysis

### 1. Mask Creation and Data Structure

**Location**: [mple.py:70-128](mple.py#L70-L128)

The mask is created in `_build_fit_eval_context()`:

```python
beta_update_mask = np.ones_like(x_array, dtype=bool)
if bool(beta_mask_pre_s) and s_index > 0:
    beta_update_mask[:s_index, :] = False  # Mask t < s
if bool(beta_mask_post_e) and e_index < t_steps:
    beta_update_mask[e_index:, :] = False  # Mask t >= e
```

The mask is stored in `_FitEvalContext` dataclass (line 57).

**Validation**: ✅ CORRECT
- `beta_mask_pre_s` correctly masks observations with `t < s`
- `beta_mask_post_e` correctly masks observations with `t >= e`
- Masks are applied before the forward model sees the data

### 2. Interaction with CV Loss Masks

**Location**: [mple.py:110](mple.py#L110)

When both beta masking and loss masks (from CV folds) are present:

```python
beta_update_mask = beta_update_mask & resolved_loss_mask
```

**Finding**: The beta mask is combined with any loss mask via bitwise AND. This means:
- Beta updates are restricted to entries that are BOTH in the beta mask window AND not held out for validation
- This is correct behavior for CV scenarios

### 3. Impact on Forward Model

**Location**: [mple.py:165-176](mple.py#L165-L176)

The forward model (`_compute_h_x`) computes:

```python
h_x = (
    np.asarray(field_matrix, dtype=float)
    + float(scalar_values["beta"]) * context.beta_feature  # No masking applied here
    + float(scalar_values["eta"]) * context.prev_x
    + float(scalar_values["xi"]) * context.interaction_effect_x
)
```

**Test Verification**: [tests/test_minimal_pipeline.py:1763-1798](tests/test_minimal_pipeline.py#L1763-L1798)

The test `test_beta_mask_pre_s_does_not_change_forward_model()` explicitly verifies:

```python
np.testing.assert_allclose(h_masked, h_unmasked)
```

**Validation**: ✅ CORRECT
- The forward model (h_x) is identical whether beta masking is enabled or disabled
- All observations use the full realized intervention panel

### 4. Impact on Loss Computation

**Location**: [mple.py:210-232](mple.py#L210-L232)

The loss computation in `_evaluate_full_field_loss()`:

```python
h_x = _compute_h_x(field_matrix, resolved_scalars, context)
loss_x = np.logaddexp(h_x, -h_x) - context.x * h_x
residual = np.tanh(h_x) - context.x
if context.loss_mask is not None:
    mask = context.loss_mask
    loss_x = loss_x * mask
    residual = residual * mask
smooth_loss = float(loss_x.sum() / context.outcome_size)
```

**Critical Observation**: The `beta_update_mask` is **NOT** applied to the loss computation. Only `loss_mask` (CV fold mask) is applied to the loss.

**Test Verification**: [tests/test_minimal_pipeline.py:1800-1842](tests/test_minimal_pipeline.py#L1800-L1842)

The test `test_beta_mask_pre_s_only_changes_beta_gradient()` verifies:

```python
self.assertAlmostEqual(masked_loss, unmasked_loss, places=12)
np.testing.assert_allclose(masked_residual, unmasked_residual)
```

**Validation**: ✅ CORRECT
- The smooth loss is identical whether beta masking is enabled or disabled
- The residual is identical whether beta masking is enabled or disabled

### 5. Impact on Gradient Computation

**Location**: [mple.py:179-207](mple.py#L179-L207)

The gradient computation in `_scalar_gradient_from_residual()`:

```python
if context.beta_outcome_size > 0.0:
    beta_gradient = (
        float(
            (
                residual
                * context.beta_feature
                * np.asarray(context.beta_update_mask, dtype=float)  # Masking APPLIED here
            ).sum()
        )
        / context.beta_outcome_size
    )
```

For other parameters (xi, eta), the mask is **NOT** applied:

```python
gradient_lookup = {
    "beta": beta_gradient,  # Masked
    "xi": float((residual * context.interaction_effect_x).sum()) / context.outcome_size,  # Unmasked
    "eta": float((residual * context.prev_x).sum()) / context.outcome_size,  # Unmasked
}
```

**Test Verification**: Line 1837 of the same test confirms:

```python
np.testing.assert_allclose(masked_scalar_grad[1:], unmasked_scalar_grad[1:])
```

**Validation**: ✅ CORRECT
- Beta gradient is masked according to the mask
- Xi and eta gradients are unmasked (computed on full panel)

### 6. Impact on Optimizer Step Size

**Location**: [mple.py:1340-1370](mple.py#L1340-L1370)

Step sizes are computed via Lipschitz constants. For beta:

```python
if name == "beta":
    feature = (
        context.beta_feature
        * np.asarray(context.beta_update_mask, dtype=float)
    ).reshape(-1)
    normalizer = float(context.beta_outcome_size)
```

For xi and eta:

```python
feature = (context.interaction_effect_x * active_loss_mask).reshape(-1)
normalizer = float(context.outcome_size)
```

**Finding**: Step sizes for beta are computed only over masked entries, while step sizes for other parameters use all entries. This means masking indirectly affects optimization dynamics.

### 7. Validation Metrics

**Location**: [utils/t7_validation_metrics.py:278-300](utils/t7_validation_metrics.py#L278-L300)

Critical comment in the code:

```python
# Beta masking is a fit-time optimization choice only. Reported losses are
# ordinary MPLE losses on the realized intervention panel, and predictive
# metrics/sampling use that same realized panel.
```

The validation loss computation uses:

```python
def _build_loss_kwargs(...) -> dict[str, Any]:
    return {
        "x": x,
        "z": z,  # REALIZED intervention panel (NOT masked)
        "x_0": x_0,
        "field_matrix": ...,
        "beta": float(bundle.beta),  # Estimated beta (affected by masking)
        "xi": float(bundle.xi),
        "eta": float(bundle.eta),
        "interaction_effect_x": interaction_effect_x,
        "fixed_scalar_params": {},
    }
```

**Finding**: 
- Validation loss uses the **realized** intervention panel (no masking)
- The beta parameter was estimated WITH masking, so it reflects masked optimization
- Other parameters were estimated without masking
- Validation metrics directly reflect the full MPLE loss on all observations

**Validation**: ✅ CORRECT per documented intent

### 8. "Post-S" Metrics Clarity

The README mentions "post-`s` validation metrics" in the context of beta masking (lines 501-505). However, examining the code:

**Location**: [utils/t7_validation_metrics.py:225-229](utils/t7_validation_metrics.py#L225-L229)

```python
post_s_mask = validation_mask & time_window_mask(
    t_steps=x.shape[0],
    n_nodes=x.shape[1],
    start_t=int(panel_context["s"]),  # Time window ONLY, no beta masking
)
```

**Finding**: The "post-`s`" metrics are computed by restricting to time steps `t >= s`, **NOT** by applying beta masking. This is a time-window filter, orthogonal to beta masking.

**Implication**: The README's discussion of "post-`s` validation metrics" is potentially misleading because it appears in a beta-masking section but is actually independent.

### 9. Optimizer Mode Restriction

**Location**: [mple.py:1981-1989](mple.py#L1981-L1989)

```python
if (
    (bool(beta_mask_pre_s) or bool(beta_mask_post_e))
    and artifacts.optimizer_mode
    not in {OPTIMIZER_MODE_ALTERNATING_LATENT_RANK, OPTIMIZER_MODE_NO_EXTERNAL_FIELD}
):
    raise ValueError(
        "beta-gradient-only masking is only supported for "
        "optimizer_mode in {'alternating_latent_rank', 'no_external_field'}; "
        "the other optimizer modes are deprecated for masked-beta workflows."
    )
```

**Validation**: ✅ CORRECT
- `alternating_latent_rank` and `no_external_field` support beta masking
- `nuclear_norm`, `concurrent_latent_rank`, and `exact_rank_manifold` reject the flag

### 10. Configuration Persistence

**Location**: [utils/t5_parameter_bundles.py:75-149](utils/t5_parameter_bundles.py#L75-L149)

Beta masking flags are:
- Saved in `fit_realized_config.yaml` during fitting
- Loaded back when reconstructing `OutcomeParameterBundle`
- Stored in the bundle but **not** used for validation metrics

**Finding**: The bundle preserves the masking configuration for reproducibility/documentation, but it doesn't influence how validation metrics are computed.

### 11. End-to-End Fitting Behavior

**Test Verification**: [tests/test_minimal_pipeline.py:2081-2137](tests/test_minimal_pipeline.py#L2081-L2137)

The test `test_alternating_low_rank_supports_beta_gradient_masking()` shows:

```python
self.assertGreater(beta_masked, beta_unmasked)  # Line 2137
```

**Finding**: Beta masking results in different estimated beta values (higher when masking t < s). This is expected because masking upweights the post-s observations in the gradient.

## Summary of Findings

### ✅ Documentation-Code Alignment

1. **Beta masking masks observations in gradient computation**: Correct
2. **Only forward model is unaffected**: Correct
3. **Only `alternating_latent_rank` supports masking**: Outdated; `no_external_field` now supports masking too
4. **Both pre_s and post_e work as documented**: Correct

### ⚠️ Gaps and Potential Confusion

1. **README's "post-`s` validation metrics" context**: While technically correct, grouping post-`s` metrics with beta masking discussion is misleading. Post-`s` metrics are time-window restrictions, not beta-masking effects.

2. **CV ranking uses post-`s` metrics**: True, but this is **independent** of beta masking. The CV ranking favors post-`s` metrics because beta masking targets the pre-s period, so validation after the intervention is more relevant. However, this causality is not made explicit.

3. **Scalar step size implications**: Beta masking changes the Lipschitz constant computation for beta, indirectly affecting optimization dynamics. This is correct behavior but is not explicitly documented.

4. **Validation loss interpretation**: The estimated beta is affected by masking, but validation loss uses the realized panel. This means:
   - A fit with `beta_mask_pre_s=True` will have a different beta
   - But reported validation loss includes all observations
   - This can lead to scenarios where masked and unmasked fits have similar validation loss but different beta values

## Recommendations

1. **Clarify CV ranking rules**: Explicitly document that post-`s` metrics are used when beta masking is enabled, because masking focuses optimization on the post-s period. Consider renaming this as "post-intervention metrics" rather than "post-`s`" to reduce confusion.

2. **Document the validation-estimation asymmetry**: Add explicit documentation that:
   - Beta is estimated from the masked set (e.g., t >= s)
   - Validation loss uses the full realized panel with the estimated beta
   - This is intentional: masking changes which observations drive beta estimation, but evaluation is always on the full panel

3. **Consider adding a validation helper**: A function that explicitly shows what happens when you evaluate a masked fit on the full vs. masked panel could clarify the intended behavior.

4. **Update CV spec examples**: When showing `beta_mask_pre_s` in grid searches, add a comment explaining that post-`s` metrics will be prioritized in ranking.

## Code Quality Assessment

- **Correctness**: Excellent. Implementation matches the stated intent.
- **Test Coverage**: Excellent. Unit tests verify key invariants (forward model unchanged, loss unchanged, only beta gradient changes).
- **Documentation**: Good overall, with minor gaps in the README around post-`s` metric interpretation.
