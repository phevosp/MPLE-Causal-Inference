# Manual Review of the Conditional MPLE Experiment Grid

## What I audited

- I reviewed all 216 experiments listed in `experiments/latest_manifest.txt`.
- I built a parseable one-row-per-experiment table at `reports/presentation_experiment_table.csv`.
- The table includes the main fit metrics plus block-level parameter error summaries:
  - `parameter_rmse`
  - `field_rmse`
  - `interaction_fro_error`
  - `beta_abs_error`, `eta_abs_error`, `zeta_abs_error`, `psi_abs_error`
  - average and maximum absolute errors for field and interaction coefficients

## Comparison caveat

`parameter_rmse` is most interpretable when the model dimension is held fixed. Comparing `uniform + known_graph` against `shared_feature_field + shared_feature_interactions` changes the number of coefficients being fit, so the cleanest apples-to-apples comparisons are:

- within a fixed model class, varying `N`, `T`, temperature regime, or graph family
- across model classes, using `field_rmse`, `interaction_fro_error`, and block-level absolute errors to see which parts of the model break first

## Executive summary

The clearest message from this grid is that time length `T` and interaction-model complexity dominate everything else. When the interaction structure is simple and known up to one coefficient, MPLE is extremely accurate once either `T` or `N` is reasonably large. When the interaction structure is expanded to 11 basis coefficients, the dominant failure mode is not the temporal or treatment parameters: it is the interaction block itself.

The second major message is that the Frobenius-norm experiment behaved as intended. The graph families produced structurally different `||Gamma||_F` values after `||Gamma||_inf = 1` normalization:

| graph family | regime | realized `||Gamma||_F` |
| --- | --- | --- |
| complete | `fro_small` | about `1.0001` to `1.0050` |
| Erdos-Renyi (`p=0.10`) | `fro_medium` | about `1.58` to `2.77` |
| cycle | `fro_large` | `7.0711`, `22.3607`, `50.0000` for `N=100,1000,5000` |

So the Frobenius variation came from graph structure, not rescaling.

## Main takeaways

### 1. `T` is the strongest single lever for recovery

Across the entire grid, moving from `T=10` to `T=100` improved:

- mean `parameter_rmse`: `0.7156` to `0.1444`
- mean `field_rmse`: `0.2010` to `0.0349`
- mean `interaction_fro_error`: `1.2273` to `0.3222`

This effect is even clearer in matched model classes:

| model class | `T=10` mean `parameter_rmse` | `T=100` mean `parameter_rmse` | `T=10` mean `interaction_fro_error` | `T=100` mean `interaction_fro_error` |
| --- | ---: | ---: | ---: | ---: |
| `uniform + known_graph` | `0.2230` | `0.0342` | `0.2312` | `0.1693` |
| `uniform + shared_feature_interactions` | `1.4643` | `0.3282` | `2.3447` | `0.5992` |
| `shared_feature_field + shared_feature_interactions` | `1.0771` | `0.1818` | `2.0894` | `0.3711` |

Interpretation: longer time series help even in the hard rich-interaction setting, and they are the difference between “often unstable” and “usually workable.”

### 2. In the simple baseline model, larger `N` helps monotonically

For the cleanest baseline subset, `uniform + known_graph`, the average `parameter_rmse` falls steadily with `N`:

| `N` | mean `parameter_rmse` | mean `field_rmse` | mean `interaction_fro_error` |
| ---: | ---: | ---: | ---: |
| 100 | `0.2526` | `0.3797` | `0.3109` |
| 1000 | `0.1128` | `0.1785` | `0.1832` |
| 5000 | `0.0203` | `0.0247` | `0.1067` |

This is the regime where the method looks strongest. The best overall experiment was:

- `N5000_T100_baseline_temp_fro_medium_uniform_known_graph`
- `parameter_rmse = 0.0011`
- `field_rmse = 0.0000`
- `interaction_fro_error = 0.0067`

So in the simplest “known interaction basis” setting, the estimator is essentially exact at the largest scale in this grid.

### 3. Low temperature is genuinely hard, and final loss can be misleading

In the simple baseline subset, temperature regime has a very strong effect:

| temperature regime | mean `parameter_rmse` | mean `field_rmse` | mean `interaction_fro_error` |
| --- | ---: | ---: | ---: |
| `baseline_temp` | `0.0208` | `0.0178` | `0.1193` |
| `high_temp` | `0.0275` | `0.0120` | `0.1657` |
| `low_temp` | `0.3375` | `0.5531` | `0.3158` |

The critical nuance is that lower final pseudo-likelihood loss does **not** mean better parameter recovery here. Averaged over all experiments:

| temperature regime | mean final loss | mean `parameter_rmse` |
| --- | ---: | ---: |
| `high_temp` | `0.6718` | `0.3056` |
| `baseline_temp` | `0.5519` | `0.3829` |
| `low_temp` | `0.2598` | `0.6014` |

So the low-temperature runs are often easier to fit in loss terms, but harder to identify correctly. That is exactly the kind of regime where fit quality and parameter recovery separate.

### 4. Frobenius norm matters, but not in a monotone “smaller is always easier” way

Again using the simple baseline subset:

| Frobenius regime | graph family | mean `parameter_rmse` | mean `field_rmse` | mean `interaction_fro_error` |
| --- | --- | ---: | ---: | ---: |
| `fro_small` | complete | `0.2221` | `0.3722` | `0.0481` |
| `fro_medium` | Erdős-Rényi | `0.0502` | `0.0307` | `0.1826` |
| `fro_large` | cycle | `0.1134` | `0.1800` | `0.3701` |

This is an important result:

- `fro_small` is **best** for interaction-matrix recovery
- but it is **worst** for field recovery and worst overall in `parameter_rmse`
- `fro_large` is hardest for interaction recovery
- `fro_medium` is the best compromise overall

My reading is that the dense complete graph makes the interaction matrix itself easy to estimate as a highly regular object, but it also makes the field/intercept side harder to disentangle. The cycle graph does the opposite: it preserves heterogeneity, but the large Frobenius norm makes interaction recovery much less stable. The moderate Erdős-Rényi case gives the best balance.

### 5. Richer external fields are manageable; richer interaction bases are the real bottleneck

The four model classes behave very differently:

| model class | mean `parameter_rmse` | mean `field_rmse` | mean `interaction_fro_error` | mean `beta_abs_error` | mean interaction-coefficient abs. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform + known_graph` | `0.1286` | `0.1943` | `0.2003` | `0.1968` | `0.0557` |
| `shared_feature_field + known_graph` | `0.0657` | `0.0594` | `0.1966` | `0.0280` | `0.0640` |
| `uniform + shared_feature_interactions` | `0.8963` | `0.1212` | `1.4719` | `0.1060` | `0.6078` |
| `shared_feature_field + shared_feature_interactions` | `0.6295` | `0.0971` | `1.2303` | `0.0214` | `0.5701` |

Two conclusions stand out:

- Moving from a uniform field to a shared-feature field does **not** destabilize the method. With a known interaction graph, recovery remains very strong.
- Moving from a single known interaction coefficient to a shared-feature interaction basis is the main source of difficulty. The average interaction-coefficient absolute error jumps by about an order of magnitude.

This is the sharpest empirical answer in the grid: interaction flexibility is much more expensive than field flexibility.

### 6. The hard regime is specifically the interaction block, not the temporal or treatment scalars

Average absolute parameter errors support that conclusion:

| subset | `beta` | `eta` | `zeta` | `psi` | field-block avg abs. error | interaction-block avg abs. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `known_graph` | `0.1124` | `0.0188` | `0.0144` | `0.0147` | `0.0662` | `0.0598` |
| `shared_feature_interactions` | `0.0637` | `0.0123` | `0.0150` | `0.0151` | `0.0797` | `0.5890` |

The worst overall experiment makes the same point:

- experiment: `N1000_T10_low_temp_fro_large_uniform_shared_feature_interactions`
- `parameter_rmse = 4.4135`
- `interaction_fro_error = 6.1382`
- mean interaction-coefficient abs. error = `2.6085`
- but `beta_abs_error = 0.0840`, `eta_abs_error = 0.0038`, `zeta_abs_error = 0.0063`, `psi_abs_error = 0.0243`

So the catastrophic failures are not broad collapse across every parameter. They are concentrated in the interaction basis coefficients.

## Representative best cases by model class

| model class | best experiment | `parameter_rmse` | `field_rmse` | `interaction_fro_error` |
| --- | --- | ---: | ---: | ---: |
| `uniform + known_graph` | `N5000_T100_baseline_temp_fro_medium_uniform_known_graph` | `0.0011` | `0.0000` | `0.0067` |
| `shared_feature_field + known_graph` | `N5000_T100_high_temp_fro_large_shared_feature_field_known_graph` | `0.0060` | `0.0049` | `0.2453` |
| `uniform + shared_feature_interactions` | `N5000_T100_high_temp_fro_medium_uniform_shared_feature_interactions` | `0.0905` | `0.0023` | `0.1298` |
| `shared_feature_field + shared_feature_interactions` | `N5000_T100_baseline_temp_fro_small_shared_feature_field_shared_feature_interactions` | `0.0572` | `0.0117` | `0.0898` |

This is encouraging: even the richer interaction setting can become usable once `N` and `T` are both large.

## Bottom line

If I were presenting this grid as a result, I would summarize it this way:

1. The estimator is highly reliable in the simple “known graph up to one coefficient” regime, especially once `T=100` and `N` is at least moderate.
2. Increasing `T` is the most reliable way to improve recovery, and increasing `N` helps strongly in the simple model.
3. Low-temperature regimes are the hardest identification setting, even when the optimization loss looks small.
4. The Frobenius norm matters structurally, but moderate Frobenius norm is best overall; the complete graph is not the easiest case once field recovery is included.
5. Allowing a richer external field is not the main problem. Allowing a richer interaction basis is.
6. When things fail, they fail in the interaction block first. The treatment and temporal coefficients remain comparatively stable.

If we were deciding what experiment grid to extend next, I would keep the simple known-graph baseline as the reference regime, and then spend additional runs on the `shared_feature_interactions` class with larger `T`, because that is where the real statistical difficulty is showing up.
