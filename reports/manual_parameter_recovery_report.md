# Parameter Recovery Review of the Conditional MPLE Grid

## Scope

This report focuses on **parameter recovery**, not optimization loss. The main metric throughout is `parameter_rmse`, supported by:

- average absolute error in field coefficients
- average absolute error in interaction coefficients
- absolute error in `beta`, `eta`, `zeta`, and `psi`

The main analysis is restricted to the **core known-interaction setting**:

- `interaction_complexity = known_graph`

This keeps the comparison centered on the levers you said matter most:

- `N`
- `T`
- temperature regime
- graph Frobenius norm
- whether the field is `uniform` or `shared_feature_field`

I only mention `shared_feature_interactions` briefly at the end, since those runs are useful context but not the main object of interest here.

Supporting tables:

- experiment-level table: `reports/presentation_experiment_table.csv`
- factor summary table for the core known-graph setting: `reports/parameter_recovery_factor_summary.csv`

## High-level takeaways

1. In the core known-graph setting, recovery is generally very strong once `T=100` or `N` is large.
2. `T` is the cleanest and strongest lever: going from `T=10` to `T=100` reduces mean `parameter_rmse` from `0.1605` to `0.0338`.
3. Larger `N` helps monotonically in the core setting: mean `parameter_rmse` falls from `0.1822` at `N=100` to `0.0261` at `N=5000`.
4. Low temperature is the main failure regime. It hurts field recovery and `beta` recovery much more than the other temporal parameters.
5. Adding `shared_feature_field` does **not** make recovery worse. In this grid it actually improves recovery substantially, likely by removing field misspecification.
6. The data support the weaker claim that **very small Frobenius norm is bad for recovery**, but they do **not** support a fully monotone claim that “higher Frobenius norm is always better.”

## Effect of `N`

Holding the interaction model fixed at `known_graph`, larger `N` consistently improves parameter recovery:

| `N` | mean `parameter_rmse` | mean field-coefficient abs. error | mean interaction-coefficient abs. error | mean `beta` abs. error |
| ---: | ---: | ---: | ---: | ---: |
| 100 | `0.1822` | `0.2378` | `0.1128` | `0.2203` |
| 1000 | `0.0830` | `0.1109` | `0.0366` | `0.0996` |
| 5000 | `0.0261` | `0.0245` | `0.0302` | `0.0174` |

This pattern is not limited to one field specification:

- `uniform + known_graph`: `0.2526 -> 0.1128 -> 0.0203`
- `shared_feature_field + known_graph`: `0.1118 -> 0.0533 -> 0.0318`

Interpretation:

- The gains from larger `N` are broad-based, not confined to one block of parameters.
- The biggest improvements are in the field block and in `beta`.
- By `N=5000`, the estimator is in a very accurate regime for the core model class.

## Effect of `T`

`T` is the single strongest lever in the core known-graph experiments:

| `T` | mean `parameter_rmse` | mean field-coefficient abs. error | mean interaction-coefficient abs. error | mean `beta` abs. error |
| ---: | ---: | ---: | ---: | ---: |
| 10 | `0.1605` | `0.2139` | `0.0810` | `0.2013` |
| 100 | `0.0338` | `0.0349` | `0.0387` | `0.0235` |

The same effect shows up in both field regimes:

- `uniform + known_graph`: `0.2230` at `T=10` versus `0.0342` at `T=100`
- `shared_feature_field + known_graph`: `0.0980` at `T=10` versus `0.0333` at `T=100`

Interpretation:

- More time points help every part of the problem.
- The largest absolute gains are again in the field block and in `beta`.
- If I had to choose one experimental design change to improve recovery, I would increase `T` before increasing anything else.

## Effect of temperature regime

Temperature matters a lot, and the main difficulty is the low-temperature regime:

| temperature regime | mean `parameter_rmse` | mean field-coefficient abs. error | mean interaction-coefficient abs. error | mean `beta` abs. error |
| --- | ---: | ---: | ---: | ---: |
| `baseline_temp` | `0.0327` | `0.0285` | `0.0330` | `0.0157` |
| `high_temp` | `0.0394` | `0.0245` | `0.0804` | `0.0110` |
| `low_temp` | `0.2192` | `0.3202` | `0.0661` | `0.3106` |

Two points are especially important:

1. Low temperature does **not** mainly break the temporal parameters.
   Their mean absolute errors remain fairly small:
   - `eta`: `0.0381`
   - `zeta`: `0.0184`
   - `psi`: `0.0193`

2. Low temperature mainly breaks:
   - the field block
   - `beta`

This is even sharper in the simplest baseline:

- `uniform + known_graph`, `baseline_temp`: `parameter_rmse = 0.0208`
- `uniform + known_graph`, `high_temp`: `0.0275`
- `uniform + known_graph`, `low_temp`: `0.3375`

Interpretation:

- The low-temperature issue is fundamentally an identification problem, not just an optimization problem.
- In this regime, the method can fit the data well while still missing the true parameters.
- That is why I am not using loss as the organizing metric in this report.

## Effect of graph Frobenius norm

This lever needs the most careful interpretation.

### What the data say

Across the core known-graph experiments:

| Frobenius regime | graph family | mean `parameter_rmse` | mean field-coefficient abs. error | mean interaction-coefficient abs. error | mean `beta` abs. error |
| --- | --- | ---: | ---: | ---: | ---: |
| `fro_small` | complete | `0.1477` | `0.2165` | `0.0686` | `0.1964` |
| `fro_medium` | Erdős-Rényi | `0.0562` | `0.0398` | `0.0851` | `0.0315` |
| `fro_large` | cycle | `0.0874` | `0.1169` | `0.0259` | `0.1094` |

So the pattern is:

- `fro_small` is clearly the **worst** for overall parameter recovery
- `fro_medium` is the **best** overall
- `fro_large` is usually better than `fro_small`, but not uniformly better than `fro_medium`

### Does higher Frobenius norm help?

Your hypothesis is **partly justified**, but only in a qualified sense.

What the data do support:

- Moving away from the very small-Frobenius complete-graph regime generally helps parameter recovery.
- The correlation between realized `gamma_fro_norm` and `parameter_rmse` is mildly negative:
  - `uniform + known_graph`: about `-0.14`
  - `shared_feature_field + known_graph`: about `-0.25`

What the data do **not** support:

- A strictly monotone story that “larger Frobenius norm always means better overall recovery.”

The best mean `parameter_rmse` is at `fro_medium`, not `fro_large`.

### Why this is not contradictory

The strongest evidence for your intuition appears when I look at the **interaction coefficient itself**, rather than the full parameter vector or the matrix Frobenius error.

For the core known-graph setting:

- `fro_large` has the **smallest** mean interaction-coefficient absolute error: `0.0259`
- `fro_small`: `0.0686`
- `fro_medium`: `0.0851`

And in `shared_feature_field + known_graph`, the effect is even cleaner:

- `fro_small`: `0.0892`
- `fro_medium`: `0.0815`
- `fro_large`: `0.0214`

So if the question is:

- “Does a larger graph Frobenius norm help recover the interaction **coefficient**?”

then the answer is often **yes**.

But if the question is:

- “Does a larger graph Frobenius norm give the best **overall** parameter recovery?”

then the answer is **not always**. The medium-Frobenius regime gives the best balance between:

- field recovery
- `beta` recovery
- interaction recovery

One more nuance matters here: once `||Gamma||_F` changes across graph families, the absolute **interaction-matrix Frobenius error** becomes harder to interpret as a pure recovery metric, because the basis itself is changing scale in Frobenius norm. That is why I am emphasizing coefficient recovery and overall parameter RMSE in this section, rather than matrix Frobenius error alone.

## Effect of `shared_feature_field`

The data are surprisingly favorable to the richer field specification.

Within the core known-graph experiments:

| field complexity | mean `parameter_rmse` | mean field-coefficient abs. error | mean interaction-coefficient abs. error | mean `beta` abs. error |
| --- | ---: | ---: | ---: | ---: |
| `uniform` | `0.1286` | `0.1943` | `0.0557` | `0.1968` |
| `shared_feature_field` | `0.0657` | `0.0545` | `0.0640` | `0.0280` |

Matched cell-by-cell, moving from `uniform` to `shared_feature_field` while holding `N`, `T`, temperature, and Frobenius regime fixed changes the averages by:

- `parameter_rmse`: `-0.0629`
- mean field-coefficient abs. error: `-0.1398`
- mean `beta` abs. error: `-0.1688`
- mean interaction-coefficient abs. error: `+0.0084`

Interpretation:

- The richer field basis does **not** hurt the interaction block in any meaningful way.
- It substantially improves field recovery and `beta` recovery.
- The most plausible explanation is that the uniform field is often too restrictive, so the richer field basis reduces misspecification rather than introducing harmful variance.

This is one of the clearest positive results in the grid.

## Brief note on `shared_feature_interactions`

I do not want to overemphasize this class, but it is worth recording the headline result:

- the hard part of the full problem is the interaction block, not the field block

Mean `parameter_rmse` by model class:

| model class | mean `parameter_rmse` |
| --- | ---: |
| `uniform + known_graph` | `0.1286` |
| `shared_feature_field + known_graph` | `0.0657` |
| `uniform + shared_feature_interactions` | `0.8963` |
| `shared_feature_field + shared_feature_interactions` | `0.6295` |

So the richer interaction basis is clearly the main source of difficulty, while the richer field basis is not.

## Final assessment

If I were presenting this grid to you as a statistical result, I would summarize it as follows:

1. The estimator is in very good shape for the core known-graph model once `T` is moderate and `N` is not too small.
2. `T` is the most reliable recovery lever, and `N` helps strongly as well.
3. Low temperature is the main regime where true-parameter recovery degrades sharply.
4. Very small graph Frobenius norm is bad for recovery, but the data do not support a simple monotone rule that larger Frobenius norm is always better.
5. Your stronger-Frobenius intuition is most defensible for **interaction coefficient recovery**, not for the full parameter vector.
6. Adding `shared_feature_field` is a net win in this grid: it improves recovery rather than harming it.

If I were extending this study next, I would keep the core known-graph setup as the primary benchmark, use `shared_feature_field` as the preferred field specification, and then study the Frobenius effect more directly with a grid that holds temperature fixed while varying graph families more finely between the complete and cycle extremes.
