# Parameter Recovery Report

This report focuses on direct true-vs-estimated parameter recovery, not on loss.

The full flattened parameter table is available in `reports/parameter_recovery_parameter_rows.csv`.

## Main Takeaways

- In the core known-graph setting, larger `N` and larger `T` both improve recovery substantially. The best baseline experiment is `N5000_T100_baseline_temp_fro_medium_uniform_known_graph` with parameter RMSE 0.0011.
- Low temperature is the hardest regime. The worst known-graph experiment is `N100_T10_low_temp_fro_small_uniform_known_graph` with parameter RMSE 1.6636.
- `shared_feature_field` improves recovery in this grid rather than hurting it. It lowers mean parameter RMSE in the known-graph setting from 0.1286 to 0.0657.
- The Frobenius effect is not monotone. Very small Frobenius norm is bad for recovery, but the medium Frobenius regime is often best overall.

## Levers

### `N`

| N | experiments | parameter_rmse | field_rmse | interaction_fro_error | avg field abs err | avg interaction abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 36 | 0.1822 | 0.2461 | 0.2717 | 0.2378 | 0.1128 |
| 1000 | 36 | 0.0830 | 0.1115 | 0.1677 | 0.1109 | 0.0366 |
| 5000 | 36 | 0.0261 | 0.0230 | 0.1559 | 0.0245 | 0.0302 |

### `T`

| T | experiments | parameter_rmse | field_rmse | interaction_fro_error | beta abs err |
| --- | --- | --- | --- | --- | --- |
| 10 | 54 | 0.1605 | 0.2167 | 0.2376 | 0.2013 |
| 100 | 54 | 0.0338 | 0.0370 | 0.1593 | 0.0235 |

### Temperature

| temperature | experiments | parameter_rmse | field_rmse | interaction_fro_error | beta abs err | eta abs err | zeta abs err | psi abs err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_temp | 36 | 0.0327 | 0.0302 | 0.1140 | 0.0157 | 0.0089 | 0.0136 | 0.0144 |
| high_temp | 36 | 0.0394 | 0.0255 | 0.1864 | 0.0110 | 0.0094 | 0.0113 | 0.0104 |
| low_temp | 36 | 0.2192 | 0.3248 | 0.2949 | 0.3106 | 0.0381 | 0.0184 | 0.0193 |

### Frobenius Norm

| fro regime | experiments | parameter_rmse | field_rmse | interaction_fro_error | avg field abs err | avg interaction abs err | beta abs err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fro_large | 36 | 0.0874 | 0.1176 | 0.3483 | 0.1169 | 0.0259 | 0.1094 |
| fro_medium | 36 | 0.0562 | 0.0407 | 0.1782 | 0.0398 | 0.0851 | 0.0315 |
| fro_small | 36 | 0.1477 | 0.2222 | 0.0688 | 0.2165 | 0.0686 | 0.1964 |

The data support a limited version of the Frobenius hypothesis: moving away from the complete-graph regime helps, but the best overall recovery is usually at the medium regime rather than the largest Frobenius regime.

### `shared_feature_field`

| field complexity | experiments | parameter_rmse | field_rmse | interaction_fro_error | beta abs err | avg field abs err |
| --- | --- | --- | --- | --- | --- | --- |
| shared_feature_field | 54 | 0.0657 | 0.0594 | 0.1966 | 0.0280 | 0.0545 |
| uniform | 54 | 0.1286 | 0.1943 | 0.2003 | 0.1968 | 0.1943 |

## Representative True vs Estimated Tables

### Best baseline case

- `N5000_T100_baseline_temp_fro_medium_uniform_known_graph`
- parameter_rmse = 0.0011
- field_rmse = 0.0000
- interaction_fro_error = 0.0067

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| beta | 0.4000 | 0.4003 | 0.0003 | 0.0000 |
| interaction::adjacency | 0.5000 | 0.5025 | 0.0025 | 0.0000 |
| eta | 0.1000 | 0.0993 | 0.0007 | 0.0000 |
| zeta | -0.5000 | -0.4994 | 0.0006 | 0.0000 |
| psi | 0.4000 | 0.4005 | 0.0005 | 0.0000 |

### Hard low-temperature case

- `N100_T10_low_temp_fro_small_uniform_known_graph`
- parameter_rmse = 1.6636
- field_rmse = 2.8494
- interaction_fro_error = 0.1570

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 1.0000 | 3.8494 | 2.8494 | 8.1191 |
| beta | 1.1000 | 4.0011 | 2.9011 | 8.4162 |
| interaction::adjacency | 2.0000 | 1.8438 | 0.1562 | 0.0244 |
| eta | 0.7000 | 0.4953 | 0.2047 | 0.0419 |
| zeta | -1.0000 | -0.9490 | 0.0510 | 0.0026 |
| psi | 0.4000 | 0.4337 | 0.0337 | 0.0011 |

### Rich field case

- `N5000_T100_high_temp_fro_large_shared_feature_field_known_graph`
- parameter_rmse = 0.0060
- field_rmse = 0.0049
- interaction_fro_error = 0.2453

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.1750 | 0.1747 | 0.0003 | 0.0000 |
| field::linear::feature_1 | 0.1400 | 0.1255 | 0.0145 | 0.0002 |
| field::quadratic::feature_1 | -0.1250 | -0.1295 | 0.0045 | 0.0000 |
| field::linear::feature_2 | 0.0070 | 0.0093 | 0.0023 | 0.0000 |
| field::quadratic::feature_2 | 0.1250 | 0.1287 | 0.0037 | 0.0000 |
| field::linear::feature_3 | 0.1400 | 0.1360 | 0.0040 | 0.0000 |
| field::quadratic::feature_3 | -0.1250 | -0.1258 | 0.0008 | 0.0000 |
| field::linear::feature_4 | -0.0400 | -0.0494 | 0.0094 | 0.0001 |
| field::quadratic::feature_4 | 0.1250 | 0.1355 | 0.0105 | 0.0001 |
| field::linear::feature_5 | -0.1800 | -0.1745 | 0.0055 | 0.0000 |
| field::quadratic::feature_5 | 0.1850 | 0.1915 | 0.0065 | 0.0000 |
| beta | 0.1200 | 0.1206 | 0.0006 | 0.0000 |
| interaction::adjacency | 0.1500 | 0.1451 | 0.0049 | 0.0000 |
| eta | 0.0100 | 0.0103 | 0.0003 | 0.0000 |
| zeta | -0.1500 | -0.1513 | 0.0013 | 0.0000 |
| psi | 0.0500 | 0.0477 | 0.0023 | 0.0000 |

## Frobenius Comparison at Fixed `N`, `T`, and Temperature

These three experiments differ only in graph family, so they isolate the effect of the graph's Frobenius norm.

### N5000_T100_baseline_temp_fro_small_uniform_known_graph

- graph family = complete
- gamma Frobenius norm = 1.0001
- parameter_rmse = 0.0032
- field_rmse = 0.0041
- interaction_fro_error = 0.0062

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.5000 | 0.4959 | 0.0041 | 0.0000 |
| beta | 0.4000 | 0.4014 | 0.0014 | 0.0000 |
| interaction::adjacency | 0.5000 | 0.5062 | 0.0062 | 0.0000 |
| eta | 0.1000 | 0.0995 | 0.0005 | 0.0000 |
| zeta | -0.5000 | -0.4987 | 0.0013 | 0.0000 |
| psi | 0.4000 | 0.3988 | 0.0012 | 0.0000 |

### N5000_T100_baseline_temp_fro_medium_uniform_known_graph

- graph family = erdos_renyi
- gamma Frobenius norm = 2.7079
- parameter_rmse = 0.0011
- field_rmse = 0.0000
- interaction_fro_error = 0.0067

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| beta | 0.4000 | 0.4003 | 0.0003 | 0.0000 |
| interaction::adjacency | 0.5000 | 0.5025 | 0.0025 | 0.0000 |
| eta | 0.1000 | 0.0993 | 0.0007 | 0.0000 |
| zeta | -0.5000 | -0.4994 | 0.0006 | 0.0000 |
| psi | 0.4000 | 0.4005 | 0.0005 | 0.0000 |

### N5000_T100_baseline_temp_fro_large_uniform_known_graph

- graph family = cycle
- gamma Frobenius norm = 50.0000
- parameter_rmse = 0.0028
- field_rmse = 0.0038
- interaction_fro_error = 0.1396

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.5000 | 0.4962 | 0.0038 | 0.0000 |
| beta | 0.4000 | 0.4008 | 0.0008 | 0.0000 |
| interaction::adjacency | 0.5000 | 0.5028 | 0.0028 | 0.0000 |
| eta | 0.1000 | 0.1023 | 0.0023 | 0.0000 |
| zeta | -0.5000 | -0.4996 | 0.0004 | 0.0000 |
| psi | 0.4000 | 0.4043 | 0.0043 | 0.0000 |

