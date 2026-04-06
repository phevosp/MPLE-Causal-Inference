# External Field Recovery Report

This report focuses on recovery of the realized external field `alpha_i + tau_t`, rather than only on coefficient recovery or loss.

Artifacts written alongside this report:

- `parameter_recovery_parameter_rows.csv`: one row per fitted parameter
- `parameter_recovery_experiment_summary.csv`: one row per experiment
- `parameter_recovery_factor_summary.csv`: grouped summaries by lever

## Reported Statistics

- `field_l2_error`: L2 error of the full realized external field matrix `(alpha_i + tau_t)` flattened over all `(t, i)` entries
- `field_rmse`: RMSE of the full realized external field matrix `(alpha_i + tau_t)`
- `static_field_rmse`: RMSE of the unit-specific static field `alpha_i`
- `tau_rmse`: RMSE of the time-specific field `tau_t`
- `interaction_fro_error`: Frobenius error of the realized interaction matrix
- `parameter_rmse`: RMSE of the full optimizer parameter vector
- `final_loss`: final conditional pseudo-NLL

## Main Takeaways

- Best full external-field recovery by L2 error is `tau_uniform_field` with `field_l2_error = 5.6364` and `field_rmse = 0.0399`.
- Worst full external-field recovery by L2 error is `tau_shared_field` with `field_l2_error = 7.9758`.
- Best time-specific recovery is `tau_uniform_field` with `tau_rmse = 0.0427`.
- Best unit-specific static-field recovery is `tau_uniform_field` with `static_field_rmse = 0.0246`.

## Field Complexity

| value | experiments | field_l2_error | field_rmse | static_field_rmse | tau_rmse | parameter_rmse | interaction_fro_error | avg alpha abs err | avg tau abs err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared_feature_field | 2 | 7.2993 | 0.0516 | 0.0777 | 0.0790 | 0.0673 | 0.3439 | 0.0337 | 0.0672 |
| uniform | 1 | 5.6364 | 0.0399 | 0.0246 | 0.0427 | 0.0453 | 0.3979 | 0.0246 | 0.0362 |

## Experiment Summary

| experiment | N | T | field complexity | tau mode | field_l2_error | field_rmse | static_field_rmse | tau_rmse | parameter_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tau_uniform_field | 1000 | 20 | uniform | uniform_random | 5.6364 | 0.0399 | 0.0246 | 0.0427 | 0.0453 |
| tau_shared_field | 1000 | 20 | shared_feature_field | uniform_random | 7.9758 | 0.0564 | 0.1052 | 0.0891 | 0.0799 |
| tau_shared_field_alt_seed | 1000 | 20 | shared_feature_field | uniform_random | 6.6228 | 0.0468 | 0.0502 | 0.0690 | 0.0547 |

## Representative True vs Estimated Tables

### Best full-field recovery

- `tau_uniform_field`
- field_l2_error = 5.6364
- field_rmse = 0.0399
- static_field_rmse = 0.0246
- tau_rmse = 0.0427
- interaction_fro_error = 0.3979
- parameter_rmse = 0.0453

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.3000 | 0.3246 | 0.0246 | 0.0006 |
| beta | 0.3500 | 0.3500 | 0.0000 | 0.0000 |
| interaction::adjacency | 0.3500 | 0.2236 | 0.1264 | 0.0160 |
| eta | 0.0800 | 0.0805 | 0.0005 | 0.0000 |
| zeta | -0.2500 | -0.2317 | 0.0183 | 0.0003 |
| psi | 0.2000 | 0.1911 | 0.0089 | 0.0001 |

Tau block summary: mean abs error = 0.0362, median abs error = 0.0350, max abs error = 0.0824.

| tau entry | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| tau::t_0 | -0.1146 | -0.1837 | 0.0691 | 0.0048 |
| tau::t_1 | -0.1978 | -0.2146 | 0.0167 | 0.0003 |
| tau::t_2 | 0.1263 | 0.1575 | 0.0312 | 0.0010 |
| tau::t_3 | 0.1527 | 0.1192 | 0.0335 | 0.0011 |
| tau::t_4 | 0.0991 | 0.0554 | 0.0437 | 0.0019 |
| tau::t_5 | 0.1849 | 0.2337 | 0.0489 | 0.0024 |
| tau::t_6 | -0.0469 | -0.0637 | 0.0169 | 0.0003 |
| tau::t_7 | -0.1006 | -0.1532 | 0.0525 | 0.0028 |
| tau::t_8 | -0.1697 | -0.2521 | 0.0824 | 0.0068 |
| tau::t_9 | 0.1609 | 0.1973 | 0.0364 | 0.0013 |
| tau::t_10 | 0.0009 | -0.0065 | 0.0074 | 0.0001 |
| tau::t_11 | 0.1007 | 0.0381 | 0.0627 | 0.0039 |
| tau::t_12 | 0.1670 | 0.1252 | 0.0418 | 0.0017 |
| tau::t_13 | -0.1869 | -0.2588 | 0.0719 | 0.0052 |
| tau::t_14 | -0.0523 | -0.0742 | 0.0219 | 0.0005 |
| tau::t_15 | -0.1201 | -0.1103 | 0.0099 | 0.0001 |
| tau::t_16 | -0.1215 | -0.1337 | 0.0122 | 0.0001 |
| tau::t_17 | 0.1250 | 0.1313 | 0.0063 | 0.0000 |
| tau::t_18 | 0.0935 | 0.1084 | 0.0149 | 0.0002 |
| tau::t_19 | 0.1607 | 0.2048 | 0.0441 | 0.0019 |

### Worst full-field recovery

- `tau_shared_field`
- field_l2_error = 7.9758
- field_rmse = 0.0564
- static_field_rmse = 0.1052
- tau_rmse = 0.0891
- interaction_fro_error = 0.5967
- parameter_rmse = 0.0799

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.2500 | 0.3520 | 0.1020 | 0.0104 |
| field::linear::feature_1 | 0.1800 | 0.1658 | 0.0142 | 0.0002 |
| field::quadratic::feature_1 | -0.1000 | -0.0751 | 0.0249 | 0.0006 |
| field::linear::feature_2 | 0.1200 | 0.1676 | 0.0476 | 0.0023 |
| field::quadratic::feature_2 | -0.0800 | -0.0138 | 0.0662 | 0.0044 |
| field::linear::feature_3 | 0.1000 | 0.1051 | 0.0051 | 0.0000 |
| field::quadratic::feature_3 | -0.0600 | 0.0377 | 0.0977 | 0.0095 |
| field::linear::feature_4 | 0.0800 | 0.0662 | 0.0138 | 0.0002 |
| field::quadratic::feature_4 | -0.0400 | -0.0459 | 0.0059 | 0.0000 |
| field::linear::feature_5 | 0.0600 | 0.0618 | 0.0018 | 0.0000 |
| field::quadratic::feature_5 | -0.0200 | 0.0629 | 0.0829 | 0.0069 |
| beta | 0.3500 | 0.3716 | 0.0216 | 0.0005 |
| interaction::adjacency | 0.3500 | 0.1601 | 0.1899 | 0.0361 |
| eta | 0.0800 | 0.0832 | 0.0032 | 0.0000 |
| zeta | -0.2500 | -0.2426 | 0.0074 | 0.0001 |
| psi | 0.2000 | 0.1951 | 0.0049 | 0.0000 |

Tau block summary: mean abs error = 0.0776, median abs error = 0.0725, max abs error = 0.1514.

| tau entry | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| tau::t_0 | 0.1627 | 0.1466 | 0.0161 | 0.0003 |
| tau::t_1 | -0.2486 | -0.3582 | 0.1096 | 0.0120 |
| tau::t_2 | 0.0847 | -0.0009 | 0.0856 | 0.0073 |
| tau::t_3 | 0.1458 | 0.0985 | 0.0473 | 0.0022 |
| tau::t_4 | 0.1892 | 0.1311 | 0.0581 | 0.0034 |
| tau::t_5 | 0.0505 | -0.0864 | 0.1369 | 0.0187 |
| tau::t_6 | 0.0439 | -0.0305 | 0.0744 | 0.0055 |
| tau::t_7 | 0.2006 | 0.1300 | 0.0706 | 0.0050 |
| tau::t_8 | -0.0944 | -0.2458 | 0.1514 | 0.0229 |
| tau::t_9 | 0.2425 | 0.2210 | 0.0216 | 0.0005 |
| tau::t_10 | -0.0079 | -0.1593 | 0.1514 | 0.0229 |
| tau::t_11 | -0.0094 | -0.1042 | 0.0949 | 0.0090 |
| tau::t_12 | 0.1922 | 0.1804 | 0.0118 | 0.0001 |
| tau::t_13 | 0.1402 | 0.1082 | 0.0320 | 0.0010 |
| tau::t_14 | -0.1126 | -0.1567 | 0.0441 | 0.0019 |
| tau::t_15 | 0.0435 | -0.0703 | 0.1138 | 0.0129 |
| tau::t_16 | 0.0804 | 0.0172 | 0.0632 | 0.0040 |
| tau::t_17 | 0.2010 | 0.1152 | 0.0858 | 0.0074 |
| tau::t_18 | -0.0246 | -0.1652 | 0.1406 | 0.0198 |
| tau::t_19 | 0.1231 | 0.0811 | 0.0420 | 0.0018 |

### Representative shared-field case

- `tau_shared_field_alt_seed`
- field_l2_error = 6.6228
- field_rmse = 0.0468
- static_field_rmse = 0.0502
- tau_rmse = 0.0690
- interaction_fro_error = 0.0912
- parameter_rmse = 0.0547

| parameter | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| field::intercept | 0.2500 | 0.2954 | 0.0454 | 0.0021 |
| field::linear::feature_1 | 0.1800 | 0.1912 | 0.0112 | 0.0001 |
| field::quadratic::feature_1 | -0.1000 | -0.0282 | 0.0718 | 0.0052 |
| field::linear::feature_2 | 0.1200 | 0.0769 | 0.0431 | 0.0019 |
| field::quadratic::feature_2 | -0.0800 | -0.0669 | 0.0131 | 0.0002 |
| field::linear::feature_3 | 0.1000 | 0.0875 | 0.0125 | 0.0002 |
| field::quadratic::feature_3 | -0.0600 | -0.0749 | 0.0149 | 0.0002 |
| field::linear::feature_4 | 0.0800 | 0.1086 | 0.0286 | 0.0008 |
| field::quadratic::feature_4 | -0.0400 | -0.0245 | 0.0155 | 0.0002 |
| field::linear::feature_5 | 0.0600 | 0.0775 | 0.0175 | 0.0003 |
| field::quadratic::feature_5 | -0.0200 | -0.0267 | 0.0067 | 0.0000 |
| beta | 0.3500 | 0.3579 | 0.0079 | 0.0001 |
| interaction::adjacency | 0.4500 | 0.4862 | 0.0362 | 0.0013 |
| eta | 0.0800 | 0.0780 | 0.0020 | 0.0000 |
| zeta | -0.2500 | -0.2543 | 0.0043 | 0.0000 |
| psi | 0.2000 | 0.2014 | 0.0014 | 0.0000 |

Tau block summary: mean abs error = 0.0568, median abs error = 0.0495, max abs error = 0.1628.

| tau entry | true | estimate | abs error | squared error |
| --- | --- | --- | --- | --- |
| tau::t_0 | 0.2442 | 0.2227 | 0.0215 | 0.0005 |
| tau::t_1 | -0.1783 | -0.1832 | 0.0049 | 0.0000 |
| tau::t_2 | 0.1549 | 0.0815 | 0.0734 | 0.0054 |
| tau::t_3 | -0.1923 | -0.2299 | 0.0376 | 0.0014 |
| tau::t_4 | -0.0762 | -0.0681 | 0.0080 | 0.0001 |
| tau::t_5 | 0.1651 | 0.1036 | 0.0615 | 0.0038 |
| tau::t_6 | 0.0645 | 0.0287 | 0.0358 | 0.0013 |
| tau::t_7 | 0.1160 | 0.1130 | 0.0029 | 0.0000 |
| tau::t_8 | 0.0800 | 0.0428 | 0.0373 | 0.0014 |
| tau::t_9 | 0.1482 | 0.0713 | 0.0769 | 0.0059 |
| tau::t_10 | -0.1754 | -0.2416 | 0.0662 | 0.0044 |
| tau::t_11 | 0.1278 | 0.0919 | 0.0360 | 0.0013 |
| tau::t_12 | 0.1531 | 0.1182 | 0.0349 | 0.0012 |
| tau::t_13 | -0.2211 | -0.2930 | 0.0719 | 0.0052 |
| tau::t_14 | 0.0130 | -0.0806 | 0.0936 | 0.0088 |
| tau::t_15 | -0.1686 | -0.2418 | 0.0732 | 0.0054 |
| tau::t_16 | 0.1030 | -0.0210 | 0.1240 | 0.0154 |
| tau::t_17 | 0.2409 | 0.0780 | 0.1628 | 0.0265 |
| tau::t_18 | 0.2117 | 0.1766 | 0.0351 | 0.0012 |
| tau::t_19 | 0.1041 | 0.0258 | 0.0783 | 0.0061 |

