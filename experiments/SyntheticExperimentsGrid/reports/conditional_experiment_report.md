# Synthetic Experiment Summary

This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.

- Completed experiments: 4
- Field modes covered: latent_feature_matrix

## By Field Mode

| field_mode | count | median_field_rmse | median_interaction_fro_error | median_beta_abs_error | median_xi_abs_error |
| --- | --- | --- | --- | --- | --- |
| latent_feature_matrix | 4 | 0.000743 | 0.032268 | 0.003540 | 0.009505 |

## Scalar Recovery

| descriptor | field_mode | N | T | final_loss | beta_abs_error | xi_abs_error | eta_abs_error | zeta_abs_error | psi_abs_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.602116 | 0.004712 | 0.011604 | 0.000041 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.602082 | 0.005203 | 0.010579 | 0.000604 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.603398 | 0.002368 | 0.008431 | 0.001952 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.603010 | 0.000599 | 0.004688 | 0.000237 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Field And Interaction Recovery

| descriptor | field_mode | N | T | field_rmse | static_field_rmse | interaction_fro_error | tau_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.000343 |  | 0.038916 |  |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.001031 |  | 0.035475 |  |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.000456 |  | 0.029061 |  |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0_beta_masked_true | latent_feature_matrix | 3009 | 120 | 0.001356 |  | 0.016160 |  |

## Latent Diagnostics

| descriptor | N | T | estimated_field_inf_norm | bound_B | estimated_field_rank | latent_rank_cap | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0_beta_masked_true | 3009 | 120 | 1.000000 | 1.000000 | 10 | 10 | 10 |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0_beta_masked_true | 3009 | 120 | 3.000000 | 3.000000 | 10 | 10 | 10 |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0_beta_masked_true | 3009 | 120 | 1.000000 | 1.000000 | 50 | 50 | 50 |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0_beta_masked_true | 3009 | 120 | 3.000000 | 3.000000 | 50 | 50 | 50 |

