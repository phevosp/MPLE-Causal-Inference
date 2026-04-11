# Synthetic Experiment Summary

This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.

- Completed experiments: 4
- Field modes covered: latent_feature_matrix

## By Field Mode

| field_mode | count | median_field_rmse | median_interaction_fro_error | median_beta_abs_error | median_xi_abs_error |
| --- | --- | --- | --- | --- | --- |
| latent_feature_matrix | 4 | 0.000776 | 0.058727 | 0.005670 | 0.017649 |

## Scalar Recovery

| descriptor | field_mode | N | T | final_loss | beta_abs_error | xi_abs_error | eta_abs_error | zeta_abs_error | psi_abs_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0 | latent_feature_matrix | 3009 | 120 | 0.601305 | 0.000552 | 0.002038 | 0.001198 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0 | latent_feature_matrix | 3009 | 120 | 0.601278 | 0.011491 | 0.033663 | 0.000533 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0 | latent_feature_matrix | 3009 | 120 | 0.603580 | 0.009022 | 0.028735 | 0.000711 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0 | latent_feature_matrix | 3009 | 120 | 0.603328 | 0.002318 | 0.006563 | 0.000163 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Field And Interaction Recovery

| descriptor | field_mode | N | T | field_rmse | static_field_rmse | interaction_fro_error | tau_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0 | latent_feature_matrix | 3009 | 120 | 0.000406 |  | 0.006722 |  |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0 | latent_feature_matrix | 3009 | 120 | 0.001086 |  | 0.116214 |  |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0 | latent_feature_matrix | 3009 | 120 | 0.000467 |  | 0.094794 |  |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0 | latent_feature_matrix | 3009 | 120 | 0.001349 |  | 0.022659 |  |

## Latent Diagnostics

| descriptor | N | T | estimated_field_inf_norm | bound_B | estimated_field_rank | latent_rank_cap | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B1.0 | 3009 | 120 | 1.000000 | 1.000000 | 10 | 10 | 10 |
| fixed_z_complete_cov_ge_30_lag2w_trimmed_latent_rank10_B3.0 | 3009 | 120 | 3.000000 | 3.000000 | 10 | 10 | 10 |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B1.0 | 3009 | 120 | 1.000000 | 1.000000 | 50 | 50 | 50 |
| fixed_z_complete_cov_ge_40_lag2w_trimmed_latent_rank50_B3.0 | 3009 | 120 | 3.000000 | 3.000000 | 50 | 50 | 50 |

