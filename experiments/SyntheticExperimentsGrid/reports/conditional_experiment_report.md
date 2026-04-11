# Synthetic Experiment Summary

This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.

- Completed experiments: 5
- Field modes covered: latent_feature_matrix

## By Field Mode

| field_mode | count | median_field_rmse | median_interaction_fro_error | median_beta_abs_error | median_xi_abs_error | all_bounds_ok | all_ranks_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| latent_feature_matrix | 5 | 0.003764 | 0.042018 | 0.002072 | 0.017793 | yes | yes |

## Scalar Recovery

| descriptor | field_mode | N | T | final_loss | beta_abs_error | xi_abs_error | eta_abs_error | zeta_abs_error | psi_abs_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | latent_feature_matrix | 100 | 20 | 0.622201 | 0.009253 | 0.233480 | 0.005176 | 0.025670 | 0.030799 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | latent_feature_matrix | 300 | 40 | 0.609629 | 0.012429 | 0.017793 | 0.013488 | 0.012435 | 0.007412 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | latent_feature_matrix | 600 | 60 | 0.630873 | 0.002072 | 0.002956 | 0.000806 | 0.007178 | 0.000833 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | latent_feature_matrix | 600 | 60 | 0.512346 | 0.000759 | 0.012398 | 0.012872 | 0.014477 | 0.008096 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | latent_feature_matrix | 1000 | 60 | 0.619141 | 0.000493 | 0.046210 | 0.002237 | 0.009021 | 0.005744 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Field And Interaction Recovery

| descriptor | field_mode | N | T | field_rmse | static_field_rmse | interaction_fro_error | tau_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | latent_feature_matrix | 100 | 20 | 0.010557 | 0.002667 | 0.466960 |  |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | latent_feature_matrix | 300 | 40 | 0.003650 | 0.000484 | 0.042018 |  |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | latent_feature_matrix | 600 | 60 | 0.003764 | 0.000416 | 0.008812 |  |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | latent_feature_matrix | 600 | 60 | 0.001564 | 0.000158 | 0.036853 |  |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | latent_feature_matrix | 1000 | 60 | 0.004683 | 0.000860 | 0.130502 |  |

## Latent Diagnostics

| descriptor | N | T | estimated_field_inf_norm | bound_B | field_bound_ok | estimated_field_rank | latent_rank_cap | rank_ok | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | 100 | 20 | 1.000000 | 1.000000 | yes | 4 | 4 | yes | 4 |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | 300 | 40 | 1.000000 | 1.000000 | yes | 6 | 6 | yes | 6 |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | 600 | 60 | 2.000000 | 2.000000 | yes | 6 | 6 | yes | 6 |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | 600 | 60 | 1.000000 | 1.000000 | yes | 8 | 8 | yes | 8 |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | 1000 | 60 | 4.000000 | 4.000000 | yes | 10 | 10 | yes | 10 |

