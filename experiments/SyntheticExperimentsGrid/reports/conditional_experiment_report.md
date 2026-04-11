# Synthetic Experiment Summary

This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.

- Completed experiments: 8
- Field modes covered: latent_feature_matrix, uniform

## By Field Mode

| field_mode | count | median_field_rmse | median_interaction_fro_error | median_beta_abs_error | median_xi_abs_error |
| --- | --- | --- | --- | --- | --- |
| latent_feature_matrix | 5 | 0.003847 | 0.012868 | 0.004968 | 0.004329 |
| uniform | 3 | 0.072899 | 0.206300 | 0.008310 | 0.070545 |

## Scalar Recovery

| descriptor | field_mode | intervention_code | network_name | trim_scope | outcome_field_setting | N | T | final_loss | beta_abs_error | xi_abs_error | eta_abs_error | zeta_abs_error | psi_abs_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | latent_feature_matrix |  |  |  |  | 100 | 20 | 0.622217 | 0.006037 | 0.182813 | 0.001324 | 0.020957 | 0.035265 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | latent_feature_matrix |  |  |  |  | 300 | 40 | 0.609653 | 0.008312 | 0.012850 | 0.016265 | 0.018513 | 0.011418 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | latent_feature_matrix |  |  |  |  | 600 | 60 | 0.630943 | 0.002285 | 0.003876 | 0.000682 | 0.001783 | 0.005410 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | latent_feature_matrix |  |  |  |  | 600 | 60 | 0.512298 | 0.000665 | 0.004329 | 0.011730 | 0.014020 | 0.006970 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | latent_feature_matrix |  |  |  |  | 1000 | 60 | 0.619204 | 0.004968 | 0.000440 | 0.000767 | 0.005920 | 0.005198 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_uniform_n100_t20_xi0p25 | uniform |  |  |  |  | 100 | 20 | 0.618603 | 0.035526 | 0.194995 | 0.011153 | 0.039289 | 0.012810 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_uniform_n300_t40_xi0p75 | uniform |  |  |  |  | 300 | 40 | 0.606362 | 0.003250 | 0.050376 | 0.003294 | 0.005839 | 0.007901 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| generated_z_uniform_n600_t60_xi1p5 | uniform |  |  |  |  | 600 | 60 | 0.486489 | 0.008310 | 0.070545 | 0.005376 | 0.022805 | 0.010878 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Field And Interaction Recovery

| descriptor | field_mode | intervention_code | network_name | trim_scope | outcome_field_setting | N | T | field_rmse | static_field_rmse | interaction_fro_error | tau_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | latent_feature_matrix |  |  |  |  | 100 | 20 | 0.010570 |  | 0.365627 |  |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | latent_feature_matrix |  |  |  |  | 300 | 40 | 0.003606 |  | 0.030345 |  |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | latent_feature_matrix |  |  |  |  | 600 | 60 | 0.003847 |  | 0.011555 |  |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | latent_feature_matrix |  |  |  |  | 600 | 60 | 0.001618 |  | 0.012868 |  |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | latent_feature_matrix |  |  |  |  | 1000 | 60 | 0.004689 |  | 0.001241 |  |
| generated_z_uniform_n100_t20_xi0p25 | uniform |  |  |  |  | 100 | 20 | 0.160298 | 0.000000 | 0.417240 | 0.160298 |
| generated_z_uniform_n300_t40_xi0p75 | uniform |  |  |  |  | 300 | 40 | 0.052439 | 0.000000 | 0.125521 | 0.052439 |
| generated_z_uniform_n600_t60_xi1p5 | uniform |  |  |  |  | 600 | 60 | 0.072899 | 0.000000 | 0.206300 | 0.072899 |

## Latent Diagnostics

| descriptor | N | T | estimated_field_inf_norm | bound_B | estimated_field_rank | latent_rank_cap | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_z_latent_n100_t20_xi0p25_B1_rank4 | 100 | 20 | 1.000000 | 1.000000 | 4 | 4 | 4 |
| generated_z_latent_n300_t40_xi0p75_B1_rank6 | 300 | 40 | 1.000000 | 1.000000 | 6 | 6 | 6 |
| generated_z_latent_n600_t60_xi0p75_B2_rank6 | 600 | 60 | 2.000000 | 2.000000 | 6 | 6 | 6 |
| generated_z_latent_n600_t60_xi1p5_B1_rank8 | 600 | 60 | 1.000000 | 1.000000 | 8 | 8 | 8 |
| generated_z_latent_n1000_t60_xi0p75_B4_rank10 | 1000 | 60 | 4.000000 | 4.000000 | 10 | 10 | 10 |

