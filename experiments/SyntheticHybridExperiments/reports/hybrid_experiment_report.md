# Synthetic Experiment Summary

This report is intentionally experiment-level: it focuses on scalar recovery, field reconstruction quality, interaction recovery, and latent-field diagnostics instead of dumping raw `U`, `V`, or `tau` coordinates.

- Completed experiments: 6
- Field modes covered: latent_feature_matrix, uniform

## By Field Mode

| field_mode | count | median_field_rmse | median_interaction_fro_error | median_beta_abs_error | median_xi_abs_error |
| --- | --- | --- | --- | --- | --- |
| latent_feature_matrix | 3 | 0.000880 | 0.113216 | 0.001767 | 0.016038 |
| uniform | 3 | 0.019729 | 0.152650 | 0.003211 | 0.016164 |

## Scalar Recovery

| descriptor | field_mode | intervention_code | network_name | trim_scope | outcome_field_setting | N | T | final_loss | beta_abs_error | xi_abs_error | eta_abs_error | zeta_abs_error | psi_abs_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_generated_interv_uscounty_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.636414 | 0.000061 | 0.011139 | 0.003368 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| hybrid_uscounty_interv_generated_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.598116 | 0.009467 | 0.032003 | 0.000973 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| hybrid_uscounty_interv_uscounty_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.610078 | 0.001767 | 0.016038 | 0.002478 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| hybrid_generated_interv_uscounty_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.637630 | 0.003211 | 0.016164 | 0.003207 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| hybrid_uscounty_interv_generated_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.599177 | 0.004710 | 0.088403 | 0.001476 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| hybrid_uscounty_interv_uscounty_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.610324 | 0.001632 | 0.001247 | 0.000881 |  |  | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Field And Interaction Recovery

| descriptor | field_mode | intervention_code | network_name | trim_scope | outcome_field_setting | N | T | field_rmse | static_field_rmse | interaction_fro_error | tau_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_generated_interv_uscounty_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.000880 |  | 0.105198 |  |
| hybrid_uscounty_interv_generated_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.000868 |  | 0.113216 |  |
| hybrid_uscounty_interv_uscounty_net_latent | latent_feature_matrix |  |  | trimmed | latent | 3009 | 120 | 0.000919 |  | 0.151460 |  |
| hybrid_generated_interv_uscounty_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.019729 | 0.000000 | 0.152650 | 0.019729 |
| hybrid_uscounty_interv_generated_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.028868 | 0.000000 | 0.302179 | 0.028868 |
| hybrid_uscounty_interv_uscounty_net_zero | uniform |  |  | trimmed | zero | 3009 | 120 | 0.017642 | 0.000000 | 0.011777 | 0.017642 |

## Latent Diagnostics

| descriptor | N | T | estimated_field_inf_norm | bound_B | estimated_field_rank | latent_rank_cap | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_generated_interv_uscounty_net_latent | 3009 | 120 | 2.000000 | 2.000000 | 40 | 40 | 40 |
| hybrid_uscounty_interv_generated_net_latent | 3009 | 120 | 2.000000 | 2.000000 | 40 | 40 | 40 |
| hybrid_uscounty_interv_uscounty_net_latent | 3009 | 120 | 2.000000 | 2.000000 | 40 | 40 | 40 |

