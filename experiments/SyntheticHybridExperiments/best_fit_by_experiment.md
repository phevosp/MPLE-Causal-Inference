# Best Fit By Experiment

Each row is the top-ranked MPLE variant within one generated experiment.

| experiment_name | descriptor | intervention_source | graph_source | field_mode | N | T | s | variant_name | variant_slug | optimizer_mode | latent_rank | lambda_nuclear | lambda_frobenius | lambda_uv_ridge | fixed_scalar_params | ranking_mode | total_recovery_rmse | final_loss | field_rmse | interaction_fro_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_us_county_intervention_uscounty_graph_confounding | hybrid_us_county_intervention_uscounty_graph_confounding | fixed_artifact | fixed_artifact | low_rank_plus_early_treatment_confounding | 3009 | 120 | 55 | manifold_rank_2 | manifold_rank_2 | exact_rank_manifold | 2 | 0.000000 | 0.100000 | 0.000000 | {} | total_recovery_rmse | 0.475696 | 0.523967 | 0.347449 | 0.916016 | Terminated - min grad norm reached after 1594 iterations, 95.07 seconds. | best_start=1/1 |
| synthetic_base | synthetic_base | generated | generated | random_low_rank | 1000 | 100 | 0 | alternating_rank_10 | alternating_rank_10 | alternating_latent_rank | 10 | 0.000000 | 0.000000 | 18.000000 | {} | total_recovery_rmse | 0.052204 | 0.578343 | 0.001294 | 0.142923 | CONVERGED: alternating objective tolerance reached |

