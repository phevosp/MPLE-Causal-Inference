# Best Fit By Experiment

Each row is the top-ranked MPLE variant within one generated experiment.

| experiment_name | descriptor | intervention_source | graph_source | field_mode | N | T | s | variant_name | variant_slug | optimizer_mode | latent_rank | lambda_nuclear | lambda_frobenius | lambda_uv_ridge | fixed_scalar_params | ranking_mode | total_recovery_rmse | final_loss | field_rmse | interaction_fro_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_us_county_intervention_uscounty_graph_base | hybrid_us_county_intervention_uscounty_graph_base | fixed_artifact | fixed_artifact | random_low_rank | 3009 | 120 | 55 | nuclear_lambda_12e_2 | nuclear_lambda_12e_2 | nuclear_norm | 0 | 0.120000 | 0.000000 | 0.000000 | {} | total_recovery_rmse | 0.046766 | 0.517558 | 0.000000 | 0.359844 | CONVERGED: proximal objective tolerance reached |
| hybrid_us_county_intervention_uscounty_graph_confounding | hybrid_us_county_intervention_uscounty_graph_confounding | fixed_artifact | fixed_artifact | low_rank_plus_early_treatment_confounding | 3009 | 120 | 55 | manifold_rank_2 | manifold_rank_2 | exact_rank_manifold | 2 | 0.000000 | 0.100000 | 0.000000 | {} | total_recovery_rmse | 0.543940 | 0.538398 | 0.349098 | 1.081557 | Terminated - min grad norm reached after 626 iterations, 34.73 seconds. | best_start=1/1 |
| hybrid_us_county_intervention_uscounty_graph_rank_20 | hybrid_us_county_intervention_uscounty_graph_rank_20 | fixed_artifact | fixed_artifact | random_low_rank | 3009 | 120 | 55 | concurrent_rank_2 | concurrent_rank_2 | concurrent_latent_rank | 2 | 0.000000 | 0.000000 | 10.000000 | {} | total_recovery_rmse | 0.201679 | 0.460348 | 0.160862 | 0.120931 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

