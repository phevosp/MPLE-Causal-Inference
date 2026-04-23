# Best Fit By Experiment

Each row is the top-ranked MPLE variant within one generated experiment.

| experiment_name | descriptor | intervention_source | graph_source | N | T | s | variant_name | variant_slug | field_mode | latent_rank | lambda_nuclear | lambda_frobenius | B | fixed_scalar_params | ranking_mode | total_recovery_rmse | final_loss | field_rmse | interaction_fro_error | optimizer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_rank_10_B1 | synthetic_rank_10_B1 | generated | generated | 1000 | 100 | 0 | manifold_rank_1_high_pen | manifold_rank_1_high_pen | low_rank | 1 | 0.000000 | 0.100000 | 15.000000 | {'beta': 0, 'xi': 0, 'eta': 0} | total_recovery_rmse | 0.165740 | 0.689126 | 0.165740 | 0.000000 | Terminated - min grad norm reached after 78 iterations, 1.72 seconds. | best_start=2/3 |
| synthetic_rank_1_B1 | synthetic_rank_1_B1 | generated | generated | 1000 | 100 | 0 | manifold_rank_1_high_pen | manifold_rank_1_high_pen | low_rank | 1 | 0.000000 | 0.100000 | 15.000000 | {'beta': 0, 'xi': 0, 'eta': 0} | total_recovery_rmse | 0.081252 | 0.688378 | 0.081252 | 0.000000 | Terminated - min grad norm reached after 49 iterations, 1.11 seconds. | best_start=3/3 |

