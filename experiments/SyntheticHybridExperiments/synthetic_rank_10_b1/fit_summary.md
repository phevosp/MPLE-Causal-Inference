# Fit Summary: synthetic_rank_10_B1

- Best variant: `manifold_rank_1_high_pen`
- Ranking mode: `total_recovery_rmse`

| experiment_name | descriptor | variant_name | variant_slug | field_mode | latent_rank | lambda_nuclear | lambda_frobenius | B | fixed_scalar_params | ranking_mode | rank_in_experiment | is_best | total_recovery_rmse | final_loss | field_rmse | interaction_fro_error | optimizer_status | beta_abs_error | xi_abs_error | eta_abs_error | estimated_field_inf_norm | estimated_field_rank | true_field_inf_norm | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_rank_10_B1 | synthetic_rank_10_B1 | manifold_rank_1_high_pen | manifold_rank_1_high_pen | low_rank | 1 | 0.000000 | 0.100000 | 15.000000 | {'beta': 0, 'xi': 0, 'eta': 0} | total_recovery_rmse | 1 | true | 0.165740 | 0.689126 | 0.165740 | 0.000000 | Terminated - min grad norm reached after 78 iterations, 1.72 seconds. | best_start=2/3 | 0.000000 | 0.000000 | 0.000000 | 0.310360 | 1 | 1.000000 | 10 |
| synthetic_rank_10_B1 | synthetic_rank_10_B1 | manifold_rank_10_high_pen | manifold_rank_10_high_pen | low_rank | 10 | 0.000000 | 0.100000 | 15.000000 | {'beta': 0, 'xi': 0, 'eta': 0} | total_recovery_rmse | 2 | false | 0.351588 | 0.611673 | 0.351588 | 0.000000 | Terminated - min grad norm reached after 121 iterations, 2.86 seconds. | best_start=3/3 | 0.000000 | 0.000000 | 0.000000 | 2.150381 | 10 | 1.000000 | 10 |

