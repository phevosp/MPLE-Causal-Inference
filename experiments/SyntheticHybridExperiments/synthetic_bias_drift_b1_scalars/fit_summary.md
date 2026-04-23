# Fit Summary: synthetic_bias_drift_B1_scalars

- Best variant: `manifold_rank_1`
- Ranking mode: `total_recovery_rmse`

| experiment_name | descriptor | variant_name | variant_slug | optimizer_mode | field_mode | latent_rank | lambda_nuclear | lambda_frobenius | lambda_uv_ridge | fixed_scalar_params | ranking_mode | rank_in_experiment | is_best | total_recovery_rmse | final_loss | field_rmse | interaction_fro_error | optimizer_status | beta_abs_error | xi_abs_error | eta_abs_error | estimated_field_max_abs_entry | estimated_field_rank | true_field_max_abs_entry | true_field_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_bias_drift_B1_scalars | synthetic_bias_drift_B1_scalars | manifold_rank_1 | manifold_rank_1 | exact_rank_manifold | node_bias_plus_smooth_time_drift | 1 | 0.000000 | 0.100000 | 0.000000 | {} | total_recovery_rmse | 1 | true | 0.224324 | 0.566897 | 0.153957 | 0.184853 | Terminated - max iterations reached after 121.41 seconds. | best_start=1/1 | 0.004683 | 0.058621 | 0.007062 | 1.122129 | 1 | 1.000000 | 2 |
| synthetic_bias_drift_B1_scalars | synthetic_bias_drift_B1_scalars | manifold_rank_10 | manifold_rank_10 | exact_rank_manifold | node_bias_plus_smooth_time_drift | 10 | 0.000000 | 0.500000 | 0.000000 | {} | total_recovery_rmse | 2 | false | 0.329021 | 0.535943 | 0.234359 | 0.177430 | Terminated - max iterations reached after 129.32 seconds. | best_start=1/1 | 0.007831 | 0.056267 | 0.030565 | 0.978703 | 10 | 1.000000 | 2 |

