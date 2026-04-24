# USCountyVaccination Data Analysis

This folder summarizes the nationwide county-week dataset for outcome `death_rate_100k_ge_2` and interventions `complete_cov_ge_20`, `complete_cov_ge_30`, and `complete_cov_ge_40`.

## Headline facts

- Counties in scope: `3,213`
- Weeks in scope: `121`
- County-week rows: `388,773`
- Date range: `2020-01-26` through `2022-05-15`

## Outputs

- [county_population_summary.csv](county_population_summary.csv)
- [weekly_binary_shares.csv](weekly_binary_shares.csv)
- [lag2_correlations.csv](lag2_correlations.csv)
- [county_population_distribution.png](county_population_distribution.png)
- [intervention_share_complete_cov_ge_20.png](intervention_share_complete_cov_ge_20.png)
- [intervention_share_complete_cov_ge_30.png](intervention_share_complete_cov_ge_30.png)
- [intervention_share_complete_cov_ge_40.png](intervention_share_complete_cov_ge_40.png)
- [outcome_share_death_rate_100k_ge_2.png](outcome_share_death_rate_100k_ge_2.png)

## Methodology

- The source of truth is `processed/us_county_binary_panel.csv.gz`.
- The binary variables are the `pm1` columns in that processed panel.
- Intervention binaries inherit the repo's existing semantics: weeks before first observed vaccination reporting are prefilled with `-1`.
- Weekly shares use the full county panel for each week and exclude only truly missing values from the share denominator.
- The correlation uses outcome at week `t` and intervention at week `t-2`.

## County population summary

- Non-missing county populations: `3,213`
- Mean population: `100578.8`
- Median population: `26010.0`
- 10th percentile: `5189.6`
- 90th percentile: `207992.2`
- Maximum population: `10039107.0`

## Binary prevalence and missingness

| variable_code | column | eligible_count | missing_count | have_count | have_share |
| --- | --- | --- | --- | --- | --- |
| death_rate_100k_ge_2 | x_death_rate_100k_ge_2_pm1 | 388,773 | 0 | 120,976 | 0.3112 |
| complete_cov_ge_20 | z_complete_cov_ge_20_pm1 | 388,744 | 29 | 167,211 | 0.4301 |
| complete_cov_ge_30 | z_complete_cov_ge_30_pm1 | 388,744 | 29 | 142,953 | 0.3677 |
| complete_cov_ge_40 | z_complete_cov_ge_40_pm1 | 388,744 | 29 | 100,957 | 0.2597 |

## Lag-2 correlations

| intervention_code | intervention_column | lag_weeks | valid_rows | correlation |
| --- | --- | --- | --- | --- |
| complete_cov_ge_20 | z_complete_cov_ge_20_pm1 | 2 | 382,324 | 0.063489 |
| complete_cov_ge_30 | z_complete_cov_ge_30_pm1 | 2 | 382,324 | 0.082371 |
| complete_cov_ge_40 | z_complete_cov_ge_40_pm1 | 2 | 382,324 | 0.073861 |

## Notes

- Outcome shares are available for all county-weeks in the panel.
- The selected intervention columns contain only a small number of truly missing rows; those rows remain visible in the CSV outputs via `missing_count` and `missing_share`.
- The intervention time-series reflect the binary threshold definitions already used by the experiment pipeline, so early pre-vaccination weeks appear as counties not yet having the intervention.
