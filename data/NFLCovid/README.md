# NFLCovid

Nationwide county-week NFL attendance and COVID case/death dataset for MPLE experiments.

## Sources

- Attendance geography Google Sheet: `https://tinyurl.com/bdemmhx8`
- NFL schedule/calendar: `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
- Game attendance: ESPN summary API `gameInfo.attendance` via `https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}`
- COVID outcomes: `https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv`
- County geometry: `https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip` with `https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip` fallback
- ACS 2019 county covariates: `https://api.census.gov/data/2019/acs/acs5` and profile/subject companions
- CDC/ATSDR SVI 2020 county data: `https://svi.cdc.gov/Documents/Data/2020/CSV/states_counties/SVI_2020_US_county.csv`
- USDA ERS RUCC 2013: `https://www.ers.usda.gov/media/5769/2013-rural-urban-continuum-codes.xls?v=15208`

## Dataset Snapshot

### Raw attendance geography

| Metric | Value |
| --- | --- |
| attendance_row_count | 197 |
| sheet_team_count_total | 32 |
| sheet_team_count_with_geography | 18 |
| sheet_team_count_without_geography | 14 |
| unique_county_state_pairs | 178 |
| listed_share_pct_min | 47.36 |
| listed_share_pct_max | 77.71 |
| exact_match_count | 191 |
| override_match_count | 6 |
| ambiguous_match_count | 0 |
| unresolved_count | 0 |
| unsupported_in_county_panel_count | 2 |
| teams_without_geography | Arizona, Chicago, Detroit, LA Chargers, LA Rams, Las Vegas, Minnesota, NY Giants, NY Jets, New England, New Orleans, San Francisco, Seattle, Washington |

County-share totals by team:

| Team | Listed share pct |
| --- | --- |
| Pittsburgh | 47.36 |
| Indianapolis | 47.76 |
| Dallas | 51.07 |
| Atlanta | 53.66 |
| Carolina | 54.75 |
| Kansas City | 56.66 |
| Jacksonville | 60.52 |
| Cincinnati | 60.83 |
| Tampa Bay | 62.5 |
| Cleveland | 62.57 |
| Tennessee | 62.61 |
| Green Bay | 64.19 |
| Miami | 64.43 |
| Denver | 64.81 |
| Philadelphia | 65.03 |
| Houston | 69.37 |
| Baltimore | 74.03 |
| Buffalo | 77.71 |

### Game attendance

| Metric | Value |
| --- | --- |
| total_games | 269 |
| positive_attendance_games | 116 |
| zero_attendance_games | 153 |
| missing_attendance_games | 0 |
| date_range_start | 2020-09-10 |
| date_range_end | 2021-02-07 |
| excluded_arizona_games | 8 |
| excluded_arizona_attendance | 9,600 |
| zeroed_unsupported_games | 10 |
| zeroed_unsupported_attendance | 32,226 |

Game counts by type:

| Type | Games |
| --- | --- |
| CON | 2 |
| DIV | 4 |
| REG | 256 |
| SB | 1 |
| WC | 6 |

### County-game exposure

| Metric | Value |
| --- | --- |
| county_game_row_count | 1,672 |
| county_game_modeled_row_count | 1,656 |
| assigned_attendance_total | 740,604.206 |
| assigned_attendance_modeled_total | 739,404.3116 |
| unassigned_attendance_total | 518,183.794 |
| unsupported_outcome_attendance_total | 1,199.8944 |

### County-week panel

| Metric | Value |
| --- | --- |
| county_count | 3,081 |
| week_count | 47 |
| row_count | 144,807 |
| case_threshold_for_retention | 200 |
| counties_removed_below_case_threshold | 153 |

Panel missingness before fills:

| Column | Missing count | Missing share |
| --- | --- | --- |
| new_cases | 2,392 | 0.0165 |
| new_deaths | 2,392 | 0.0165 |
| case_rate_100k | 2,392 | 0.0165 |
| death_rate_100k | 2,392 | 0.0165 |
| attendance_count | 143,277 | 0.9894 |
| attendance_share_pct | 143,277 | 0.9894 |

Panel missingness after fills:

| Column | Missing count | Missing share |
| --- | --- | --- |
| new_cases | 0 | 0 |
| new_deaths | 0 | 0 |
| case_rate_100k | 0 | 0 |
| death_rate_100k | 0 | 0 |
| attendance_count | 0 | 0 |
| attendance_share_pct | 0 | 0 |

Feature missingness before imputation:

| Column | Missing count | Missing share |
| --- | --- | --- |
| population_density | 0 | 0 |
| log_population | 0 | 0 |
| svi_overall | 79 | 0.0256 |
| rucc_2013 | 2 | 0.0006 |
| senior_population | 3 | 0.001 |
| college_education | 79 | 0.0256 |

Feature missingness after imputation:

| Column | Missing count | Missing share |
| --- | --- | --- |
| population_density | 0 | 0 |
| log_population | 0 | 0 |
| svi_overall | 0 | 0 |
| rucc_2013 | 0 | 0 |
| senior_population | 0 | 0 |
| college_education | 0 | 0 |

### Network and support

| Network | Nodes | Edges | Components |
| --- | --- | --- | --- |
| contiguity | 3,081 | 8,814 | 12 |
| knn_8 | 3,081 | 13,781 | 2 |
| distance_kernel_8 | 3,081 | 13,781 | 2 |

Binary threshold diagnostics:

| Type | Column | Eligible rows | Eligible share | Positive share | Transition rate |
| --- | --- | --- | --- | --- | --- |
| outcome | x_death_rate_100k_ge_2_pm1 | 144,807 | 1 | 0.3515 | 0.235 |
| intervention | z_attendance_share_pct_ge_0p5_pm1 | 144,807 | 1 | 0.0003 | 0.0005 |

Realized dense-support summary:

| Outcome | Intervention | Lag | Requested nodes | Realized nodes | Requested weeks | Realized weeks |
| --- | --- | --- | --- | --- | --- | --- |
| death_rate_100k_ge_2 | attendance_share_pct_ge_0p5 | 0w | 3,081 | 3,081 | 47 | 47 |
| death_rate_100k_ge_2 | attendance_share_pct_ge_0p5 | 1w | 3,081 | 3,081 | 47 | 46 |
| death_rate_100k_ge_2 | attendance_share_pct_ge_0p5 | 2w | 3,081 | 3,081 | 47 | 45 |
| death_rate_100k_ge_2 | attendance_share_pct_ge_0p5 | 3w | 3,081 | 3,081 | 47 | 44 |
| death_rate_100k_ge_2 | attendance_share_pct_ge_0p5 | 4w | 3,081 | 3,081 | 47 | 43 |

## Processed Outputs

- `nfl_team_county_fan_shares.csv.gz`
- `nfl_game_attendance.csv.gz`
- `nfl_county_game_exposure.csv.gz`
- `nfl_county_weekly_nyt.csv.gz`
- `nfl_county_weekly_panel.csv.gz`
- `nfl_county_feature_basis.csv.gz`
- `nfl_county_feature_dictionary.csv`
- `nfl_covid_counties.gpkg`
- `nfl_covid_county_centroids.csv`
- `nfl_covid_county_contiguity_adjacency.csv.gz`
- `nfl_covid_county_knn_8_adjacency.csv.gz`
- `nfl_covid_county_distance_kernel_8_adjacency.csv.gz`
- `nfl_covid_county_network_summary.csv`
- `processing_summary.json`
