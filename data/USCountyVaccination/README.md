# USCountyVaccination

Nationwide county-week real-data package and experiment pipeline for MPLE, parallel to the Ohio county workflow but built over the full US county-equivalent geography.

## Active Experiment Scope

- Unit of analysis: county-week
- Geography scope: all county-equivalent units with valid 5-digit FIPS in the shared geography/NYT/vaccination overlap
  - Optional experiment-time trim: `--trim` keeps only mainland US counties with `total_population >= 2000`
- Core date window: week ends `2020-01-26` through `2022-05-15`
- Active outcome families:
  - `case_rate_100k >= 100`
  - `case_rate_100k >= 200`
  - `death_rate_100k >= 2`
- Active intervention families:
  - `complete_cov >= 10`
  - `complete_cov >= 20`
  - `complete_cov >= 30`
  - `complete_cov >= 40`
  - `complete_cov >= 50`
  - `complete_cov >= 60`
  - `complete_cov >= 70`
  - `complete_cov >= 80`
  - `partial_cov >= 10`
  - `partial_cov >= 20`
  - `partial_cov >= 30`
  - `partial_cov >= 40`
  - `partial_cov >= 50`
  - `partial_cov >= 60`
  - `partial_cov >= 70`
  - `partial_cov >= 80`
- Active lag grid: `0w`, `1w`, `2w`, `3w`, `4w`
- Default network for fitting: `contiguity`

The processed weekly panel still carries continuous helper columns such as `complete_cov_delta`, `booster_cov_delta`, `prev_case_rate_100k`, and `case_growth_ratio`, but the active nationwide experiment grid no longer materializes delta-based intervention experiments or growth-based outcome experiments.

## Sources

- COVID outcomes: NYT county archive
  - `https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv`
- Primary vaccination source: CDC county vaccination dataset `8xkx-amqh`
  - `https://data.cdc.gov/resource/8xkx-amqh.csv`
- Vaccination filler source: Bansal Lab county vaccination time series
  - `https://media.githubusercontent.com/media/bansallab/vaccinetracking/main/vacc_data/data_county_timeseries.csv`
  - The ordinary GitHub raw URL serves a Git LFS pointer rather than the CSV payload, so the `media.githubusercontent.com` URL is required.
- Geography: Census TIGER county shapefiles
  - Primary: `https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip`
  - Fallback: `https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip`
- County covariates: Census ACS 2021 county APIs
  - `https://api.census.gov/data/2021/acs/acs5`
  - `https://api.census.gov/data/2021/acs/acs5/subject`
  - `https://api.census.gov/data/2021/acs/acs5/profile`

## Construction

### 1. County identifiers

- Every source is normalized to 5-digit county FIPS strings before any join.
- NYT rows with `county == "Unknown"` or missing FIPS are dropped.
- Bansal rows are restricted to `GEOFLAG == "County"`.
- The county master table is the FIPS intersection of:
  - TIGER geometry
  - NYT county data
  - vaccination weekly data

### 2. NYT outcomes

- NYT cumulative `cases` and `deaths` are differenced within county to form daily `new_cases` and `new_deaths`.
- Negative daily revisions are treated as corrections and clipped to zero after differencing.
- Daily data are aggregated to county-week by ISO year/week:
  - `new_cases`: weekly sum
  - `new_deaths`: weekly sum
  - `cases`: weekly max cumulative value
  - `deaths`: weekly max cumulative value
- `WeekStartDate` and `WeekEndDate` are reconstructed from ISO week keys.
- Weekly outcome rates are:
  - `case_rate_100k = 100000 * new_cases / population`
  - `death_rate_100k = 100000 * new_deaths / population`

### 3. Vaccination data

- Bansal county rows are pivoted from `CASE_TYPE` into:
  - `complete_count`, `complete_cov`
  - `partial_count`, `partial_cov`
  - `booster_count`, `booster_cov`
- If multiple Bansal dates fall in the same county-week, the latest `source_date` is retained.
- CDC vaccination rows are converted to county-week by taking the last available row in each county/week and mapping CDC fields into the same canonical schema.
- The canonical nationwide vaccination table is an outer merge of CDC and Bansal on:
  - `fips`
  - `iso_year`
  - `iso_week`
  - `WeekStartDate`
  - `WeekEndDate`
- Field precedence is:
  - use CDC when present
  - otherwise use Bansal as a gap fill
- Row-level provenance is stored in `vaccination_source`:
  - `cdc`
  - `bansal_fill`
- The resolved table source reported in `processing_summary.json` is therefore `cdc_with_bansal_fill`.

### 4. Joined county-week panel

- The joined weekly panel is built as the full county-by-week cartesian product over the core date window, then left-joining:
  - weekly NYT outcomes
  - weekly vaccination fields
- The canonical join keys are:
  - `fips`
  - `iso_year`
  - `iso_week`
  - `WeekStartDate`
  - `WeekEndDate`
- Population is carried from the vaccination side when present and otherwise filled from the NYT-side population used in weekly rate construction.

### 5. Networks

- `contiguity` is built from TIGER county polygons in projected CRS `EPSG:5070` using polygon-touching adjacency.
- `knn_8` and `distance_kernel_8` are built from projected county centroids.
- The nationwide contiguity graph is not fully connected because of Alaska, Hawaii, islands, and territories.

### 6. Feature basis

- The intended county feature basis is:
  - `population_density`
  - `senior_population`
  - `college_education`
  - `poverty_rate`
  - `log_population`
  - `median_household_income`
  - `cdc_svi_2022_overall`
  - `usda_ers_rucc_2023`
- `population_density` is computed as `total_population / land_area_sq_km`, where `land_area_sq_km` comes from TIGER `ALAND`.
- `senior_population` is the ACS 2021 share of county residents age 65 and older.
- `college_education` is the ACS 2021 share of adults age 25 and older with a bachelor's degree or higher.
- `cdc_svi_2022_overall` is the CDC/ATSDR SVI 2022 overall county percentile ranking (`RPL_THEMES`), joined from the U.S. county file plus Puerto Rico county file.
- `usda_ers_rucc_2023` is the USDA ERS 2023 Rural-Urban Continuum Code, kept as one ordinal urbanicity feature.
- `log_population` is computed as `log1p(total_population)`.
- Continuous county-feature missingness is median-imputed columnwise; RUCC is mode-imputed.

## Current Processed Dataset

From `data/USCountyVaccination/processed/processing_summary.json`:

- `county_count_geometry = 3213`
- `county_count_weekly_panel = 3213`
- `state_count = 53`
- `daily_nyt_rows = 2,475,577`
- `weekly_nyt_rows = 228,123`
- `vaccination_weekly_rows = 228,123`
- `cdc_weekly_rows = 233,052`
- `bansal_weekly_rows = 207,104`
- `panel_rows = 228,123`
- `vaccination_observed_rows = 228,054`
- `booster_observed_rows = 73,772`
- `feature_basis_mode = acs_2021`

Network summary:

- `contiguity`: `3213` nodes, `9436` edges, `11` connected components
- `knn_8`: `3213` nodes, `14310` edges, `2` connected components
- `distance_kernel_8`: `3213` nodes, `14310` edges, `2` connected components

## Data Completeness

### Outcome availability

After rebuilding the binary panel from the current weekly panel, the active threshold outcomes are available for all `3213` counties over all `71` core weeks:

- `x_case_rate_100k_ge_100_pm1`: `228,123 / 228,123` rows
- `x_case_rate_100k_ge_200_pm1`: `228,123 / 228,123` rows
- `x_death_rate_100k_ge_2_pm1`: `228,123 / 228,123` rows

This means the current nationwide limitation is not on the NYT outcome side. It is on the vaccination side.

### Vaccination availability in the CDC-first filled panel

For `complete_cov`, the CDC-plus-Bansal-filled panel is almost complete but not perfectly complete:

- `228,054 / 228,123` county-weeks have non-missing `complete_cov`
- only `69` county-weeks are missing `complete_cov`

Those missing rows are concentrated in just `10` county-equivalent units:

- Hawaii counties: `5`
  - missing for the first `8` core weeks
- Texas counties `King` and `Loving`: `2`
  - each missing for `1` late week
- US Virgin Islands counties: `3`
  - missing for the final `9` core weeks

Weekly `complete_cov` availability in the filled panel:

- `2021-01-10` through `2021-02-28`: `3208 / 3213` counties observed
- `2021-03-07` through `2022-03-13`: `3213 / 3213` counties observed
- `2022-03-20`: `3208 / 3213` counties observed
- `2022-03-27` through `2022-05-15`: `3210 / 3213` counties observed

### CDC-first provenance and Bansal filler usage

The current canonical vaccination table is overwhelmingly CDC-native:

- `227,445` county-weeks come directly from CDC
- `678` county-weeks are filled from Bansal

The Bansal filler currently touches only `30` county-equivalent units:

- Virginia independent cities: `9`
- California counties: `8`
- Hawaii counties: `5`
- Massachusetts counties: `3`
- Virgin Islands counties: `3`
- Texas counties: `2`

Pure CDC support by itself would still be strong, but slightly smaller:

- CDC only, `0w` lag: best dense suffix is `3183` counties for `71` weeks
- CDC only, `1w` lag: best dense suffix is `3183` counties for `70` weeks
- CDC plus Bansal fill, `0w` lag: best dense suffix is `3203` counties for `71` weeks
- CDC plus Bansal fill, `1w` lag: best dense suffix is `3203` counties for `70` weeks

So after the precedence switch, Bansal is no longer doing most of the nationwide rescue work. It is now a relatively small filler layer that adds about `20` more complete-case counties to the core threshold support.

### Bansal-only completeness

Raw Bansal data are not complete on their own for a nationwide county-week panel.

Within the core window:

- Bansal weekly table contains `207,104` county-weeks
- Bansal covers `3142` counties at least once
- Bansal spans `51` state or territory codes in the raw weekly table
- Only `795` counties have all `71` core weeks in raw Bansal

Weekly Bansal county coverage ramps up sharply over early 2021:

- `2021-01-10`: `796` counties
- `2021-01-17`: `1365`
- `2021-01-24`: `1913`
- `2021-01-31`: `2455`
- `2021-02-07`: `2699`
- `2021-03-07`: `2858`
- `2022-05-15`: `3142`

Late in the window, raw Bansal is much denser but still not fully nationwide.

What a CDC-primary table changes relative to raw Bansal:

- At the start of the core window, the CDC-primary filled panel has `3208` observed `complete_cov` rows per week while raw Bansal has only `790`
- At the end of the core window, raw Bansal reaches `3132` rows per week while the CDC-primary filled panel retains `3210`

Complete-case support comparison for the `complete_cov >= 30` intervention against NYT outcomes:

- Bansal only, `0w` lag: best dense suffix is `2749` counties for `66` weeks starting `2021-02-14`
- Bansal only, `1w` lag: best dense suffix is `2751` counties for `65` weeks starting `2021-02-21`
- CDC plus Bansal fill, `0w` lag: best dense suffix is `3203` counties for `71` weeks starting `2021-01-10`
- CDC plus Bansal fill, `1w` lag: best dense suffix is `3203` counties for `70` weeks starting `2021-01-17`

So the nationwide threshold experiments remain close to fully national under the CDC-primary design, and Bansal now acts as a narrow backfill layer rather than the dominant source.

## Why The Core `1w` Experiment Has 70 Calendar Weeks

The current core `1w` smoke-test experiment is:

- outcome: `case_rate_100k >= 100`
- intervention: `complete_cov >= 30`
- lag: `1w`
- network: `contiguity`

Its realized metadata currently show:

- requested counties: `3213`
- realized counties: `3203`
- requested calendar weeks: `71`
- realized calendar weeks: `70`

The reason it uses `70` weeks rather than `71` is simple and mechanical:

- a `1w` intervention lag means the first requested week, `2021-01-10`, has no lagged intervention value by construction
- the realized dense suffix therefore starts at `2021-01-17`
- all later lags behave analogously:
  - `0w` keeps `71` weeks
  - `1w` keeps `70`
  - `2w` keeps `69`
  - `3w` keeps `68`
  - `4w` keeps `67`

The remaining loss is in counties, not weeks. The current `1w` core support excludes exactly these `10` counties:

- Hawaii: `15001`, `15003`, `15005`, `15007`, `15009`
- Texas: `48269`, `48301`
- Virgin Islands: `78010`, `78020`, `78030`

That is why the active filled-panel support is `3203` counties by `70` weeks for the `1w` core threshold experiment.

## ACS Covariates: Failure Mode And Fix

The earlier nationwide build fell back to `population_only` because the ACS fetch loop attempted the same county-level ACS calls for every state or territory FIPS in the county master table, including `78` for the US Virgin Islands.

Observed failure mode:

- All standard states, DC, and Puerto Rico returned normal JSON payloads.
- For state FIPS `78`, the ACS 2021 county endpoints returned HTTP `204` with an empty body.
- The original code called `response.json()` unconditionally.
- That empty response triggered:
  - `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
- The exception then caused a full fallback to `population_only`.

Current fix:

- The ACS fetch helper now treats empty or `204` responses as valid empty tables when expected columns are known.
- This allows the ACS build to succeed for the supported states and territories.
- The missing Virgin Islands county covariates are then median-imputed, just like any other residual missing ACS cells.

Result:

- `feature_basis_mode` is now `acs_2021`
- The experiment basis can now include:
  - `population_density`
  - `senior_population`
  - `college_education`
  - `poverty_rate`
  - `log_population`
  - `median_household_income`
  - `cdc_svi_2022_overall`
  - `usda_ers_rucc_2023`

In the current processed feature table, the three Virgin Islands rows are present in `acs_2021` mode and contain median-imputed ACS values rather than forcing the whole nationwide run back to `population_only`.

## Assumptions

- County identity is determined by 5-digit FIPS.
- ISO week keys are the canonical weekly alignment layer across NYT and vaccination sources.
- CDC is preferred whenever it has a county-week value; Bansal is used only as a fill source.
- Weekly experiment materialization is done on dense county-by-week suffixes after lagging so that saved `x` and `z` arrays contain no missing values.
- The current nationwide runner chooses the realized support that maximizes county-weeks, recorded in metadata as `max_complete_suffix_by_node_week_area`.

## Processed Outputs

Key outputs under `data/USCountyVaccination/processed/`:

- `us_county_daily_nyt.csv.gz`
- `us_county_weekly_nyt.csv.gz`
- `us_county_weekly_vaccination.csv.gz`
- `us_county_weekly_panel.csv.gz`
- `us_county_binary_panel.csv.gz`
- `us_county_binary_threshold_diagnostics.csv`
- `us_county_binary_threshold_diagnostics.md`
- `us_county_feature_basis.csv.gz`
- `us_county_feature_dictionary.csv`
- `us_counties.gpkg`
- `us_county_centroids.csv`
- `us_county_contiguity_adjacency.csv.gz`
- `us_county_knn_8_adjacency.csv.gz`
- `us_county_distance_kernel_8_adjacency.csv.gz`
- `processing_summary.json`

Experiment folders are written under `experiments/USCountyVaccination_US/`.
The nationwide runner now reads the processed binary panel, feature basis,
centroid table, and cached network edge lists directly; the GeoPackage is only
needed during preprocessing.

The current default intervention grid includes:

- complete vaccination thresholds: `10`, `30`, `40`, `50`, `60`, `70`, `80`
- first-dose thresholds: `10`, `30`, `40`, `50`, `60`, `70`, `80`

## Rebuild

Prepare processed data:

```bash
pixi run python data/USCountyVaccination/prepare_us_county_vaccination_data.py
```

Rebuild binary threshold columns:

```bash
pixi run python data/USCountyVaccination/build_binary_outcomes.py
```

Materialize the default nationwide threshold-only grid:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py
```

Materialize the mainland-only trimmed grid with `total_population >= 2000`:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed
```

Run MPLE while materializing:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py --run_mple
```

Write combined CSV and Markdown reports for one experiment root:

```bash
pixi run python data/USCountyVaccination/summarize_mple_experiments.py
```

Each materialized experiment folder now writes both `binary_definition_summary.csv`
and `binary_definition_summary.md`, including positive-share and transition-rate
diagnostics for the aligned outcome/intervention binary pair.

Smoke-test one current nationwide threshold specification:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --run_mple \
  --interventions complete_cov_ge_30 \
  --outcomes case_rate_100k_ge_100 \
  --lags 1w \
  --max_experiments 1 \
  --overwrite
```
