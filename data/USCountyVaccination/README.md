# USCountyVaccination

Nationwide county-week data preparation and experiment materialization for the MPLE pipeline.

This workflow turns public US county COVID and vaccination sources into:

- processed weekly county tables
- binary threshold panels
- reusable realized outcome, intervention, and network artifacts
- experiment folders under `experiments/USCountyVaccination_US/`
- optional MPLE fits and summary reports

## Current Scope

The active nationwide scope is defined in `data/USCountyVaccination/common.py`.

Core window:

- `WeekEndDate` from `2020-01-26` through `2022-05-15`

Default outcome definitions:

- `case_rate_100k >= 100`
- `case_rate_100k >= 200`
- `death_rate_100k >= 2`

Default intervention definitions:

- complete coverage thresholds: `10`, `20`, `30`, `40`, `50`, `60`, `70`, `80`
- partial coverage thresholds: `10`, `20`, `30`, `40`, `50`, `60`, `70`, `80`

Default lag grids:

- core interventions: `0w`, `1w`, `2w`, `3w`, `4w`
- booster interventions: currently disabled in the default grid

Network artifacts built during preprocessing:

- `contiguity`
- `knn_8`
- `distance_kernel_8`

Default network materialized by the experiment runner:

- `contiguity`

Default experiment root:

- `experiments/USCountyVaccination_US`

Optional trim mode:

- mainland US counties only
- `total_population >= 2000`

## Data Sources

- New York Times county COVID archive
- CDC county vaccination dataset `8xkx-amqh`
- Bansal Lab county vaccination time series used as a fill source in the default CDC-first workflow
- Census TIGER county shapefiles
- Census ACS 2021 county APIs
- CDC/ATSDR SVI 2022 county files
- USDA ERS RUCC 2023 county file

The concrete URLs live in `data/USCountyVaccination/common.py`.

## Workflow

### 1. Prepare Processed County-Week Tables

Entry point:

- `data/USCountyVaccination/prepare_us_county_vaccination_data.py`

Default command:

```bash
pixi run python data/USCountyVaccination/prepare_us_county_vaccination_data.py
```

Default behavior:

- downloads or reuses raw source files
- builds a county master table from the overlap of geometry, NYT outcomes, and vaccination support
- aggregates NYT daily data to county-week outcomes
- normalizes CDC and Bansal vaccination data to county-week format
- uses CDC as the primary vaccination source and fills remaining gaps from Bansal
- builds county feature tables
- builds geometry-derived networks and centroids
- writes processed outputs under `data/USCountyVaccination/processed/`

Important flags:

- `--vaccination_source cdc|bansal`
- `--reuse_processed_tables`
- `--reuse_processed_networks`
- `--reuse_processed_features`

Notes:

- `--vaccination_source cdc` is the default and produces a CDC-first table with Bansal fill
- `--vaccination_source bansal` uses raw Bansal weekly data as the canonical vaccination source instead

### 2. Build Binary Threshold Columns

Entry point:

- `data/USCountyVaccination/build_binary_outcomes.py`

Command:

```bash
pixi run python data/USCountyVaccination/build_binary_outcomes.py
```

This step:

- reads `processed/us_county_weekly_panel.csv.gz`
- writes `processed/us_county_binary_panel.csv.gz`
- writes threshold diagnostics as CSV and Markdown

Binary semantics:

- `+1` means above threshold
- `-1` means below threshold
- intervention columns are filled with `-1` before first observed reporting for that county

### 3. Optional Descriptive Analysis

Descriptive summary for the currently highlighted threshold variables:

```bash
pixi run python data/USCountyVaccination/create_data_analysis_summary.py
```

Pre-vaccination low-rank diagnostics:

```bash
pixi run python data/USCountyVaccination/analyze_pre_vaccination_low_rank.py
```

The analysis outputs are written under:

- `data/USCountyVaccination/data_analysis/`

### 4. Materialize Experiment Folders

Entry point:

- `data/USCountyVaccination/run_us_county_vaccination_experiments.py`

Default command:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py
```

This runner:

- reads the processed binary panel, feature basis, centroid table, and cached edge lists
- writes reusable realized artifacts for outcomes, interventions, and networks
- selects a dense county-by-week suffix after lagging so saved experiment panels contain no missing `x` or `z`
- writes reusable shared panels
- writes one experiment folder per `(outcome, intervention, lag, network)` combination
- optionally runs `mple.py` for each experiment
- records a manifest at `experiments/USCountyVaccination_US/manifest.csv`

The support-selection rule recorded in metadata is:

- `max_complete_suffix_by_node_week_area`

## Experiment Layout

The runner writes these reusable artifact roots beneath the output root:

- `realized_outcomes/`
- `realized_interventions/`
- `realized_networks/`
- `shared_panels/`

Each experiment folder contains:

- `realized_config.yaml`
- `experiment_metadata.yaml`
- `field_artifacts.npz`
- `gamma_matrix_sparse.npz`
- `adjacency_edge_list.csv.gz`
- `binary_definition_summary.csv`
- `binary_definition_summary.md`

When a shared panel exists, metadata points to:

- `shared_panel_path`
- `shared_x0_path`
- `shared_z0_path`
- `shared_node_index_path`
- `shared_time_index_path`

Shared panel folders contain:

- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`
- `panel_data.csv.gz`
- `node_index.csv`
- `time_index.csv`
- `panel_metadata.yaml`

## Field Modes

`run_us_county_vaccination_experiments.py` supports two field parameterizations:

- `--field_mode additive`
- `--field_mode latent_feature_matrix`

`additive`:

- uses the processed county feature basis
- writes a standard `field_artifacts.npz` backed by the feature basis

`latent_feature_matrix`:

- disables the additive county basis
- fits a low-rank latent field instead
- uses `--latent_rank`
- uses `--latent_B` as the global bound `B`

## Runner Flags

Useful experiment-runner flags:

- `--run_mple`
- `--overwrite`
- `--steps`
- `--tol`
- `--seed`
- `--max_experiments`
- `--lags`
- `--outcomes`
- `--interventions`
- `--networks`
- `--output_root`
- `--trim`
- `--field_mode`
- `--latent_rank`
- `--latent_B`
- `--beta_mask_pre_intervention`
- `--beta_mask_rescale`

Two additional flags are currently passed through into saved configs and metadata:

- `--tau_zero_mean`
- `--tau_smoothness_lambda`

At the moment those `tau_*` settings are recorded by the real-data runner, but the current top-level `mple.py` path does not consume them.

## Fitting Behavior

When `--run_mple` is enabled, the runner first attempts the full outcome-plus-intervention fit.

If that full fit fails, it automatically retries with:

- `mple.py --outcome_only`

The manifest records:

- `full_fit_status`
- `outcome_only_fit_status`
- `fallback_run`

`fallback_run = true` means the full fit failed and the experiment was rerun outcome-only.

## Processed Outputs

Key processed files under `data/USCountyVaccination/processed/`:

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
- `us_county_network_summary.csv`
- `processing_summary.json`

For the current snapshot, the most reliable machine-readable summaries are:

- `processed/processing_summary.json`
- `processed/us_county_network_summary.csv`
- `data_analysis/summary.md`

## Common Commands

Prepare processed data:

```bash
pixi run python data/USCountyVaccination/prepare_us_county_vaccination_data.py
```

Build binary threshold columns:

```bash
pixi run python data/USCountyVaccination/build_binary_outcomes.py
```

Create the descriptive analysis summary:

```bash
pixi run python data/USCountyVaccination/create_data_analysis_summary.py
```

Run the pre-vaccination low-rank analysis on the full county set:

```bash
pixi run python data/USCountyVaccination/analyze_pre_vaccination_low_rank.py
```

Run the same low-rank analysis on the trimmed county set:

```bash
pixi run python data/USCountyVaccination/analyze_pre_vaccination_low_rank.py \
  --node_index_path experiments/USCountyVaccination_US/realized_outcomes/outcome_death_rate_100k_ge_2__scope_trimmed/node_index.csv \
  --scope_label trimmed \
  --output_dir data/USCountyVaccination/data_analysis/pre_vaccination_low_rank_trimmed
```

Materialize the default experiment grid:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py
```

Materialize the trimmed grid:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed
```

Run a small subset with MPLE:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --run_mple \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_20 \
  --lags 2w \
  --max_experiments 1 \
  --overwrite
```

Run a latent-field fit:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --trim \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_20 \
  --lags 2w \
  --max_experiments 1 \
  --field_mode latent_feature_matrix \
  --latent_rank 6 \
  --latent_B 1.5 \
  --beta_mask_pre_intervention \
  --beta_mask_rescale \
  --run_mple \
  --overwrite
```

Materialize multiple network variants:

```bash
pixi run python data/USCountyVaccination/run_us_county_vaccination_experiments.py \
  --networks contiguity knn_8 distance_kernel_8 \
  --max_experiments 3
```

Write combined MPLE reports for an experiment root:

```bash
pixi run python data/USCountyVaccination/summarize_mple_experiments.py
```

Or point the summarizer at a different experiment root:

```bash
pixi run python data/USCountyVaccination/summarize_mple_experiments.py \
  --experiments_root experiments/USCountyVaccination_US_trimmed
```

## Interpretation Notes

- The canonical weekly alignment layer is ISO year/week with explicit `WeekStartDate` and `WeekEndDate`.
- The default nationwide vaccination table is CDC-first with Bansal fill, not Bansal-only.
- Realized experiment support is chosen after lagging so saved `x`, `z`, `x_0`, and `z_0` contain no missing values.
- Preprocessing builds all three network artifacts, but the experiment runner defaults to `contiguity` unless `--networks` is overridden.
