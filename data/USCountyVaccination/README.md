# USCountyVaccination

Nationwide county-week data preparation and experiment materialization for the MPLE pipeline.

This workflow turns public US county COVID and vaccination sources into:

- processed weekly county tables
- binary threshold panels
- reusable realized outcome, intervention, and network artifacts
- shared-pipeline experiment folders under `experiments/USCountyVaccination_US_trimmed/`
- a shared-compatible `generation_manifest.csv`
- reusable counterfactual intervention libraries
- MPLE fits and posterior-predictive counterfactual summaries through the same top-level runners used by synthetic and hybrid experiments

## Current Scope

The active nationwide scope is defined in `data/USCountyVaccination/common.py`.

Core window:

- `WeekEndDate` from `2020-01-26` through `2022-05-15`

Available outcome definitions:

- `case_rate_100k >= 100`
- `case_rate_100k >= 200`
- `death_rate_100k >= 2`

Default v1 experiment outcome:

- `death_rate_100k_ge_2`

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

- `experiments/USCountyVaccination_US_trimmed`

Default county scope:

- mainland US counties only
- `total_population >= 2000`

Use `--no-trim` to materialize the full county support instead.

## Data Sources

- New York Times county COVID archive
- CDC county vaccination dataset `8xkx-amqh`
- Bansal Lab county vaccination time series used as a fill source in the default CDC-first workflow
- Census TIGER county shapefiles

The concrete URLs live in `data/USCountyVaccination/common.py`.

## Workflow

The USCountyVaccination path is now explicitly staged. Each stage writes artifacts consumed by the next stage; MPLE fitting and counterfactual simulation use the same top-level shared runners as the synthetic/hybrid workflow.

Implementation helpers are intentionally not runnable pipeline stages:

- `common.py` stores USCounty constants, paths, specs, naming helpers, and small shared utilities.
- `processed_data.py` builds processed county-week tables, network source tables, and binary `-1/+1` threshold panels.
- `experiment_artifacts.py` builds/loads realized artifacts, shared panels, and final shared-pipeline experiment folders.

### 1. Load Raw Data

Entry point:

- `data/USCountyVaccination/load_raw_data.py`

Default command:

```bash
pixi run python data/USCountyVaccination/load_raw_data.py
```

This downloads or reuses raw NYT, CDC, Bansal, and Census TIGER files under `data/USCountyVaccination/raw/`.

### 2. Preprocess And Realize Artifacts

Entry point:

- `data/USCountyVaccination/preprocess_us_county_vaccination_data.py`

Default command:

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

This step:

- builds processed county-week tables under `data/USCountyVaccination/processed/`
- builds binary threshold columns with `+1` above threshold and `-1` below threshold
- writes threshold diagnostics as CSV and Markdown
- writes `realized_outcomes/`
- writes `realized_interventions/`
- writes `realized_networks/`
- writes `shared_panels/`

Important flags:

- `--vaccination_source cdc|bansal`
- `--reuse_processed_tables`
- `--reuse_processed_networks`
- `--outcomes`
- `--interventions`
- `--lags`
- `--networks`
- `--trim | --no-trim`

### 3. Create Experiment Folders

Entry point:

- `data/USCountyVaccination/create_us_county_vaccination_experiments.py`

Default command:

```bash
pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

This step loads saved realized artifacts and shared panels, then writes one experiment folder per `(outcome, intervention, lag, network)` combination. It records the shared pipeline manifest:

- `experiments/USCountyVaccination_US_trimmed/generation_manifest.csv`

The support-selection rule recorded in metadata is:

- `max_complete_suffix_by_node_week_area`

Optional start-date slicing is built into this step. Pass one or more `--start_dates YYYY-MM-DD` values to materialize only sliced experiments. Each requested date resolves to the first available modeled `WeekEndDate >= requested date`, and the resulting experiment names are suffixed as `__start_YYYY_MM_DD`.

Example:

```bash
pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --start_dates 2020-09-06 2021-01-03 \
  --overwrite
```

### 4. Run MPLE Fits

Use the shared fit runner:

```bash
GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

### 5. Counterfactual Pipeline

Build intervention scenarios:

```bash
pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite
```

Run counterfactual posterior predictive simulation:

```bash
GEN_MANIFEST=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FIT_MANIFEST=experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
TARGET_PAIRS_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```

### Optional Descriptive Analysis

Analysis entrypoints now live under `data/USCountyVaccination/data_analysis/`:

```bash
pixi run python data/USCountyVaccination/data_analysis/create_data_analysis_summary.py
pixi run python data/USCountyVaccination/data_analysis/analyze_pre_vaccination_low_rank.py
```

## Experiment Layout

The preprocessing/realization stage writes these reusable artifact roots beneath the output root:

- `realized_outcomes/`
- `realized_interventions/`
- `realized_networks/`
- `shared_panels/`

Each experiment folder contains:

- `realized_config.yaml`
- `experiment_metadata.yaml`
- `panel_data.npz`
- `x_0.npy`
- `z_0.npy`
- `node_index.csv`
- `time_index.csv`
- `field_artifacts.npz`
- `gamma_matrix_sparse.npz`
- `adjacency_edge_list.csv.gz`
- `panel_data.csv.gz`
- `binary_definition_summary.csv`
- `binary_definition_summary.md`

USCountyVaccination experiment metadata records `has_truth: false`. The root-level `field_artifacts.npz` is a concrete zero field placeholder with the correct `(T, N)` shape so the shared loaders can treat real-data and synthetic experiment roots uniformly. Fit-time latent rank and bounds are controlled by the fit spec, not by synthetic truth artifacts.

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

The duplicated root-level artifacts are the inputs consumed by:

- `run_fit_pipeline.py`
- `run_intervention_library.py`
- `run_posterior_predictive.py`
- `report_posterior_predictive.py`

USCountyVaccination materialization does not choose a field parameterization. It always writes the same real-data experiment contract with `has_truth: false` and a zero `field_artifacts.npz` compatibility placeholder. For the shared manifest workflow, edit `data/USCountyVaccination/experiment_configs/fits_spec.yaml` to choose MPLE variants such as latent rank, optimizer mode, and optional downstream fit settings like `estimation.beta_mask_pre_s`.

## Runner Flags

Useful `create_us_county_vaccination_experiments.py` flags:

- `--overwrite`
- `--max_experiments`
- `--lags`
- `--outcomes`
- `--interventions`
- `--networks`
- `--output_root`
- `--start_dates`
- `--trim`

## Fitting Behavior

When `--run_mple` is enabled, the runner fits the outcome pseudo-likelihood only.

## Shared MPLE And Counterfactual Workflow

The preferred real-data workflow is now the same as the synthetic/hybrid workflow, using the generated `generation_manifest.csv`.

Fit MPLE variants:

```bash
GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

Build reusable intervention scenarios:

```bash
pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite
```

Run fit-based counterfactual posterior predictive simulation:

```bash
GEN_MANIFEST=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FIT_MANIFEST=experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
TARGET_PAIRS_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```

The included intervention-library template materializes:

- `observed_experiment`
- `all_minus_ones`
- `all_ones`
- `all_ones_from_s`
- `single_unit_0_all_ones`

Saved interventions use the model's internal `-1/+1` coding. Posterior predictive targets for USCountyVaccination should use `source_type=fit`; `source_type=truth` is rejected because real-data experiments set `has_truth: false`.

## Optional Start-Date Slicing

If you want to run the applied pipeline from later calendar weeks, create sliced experiment roots directly during materialization with `--start_dates` and then use the normal shared fit/intervention/posterior-predictive stages on the resulting `generation_manifest.csv`.

Slicing behavior:

- requested dates are interpreted as ISO `YYYY-MM-DD`
- each date resolves forward to the first available modeled `WeekEndDate >= requested date`
- slicing fails if the request is after the final available modeled week
- slicing also fails if it would leave zero transition weeks to fit
- sliced experiment names are suffixed as `__start_YYYY_MM_DD`

Each sliced experiment records:

- `requested_start_date`
- `resolved_start_week_end_date`
- `start_index`
- `dropped_transition_weeks_for_start`

This keeps start-date control inside the standard US county materialization path instead of using a separate sensitivity runner.

## Processed Outputs

Key processed files under `data/USCountyVaccination/processed/`:

- `us_county_daily_nyt.csv.gz`
- `us_county_weekly_nyt.csv.gz`
- `us_county_weekly_vaccination.csv.gz`
- `us_county_weekly_panel.csv.gz`
- `us_county_binary_panel.csv.gz`
- `us_county_binary_threshold_diagnostics.csv`
- `us_county_node_geography.csv.gz`
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

Load raw data:

```bash
pixi run python data/USCountyVaccination/load_raw_data.py
```

Preprocess and materialize realized artifacts/shared panels:

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

Create the descriptive analysis summary:

```bash
pixi run python data/USCountyVaccination/data_analysis/create_data_analysis_summary.py
```

Run the pre-vaccination low-rank analysis on the full county set:

```bash
pixi run python data/USCountyVaccination/data_analysis/analyze_pre_vaccination_low_rank.py
```

Run the same low-rank analysis on the trimmed county set:

```bash
pixi run python data/USCountyVaccination/data_analysis/analyze_pre_vaccination_low_rank.py \
  --node_index_path experiments/USCountyVaccination_US_trimmed/realized_outcomes/outcome_death_rate_100k_ge_2__scope_trimmed/node_index.csv \
  --scope_label trimmed \
  --output_dir data/USCountyVaccination/data_analysis/pre_vaccination_low_rank_trimmed
```

Create the default trimmed death-rate/vaccine-rate experiment folders:

```bash
pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCountyVaccination_US_trimmed \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

Create the full county-support experiment folders:

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --no-trim \
  --output_root experiments/USCountyVaccination_US_full \
  --outcomes death_rate_100k_ge_2 \
  --overwrite

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --no-trim \
  --output_root experiments/USCountyVaccination_US_full \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

Create a small subset and fit it through the shared MPLE runner:

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_20 \
  --lags 2w \
  --max_experiments 1 \
  --overwrite

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_20 \
  --lags 2w \
  --max_experiments 1 \
  --overwrite

GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

Run the shared fit, intervention-library, and counterfactual workflow:

```bash
GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh

pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

GEN_MANIFEST=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FIT_MANIFEST=experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
TARGET_PAIRS_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```

Materialize multiple network variants into realized artifacts and experiment folders:

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --networks contiguity knn_8 distance_kernel_8 \
  --max_experiments 3

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --networks contiguity knn_8 distance_kernel_8 \
  --max_experiments 3
```

Shared MPLE fitting writes per-experiment fit artifacts and updates the root-level `fit_manifest.csv`:

```bash
GENERATION_MANIFEST_PATH=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FITS_SPEC_PATH=data/USCountyVaccination/experiment_configs/fits_spec.yaml \
FIT_OVERWRITE=true \
bash submit_fit_jobs.sh
```

For counterfactual summaries, build intervention-library entries and run the shared posterior-predictive runner:

```bash
pixi run python run_intervention_library.py \
  --generation_manifest_path experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
  --spec_path data/USCountyVaccination/experiment_configs/intervention_library_spec.yaml \
  --overwrite

GEN_MANIFEST=experiments/USCountyVaccination_US_trimmed/generation_manifest.csv \
FIT_MANIFEST=experiments/USCountyVaccination_US_trimmed/fit_manifest.csv \
TARGET_PAIRS_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_target_pairs.csv \
POSTERIOR_PREDICTIVE_SPEC_PATH=data/USCountyVaccination/experiment_configs/posterior_predictive_spec.yaml \
bash submit_posterior_predictive_jobs.sh
```

## Interpretation Notes

- The canonical weekly alignment layer is ISO year/week with explicit `WeekStartDate` and `WeekEndDate`.
- The default nationwide vaccination table is CDC-first with Bansal fill, not Bansal-only.
- Realized experiment support is chosen after lagging so saved `x`, `z`, `x_0`, and `z_0` contain no missing values.
- Preprocessing builds all three network artifacts, but the experiment runner defaults to `contiguity` unless `--networks` is overridden.
