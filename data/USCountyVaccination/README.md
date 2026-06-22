# USCountyVaccination

Nationwide county-week data preparation and experiment materialization for the MPLE pipeline.

This workflow turns public US county COVID and vaccination sources into:

- processed weekly county tables
- binary threshold panels
- reusable realized outcome, intervention, and network artifacts
- shared-pipeline experiment folders
- a shared-compatible `generation_manifest.csv`
- MPLE fits and posterior-predictive or counterfactual summaries through the same top-level runners used by synthetic and hybrid experiments

## Overview

The US-county pipeline stages raw public data into experiment roots that match the same downstream contract used by synthetic and hybrid experiments. The active nationwide scope lives in `data/USCountyVaccination/common.py`, and the main materialization entrypoints are:

- `data/USCountyVaccination/load_raw_data.py`
- `data/USCountyVaccination/preprocess_us_county_vaccination_data.py`
- `data/USCountyVaccination/create_us_county_vaccination_experiments.py`

## Data

The loader and preprocessing scripts make a few pipeline-level assumptions so every realized artifact lands on a shared county-week support:

- Raw sources are NYT county COVID time series, CDC county vaccination snapshots, Bansal county vaccination time series, and Census TIGER county geometry.
- The modeled calendar is fixed to Sunday-ending ISO weeks from `2020-01-26` through `2022-05-15`.
- The working county set is the intersection of counties present in geometry, NYT outcomes, and the selected vaccination table.
- With the default `--vaccination_source cdc`, the pipeline does not use CDC alone. It builds a CDC weekly table and fills missing county-week vaccination values from the Bansal series.
- Weekly vaccination values are taken as the last available record within each county-week.
- Weekly case and death outcomes are aggregated from NYT daily rows and converted to per-100k rates using the county population lookup derived from the vaccination table.
- Binary outcomes and interventions use the shared `+1 / -1 / missing` convention. For interventions, weeks before a county's first observed vaccination report are explicitly filled with `-1` rather than left missing.
- `--trim` applies the mainland-population filter used by the committed USCounty experiments: keep mainland counties only and require `total_population >= 2000`.

The key data-selection flags are shared across `preprocess_us_county_vaccination_data.py` and `create_us_county_vaccination_experiments.py`, so the same subset should usually be passed to both commands:

- `--outcomes`: selects which binary outcome definitions to realize. Current built-in options are `case_rate_100k_ge_100`, `case_rate_100k_ge_200`, and `death_rate_100k_ge_2`.
- `--interventions`: selects which vaccination-threshold interventions to realize. Current committed USCounty workflows mostly use thresholded `complete_cov_*` definitions, but `partial_cov_*` definitions are also available.
- `--lags`: selects intervention lag codes such as `0w`, `1w`, or `2w`. In these experiments the lag is applied to the intervention only, not the outcome. Core vaccination interventions support `0w` through `4w`.
- `--networks`: selects the network family used when realizing shared artifacts and experiment folders. `contiguity` uses county boundary touch, `knn_8` uses 8-nearest centroid neighbors, and `distance_kernel_8` uses centroid-based distance-kernel weights over 8 neighbors.
- `--trim` or `--no-trim`: chooses between the trimmed mainland support and the full available county set. The committed revision specs assume trimmed support.
- `--vaccination_source`: chooses the vaccination input builder. `cdc` is the default and resolves to CDC with Bansal backfill; `bansal` uses the Bansal weekly series directly.

## Workflow

1. Load raw NYT, CDC, Bansal, and Census geography inputs with `load_raw_data.py`.
2. Preprocess county-week tables, threshold panels, realized artifacts, and shared panels with `preprocess_us_county_vaccination_data.py`.
3. Materialize shared-pipeline experiment folders and `generation_manifest.csv` with `create_us_county_vaccination_experiments.py`.
4. Run any downstream stages (MPLE fits, test set evaluations, counterfactual simulations, etc.) with the same shared runners used by synthetic and hybrid experiments (see [README.md](../../README.md)).

## Example Workflows

### Default Trimmed Pipeline

```bash
pixi run python data/USCountyVaccination/load_raw_data.py

pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --overwrite

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --overwrite
```

### Trimmed Start-Date Slice Matching The Existing USCounty Experiment

```bash
pixi run python data/USCountyVaccination/preprocess_us_county_vaccination_data.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_30 \
  --lags 2w \
  --networks distance_kernel_8 \
  --overwrite

pixi run python data/USCountyVaccination/create_us_county_vaccination_experiments.py \
  --trim \
  --output_root experiments/USCounty \
  --outcomes death_rate_100k_ge_2 \
  --interventions complete_cov_ge_30 \
  --lags 2w \
  --networks distance_kernel_8 \
  --start_dates 2020-03-01 \
  --overwrite
```

## Notes

- US-county experiments set `has_truth: false`, so `source_type=truth` posterior-predictive targets are intentionally unsupported.
- Each experiment root includes a zero `field_artifacts.npz` compatibility artifact so the shared loaders can treat real-data and synthetic roots uniformly.
- Downstream fitting, intervention-library construction, posterior predictive simulation, split construction, CV, and test evaluation all use the same shared runners as the synthetic and hybrid workflows (see [README.md](../../README.md)).
