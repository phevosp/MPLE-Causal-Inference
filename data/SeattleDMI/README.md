# SeattleDMI

This folder contains a local copy of the `seattledmi` dataset from the `microsynth` R package, plus Seattle/King County geographic files that let us map the block-level crime and demographic data to block geometries and a known network.

## Sources

- `microsynth` package data:
  - package source archive: `https://michaelwrobbins.r-universe.dev/src/contrib/microsynth_2.0.51.tar.gz`
  - dataset object: `seattledmi`
  - package paper: <https://www.jstatsoft.org/article/view/v097i02>
- Seattle geographic files:
  - 2010 King County census block shapefile bundle:
    `https://www.seattle.gov/documents/Departments/OPCD/Demographics/GeographicFilesandMaps/KingCountyBlocksShapefiles.zip`
  - Seattle census block to neighborhood correlation workbook:
    `https://www.seattle.gov/documents/Departments/OPCD/Demographics/GeographicFilesandMaps/SeattleCensusBlocksandNeighborhoodCorrelationFile.xlsx`
  - source landing page:
    <https://www.seattle.gov/opcd/population-and-demographics/geographic-files-and-maps#2010census>

## What The SeattleDMI Data Contains

The `seattledmi` dataset is a quarterly panel of Seattle census blocks used in the `microsynth` package to study a 2013 Drug Market Intervention in Seattle's International District.

Current realized counts in this folder:

- `154,272` panel rows
- `9,642` unique census blocks
- `16` quarterly time periods
- `39` treated blocks
- treated blocks switch from `Intervention = 0` to `Intervention = 1` at quarter `13`

The raw package documentation describes the main variables as:

- `ID`: census block identifier
- `time`: quarter index
- `Intervention`: treatment indicator
- crime outcomes such as `i_robbery`, `i_felony`, `i_drugs`, `any_crime`
- 2010 Census block-level demographic covariates such as `TotalPop`, `BLACK`, `HISPANIC`, `HOUSEHOLDS`, `RENTER_HOU`

## Join Logic

The key mapping between SeattleDMI and the geographic files is the 2010 census block identifier:

- `seattledmi.ID` from the microsynth dataset
- `GEOID10` in the King County block shapefile
- `GEOID10` in the Seattle block-to-neighborhood correlation workbook

In preprocessing, `ID` is converted to a zero-padded 15-digit string and saved as `GEOID10`.

Join coverage:

- all `9,642 / 9,642` SeattleDMI blocks match the King County block shapefile
- `9,641 / 9,642` SeattleDMI blocks match the Seattle neighborhood correlation workbook
- the one block missing neighborhood metadata is `530330264001005`

That means the geometry join is complete, and the neighborhood crosswalk is effectively complete except for one flagged block.

## Saved Layout

### Raw

- `raw/microsynth/microsynth_2.0.51.tar.gz`
  - downloaded package source archive
- `raw/microsynth/microsynth/data/seattledmi.rda`
  - extracted package dataset file
- `raw/microsynth/microsynth/man/seattledmi.Rd`
  - extracted package documentation
- `raw/geography/KingCountyBlocksShapefiles.zip`
  - downloaded Seattle/King County block shapefile bundle
- `raw/geography/KingCountyBlocksShapefiles/`
  - extracted shapefile components, including `kc_block_10.shp`
- `raw/geography/SeattleCensusBlocksandNeighborhoodCorrelationFile.xlsx`
  - downloaded correlation workbook

### Processed

- `processed/seattledmi_panel.csv.gz`
  - one row per `(block, quarter)` observation
  - includes `GEOID10`, `time`, `Intervention`, crime outcomes, and demographic covariates
- `processed/seattledmi_block_features.csv`
  - one row per block
  - contains static demographic covariates plus:
    - `treated_ever`
    - `intervention_start_time`
- `processed/seattledmi_block_preperiod_crime.csv`
  - one row per block
  - pre-intervention summaries for the crime outcomes
  - for each outcome, both `_pre_sum` and `_pre_mean` are included over quarters `1` through `12`
- `processed/seattledmi_binary_outcomes.csv.gz`
  - one row per `(block, quarter)` observation
  - currently includes the requested `{-1,+1}` outcomes, with:
    - `+1` = above-threshold outcome
    - `-1` = below-threshold outcome
    - `i_drugs_gt_0_pm1`
    - `any_crime_gt_0_pm1`
    - `any_crime_gt_1_pm1`
    - `any_crime_gt_2_pm1`
    - `any_crime_gt_3_pm1`
    - `any_crime_gt_district_mean_pm1`
    - `any_crime_gt_block_mean_pm1`
- `processed/seattledmi_binary_threshold_summary.csv`
  - threshold diagnostics for candidate binary outcome rules
- `processed/seattledmi_binary_threshold_summary.md`
  - short interpretation of the threshold diagnostics
- `processed/seattledmi_district_mean_thresholds.csv`
  - one row per neighborhood district
  - stores the district-level mean of `i_drugs` and `any_crime`
  - used to create district-relative binary thresholds
- `processed/seattledmi_block_mean_thresholds.csv`
  - one row per block
  - stores the block-specific mean of `i_drugs` and `any_crime`
  - used to create block-relative binary thresholds
- `processed/seattledmi_block_crosswalk.csv`
  - one row per block
  - non-spatial join table linking block IDs to tract/block-group identifiers, Seattle neighborhood metadata, and centroid coordinates
  - includes `has_neighborhood_match`
- `processed/seattledmi_block_centroids.csv`
  - one row per block
  - cached projected centroid coordinates used directly by the Seattle experiment runner
- `processed/seattledmi_blocks.gpkg`
  - GeoPackage with one geometry row per SeattleDMI block
  - contains the joined geometry plus static features and neighborhood metadata
  - layer name: `blocks`
- `processed/seattledmi_block_adjacency.csv.gz`
  - undirected edge list for a known network over SeattleDMI blocks
  - built using queen contiguity: two blocks share an edge if their polygons touch at any boundary point
- `processed/seattledmi_block_knn_8_adjacency.csv.gz`
- `processed/seattledmi_block_knn_16_adjacency.csv.gz`
  - cached undirected k-nearest-neighbor edge lists built from projected centroids
- `processed/seattledmi_block_distance_kernel_8_adjacency.csv.gz`
- `processed/seattledmi_block_distance_kernel_16_adjacency.csv.gz`
  - cached weighted centroid-distance-kernel edge lists on the same nearest-neighbor supports
- `processed/seattledmi_network_summary.csv`
  - node, edge, and connected-component counts for every saved Seattle network
- `processed/seattledmi_blocks_missing_correlation.csv`
  - the blocks that failed to match the neighborhood workbook
- `processed/processing_summary.json`
  - compact summary of row counts and join coverage

## How The Network Is Realized

The known network saved here is a geographic contiguity network over the SeattleDMI blocks:

- nodes: the `9,642` blocks in the SeattleDMI dataset
- edges: pairs of blocks whose census-block polygons touch
- saved as an edge list in `processed/seattledmi_block_adjacency.csv.gz`

This is a natural starting point for a known interaction graph because it depends only on public geography, not on outcomes.

The preprocessing step now writes the default contiguity, k-nearest-neighbor,
and centroid-distance-kernel edge lists up front, so the experiment runner can
reuse them directly instead of rebuilding them for every launch.

## Reproducibility

The processing script is:

- `prepare_seattledmi.py`
- `build_binary_outcomes.py`

Run it from the repo root with:

```bash
pixi run python data/SeattleDMI/prepare_seattledmi.py
```

It downloads the raw files if needed, extracts them, rebuilds the processed
tables, and rewrites the GeoPackage, centroid table, and cached Seattle network
edge lists.

To rebuild the binary outcome files and threshold diagnostics:

```bash
pixi run python data/SeattleDMI/build_binary_outcomes.py
```

The district-relative rule retained for experiments is:

- `any_crime_gt_district_mean_pm1`
- `any_crime_gt_block_mean_pm1`

This compares each block-quarter `any_crime` count to the mean level of `any_crime` in the block's Seattle neighborhood district.
The block-relative variant instead compares each block-quarter `any_crime` count to that exact block's own mean over time.
The analogous `i_drugs` district-mean rule is not kept in the runnable experiment set because it is effectively identical to `i_drugs_gt_0_pm1`.
The saved sign convention is:

- `+1`: above-threshold outcome
- `-1`: below-threshold outcome

## Real-Data MPLE Pipeline

The SeattleDMI folder also includes:

- `run_mple_pipeline.py`

This script builds full-network MPLE experiment folders under `experiments/SeattleDMI/` using:

- multiple binary outcomes from `processed/seattledmi_binary_outcomes.csv.gz`
- multiple known-network variants over the same `9,642` blocks
- an observed field basis built only from static SeattleDMI block covariates

Currently supported network variants are:

- `contiguity`
- `knn_8`
- `knn_16`
- `centroid_distance_kernel_8`
- `centroid_distance_kernel_16`

The centroid-distance kernel graphs are sparse kernels built from projected block-centroid distances and then restricted to the `k` nearest neighbors for scalability.

Each experiment folder contains:

- `panel_data.npz`
  - the observed `x` and `z` panel arrays in the same format used by `mple.py`
- `panel_data.csv.gz`
  - human-readable block-quarter panel for the chosen binary outcome and the observed intervention
- `x_0.npy`, `z_0.npy`
  - quarter-1 initial conditions
- `gamma_matrix_sparse.npz`
  - the normalized known network used for that experiment
- `interaction_basis_sparse.npz`
  - identical to the known network for the single-template interaction-basis case
- `field_basis.npy`
  - observed, infinity-normalized field templates
- `field_basis_names.npy`
- `interaction_basis_names.npy`
- `adjacency_edge_list.csv.gz`
  - the saved edge list actually used for the chosen Seattle network
- `node_index.csv`
  - mapping from node index back to `GEOID10` and neighborhood metadata
- `time_index.csv`
  - mapping from model time index back to original SeattleDMI quarter
- `realized_config.yaml`
- `experiment_metadata.yaml`
- `binary_definition_summary.csv`
- `binary_definition_summary.md`
  - positive-share and transition-rate diagnostics for the saved outcome and intervention binaries

### Current Field Basis

The external field is modeled as a linear combination of static block-level features only.
The current basis includes:

- `total_pop`
- `black_share`
- `hispanic_share`
- `male_1521_share`
- `family_household_share`
- `female_household_share`
- `renter_share`
- `vacant_share`

Each feature is centered across blocks and then normalized to infinity norm `1`.
No pre-intervention crime summaries are included in the current MPLE field basis.

For experiments that should use only the time-varying `tau_t` terms and no feature-based
external field, the builder script also supports `--field_basis_mode zero`. In that mode the
field basis is empty and the external field is driven entirely by the time-specific `tau_t`
terms.

### Current Outcome Construction

Each experiment folder uses exactly one binary outcome and exactly one fixed known network.
The pipeline can iterate over many outcomes or many network choices, but each saved folder corresponds to one specific pair.
The current experiment tree is grouped by mode under `experiments/SeattleDMI/`:

- `static/<outcome>__<network>/`
- `outcome_only/<outcome>__<network>/`
- `zero_basis/<outcome>__<network>/`
- `outcome_only_zero_basis/<outcome>__<network>/`

The default outcome options are:

- `i_drugs_gt_0_pm1`
- `any_crime_gt_0_pm1`
- `any_crime_gt_1_pm1`
- `any_crime_gt_2_pm1`
- `any_crime_gt_3_pm1`
- `any_crime_gt_district_mean_pm1`
- `any_crime_gt_block_mean_pm1`

The sign convention for `x` is:

- `+1`: above-threshold outcome, meaning the crime count is above the chosen threshold
- `-1`: below-threshold outcome, meaning the crime count is at or below the chosen threshold

### Current Network Construction

The node set is the full SeattleDMI block set of `9,642` census blocks.
All network variants are symmetrized, have zero diagonal, and are normalized to infinity norm `1`.

- `contiguity`
  - built from the saved queen-contiguity edge list in `processed/seattledmi_block_adjacency.csv.gz`
  - two blocks are adjacent when their polygons touch
- `knn_8` and `knn_16`
  - loaded from the cached projected-centroid edge lists in `processed/seattledmi_block_knn_*.csv.gz`
  - each block is connected to its `k` nearest neighbors, then the graph is symmetrized
- `centroid_distance_kernel_8` and `centroid_distance_kernel_16`
  - loaded from the cached projected-centroid kernel edge lists in `processed/seattledmi_block_distance_kernel_*.csv.gz`
  - edge weights are `exp(-distance / median_distance)`
  - the kernel is truncated to the `k` nearest neighbors for scalability, then symmetrized

### How One Experiment Folder Is Built

For one chosen outcome column and one chosen network:

1. The script loads `processed/seattledmi_binary_outcomes.csv.gz`, `processed/seattledmi_block_features.csv`, `processed/seattledmi_block_crosswalk.csv`, and `processed/seattledmi_block_centroids.csv`.
2. Blocks are sorted by `GEOID10` to create a fixed node ordering.
3. The selected binary outcome becomes the `x` panel.
4. `Intervention` is converted from `{0,1}` to `{-1,+1}` to create the `z` panel, where `-1` means no intervention and `+1` means intervention.
5. Quarter `1` is saved separately as `x_0.npy` and `z_0.npy`.
6. Quarters `2` through `16` are saved in `panel_data.npz` as the fitted panel.
7. The pre-intervention cutoff `s` is computed from the first quarter in which any block receives treatment.
8. The selected known network edge list is loaded from the processed cache and normalized.
9. The static field basis is built and normalized, or reduced to an empty basis if
   `--field_basis_mode zero` is used.
10. The folder is written under `experiments/SeattleDMI/<mode>/<outcome>__<network>/`.

The runner also supports the same optional `tau` controls used in the other
real-data pipelines:

- `--tau_zero_mean`
- `--tau_smoothness_lambda <float>`

### How MPLE Uses The Folder

`mple.py` expects the current panel and basis artifacts written by this repository.
For SeattleDMI folders it:

- loads `panel_data.npz`
- loads `gamma_matrix_sparse.npz`
- loads the saved field and interaction basis artifacts
- treats `experiment_metadata.yaml: has_truth: false` as a signal that no ground-truth parameter comparison is available
- writes estimated parameters, fitted interaction objects, and summary tables without truth-based recovery metrics

### Reporting Layout

The Seattle summary script now writes reports in the same grouped layout as the
experiment tree. If you point it at `experiments/SeattleDMI`, it writes:

- a combined report at `experiments/SeattleDMI/reports/`
- group-specific reports at:
  - `experiments/SeattleDMI/static/reports/`
  - `experiments/SeattleDMI/outcome_only/reports/`
  - `experiments/SeattleDMI/zero_basis/reports/`
  - `experiments/SeattleDMI/outcome_only_zero_basis/reports/`

By default, the report contains all experiment rows found under the Seattle root.
If you want a single custom output location, you can still pass `--output_csv`
and `--output_md` explicitly.

To build the experiment folders only:

```bash
pixi run python data/SeattleDMI/run_mple_pipeline.py
```

To build them and immediately fit MPLE:

```bash
pixi run python data/SeattleDMI/run_mple_pipeline.py --run_mple
```

To write the combined Seattle CSV and Markdown reports:

```bash
pixi run python data/SeattleDMI/summarize_mple_experiments.py
```

For a small smoke test:

```bash
pixi run python data/SeattleDMI/run_mple_pipeline.py \
  --outcomes any_crime_gt_0_pm1 \
  --networks contiguity \
  --output_root experiments/SeattleDMI_smoke \
  --manifest_path experiments/SeattleDMI_smoke/manifest.csv \
  --run_mple \
  --steps 25 \
  --overwrite
```

This will write the smoke experiment under `experiments/SeattleDMI_smoke/static/`.
