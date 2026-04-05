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
  - currently includes the requested `{-1,+1}` outcomes:
    - `i_drugs_gt_0_pm1`
    - `any_crime_gt_0_pm1`
- `processed/seattledmi_binary_threshold_summary.csv`
  - threshold diagnostics for candidate binary outcome rules
- `processed/seattledmi_binary_threshold_summary.md`
  - short interpretation of the threshold diagnostics
- `processed/seattledmi_district_mean_thresholds.csv`
  - one row per neighborhood district
  - stores the district-level mean of `i_drugs` and `any_crime`
  - used to create district-relative binary thresholds
- `processed/seattledmi_block_crosswalk.csv`
  - one row per block
  - non-spatial join table linking block IDs to tract/block-group identifiers, Seattle neighborhood metadata, and centroid coordinates
  - includes `has_neighborhood_match`
- `processed/seattledmi_blocks.gpkg`
  - GeoPackage with one geometry row per SeattleDMI block
  - contains the joined geometry plus static features and neighborhood metadata
  - layer name: `blocks`
- `processed/seattledmi_block_adjacency.csv.gz`
  - undirected edge list for a known network over SeattleDMI blocks
  - built using queen contiguity: two blocks share an edge if their polygons touch at any boundary point
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

If you want a different known network, the saved geometry file also supports alternatives such as:

- centroid-distance kernels
- k-nearest-neighbor graphs
- tract-restricted or neighborhood-restricted adjacency

## Reproducibility

The processing script is:

- `prepare_seattledmi.py`
- `build_binary_outcomes.py`

Run it from the repo root with:

```bash
pixi run python data/SeattleDMI/prepare_seattledmi.py
```

It downloads the raw files if needed, extracts them, rebuilds the processed tables, and rewrites the GeoPackage and adjacency edge list.

To rebuild the binary outcome files and threshold diagnostics:

```bash
pixi run python data/SeattleDMI/build_binary_outcomes.py
```

The district-relative rules created by that script are:

- `i_drugs_gt_district_mean_pm1`
- `any_crime_gt_district_mean_pm1`

These compare each block-quarter count to the mean level of that outcome in the block's Seattle neighborhood district.
