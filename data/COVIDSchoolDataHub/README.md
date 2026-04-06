# COVIDSchoolDataHub

This folder contains a local data package built from the COVID School Data Hub district learning-model files, the COVID School Data Hub monthly in-person share files, the COVID School Data Hub community case-rate data, and school-district geography for Ohio and Massachusetts. The package is organized so we can later construct state-level MPLE experiments with:

- observed district learning-mode interventions
- observed district monthly in-person-share interventions
- district-week COVID community outcomes
- multiple known-network options derived from public school-district geography

The layout mirrors the structure used in `data/SeattleDMI/`:

- `raw/` stores the downloaded source files unchanged
- `processed/` stores cleaned tables, crosswalks, geometry-backed district files, and adjacency-ready network artifacts

The current raw files in this folder were downloaded and processed on `2026-04-06`.

## Sources

### COVID School Data Hub

- data-resources landing page:
  <https://www.covidschooldatahub.com/data-resources>
- district learning-model files:
  <https://assets.ctfassets.net/9fbw4onh0qc1/3JXV9ahOubLLnh9aHTHgKv/6e3c8a2baf1f2e0517edd9e454ee5c74/CSDH_District_Files_-_CSV.zip>
- district monthly in-person shares:
  <https://assets.ctfassets.net/9fbw4onh0qc1/4LRV2nKQOBoCudvyVxLx6w/fdb67c6da6252d520b83d173f3a41237/District_Monthly_Shares_03.08.23.csv>
- community case-rate data by district:
  <https://downloads.ctfassets.net/9fbw4onh0qc1/1FyYF7Qqmn2fXfWYqcqZUB/d2f9ec9d4a78bdedbc93869396393c09/Matched_Districts_and_Case_Rates.zip>
- community case-rate codebook:
  <https://assets.ctfassets.net/9fbw4onh0qc1/3vad828a7tYRJ2F7Qeqfbh/1c4612a5918ac618e9b05eeae46344ee/Cate_Rate_Codebook.xlsx>
- NCES CCD district demographics, 2020-21:
  <https://assets.ctfassets.net/9fbw4onh0qc1/6lSX82GvL9tPRpSE9VNkB/24cd9aea4dc7af91be9344c3ccd661f0/NCES_2020-2021_District_Demographics.csv>
- district merge-code file:
  <https://assets.ctfassets.net/9fbw4onh0qc1/1lKC4sytDUc3laSmTxhSFl/72189d83203a5ce045f1cf4739914469/district_code_to_combine_learning_model.do>

### Massachusetts geography

- official geometry item:
  <https://www.arcgis.com/home/item.html?id=145c945f4fa744e8951c47b696c73758>
- official feature service:
  <https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/AGOL/Public_School_Districts/FeatureServer>

The script downloads and combines the first three official layers:

- `K-12 School Districts`
- `Secondary School Districts`
- `Elementary School Districts`

These layers expose `ORG4CODE`, `ORG8CODE`, and `DISTRICT_NAME`, which are the key fields used for the Massachusetts official crosswalk.

### Ohio geography

- official feature service:
  <https://maps.ohio.gov/arcgis/rest/services/Hosted/Ohio_School_Districts_2025/FeatureServer/0>

The Ohio official layer exposes:

- `irn`
- `taxid`
- `name`

The crosswalk uses `irn` as the primary join field.

### Standardized EDGE school district boundaries

- NCES EDGE school district boundaries:
  <https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries>
- EDGE composite zip used by this package:
  <https://nces.ed.gov/programs/edge/data/EDGE_SCHOOLDISTRICT_TL21_SY2021.zip>

This is the canonical standardized geometry source for the package.

## What The Data Contains

The two core CSDH resources used here are:

- district learning-model periods:
  one row per district-period, with district identifiers, dates, and learning-model labels
- district monthly in-person shares:
  one row per district-month, with `share_inperson`, `share_hybrid`, and `share_virtual`
- community case-rate records:
  one row per district-week-ZIP, with weekly community COVID case-rate fields assigned to ZIPs attached to each NCES district ID

Current realized counts in this package:

- `173,346` district learning-period rows across `6,791` districts in `24` learning-data states
- `1,127,802` district-week-ZIP case-rate rows across `19,786` districts in `51` case-data jurisdictions
- `280,562` joined district-week-ZIP rows after matching learning periods to overlapping case-rate weeks
- `1,127,802` joined district-week-ZIP rows after matching monthly in-person shares to weekly case-rate rows

Ohio realized counts:

- `25,578` learning rows across `609` districts
- `60,078` case-rate rows across `1,054` districts
- `26,187` joined rows across the same `609` learning districts
- learning date range: `2020-08-02` to `2021-05-22`
- joined date range: `2020-07-27` to `2021-05-23`

Massachusetts realized counts:

- `7,141` learning rows across `421` districts
- `24,624` case-rate rows across `432` districts
- `15,548` joined rows across the same `421` learning districts
- learning date range: `2020-09-03` to `2021-05-26`
- joined date range: `2020-08-31` to `2021-05-30`

## Join Logic

### Learning-model processing

The district learning-model zip contains one CSV per state. The preprocessing script:

- loads every district-level CSV in the archive
- keeps district rows only
- standardizes:
  - `NCESDistrictID`
  - `StateAssignedDistrictID`
  - `PeriodStartDate`
  - `PeriodEndDate`
- preserves the original learning-model columns, including grade-specific learning-model fields where present

### Community case-rate processing

The community case-rate file is stored in one pipe-delimited CSV. The preprocessing script:

- loads the full district-week-ZIP file with the correct `|` delimiter
- standardizes:
  - `NCESDistrictID` from `leaid`
  - `StateAssignedDistrictID` from `state_leaid`
  - `zip`
  - `WeekStartDate`
  - `WeekEndDate`
- preserves the case-rate, testing, and ZIP-allocation fields from the source

### Monthly in-person-share processing

The district monthly-share file contains one row per district-month. The preprocessing script:

- loads the CSDH monthly in-person share CSV
- standardizes:
  - `NCESDistrictID`
  - `StateAbbrev`
  - `Month`
  - `MonthStartDate`
  - `MonthEndDate`
- preserves the monthly intervention-share fields:
  - `share_inperson`
  - `share_hybrid`
  - `share_virtual`
- joins those shares onto the weekly case-rate rows using the month of each `WeekStartDate`

### Learning-to-case join

The joined table is built at the most detailed case-data level: district-week-ZIP.

For each district learning period, the script:

- matches to the case-rate week calendar within the same state
- keeps weeks whose date interval overlaps the learning period
- resolves overlap ambiguity by assigning each district-week to the learning period with the largest number of overlapping calendar days
- merges that district-week assignment onto the district-week-ZIP case-rate rows

This produces a clean district-week-ZIP panel with one learning-model assignment per district-week.

### Geometry crosswalks

The primary district key throughout the package is `NCESDistrictID`.

Massachusetts official crosswalk:

- primary rule:
  `StateAssignedDistrictID` zero-padded to 8 digits and matched to `ORG8CODE`
- fallback:
  normalized district-name matching where the official name is unique

Ohio official crosswalk:

- primary rule:
  `StateAssignedDistrictID` zero-padded to 6 digits and matched to official `irn`
- fallback:
  normalized district-name matching where the official name is unique

Standardized EDGE crosswalk:

- direct match on `NCESDistrictID`

## Geometry Coverage

Ohio has complete coverage under both geometry options:

- official geometry match: `609 / 609` districts (`100%`)
- standardized geometry match: `609 / 609` districts (`100%`)

Massachusetts has partial coverage under both geometry options:

- official geometry match: `290 / 421` districts (`68.9%`)
- standardized geometry match: `290 / 421` districts (`68.9%`)

The unmatched Massachusetts districts are overwhelmingly charter-style districts rather than municipal or regional school districts. Examples include:

- `Academy Of the Pacific Rim Charter Public (District)`
- `Benjamin Banneker Charter Public (District)`
- `Cape Cod Lighthouse Charter (District)`
- `Innovation Academy Charter (District)`

So the current Massachusetts geometry package is strong for municipal and regional districts, but it does not fully cover charters.

## Saved Layout

### Raw

- `raw/csdh/CSDH_District_Files_-_CSV.zip`
  - district learning-model zip from COVID School Data Hub
- `raw/csdh/Matched_Districts_and_Case_Rates.zip`
  - community case-rate zip from COVID School Data Hub
- `raw/csdh/Cate_Rate_Codebook.xlsx`
  - case-rate codebook
- `raw/csdh/NCES_2020-2021_District_Demographics.csv`
  - NCES district master file
- `raw/csdh/district_code_to_combine_learning_model.do`
  - CSDH merge-code file
- `raw/geography/massachusetts/massachusetts_public_school_districts_k12.geojson`
- `raw/geography/massachusetts/massachusetts_public_school_districts_secondary.geojson`
- `raw/geography/massachusetts/massachusetts_public_school_districts_elementary.geojson`
  - official Massachusetts district geometry downloads
- `raw/geography/ohio/ohio_school_districts_2025.geojson`
  - official Ohio district geometry download
- `raw/geography/edge/EDGE_SCHOOLDISTRICT_TL21_SY2021.zip`
  - standardized EDGE school-district boundary zip used to create the alternate geometry option

### Processed

- `processed/csdh_district_learning_periods.csv.gz`
  - cleaned national learning-model table
- `processed/csdh_district_monthly_shares.csv.gz`
  - cleaned national monthly share table
- `processed/csdh_case_rates_by_district_zip_week.csv.gz`
  - cleaned national district-week-ZIP case-rate table
- `processed/csdh_learning_case_joined.csv.gz`
  - joined district-week-ZIP panel with one learning assignment per district-week
- `processed/csdh_learning_case_joined_monthly_shares.csv.gz`
  - joined district-week-ZIP panel with a monthly in-person share assigned to each week
- `processed/csdh_learning_case_joined_district_week.csv.gz`
  - district-week aggregate built from ZIP-level rows using `tot_zip_week` weights
- `processed/csdh_nces_district_master.csv`
  - cleaned NCES district master
- `processed/csdh_*_ohio.csv.gz`
  - Ohio-only subsets of the main learning, case, and joined tables
- `processed/csdh_*_massachusetts.csv.gz`
  - Massachusetts-only subsets of the main learning, case, and joined tables
- `processed/csdh_district_monthly_shares_*`
  - Ohio and Massachusetts subsets of the monthly-share table
- `processed/csdh_learning_case_joined_monthly_shares_*`
  - Ohio and Massachusetts subsets of the monthly-share joined weekly panel
- `processed/ohio_district_crosswalk.csv`
- `processed/massachusetts_district_crosswalk.csv`
  - district crosswalks with official and standardized geometry match flags
- `processed/ohio_districts_official.gpkg`
- `processed/ohio_districts_standardized.gpkg`
- `processed/massachusetts_districts_official.gpkg`
- `processed/massachusetts_districts_standardized.gpkg`
  - geometry-backed district files with attached `NCESDistrictID`
- `processed/*_centroids.csv`
  - centroid tables for each state / geometry source
- `processed/*_contiguity_adjacency.csv.gz`
  - polygon-contiguity edge lists
- `processed/*_knn_8_adjacency.csv.gz`
  - 8-nearest-neighbor edge lists from projected centroids
- `processed/*_distance_kernel_8_adjacency.csv.gz`
  - sparse centroid-distance-kernel edge lists over the same 8-nearest-neighbor support
- `processed/edge_acsed_district_features.csv.gz`
  - district-level EDGE ACS-ED feature table used as the main external-field basis candidate
- `processed/saipe_district_features.csv.gz`
  - supplemental Census SAIPE poverty and population feature table
- `processed/district_feature_basis.csv.gz`
  - combined clean district feature basis table for later MPLE construction
- `processed/district_feature_dictionary.csv`
  - compact dictionary for the district feature columns
- `processed/state_feature_coverage.csv`
  - Ohio and Massachusetts feature-join coverage diagnostics
- `processed/state_dataset_summary.csv`
  - Ohio and Massachusetts row counts, district counts, and date ranges
- `processed/state_join_coverage.csv`
  - Ohio and Massachusetts geometry coverage diagnostics
- `processed/state_network_summary.csv`
  - node counts, edge counts, and connected-component counts for each saved network
- `processed/processing_summary.json`
  - compact package summary used for quick auditing

## How The Network Options Are Realized

For each state and geometry source, the package writes three network-ready variants:

- `contiguity`
  - undirected polygon-touching network from district boundaries
- `knn_8`
  - undirected 8-nearest-neighbor graph from projected district centroids
- `distance_kernel_8`
  - weighted distance-kernel graph on the same 8-nearest-neighbor support

Current network summaries:

- Massachusetts official:
  - `290` nodes
  - contiguity: `821` edges, `5` connected components
  - `knn_8`: `1,353` edges, connected
  - distance-kernel: `1,353` weighted edges, connected
- Massachusetts standardized:
  - `290` nodes
  - contiguity: `850` edges, `2` connected components
  - `knn_8`: `1,354` edges, connected
  - distance-kernel: `1,354` weighted edges, connected
- Ohio official:
  - `609` nodes
  - contiguity: `1,680` edges, `5` connected components
  - `knn_8`: `2,853` edges, connected
  - distance-kernel: `2,853` weighted edges, connected
- Ohio standardized:
  - `609` nodes
  - contiguity: `1,713` edges, `3` connected components
  - `knn_8`: `2,856` edges, connected
  - distance-kernel: `2,856` weighted edges, connected

These files are intended as adjacency-ready inputs for later state-level MPLE experiments.

## Reproducibility

The processing script is:

- `prepare_csdh_data.py`

Run it from the repo root with:

```bash
pixi run python data/COVIDSchoolDataHub/prepare_csdh_data.py
```

For a fast rebuild that reuses the already-written learning, case, feature, and network tables:

```bash
pixi run python data/COVIDSchoolDataHub/prepare_csdh_data.py --reuse_processed_tables --reuse_processed_features --reuse_processed_networks
```

It will:

- download the requested COVID School Data Hub files
- download the Ohio and Massachusetts district geography
- download the required EDGE standardized school district boundaries
- rebuild the cleaned learning, case, and joined tables
- rebuild the crosswalks, GeoPackages, centroids, and adjacency-ready edge lists
- rebuild the district feature tables and feature dictionary
- rebuild the monthly-share tables and the monthly-share joined panel
- rewrite the package summary files

This package stops at data acquisition, joins, and adjacency-ready geometry. It does not yet threshold interventions or outcomes, and it does not run MPLE.
