"""Prepare the SeattleDMI block-level analysis package."""

from __future__ import annotations

import json
import sys
import tarfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyreadr

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils import (  # noqa: E402
    build_knn_and_kernel_edges,
    build_touching_edge_list,
    count_connected_components,
    download_if_missing,
    standardize_id,
)


MICROSYNTH_URL = "https://michaelwrobbins.r-universe.dev/src/contrib/microsynth_2.0.51.tar.gz"
GEOGRAPHY_ZIP_URL = "https://www.seattle.gov/documents/Departments/OPCD/Demographics/GeographicFilesandMaps/KingCountyBlocksShapefiles.zip"
CORRELATION_URL = "https://www.seattle.gov/documents/Departments/OPCD/Demographics/GeographicFilesandMaps/SeattleCensusBlocksandNeighborhoodCorrelationFile.xlsx"


def ensure_directories(base_dir: Path) -> dict[str, Path]:
    """Create the raw and processed directories used by the SeattleDMI pipeline."""
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    microsynth_dir = raw_dir / "microsynth"
    geography_dir = raw_dir / "geography"
    extracted_shape_dir = geography_dir / "KingCountyBlocksShapefiles"

    for path in [raw_dir, processed_dir, microsynth_dir, geography_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "microsynth": microsynth_dir,
        "geography": geography_dir,
        "extracted_shape": extracted_shape_dir,
    }


def extract_if_missing(paths: dict[str, Path]) -> tuple[Path, Path]:
    """Extract the microsynth package contents and the geography zip as needed."""
    microsynth_tar = paths["microsynth"] / "microsynth_2.0.51.tar.gz"
    microsynth_extract_root = paths["microsynth"] / "microsynth"
    if not microsynth_extract_root.exists():
        with tarfile.open(microsynth_tar, "r:gz") as archive:
            archive.extractall(paths["microsynth"])

    geography_zip = paths["geography"] / "KingCountyBlocksShapefiles.zip"
    if not paths["extracted_shape"].exists():
        with zipfile.ZipFile(geography_zip, "r") as archive:
            archive.extractall(paths["extracted_shape"])

    rda_path = microsynth_extract_root / "data" / "seattledmi.rda"
    shp_path = next(paths["extracted_shape"].glob("*.shp"))
    return rda_path, shp_path


def load_seattledmi_panel(rda_path: Path) -> pd.DataFrame:
    """Load the SeattleDMI panel from the microsynth package data archive."""
    panel = pyreadr.read_r(str(rda_path))["seattledmi"].copy()
    panel["ID"] = panel["ID"].round().astype("int64")
    panel["GEOID10"] = standardize_id(panel["ID"], width=15)
    panel["time"] = panel["time"].astype("int64")
    panel["Intervention"] = panel["Intervention"].astype("int64")
    return panel.sort_values(["GEOID10", "time"]).reset_index(drop=True)


def build_block_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the panel to one row per block with static covariates and treatment metadata."""
    static_columns = [
        "GEOID10",
        "ID",
        "TotalPop",
        "BLACK",
        "HISPANIC",
        "Males_1521",
        "HOUSEHOLDS",
        "FAMILYHOUS",
        "FEMALE_HOU",
        "RENTER_HOU",
        "VACANT_HOU",
    ]
    static = panel[static_columns].drop_duplicates(subset=["GEOID10"]).copy()
    intervention_start = (
        panel.loc[panel["Intervention"] > 0]
        .groupby("GEOID10")["time"]
        .min()
        .rename("intervention_start_time")
    )
    static = static.merge(intervention_start, on="GEOID10", how="left")
    static["treated_ever"] = static["intervention_start_time"].notna().astype(int)
    static["intervention_start_time"] = static["intervention_start_time"].fillna(0).astype(int)
    return static


def build_preperiod_crime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre-intervention crime outcomes to one row per block."""
    outcome_columns = [
        "i_robbery",
        "i_aggassau",
        "i_burglary",
        "i_larceny",
        "i_felony",
        "i_misdemea",
        "i_drugsale",
        "i_drugposs",
        "i_drugs",
        "any_crime",
    ]
    pre_panel = panel.loc[panel["time"] <= 12, ["GEOID10", *outcome_columns]].copy()
    aggregated = pre_panel.groupby("GEOID10", as_index=False).agg(["sum", "mean"])
    aggregated.columns = [
        "GEOID10",
        *[
            f"{column}_pre_{agg}"
            for column in outcome_columns
            for agg in ("sum", "mean")
        ],
    ]
    return aggregated


def load_correlation_workbook(correlation_path: Path) -> pd.DataFrame:
    """Load Seattle's block-to-neighborhood correlation workbook."""
    correlation = pd.read_excel(correlation_path, sheet_name="CENSUS_2010_BLOCKS")
    correlation["GEOID10"] = (
        correlation["GEOID10"].astype(str).str.replace(".0", "", regex=False).str.zfill(15)
    )
    return correlation


def load_block_geometries(shp_path: Path) -> gpd.GeoDataFrame:
    """Load the King County 2010 census block shapefile."""
    gdf = gpd.read_file(shp_path)
    gdf["GEOID10"] = gdf["GEOID10"].astype(str).str.zfill(15)
    return gdf


def build_joined_geography(
    block_features: pd.DataFrame,
    correlation: pd.DataFrame,
    block_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Join SeattleDMI block features to block geometries and neighborhood metadata."""
    joined = block_gdf.merge(block_features, on="GEOID10", how="inner", validate="one_to_one")
    joined = joined.merge(correlation, on="GEOID10", how="left", suffixes=("", "_corr"))
    joined["has_neighborhood_match"] = joined["NEIGHBORHOOD_DISTRICT_NAME"].notna().astype(int)
    centroids = joined.to_crs(2285).centroid.to_crs(joined.crs)
    joined["centroid_lon"] = centroids.x
    joined["centroid_lat"] = centroids.y
    return joined


def build_centroid_table(joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Save projected and lon/lat centroid coordinates so experiment runs can skip the GeoPackage."""
    projected = joined_gdf.to_crs(2285).copy()
    centroids = projected.geometry.centroid
    centroids_ll = gpd.GeoSeries(centroids, crs=projected.crs).to_crs(joined_gdf.crs)
    table = pd.DataFrame(
        {
            "GEOID10": projected["GEOID10"].astype(str).str.zfill(15),
            "centroid_x": centroids.x.to_numpy(),
            "centroid_y": centroids.y.to_numpy(),
            "centroid_lon": centroids_ll.x.to_numpy(),
            "centroid_lat": centroids_ll.y.to_numpy(),
        }
    )
    return table.sort_values("GEOID10").reset_index(drop=True)


def build_network_artifacts(
    centroids: pd.DataFrame,
    contiguity: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build the reusable Seattle network edge lists and a compact summary table."""
    knn_8, kernel_8 = build_knn_and_kernel_edges(
        centroids,
        id_column="GEOID10",
        x_column="centroid_x",
        y_column="centroid_y",
        k=8,
    )
    knn_16, kernel_16 = build_knn_and_kernel_edges(
        centroids,
        id_column="GEOID10",
        x_column="centroid_x",
        y_column="centroid_y",
        k=16,
    )
    network_tables = {
        "contiguity": contiguity.sort_values(["GEOID10", "neighbor_GEOID10"]).reset_index(drop=True),
        "knn_8": knn_8.rename(columns={"neighbor_id": "neighbor_GEOID10"}),
        "knn_16": knn_16.rename(columns={"neighbor_id": "neighbor_GEOID10"}),
        "centroid_distance_kernel_8": kernel_8.rename(columns={"neighbor_id": "neighbor_GEOID10"}),
        "centroid_distance_kernel_16": kernel_16.rename(columns={"neighbor_id": "neighbor_GEOID10"}),
    }

    node_ids = centroids["GEOID10"].tolist()
    summary_rows: list[dict[str, int | str]] = []
    for name, edges in network_tables.items():
        summary_rows.append(
            {
                "network_name": name,
                "node_count": int(len(node_ids)),
                "edge_count": int(len(edges)),
                "connected_components": int(
                    count_connected_components(node_ids, edges, "GEOID10", "neighbor_GEOID10")
                ),
            }
        )
    return network_tables, pd.DataFrame(summary_rows)


def build_crosswalk(joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create a lightweight non-spatial block crosswalk for merging and auditing."""
    columns = [
        "GEOID10",
        "ID",
        "treated_ever",
        "intervention_start_time",
        "TRACTCE10",
        "BLOCKCE10",
        "TRACT_10",
        "TRBL",
        "TRBG_10",
        "BG",
        "URBAN_VILLAGE_NUMBER",
        "URBAN_VILLAGE_NAME",
        "URBAN_VILLAGE_TYPE",
        "CRA_NO",
        "CRA_NAME",
        "NEIGHBORHOODS_INCLUDED",
        "NEIGHBORHOOD_DISTRICT_NUMBER",
        "NEIGHBORHOOD_DISTRICT_NAME",
        "has_neighborhood_match",
        "centroid_lon",
        "centroid_lat",
    ]
    existing = [column for column in columns if column in joined_gdf.columns]
    crosswalk = joined_gdf[existing].copy()
    if "ID" in crosswalk.columns:
        crosswalk["ID"] = crosswalk["ID"].astype("int64")
    return crosswalk.sort_values("GEOID10").reset_index(drop=True)


def save_outputs(
    paths: dict[str, Path],
    panel: pd.DataFrame,
    block_features: pd.DataFrame,
    preperiod_features: pd.DataFrame,
    crosswalk: pd.DataFrame,
    centroids: pd.DataFrame,
    joined_gdf: gpd.GeoDataFrame,
    adjacency: pd.DataFrame,
    network_tables: dict[str, pd.DataFrame],
    network_summary: pd.DataFrame,
    correlation: pd.DataFrame,
) -> dict[str, int]:
    """Write processed SeattleDMI files and a compact processing summary."""
    processed = paths["processed"]
    panel.to_csv(processed / "seattledmi_panel.csv.gz", index=False)
    block_features.to_csv(processed / "seattledmi_block_features.csv", index=False)
    preperiod_features.to_csv(processed / "seattledmi_block_preperiod_crime.csv", index=False)
    crosswalk.to_csv(processed / "seattledmi_block_crosswalk.csv", index=False)
    centroids.to_csv(processed / "seattledmi_block_centroids.csv", index=False)
    adjacency.to_csv(processed / "seattledmi_block_adjacency.csv.gz", index=False)
    network_tables["knn_8"].to_csv(processed / "seattledmi_block_knn_8_adjacency.csv.gz", index=False)
    network_tables["knn_16"].to_csv(processed / "seattledmi_block_knn_16_adjacency.csv.gz", index=False)
    network_tables["centroid_distance_kernel_8"].to_csv(
        processed / "seattledmi_block_distance_kernel_8_adjacency.csv.gz",
        index=False,
    )
    network_tables["centroid_distance_kernel_16"].to_csv(
        processed / "seattledmi_block_distance_kernel_16_adjacency.csv.gz",
        index=False,
    )
    network_summary.to_csv(processed / "seattledmi_network_summary.csv", index=False)
    joined_gdf.to_file(processed / "seattledmi_blocks.gpkg", layer="blocks", driver="GPKG")

    missing_corr = crosswalk.loc[crosswalk["has_neighborhood_match"] == 0, ["GEOID10"]].copy()
    missing_corr.to_csv(processed / "seattledmi_blocks_missing_correlation.csv", index=False)

    summary = {
        "panel_rows": int(len(panel)),
        "unique_blocks": int(panel["GEOID10"].nunique()),
        "treated_blocks": int(block_features["treated_ever"].sum()),
        "time_periods": int(panel["time"].nunique()),
        "adjacency_edges": int(len(adjacency)),
        "network_rows": int(len(network_summary)),
        "blocks_with_neighborhood_match": int(crosswalk["has_neighborhood_match"].sum()),
        "blocks_missing_neighborhood_match": int((crosswalk["has_neighborhood_match"] == 0).sum()),
        "correlation_rows": int(len(correlation)),
    }
    (processed / "processing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Download, preprocess, and save the SeattleDMI data with geographic joins."""
    base_dir = Path(__file__).resolve().parent
    paths = ensure_directories(base_dir)

    download_if_missing(MICROSYNTH_URL, paths["microsynth"] / "microsynth_2.0.51.tar.gz")
    download_if_missing(GEOGRAPHY_ZIP_URL, paths["geography"] / "KingCountyBlocksShapefiles.zip")
    download_if_missing(
        CORRELATION_URL,
        paths["geography"] / "SeattleCensusBlocksandNeighborhoodCorrelationFile.xlsx",
    )

    rda_path, shp_path = extract_if_missing(paths)
    panel = load_seattledmi_panel(rda_path)
    block_features = build_block_features(panel)
    preperiod_features = build_preperiod_crime_features(panel)
    correlation = load_correlation_workbook(
        paths["geography"] / "SeattleCensusBlocksandNeighborhoodCorrelationFile.xlsx"
    )
    block_gdf = load_block_geometries(shp_path)
    joined_gdf = build_joined_geography(block_features, correlation, block_gdf)
    crosswalk = build_crosswalk(joined_gdf)
    centroids = build_centroid_table(joined_gdf)
    adjacency = build_touching_edge_list(
        joined_gdf,
        id_column="GEOID10",
        neighbor_column="neighbor_GEOID10",
    )
    network_tables, network_summary = build_network_artifacts(centroids, adjacency)
    summary = save_outputs(
        paths,
        panel,
        block_features,
        preperiod_features,
        crosswalk,
        centroids,
        joined_gdf,
        adjacency,
        network_tables,
        network_summary,
        correlation,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
