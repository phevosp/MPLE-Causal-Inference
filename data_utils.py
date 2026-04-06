"""Shared data-preparation helpers for the repository's preprocessing scripts.

This module collects small, reusable utilities for loading, normalizing, and
joining data across the synthetic, SeattleDMI, and COVID School Data Hub
pipelines.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape


def download_if_missing(url: str, destination: Path) -> None:
    """Download one file to disk unless it already exists locally."""
    if destination.exists():
        return

    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)


def standardize_id(series: pd.Series, width: int | None = None) -> pd.Series:
    """Convert a mixed-type identifier column to a clean string representation."""
    cleaned = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    if width is not None:
        cleaned = cleaned.str.zfill(width)
    return cleaned


def normalize_name(value: str | float | None) -> str:
    """Normalize a district name for fallback matching across files."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).upper()
    text = text.replace("&", "AND")
    text = re.sub(r"\(DISTRICT\)", "", text)
    text = re.sub(r"\bSCHOOL DISTRICT\b", "", text)
    text = re.sub(r"\bPUBLIC SCHOOLS\b", "", text)
    text = re.sub(r"\bCITY OF\b", "", text)
    text = re.sub(r"\bEXEMPTED VILLAGE\b", "EX VILL", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_numeric_text(value: str | None) -> float | None:
    """Parse a formatted numeric string into a float."""
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if cleaned in {"", "-", ".", "-."}:
        return None
    return float(cleaned)


def center_and_normalize_vector_infinity(vector: np.ndarray) -> np.ndarray:
    """Center a vector and scale it to infinity norm one."""
    values = np.asarray(vector, dtype=float)
    if np.all(np.isnan(values)):
        return np.zeros_like(values)
    values = np.nan_to_num(values, nan=np.nanmean(values))
    values = values - np.nanmean(values)
    norm = float(np.linalg.norm(values, ord=np.inf))
    if norm < 1e-12:
        return np.zeros_like(values)
    return values / norm


def normalize_vector_infinity(vector: np.ndarray) -> np.ndarray:
    """Scale a vector by its infinity norm without centering."""
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values, ord=np.inf))
    if norm < 1e-12:
        return np.zeros_like(values)
    return values / norm


def normalize_sparse_matrix_infinity(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Scale one sparse matrix so its infinity norm is one."""
    csr = matrix.tocsr().astype(float)
    row_sums = np.asarray(np.abs(csr).sum(axis=1)).ravel()
    norm = float(row_sums.max()) if row_sums.size else 0.0
    if norm < 1e-12:
        return csr
    return (csr / norm).tocsr()


def safe_ratio(numerator, denominator) -> np.ndarray:
    """Compute a safe ratio feature, returning zeros where the denominator vanishes."""
    num = np.nan_to_num(np.asarray(numerator, dtype=float), nan=0.0)
    den = np.nan_to_num(np.asarray(denominator, dtype=float), nan=0.0)
    out = np.zeros_like(num, dtype=float)
    valid = np.abs(den) > 1e-12
    out[valid] = num[valid] / den[valid]
    return out


def fetch_arcgis_geojson(layer_url: str, out_fields: str = "*") -> gpd.GeoDataFrame:
    """Download a full ArcGIS feature layer as GeoJSON using paginated queries."""
    count_response = requests.get(
        f"{layer_url}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=300,
    )
    count_response.raise_for_status()
    total_count = int(count_response.json()["count"])

    features: list[dict[str, object]] = []
    offset = 0
    batch_size = 500
    while offset < total_count:
        response = requests.get(
            f"{layer_url}/query",
            params={
                "where": "1=1",
                "outFields": out_fields,
                "returnGeometry": "true",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": batch_size,
            },
            timeout=300,
        )
        response.raise_for_status()
        features.extend(response.json()["features"])
        offset += batch_size

    records = []
    for feature in features:
        props = feature["properties"].copy()
        props["geometry"] = shape(feature["geometry"])
        records.append(props)
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def save_raw_geojson(gdf: gpd.GeoDataFrame, destination: Path) -> None:
    """Write a GeoDataFrame to raw GeoJSON storage."""
    gdf.to_file(destination, driver="GeoJSON")


def build_touching_edge_list(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    neighbor_column: str,
    geometry_column: str = "geometry",
) -> pd.DataFrame:
    """Build an undirected contiguity edge list from polygon boundaries."""
    left = gdf[[id_column, geometry_column]].copy()
    right = left.rename(columns={id_column: neighbor_column})
    joined = gpd.sjoin(left, right, how="inner", predicate="touches")
    edges = joined[[id_column, neighbor_column]].copy()
    edges = edges.loc[edges[id_column] < edges[neighbor_column]].drop_duplicates()
    return edges.sort_values([id_column, neighbor_column]).reset_index(drop=True)


def count_connected_components(
    nodes: list[str] | pd.Series,
    edges: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> int:
    """Count connected components in an undirected graph represented by an edge list."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    node_list = list(nodes)
    for node in node_list:
        adjacency[node]
    for _, row in edges.iterrows():
        source = row[source_column]
        target = row[target_column]
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen: set[str] = set()
    components = 0
    for node in node_list:
        if node in seen:
            continue
        components += 1
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current] - seen)
    return components


def build_knn_and_kernel_edges(
    centroids: pd.DataFrame,
    id_column: str,
    x_column: str,
    y_column: str,
    k: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build k-nearest-neighbor and distance-kernel edge lists from centroid coordinates."""
    if len(centroids) <= 1:
        empty = pd.DataFrame(columns=[id_column, "neighbor_id", "weight"])
        return empty.copy(), empty.copy()

    coords = centroids[[x_column, y_column]].to_numpy(dtype=float)
    node_ids = centroids[id_column].to_numpy()
    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distance_matrix, np.inf)

    knn_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    finite_distances = distance_matrix[np.isfinite(distance_matrix)]
    median_distance = float(np.median(finite_distances)) if finite_distances.size else 1.0

    for i, node_id in enumerate(node_ids):
        neighbor_idx = np.argsort(distance_matrix[i])[:k]
        for j in neighbor_idx:
            source, target = sorted([node_id, node_ids[j]])
            dist = float(distance_matrix[i, j])
            knn_rows.append({id_column: source, "neighbor_id": target, "weight": 1.0})
            kernel_rows.append(
                {
                    id_column: source,
                    "neighbor_id": target,
                    "weight": float(np.exp(-dist / median_distance)) if median_distance > 0 else 1.0,
                }
            )

    knn = pd.DataFrame(knn_rows).drop_duplicates([id_column, "neighbor_id"])
    kernel = (
        pd.DataFrame(kernel_rows)
        .groupby([id_column, "neighbor_id"], as_index=False)["weight"]
        .max()
    )
    return (
        knn.sort_values([id_column, "neighbor_id"]).reset_index(drop=True),
        kernel.sort_values([id_column, "neighbor_id"]).reset_index(drop=True),
    )
