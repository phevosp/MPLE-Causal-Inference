"""Small utility helpers used by the active USCountyVaccination pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy import sparse


def download_if_missing(url: str, destination: Path) -> None:
    """Download one file unless it already exists."""
    if destination.exists():
        return
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)


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


def normalize_sparse_matrix_infinity(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Scale one sparse matrix so its infinity norm is one."""
    csr = matrix.tocsr().astype(float)
    row_sums = np.asarray(np.abs(csr).sum(axis=1)).ravel()
    norm = float(row_sums.max()) if row_sums.size else 0.0
    if norm < 1e-12:
        return csr
    return (csr / norm).tocsr()


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
    distance_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distance_matrix, np.inf)
    finite_distances = distance_matrix[np.isfinite(distance_matrix)]
    median_distance = float(np.median(finite_distances)) if finite_distances.size else 1.0

    knn_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    for i, node_id in enumerate(node_ids):
        for j in np.argsort(distance_matrix[i])[:k]:
            source, target = sorted([node_id, node_ids[j]])
            distance = float(distance_matrix[i, j])
            knn_rows.append({id_column: source, "neighbor_id": target, "weight": 1.0})
            kernel_rows.append(
                {
                    id_column: source,
                    "neighbor_id": target,
                    "weight": float(np.exp(-distance / median_distance)) if median_distance > 0 else 1.0,
                }
            )

    knn = pd.DataFrame(knn_rows).drop_duplicates([id_column, "neighbor_id"])
    kernel = pd.DataFrame(kernel_rows).groupby([id_column, "neighbor_id"], as_index=False)["weight"].max()
    return (
        knn.sort_values([id_column, "neighbor_id"]).reset_index(drop=True),
        kernel.sort_values([id_column, "neighbor_id"]).reset_index(drop=True),
    )
