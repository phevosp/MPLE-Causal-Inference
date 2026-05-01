"""Shared utilities for loading, merging, and expanding pipeline YAML specs and CSV manifests.

All pipeline entry points use `expand_named_entries` to resolve the base+overrides pattern
in fits_spec.yaml, generation_spec.yaml, etc., and `read/write_csv_manifest` for artifact
tracking between stages.
"""

from __future__ import annotations

import csv
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "item"


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Expected a mapping-like YAML object.")
    return deepcopy(container)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = OmegaConf.load(Path(path))
    container = OmegaConf.to_container(spec, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"Spec at {path} must be a mapping.")
    return container


def expand_named_entries(
    spec_path: str | Path,
    entries_key: str,
) -> list[dict[str, Any]]:
    spec = load_spec(spec_path)
    base = _as_plain_dict(spec.get("base"))
    entries = spec.get(entries_key, [])
    if not isinstance(entries, list):
        raise ValueError(f"'{entries_key}' must be a list in {spec_path}.")
    expanded: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        plain_entry = _as_plain_dict(entry)
        name = plain_entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Entry {index} in '{entries_key}' must define a non-empty 'name'."
            )
        merged = deep_merge(base, plain_entry)
        merged["name"] = name
        merged["slug"] = slugify(name)
        expanded.append(merged)
    return expanded


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_csv_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = ensure_parent_dir(path)
    if not rows:
        fieldnames = ["name"]
    else:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


_VALID_OPTIMIZER_MODES = frozenset(
    {
        "no_external_field",
        "nuclear_norm",
        "exact_rank_manifold",
        "alternating_latent_rank",
        "concurrent_latent_rank",
    }
)


def validate_fit_variant_dict(variant: dict[str, Any]) -> None:
    name = str(variant.get("name", "<unnamed>"))
    mode = str(variant.get("optimizer_mode", "no_external_field"))
    if mode not in _VALID_OPTIMIZER_MODES:
        raise ValueError(
            f"Variant '{name}': optimizer_mode '{mode}' is not valid. "
            f"Must be one of: {sorted(_VALID_OPTIMIZER_MODES)}."
        )
    rank = int(variant.get("latent_rank", 0))
    if mode in {
        "exact_rank_manifold",
        "alternating_latent_rank",
        "concurrent_latent_rank",
    } and rank <= 0:
        raise ValueError(
            f"Variant '{name}': latent_rank must be >= 1 for optimizer_mode='{mode}' (got {rank})."
        )
    for param in ("lambda_nuclear", "lambda_frobenius", "lambda_uv_ridge"):
        val = float(variant.get(param, 0.0))
        if val < 0.0:
            raise ValueError(
                f"Variant '{name}': {param} must be non-negative (got {val})."
            )
    if (
        variant.get("v_column_l2_max", None) is not None
        and float(variant["v_column_l2_max"]) <= 0.0
    ):
        raise ValueError(
            f"Variant '{name}': v_column_l2_max must be positive "
            f"(got {variant['v_column_l2_max']})."
        )
    if mode != "nuclear_norm" and float(variant.get("lambda_nuclear", 0.0)) != 0.0:
        raise ValueError(
            f"Variant '{name}': lambda_nuclear is only valid for optimizer_mode='nuclear_norm'."
        )
    if mode != "exact_rank_manifold" and float(variant.get("lambda_frobenius", 0.0)) != 0.0:
        raise ValueError(
            f"Variant '{name}': lambda_frobenius is only valid for optimizer_mode='exact_rank_manifold'."
        )
    if (
        mode not in {"alternating_latent_rank", "concurrent_latent_rank"}
        and float(variant.get("lambda_uv_ridge", 0.0)) != 0.0
    ):
        raise ValueError(
            f"Variant '{name}': lambda_uv_ridge is only valid for optimizer_mode="
            "'alternating_latent_rank' or 'concurrent_latent_rank'."
        )


def validate_fits_spec(spec_path: str | Path) -> None:
    """Validate a fits_spec.yaml at load time and raise ValueError on the first problem found.

    Checks that each variant has a valid optimizer_mode, that rank-based modes have a
    positive latent_rank, and that all regularization values are non-negative. Call this
    at the top of run_fit_pipeline.py before processing any experiments.
    """
    variants = expand_named_entries(spec_path, "variants")
    for variant in variants:
        validate_fit_variant_dict(variant)


def validate_cv_spec(spec_path: str | Path) -> None:
    searches = expand_named_entries(spec_path, "searches")
    for search in searches:
        grid = search.get("grid", {})
        if not isinstance(grid, dict) or not grid:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' must define a non-empty grid mapping."
            )

        flattened_candidates: list[dict[str, Any]] = []

        def _walk_grid(node: dict[str, Any], path: list[str], out: list[tuple[list[str], list[Any]]]) -> None:
            for key, value in node.items():
                key_path = [*path, str(key)]
                if isinstance(value, dict):
                    _walk_grid(value, key_path, out)
                    continue
                if not isinstance(value, list) or not value:
                    dotted = ".".join(key_path)
                    raise ValueError(
                        f"Search '{search.get('name', '<unnamed>')}' grid leaf '{dotted}' "
                        "must be a non-empty list."
                    )
                out.append((key_path, value))

        leaves: list[tuple[list[str], list[Any]]] = []
        _walk_grid(grid, [], leaves)
        for key_path, values in leaves:
            flattened_candidates.append({"path": ".".join(key_path), "values": values})
        if not flattened_candidates:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' grid does not contain any list-valued leaves."
            )
