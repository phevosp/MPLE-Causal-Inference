"""Pipeline-spec expansion, validation, and lookup helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.t0_config_utils import (
    deep_merge_mappings,
    load_yaml_mapping,
    to_plain_mapping,
)
from utils.t0_string_utils import slugify
from utils.t6_split_management import VALID_SPLIT_KINDS, normalize_split_kind


def expand_named_entries(
    spec_path: str | Path,
    entries_key: str,
) -> list[dict[str, Any]]:
    spec = load_yaml_mapping(spec_path)
    base = to_plain_mapping(spec.get("base"))
    entries = spec.get(entries_key, [])
    if not isinstance(entries, list):
        raise ValueError(f"'{entries_key}' must be a list in {spec_path}.")
    expanded: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        plain_entry = to_plain_mapping(entry)
        name = plain_entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Entry {index} in '{entries_key}' must define a non-empty 'name'."
            )
        merged = deep_merge_mappings(base, plain_entry)
        merged["name"] = name
        merged["slug"] = slugify(name)
        expanded.append(merged)
    return expanded


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
    if (
        mode != "exact_rank_manifold"
        and float(variant.get("lambda_frobenius", 0.0)) != 0.0
    ):
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
    """Validate a fits_spec.yaml at load time."""
    variants = expand_named_entries(spec_path, "variants")
    for variant in variants:
        validate_fit_variant_dict(variant)


def validate_cv_spec(spec_path: str | Path) -> None:
    searches = expand_named_entries(spec_path, "searches")
    for search in searches:
        split_kind = normalize_split_kind(search.get("split_kind", "train_cv"))
        if split_kind not in VALID_SPLIT_KINDS:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' split_kind must be one of "
                f"{sorted(VALID_SPLIT_KINDS)}."
            )
        if "outer_num_folds" in search and int(search["outer_num_folds"]) <= 0:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' outer_num_folds must be positive."
            )
        if "test_fold_id" in search and int(search["test_fold_id"]) <= 0:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' test_fold_id must be positive."
            )
        validation_sampling = search.get("validation_sampling", {})
        if validation_sampling is None:
            validation_sampling = {}
        if not isinstance(validation_sampling, dict):
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' validation_sampling must be a mapping."
            )
        for key in ("num_samples", "gibbs_sweeps"):
            if key in validation_sampling and int(validation_sampling[key]) <= 0:
                raise ValueError(
                    f"Search '{search.get('name', '<unnamed>')}' validation_sampling.{key} "
                    "must be positive."
                )
        if "seed" in validation_sampling:
            int(validation_sampling["seed"])
        grid = search.get("grid", {})
        if not isinstance(grid, dict) or not grid:
            raise ValueError(
                f"Search '{search.get('name', '<unnamed>')}' must define a non-empty grid mapping."
            )

        flattened_candidates: list[dict[str, Any]] = []

        def _walk_grid(
            node: dict[str, Any],
            path: list[str],
            out: list[tuple[list[str], list[Any]]],
        ) -> None:
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


def load_search_from_spec(
    cv_spec_path: str | Path,
    search_slug: str,
) -> dict[str, Any]:
    validate_cv_spec(cv_spec_path)
    searches = expand_named_entries(cv_spec_path, "searches")
    for search in searches:
        search["_spec_path"] = str(Path(cv_spec_path).resolve())
    search = next((item for item in searches if item.get("slug") == search_slug), None)
    if search is None:
        raise ValueError(f"Search slug '{search_slug}' not found in {cv_spec_path}.")
    return search


def best_candidate_path_for_search(
    experiment_root: str | Path,
    cv_spec_path: str | Path,
    search_slug: str,
    *,
    execution_mode: str = "cv",
) -> Path:
    search = load_search_from_spec(cv_spec_path, search_slug)
    root_name_key = "cv_root_name" if str(execution_mode).strip().lower() == "cv" else "validation_root_name"
    default_root_name = "cv_runs" if root_name_key == "cv_root_name" else "validation_runs"
    return (
        Path(experiment_root).resolve()
        / str(search.get(root_name_key, default_root_name))
        / str(search["slug"])
        / "best_candidate.yaml"
    ).resolve()
