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
