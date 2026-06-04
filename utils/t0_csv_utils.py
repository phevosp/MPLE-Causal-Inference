"""CSV file writing and formatting utilities."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from utils.t0_path_utils import io_path


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write rows to a CSV file with specified columns."""
    os.makedirs(io_path(path.parent), exist_ok=True)
    with open(io_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in columns} for row in rows]
        )


def _fmt(value: object) -> str:
    """Format a value for display in CSV/reports."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
