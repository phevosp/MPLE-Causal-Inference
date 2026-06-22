"""CSV file writing and formatting utilities."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from utils.t0_path_utils import io_path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read rows from a CSV file as dictionaries."""
    with open(io_path(path), "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    """Write rows to a CSV file, inferring columns when not provided."""
    if columns is None:
        if not rows:
            fieldnames = ["name"]
        else:
            fieldnames = []
            seen: set[str] = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
    else:
        fieldnames = list(columns)
    os.makedirs(io_path(path.parent), exist_ok=True)
    with open(io_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in fieldnames} for row in rows]
        )


def write_csv(path: str | Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write rows to a CSV file with specified columns."""
    write_csv_rows(path, rows, columns=columns)
