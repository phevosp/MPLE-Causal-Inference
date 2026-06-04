"""Path resolution and file existence utilities."""

from __future__ import annotations

import os
from pathlib import Path


def io_path(path: str | Path) -> str:
    """Resolve a path to an absolute, platform-safe string.

    On Windows, prepends UNC prefix for long paths. On Unix, returns absolute path.
    """
    candidate = Path(path)
    if os.name == "nt":
        resolved = str(candidate.resolve())
        if not resolved.startswith("\\\\?\\"):
            return "\\\\?\\" + resolved
        return resolved
    if candidate.is_absolute():
        return str(candidate)
    return os.path.abspath(os.fspath(candidate))


def path_exists(path: str | Path) -> bool:
    """Check if a path exists, using platform-safe resolution."""
    return os.path.exists(io_path(path))


def first_existing_path(*paths: str | Path) -> Path:
    """Return the first path from the list that exists.

    Raises FileNotFoundError if none of the paths exist.
    """
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of the expected paths: "
        + ", ".join(str(Path(path)) for path in paths)
    )
