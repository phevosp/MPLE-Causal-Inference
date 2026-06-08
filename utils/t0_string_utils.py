"""Generic string normalization helpers."""

from __future__ import annotations

import re


def slugify(text: str, fallback: str = "item") -> str:
    """Convert a label into a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or fallback
