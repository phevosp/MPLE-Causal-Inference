"""Fallback remapping for ORCD-rooted paths into the local repository."""

from __future__ import annotations

from pathlib import Path


ORCD_REPO_ROOT = "/orcd/home/002/phevosp/MPLE-Causal-Inference"
LOCAL_REPO_ROOT = "C:/Users/phevo/Documents/MIT/Code/MPLE-Causal-Inference"


def resolve_orcd_local_path(path: str | Path) -> Path:
    """Resolve a path, falling back from the ORCD repo root to the local repo root."""
    original_text = str(path)
    original_candidate = Path(original_text)
    if original_candidate.exists():
        return original_candidate.resolve()

    normalized_text = original_text.replace("\\", "/")
    remapped_text = normalized_text
    if normalized_text.startswith(ORCD_REPO_ROOT):
        remapped_text = LOCAL_REPO_ROOT + normalized_text[len(ORCD_REPO_ROOT) :]

    remapped_candidate = Path(remapped_text)
    if remapped_candidate.exists():
        return remapped_candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve path. "
        f"Original candidate: {original_candidate}. "
        f"Remapped candidate: {remapped_candidate}."
    )
