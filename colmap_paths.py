"""
Shared helpers for locating COLMAP sparse reconstruction directories.

Handles common layouts (MuSHRoom Zenodo archives, pycolmap export under
colmap_workspace/sparse/0, and nested sparse/0/0 folders).
"""

from __future__ import annotations

from pathlib import Path

# Files present in a valid COLMAP sparse model (binary or text export).
COLMAP_MARKERS = ("cameras.bin", "cameras.txt", "images.bin", "images.txt")

# Search order: nested Zenodo layout first, then standard sparse/0, then sparse/.
SPARSE_MODEL_CANDIDATES = (
    Path("sparse") / "0" / "0",
    Path("sparse") / "0",
    Path("sparse"),
)


def is_colmap_model(path: Path) -> bool:
    """True if path contains at least one COLMAP sparse model marker file."""
    return any((path / name).exists() for name in COLMAP_MARKERS)


def find_colmap_model(root: Path, *, raise_if_missing: bool = True) -> Path | None:
    """
    Return the COLMAP sparse model directory under root.

    Checks sparse/0/0, sparse/0, and sparse/ in that order.
    When raise_if_missing is False, returns None if no model is found.
    """
    base = root.expanduser().resolve()
    for relative in SPARSE_MODEL_CANDIDATES:
        candidate = base / relative
        if is_colmap_model(candidate):
            return candidate.resolve()

    if raise_if_missing:
        tried = ", ".join(str(p) for p in SPARSE_MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"No COLMAP sparse model found under {base}. "
            f"Expected {COLMAP_MARKERS[0]} or {COLMAP_MARKERS[1]} under one of: {tried}"
        )
    return None
