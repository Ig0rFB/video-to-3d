"""
Resolve image and COLMAP paths for the MuSHRoom dataset.

See: https://github.com/TUTvision/MuSHRoom
Zenodo COLMAP poses: https://zenodo.org/records/13986996
"""

from __future__ import annotations

import argparse
from pathlib import Path

COLMAP_MARKERS = ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
VALID_DEVICES = ("iphone", "kinect")
VALID_CAPTURES = ("long_capture", "short_capture")


def _has_colmap_model(path: Path) -> bool:
    return any((path / name).exists() for name in COLMAP_MARKERS)


def find_colmap_model(capture_dir: Path) -> Path:
    """Return the COLMAP sparse model directory (handles sparse/0 and sparse/0/0)."""
    candidates = [
        capture_dir / "sparse" / "0" / "0",
        capture_dir / "sparse" / "0",
        capture_dir / "sparse",
    ]
    for candidate in candidates:
        if _has_colmap_model(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f"No COLMAP sparse model found under {capture_dir}/sparse. "
        "Expected cameras.bin/txt under sparse/0 or sparse/0/0 "
        "(Zenodo iPhone COLMAP release)."
    )


def find_image_dir(capture_dir: Path) -> Path:
    images = capture_dir / "images"
    if not images.is_dir():
        raise FileNotFoundError(
            f"Missing {images}. MuSHRoom training needs the RGB sequence from the "
            "main Zenodo room download (images/ per capture), not only the COLMAP pose archive."
        )
    if not any(images.iterdir()):
        raise FileNotFoundError(f"{images} is empty.")
    return images.resolve()


def resolve_capture_dir(
    mushroom_path: Path,
    device: str = "iphone",
    capture: str = "long_capture",
) -> Path:
    """
    Normalise user input to a capture directory, e.g.
    .../coffee_room/iphone/long_capture
    """
    path = mushroom_path.expanduser().resolve()

    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")
    if capture not in VALID_CAPTURES:
        raise ValueError(f"capture must be one of {VALID_CAPTURES}, got {capture!r}")

    if path.name in VALID_CAPTURES:
        return path
    if path.name in VALID_DEVICES and (path / capture).is_dir():
        return path / capture

    for base in (path, path / device):
        candidate = base / capture
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find MuSHRoom capture at {path} with device={device!r}, capture={capture!r}. "
        f"Pass the capture folder directly (e.g. .../coffee_room/iphone/long_capture) "
        f"or a room root (e.g. .../coffee_room)."
    )


def resolve_mushroom_paths(
    mushroom_path: str | Path,
    device: str = "iphone",
    capture: str = "long_capture",
) -> tuple[Path, Path]:
    """
    Return (image_dir, colmap_model_path) for ns-process-data / splatfacto.

    image_dir: directory of RGB frames (capture/images/)
    colmap_model_path: sparse reconstruction folder (cameras.bin, images.bin, ...)
    """
    capture_dir = resolve_capture_dir(Path(mushroom_path), device=device, capture=capture)
    return find_image_dir(capture_dir), find_colmap_model(capture_dir)


def add_mushroom_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mushroom",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "MuSHRoom dataset path: room root (e.g. .../coffee_room) or capture folder "
            "(.../iphone/long_capture). Skips frame extraction and COLMAP; uses existing "
            "images/ and sparse/ poses."
        ),
    )
    parser.add_argument(
        "--mushroom-device",
        type=str,
        default="iphone",
        choices=VALID_DEVICES,
        help="Sensor when --mushroom points at a room root (default: iphone).",
    )
    parser.add_argument(
        "--mushroom-capture",
        type=str,
        default="long_capture",
        choices=VALID_CAPTURES,
        help="Sequence when --mushroom points at a room root (default: long_capture).",
    )
