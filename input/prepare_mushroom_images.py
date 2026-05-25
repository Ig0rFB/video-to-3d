#!/usr/bin/env python3
"""
Build long_capture/images/ from existing sdf_dataset *_rgb.png files.

MuSHRoom COLMAP poses reference frame_XXXXX.jpg; Zenodo *_iphone_our archives
often ship RGB only under sdf_dataset_train_interp_4/ as NNNNNN_rgb.png.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

# Project root on path for mushroom_paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mushroom_paths import (  # noqa: E402
    add_mushroom_arguments,
    find_colmap_model,
    resolve_capture_dir,
)

FRAME_RE = re.compile(r"frame_(\d+)\.jpe?g", re.IGNORECASE)
SDF_DIRS = ("sdf_dataset_train_interp_4", "sdf_dataset_all_interp_4")


def _colmap_frame_names(colmap_model: Path) -> list[str]:
    images_txt = colmap_model / "images.txt"
    if not images_txt.is_file():
        raise FileNotFoundError(f"Missing {images_txt}")

    names: list[str] = []
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[-1])
    if not names:
        raise RuntimeError(f"No image entries parsed from {images_txt}")
    return names


def _find_sdf_dir(capture_dir: Path) -> Path:
    for name in SDF_DIRS:
        candidate = capture_dir / name
        if candidate.is_dir() and any(candidate.glob("*_rgb.png")):
            return candidate
    raise FileNotFoundError(
        f"No *_rgb.png under {capture_dir}/{{{', '.join(SDF_DIRS)}}}"
    )


def _sdf_rgb_path(sdf_dir: Path, frame_name: str) -> Path:
    match = FRAME_RE.search(frame_name)
    if not match:
        raise ValueError(f"Unexpected COLMAP image name: {frame_name}")
    index = int(match.group(1))
    src = sdf_dir / f"{index:06d}_rgb.png"
    if not src.is_file():
        raise FileNotFoundError(f"Missing SDF RGB for {frame_name}: {src}")
    return src


def prepare_images(
    capture_dir: Path,
    *,
    force: bool = False,
    jpeg_quality: int = 95,
) -> Path:
    images_dir = capture_dir / "images"
    if images_dir.is_dir() and any(images_dir.iterdir()) and not force:
        print(f"[skip] {images_dir} already has files (use --force to rebuild)")
        return images_dir

    colmap_model = find_colmap_model(capture_dir)
    sdf_dir = _find_sdf_dir(capture_dir)
    frame_names = _colmap_frame_names(colmap_model)

    if images_dir.exists():
        for old in images_dir.iterdir():
            if old.is_file():
                old.unlink()
    else:
        images_dir.mkdir(parents=True)

    print(f"[prepare] {capture_dir.name}")
    print(f"  COLMAP frames: {len(frame_names)}")
    print(f"  SDF source:    {sdf_dir.name}")
    print(f"  Output:        {images_dir}")

    for i, frame_name in enumerate(frame_names, start=1):
        src = _sdf_rgb_path(sdf_dir, frame_name)
        # COLMAP names use .jpg; write JPEG from PNG source
        dest = images_dir / frame_name
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(dest, format="JPEG", quality=jpeg_quality)
        if i % 50 == 0 or i == len(frame_names):
            print(f"  … {i}/{len(frame_names)}")

    print(f"[done] {len(frame_names)} images in {images_dir}")
    return images_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mushroom_arguments(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild images/ even when it already contains files.",
    )
    parser.add_argument(
        "--all-rooms",
        action="store_true",
        help="Process every room under input/MuSHRoom/room_datasets/ that has SDF RGB.",
    )
    args = parser.parse_args()

    if args.all_rooms:
        root = Path(__file__).resolve().parent / "MuSHRoom" / "room_datasets"
        captures = []
        for room_dir in sorted(root.iterdir()):
            if not room_dir.is_dir():
                continue
            cap = room_dir / "iphone" / "long_capture"
            if cap.is_dir():
                try:
                    _find_sdf_dir(cap)
                    captures.append(cap)
                except FileNotFoundError:
                    pass
        if not captures:
            print("No rooms with sdf_dataset *_rgb.png found.", file=sys.stderr)
            return 1
    elif args.mushroom:
        captures = [resolve_capture_dir(Path(args.mushroom), args.mushroom_device, args.mushroom_capture)]
    else:
        parser.error("Pass --mushroom PATH or --all-rooms")

    for capture_dir in captures:
        try:
            prepare_images(capture_dir, force=args.force)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[error] {capture_dir}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
