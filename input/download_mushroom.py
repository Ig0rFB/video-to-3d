#!/usr/bin/env python3
"""
Download and extract the MuSHRoom iPhone COLMAP dataset from Zenodo.

Record: https://zenodo.org/records/13986996
Archive: room_datasets_iphone_colmap.tar (~379 MB)
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

INPUT_DIR = Path(__file__).resolve().parent
MUSHROOM_ROOT = INPUT_DIR / "MuSHRoom"
ROOM_DATASETS = MUSHROOM_ROOT / "room_datasets"
DOWNLOAD_URL = (
    "https://zenodo.org/records/13986996/files/room_datasets_iphone_colmap.tar?download=1"
)
TAR_PATH = INPUT_DIR / "room_datasets_iphone_colmap.tar"


def dataset_exists() -> bool:
    """True if room_datasets with at least one iPhone COLMAP capture is present."""
    if not ROOM_DATASETS.is_dir():
        return False
    for room_dir in ROOM_DATASETS.iterdir():
        if not room_dir.is_dir():
            continue
        sparse = room_dir / "iphone" / "long_capture" / "sparse"
        if sparse.is_dir() and any(sparse.rglob("cameras.bin")) or any(
            sparse.rglob("cameras.txt")
        ):
            return True
    return False


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024

        if tqdm is not None and total > 0:
            with open(dest, "wb") as out, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
        else:
            downloaded = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = 100 * downloaded / total
                        print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
            if total > 0:
                print()


def _normalize_layout(extract_root: Path) -> None:
    """Ensure data lives at MuSHRoom/room_datasets/."""
    nested = extract_root / "MuSHRoom" / "room_datasets"
    flat = extract_root / "room_datasets"

    if nested.is_dir():
        source = nested
    elif flat.is_dir():
        source = flat
    else:
        raise RuntimeError(
            f"Unexpected archive layout under {extract_root}. "
            "Expected room_datasets/ or MuSHRoom/room_datasets/."
        )

    MUSHROOM_ROOT.mkdir(parents=True, exist_ok=True)
    if ROOM_DATASETS.exists():
        if dataset_exists():
            print(f"Dataset already at {ROOM_DATASETS}, leaving in place.")
            return
        shutil.rmtree(ROOM_DATASETS)

    if source.resolve() == ROOM_DATASETS.resolve():
        return

    shutil.move(str(source), str(ROOM_DATASETS))
    # Remove empty MuSHRoom wrapper if the archive only had MuSHRoom/room_datasets
    leftover = extract_root / "MuSHRoom"
    if leftover.is_dir() and not any(leftover.iterdir()):
        leftover.rmdir()


def extract_tar(tar_path: Path) -> None:
    print(f"Extracting {tar_path} → {INPUT_DIR}")
    with tarfile.open(tar_path, "r") as tar:
        # filter= requires Python 3.12+; project pins 3.10
        tar.extractall(path=INPUT_DIR)
    _normalize_layout(INPUT_DIR)


def main() -> int:
    if dataset_exists():
        print(f"MuSHRoom COLMAP dataset already present at {ROOM_DATASETS}")
        return 0

    if not TAR_PATH.is_file():
        try:
            _download(DOWNLOAD_URL, TAR_PATH)
        except urllib.error.URLError as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"Using existing archive: {TAR_PATH}")

    try:
        extract_tar(TAR_PATH)
    except (tarfile.TarError, OSError, RuntimeError) as exc:
        print(f"Extract failed: {exc}", file=sys.stderr)
        return 1

    if dataset_exists():
        print(f"Done. Dataset ready at {ROOM_DATASETS}")
        print(
            "Note: this archive is COLMAP poses only. RGB images/ come from the "
            "main MuSHRoom Zenodo room downloads for full training."
        )
        return 0

    print("Extract finished but dataset layout could not be verified.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
