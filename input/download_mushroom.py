#!/usr/bin/env python3
"""
MuSHRoom iPhone dataset downloader (COLMAP poses + RGB images).

This script prepares everything needed for:
  uv run --no-sync python pipeline.py --mushroom input/MuSHRoom/room_datasets/<room>

It runs in two phases:

  Phase 1 — COLMAP sparse models (all rooms, one archive, ~379 MB)
    Zenodo: https://zenodo.org/records/13986996
    File:   room_datasets_iphone_colmap.tar
    Installs: input/MuSHRoom/room_datasets/<room>/iphone/long_capture/sparse/...

  Phase 2 — RGB image sequences (one archive per room, ~150–400 MB each)
    Zenodo: https://zenodo.org/records/10230733  ("iPhone dataset (Basic Data)")
    Files:  <room>_iphone.tar.gz  (contains images/, depth/, test.txt, …)
    Installs: input/MuSHRoom/room_datasets/<room>/iphone/long_capture/images/

    Note: the larger <room>_iphone_our.tar.gz archives on Zenodo 10151161 are SDF
    derivatives only — they do not include images/ and will not satisfy the pipeline.

Skipped automatically when the target path already exists (use --force to re-download).
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colmap_paths import find_colmap_model

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR = Path(__file__).resolve().parent
MUSHROOM_ROOT = INPUT_DIR / "MuSHRoom"
ROOM_DATASETS = MUSHROOM_ROOT / "room_datasets"
ARCHIVE_DIR = INPUT_DIR / "_mushroom_archives"

# Phase 1: COLMAP poses for all iPhone long_capture sequences
COLMAP_RECORD = "13986996"
COLMAP_FILENAME = "room_datasets_iphone_colmap.tar"
COLMAP_URL = (
    f"https://zenodo.org/records/{COLMAP_RECORD}/files/{COLMAP_FILENAME}?download=1"
)
COLMAP_ARCHIVE = ARCHIVE_DIR / COLMAP_FILENAME

# Phase 2: per-room RGB (images/, depth/, test.txt) — MuSHRoom "Basic Data" iPhone release
IPHONE_RECORD = "10230733"
IPHONE_ROOM_ARCHIVES: dict[str, tuple[str, str]] = {
    room: (IPHONE_RECORD, f"{room}_iphone.tar.gz")
    for room in (
        "coffee_room",
        "honka",
        "computer",
        "kokko",
        "vr_room",
        "koivu",
        "activity",
        "classroom",
        "sauna",
        "olohuone",
    )
}

DEVICE = "iphone"
CAPTURE = "long_capture"


def _zenodo_url(record_id: str, filename: str) -> str:
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def _log(section: str, message: str) -> None:
    print(f"\n[{section}] {message}")


def _download(url: str, dest: Path, force: bool = False) -> None:
    if dest.is_file() and not force:
        _log("download", f"Reusing cached archive: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    _log("download", f"Fetching {url}")
    _log("download", f"  → {dest}")

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
                        print(
                            f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                            end="",
                        )
            if total > 0:
                print()


def room_colmap_dir(room: str) -> Path:
    return ROOM_DATASETS / room / DEVICE / CAPTURE


def room_images_dir(room: str) -> Path:
    return room_colmap_dir(room) / "images"


def colmap_ready(room: str) -> bool:
    return find_colmap_model(room_colmap_dir(room), raise_if_missing=False) is not None


def images_ready(room: str) -> bool:
    images = room_images_dir(room)
    return images.is_dir() and any(images.iterdir())


def all_colmap_ready() -> bool:
    return all(colmap_ready(room) for room in IPHONE_ROOM_ARCHIVES)


def _normalize_colmap_layout(extract_root: Path) -> None:
    """Move extracted room_datasets/ tree to MuSHRoom/room_datasets/."""
    nested = extract_root / "MuSHRoom" / "room_datasets"
    flat = extract_root / "room_datasets"

    if nested.is_dir():
        source = nested
    elif flat.is_dir():
        source = flat
    else:
        raise RuntimeError(
            f"Unexpected COLMAP archive layout under {extract_root}. "
            "Expected room_datasets/ or MuSHRoom/room_datasets/."
        )

    MUSHROOM_ROOT.mkdir(parents=True, exist_ok=True)
    if ROOM_DATASETS.exists():
        # Merge: do not delete existing room trees (e.g. images added earlier)
        for room_dir in source.iterdir():
            if not room_dir.is_dir():
                continue
            dest_room = ROOM_DATASETS / room_dir.name
            _merge_tree(room_dir, dest_room)
    else:
        shutil.move(str(source), str(ROOM_DATASETS))

    leftover = extract_root / "MuSHRoom"
    if leftover.is_dir() and not any(leftover.iterdir()):
        leftover.rmdir()


def _merge_tree(src: Path, dest: Path) -> None:
    """Copy src into dest, overwriting files but keeping unrelated dest files."""
    src = src.resolve()
    dest = dest.resolve()
    if src == dest:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dest / rel
        if target.exists() and os.path.samefile(item, target):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def phase1_download_colmap(force: bool) -> None:
    _log("phase 1", "COLMAP sparse models (all rooms)")
    if all_colmap_ready() and not force:
        _log("phase 1", f"Already complete under {ROOM_DATASETS} — skipping.")
        return

    _download(COLMAP_URL, COLMAP_ARCHIVE, force=force)
    _log("phase 1", f"Extracting {COLMAP_ARCHIVE.name}")
    with tempfile.TemporaryDirectory(prefix="mushroom_colmap_", dir=INPUT_DIR) as tmp:
        with tarfile.open(COLMAP_ARCHIVE, "r") as tar:
            tar.extractall(path=tmp)
        _normalize_colmap_layout(Path(tmp))

    missing = [r for r in IPHONE_ROOM_ARCHIVES if not colmap_ready(r)]
    if missing:
        raise RuntimeError(f"COLMAP still missing for rooms: {', '.join(missing)}")
    _log("phase 1", "Done — sparse/ poses installed for all rooms.")


def _archive_is_complete(archive_path: Path) -> bool:
    """Return True if the .tar.gz passes gzip integrity check."""
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        return False
    try:
        with gzip.open(archive_path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except (OSError, gzip.BadGzipFile, EOFError):
        return False


def _find_local_archive(room: str) -> Path | None:
    """Find a downloaded per-room tarball under input/ (any common filename)."""
    # Prefer Basic Data archives; do not auto-pick *_iphone_our (SDF-only, no images/).
    patterns = (
        f"{room}_iphone.tar.gz",
        f"{room}_iphone.tar",
    )
    search_dirs = (ARCHIVE_DIR, MUSHROOM_ROOT, INPUT_DIR)
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    return None


def _find_long_capture(extract_root: Path, room: str) -> Path:
    """Locate <room>/iphone/long_capture in an extracted per-room archive."""
    for direct in (
        extract_root / "room_datasets" / room / DEVICE / CAPTURE,
        extract_root / room / DEVICE / CAPTURE,
    ):
        if direct.is_dir():
            return direct

    for candidate in extract_root.rglob(CAPTURE):
        if candidate.is_dir() and room in str(candidate):
            parent = candidate.parent
            if parent.name == DEVICE:
                return candidate

    matches = list(extract_root.rglob(f"{CAPTURE}/images"))
    if matches:
        return matches[0].parent

    raise RuntimeError(
        f"Could not find {room}/{DEVICE}/{CAPTURE} in extracted archive under {extract_root}"
    )


def _merge_room_capture(src_capture: Path, room: str) -> None:
    """
    Copy RGB images (and test.txt if present) into room_datasets.
    Does not overwrite existing COLMAP sparse/ data.
    """
    dest_capture = room_colmap_dir(room)
    dest_capture.mkdir(parents=True, exist_ok=True)

    src_images = src_capture / "images"
    if not src_images.is_dir():
        raise RuntimeError(f"No images/ in {src_capture}")

    dest_images = dest_capture / "images"
    if dest_images.exists():
        shutil.rmtree(dest_images)
    _log("merge", f"Copying images → {dest_images}")
    shutil.copytree(src_images, dest_images)

    src_test = src_capture / "test.txt"
    if src_test.is_file():
        shutil.copy2(src_test, dest_capture / "test.txt")


def _extract_room_archive(archive_path: Path, room: str) -> None:
    _log(room, f"Extracting {archive_path.name} …")
    with tempfile.TemporaryDirectory(prefix=f"mushroom_{room}_", dir=INPUT_DIR) as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=tmp_path)
        src_capture = _find_long_capture(tmp_path, room)
        _merge_room_capture(src_capture, room)

    if not images_ready(room):
        raise RuntimeError(f"images/ still missing after merge for {room}")
    count = len(list(room_images_dir(room).iterdir()))
    _log(room, f"Done — {count} files in {room_images_dir(room)}")


def phase2_download_images(rooms: list[str], force: bool, download: bool = True) -> None:
    _log("phase 2", f"RGB image sequences ({len(rooms)} room(s))")
    _log("phase 2", f"Zenodo record {IPHONE_RECORD}; each archive is roughly 150–400 MB.")

    for room in rooms:
        if images_ready(room) and not force:
            _log(room, f"images/ already present — skipping.")
            continue

        record_id, filename = IPHONE_ROOM_ARCHIVES[room]
        archive_path = ARCHIVE_DIR / filename

        local = _find_local_archive(room)
        if local is not None and local.resolve() != archive_path.resolve():
            _log(room, f"Found local archive: {local}")
            if not archive_path.is_file() or force:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                if archive_path.exists():
                    archive_path.unlink()
                shutil.copy2(local, archive_path)

        if archive_path.is_file() and _archive_is_complete(archive_path):
            _log(room, f"Using complete archive: {archive_path}")
        elif archive_path.is_file() and not _archive_is_complete(archive_path):
            _log(room, f"Incomplete archive ({archive_path.stat().st_size / 1e9:.2f} GB) — will resume download")
            if not download:
                raise RuntimeError(
                    f"{archive_path} is incomplete. Re-run without --extract-local to resume download, "
                    f"or: curl -L -C - -o '{archive_path}' '{_zenodo_url(record_id, filename)}'"
                )
        elif download:
            url = _zenodo_url(record_id, filename)
            _log(room, f"Downloading {filename} (Zenodo record {record_id})")
            _download(url, archive_path, force=force)
        else:
            raise RuntimeError(
                f"No complete archive for {room}. Place {filename} in {ARCHIVE_DIR} or {MUSHROOM_ROOT}"
            )

        if not _archive_is_complete(archive_path):
            raise RuntimeError(
                f"{archive_path} failed integrity check (download incomplete). "
                f"Resume with: curl -L -C - -o '{archive_path}' '{_zenodo_url(record_id, filename)}'"
            )

        _extract_room_archive(archive_path, room)


def phase2_extract_local(rooms: list[str], force: bool) -> None:
    """Extract already-downloaded .tar.gz files from input/MuSHRoom or _mushroom_archives/."""
    _log("extract-local", f"Preparing {len(rooms)} room(s) from on-disk archives")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    phase2_download_images(rooms, force=force, download=False)


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("MuSHRoom dataset status")
    print("=" * 60)
    for room in IPHONE_ROOM_ARCHIVES:
        c = "yes" if colmap_ready(room) else "NO"
        i = "yes" if images_ready(room) else "NO"
        print(f"  {room:14}  COLMAP: {c:3}   images: {i}")
    print("=" * 60)
    ready = [
        r
        for r in IPHONE_ROOM_ARCHIVES
        if colmap_ready(r) and images_ready(r)
    ]
    if ready:
        example = ready[0]
        print(
            f"\nExample training command:\n"
            f"  uv run --no-sync python pipeline.py "
            f"--mushroom {ROOM_DATASETS / example}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--room",
        choices=sorted(IPHONE_ROOM_ARCHIVES),
        help="Download images for a single room only (COLMAP phase still runs if needed).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even when data already exists.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Skip phase 1 (COLMAP); only fetch RGB archives.",
    )
    parser.add_argument(
        "--colmap-only",
        action="store_true",
        help="Skip phase 2 (RGB images); only fetch COLMAP archive.",
    )
    parser.add_argument(
        "--extract-local",
        action="store_true",
        help="Only extract existing per-room .tar.gz files (no Zenodo download). Implies --images-only.",
    )
    args = parser.parse_args()

    if args.extract_local:
        args.images_only = True

    if args.images_only and args.colmap_only:
        print("Cannot use --images-only and --colmap-only together.", file=sys.stderr)
        return 1

    rooms = [args.room] if args.room else list(IPHONE_ROOM_ARCHIVES)

    print("MuSHRoom downloader")
    print(f"  Target directory: {ROOM_DATASETS}")
    print(f"  Archive cache:    {ARCHIVE_DIR}")

    try:
        if not args.images_only:
            phase1_download_colmap(force=args.force)
        if not args.colmap_only:
            if args.extract_local:
                phase2_extract_local(rooms, force=args.force)
            else:
                phase2_download_images(rooms, force=args.force, download=True)
    except (urllib.error.URLError, tarfile.TarError, OSError, RuntimeError) as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        return 1

    print_summary()

    if not args.colmap_only and not all(images_ready(r) for r in rooms):
        return 1
    if not args.images_only and not all_colmap_ready():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
