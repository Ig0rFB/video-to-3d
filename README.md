# video3d

Reconstruct a geometrically coherent 3D indoor scene from a short phone-captured video using COLMAP structure-from-motion and 3D Gaussian Splatting (nerfstudio `splatfacto` / gsplat).

## Requirements

- **Python 3.10** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — package and environment management
- **ffmpeg** — frame extraction
- **COLMAP** — camera poses and sparse reconstruction
- **macOS:** Homebrew for system deps (`brew install ffmpeg colmap`)
- **Linux:** `sudo apt-get install -y ffmpeg colmap`

Hardware: CUDA, Apple Silicon MPS, or CPU. The active backend is chosen automatically (see `device.py`).

## Installation

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the project
cd video-to-3d

# Reproduce the environment from the lockfile
uv sync --prerelease=allow

# Verify device detection (expect MPS: True on Apple Silicon)
uv run python -c "from device import get_device; get_device()"

# Verify system tools
colmap help
ffmpeg -version

# Verify nerfstudio CLI
uv run ns-train --help
```

> **Note:** `nerfstudio` on arm64 requires pre-release resolution. Use `uv sync --prerelease=allow` (or `uv add --prerelease=allow` when adding packages).

## Usage

1. Place your video at `input/room.mp4` (or pass a custom path to individual steps).
2. Run the full pipeline:

```bash
uv run python pipeline.py
```

Or run stages manually:

```bash
uv run python 01_extract_frames.py --video input/room.mp4 --fps 2
uv run python 02_run_colmap.py
uv run python 03_train_gaussian.py
uv run python 05_export.py --checkpoint_dir outputs/nerfstudio_data/splatfacto/<timestamp>
```

Use the latest timestamped folder under `outputs/nerfstudio_data/splatfacto/` (not `outputs/splatfacto/latest-run`).

For videos shorter than 30 seconds, try `--fps 3` or `--fps 4` in step 01 if COLMAP registers too few images.

**Training time:** ~20 min on CUDA (RTX 4090). Do not interrupt `03_train_gaussian.py`.

### Apple Silicon (MPS)

Steps **01** (frames) and **02** (COLMAP) run fully on an M5/M-series Mac. **03** (`splatfacto`) uses **gsplat’s CUDA rasteriser**, which is not built on macOS without an NVIDIA GPU — so Gaussian training must run on a **CUDA Linux/Windows machine** (or cloud GPU), not on MPS alone.

After `uv sync`, apply the nerfstudio MPS init fix (safe on all platforms):

```bash
uv run python patch_nerfstudio_mps.py
```

Typical workflow on a Mac:

1. Run `01_extract_frames.py` and `02_run_colmap.py` (or `pipeline.py` up to COLMAP).
2. Copy `nerfstudio_data/` to a CUDA host.
3. There: `uv sync --prerelease=allow && uv run python patch_nerfstudio_mps.py && uv run python 03_train_gaussian.py`
4. Copy `outputs/` back and run `05_export.py` with the checkpoint path below.

`03_train_gaussian.py` exits early on MPS/CPU when gsplat CUDA is missing, with this message, instead of failing mid-run.

### MuSHRoom dataset (skip steps 01 and 02)

One script downloads **COLMAP poses and RGB images** for all iPhone rooms (skips anything already on disk):

```bash
uv run python input/download_mushroom.py
```

| Phase | Zenodo | Size |
|-------|--------|------|
| COLMAP `sparse/` (all 10 rooms) | [13986996](https://zenodo.org/records/13986996) | ~379 MB |
| RGB `images/` per room | [10230733](https://zenodo.org/records/10230733) (`<room>_iphone.tar.gz`) | ~150–400 MB each |

Do not use the larger `*_iphone_our.tar.gz` files on [10151161](https://zenodo.org/records/10151161) for training — those archives contain SDF derivatives only, not `images/`.

Useful options:

```bash
# One room only (e.g. coffee_room, ~4 GB images)
uv run python input/download_mushroom.py --room coffee_room

# COLMAP only or images only
uv run python input/download_mushroom.py --colmap-only
uv run python input/download_mushroom.py --images-only --room coffee_room

# SDF-only download (*_iphone_our): build images/ from *_rgb.png (COLMAP frame names)
uv run python input/prepare_mushroom_images.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

Archives are cached under `input/_mushroom_archives/`. See the script docstring for step-by-step behaviour.

To train without multi‑GB downloads, use your own video: `uv run python pipeline.py --video input/your.mp4`.

When both `images/` and `sparse/` exist, pass `--mushroom` to skip frame extraction and COLMAP:

```bash
# Room root (defaults: iphone + long_capture)
uv run python pipeline.py --mushroom input/MuSHRoom/room_datasets/coffee_room

# Or point at the capture folder directly
uv run python pipeline.py \
  --mushroom input/MuSHRoom/room_datasets/coffee_room/iphone/long_capture

# Kinect short sequence
uv run python pipeline.py \
  --mushroom input/MuSHRoom/room_datasets/sauna \
  --mushroom-device kinect \
  --mushroom-capture short_capture
```

Train only (same paths as pipeline):

```bash
uv run python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

## Outputs

After a successful run, check `export/`:

| File | Description |
|------|-------------|
| `export/point_cloud.ply` | Gaussian splat point cloud — open in [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.cloudcompare.org/) |
| `export/render.mp4` | Novel-view spiral render around the scene |
| `export/examples/` | Example input frame and output render for documentation |

Intermediate artefacts (`frames/`, `colmap_workspace/`, `nerfstudio_data/`, `outputs/`) are gitignored.

## Optional: semantic lifting

After core export works:

```bash
uv add --prerelease=allow groundingdino-py
uv add "git+https://github.com/facebookresearch/sam2.git"
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

uv run python 04_semantic_lift.py --checkpoint_dir outputs/splatfacto/latest-run
```

## Example

Add one extracted frame and one spiral render to `export/examples/` after your first successful run (these paths are gitignored with `export/` — copy or symlink for README screenshots as needed).

## Hardware notes

- Device selection is centralised in `device.py` (`cuda` → `mps` → `cpu`).
- `pipeline.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before any PyTorch import so unsupported MPS ops fall back to CPU.
- **splatfacto / gsplat:** training needs CUDA; MPS is used for COLMAP and data prep only on Mac.
- Do not hardcode device strings elsewhere in the codebase.

## Project layout

```
input/              # raw .mp4
frames/             # extracted JPEGs
colmap_workspace/   # COLMAP sparse model
nerfstudio_data/    # ns-process-data output
outputs/            # training checkpoints
export/             # PLY + renders
device.py           # shared device utility
pipeline.py         # orchestration
01_extract_frames.py … 05_export.py
DESIGN.md           # design rationale (after pipeline verified)
```
