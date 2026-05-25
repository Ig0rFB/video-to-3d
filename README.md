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
uv run python 05_export.py --checkpoint_dir outputs/splatfacto/latest-run
```

For videos shorter than 30 seconds, try `--fps 3` or `--fps 4` in step 01 if COLMAP registers too few images.

**Training time:** ~20 min on CUDA (RTX 4090), ~60–90 min on MPS (M5 Pro). Do not interrupt `03_train_gaussian.py`.

### MuSHRoom dataset (skip steps 01 and 02)

Download the iPhone COLMAP pose archive (if missing):

```bash
uv run python input/download_mushroom.py
```

This fetches [room_datasets_iphone_colmap.tar](https://zenodo.org/records/13986996/files/room_datasets_iphone_colmap.tar?download=1) (~379 MB) into `input/MuSHRoom/room_datasets/`.

If you have the [MuSHRoom](https://github.com/TUTvision/MuSHRoom) indoor room dataset with pre-computed COLMAP poses and RGB frames, pass `--mushroom` to skip frame extraction and COLMAP. You need **both**:

- `images/` under each capture (main [Zenodo room download](https://zenodo.org/communities/mushroom))
- `sparse/` COLMAP model ([Zenodo COLMAP poses](https://zenodo.org/records/13986996) — often under `sparse/0/0/`)

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
