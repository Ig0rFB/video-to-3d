# video3d

Reconstruct a geometrically coherent 3D indoor scene from a short phone-captured video using COLMAP structure-from-motion and 3D Gaussian Splatting (nerfstudio `splatfacto` / gsplat).

## Requirements

- **Python 3.10** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — package and environment management
- **ffmpeg** — frame extraction
- **COLMAP** — camera poses and sparse reconstruction
- **Linux:** `sudo apt-get install -y ffmpeg colmap`

Hardware: **CUDA-enabled NVIDIA GPU**. This project is configured to run on CUDA only.

| Guide | Contents |
|-------|----------|
| [docs/CLOUD_GPU.md](docs/CLOUD_GPU.md) | vast.ai / Linux CUDA one-shot setup and troubleshooting |
| [docs/MUSHROOM.md](docs/MUSHROOM.md) | MuSHRoom download, Zenodo archives, train commands |
| [DESIGN.md](DESIGN.md) | Stack choices, hardware strategy, future work |

## Installation

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the project
cd video-to-3d

# Reproduce the environment from the lockfile
uv sync --prerelease=allow

# Preflight
uv run --no-sync python scripts/ensure_env.py

# Verify system tools
colmap help
ffmpeg -version
```

> **PyTorch:** Linux CUDA hosts use **torch 2.5.1 + cu124** from `pyproject.toml` (not the latest PyPI CUDA 13 wheel). After `uv sync`, run pipeline commands with **`uv run --no-sync`** so uv does not re-resolve torch. Plain `uv run` without `--no-sync` may still change wheels if you edit dependencies.

## Usage

1. Place your video at `input/room.mp4` (or pass a custom path to individual steps).
2. Run the full pipeline:

```bash
uv run --no-sync python pipeline.py
```

Or run stages manually:

```bash
uv run --no-sync python 01_extract_frames.py --video input/room.mp4 --fps 2
uv run --no-sync python 02_run_colmap.py
uv run --no-sync python 03_train_gaussian.py
uv run --no-sync python 05_export.py --checkpoint_dir outputs/nerfstudio_data/splatfacto/<timestamp>
```

Use the latest timestamped folder under `outputs/nerfstudio_data/splatfacto/` (not `outputs/splatfacto/latest-run`).

For videos shorter than 30 seconds, try `--fps 3` or `--fps 4` in step 01 if COLMAP registers too few images.

**Training time:** ~20 min on CUDA (RTX 4090). Do not interrupt `03_train_gaussian.py`.

**Export:** `05_export.py` uses `ns-render interpolate` (not `spiral`). Video render can take a while on large scenes; PLY-only: `--skip-render`.

### Cloud GPU (vast.ai, etc.)

See **[docs/CLOUD_GPU.md](docs/CLOUD_GPU.md)** for the full checklist. Short version:

```bash
git clone https://github.com/Ig0rFB/video-to-3d.git && cd video-to-3d
./scripts/setup_cloud.sh
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

### MuSHRoom dataset (skip steps 01 and 02)

See **[docs/MUSHROOM.md](docs/MUSHROOM.md)** for Zenodo IDs and layout.

One script downloads **COLMAP poses and RGB images** for all iPhone rooms (skips anything already on disk):

```bash
uv run --no-sync python input/download_mushroom.py
```

| Phase | Zenodo | Size |
|-------|--------|------|
| COLMAP `sparse/` (all 10 rooms) | [13986996](https://zenodo.org/records/13986996) | ~379 MB |
| RGB `images/` per room | [10230733](https://zenodo.org/records/10230733) (`<room>_iphone.tar.gz`) | ~150–400 MB each |

Do not use the larger `*_iphone_our.tar.gz` files on [10151161](https://zenodo.org/records/10151161) for training — those archives contain SDF derivatives only, not `images/`.

Useful options:

```bash
# One room only (e.g. coffee_room, ~4 GB images)
uv run --no-sync python input/download_mushroom.py --room coffee_room

# COLMAP only or images only
uv run --no-sync python input/download_mushroom.py --colmap-only
uv run --no-sync python input/download_mushroom.py --images-only --room coffee_room

# SDF-only download (*_iphone_our): build images/ from *_rgb.png (COLMAP frame names)
uv run --no-sync python input/prepare_mushroom_images.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

Archives are cached under `input/_mushroom_archives/`. See the script docstring for step-by-step behaviour.

To train without multi‑GB downloads, use your own video: `uv run --no-sync python pipeline.py --video input/your.mp4`.

When both `images/` and `sparse/` exist, pass `--mushroom` to skip frame extraction and COLMAP:

```bash
# Room root (defaults: iphone + long_capture)
uv run --no-sync python pipeline.py --mushroom input/MuSHRoom/room_datasets/coffee_room

# Or point at the capture folder directly
uv run --no-sync python pipeline.py \
  --mushroom input/MuSHRoom/room_datasets/coffee_room/iphone/long_capture

# Kinect short sequence
uv run --no-sync python pipeline.py \
  --mushroom input/MuSHRoom/room_datasets/sauna \
  --mushroom-device kinect \
  --mushroom-capture short_capture
```

Train only (same paths as pipeline):

```bash
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

## Outputs

After a successful run, check `export/`:

| File | Description |
|------|-------------|
| `export/point_cloud.ply` | Gaussian splat point cloud — open in [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.cloudcompare.org/) |
| `export/render.mp4` | Novel-view video (camera path interpolated from training poses) |
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

uv run --no-sync python 04_semantic_lift.py --checkpoint_dir outputs/nerfstudio_data/splatfacto/<timestamp>
```

## Example

Add one extracted frame and one spiral render to `export/examples/` after your first successful run (these paths are gitignored with `export/` — copy or symlink for README screenshots as needed).

## Hardware notes

- Device selection is centralised in `device.py` (`cuda` → `cpu`).
- **splatfacto / gsplat:** training requires CUDA.
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
env_utils.py        # COLMAP / CLI / CUDA preflight helpers
scripts/            # setup_cloud.sh, ensure_env.py
docs/               # CLOUD_GPU.md, MUSHROOM.md
pipeline.py         # orchestration (uses uv run --no-sync)
01_extract_frames.py … 05_export.py
DESIGN.md           # design rationale
```
