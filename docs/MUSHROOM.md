# MuSHRoom dataset — download and train

See also [README.md](../README.md) for full project setup.

## Layout (after download)

```text
input/MuSHRoom/room_datasets/<room>/iphone/long_capture/
  images/          # RGB frames (required)
  sparse/0/0/      # COLMAP model (phase 1)
  test.txt         # optional eval split
```

## Quick start (cloud GPU)

```bash
git clone https://github.com/Ig0rFB/video-to-3d.git && cd video-to-3d
./scripts/setup_cloud.sh

uv run --no-sync python input/download_mushroom.py --room coffee_room
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

Checkpoints: `outputs/splatfacto/<timestamp>/` (export with `05_export.py` or `--checkpoint-dir latest`)

## Download commands

```bash
# All rooms (COLMAP + images where not cached)
uv run --no-sync python input/download_mushroom.py

# One room
uv run --no-sync python input/download_mushroom.py --room coffee_room

# COLMAP or images only
uv run --no-sync python input/download_mushroom.py --colmap-only
uv run --no-sync python input/download_mushroom.py --images-only --room coffee_room

# Re-extract archives already under input/_mushroom_archives/
uv run --no-sync python input/download_mushroom.py --room coffee_room --extract-local
```

## Zenodo archives

| Phase | Record | File pattern |
|-------|--------|----------------|
| COLMAP | [13986996](https://zenodo.org/records/13986996) | `room_datasets_iphone_colmap.tar` |
| RGB | [10230733](https://zenodo.org/records/10230733) | `<room>_iphone.tar.gz` |

Do **not** use `*_iphone_our.tar.gz` from [10151161](https://zenodo.org/records/10151161) for training — no `images/` folder. If you only have SDF packs, build symlinks:

```bash
uv run --no-sync python input/prepare_mushroom_images.py \
  --mushroom input/MuSHRoom/room_datasets/coffee_room
```

## Train

```bash
# Full pipeline (train + export if checkpoint path set)
uv run --no-sync python pipeline.py --mushroom input/MuSHRoom/room_datasets/coffee_room

# Train only
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

Both `images/` and a valid `sparse/` model must exist. Check status in the downloader summary table.

## Kinect rooms

```bash
uv run --no-sync python pipeline.py \
  --mushroom input/MuSHRoom/room_datasets/sauna \
  --mushroom-device kinect \
  --mushroom-capture short_capture
```
