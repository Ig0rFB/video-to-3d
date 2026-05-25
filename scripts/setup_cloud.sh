#!/usr/bin/env bash
# One-shot setup for a fresh Linux CUDA cloud instance (vast.ai, Lambda, etc.).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> System packages (ffmpeg, COLMAP)"
if ! command -v colmap >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg colmap
  else
    echo "Install ffmpeg and colmap manually, then re-run this script."
    exit 1
  fi
fi

echo "==> Python environment (uv sync — torch cu124 pinned in pyproject.toml)"
uv sync --prerelease=allow

echo "==> nerfstudio MPS patch (harmless on CUDA)"
uv run --no-sync python patch_nerfstudio_mps.py

echo "==> Verify / repair CUDA PyTorch in .venv"
uv run --no-sync python scripts/ensure_env.py --fix-cuda --require-cuda

echo ""
echo "Ready. Train with:"
echo "  uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room"
echo "Avoid plain 'uv run' without --no-sync before training (it can re-resolve torch)."
