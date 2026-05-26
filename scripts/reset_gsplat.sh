#!/usr/bin/env bash
# Restore gsplat from the lockfile, then apply the Mac CPU/MPS viewer patch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

echo "==> Reinstall gsplat (from pyproject lock)"
uv pip uninstall --python "$PY" gsplat 2>/dev/null || true
uv sync --prerelease=allow --no-install-project 2>/dev/null || uv sync --prerelease=allow

echo "==> Apply non-CUDA gsplat patch"
uv run --no-sync python patch_gsplat_non_cuda.py
