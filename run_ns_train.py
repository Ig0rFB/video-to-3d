"""
Entry point for ns-train that applies Apple Silicon / non-CUDA patches first.

Usage: uv run --no-sync python run_ns_train.py splatfacto --data ... [ns-train args]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from patch_gsplat_non_cuda import patch_gsplat_rendering
from patch_nerfstudio_mps import patch_splatfacto

patch_gsplat_rendering()
patch_splatfacto()

from nerfstudio.scripts.train import entrypoint

if __name__ == "__main__":
    entrypoint()
