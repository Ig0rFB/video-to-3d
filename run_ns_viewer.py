"""
Start ns-viewer with project patches (torch.load + splatfacto MPS init).

Usage:
  uv run --no-sync python run_ns_viewer.py --load-config \\
    outputs/nerfstudio_data/splatfacto/<timestamp>/config.yml
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from patch_torch_load import patch_torch_load
from patch_nerfstudio_mps import patch_splatfacto

patch_torch_load()
patch_splatfacto()

from nerfstudio.scripts.viewer.run_viewer import entrypoint

if __name__ == "__main__":
    entrypoint()
