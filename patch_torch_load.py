"""
Nerfstudio checkpoints use pickled numpy objects; PyTorch 2.6+ defaults weights_only=True.

Apply before any nerfstudio import that calls torch.load.
"""

from __future__ import annotations

import torch

_PATCHED = "_video3d_torch_load_patched"


def patch_torch_load() -> None:
    if getattr(torch.load, _PATCHED, False):
        return
    original = torch.load

    def load(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    setattr(load, _PATCHED, True)
    torch.load = load  # type: ignore[assignment]
    print("[patch] torch.load defaults to weights_only=False for nerfstudio checkpoints")


if __name__ == "__main__":
    patch_torch_load()
