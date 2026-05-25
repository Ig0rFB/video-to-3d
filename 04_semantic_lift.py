"""
Optional semantic lifting: Grounding DINO + SAM2 → 3D Gaussian labels.

Implement after the core pipeline produces a valid PLY export.
Requires: groundingdino-py, sam2, and checkpoints/sam2.1_hiera_large.pt
"""

import argparse

from device import get_device

TARGET_CLASSES = [
    "table",
    "chair",
    "sofa",
    "shelf",
    "door",
    "window",
    "floor",
    "wall",
    "ceiling",
    "lamp",
]


def lift_semantics(
    checkpoint_dir: str,
    frames_dir: str,
    output_dir: str,
) -> None:
    """
    Placeholder for semantic lifting pipeline:
    1. Grounding DINO on keyframes → boxes + labels
    2. SAM2 on boxes → per-instance masks
    3. Project Gaussian centres to cameras, majority-vote labels
    4. Export colour-coded PLY to export/
    """
    device = get_device()
    raise NotImplementedError(
        f"Semantic lifting not yet implemented (device={device}). "
        f"Run core pipeline first, then implement lift for "
        f"checkpoint={checkpoint_dir}, frames={frames_dir}, output={output_dir}, "
        f"classes={TARGET_CLASSES}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--frames_dir", default="frames/")
    parser.add_argument("--output_dir", default="semantic/")
    args = parser.parse_args()
    lift_semantics(args.checkpoint_dir, args.frames_dir, args.output_dir)
