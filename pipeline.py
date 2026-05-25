import argparse
import os
import subprocess
import sys

# Must be set before any torch imports — no-op on CUDA/CPU, enables MPS op fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from mushroom_paths import add_mushroom_arguments, resolve_mushroom_paths

DEFAULT_VIDEO = "input/room.mp4"
DEFAULT_CHECKPOINT = "outputs/splatfacto/latest-run"


def build_steps(
    video: str,
    mushroom: str | None,
    mushroom_device: str,
    mushroom_capture: str,
    checkpoint_dir: str,
) -> list[list[str]]:
    if mushroom:
        image_dir, colmap_model = resolve_mushroom_paths(
            mushroom,
            device=mushroom_device,
            capture=mushroom_capture,
        )
        print(f"[mushroom] image_dir: {image_dir}")
        print(f"[mushroom] colmap_model: {colmap_model}")
        train_step = [
            "uv",
            "run",
            "python",
            "03_train_gaussian.py",
            "--image_dir",
            str(image_dir),
            "--colmap-model-path",
            str(colmap_model),
        ]
        return [
            train_step,
            [
                "uv",
                "run",
                "python",
                "05_export.py",
                "--checkpoint_dir",
                checkpoint_dir,
            ],
        ]

    return [
        [
            "uv",
            "run",
            "python",
            "01_extract_frames.py",
            "--video",
            video,
            "--fps",
            "2",
        ],
        ["uv", "run", "python", "02_run_colmap.py"],
        ["uv", "run", "python", "03_train_gaussian.py"],
        [
            "uv",
            "run",
            "python",
            "05_export.py",
            "--checkpoint_dir",
            checkpoint_dir,
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run video-to-3D reconstruction (ffmpeg → COLMAP → splatfacto → export)."
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"Input video for the default pipeline (default: {DEFAULT_VIDEO}).",
    )
    add_mushroom_arguments(parser)
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT,
        help=f"Trained splatfacto run for export (default: {DEFAULT_CHECKPOINT}).",
    )
    args = parser.parse_args()

    steps = build_steps(
        video=args.video,
        mushroom=args.mushroom,
        mushroom_device=args.mushroom_device,
        mushroom_capture=args.mushroom_capture,
        checkpoint_dir=args.checkpoint_dir,
    )

    for step in steps:
        print(f"\n>>> {' '.join(step)}\n")
        result = subprocess.run(step, check=False)
        if result.returncode != 0:
            print(f"\nStep failed: {' '.join(step)}")
            print("Resolve the error above before continuing.")
            sys.exit(1)

    print("\nPipeline complete. Check export/ for outputs.")


if __name__ == "__main__":
    main()
