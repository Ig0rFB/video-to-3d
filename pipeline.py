import argparse
import subprocess
import sys

from checkpoint_paths import LATEST_CHECKPOINT, resolve_checkpoint_dir
from mushroom_paths import add_mushroom_arguments, resolve_mushroom_paths

DEFAULT_VIDEO = "input/video.mp4"


def uv_python(*script_args: str) -> list[str]:
    """Run a project script without re-syncing the venv (avoids overwriting pinned CUDA torch)."""
    return ["uv", "run", "--no-sync", "python", *script_args]


def build_steps(
    video: str,
    mushroom: str | None,
    mushroom_device: str,
    mushroom_capture: str,
) -> list[list[str]]:
    if mushroom:
        image_dir, colmap_model = resolve_mushroom_paths(
            mushroom,
            device=mushroom_device,
            capture=mushroom_capture,
        )
        print(f"[mushroom] image_dir: {image_dir}")
        print(f"[mushroom] colmap_model: {colmap_model}")
        train_step = uv_python(
            "03_train_gaussian.py",
            "--image-dir",
            str(image_dir),
            "--colmap-model-path",
            str(colmap_model),
        )
        return [train_step, ["__export__"]]

    return [
        uv_python("01_extract_frames.py", "--video", video, "--fps", "2"),
        uv_python("02_run_colmap.py"),
        uv_python("03_train_gaussian.py"),
        ["__export__"],
    ]


def _run_step(step: list[str], checkpoint_dir: str) -> int:
    if step == ["__export__"]:
        resolved = resolve_checkpoint_dir(checkpoint_dir)
        step = uv_python("05_export.py", "--checkpoint-dir", str(resolved))
    print(f"\n>>> {' '.join(step)}\n")
    result = subprocess.run(step, check=False)
    if result.returncode != 0:
        print(f"\nStep failed: {' '.join(step)}")
        print("Resolve the error above before continuing.")
    return result.returncode


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
        default=LATEST_CHECKPOINT,
        help=(
            f"Splatfacto run for export (default: {LATEST_CHECKPOINT} — resolved after training "
            "from outputs/splatfacto/<timestamp>/)."
        ),
    )
    args = parser.parse_args()

    steps = build_steps(
        video=args.video,
        mushroom=args.mushroom,
        mushroom_device=args.mushroom_device,
        mushroom_capture=args.mushroom_capture,
    )

    for step in steps:
        if _run_step(step, args.checkpoint_dir) != 0:
            sys.exit(1)

    print("\nPipeline complete. Check export/ for outputs.")


if __name__ == "__main__":
    main()
