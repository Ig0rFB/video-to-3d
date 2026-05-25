import argparse
import subprocess
from pathlib import Path

from device import get_device
from mushroom_paths import add_mushroom_arguments, resolve_mushroom_paths


def train(
    image_dir: str,
    colmap_model_path: str,
    output_dir: str,
    ns_data_dir: str = "nerfstudio_data",
) -> None:
    device = get_device()

    subprocess.run(
        [
            "ns-process-data",
            "images",
            "--data",
            image_dir,
            "--output-dir",
            ns_data_dir,
            "--colmap-model-path",
            colmap_model_path,
            "--no-gpu",
        ],
        check=True,
    )

    subprocess.run(
        [
            "ns-train",
            "splatfacto",
            "--data",
            ns_data_dir,
            "--output-dir",
            output_dir,
            "--max-num-iterations",
            "30000",
            "--pipeline.model.cull-alpha-thresh",
            "0.005",
            "--machine.device",
            device,
        ],
        check=True,
    )


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    if args.mushroom:
        image_dir, colmap_model = resolve_mushroom_paths(
            args.mushroom,
            device=args.mushroom_device,
            capture=args.mushroom_capture,
        )
        return str(image_dir), str(colmap_model)

    colmap_model = str(Path(args.colmap_dir) / "sparse" / "0")
    if not _has_colmap_at(colmap_model):
        nested = str(Path(args.colmap_dir) / "sparse" / "0" / "0")
        if _has_colmap_at(nested):
            colmap_model = nested
    return args.image_dir, colmap_model


def _has_colmap_at(path: str) -> bool:
    p = Path(path)
    return (p / "cameras.bin").exists() or (p / "cameras.txt").exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="frames/")
    parser.add_argument(
        "--colmap_dir",
        default="colmap_workspace/",
        help="Workspace from 02_run_colmap (uses sparse/0 under this dir).",
    )
    parser.add_argument(
        "--colmap-model-path",
        default=None,
        help="Explicit COLMAP sparse model directory (overrides --colmap_dir).",
    )
    add_mushroom_arguments(parser)
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--ns_data_dir", default="nerfstudio_data")
    args = parser.parse_args()

    if args.colmap_model_path:
        image_dir = args.image_dir
        colmap_model_path = args.colmap_model_path
    else:
        image_dir, colmap_model_path = _resolve_paths(args)

    train(image_dir, colmap_model_path, args.output_dir, args.ns_data_dir)
