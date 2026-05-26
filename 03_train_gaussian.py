import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from device import get_device, gsplat_cuda_available
from env_utils import (
    check_cuda_torch,
    cuda_torch_install_hint,
    require_colmap_binary,
    resolve_cli,
)
from colmap_paths import find_colmap_model
from mushroom_paths import add_mushroom_arguments, resolve_mushroom_paths

# Path relative to --output-dir for ns-process-data when reusing an existing COLMAP model
NS_COLMAP_REL = Path("colmap/sparse/0")


def _link_colmap_into_ns_data(colmap_model_path: Path, ns_data_dir: Path) -> str:
    """
    nerfstudio expects --colmap-model-path relative to the output dir when --skip-colmap is set.
    Symlink our sparse model into nerfstudio_data/colmap/sparse/0.
    """
    dest = ns_data_dir / NS_COLMAP_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(colmap_model_path.resolve())
    return str(NS_COLMAP_REL)


def train(
    image_dir: str,
    colmap_model_path: str,
    output_dir: str,
    ns_data_dir: str = "nerfstudio_data",
) -> None:
    device = get_device()
    if device != "cuda":
        raise SystemExit(
            "\n[splatfacto] This project is configured to run on CUDA-enabled machines only.\n\n"
            "Make sure your environment can see an NVIDIA GPU:\n"
            "  nvidia-smi\n"
            "  uv run --no-sync python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"\n"
        )

    cuda_issues = check_cuda_torch(strict=True)
    if cuda_issues:
        raise SystemExit(
            "\n[splatfacto] CUDA PyTorch is not usable on this host:\n"
            + "\n".join(f"  - {m}" for m in cuda_issues)
            + "\n\n"
            + cuda_torch_install_hint()
            + "\n"
        )

    if not gsplat_cuda_available():
        raise SystemExit(
            "\n[splatfacto] gsplat's CUDA rasteriser is not available on this machine.\n\n"
            "Fix by reinstalling gsplat in this environment, and ensure you are on a CUDA host.\n"
        )

    ns_path = Path(ns_data_dir)
    colmap_rel = _link_colmap_into_ns_data(Path(colmap_model_path), ns_path)
    require_colmap_binary()
    ns_process_data = resolve_cli("ns-process-data")

    subprocess.run(
        [
            ns_process_data,
            "images",
            "--data",
            image_dir,
            "--output-dir",
            ns_data_dir,
            "--skip-colmap",
            "--colmap-model-path",
            colmap_rel,
            "--no-gpu",
        ],
        check=True,
    )

    subprocess.run(
        [
            resolve_cli("ns-train"),
            "splatfacto",
            "--data",
            ns_data_dir,
            "--output-dir",
            output_dir,
            "--max-num-iterations",
            "30000",
            "--pipeline.model.cull-alpha-thresh",
            "0.005",
            "--machine.device-type",
            "cuda",
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

    colmap_model = str(find_colmap_model(Path(args.colmap_dir)))
    return args.image_dir, colmap_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train splatfacto (use: uv run --no-sync python 03_train_gaussian.py …)"
    )
    parser.add_argument("--image-dir", default="frames/")
    parser.add_argument(
        "--colmap-dir",
        default="colmap_workspace/",
        help="Workspace from 02_run_colmap (uses sparse/0 under this dir).",
    )
    parser.add_argument(
        "--colmap-model-path",
        default=None,
        help="Explicit COLMAP sparse model directory (overrides --colmap-dir).",
    )
    add_mushroom_arguments(parser)
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--ns-data-dir", default="nerfstudio_data")
    args = parser.parse_args()

    if args.colmap_model_path:
        image_dir = args.image_dir
        colmap_model_path = args.colmap_model_path
    else:
        image_dir, colmap_model_path = _resolve_paths(args)

    train(image_dir, colmap_model_path, args.output_dir, args.ns_data_dir)
