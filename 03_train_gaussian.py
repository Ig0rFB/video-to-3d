import argparse
import os
import subprocess
import sys
from pathlib import Path

from device import get_device, gsplat_cuda_available
from mushroom_paths import add_mushroom_arguments, resolve_mushroom_paths
from patch_nerfstudio_mps import patch_splatfacto

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
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    patch_splatfacto()

    if device in ("mps", "cpu") and not gsplat_cuda_available():
        raise SystemExit(
            "\n[splatfacto] gsplat's CUDA rasteriser is not available on this machine "
            "(expected on Apple Silicon without an NVIDIA GPU).\n\n"
            "COLMAP and ns-process-data can still run here. For Gaussian training, use a "
            "Linux host with CUDA, or copy `nerfstudio_data/` to such a machine and run:\n"
            "  uv run python 03_train_gaussian.py\n"
            "  (on the CUDA host, `get_device()` will select cuda automatically.)\n\n"
            "See README.md → Apple Silicon.\n"
        )

    ns_path = Path(ns_data_dir)
    colmap_rel = _link_colmap_into_ns_data(Path(colmap_model_path), ns_path)

    subprocess.run(
        [
            "ns-process-data",
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
            sys.executable,
            str(Path(__file__).resolve().parent / "run_ns_train.py"),
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
