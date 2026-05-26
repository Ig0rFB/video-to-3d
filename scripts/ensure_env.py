#!/usr/bin/env python3
"""
Preflight checks for video-to-3d. Safe to run after every `uv sync`.

  uv run --no-sync python scripts/ensure_env.py
  uv run --no-sync python scripts/ensure_env.py --fix-cuda --require-cuda
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env_utils import (  # noqa: E402
    CUDA_TORCH_INDEX,
    CUDA_TORCH_VERSION,
    VENV_BIN,
    check_cuda_torch,
    require_colmap_binary,
    require_ffmpeg,
    resolve_cli,
)


def _uv_cmd() -> str:
    return shutil.which("uv") or "uv"


def _fix_cuda_torch() -> None:
    py = Path(sys.executable)
    uv = _uv_cmd()

    subprocess.run(
        [uv, "pip", "uninstall", "--python", str(py), "torch", "torchvision", "torchaudio"],
        cwd=ROOT,
        check=False,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(py),
            "--reinstall",
            f"torch=={CUDA_TORCH_VERSION}",
            "torchvision==0.20.1",
            f"torchaudio=={CUDA_TORCH_VERSION}",
            "--index-url",
            CUDA_TORCH_INDEX,
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"[ensure_env] Installed torch {CUDA_TORCH_VERSION} from {CUDA_TORCH_INDEX}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify system tools and Python env.")
    parser.add_argument(
        "--fix-cuda",
        action="store_true",
        help="Reinstall pinned CUDA 12.4 PyTorch wheels into .venv (Linux GPU hosts).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA PyTorch is not working (use on cloud GPU instances).",
    )
    parser.add_argument(
        "--skip-system",
        action="store_true",
        help="Skip ffmpeg/COLMAP checks.",
    )
    args = parser.parse_args()

    if args.fix_cuda:
        if not sys.platform.startswith("linux"):
            raise SystemExit("--fix-cuda is only for Linux CUDA hosts.")
        _fix_cuda_torch()

    ok = True

    if not args.skip_system:
        try:
            require_ffmpeg()
            print("[ensure_env] ffmpeg: ok")
        except SystemExit as exc:
            print(f"[ensure_env] {exc}", file=sys.stderr)
            ok = False
        try:
            require_colmap_binary()
            print("[ensure_env] colmap: ok")
        except SystemExit as exc:
            print(f"[ensure_env] {exc}", file=sys.stderr)
            ok = False

    try:
        resolve_cli("ns-process-data")
        print("[ensure_env] ns-process-data: ok")
    except SystemExit as exc:
        print(f"[ensure_env] {exc}", file=sys.stderr)
        ok = False

    try:
        resolve_cli("ns-train")
        print("[ensure_env] ns-train: ok")
    except SystemExit:
        print("[ensure_env] ns-train: not found (training uses run_ns_train.py)")

    import torch

    print(
        f"[ensure_env] torch {torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"cuda_ver={getattr(torch.version, 'cuda', None)}"
    )

    if args.require_cuda or args.fix_cuda:
        issues = check_cuda_torch(strict=True)
        for msg in issues:
            print(f"[ensure_env] ERROR: {msg}", file=sys.stderr)
            ok = False
        if ok and (args.require_cuda or args.fix_cuda):
            print("[ensure_env] CUDA PyTorch: ok")

    if not ok:
        raise SystemExit(1)

    print("[ensure_env] All checks passed.")


if __name__ == "__main__":
    main()
