"""Shared environment checks and venv CLI resolution for pipeline scripts."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_BIN = Path(sys.executable).resolve().parent

# Pinned in pyproject.toml / uv.lock for Linux CUDA hosts
CUDA_TORCH_VERSION = "2.5.1"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


def resolve_cli(name: str) -> str:
    """Resolve a console script from the active venv (works without `uv run` on PATH)."""
    venv_exe = VENV_BIN / name
    if venv_exe.is_file():
        return str(venv_exe)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(
        f"Cannot find `{name}`. From the project root run:\n"
        f"  uv sync --prerelease=allow\n"
        f"Then retry, or use: uv run --no-sync python <script>"
    )


def require_colmap_binary() -> None:
    if shutil.which("colmap") is None:
        raise SystemExit(
            "COLMAP is not on PATH (required by ns-process-data even with --skip-colmap).\n\n"
            "  Linux:  sudo apt-get update && sudo apt-get install -y colmap\n"
            "See https://colmap.github.io/install.html"
        )


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is not on PATH.\n\n"
            "  Linux:  sudo apt-get install -y ffmpeg"
        )


def check_cuda_torch(*, strict: bool = False) -> list[str]:
    """
    Return human-readable problems with the current PyTorch CUDA install.
    When strict=True, also treat missing +cu build on Linux as an error.
    """
    import torch

    issues: list[str] = []
    if not torch.cuda.is_available():
        issues.append(
            "PyTorch cannot see a CUDA GPU (torch.cuda.is_available() is False)."
        )
        if sys.platform.startswith("linux"):
            issues.append(
                f"On Linux GPU hosts, install pinned wheels:\n"
                f"  uv run --no-sync python scripts/ensure_env.py --fix-cuda\n"
                f"Or: ./scripts/setup_cloud.sh"
            )
        return issues

    version = torch.__version__
    cuda_ver = getattr(torch.version, "cuda", None)
    if strict and sys.platform.startswith("linux"):
        if "+cu" not in version:
            issues.append(
                f"PyTorch build looks CPU-only ({version}). "
                f"Expected {CUDA_TORCH_VERSION}+cu124 on Linux CUDA hosts."
            )
        elif cuda_ver and not str(cuda_ver).startswith("12."):
            issues.append(
                f"PyTorch CUDA runtime is {cuda_ver}; this project pins cu124 for cloud GPUs."
            )

    try:
        torch.zeros(1, device="cuda")
    except Exception as exc:  # noqa: BLE001 — surface driver/wheel mismatch
        issues.append(f"CUDA tensor allocation failed: {exc}")

    return issues


def cuda_torch_install_hint() -> str:
    py = VENV_BIN / "python"
    return (
        f"uv run --no-sync python scripts/ensure_env.py --fix-cuda\n"
        f"# or: uv pip install --python {py} --reinstall "
        f"torch=={CUDA_TORCH_VERSION} torchvision==0.20.1 torchaudio=={CUDA_TORCH_VERSION} "
        f"--index-url {CUDA_TORCH_INDEX}"
    )
