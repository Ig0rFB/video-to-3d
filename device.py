import torch


def gsplat_cuda_available() -> bool:
    """True when gsplat's compiled CUDA extension loaded (required for splatfacto training)."""
    if not torch.cuda.is_available():
        return False
    try:
        from gsplat.cuda._backend import _C
    except (ImportError, RuntimeError, OSError):
        return False
    return _C is not None


def get_device() -> str:
    """
    Returns the device string: 'cuda' or 'cpu'.
    This project is intended to run on CUDA-enabled machines.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[device] Using: {device}")
    return device
