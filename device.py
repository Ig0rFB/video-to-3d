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
    Returns the best available device string: 'cuda', 'mps', or 'cpu'.
    All pipeline scripts import this function — never hardcode device elsewhere.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[device] Using: {device}")
    return device
