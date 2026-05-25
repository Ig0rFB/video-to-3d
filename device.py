import torch


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
