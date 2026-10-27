import os

import torch


def pick_device() -> torch.device:
    """Best available accelerator, overridable with VIBEBG_DEVICE (cpu, mps, cuda)."""
    forced = os.environ.get("VIBEBG_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
