import gc
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


def free_device_memory():
    """Hand memory cached by dropped models back to the system."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
