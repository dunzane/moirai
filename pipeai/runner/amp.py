import torch
from contextlib import contextmanager
from typing import Optional


@contextmanager
def autocast(device_type: Optional[str] = None,
             dtype: Optional[torch.dtype] = None,
             enabled: bool = True,
             cache_enabled: Optional[bool] = None):
    """Simplified autocast wrapper for PyTorch >= 1.10.

    This wrapper provides a unified interface for mixed-precision training
    with CPU or CUDA devices. Other device backends (MLU, NPU, MUSA, etc.)
    are not supported in this simplified version.

    Args:
        device_type (str, optional): 'cuda' or 'cpu'. Defaults to a current device.
        dtype (torch.dtype, optional): Precision type (float16 or bfloat16).
        enabled (bool): Whether autocast should be enabled. Default: True.
        cache_enabled (bool, optional): Whether to enable the autocast cache.
    """
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Default dtypes
    if device_type == "cuda":
        if dtype is None:
            dtype = torch.float16
    elif device_type == "cpu":
        if dtype is None:
            dtype = torch.bfloat16
        if dtype != torch.bfloat16:
            raise ValueError("CPU autocast only supports torch.bfloat16.")
    else:
        if enabled:
            raise ValueError(f"Unsupported device_type {device_type}. Use 'cuda' or 'cpu'.")

    if cache_enabled is None:
        cache_enabled = torch.is_autocast_cache_enabled()

    with torch.autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=cache_enabled
    ):
        yield
