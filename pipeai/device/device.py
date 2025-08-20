import logging
import os
from typing import Union, Optional, Any, Dict, List, Tuple

import torch
from torch import nn

from pipeai.version import __version__

# Supported device types
SUPPORTED_DEVICE_TYPES = {"gpu", "mps", "cpu"}
DEVICE_TYPE = "gpu"


def get_device_type() -> str:
    """Get the current device type.

    Returns:
        str: Current device type ('gpu', 'mps', 'cpu')
    """
    return DEVICE_TYPE


def set_device_type(device_type: str) -> None:
    """Set the device type.

    Args:
        device_type (str): Device type, must be one of 'gpu', 'mps', 'cpu'

    Raises:
        ValueError: If the device type is not supported
        TypeError: If the argument type is incorrect
    """
    assert device_type in SUPPORTED_DEVICE_TYPES, f"pipeai at version={__version__} unsupported device type '{device_type}'. "
    global DEVICE_TYPE
    DEVICE_TYPE = device_type


def get_device_count() -> int:
    """Get the number of available devices.

    Returns:
        int: Number of devices

    Raises:
        RuntimeError: If the device type is unknown or unavailable
    """
    if DEVICE_TYPE == "gpu":
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "mps":
        return 1 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 0
    elif DEVICE_TYPE == "cpu":
        return 1
    else:
        raise RuntimeError(f"Unknown device type: '{DEVICE_TYPE}'")


def set_device(device_id: int) -> None:
    """Set the current device.

    Args:
        device_id (int): Device ID

    Raises:
        TypeError: If the argument type is incorrect
        ValueError: If the device ID is invalid
        RuntimeError: If the device type does not support setting devices
    """
    if not isinstance(device_id, int):
        raise TypeError(f"device_id must be an integer, got {type(device_id).__name__}")

    if DEVICE_TYPE == "gpu":
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid GPU device_id {device_id}. "
                f"Available devices: 0-{torch.cuda.device_count() - 1}"
            )
        torch.cuda.set_device(device_id)
    elif DEVICE_TYPE in ("mps", "cpu"):
        if device_id != 0:
            raise ValueError(f"For {DEVICE_TYPE} device, only device_id=0 is supported, got {device_id}")
        pass
    else:
        raise RuntimeError(f"Unknown device type: '{DEVICE_TYPE}'")


def to_device(
        src: Union[torch.Tensor, nn.Module],
        device_id: Optional[int] = None,
        non_blocking: bool = False
) -> Union[torch.Tensor, nn.Module]:
    """Move a tensor or model to the specified device.

    Args:
        src: Tensor or model to move
        device_id: Device ID, use a current device if None
        non_blocking: Whether to use non-blocking transfer (only effective for tensors)

    Returns:
        Tensor or model moved to the target device

    Raises:
        TypeError: If argument types are incorrect
        ValueError: If the device ID is invalid
        RuntimeError: If the device is unavailable
    """
    if not isinstance(src, (torch.Tensor, nn.Module)):
        raise TypeError(f"src must be torch.Tensor or nn.Module, got {type(src).__name__}")

    if device_id is not None and not isinstance(device_id, int):
        raise TypeError(f"device_id must be an integer or None, got {type(device_id).__name__}")

    if not isinstance(non_blocking, bool):
        raise TypeError(f"non_blocking must be a boolean, got {type(non_blocking).__name__}")

    # Only tensors support the non_blocking argument
    kwargs = {"non_blocking": non_blocking} if isinstance(src, torch.Tensor) else {}

    if DEVICE_TYPE == "gpu":
        if device_id is None:
            return src.cuda(**kwargs)
        else:
            if device_id < 0 or device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid GPU device_id {device_id}. "
                    f"Available devices: 0-{torch.cuda.device_count() - 1}"
                )
            return src.to(f"cuda:{device_id}", **kwargs)
    elif DEVICE_TYPE == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS device is not available on this system")
        if device_id is not None and device_id != 0:
            raise ValueError(f"For MPS device, only device_id=0 or None is supported, got {device_id}")
        return src.to("mps", **kwargs)
    elif DEVICE_TYPE == "cpu":
        if device_id is not None and device_id != 0:
            raise ValueError(f"For CPU device, only device_id=0 or None is supported, got {device_id}")
        return src.cpu()
    else:
        raise RuntimeError(f"Unknown device type: '{DEVICE_TYPE}'")


def init_stream() -> torch.cuda.Stream:
    """Initialize a CUDA stream.

    Returns:
        torch.cuda.Stream: CUDA stream object

    Raises:
        RuntimeError: If the device type does not support streams
    """
    if DEVICE_TYPE == "gpu":
        return torch.cuda.Stream()
    else:
        raise RuntimeError(f"Streams are only supported for GPU devices, current device type: '{DEVICE_TYPE}'")


def stream(st: torch.cuda.Stream):
    """Set a context manager for the current CUDA stream.

    Args:
        st: CUDA stream object

    Returns:
        CUDA stream context manager

    Raises:
        TypeError: If the argument type is incorrect
        RuntimeError: If the device type does not support streams
    """
    if DEVICE_TYPE == "gpu":
        if not isinstance(st, torch.cuda.Stream):
            raise TypeError(f"st must be torch.cuda.Stream, got {type(st).__name__}")
        return torch.cuda.stream(st)
    else:
        raise RuntimeError(f"Streams are only supported for GPU devices, current device type: '{DEVICE_TYPE}'")


def current_stream() -> torch.cuda.Stream:
    """Get the current CUDA stream.

    Returns:
        torch.cuda.Stream: Current CUDA stream object

    Raises:
        RuntimeError: If the device type does not support streams
    """
    if DEVICE_TYPE == "gpu":
        return torch.cuda.current_stream()
    else:
        raise RuntimeError(f"Streams are only supported for GPU devices, current device type: '{DEVICE_TYPE}'")


def set_device_manual_seed(seed: int) -> None:
    """Set the random seed for the device.

    Args:
        seed: Random seed

    Raises:
        TypeError: If the argument type is incorrect
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    torch.manual_seed(seed)

    if DEVICE_TYPE == "gpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif DEVICE_TYPE == "mps":
        # MPS uses a unified random seed
        torch.mps.manual_seed(seed)
    # CPU is enough with torch.manual_seed


def data_to_device(data: Any) -> Any:
    """Recursively move tensors in a data structure to the current device.

    Args:
        data: Data structure containing tensors (dict, list, tuple, tensor, etc.)

    Returns:
        Data structure moved to the target device
    """
    if isinstance(data, dict):
        return {k: data_to_device(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [data_to_device(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(data_to_device(item) for item in data)
    elif isinstance(data, torch.Tensor):
        return to_device(data, non_blocking=True)
    else:
        # For other types, return as is
        return data


def set_visible_devices(devices: Optional[str], logger: logging.Logger) -> None:
    """Set visible devices.

    Args:
        devices: Device string, e.g., '0,1,2,3'. Use all devices if None.
        logger: Logger instance

    Raises:
        TypeError: If argument types are incorrect
        RuntimeError: If the device type does not support setting visible devices
    """
    if devices is not None and not isinstance(devices, str):
        raise TypeError(f"devices must be a string or None, got {type(devices).__name__}")

    if not isinstance(logger, logging.Logger):
        raise TypeError(f"logger must be logging.Logger, got {type(logger).__name__}")

    if DEVICE_TYPE == "gpu":
        env_var = "CUDA_VISIBLE_DEVICES"
        if devices is not None:
            os.environ[env_var] = devices
            logger.info(f"Set visible GPU devices to: {devices}")
        else:
            logger.info("Using all available GPU devices")
    elif DEVICE_TYPE in ("mps", "cpu"):
        if devices is not None:
            logger.warning(f"Setting visible devices is not applicable for {DEVICE_TYPE} device type, ignoring")
        else:
            logger.info(f"Using {DEVICE_TYPE} device")
    else:
        raise RuntimeError(f"Unknown device type: '{DEVICE_TYPE}'")


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about the current device.

    Returns:
        Dictionary containing device information
    """
    info = {
        "device_type": DEVICE_TYPE,
        "device_count": get_device_count()
    }

    if DEVICE_TYPE == "gpu":
        info.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        })
    elif DEVICE_TYPE == "mps":
        info.update({
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        })

    return info
