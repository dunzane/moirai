import os
import torch

try:
    import torch_npu  # noqa: F401
    import torch_npu.npu.utils as npu_utils  # noqa: F401

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv('NPUJITCompile', False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
except (ImportError, ModuleNotFoundError):
    IS_NPU_AVAILABLE = False

try:
    import torch_mlu  # noqa: F401

    IS_MLU_AVAILABLE = hasattr(torch, 'mlu') and torch.mlu.is_available()
except (ImportError, ModuleNotFoundError):
    IS_MLU_AVAILABLE = False

try:
    import torch_dipu  # noqa: F401

    IS_DIPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    IS_DIPU_AVAILABLE = False

try:
    import torch_musa  # noqa: F401

    IS_MUSA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    IS_MUSA_AVAILABLE = False


def is_cuda_available() -> bool:
    """Returns True if cuda devices exist."""
    return torch.cuda.is_available()


def is_mlu_available() -> bool:
    """Returns True if Cambricon PyTorch and mlu devices exist."""
    return IS_MLU_AVAILABLE


def is_npu_available() -> bool:
    """Returns True if Ascend PyTorch and npu devices exist."""
    return IS_NPU_AVAILABLE


def is_mps_available() -> bool:
    """Return True if mps devices exist.

    It's specialized for mac m1 chips and require torch version 1.12 or higher.
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def is_dipu_available() -> bool:
    return IS_DIPU_AVAILABLE


def is_musa_available() -> bool:
    return IS_MUSA_AVAILABLE
