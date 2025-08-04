# pylint: disable=inconsistent-quotes
import os
import random
from typing import Dict, Any

import numpy as np
import torch
from packaging.version import parse as parse_version

from moirai.device import get_device_type, set_device_manual_seed
from moirai.dist import get_rank
from moirai.logging import get_logger

_DEFAULT_ENV_LOGGER = get_logger('moirai-utils-env')


def set_tf32_mode(tf32_mode: bool):
    """Enable or disable TF32 mode on supported GPUs and torch versions."""
    device = get_device_type()
    torch_version = parse_version(torch.__version__)

    if tf32_mode:
        if device != 'gpu':
            raise RuntimeError(f"TF32 mode is only supported on GPUs. Current device: {device}")
        if torch_version < parse_version("1.7.0"):
            raise RuntimeError(f"TF32 requires torch>=1.7.0, got {torch.__version__}")
        _DEFAULT_ENV_LOGGER.info('TF32 mode: ENABLED')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        if device == 'gpu':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            _DEFAULT_ENV_LOGGER.info('TF32 mode: DISABLED')


def setup_determinacy(
    seed: int,
    deterministic: bool = False,
    cudnn_enabled: bool = True,
    cudnn_benchmark: bool = True,
    cudnn_deterministic: bool = False
):
    """Set all random seeds and configure deterministic behavior."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_device_manual_seed(seed)

    device = get_device_type()
    torch_version = parse_version(torch.__version__)

    if deterministic:
        if device == 'gpu':
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        if torch_version >= parse_version('1.8.0'):
            torch.use_deterministic_algorithms(True)
        elif torch_version >= parse_version('1.7.0'):
            torch.set_deterministic(True)

        _DEFAULT_ENV_LOGGER.info('Deterministic algorithms: ENABLED')

    if device == 'gpu':
        if not cudnn_enabled:
            torch.backends.cudnn.enabled = False
            _DEFAULT_ENV_LOGGER.info('CUDNN: DISABLED')
        if not cudnn_benchmark:
            torch.backends.cudnn.benchmark = False
            _DEFAULT_ENV_LOGGER.info('CUDNN Benchmark: DISABLED')
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            _DEFAULT_ENV_LOGGER.info('CUDNN Deterministic: ENABLED')


def _get_nested(config: Dict[str, Any], keys: str, default=None):
    """Support nested dict key access like 'CUDNN.ENABLED'"""
    val = config
    for k in keys.split('.'):
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def set_env(env_cfg: Dict[str, Any]):
    """Setup runtime env: tf32, seed, determinacy from config dict."""
    tf32 = env_cfg.get('TF32', False)
    seed = env_cfg.get('SEED')

    set_tf32_mode(tf32)

    if seed is not None:
        rank_seed = seed + get_rank()
        setup_determinacy(
            seed=rank_seed,
            deterministic=env_cfg.get('DETERMINISTIC', False),
            cudnn_enabled=_get_nested(env_cfg, 'CUDNN.ENABLED', True),
            cudnn_benchmark=_get_nested(env_cfg, 'CUDNN.BENCHMARK', True),
            cudnn_deterministic=_get_nested(env_cfg, 'CUDNN.DETERMINISTIC', False),
        )
