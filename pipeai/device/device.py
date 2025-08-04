import logging
import os
from typing import Union

import torch
from torch import nn

try:
    __import__('torch_mlu')
except ImportError:
    pass

_DEVICE_TYPE = 'gpu'


def get_device_type() -> str:
    return _DEVICE_TYPE


def set_device_type(device_type: str):
    global _DEVICE_TYPE
    if device_type not in ['gpu', 'mlu', 'cpu']:
        raise ValueError('Unknown device type!')
    _DEVICE_TYPE = device_type


def get_device_count() -> int:
    if _DEVICE_TYPE == 'gpu':
        return torch.cuda.device_count()
    elif _DEVICE_TYPE == 'mlu':
        return torch.mlu.device_count()
    elif _DEVICE_TYPE == 'cpu':
        return 0
    else:
        raise ValueError('Unknown device type!')


def set_device(device_id: int):
    if _DEVICE_TYPE == 'gpu':
        torch.cuda.set_device(device_id)
    elif _DEVICE_TYPE == 'mlu':
        torch.mlu.set_device(device_id)
    else:
        raise ValueError('Unknown device type!')


def to_device(src: Union[torch.Tensor, nn.Module], device_id: int = None,
              non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
    kwargs = {'non_blocking': non_blocking} if isinstance(src, torch.Tensor) else {}
    if _DEVICE_TYPE == 'gpu':
        if device_id is None:
            return src.cuda(**kwargs)
        else:
            return src.to('cuda:{:d}'.format(device_id), **kwargs)
    elif _DEVICE_TYPE == 'mlu':
        if device_id is None:
            return src.mlu(**kwargs)
        else:
            return src.to('mlu:{:d}'.format(device_id), **kwargs)
    elif _DEVICE_TYPE == 'cpu':
        return src.cpu()
    else:
        raise ValueError('Unknown device type!')


def init_stream():
    if _DEVICE_TYPE == 'gpu':
        return torch.cuda.Stream()
    elif _DEVICE_TYPE == 'mlu':
        return torch.mlu.Stream()
    else:
        raise ValueError('Unknown device type!')


def stream(st):
    if _DEVICE_TYPE == 'gpu':
        return torch.cuda.stream(st)
    elif _DEVICE_TYPE == 'mlu':
        return torch.mlu.stream(st)
    else:
        raise ValueError('Unknown device type!')


def current_stream():
    if _DEVICE_TYPE == 'gpu':
        return torch.cuda.current_stream()
    elif _DEVICE_TYPE == 'mlu':
        return torch.mlu.current_stream()
    else:
        raise ValueError('Unknown device type!')


def set_device_manual_seed(seed: int):
    torch.manual_seed(seed)
    if _DEVICE_TYPE == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif _DEVICE_TYPE == 'mlu':
        torch.mlu.manual_seed(seed)
        torch.mlu.manual_seed_all(seed)


def data_to_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data_to_device(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = data_to_device(v)
    elif isinstance(data, tuple):
        data = tuple(data_to_device(list(data)))
    elif isinstance(data, torch.Tensor):
        data = to_device(data, non_blocking=True)
    return data


def set_visible_devices(devices: str, logger: logging.Logger):
    """Set environment variable `CUDA_VISIBLE_DEVICES` to select GPU devices.

    Examples:
        set_devices('0,1,2,3')

    Args:
        devices (str): environment variable `CUDA_VISIBLE_DEVICES` value
        logger: output
    """
    if devices is not None:
        os.environ[{
            'gpu': 'CUDA_VISIBLE_DEVICES',
            'mlu': 'MLU_VISIBLE_DEVICES'
        }[get_device_type()]] = devices
        logger.info('Use devices {}.'.format(devices))
    else:
        logger.info('Use all devices.')
