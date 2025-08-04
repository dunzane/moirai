# pylint: disable=inconsistent-quotes
import traceback
from typing import Callable, Dict, Union, Tuple

from moirai.config import init_cfg
from moirai.device import set_device_type,set_visible_devices
from moirai.dist import is_master,dist_wrap
from moirai.logging import get_logger


def training_func(cfg: Dict):
    """Start a training process.

    1. Initialize runner from `cfg`.
    2. Initialize logger (after creating necessary directories).
    3. Call the `train()` method on the runner.

    Args:
        cfg (Dict): Configuration dictionary.
    """
    logger = get_logger('moirai-launcher')
    if is_master():
        logger.info(f'Initializing runner {cfg["RUNNER"]}')

    runner = cfg['RUNNER'](cfg)
    runner.build_logger(logger_name='moirai-training', log_file_name='training_log')

    try:
        runner.train()
    except BaseException as e:
        runner.logger.error(traceback.format_exc())
        raise RuntimeError("Training failed with an unhandled exception.") from e


def launch_training(cfg: Union[Dict, str], devices: str = None, node_rank: int = 0):
    """Launch a training process with optional distributed support.

    Supports DDP (NCCL backend) if GPU_NUM > 1.

    Args:
        cfg (Union[Dict, str]): Configuration dictionary or path.
        devices (str, optional): CUDA_VISIBLE_DEVICES setting.
        node_rank (int): Current node rank in distributed setting.
    """
    logger = get_logger('moirai-launcher')
    logger.info('Launching Moirai training.')

    cfg = init_cfg(cfg, node_rank == 0)

    # Device resolution priority: DEVICE > (GPU_NUM / MLU_NUM) > CPU
    if 'DEVICE' in cfg:
        set_device_type(cfg['DEVICE'])
        device_num = cfg.get('DEVICE_NUM', 0)
    elif cfg.get('GPU_NUM', 0) > 0:
        set_device_type('gpu')
        device_num = cfg['GPU_NUM']
    elif cfg.get('MLU_NUM', 0) > 0:
        set_device_type('mlu')
        device_num = cfg['MLU_NUM']
    else:
        set_device_type('cpu')
        device_num = 0

    if devices and device_num > 0:
        set_visible_devices(devices,logger)

    dist_wrapper = dist_wrap(
        training_func,
        node_num=cfg.get('DIST_NODE_NUM', 1),
        device_num=device_num,
        node_rank=node_rank,
        dist_backend=cfg.get('DIST_BACKEND'),
        init_method=cfg.get('DIST_INIT_METHOD'),
    )
    dist_wrapper(cfg)


def launch_runner(cfg: Union[Dict, str],
                  fn: Callable,
                  args: Tuple = (),
                  device_type: str = 'gpu',
                  devices: str = None):
    """Initialize runner and invoke given function `fn`.

    Args:
        cfg (Union[Dict, str]): Configuration dictionary or path.
        fn (Callable): Callback function called as `fn(cfg, runner, *args)`.
        args (Tuple): Additional arguments passed to `fn`.
        device_type (str): One of ['cpu', 'gpu', 'mlu'].
        devices (str, optional): CUDA_VISIBLE_DEVICES setting.
    """
    logger = get_logger('moirai-launcher')
    logger.info('Launching Moirai runner.')

    cfg = init_cfg(cfg, True)
    set_device_type(device_type)

    if devices and device_type != 'cpu':
        set_visible_devices(devices,logger)

    runner = cfg['RUNNER'](cfg)
    return fn(cfg, runner, *args)
