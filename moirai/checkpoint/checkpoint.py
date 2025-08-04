# pylint: disable=inconsistent-quotes
import os
import re
import glob
import shutil
from logging import Logger
from typing import Dict, List, Tuple, Union, Optional

import torch

from moirai.logging import get_logger

_DEFAULT_CKPT_LOGGER = get_logger('moirai-checkpoint')


def get_ckpt_save_dir(cfg: Dict):
    """Get real ckpt save dir with MD5.

    Args:
        cfg (Dict): config.

    Returns:
        str: Real ckpt save dir
    """

    return os.path.join(cfg['TRAIN']['CKPT_SAVE_DIR'], cfg['MD5'])


def get_last_ckpt_path(ckpt_save_dir: str, name_pattern: str = r'^.+_[\d]*.pt$') -> str:
    r"""Get last checkpoint path in `ckpt_save_dir`
    checkpoint files will be sorted by name

    Args:
        ckpt_save_dir (str): checkpoint save directory
        name_pattern (str): re pattern for checkpoint file name, default is r'^.+_[\d]*.pt$'

    Returns:
        checkpoint path (str): last checkpoint path in `ckpt_save_dir`
    """

    ckpt_list = [f for f in os.listdir(ckpt_save_dir) if re.search(name_pattern, f) is not None]
    ckpt_list.sort()
    return os.path.join(ckpt_save_dir, ckpt_list[-1])


def load_ckpt(
        ckpt_save_dir: str,
        ckpt_path: Optional[str] = None,
        logger: Logger = _DEFAULT_CKPT_LOGGER,
        device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    Load a model checkpoint from a file.

    Args:
        ckpt_save_dir (str): Directory where checkpoints are saved.
        ckpt_path (str, optional): Specific checkpoint file path. If None, load the last checkpoint.
        logger (Logger): Logger instance for logging.
        device (str or torch.device, optional): Device to map checkpoint to. Default is
            auto ("cuda" if available else "cpu").

    Returns:
        dict: Checkpoint dictionary.
    """
    if ckpt_path is None:
        if not ckpt_save_dir:
            raise ValueError("`ckpt_save_dir` must be provided when `ckpt_path` is None")
        ckpt_path = get_last_ckpt_path(ckpt_save_dir)

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint path does not exist: {ckpt_path}")
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    logger.info(f"Loading checkpoint from: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        logger.exception(f"Failed to load checkpoint: {e}")
        raise e

    return checkpoint


def save_ckpt(
        ckpt: Dict,
        ckpt_path: str,
        logger: Logger = _DEFAULT_CKPT_LOGGER
):
    """
    Save model checkpoint to file.

    Args:
        ckpt (Dict): Checkpoint dictionary to save.
        ckpt_path (str): Destination file path.
        logger (Logger): Logger instance.
    """
    try:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(ckpt, ckpt_path)
        logger.info(f"Checkpoint saved to: {ckpt_path}")
    except Exception as e:
        logger.exception(f"Failed to save checkpoint to {ckpt_path}: {e}")
        raise e


def need_to_remove_last_ckpt(
        last_epoch: int,
        ckpt_save_strategy: Union[int, List[int], Tuple[int]]
) -> bool:
    """
    Determine whether the last checkpoint should be removed.

    Args:
        last_epoch (int): Index of the last epoch.
        ckpt_save_strategy (Union[int, List[int], Tuple[int]]): Checkpoint saving strategy.

    Returns:
        bool: True if the last checkpoint should be removed, else False.
    """
    if ckpt_save_strategy is None:
        return True
    elif isinstance(ckpt_save_strategy, int):
        return last_epoch % ckpt_save_strategy != 0
    elif isinstance(ckpt_save_strategy, (list, tuple)):
        return last_epoch not in ckpt_save_strategy
    else:
        raise TypeError("ckpt_save_strategy must be None, int, list, or tuple")


def backup_last_ckpt(
        last_ckpt_path: str,
        epoch: int,
        ckpt_save_strategy: Union[int, List[int], Tuple[int]],
        logger: Logger = _DEFAULT_CKPT_LOGGER
):
    """
    Backup the last checkpoint if it is about to be removed.
    The original checkpoint will be renamed with `.bak` extension.

    Args:
        last_ckpt_path (str): Path to the last checkpoint file.
        epoch (int): Current epoch number.
        ckpt_save_strategy (Union[int, List[int], Tuple[int]]): Checkpoint saving strategy.
        logger (Logger): Logger instance.
    """
    last_epoch = epoch - 1

    if last_epoch <= 0:
        return  # No checkpoint to back up

    if need_to_remove_last_ckpt(last_epoch, ckpt_save_strategy):
        bak_path = last_ckpt_path + ".bak"
        try:
            if os.path.exists(last_ckpt_path):
                shutil.move(last_ckpt_path, bak_path)
                logger.info(f"Backed up last checkpoint: {last_ckpt_path} -> {bak_path}")
            else:
                logger.warning(f"Last checkpoint not found: {last_ckpt_path}, skip backup.")
        except Exception as e:
            logger.exception(f"Failed to backup checkpoint {last_ckpt_path}: {e}")
            raise e


def clear_ckpt(
        ckpt_save_dir: str,
        name_pattern: str = '*.pt.bak',
        logger: Logger = _DEFAULT_CKPT_LOGGER
):
    """
    Clear all checkpoint backup files matching the given pattern.

    Args:
        ckpt_save_dir (str): Directory where checkpoints are saved.
        name_pattern (str): Pattern for backup files (default: '*.pt.bak').
        logger (Logger): Logger instance.
    """
    search_path = os.path.join(ckpt_save_dir, name_pattern)
    ckpt_list = glob.glob(search_path)

    if not ckpt_list:
        logger.info(f"No backup checkpoints found in {ckpt_save_dir} matching '{name_pattern}'")
        return

    for ckpt_file in ckpt_list:
        try:
            os.remove(ckpt_file)
            logger.info(f"Removed backup checkpoint: {ckpt_file}")
        except Exception as e:
            logger.exception(f"Failed to remove {ckpt_file}: {e}")
            raise e
