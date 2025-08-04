# pylint: disable=inconsistent-quotes
import copy
import hashlib
import json
import os
import shutil
import sys
import types
import importlib.util
from datetime import datetime
from typing import Dict, Set, List, Union, Any, TypeVar

from moirai.checkpoint import get_ckpt_save_dir
from moirai.config import Config

TRAINING_INDEPENDENT_FLAG = '_TRAINING_INDEPENDENT'

DEFAULT_TRAINING_INDEPENDENT_KEYS = {
    'DIST_BACKEND',
    'DIST_INIT_METHOD',
    'TRAIN.CKPT_SAVE_STRATEGY',
    'TRAIN.DATA.NUM_WORKERS',
    'TRAIN.DATA.PIN_MEMORY',
    'TRAIN.DATA.PREFETCH',
    'VAL'
}


def get_training_dependent_config(
        cfg: Dict,
        except_keys: Union[Set[str], List[str]] = None
) -> Dict:
    """
    Extract training-dependent parts of the configuration.

    Recursively removes keys from the config that are known to be training-independent,
    including those listed in DEFAULT_TRAINING_INDEPENDENT_KEYS and those dynamically
    marked by TRAINING_INDEPENDENT_FLAG in the config.

    Args:
        cfg (Dict): The complete configuration dictionary.
        except_keys (Set[str] or List[str], optional): Additional keys to exclude.

    Returns:
        Dict: A deep copy of the config containing only training-dependent keys.
    """
    cfg_copy = copy.deepcopy(cfg)

    exclude_keys = set(DEFAULT_TRAINING_INDEPENDENT_KEYS)
    if cfg_copy.get(TRAINING_INDEPENDENT_FLAG):
        exclude_keys.update(cfg_copy.pop(TRAINING_INDEPENDENT_FLAG))

    if except_keys:
        exclude_keys.update(except_keys)

    def _filter_recursive(d: Dict, prefix: str = '') -> Dict:
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if any(full_key == ex or full_key.startswith(f"{ex}.") for ex in exclude_keys):
                continue
            if isinstance(v, dict):
                filtered = _filter_recursive(v, full_key)
                if filtered:
                    result[k] = filtered
            else:
                result[k] = v
        return result

    return _filter_recursive(cfg_copy)


def config_str(cfg: Dict, indent: str = '') -> str:
    """
    Generate a human-readable string representation of the configuration.

    Args:
        cfg (Dict): Configuration dictionary.
        indent (str): Indentation used for nested configs.

    Returns:
        str: Formatted configuration string.
    """
    lines = []
    for k, v in cfg.items():
        if k == TRAINING_INDEPENDENT_FLAG:
            continue
        if isinstance(v, dict):
            lines.append(f"{indent}{k}:")
            lines.append(config_str(v, indent + '  '))
        elif isinstance(v, types.FunctionType):
            lines.append(f"{indent}{k}: {v.__name__}")
        else:
            lines.append(f"{indent}{k}: {v}")
    return '\n'.join(lines)


def config_md5(cfg: Any) -> str:
    """Compute MD5 hash of a config object in a deterministic way.
    Handles common non-JSON-serializable types like TypeVar, set, datetime, etc.
    """

    def to_serializable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return sorted(to_serializable(v) for v in obj)  # Convert set to sorted list for determinism
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO string
        elif isinstance(obj, TypeVar):
            return f"TypeVar({obj.__name__})"  # Represent TypeVar as a string
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return to_serializable(obj.__dict__)
        else:
            # Fallback: convert to string (may lose some information)
            return str(obj)

    cfg_serialized = to_serializable(cfg)
    cfg_json = json.dumps(cfg_serialized, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(cfg_json.encode('utf-8')).hexdigest()


def save_config_str(cfg: Dict, file_path: str) -> None:
    """
    Save the stringified configuration to a file.

    Args:
        cfg (Dict): Configuration dictionary.
        file_path (str): Path to save the config file.
    """
    with open(file_path, 'w') as f:
        f.write(config_str(cfg))


def copy_config_file(cfg_file_path: str, save_dir: str):
    """
    Copy the original config file to the target save directory.

    Args:
        cfg_file_path (str): Source config file path.
        save_dir (str): Destination directory.
    """
    if os.path.isfile(cfg_file_path) and os.path.isdir(save_dir):
        cfg_file_name = os.path.basename(cfg_file_path)
        shutil.copyfile(cfg_file_path, os.path.join(save_dir, cfg_file_name))


def import_config(path: str, verbose: bool = True):
    """
    Import a Python configuration file and return the CFG object.

    The function supports both relative module paths (e.g., "configs.default")
    and absolute file paths (e.g., "./configs/default.py"). It assumes the
    config file defines a variable named `CFG`.

    Args:
        path (str): Path to the config file or module.
        verbose (bool): Whether to print the config as a formatted string.

    Returns:
        Any: The imported configuration object `CFG`.

    Raises:
        FileNotFoundError: If the specified config path does not exist.
        AttributeError: If the imported config does not define `CFG`.
    """
    if os.path.isfile(path):
        abs_path = os.path.abspath(path)
        spec = importlib.util.spec_from_file_location("config", abs_path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        cfg = cfg_module.CFG
    else:
        if path.endswith('.py'):
            path = path[:-3].replace('/', '.').replace('\\', '.')
        cfg_name = path.split('.')[-1]
        cfg_module = __import__(path, fromlist=[cfg_name])
        cfg = cfg_module.CFG

    if verbose:
        print(config_str(cfg))
    return cfg


def convert_config(cfg: Dict) -> Config:
    """
    Convert a dictionary into a Config object and attach its MD5.

    Args:
        cfg (Dict): Configuration dictionary.

    Returns:
        Config: Converted configuration.
    """
    if not isinstance(cfg, Config):
        cfg = Config(cfg)
    if cfg.get('MD5') is None:
        cfg['MD5'] = config_md5(cfg)
    return cfg


def init_cfg(cfg: Union[Dict, str], save: bool = False) -> Config:
    """
    Initialize configuration, convert to Config, compute MD5, and optionally save.

    Args:
        cfg (Dict or str): Config dictionary or path to config module.
        save (bool): Whether to save config and copy the original file.

    Returns:
        Config: Initialized configuration with MD5.
    """
    if isinstance(cfg, str):
        cfg_path = cfg
        cfg = import_config(cfg, verbose=save)
    else:
        cfg_path = None

    cfg = convert_config(cfg)
    ckpt_dir = get_ckpt_save_dir(cfg)

    if save and not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        save_config_str(cfg, os.path.join(ckpt_dir, 'cfg.txt'))
        if cfg_path:
            copy_config_file(cfg_path, ckpt_dir)

    return cfg
