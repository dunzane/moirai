import functools

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Union, Optional, Callable

import torch

from pipeai import Config


class BaseVisBackend(metaclass=ABCMeta):
    """Base class for visualization backend.

    All backends must inherit ``BaseVisBackend`` and implement
    the required functions.

    Args:
        save_dir (str, optional): The root directory to save
            the files produced by the backend.
    """

    def __init__(self, save_dir: str = None):
        self._save_dir = save_dir
        self._env_initialized = False

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this visualization
        backend.

        The experiment attribute can get the visualization backend, such as
        wandb, tensorboard. If you want to write other data, such as writing a
        table, you can directly get the visualization backend through
        experiment.
        """
        pass

    @abstractmethod
    def _init_env(self) -> Any:
        """Setup env for VisBackend."""
        pass

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module,
                  data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        pass

    def close(self) -> None:
        """Close an opened object."""
        pass


def force_init_env(old_func: Callable) -> Any:
    """Those methods decorated by ``force_init_env`` will be forced to call
    ``_init_env`` if the instance has not been fully initiated. This function
    will decorated all the `add_xxx` method and `experiment` method, because
    `VisBackend` is initialized only when used its API.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``_init_env`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `_init_env` method.
        if not hasattr(obj, '_init_env'):
            raise AttributeError(f'{type(obj)} does not have _init_env '
                                 'method.')
        # If instance does not have `_env_initialized` attribute or
        # `_env_initialized` is False, call `_init_env` and set
        # `_env_initialized` to True
        if not getattr(obj, '_env_initialized', False):
            obj._init_env()  # type: ignore
            obj._env_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper
