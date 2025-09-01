import copy
import os
import numpy as np
import os.path as osp
from json import dump
from PIL import Image
from typing import Union, Optional

import torch

from pipeai import Config
from pipeai.config import save_config_str
from pipeai.registry import VISBACKENDS
from pipeai.visualization import BaseVisBackend, force_init_env


@VISBACKENDS.register_module()
class LocalVisBackend(BaseVisBackend):
    """Local visualization backend class.

    It writes images, configs, scalars, etc.
    to the local hard disk. You can get the drawing backend
    through the experiment property for custom drawing.

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. If it is None, it means no data
            is stored.
        img_save_dir (str): The directory to save images.
            Defaults to 'vis_image'.
        config_save_file (str): The file name to save config.
            Defaults to 'config.py'.
        scalar_save_file (str): The file name to save scalar values.
            Defaults to 'scalars.json'.
    """

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json'):
        assert config_save_file.split('.')[-1] == 'py'
        assert scalar_save_file.split('.')[-1] == 'json'

        super().__init__(save_dir)
        self._img_save_dir = img_save_dir
        self._config_save_file = config_save_file
        self._scalar_save_file = scalar_save_file

    def _init_env(self):
        """Initialize the environment for local visualization."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)
        self._img_save_dir = osp.join(self._save_dir, self._img_save_dir)  # type: ignore
        self._config_save_file = osp.join(self._save_dir, self._config_save_file)  # type: ignore
        self._scalar_save_file = osp.join(self._save_dir, self._scalar_save_file)  # type: ignore

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'LocalVisBackend':
        """Return the experiment object associated with this visualization backend."""
        return self

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to disk.

        Args:
            config (Config): The Config object.
        """
        assert isinstance(config, Config)
        save_config_str(config, self._config_save_file)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record an image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. Must be RGB format with dtype uint8.
            step (int): Global step value to record. Defaults to 0.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.dtype == np.uint8, "Image must be of dtype uint8."
        assert image.ndim in (2, 3), "Image must be 2D (grayscale) or 3D (RGB)."

        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f"{name}_{step}.png"
        save_path = osp.join(self._img_save_dir, save_file_name)

        # Use PIL to save RGB/Grayscale image
        img = Image.fromarray(image)
        img.save(save_path)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record a scalar value to disk.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                value = value.tolist()
        self._dump({name: value, 'step': step}, self._scalar_save_file)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record multiple scalars to disk.

        The scalar dict will be written to the default file and
        to an extra file if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pairs of scalar tags and values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): If specified, scalars will also be
                saved to this file. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict = copy.deepcopy(scalar_dict)
        scalar_dict.setdefault('step', step)

        if file_path is not None:
            assert file_path.split('.')[-1] == 'json'
            new_save_file_path = osp.join(self._save_dir, file_path)  # type: ignore
            assert new_save_file_path != self._scalar_save_file, \
                '``file_path`` and ``scalar_save_file`` have the same name, please set ``file_path`` to another value'
            self._dump(scalar_dict, new_save_file_path)
        self._dump(scalar_dict, self._scalar_save_file)

    @staticmethod
    def _dump(value_dict: dict, file_path: str) -> None:
        """Dump a dict to file as JSON (line-by-line JSONL format).

        Args:
            value_dict (dict): The dictionary to save.
            file_path (str): The file path to save data.
        """
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as f:
            dump(value_dict, f)
            f.write('\n')


@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseVisBackend):
    """Tensorboard visualization backend class.

    It can write images, config, scalars, etc. to a
    tensorboard file.

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend.
    """

    def __init__(self, save_dir: str):
        super().__init__(save_dir)
        self._tensorboard = None
        self._SummaryWriter = None

    def _init_env(self):
        """Initialize the environment for tensorboard visualization."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        # Try to import SummaryWriter from torch first, fallback to tensorboardX
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._SummaryWriter = SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
                self._SummaryWriter = SummaryWriter
            except ImportError as e:
                raise ImportError(
                    'Tensorboard is not available. Please install with:\n'
                    '  pip install tensorboard\n'
                    'or:\n'
                    '  pip install tensorboardX'
                ) from e

        self._tensorboard = self._SummaryWriter(self._save_dir)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return Tensorboard object."""
        return self._tensorboard

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to tensorboard.

        Args:
            config (Config): The Config object
        """
        self._tensorboard.add_text('config', config.pretty_text)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to tensorboard.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. Must be HWC RGB or grayscale.
            step (int): Global step value to record. Defaults to 0.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.dtype in (np.uint8, np.float32), \
            "Image must be of dtype uint8 or float32."
        assert image.ndim in (2, 3), \
            "Image must be 2D (grayscale) or 3D (HWC RGB)."

        self._tensorboard.add_image(name, image, step, dataformats='HWC')

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to tensorboard.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()
        elif isinstance(value, (int, float, np.number)):
            value = float(value)
        else:
            warnings.warn(
                f"Got {type(value)}, but numpy array, torch tensor, "
                f"int or float are expected. Skip it!"
            )
            return

        self._tensorboard.add_scalar(name, value, step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record multiple scalars to tensorboard.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, \
            "Please set it directly through the step parameter."

        for key, value in scalar_dict.items():
            self.add_scalar(key, value, step)

    def close(self):
        """Close the tensorboard writer if it is open."""
        if self._tensorboard is not None:
            self._tensorboard.close()
            self._tensorboard = None


@VISBACKENDS.register_module()
class SwanlabBackend(BaseVisBackend):
    """Swanlab visualization backend class.

    This backend integrates with Swanlab to log scalars, images, and configs.

    Args:
        save_dir (str): Directory to save offline Swanlab logs if needed.
        project (str, optional): Project name for Swanlab.
        experiment_name (str, optional): Experiment/run name.
    """

    def __init__(self, save_dir: str,
                 project: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        super().__init__(save_dir)
        self._swanlab = None
        self._project = project
        self._experiment_name = experiment_name

    def _init_env(self):
        """Initialize the environment for Swanlab visualization."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        try:
            import swanlab
        except ImportError:
            raise ImportError(
                "You are trying to use Swanlab which is not currently installed. "
                "Please install it with: pip install swanlab"
            )

        # Initialize Swanlab experiment (similar to wandb.init)
        self._swanlab = swanlab.init(
            project=self._project or "default_project",
            experiment_name=self._experiment_name or "default_run",
            dir=self._save_dir
        )

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return the Swanlab run object."""
        return self._swanlab

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to Swanlab.

        Args:
            config (Config): The Config object
        """
        # Swanlab logs configs via log() with a dict
        self._swanlab.config.update(dict(config))  # ensure Config can be cast to dict

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to Swanlab.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. Must be HWC RGB or grayscale.
            step (int): Global step value to record. Defaults to 0.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.dtype in (np.uint8, np.float32), \
            "Image must be of dtype uint8 or float32."
        assert image.ndim in (2, 3), \
            "Image must be 2D (grayscale) or 3D (HWC RGB)."

        import swanlab
        # Swanlab logs images with swanlab.Image
        self._swanlab.log({name: swanlab.Image(image)}, step=step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to Swanlab.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()
        elif isinstance(value, (int, float, np.number)):
            value = float(value)
        else:
            warnings.warn(
                f"Got {type(value)}, but numpy array, torch tensor, "
                f"int or float are expected. Skip it!"
            )
            return

        self._swanlab.log({name: value}, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record multiple scalars to Swanlab.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, \
            "Please set it directly through the step parameter."

        self._swanlab.log(scalar_dict, step=step)

    def close(self):
        """Close the Swanlab run if it is active."""
        if self._swanlab is not None:
            self._swanlab.finish()
            self._swanlab = None
