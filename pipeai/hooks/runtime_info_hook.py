from typing import Optional, Any, Dict

import numpy as np
import torch

from pipeai.hooks import Hook, DATA_BATCH
from pipeai.registry import HOOKS
from pipeai.utils import get_git_hash
from pipeai.version import __version__


@HOOKS.register_module()
class RuntimeInfoHook(Hook):
    """A hook that updates runtime information into message hub.

    This hook provides essential runtime information such as current epoch,
    iteration, max epochs/iters, dataset meta, and log variables. Other
    components that cannot access the runner directly can retrieve this
    information from the message hub.
    """

    priority = 'VERY_HIGH'

    def __init__(self):
        super().__init__()
        self.last_loop_stage = None

    def before_run(self, runner) -> None:
        """Update meta information before training starts.

        Args:
            runner (Runner): The runner of the training process.
        """
        cfg_text = getattr(runner.cfg, "pretty_text", str(runner.cfg))
        metainfo = dict(
            cfg=cfg_text,
            seed=runner.seed,
            experiment_name=runner.experiment_name,
            pipeai_version=__version__ + get_git_hash(),
        )
        runner.message_hub.update_info_dict(metainfo)
        self.last_loop_stage = None

    def before_train(self, runner) -> None:
        """Update training state information before training begins."""
        runner.message_hub.update_info("loop_stage", "train")
        runner.message_hub.update_info("epoch", runner.epoch)
        runner.message_hub.update_info("iter", runner.iter)
        runner.message_hub.update_info("max_epochs", runner.max_epochs)
        runner.message_hub.update_info("max_iters", runner.max_iters)

        if hasattr(runner.train_dataloader.dataset, "metainfo"):
            runner.message_hub.update_info(
                "dataset_meta", runner.train_dataloader.dataset.metainfo
            )

    def before_train_epoch(self, runner) -> None:
        """Update current epoch before each training epoch."""
        runner.message_hub.update_info("epoch", runner.epoch)

    def before_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
    ) -> None:
        """Update iteration and learning rate before each training iteration."""
        runner.message_hub.update_info("iter", runner.iter)

        lr_dict = runner.optim_wrapper.get_lr()
        assert isinstance(
            lr_dict, dict
        ), (
            "`runner.optim_wrapper.get_lr()` should return a dict "
            f"but got {type(lr_dict)}. Please check your optimizer wrapper."
        )

        for name, lr in lr_dict.items():
            # lr may be a list/tuple (per param group) -> use first element
            value = lr[0] if isinstance(lr, (list, tuple)) else lr
            runner.message_hub.update_scalar(f"train/{name}", value)

    def after_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
            outputs: Optional[dict] = None,
    ) -> None:
        """Update log variables in model outputs after each iteration."""
        if outputs is not None:
            for key, value in outputs.items():
                if self._is_scalar(value):
                    runner.message_hub.update_scalar(f"train/{key}", value)
                else:
                    runner.message_hub.update_info(f"train/{key}", value)

    def after_train(self, runner) -> None:
        """Remove loop stage after training ends."""
        runner.message_hub.pop_info("loop_stage")

    def before_val(self, runner) -> None:
        """Set loop stage to validation."""
        self.last_loop_stage = runner.message_hub.get_info("loop_stage")
        runner.message_hub.update_info("loop_stage", "val")

    def after_val_epoch(
            self,
            runner,
            metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update validation metrics after each validation epoch."""
        if metrics is not None:
            for key, value in metrics.items():
                if self._is_scalar(value):
                    runner.message_hub.update_scalar(f"val/{key}", value)
                else:
                    runner.message_hub.update_info(f"val/{key}", value)

    def after_val(self, runner) -> None:
        """Reset loop stage after validation finishes."""
        if self.last_loop_stage == "train":
            runner.message_hub.update_info("loop_stage", self.last_loop_stage)
            self.last_loop_stage = None
        else:
            runner.message_hub.pop_info("loop_stage")

    def before_test(self, runner) -> None:
        """Set loop stage to test."""
        runner.message_hub.update_info("loop_stage", "test")

    def after_test_epoch(
            self,
            runner,
            metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update test metrics after each test epoch."""
        if metrics is not None:
            for key, value in metrics.items():
                if self._is_scalar(value):
                    runner.message_hub.update_scalar(f"test/{key}", value)
                else:
                    runner.message_hub.update_info(f"test/{key}", value)

    def after_test(self, runner) -> None:
        """Remove loop stage after testing ends."""
        runner.message_hub.pop_info("loop_stage")

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        """Check if the given value is a scalar."""
        if isinstance(value, np.ndarray):
            return value.size == 1
        elif isinstance(value, (int, float, np.generic)):
            return True
        elif isinstance(value, torch.Tensor):
            return value.numel() == 1
        return False
