from typing import Optional, Dict, Union, List
from torch.optim import lr_scheduler

from pipeai.hooks import Hook, DATA_BATCH
from pipeai.registry import HOOKS


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """A hook to update optimizer parameters like learning rate or momentum
    using PyTorch's built-in lr_scheduler.

    This hook supports both iteration-based and epoch-based stepping.
    For schedulers such as ReduceLROnPlateau, validation metrics will be
    passed in after a validation epoch.
    """

    priority = 'LOW'

    def _step(self, schedulers: Union[List, Dict], mode: str, metrics: Optional[Dict] = None):
        """Iterate through a list or dict of schedulers and step them."""
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                self._step_one(scheduler, mode, metrics)
        elif isinstance(schedulers, dict):
            for sub_schedulers in schedulers.values():
                for scheduler in sub_schedulers:
                    self._step_one(scheduler, mode, metrics)
        else:
            raise TypeError(
                f'runner.param_schedulers should be a list or dict, but got {type(schedulers)}')

    @staticmethod
    def _step_one(scheduler, mode: str, metrics: Optional[Dict] = None):
        """Step a single scheduler based on the current mode."""
        # Schedulers like ReduceLROnPlateau need metrics from validation
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            if mode == "val" and metrics is not None:
                # Note: Assumes the first key in metrics is the one to monitor.
                # It's the user's responsibility to ensure metric keys are consistent.
                monitor_key = next(iter(metrics))
                scheduler.step(metrics[monitor_key])
    # Other standard schedulers are stepped by iteration or epoch
        elif mode in ("iter", "epoch"):
            scheduler.step()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Step schedulers that are updated per iteration."""
        if runner.param_schedulers is not None:
            self._step(runner.param_schedulers, mode="iter")

    def after_train_epoch(self, runner) -> None:
        """Step schedulers that are updated per epoch."""
        if runner.param_schedulers is not None:
            self._step(runner.param_schedulers, mode="epoch")

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Step schedulers that depend on validation metrics."""
        if runner.param_schedulers is not None:
            self._step(runner.param_schedulers, mode="val", metrics=metrics)