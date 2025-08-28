import time
from typing import Optional, Union, Sequence

from pipeai.hooks import Hook, DATA_BATCH
from pipeai.registry import HOOKS


@HOOKS.register_module()
class IterTimerHook(Hook):
    """Hook for logging iteration-level timing information.

    Tracks:
        - ``data_time``: time spent loading a batch of data.
        - ``time``: total time for one training/validation/test iteration.
        - Estimated time of arrival (``eta``) based on iteration averages.
    """

    priority = 'NORMAL'

    def __init__(self) -> None:
        super().__init__()
        self.total_train_time: float = 0.0
        self.total_eval_time: float = 0.0
        self.start_iter: int = 0
        self.iter_start_time: float = 0.0

    def before_train(self, runner) -> None:
        """Record the starting iteration index at the beginning of training."""
        self.start_iter = runner.iter

    def _before_epoch(self, runner, stage: str = 'train') -> None:
        """Mark the start time before an epoch begins."""
        self.iter_start_time = time.time()

    def _after_epoch(self, runner, stage: str = 'train') -> None:
        """Reset evaluation timer after each epoch."""
        self.total_eval_time = 0.0

    def _before_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
            stage: str = 'train'
    ) -> None:
        """Log the time taken to load the current batch of data."""
        runner.message_hub.update_scalar(
            f'{stage}/data_time', time.time() - self.iter_start_time
        )

    def _after_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Union[dict, Sequence]] = None,
            stage: str = 'train'
    ) -> None:
        """Log iteration time and update ETA based on average iteration duration."""
        hub = runner.message_hub
        hub.update_scalar(f'{stage}/time', time.time() - self.iter_start_time)
        iter_time = hub.get_scalar(f'{stage}/time').current()

        if stage == 'train':
            self.total_train_time += iter_time
            avg_iter_time = self.total_train_time / (
                    runner.iter - self.start_iter + 1
            )
            eta_seconds = avg_iter_time * (runner.max_iters - runner.iter - 1)

            hub.update_info('eta', eta_seconds)
        else:
            cur_dataloader = (
                runner.val_dataloader if stage == 'val'
                else runner.test_dataloader
            )
            self.total_eval_time += iter_time
            avg_iter_time = self.total_eval_time / (batch_idx + 1)
            eta_seconds = avg_iter_time * (len(cur_dataloader) - batch_idx - 1)

            hub.update_info('eta', eta_seconds)
