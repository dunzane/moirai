from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import DataLoader

from pipeai.registry import LOOPS
from pipeai.data import InfiniteDataloaderIterator
from pipeai.evaluator import Evaluator, AvgMeter
from .amp import autocast
from .base_loop import BaseLoop


@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (DataLoader or dict): A dataloader object or a dict to build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1):
        super().__init__(runner, dataloader)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = (self._max_epochs * len(self.dataloader)
                           if hasattr(self.dataloader, '__len__') else None)

        self._epoch = 0
        self._iter = 0
        self._val_begin = val_begin
        self._val_interval = val_interval
        self._stop_training = False  # Updated by `EarlyStoppingHook` when it is enabled.

        if hasattr(self._runner, "visualizer"):
            if hasattr(self._dataloader.dataset, 'metainfo'):
                self._runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo
            else:
                self.logger.warning(
                    f'Dataset {self._dataloader.dataset.__class__.__name__} has no '
                    '`metainfo`. ``dataset_meta`` in visualizer will be None.'
                )

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self._runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self._stop_training:
            self._run_epoch()

            if (self._runner.val_loop is not None
                    and self._epoch >= self._val_begin
                    and (self._epoch % self._val_interval == 0
                         or self._epoch == self._max_epochs)):
                self._runner.val_loop.run()

            if getattr(self._runner, "should_stop", False):
                self.logger.info(f"Early stopping at epoch {self._epoch}")
                break

        self._runner.call_hook('after_train')
        return self._runner.model

    def _run_epoch(self):
        """Run one epoch."""
        self._runner.call_hook('before_train_epoch')
        self._runner.model.train()

        for idx, data_batch in enumerate(self._dataloader):
            self._run_iter(batch_idx=idx, batch_data=data_batch)
            if self._stop_training:
                break

        self._runner.call_hook('after_train_epoch')
        self._epoch += 1

    def _run_iter(self, batch_idx: int, batch_data: Sequence[dict]) -> None:
        """Run one iteration."""
        self._runner.call_hook(
            'before_train_iter',
            batch_idx=batch_idx,
            batch_data=batch_data
        )

        outputs = self._runner.model.train_step(
            batch_data=batch_data,
            optim_wrapper=self.runner.optim_wrapper
        )

        if not isinstance(outputs, dict) or 'loss' not in outputs:
            self.logger.warning(
                "train_step should return a dict with key 'loss'."
            )

        self._runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            batch_data=batch_data,
            outputs=outputs
        )

        self._iter += 1


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iteration-based training.

    Args:
        runner (Runner): Reference to the runner.
        dataloader (DataLoader or dict): DataLoader object or dict to build a dataloader.
        max_iters (int): Total number of training iterations.
        val_begin (int): Iteration to begin validation. Defaults to 1.
        val_interval (int): Validation interval in iterations. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: int,
                 val_begin: int = 1,
                 val_interval: int = 1):
        super().__init__(runner, dataloader)

        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, f"`max_iters` should be an integer, but got {max_iters}"

        # For compatibility with epoch-based training
        self._max_epochs, self._epoch = 1, 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False

        # Bind dataset metadata to visualizer if available
        if hasattr(self._runner, "visualizer"):
            if hasattr(self._dataloader.dataset, 'metainfo'):
                self._runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo
            else:
                self.logger.warning(
                    f"Dataset {self._dataloader.dataset.__class__.__name__} has no `metainfo`. "
                    "dataset_meta in visualizer will be None."
                )

        # Wrap dataloader with an infinite iterator
        self.dataloader_iterator = InfiniteDataloaderIterator(self._dataloader)

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def epoch(self):
        # Compute equivalent epoch based on number of iterations
        if hasattr(self._dataloader, '__len__'):
            return self._iter // len(self._dataloader)
        return self._epoch

    @property
    def iter(self):
        return self._iter

    def run(self) -> None:
        """Launch iteration-based training."""
        self._runner.call_hook('before_train')

        # Treat the entire training process as a single large epoch
        self._runner.call_hook('before_train_epoch')

        # Skip already trained iterations if resuming
        if self._iter > 0:
            self.logger.warning(f"Advance dataloader {self._iter} steps to skip already trained data.")
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        while self._iter < self._max_iters and not self.stop_training:
            self._runner.model.train()

            # Fetch next batch
            batch_data = next(self.dataloader_iterator)

            # Run one training iteration
            self._run_iter(batch_data)

            # Early stopping check
            if getattr(self._runner, "should_stop", False):
                self.logger.info(f"Early stopping at iteration {self._iter}")
                break

            # Validation check
            if (self._runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self._runner.val_loop.run()

        self._runner.call_hook('after_train_epoch')
        self._runner.call_hook('after_train')

    def _run_iter(self, batch_data: Sequence[dict]) -> None:
        """Run one iteration of training."""
        self._runner.call_hook(
            'before_train_iter',
            batch_idx=self._iter,
            batch_data=batch_data
        )

        # Perform the training step
        outputs = self._runner.model.train_step(
            batch_data=batch_data,
            optim_wrapper=self._runner.optim_wrapper
        )

        # Check outputs contain 'loss'
        if not isinstance(outputs, dict) or 'loss' not in outputs:
            self.logger.warning("train_step should return a dict containing key 'loss'.")

        self._runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            batch_data=batch_data,
            outputs=outputs
        )

        self._iter += 1


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator

        # Bind dataset metadata to visualizer if available
        if hasattr(self._runner, "visualizer"):
            if hasattr(self._dataloader.dataset, 'metainfo'):
                self._runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo
            else:
                self.logger.warning(
                    f"Dataset {self._dataloader.dataset.__class__.__name__} has no `metainfo`. "
                    "dataset_meta in visualizer will be None."
                )

        self.fp16 = fp16
        self.val_loss: Dict[str, AvgMeter] = dict()
        self.metrics = None

    def run(self) -> dict:
        """Launch validation."""
        self._runner.call_hook('before_valid')
        self._runner.call_hook('before_val_epoch')

        self._runner.model.eval()

        # reset all val-stage metrics
        self.evaluator.reset(stage='val')

        for idx, data_batch in enumerate(self.dataloader):
            self._run_iter(idx, data_batch)

        # compute aggregated metrics for the validation stage
        self.metrics = self.evaluator.evaluate(stage='val')

        self._runner.call_hook('after_val_epoch', metrics=self.metrics)
        self._runner.call_hook('after_val')

        return self.metrics

    @torch.no_grad()
    def _run_iter(self, batch_idx: int, batch_data: Sequence[dict]):
        """Launch iteration-based validation."""
        self._runner.call_hook(
            'before_val_iter',
            batch_idx=batch_idx,
            batch_data=batch_data,
        )

        with autocast(enabled=self.fp16):  # type: ignore
            outputs = self.runner.model.val_step(batch_data)  # return dict

        for name, value in outputs.items():
            self.evaluator.process(stage='val', name=name, value=value)

        self._runner.call_hook(
            'after_val_iter',
            batch_idx=batch_idx,
            data_batch=batch_data,
            outputs=outputs
        )


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (DataLoader or dict): Dataloader object or dict to build it.
        evaluator (Evaluator or dict or list): For computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        # build evaluator if needed
        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)
        else:
            self.evaluator = evaluator

        # bind dataset metadata to visualizer
        if hasattr(self._runner, "visualizer") and hasattr(self._dataloader.dataset, 'metainfo'):
            self._runner.visualizer.dataset_meta = self._dataloader.dataset.metainfo

        self.fp16 = fp16

    def run(self) -> Dict[str, float]:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # reset metrics for test stage
        self.evaluator.reset(stage='test')

        for idx, data_batch in enumerate(self.dataloader):
            self._run_iter(idx, data_batch)

        # compute aggregated metrics
        metrics = self.evaluator.evaluate(stage='test')

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def _run_iter(self, batch_idx: int, batch_data: Sequence[dict]):
        """Launch iteration-based test."""
        self._runner.call_hook(
            'before_test_iter',
            batch_idx=batch_idx,
            batch_data=batch_data,
        )

        with autocast(enabled=self.fp16):  # type: ignore
            outputs = self.runner.model.val_step(batch_data)

        # process outputs into evaluator
        for name, value in outputs.items():
            self.evaluator.process(stage='test', name=name, value=value)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=batch_idx,
            data_batch=batch_data,
            outputs=outputs
        )
