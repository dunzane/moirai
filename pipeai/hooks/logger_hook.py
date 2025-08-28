import os
import numpy as np
from json import dump
from pathlib import Path
import os.path as osp
from typing import Optional, Union, Sequence, Dict, OrderedDict

import torch

from pipeai.hooks import Hook,DATA_BATCH
from pipeai.registry import HOOKS
from pipeai.utils import is_seq_of


SUFFIX_TYPE = Union[Sequence[str], str]


@HOOKS.register_module()
class LoggerHook(Hook):
    """Collect logs from Runner and write them to terminal, JSON file, tensorboard, etc.

    This simplified version only supports local storage via `out_dir`.

    Args:
        interval (int): Logging interval (every k iterations). Defaults to 10.
        ignore_last (bool): Whether to ignore the last iterations in each epoch
            if the remaining iterations are less than `interval`. Defaults to True.
        interval_exp_name (int): Interval to log experiment name. Defaults to 1000.
        out_dir (str or Path): Root directory to save logs. Required.
        out_suffix (str or Sequence[str]): File suffixes to include when copying logs.
            Defaults to ('.json', '.log', '.py', 'yaml').
        keep_local (bool): Whether to keep local logs after copying. Defaults to True.
        log_metric_by_epoch (bool): Whether to log metrics by epoch instead of iteration. Defaults to True.
    """

    priority = 'BELOW_NORMAL'

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[Union[str, Path]] = None,
                 out_suffix: SUFFIX_TYPE = ('.json', '.log', '.py', 'yaml'),
                 keep_local: bool = True,
                 log_metric_by_epoch: bool = True):
        super().__init__()

        if not isinstance(interval, int) or interval <= 0:
            raise ValueError('interval must be a positive integer')
        if not isinstance(ignore_last, bool):
            raise TypeError('ignore_last must be a boolean')
        if not isinstance(interval_exp_name, int) or interval_exp_name <= 0:
            raise ValueError('interval_exp_name must be a positive integer')
        if out_dir is None or not isinstance(out_dir, (str, Path)):
            raise ValueError('out_dir must be specified as a str or Path')
        if not isinstance(keep_local, bool):
            raise TypeError('keep_local must be a boolean')
        if not (isinstance(out_suffix, str) or is_seq_of(out_suffix, str)):
            raise TypeError('out_suffix should be a string or a sequence of strings')

        self.interval = interval
        self.ignore_last = ignore_last
        self.interval_exp_name = interval_exp_name
        self.out_dir = str(out_dir) if isinstance(out_dir, Path) else out_dir
        self.out_suffix = out_suffix
        self.keep_local = keep_local
        self.json_log_path: Optional[str] = None
        self.log_metric_by_epoch = log_metric_by_epoch

    def before_run(self, runner) -> None:
        """Prepare log directory and JSON log file before training starts.

        Args:
            runner (Runner): The runner object of the training process.
        """
        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        self.out_dir = osp.join(self.out_dir, basename)
        os.makedirs(self.out_dir, exist_ok=True)
        runner.logger.info(f'Text logs will be saved to {self.out_dir} after training.')
        self.json_log_path = osp.join(self.out_dir, f'{runner.timestamp}.json')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Log training information after each iteration.

        Args:
            runner (Runner): The runner object of the training process.
            batch_idx (int): Current batch index.
            data_batch (DATA_BATCH, optional): Input data batch. Defaults to None.
            outputs (dict, optional): Model outputs for this iteration. Defaults to None.
        """
        log_needed = (
                self.every_n_train_iters(runner, self.interval_exp_name) or
                self.end_of_epoch(runner.train_dataloader, batch_idx)
        )
        if log_needed:
            runner.logger.info(f'Exp name: {runner.experiment_name}')

        if self.every_n_inner_iters(batch_idx, self.interval) or \
                (self.end_of_epoch(runner.train_dataloader, batch_idx) and
                 (not self.ignore_last or len(runner.train_dataloader) <= self.interval)):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, 'train')
            runner.logger.info(log_str)
            runner.visualizer.add_scalars(tag, step=runner.iter + 1, file_path=self.json_log_path)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """Log validation information after each iteration.

        Args:
            runner (Runner): The runner object of the validation process.
            batch_idx (int): Current batch index.
            data_batch (DATA_BATCH, optional): Input data batch. Defaults to None.
            outputs (Sequence, optional): Model outputs for this iteration. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, 'val')
            runner.logger.info(log_str)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Log test information after each iteration.

        Args:
            runner (Runner): The runner object of the test process.
            batch_idx (int): Current batch index.
            data_batch (DATA_BATCH, optional): Input data batch. Defaults to None.
            outputs (Sequence, optional): Model outputs for this iteration. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, 'test')
            runner.logger.info(log_str)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics after each epoch.

        Args:
            runner (Runner): The runner object of the validation process.
            metrics (Dict[str, float], optional): Dictionary of metrics. Defaults to None.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)

        step = runner.epoch if self.log_metric_by_epoch else runner.iter
        runner.visualizer.add_scalars(tag, step=step, file_path=self.json_log_path)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """Log test metrics after each epoch and dump JSON.

        Args:
            runner (Runner): The runner object of the test process.
            metrics (Dict[str, float], optional): Dictionary of metrics. Defaults to None.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), 'test', with_non_scalar=True)
        runner.logger.info(log_str)

        processed = self._process_tags(tag)
        with open(self.json_log_path, 'w') as f:
            dump(processed, f)

    def after_run(self, runner) -> None:
        """Finalize logging after training/testing/validation.

        This method closes the visualizer and copies log files to `out_dir`.

        Args:
            runner (Runner): The runner object of the process.
        """
        runner.visualizer.close()

        if self.out_dir is None:
            return

        for suffix in self.out_suffix if isinstance(self.out_suffix, (list, tuple)) else [self.out_suffix]:
            for file in Path(runner._log_dir).glob(f'*{suffix}'):
                out_file = osp.join(self.out_dir, file.name)
                with open(file, 'r') as f_src, open(out_file, 'w') as f_dst:
                    f_dst.write(f_src.read())
                runner.logger.info(f'File {file} copied to {out_file}')
                if not self.keep_local:
                    os.remove(file)
    @staticmethod
    def _process_tags(tags: dict) -> dict:
        """Recursively processes tag values to ensure JSON serializability.

        This method converts non-serializable types like torch.Tensor and
        np.ndarray into lists, while filtering out unsupported types.

        Args:
            tags (dict): The dictionary of tags to process.

        Returns:
            dict: A dictionary with JSON-serializable values.
        """

        def process_val(value):
            if isinstance(value, (list, tuple)):
                return [process_val(v) for v in value]
            elif isinstance(value, dict):
                return {k: process_val(v) for k, v in value.items()}
            elif isinstance(value, (str, int, float, bool)) or value is None:
                return value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                return value.tolist()
            else:
                return None

        return OrderedDict(process_val(tags))  # type: ignore
