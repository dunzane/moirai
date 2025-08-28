import copy
import datetime
import re
from typing import List, Optional, Tuple, Dict, Any
from collections import OrderedDict
from itertools import chain

import numpy as np
import torch

from pipeai.device import is_cuda_available, get_max_cuda_memory
from pipeai.device.available import is_musa_available
from pipeai.registry import LOG_PROCESSORS


@LOG_PROCESSORS.register_module()
class LogProcessor:

    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None,
                 num_digits: int = 4,
                 log_with_hierarchy: bool = False,
                 mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
        from pipeai.logging import get_logger
        self.logger = get_logger(f'pipeai-{__class__.__name__}')

        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self.num_digits = num_digits
        self.log_with_hierarchy = log_with_hierarchy
        self.mean_pattern = mean_pattern

        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int, stage: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            stage (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert stage in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, stage)

        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, stage, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{stage}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, stage)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if stage in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, stage)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({stage}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({stage}) '
                           f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            if stage == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({stage}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, stage)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({stage}) [{cur_iter_str}/{dataloader_len}]  ')
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                        f'data_time: '
                        f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda/musa is available,
        # the max memory occupied should be calculated.
        if is_cuda_available() or is_musa_available():
            max_memory = self._get_max_memory(runner)
            log_str += f'memory: {max_memory}  '
            tag['memory'] = max_memory

        # Loop left keys to fill `log_str`.
        if stage in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if stage == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    def get_log_after_epoch(self,
                            runner,
                            batch_idx: int,
                            stage: str,
                            with_non_scalar: bool = False) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner phase.
            batch_idx (int): The index of the current batch in the current loop.
            stage (str): Current mode of runner, train, test or val.
            with_non_scalar (bool): Whether to include non-scalar infos in the
                returned tag. Defaults to False.

        Returns:
            Tuple[dict, str]: Formatted log dict/string which will be recorded by
                :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert stage in ['val', 'test'], f'_get_metric_log_str` only accept val or test mode, but got {stage}'
        dataloader_len = self._get_dataloader_size(runner, stage)

        # format log header.
        # by_epoch == True
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # by_epoch == False
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if stage == 'val':
                cur_epoch = self._get_epoch(runner, stage)
                log_str = (f'Epoch({stage}) [{cur_epoch}][{dataloader_len}/'
                           f'{dataloader_len}]  ')
            else:
                log_str = (
                    f'Epoch({stage}) [{dataloader_len}/{dataloader_len}]  ')
        else:
            log_str = f'Iter({stage}) [{dataloader_len}/{dataloader_len}]  '

        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        custom_keys = [
            self._remove_prefix(cfg['data_src'], f'{stage}/')  # remove prefix
            for cfg in custom_cfg_copy
        ]

        # count the averaged time and data_time by epoch
        if 'time' not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src='time', window_size='epoch', method_name='mean'))
        if 'data_time' not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src='data_time', window_size='epoch', method_name='mean'))
        parsed_cfg = self._parse_windows_size(runner, batch_idx, custom_cfg_copy)
        # tag is used to write log information to different backends-.
        ori_tag = self._collect_scalars(runner, parsed_cfg, stage, self.log_with_hierarchy)
        non_scalar_tag = self._collect_non_scalars(runner, stage)
        tag = OrderedDict()
        time_tag = OrderedDict()
        for k, v in ori_tag.items():
            if k in (f'{stage}/time', f'{stage}/data_time', 'time',
                     'data_time'):
                time_tag[k] = v
            else:
                tag[k] = v
        # log other messages.
        log_items = []
        log_str += '  '
        for name, val in chain(tag.items(), non_scalar_tag.items(),
                               time_tag.items()):
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            if isinstance(val, (torch.Tensor, np.ndarray)):
                # newline to display tensor and array.
                val = f'\n{val}\n'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)

        if with_non_scalar:
            tag.update(non_scalar_tag)
        tag.update(time_tag)
        return tag, log_str

    def _parse_windows_size(self,
                            runner,
                            batch_idx: int,
                            custom_cfg: Optional[list] = None) -> list:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
            custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
                to keep backward compatibility.
        """
        if custom_cfg is None:
            custom_cfg = copy.deepcopy(self.custom_cfg)
        else:
            custom_cfg = copy.deepcopy(custom_cfg)
        for log_cfg in custom_cfg:
            window_size = log_cfg.get('window_size', None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == 'epoch':
                log_cfg['window_size'] = batch_idx + 1
            elif window_size == 'global':
                log_cfg['window_size'] = runner.iter + 1
            else:
                raise TypeError(
                    'window_size should be int, epoch or global, but got '
                    f'invalid {window_size}')
        return custom_cfg

    def _collect_scalars(self,
                         custom_cfg: List[dict],
                         runner,
                         stage: str,
                         reserve_prefix: bool = False) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            stage (str): Current mode of runner.
            reserve_prefix (bool): Whether to reserve the prefix of the key.

        Returns:
            dict: Statistical values of logs.
        """
        custom_cfg = copy.deepcopy(custom_cfg)
        tag = OrderedDict()
        # history_scalars of train/val/test phase.
        history_scalars = runner.message_hub.log_scalars
        # corresponding mode history_scalars
        mode_history_scalars = OrderedDict()
        # extract log scalars and remove prefix to `mode_history_scalars`
        # according to mode.
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(stage):
                if not reserve_prefix:
                    key = self._remove_prefix(prefix_key, f'{stage}/')
                else:
                    key = prefix_key
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # Update the latest learning rate and smoothed time logs.
            if re.search(self.mean_pattern, key) is not None:
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # Default statistic method is current.
                tag[key] = mode_history_scalars[key].current()
        # Update custom keys.
        for log_cfg in custom_cfg:
            data_src = log_cfg.pop('data_src')
            log_name = log_cfg.pop('log_name', data_src)
            if reserve_prefix:
                data_src = f'{stage}/{data_src}'
                log_name = f'{stage}/{log_name}'
            # log item in custom_cfg could only exist in train or val
            # mode.
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(
                    **log_cfg)
        return tag

    def _collect_non_scalars(self,
                             runner,
                             stage: str) -> dict:
        """Collect log information to compose a dict according to stage."""
        infos = runner.message_hub.runtime_info
        mode_infos = OrderedDict()
        for prefix_key, value in infos.items():
            if prefix_key.startswith(stage):
                if self.log_with_hierarchy:
                    key = prefix_key
                else:
                    key = self._remove_prefix(prefix_key, f'{stage}/')
                mode_infos[key] = value
        return mode_infos

    def _get_iter(self, runner, batch_idx: int):
        """Get current iteration index."""
        if self.by_epoch:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_dataloader_size(self, runner, stage) -> int:
        """Get dataloader size of current loop."""
        return len(self._get_cur_loop(runner=runner, stage=stage).dataloader)

    @staticmethod
    def _remove_prefix(string: str, prefix: str):
        """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string

    @staticmethod
    def _get_cur_loop(runner, stage: str):
        """Get current loop according to stage."""
        if stage == 'train':
            return runner.train_loop
        elif stage == 'val':
            return runner.val_loop
        else:
            return runner.test_loop

    @staticmethod
    def _get_epoch(runner, stage: str):
        """Get current epoch according to stage."""
        if stage == 'train':
            epoch = runner.epoch + 1
        elif stage == 'val':
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                epoch = 0
            else:
                # normal val stage
                # runner.epoch += 1 has been done before validation
                epoch = runner.epoch
        else:
            raise ValueError(f"runner stage should be 'train' or 'val', but got {stage}")
        return epoch

    @staticmethod
    def _get_max_memory(runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for a given device."""
        device = getattr(runner.model, 'output_device', None)
        return get_max_cuda_memory(device)

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                assert log_name not in check_set, (
                    f'Found duplicate {log_name} for {data_src}. Please check'
                    'your `custom_cfg` for `log_processor`. You should '
                    f'neither define duplicate `{log_name}` for {data_src} '
                    f'nor do not define any {log_name} for multiple '
                    f'{data_src}, See more information in the docstring of '
                    'LogProcessor')

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()
