import io
import os
import shutil
import hashlib
import os.path as osp
from numpy import inf
from pathlib import Path
from collections import deque
from typing import Dict, Optional, Union, List, Sequence, Callable

import torch

from .hook import Hook
from pipeai.utils import is_list_of, is_seq_of
from pipeai.dist import is_main_process, master_only
from pipeai.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Saves checkpoints periodically to the local filesystem.

    This hook provides functionality to save checkpoints at regular intervals,
    track and save the best-performing models based on validation metrics,
    manage the total number of checkpoints, and publish cleaned model files.
    """

    priority = 'VERY_LOW'

    # --- Predefined mapping for comparison rules and initial values ---
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 filename_tmpl: Optional[str] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 **kwargs) -> None:
        """Initializes the CheckpointHook.

        Args:
            interval (int): The saving interval. Defaults to -1 (no saving).
            by_epoch (bool): If True, saves every `interval` epochs. Otherwise,
                saves every `interval` iterations. Defaults to True.
            save_optimizer (bool): Whether to save the optimizer's state dict.
                Defaults to True.
            save_param_scheduler (bool): Whether to save the parameter
                scheduler's state dict. Defaults to True.
            out_dir (str or Path, optional): Directory to save checkpoints.
                If None, uses `runner.work_dir`. Defaults to None.
            max_keep_ckpts (int): The maximum number of checkpoints to keep.
                -1 means no limit. Defaults to -1.
            save_last (bool): Whether to save a final checkpoint at the end of
                training. Defaults to True.
            save_best (str or List[str], optional): Metric name(s) to use for
                saving the best checkpoint. Use 'auto' to automatically select
                the first metric. Defaults to None.
            rule (str or List[str], optional): Comparison rule for `save_best`.
                Options are 'greater' or 'less'. If None, the rule is inferred
                from the metric name. Defaults to None.
            greater_keys (Sequence[str], optional): Metric keys that should use
                the 'greater' rule by default.
            less_keys (Sequence[str], optional): Metric keys that should use
                the 'less' rule by default.
            filename_tmpl (str, optional): Template for checkpoint filenames.
                Defaults to 'epoch_{}.pth' or 'iter_{}.pth'.
            published_keys (str or List[str], optional): Keys to keep when
                publishing the model. If None, no publishing is done.
            save_begin (int): The step (epoch or iter) to start saving
                checkpoints. Defaults to 0.
            **kwargs: Additional arguments for `runner.save_checkpoint`.
        """
        super().__init__()
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self.out_dir = str(out_dir) if out_dir is not None else None
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs

        if filename_tmpl is None:
            self.filename_tmpl = 'epoch_{}.pth' if self.by_epoch else 'iter_{}.pth'
        else:
            self.filename_tmpl = filename_tmpl

        assert (isinstance(save_best, str) or is_list_of(save_best, str)
                or save_best is None), \
            f'"save_best" must be a str, list of str, or None, but got {type(save_best)}'
        if isinstance(save_best, list):
            if 'auto' in save_best:
                assert len(save_best) == 1, 'Only one "auto" is allowed in "save_best"'
            assert len(save_best) == len(set(save_best)), 'Duplicate keys found in "save_best"'
        elif save_best is not None:
            save_best = [save_best]
        self.save_best = save_best

        assert (isinstance(rule, str) or is_list_of(rule, str) or rule is None), \
            f'"rule" must be a str, list of str, or None, but got {type(rule)}'
        if isinstance(rule, list):
            assert len(rule) in [1, len(self.save_best) if self.save_best else 1], \
                'Length of "rule" must be 1 or match the length of "save_best"'
        else:
            rule = [rule]

        self.greater_keys = self._default_greater_keys if greater_keys is None else \
            (greater_keys,) if not isinstance(greater_keys, (list, tuple)) else greater_keys
        self.less_keys = self._default_less_keys if less_keys is None else \
            (less_keys,) if not isinstance(less_keys, (list, tuple)) else less_keys

        if self.save_best is not None:
            self.is_better_than: Dict[str, Callable] = {}
            self._init_rule(rule, self.save_best)
            if len(self.key_indicators) == 1:
                self.best_ckpt_path: Optional[str] = None
            else:
                self.best_ckpt_path_dict: Dict[str, Optional[str]] = {}

        if not (isinstance(published_keys, str) or is_seq_of(published_keys, str) or published_keys is None):
            raise TypeError(
                f'"published_keys" must be a str, a sequence of str, or None, but got {type(published_keys)}')
        if isinstance(published_keys, str):
            published_keys = [published_keys]
        elif isinstance(published_keys, (list, tuple)):
            assert len(published_keys) == len(set(published_keys)), 'Duplicate keys found in "published_keys"'
        self.published_keys = published_keys

        self.last_ckpt = None
        if save_begin < 0:
            raise ValueError(f'"save_begin" must be >= 0, but got {save_begin}')
        self.save_begin = save_begin

    def before_train(self, runner) -> None:
        """Prepares the checkpoint directory and restores state before training.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = runner.message_hub.get_info('best_ckpt')
            else:
                self.best_ckpt_path_dict = {}
                for key in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key}'
                    self.best_ckpt_path_dict[key] = runner.message_hub.get_info(best_ckpt_name)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids', [])
            while len(keep_ckpt_ids) > self.max_keep_ckpts and is_main_process():
                step = keep_ckpt_ids.pop(0)
                path = osp.join(self.out_dir, self.filename_tmpl.format(step))
                if osp.isfile(path):
                    os.remove(path)
                elif osp.isdir(path):
                    shutil.rmtree(path)
            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids, maxlen=self.max_keep_ckpts)
        else:
            self.keep_ckpt_ids: deque = deque()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Saves a checkpoint after a training iteration if conditions are met.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The current batch index.
            data_batch (DATA_BATCH, optional): Data from the dataloader.
            outputs (dict, optional): Outputs from the model.
        """
        if self.by_epoch:
            return

        if self.every_n_train_iters(runner, self.interval, self.save_begin) or \
                (self.save_last and self.is_last_train_iter(runner)):
            runner.logger.info(f'Saving checkpoint at iteration {runner.iter + 1}')
            self._save_checkpoint(runner)

    def after_train_epoch(self, runner) -> None:
        """Saves a checkpoint after a training epoch if conditions are met.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.interval, self.save_begin) or \
                (self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(f'Saving checkpoint at epoch {runner.epoch + 1}')
            self._save_checkpoint(runner)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Saves the best checkpoint based on validation metrics.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict, optional): A dictionary of evaluation metrics.
        """
        if not metrics:
            runner.logger.warning('Metrics are empty; skipping best checkpoint saving.')
            return
        self._save_best_checkpoint(runner, metrics)

    def after_train(self, runner) -> None:
        """Publishes checkpoints after training completes.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.published_keys is None:
            return

        if self.save_last and self.last_ckpt is not None:
            self._publish_model(runner, self.last_ckpt)

        if getattr(self, 'best_ckpt_path', None) is not None:
            self._publish_model(runner, str(self.best_ckpt_path))

        if getattr(self, 'best_ckpt_path_dict', None) is not None:
            for best_ckpt in self.best_ckpt_path_dict.values():
                self._publish_model(runner, best_ckpt)

    def _init_rule(self, rules, key_indicators) -> None:
        """Initializes comparison rules for tracking the best metrics.

        Args:
            rules (List[str or None]): List of rules ('greater' or 'less').
            key_indicators (List[str]): List of metric names to track.
        """
        if len(rules) == 1:
            rules = rules * len(key_indicators)

        self.rules = []
        for rule, key in zip(rules, key_indicators):
            if rule not in self.rule_map and rule is not None:
                raise KeyError(f'Rule must be "greater", "less", or None, but got {rule}')

            if rule is None and key != 'auto':
                key_lc = key.lower()
                greater = {k.lower() for k in self.greater_keys}
                less = {k.lower() for k in self.less_keys}

                if key_lc in greater or any(k in key_lc for k in greater):
                    rule = 'greater'
                elif key_lc in less or any(k in key_lc for k in less):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer rule for key "{key}". Please specify it explicitly.')

            if rule is not None:
                self.is_better_than[key] = self.rule_map[rule]
            self.rules.append(rule)
        self.key_indicators = key_indicators

    def _save_checkpoint(self, runner) -> None:
        """Saves the current periodic checkpoint and manages history.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            step = runner.epoch + 1
            meta = {'epoch': step, 'iter': runner.iter}
        else:
            step = runner.iter + 1
            meta = {'epoch': runner.epoch, 'iter': step}
        self._save_checkpoint_with_step(runner, step, meta)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Saves the best checkpoint if the current metric is better.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): A dictionary of evaluation metrics.
        """
        if not self.save_best:
            return

        cur_type, cur_time = ('epoch', runner.epoch) if self.by_epoch else ('iter', runner.iter)
        meta = {'epoch': runner.epoch, 'iter': runner.iter}
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        best_ckpt_updated = False
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics.get(key_indicator)
            if key_score is None:
                continue

            if len(self.key_indicators) == 1:
                best_score_key, runtime_ckpt_key = 'best_score', 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict.get(key_indicator)

            best_score = runner.message_hub.get_info(best_score_key, self.init_value_map[rule])
            if not self.is_better_than[key_indicator](key_score, best_score):
                continue

            best_ckpt_updated = True
            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and is_main_process():
                if osp.isfile(best_ckpt_path):
                    os.remove(best_ckpt_path)
                elif osp.isdir(best_ckpt_path):
                    shutil.rmtree(best_ckpt_path)
                runner.logger.info(f'Removed previous best checkpoint: {best_ckpt_path}')

            best_ckpt_name = f'best_{key_indicator}_{cur_type}{cur_time}.pth'
            new_best_path = osp.join(self.out_dir, best_ckpt_name)
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = new_best_path
            else:
                self.best_ckpt_path_dict[key_indicator] = new_best_path
            runner.message_hub.update_info(runtime_ckpt_key, new_best_path)

            runner.save_checkpoint(
                self.out_dir, filename=best_ckpt_name, save_optimizer=False,
                save_param_scheduler=False, meta=meta, by_epoch=False)

            runner.logger.info(
                f'New best checkpoint saved: {best_score:0.4f} {key_indicator} at '
                f'{cur_time} {cur_type} to {best_ckpt_name}.')

        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(runner, cur_time, meta)

    def _save_checkpoint_with_step(self, runner, step, meta):
        """Saves a checkpoint for a given step.

        Args:
            runner (Runner): The runner of the training process.
            step (int): The current step (epoch or iteration).
            meta (dict): Metadata for the checkpoint.
        """
        if self.max_keep_ckpts > 0 and step not in self.keep_ckpt_ids:
            if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                old_step = self.keep_ckpt_ids.popleft()
                ckpt_path = osp.join(self.out_dir, self.filename_tmpl.format(old_step))
                if osp.exists(ckpt_path):
                    if osp.isfile(ckpt_path):
                        os.remove(ckpt_path)
                    else:
                        shutil.rmtree(ckpt_path)
            self.keep_ckpt_ids.append(step)
            runner.message_hub.update_info('keep_ckpt_ids', list(self.keep_ckpt_ids))

        # Save the checkpoint
        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = osp.join(self.out_dir, ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)
        runner.save_checkpoint(
            self.out_dir, ckpt_filename, save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler, meta=meta,
            by_epoch=self.by_epoch, **self.args)

        # Update the 'last_checkpoint' file for easy resuming
        if is_main_process():
            with open(osp.join(runner.work_dir, 'last_checkpoint'), 'w') as f:
                f.write(self.last_ckpt)

    @master_only
    def _publish_model(self, runner, ckpt_path: str) -> None:
        """Publishes a model by cleaning and versioning the checkpoint file.

        Args:
            runner (Runner): The runner of the training process.
            ckpt_path (str): Path to the checkpoint to publish.
        """
        from pipeai.checkpoint import load_checkpoint, save_checkpoint
        if not osp.exists(ckpt_path):
            runner.logger.warning(f'Checkpoint not found for publishing: {ckpt_path}')
            return

        # Load checkpoint to CPU
        checkpoint = load_checkpoint(ckpt_path, map_location='cpu')
        assert self.published_keys is not None

        # Remove keys not in the `published_keys` list
        removed_keys = [k for k in list(checkpoint.keys()) if k not in self.published_keys]
        for key in removed_keys:
            checkpoint.pop(key)
        if removed_keys:
            self.logger.info(f'Removed keys from published checkpoint: {removed_keys}')

        # Compute SHA256 hash for versioning and create the final path
        sha = hashlib.sha256(torch.save(checkpoint, io.BytesIO()).getvalue()).hexdigest()  # type: ignore
        final_path = osp.splitext(ckpt_path)[0] + f'-{sha[:8]}.pth'

        # Save the cleaned and versioned checkpoint
        save_checkpoint(checkpoint, final_path)
        self.logger.info(f'Published checkpoint {ckpt_path} to {final_path}.')
