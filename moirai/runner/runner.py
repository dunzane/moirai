# pylint: disable=inconsistent-quotes
import copy
import logging
import os
import time
from abc import abstractmethod
from typing import Union, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from moirai.checkpoint import get_ckpt_save_dir, backup_last_ckpt, save_ckpt, clear_ckpt, load_ckpt
from moirai.config import Config
from moirai.data import build_data_loader_ddp, build_data_loader, DevicePrefetcher
from moirai.dist import get_local_rank, master_only, is_master
from moirai.evaluator import MeterPool
from moirai.logging import get_logger
from moirai.optim.builder import build_optim, build_lr_scheduler
from moirai.utils import set_env
from moirai.device import to_device
from moirai.utils.timer import TimePredictor


class Runner:
    """Base Runner under Pytorch"""

    def __init__(self, cfg: Config):
        """Initialize the base Runner.

        Args:
            cfg (Config): Configuration object.
        """
        self.cfg = copy.deepcopy(cfg)

        # Default logger
        self.logger = get_logger('moirai')

        # Set environment variables
        set_env(cfg.get('ENV', {}))

        # Model related
        self.model_name = cfg['MODEL.NAME']
        self.model = self.build_model(cfg)

        # Checkpoint related
        self.ckpt_save_dir = get_ckpt_save_dir(cfg)
        if not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)
        self.logger.info(f"Set ckpt save dir: '{self.ckpt_save_dir}'")
        self.ckpt_save_strategy = None

        # Training state
        self.num_epochs = None
        self.start_epoch = None

        # Val state
        self.val_interval = 0

        # Optimizer and scheduler
        self.optim = None
        self.scheduler = None
        self.clip_grad_param = None

        # Dataloaders
        self.train_data_loader = None
        self.val_data_loader = None

        # Monitoring tools
        self.meter_pool = None
        self.best_metrics = None
        self.tensorboard_writer = None

    def build_logger(self, logger: logging.Logger = None, logger_name: str = None,
                     log_file_name: str = None, log_level: int = logging.INFO):
        """Build or replace the logger for this runner.

        Args:
            logger (logging.Logger, optional): If provided, use this logger directly.
            logger_name (str, optional): Name for a new logger instance.
            log_file_name (str, optional): If set, log to a file under `ckpt_save_dir`.
            log_level (int, optional): Logging level, e.g., logging.INFO or logging.DEBUG.

        Raises:
            TypeError: If neither `logger` nor `logger_name` is provided.
        """
        if logger is not None:
            self.logger = logger
        elif logger_name is not None:
            if log_file_name is not None:
                log_file_name = f"{log_file_name}_{time.strftime('%Y%m%d%H%M%S')}.log"
                log_file_path = os.path.join(self.ckpt_save_dir, log_file_name)
            else:
                log_file_path = None
            self.logger = get_logger(logger_name, log_file_path, log_level)
        else:
            raise TypeError("At least one of `logger` or `logger_name` must be provided.")

    def build_model(self, model_cfg: Config):
        """Build the model and apply device placement (GPU, DDP if available).

        Calls `define_model()` to instantiate the model,
        moves it to the appropriate device, and wraps it in DDP if needed.

        Args:
            model_cfg (Config): Configuration for the model.

        Returns:
            nn.Module: Instantiated and device-prepared model.
        """
        self.logger.info('Building model.')
        model = self.define_model()
        model = to_device(model)

        if torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=model_cfg.get('MODEL.DDP_FIND_UNUSED_PARAMETERS', False)
            )
        return model

    @abstractmethod
    def define_model(self) -> nn.Module:
        """Define the model architecture (to be implemented in subclass).

        This method must be implemented by subclasses to provide a model
        based on the configuration.

        Returns:
            nn.Module: The model instance.
        """
        pass

    @abstractmethod
    def build_train_dataset(self) -> Dataset:
        """
        Build the dataset used for training.

        Returns:
            Dataset: The constructed training dataset.
        """
        pass

    @staticmethod
    def build_val_dataset(cfg: Config) -> Dataset:
        """
        Build the dataset used for validation. This method is optional and
        can be overridden by subclass if validation is required.

        Args:
            cfg (Config): The configuration for building the validation dataset.

        Returns:
            Dataset: The constructed validation dataset.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError()

    def build_train_dataloader(self, cfg: Config) -> DataLoader:
        """
        Build the training dataloader.

        This method first constructs the training dataset by calling
        `self.build_train_dataset`, then builds the corresponding dataloader
        based on whether distributed training is initialized.

        Args:
            cfg (Config): Configuration containing training dataset and loader settings.

        Returns:
            DataLoader: The training dataloader instance.
        """
        self.logger.info('Building training data loader.')
        dataset = self.build_train_dataset()
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return build_data_loader(dataset, cfg['TRAIN.DATA'])

    def build_val_dataloader(self) -> DataLoader:
        """
        Build the validation dataloader.

        This method constructs the validation dataset by calling
        `self.build_val_dataset`, then builds the corresponding dataloader.

        Args:
            cfg (Config): Configuration containing validation dataset and loader settings.

        Returns:
            DataLoader: The validation dataloader instance.
        """
        self.logger.info('Building val data loader.')
        dataset = self.build_val_dataset(self.cfg)
        return build_data_loader(dataset, self.cfg['VAL.DATA'])

    @master_only
    def save_model(self, epoch: int):
        """Save checkpoint for the given epoch.

        Checkpoint structure:
            {
                'epoch': int,
                'model_state_dict': dict,
                'optim_state_dict': dict,
                'best_metrics': dict
            }

        Checkpoint file will be saved as:
            {ckpt_save_dir}/{model_name}_{epoch:0{len(str(num_epochs))}d}.pt

        Args:
            epoch (int): Current epoch index.
        """

        def _get_ckpt_path(e: int) -> str:
            """Generate checkpoint path string for given epoch."""
            epoch_str = str(e).zfill(len(str(self.num_epochs)))
            ckpt_name = f'{self.model_name}_{epoch_str}.pt'
            return os.path.join(self.ckpt_save_dir, ckpt_name)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics
        }

        # Backup previous checkpoint
        last_ckpt_path = _get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # Save current checkpoint
        ckpt_path = _get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # Periodically clear old checkpoints
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    def load_model_resume(self, strict: bool = True):
        """Resume training by loading the latest checkpoint.

        This includes restoring model weights, optimizer state, scheduler state,
        the best metrics, and the start epoch.

        Args:
            strict (bool, optional): Whether to strictly enforce that the keys in
                :attr:`state_dict` match the keys returned by the model's
                :meth:`state_dict` method. Default: True.
        """
        try:
            ckpt = load_ckpt(self.ckpt_save_dir, logger=self.logger)

            model = self.model.module if isinstance(self.model, DDP) else self.model
            model.load_state_dict(ckpt['model_state_dict'], strict=strict)

            self.optim.load_state_dict(ckpt['optim_state_dict'])
            self.start_epoch = ckpt.get('epoch', 0)

            if 'best_metrics' in ckpt:
                self.best_metrics = ckpt['best_metrics']

            if self.scheduler is not None:
                self.scheduler.last_epoch = self.start_epoch

            self.logger.info(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}.")

        except (IndexError, OSError, KeyError) as e:
            self.logger.warning(f"Failed to resume training: {e}")

    def load_model(self, ckpt_path: str = None, strict: bool = True):
        """Load model weights from checkpoint.

        If `ckpt_path` is not provided, load the latest checkpoint in `self.ckpt_save_dir`;
        otherwise, load from the specified path.

        Args:
            ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
            strict (bool, optional): Whether to strictly enforce that the keys in
                `state_dict` match the model's keys. Defaults to True.

        Raises:
            OSError: If the checkpoint file does not exist or cannot be loaded.
        """
        try:
            checkpoint_dict = load_ckpt(
                self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger
            )
            state_dict = checkpoint_dict['model_state_dict']
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(state_dict, strict=strict)
            else:
                self.model.load_state_dict(state_dict, strict=strict)
            self.logger.info(f'Loaded model weights from checkpoint: {ckpt_path or "latest"}')
        except (IndexError, OSError) as e:
            raise OSError('Checkpoint file does not exist or failed to load.') from e

    def train(self):
        """Train the model

        Train process:
        [init_training]
        for in train_epoch:
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]
        """
        self.init_training()

        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1

            self.on_epoch_start(epoch)
            epoch_start_time = time.time()

            self.model.train()  # start training

            # tqdm process bar
            if self.cfg.get('TRAIN.DATA.DEVICE_PREFETCH', False):
                data_loader = DevicePrefetcher(self.train_data_loader)
            else:
                data_loader = self.train_data_loader
            data_loader = tqdm(data_loader) if get_local_rank() == 0 else data_loader

            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # epoch time
            epoch_end_time = time.time()
            self.update_epoch_meter('train_time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            # estimate training finish time
            expected_end_time = train_time_predictor.get_expected_end_time(epoch)
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end()

    @abstractmethod
    def init_training(self):
        """Initialize the training"""
        self.logger.info('Initializing training.')

        # init training param
        self.num_epochs = self.cfg['TRAIN.NUM_EPOCHS']
        self.start_epoch = 0
        self.ckpt_save_strategy = self.cfg.get('TRAIN.CKPT_SAVE_STRATEGY')
        self.best_metrics = {}
        self.clip_grad_param = self.cfg.get('TRAIN.CLIP_GRAD_PARAM')
        if self.clip_grad_param is not None:
            self.logger.info('Set clip grad, param: {}'.format(self.clip_grad_param))

        # train data loader
        self.train_data_loader = self.build_train_dataloader(self.cfg)
        self.register_epoch_meter('train_time', 'train', '{:.2f} (s)', plt=False)

        # create optim
        self.optim = build_optim(self.cfg['TRAIN.OPTIM'], self.model)
        self.logger.info('Set optim: {}'.format(self.optim))

        # create lr_scheduler
        if self.cfg.has('TRAIN.LR_SCHEDULER'):
            self.scheduler = build_lr_scheduler(self.cfg['TRAIN.LR_SCHEDULER'], self.optim)
            self.logger.info('Set lr_scheduler: {}'.format(self.scheduler))
            self.register_epoch_meter('lr', 'train', '{:.2e}')

        # fine tune
        if self.cfg.has('TRAIN.FINETUNE_FROM'):
            self.load_model(self.cfg['TRAIN.FINETUNE_FROM'], self.cfg.get('TRAIN.FINETUNE_STRICT_LOAD', True))
            self.logger.info('Start fine tuning')

        # resume
        self.load_model_resume()

        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

        # init validation
        if self.cfg.has('VAL'):
            self.init_validation()

    @master_only
    def init_validation(self):
        """Initialize validation"""
        self.logger.info('Initializing validation.')
        self.val_interval = self.cfg.get('VAL.INTERVAL', 1)
        self.register_epoch_meter('val_time', 'val', '{:.2f} (s)', plt=False)

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        if self.meter_pool is None:
            self.meter_pool = MeterPool()
        self.meter_pool.register(name, meter_type, fmt, plt)

    def on_epoch_start(self, epoch: int):
        """Callback at the start of an epoch.

        Args:
            epoch (int): current epoch
        """
        # print epoch num info
        self.logger.info('Epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        # update lr meter
        if self.scheduler is not None:
            self._update_epoch_meter('lr', self.scheduler.get_last_lr()[0])

        # set epoch for sampler in distributed mode
        # see https://pytorch.org/docs/stable/data.html
        sampler = self.train_data_loader.sampler
        if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
            sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """
        # print train meters
        self._print_epoch_meters('train')
        # tensorboard plt meters
        self._plt_epoch_meters('train', epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # save model
        self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    def on_training_start(self):
        """Callback at the start of training.
        """
        pass

    def on_training_end(self):
        """Callback at the end of training.
        """

        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

    @master_only
    def _update_epoch_meter(self, name, value, n=1):
        self.meter_pool.update(name, value, n)

    @master_only
    def _print_epoch_meters(self, meter_type):
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def _plt_epoch_meters(self, meter_type, step):
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer)

    @abstractmethod
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """It must be implemented to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """
        pass

    def backward(self, loss: torch.Tensor):
        """Backward and update params.

        Args:
            loss (torch.Tensor): loss
        """
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad_param is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
        self.optim.step()

    @torch.no_grad()
    @master_only
    def validate(self, train_epoch: Optional[int] = None):
        """Validate model.

        Args:
            train_epoch (int, optional): current epoch if in a training process.
        """
        # init validation if not in a training process
        if train_epoch is None:
            self.init_validation()

        self.logger.info('Start validation.')
        self.on_validating_start(train_epoch)

        val_start_time = time.time()
        self.model.eval()

        # tqdm process bar
        data_iter = tqdm(self.val_data_loader)

        # val loop
        for iter_index, data in enumerate(data_iter):
            self.val_iters(iter_index, data)

        val_end_time = time.time()
        self.update_epoch_meter('val_time', val_end_time - val_start_time)
        # print val meters
        self._print_epoch_meters('val')
        if train_epoch is not None:
            # tensorboard plt meters
            self._plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end(train_epoch)

    @master_only
    def on_validating_start(self, train_epoch: Optional[int]):
        """Callback at the start of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """
        pass

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        pass

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """It can be implemented to define validating detail (not necessary).

        Args:
            iter_index (int): current iter.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
        """
        raise NotImplementedError()

    @master_only
    def update_epoch_meter(self, name, value, n=1):
        self.meter_pool.update(name, value, n)

    @master_only
    def reset_epoch_meters(self):
        self.meter_pool.reset()

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save the best model while training.

        Examples:
            >>> def on_validating_end(self, train_epoch: Optional[int]):
            >>>     if train_epoch is not None:
            >>>         self.save_best_model(train_epoch, 'val/loss', greater_best=False)

        Args:
            epoch (int): current epoch.
            metric_name (str): metric name used to measure the model, must be registered in `epoch_meter`.
            greater_best (bool, optional): `True` means greater value is best, such as `acc`
                `False` means lower value is best, such as `loss`. Defaults to True.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
