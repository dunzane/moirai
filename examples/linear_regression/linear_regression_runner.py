from typing import Union, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from examples.linear_regression.dataset import LinearDataset
from pipeai import Runner, Config
from pipeai.device import to_device


class LinearRegressionRunner(Runner):
    """LinearRegressionRunner"""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.loss = nn.MSELoss()
        self.loss = to_device(self.loss)

    def init_training(self):
        super().init_training()
        self.register_epoch_meter('train_loss', 'train', '{:.2f}')

    def define_model(self) -> nn.Module:
        return nn.Linear(1, 1)

    def build_train_dataset(self) -> Dataset:
        return LinearDataset(self.cfg['TRAIN']['DATA']['K'],
                             self.cfg['TRAIN']['DATA']['B'],
                             self.cfg['TRAIN']['DATA']['NUM'])

    def train_iters(self, epoch: int,
                    iter_index: int,
                    data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        x, y = data
        x = to_device(x)
        y = to_device(y)

        output = self.model(x)
        loss = self.loss(output, y)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def on_training_end(self):
        super().on_training_end()
        self.logger.info('Result: k: {}, b: {}'.format(self.model.weight.item(), self.model.bias.item()))

