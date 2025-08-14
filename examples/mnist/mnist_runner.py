from typing import Union, Tuple, Dict

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset

from examples.mnist.conv_net import ConvNet
from pipeai import Runner, Config
from pipeai.device import to_device


class MNISTRunner(Runner):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.loss = None

    def define_model(self) -> nn.Module:
        return {
            'conv_net': ConvNet
        }[self.cfg['MODEL']['NAME']](**self.cfg['MODEL'].get('PARAM', {}))

    def init_training(self):
        super().init_training()

        self.loss = nn.NLLLoss()
        self.loss = to_device(self.loss)

        self.register_epoch_meter('train_loss', 'train', '{:.2f}')

    def build_train_dataset(self) -> Dataset:
        return torchvision.datasets.MNIST(
            self.cfg['TRAIN']['DATA']['DIR'], train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        input_, target_ = data
        input_ = to_device(input_)
        target_ = to_device(target_)

        output = self.model(input_)
        loss = self.loss(output, target_)
        self.update_epoch_meter('train_loss', loss.item())
        return loss

    def init_validation(self):
        """Initialize validation.

        Including validation meters, etc.
        """
        super().init_validation()

        self.register_epoch_meter('val_acc', 'val', '{:.2f}%')

    def build_val_dataset(self):
        """Build MNIST val dataset

        Returns:
            train dataset (Dataset)
        """

        return torchvision.datasets.MNIST(
            self.cfg['VAL']['DATA']['DIR'], train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """

        input_, target_ = data
        input_ = to_device(input_)
        target_ = to_device(target_)

        output = self.model(input_)
        pred = output.data.max(1, keepdim=True)[1]
        self.update_epoch_meter('val_acc', 100 * pred.eq(target_.data.view_as(pred)).sum())
