from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader


class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should
    overwrite the: meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        from pipeai.logging import get_logger
        self.logger = get_logger('pipeai-loop')

        self._runner = runner

        self._dataloader = None
        if isinstance(dataloader, dict):
            # Determine whether different ranks use different seed.
            diff_rank_seed = runner.randomness_cfg.get(
                'diff_rank_seed', False)
            self._dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self._dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @property
    def dataloader(self):
        return self._dataloader

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""
