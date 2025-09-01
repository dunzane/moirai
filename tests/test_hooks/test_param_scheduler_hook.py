# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import MagicMock, patch
from torch.optim import lr_scheduler

from pipeai.hooks import ParamSchedulerHook


class TestParamSchedulerHook(unittest.TestCase):
    """Unit tests for the ParamSchedulerHook."""

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()
        self.hook = ParamSchedulerHook()

    def test_after_train_iter_steps_schedulers(self):
        mock_scheduler = MagicMock()
        self.runner.param_schedulers = [mock_scheduler]

        self.hook.after_train_iter(self.runner, batch_idx=0)

        mock_scheduler.step.assert_called_once()

    def test_after_train_epoch_steps_schedulers(self):
        mock_scheduler = MagicMock()
        self.runner.param_schedulers = [mock_scheduler]

        self.hook.after_train_epoch(self.runner)

        mock_scheduler.step.assert_called_once()

    def test_after_val_epoch_steps_plateau_scheduler(self):
        mock_plateau_scheduler = MagicMock(spec=lr_scheduler.ReduceLROnPlateau)
        mock_standard_scheduler = MagicMock(spec=lr_scheduler.StepLR)

        self.runner.param_schedulers = [mock_plateau_scheduler, mock_standard_scheduler]
        mock_metrics = {'val/accuracy': 0.95}

        self.hook.after_val_epoch(self.runner, metrics=mock_metrics)

        mock_plateau_scheduler.step.assert_called_once_with(0.95)
        mock_standard_scheduler.step.assert_not_called()

    def test_handles_dict_of_schedulers(self):
        mock_scheduler1 = MagicMock()
        mock_scheduler2 = MagicMock()
        self.runner.param_schedulers = {
            'optimizer1': [mock_scheduler1],
            'optimizer2': [mock_scheduler2]
        }

        self.hook.after_train_iter(self.runner, batch_idx=0)

        mock_scheduler1.step.assert_called_once()
        mock_scheduler2.step.assert_called_once()

    def test_does_nothing_if_no_schedulers(self):
        self.runner.param_schedulers = None
        try:
            self.hook.after_train_iter(self.runner, batch_idx=0)
            self.hook.after_train_epoch(self.runner)
            self.hook.after_val_epoch(self.runner)
        except Exception as e:
            self.fail(f"Hook raised an unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()
