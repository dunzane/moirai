import unittest
from unittest.mock import MagicMock, patch

from pipeai.hooks import IterTimerHook


class MockScalar:
    """Mock scalar object that mimics .current() behavior."""

    def __init__(self, value):
        self._value = value

    def current(self):
        return self._value


class MockMessageHub:
    """Mock message hub to store scalars and infos."""

    def __init__(self):
        self.scalars = {}
        self.infos = {}

    def update_scalar(self, name, value):
        self.scalars[name] = MockScalar(value)

    def get_scalar(self, name):
        return self.scalars[name]

    def update_info(self, name, value):
        self.infos[name] = value


class TestIterTimerHook(unittest.TestCase):
    """Unit tests for the IterTimerHook."""

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()
        self.runner.max_iters = 100
        self.runner.message_hub = MockMessageHub()
        # Simulate dataloaders
        self.runner.val_dataloader = list(range(3))  # 3 batches
        self.runner.test_dataloader = list(range(2))  # 2 batches
        self.hook = IterTimerHook()

    @patch('pipeai.hooks.iter_timer_hook.time.time')
    def test_train_iteration_timing(self, mock_time):
        """Test timing and ETA calculation in training phase."""
        mock_time.side_effect = [1000.0, 1000.2, 1001.0]

        self.runner.iter = 0
        self.hook.before_train(self.runner)

        self.hook._before_epoch(self.runner, stage='train')  # 1000.0
        self.hook._before_iter(self.runner, 0, stage='train')  # 1000.2
        self.runner.iter = 1
        self.hook._after_iter(self.runner, 0, stage='train')  # 1001.0

        hub = self.runner.message_hub
        self.assertAlmostEqual(hub.get_scalar('train/data_time').current(), 0.2, places=2)
        self.assertAlmostEqual(hub.get_scalar('train/time').current(), 1.0, places=2)

        # ETA = (total_train_time / (runner.iter - start_iter + 1)) * (max_iters - iter - 1)
        #      = (1.0 / (1 - 0 + 1)) * (100 - 1 - 1)
        #      = 0.5 * 98 = 49.0
        self.assertAlmostEqual(hub.infos['eta'], 49.0, places=2)

    @patch('pipeai.hooks.iter_timer_hook.time.time')
    def test_val_iteration_timing_and_reset(self, mock_time):
        """Test timing for a validation loop and ensure the eval timer is reset."""
        mock_time.side_effect = [2000.0, 2000.1, 2000.5]
        self.runner.val_dataloader = [1, 2, 3]

        self.hook._before_epoch(self.runner, stage='val')       # t=2000.0
        self.hook._before_iter(self.runner, 0, stage='val')     # t=2000.1
        self.hook._after_iter(self.runner, 0, stage='val')      # t=2000.5

        hub = self.runner.message_hub
        self.assertAlmostEqual(hub.get_scalar('val/data_time').current(), 0.1)
        self.assertAlmostEqual(hub.get_scalar('val/time').current(), 0.5)

        # --- Assertions for ETA ---
        # ETA = avg_iter_time * (len(dataloader) - batch_idx - 1)
        # avg_iter_time = total_eval_time / (batch_idx + 1) = 0.5 / 1 = 0.5
        # eta = 0.5 * (3 - 0 - 1) = 1.0
        self.assertAlmostEqual(hub.infos['eta'], 1.0)

        # --- Test Reset ---
        # Ensure the timer is reset after the epoch
        self.hook._after_epoch(self.runner, stage='val')
        self.assertEqual(self.hook.total_eval_time, 0.0)

    @patch('pipeai.hooks.iter_timer_hook.time.time')
    def test_test_iteration_timing(self, mock_time):
        mock_time.side_effect = [3000.0, 3000.1, 3000.6]
        self.runner.test_dataloader = [1, 2, 3, 4]

        self.hook._before_epoch(self.runner, stage='test')    # t=3000.0
        self.hook._before_iter(self.runner, 0, stage='test')  # t=3000.1
        self.hook._after_iter(self.runner, 0, stage='test')   # t=3000.6

        hub = self.runner.message_hub
        self.assertAlmostEqual(hub.get_scalar('test/data_time').current(), 0.1)
        self.assertAlmostEqual(hub.get_scalar('test/time').current(), 0.6)

        # --- Assertions for ETA ---
        # avg_iter_time = total_eval_time / (batch_idx + 1) = 0.6 / 1 = 0.6
        # eta = avg_iter_time * (len(dataloader) - batch_idx - 1)
        # eta = 0.6 * (4 - 0 - 1) = 0.6 * 3 = 1.8
        # Use hub.info, not hub.infos
        self.assertAlmostEqual(hub.infos['eta'], 1.8)

    def test_after_epoch_resets_eval_time(self):
        """Test that eval time is reset after each epoch."""
        self.hook.total_eval_time = 123.0
        self.hook._after_epoch(self.runner, stage='val')
        self.assertEqual(self.hook.total_eval_time, 0.0)


if __name__ == '__main__':
    unittest.main()
