# pylint: disable=invalid-name, inconsistent-quotes, unused-argument, protected-access
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import MagicMock

from pipeai.hooks import EarlyStoppingHook


class TestEarlyStoppingHook(unittest.TestCase):

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()
        # The hook interacts with the runner's train_loop and logger
        self.runner.train_loop = MagicMock()
        self.runner.logger = MagicMock()
        self.runner.train_loop.stop_training = False

    def test_stop_after_patience_greater_rule(self):
        """Test that training stops after `patience` epochs of no improvement."""
        hook = EarlyStoppingHook(monitor='acc', rule='greater', patience=3, min_delta=0.0)
        hook.before_run(self.runner)

        # Initial best score
        hook.after_val_epoch(self.runner, metrics={'acc': 0.8})
        self.assertEqual(hook.best_score, 0.8)
        self.assertEqual(hook.wait_count, 0)
        self.assertFalse(self.runner.train_loop.stop_training)

        # No improvement, wait_count = 1
        hook.after_val_epoch(self.runner, metrics={'acc': 0.79})
        self.assertEqual(hook.wait_count, 1)
        self.assertFalse(self.runner.train_loop.stop_training)

        # No improvement, wait_count = 2
        hook.after_val_epoch(self.runner, metrics={'acc': 0.78})  # Equal is not an improvement
        self.assertEqual(hook.wait_count, 2)
        self.assertFalse(self.runner.train_loop.stop_training)

        # No improvement, wait_count = 3, stop training
        hook.after_val_epoch(self.runner, metrics={'acc': 0.75})
        self.assertEqual(hook.wait_count, 3)
        self.assertTrue(self.runner.train_loop.stop_training)
        self.runner.logger.info.assert_called_once()

    def test_patience_counter_resets_on_improvement(self):
        """Test that the patience counter resets when the metric improves."""
        hook = EarlyStoppingHook(monitor='acc', rule='greater', patience=3, min_delta=0.0)
        hook.before_run(self.runner)

        hook.after_val_epoch(self.runner, metrics={'acc': 0.8})  # Best: 0.8
        hook.after_val_epoch(self.runner, metrics={'acc': 0.7})  # Wait: 1
        self.assertEqual(hook.wait_count, 1)

        # Metric improves, counter should reset
        hook.after_val_epoch(self.runner, metrics={'acc': 0.9})
        self.assertEqual(hook.best_score, 0.9)
        self.assertEqual(hook.wait_count, 0)
        self.assertFalse(self.runner.train_loop.stop_training)

    def test_stop_after_patience_less_rule(self):
        """Test the stopping logic for a metric that should be minimized."""
        hook = EarlyStoppingHook(monitor='loss', rule='less', patience=2, min_delta=0.0)
        hook.before_run(self.runner)

        hook.after_val_epoch(self.runner, metrics={'loss': 1.0})  # Best: 1.0
        self.assertFalse(self.runner.train_loop.stop_training)

        hook.after_val_epoch(self.runner, metrics={'loss': 1.1})  # No improvement, wait: 1
        self.assertEqual(hook.wait_count, 1)
        self.assertFalse(self.runner.train_loop.stop_training)

        hook.after_val_epoch(self.runner, metrics={'loss': 1.05})  # No improvement, wait: 2
        self.assertTrue(self.runner.train_loop.stop_training)
        self.runner.logger.info.assert_called_once()

    def test_min_delta_logic(self):
        """Test that improvements smaller than `min_delta` are ignored."""
        hook = EarlyStoppingHook(monitor='acc', rule='greater', patience=2, min_delta=0.1)
        hook.before_run(self.runner)

        hook.after_val_epoch(self.runner, metrics={'acc': 0.8})  # Best: 0.8
        self.assertFalse(self.runner.train_loop.stop_training)

        # Improvement of 0.05 is less than min_delta of 0.1, so it's not an "improvement"
        hook.after_val_epoch(self.runner, metrics={'acc': 0.85})
        self.assertEqual(hook.wait_count, 1)
        self.assertFalse(self.runner.train_loop.stop_training)

        # Another insignificant improvement triggers the stop condition
        hook.after_val_epoch(self.runner, metrics={'acc': 0.86})
        self.assertTrue(self.runner.train_loop.stop_training)

    def test_stopping_threshold(self):
        """Test that training stops immediately when `stopping_threshold` is met."""
        hook = EarlyStoppingHook(monitor='acc', rule='greater', stopping_threshold=0.95)
        hook.before_run(self.runner)

        hook.after_val_epoch(self.runner, metrics={'acc': 0.9})
        self.assertFalse(self.runner.train_loop.stop_training)

        # Metric surpasses the threshold, training should stop
        hook.after_val_epoch(self.runner, metrics={'acc': 0.96})
        self.assertTrue(self.runner.train_loop.stop_training)
        self.runner.logger.info.assert_called_once_with(
            'Stopping threshold reached: `acc` = 0.96 is greater than 0.95.')

    def test_check_finite(self):
        """Test that training stops if the metric is not a finite number."""
        hook = EarlyStoppingHook(monitor='loss', rule='less', check_finite=True)
        hook.before_run(self.runner)

        # Should stop for infinity
        hook.after_val_epoch(self.runner, metrics={'loss': float('inf')})
        self.assertTrue(self.runner.train_loop.stop_training)

        # Reset and test for NaN
        self.runner.train_loop.stop_training = False
        hook.after_val_epoch(self.runner, metrics={'loss': float('nan')})
        self.assertTrue(self.runner.train_loop.stop_training)

    def test_rule_inference(self):
        """Test the automatic inference of the comparison rule."""
        hook_acc = EarlyStoppingHook(monitor='val/acc')
        self.assertEqual(hook_acc.rule, 'greater')

        hook_loss = EarlyStoppingHook(monitor='train_loss')
        self.assertEqual(hook_loss.rule, 'less')

        hook_map = EarlyStoppingHook(monitor='mAP')
        self.assertEqual(hook_map.rule, 'greater')

        with self.assertRaises(ValueError):
            EarlyStoppingHook(monitor='my_unknown_metric')

    def test_strict_mode(self):
        """Test the behavior of `strict` mode when the metric is missing."""
        # With strict=True, it should raise an error.
        strict_hook = EarlyStoppingHook(monitor='acc', strict=True)
        with self.assertRaises(RuntimeError):
            strict_hook.after_val_epoch(self.runner, metrics={'loss': 0.5})

        # With strict=False, it should issue a warning and continue.
        non_strict_hook = EarlyStoppingHook(monitor='acc', strict=False)
        with self.assertWarns(UserWarning):
            non_strict_hook.after_val_epoch(self.runner, metrics={'loss': 0.5})
        self.assertFalse(self.runner.train_loop.stop_training)


if __name__ == '__main__':
    unittest.main()
