# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import MagicMock, patch

from pipeai.hooks import ProfilerHook


class TestProfilerHook(unittest.TestCase):
    """Unit tests for the ProfilerHook."""

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()

    @patch('pipeai.hooks.profiler_hook.profile')
    def test_before_train_initializes_and_starts_profiler(self, mock_profile):
        mock_profiler_instance = MagicMock()
        mock_profile.return_value = mock_profiler_instance

        hook = ProfilerHook(record_shapes=False, with_stack=True)
        hook.before_train(self.runner)

        mock_profile.assert_called_once()
        called_kwargs = mock_profile.call_args[1]
        self.assertEqual(called_kwargs['record_shapes'], False)
        self.assertEqual(called_kwargs['with_stack'], True)

        mock_profiler_instance.__enter__.assert_called_once()

        self.assertIsNotNone(hook.profiler)

    def test_after_train_iter_steps_profiler(self):
        hook = ProfilerHook()
        mock_profiler = MagicMock()
        hook.profiler = mock_profiler

        hook.after_train_iter(self.runner, batch_idx=0)

        mock_profiler.step.assert_called_once()

    def test_after_train_stops_and_clears_profiler(self):
        hook = ProfilerHook()
        mock_profiler = MagicMock()
        hook.profiler = mock_profiler

        hook.after_train(self.runner)

        mock_profiler.__exit__.assert_called_once_with(None, None, None)

        self.assertIsNone(hook.profiler)

    def test_hooks_do_nothing_if_profiler_is_none(self):
        hook = ProfilerHook()
        self.assertIsNone(hook.profiler)

        try:
            hook.after_train_iter(self.runner, batch_idx=0)
            hook.after_train(self.runner)
        except Exception as e:
            self.fail(f"Hook raised an unexpected exception when profiler was None: {e}")


if __name__ == '__main__':
    unittest.main()
