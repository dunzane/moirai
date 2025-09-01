# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from parameterized import parameterized

from pipeai.hooks import LoggerHook


class TestLoggerHook(unittest.TestCase):
    """Unit tests for the LoggerHook."""

    def setUp(self):
        """Set up a temporary directory and a mock runner for each test."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock the runner object with necessary attributes
        self.runner = MagicMock()
        self.runner.work_dir = self.temp_dir
        self.runner.timestamp = '20250101_120000'
        self.runner.experiment_name = 'test_exp'
        self.runner.logger = MagicMock()
        self.runner.log_processor = MagicMock()
        self.runner.visualizer = MagicMock()
        self.runner.train_dataloader = [1] * 50  # length 50
        self.runner.val_dataloader = [1] * 20  # length 20
        self.runner.test_dataloader = [1] * 20  # length 20
        self.runner.iter = 0
        self.runner.epoch = 1

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_before_run_creates_directory_and_sets_path(self):
        """Test if the output directory and JSON path are correctly set up."""
        hook = LoggerHook(out_dir=self.temp_dir)
        hook.before_run(self.runner)

        expected_out_dir = os.path.join(self.temp_dir, os.path.basename(self.temp_dir))
        self.assertTrue(os.path.exists(expected_out_dir))
        self.assertEqual(hook.out_dir, expected_out_dir)
        expected_json_path = os.path.join(expected_out_dir, '20250101_120000.json')
        self.assertEqual(hook.json_log_path, expected_json_path)

    @parameterized.expand([(9, True), (8, False)])
    def test_after_train_iter_logs_on_interval(self, batch_idx, should_log):
        """Test if logging occurs only at the specified interval."""
        hook = LoggerHook(interval=10, out_dir=self.temp_dir)
        hook.before_run(self.runner)  # to set json_log_path
        self.runner.iter = batch_idx

        self.runner.log_processor.get_log_after_iter.return_value = ({}, "dummy_log_string")

        hook.after_train_iter(self.runner, batch_idx)

        if should_log:
            self.runner.log_processor.get_log_after_iter.assert_called_once()
            self.runner.visualizer.add_scalars.assert_called_once()
        else:
            self.runner.log_processor.get_log_after_iter.assert_not_called()
            self.runner.visualizer.add_scalars.assert_not_called()


if __name__ == '__main__':
    unittest.main()
