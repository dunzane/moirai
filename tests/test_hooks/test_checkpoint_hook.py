# pylint: disable=invalid-name, inconsistent-quotes, unused-argument, protected-access
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

from pipeai.hooks import CheckpointHook


class TestCheckpointHook(unittest.TestCase):
    """Unit tests for the CheckpointHook class."""

    def setUp(self):
        self.runner = MagicMock()

        self.temp_dir = tempfile.mkdtemp()
        self.runner.work_dir = self.temp_dir

        self.runner.epoch = 0
        self.runner.iter = 0
        self.runner.max_epochs = 10
        self.runner.max_iters = 100

        self.message_hub_storage = {}

        def get_info(key, default=None):
            return self.message_hub_storage.get(key, default)

        def update_info(key, value):
            self.message_hub_storage[key] = value

        self.runner.message_hub.get_info = get_info
        self.runner.message_hub.update_info = update_info

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.message_hub_storage.clear()

    def test_save_by_epoch(self):
        hook = CheckpointHook(interval=2, by_epoch=True)
        hook.before_train(self.runner)

        for i in range(5):
            self.runner.epoch = i
            hook.after_train_epoch(self.runner)

        self.assertEqual(self.runner.save_checkpoint.call_count, 2)
        self.runner.save_checkpoint.assert_has_calls([
            call(self.temp_dir, 'epoch_2.pth', save_optimizer=True, save_param_scheduler=True,
                 meta={'epoch': 2, 'iter': 0}, by_epoch=True),
            call(self.temp_dir, 'epoch_4.pth', save_optimizer=True, save_param_scheduler=True,
                 meta={'epoch': 4, 'iter': 0}, by_epoch=True)
        ])

    def test_save_best_checkpoint(self):
        hook = CheckpointHook(save_best='acc', rule='greater')
        hook.before_train(self.runner)

        hook.after_val_epoch(self.runner, metrics={'acc': 0.8})
        self.runner.save_checkpoint.assert_called_once()
        self.assertIn('best_acc', self.runner.save_checkpoint.call_args[1]['filename'])

        self.runner.save_checkpoint.reset_mock()
        hook.after_val_epoch(self.runner, metrics={'acc': 0.7})
        self.runner.save_checkpoint.assert_not_called()

        self.runner.save_checkpoint.reset_mock()
        hook.after_val_epoch(self.runner, metrics={'acc': 0.9})
        self.runner.save_checkpoint.assert_called_once()
        self.assertIn('best_acc', self.runner.save_checkpoint.call_args[1]['filename'])

        self.assertEqual(self.message_hub_storage['best_score'], 0.9)

    @patch("pipeai.hooks.checkpoint_hook.os.path.exists", return_value=True)
    @patch("pipeai.hooks.checkpoint_hook.os.path.isfile", return_value=True)
    @patch("pipeai.hooks.checkpoint_hook.os.remove")
    def test_max_keep_ckpts(self, mock_remove, mock_isfile, mock_exists):
        hook = CheckpointHook(interval=1, max_keep_ckpts=2, by_epoch=True)
        hook.before_train(self.runner)

        self.runner.epoch = 0
        hook.after_train_epoch(self.runner)
        self.runner.epoch = 1
        hook.after_train_epoch(self.runner)
        mock_remove.assert_not_called()
        self.assertEqual(len(hook.keep_ckpt_ids), 2)

        self.runner.epoch = 2
        hook.after_train_epoch(self.runner)

        expected_path_to_remove = os.path.join(self.temp_dir, 'epoch_1.pth')
        mock_remove.assert_called_once_with(expected_path_to_remove)

        self.assertEqual(list(hook.keep_ckpt_ids), [2, 3])


if __name__ == '__main__':
    unittest.main()
