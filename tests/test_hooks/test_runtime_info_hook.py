import unittest
from unittest.mock import MagicMock, patch

from pipeai.hooks import RuntimeInfoHook


class TestRuntimeInfoHook(unittest.TestCase):
    """Unit tests for the RuntimeInfoHook."""

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()
        self.runner.message_hub = MagicMock()
        self.runner.cfg.pretty_text = "config_text"
        self.runner.seed = 42
        self.runner.experiment_name = "test_exp"
        self.runner.epoch = 5
        self.runner.iter = 100
        self.runner.max_epochs = 10
        self.runner.max_iters = 1000

        # Mock dataloader with a metainfo attribute
        mock_dataset = MagicMock()
        mock_dataset.metainfo = {'classes': ['cat', 'dog']}
        self.runner.train_dataloader.dataset = mock_dataset

        # Mock optimizer wrapper
        self.runner.optim_wrapper.get_lr.return_value = {'lr': [0.01]}

        self.hook = RuntimeInfoHook()

    @patch('pipeai.hooks.runtime_info_hook.get_git_hash', new=lambda: "+abc1234")
    @patch('pipeai.hooks.runtime_info_hook.__version__', new="1.0.0")
    def test_before_run_updates_meta_info(self):
        self.hook.before_run(self.runner)

        expected_metainfo = {
            'cfg': 'config_text',
            'seed': 42,
            'experiment_name': 'test_exp',
            'pipeai_version': '1.0.0+abc1234'
        }
        self.runner.message_hub.update_info_dict.assert_called_once_with(expected_metainfo)

    def test_before_train_updates_train_state(self):
        self.hook.before_train(self.runner)

        self.runner.message_hub.update_info.assert_any_call("loop_stage", "train")
        self.runner.message_hub.update_info.assert_any_call("epoch", 5)
        self.runner.message_hub.update_info.assert_any_call("iter", 100)
        self.runner.message_hub.update_info.assert_any_call("max_epochs", 10)
        self.runner.message_hub.update_info.assert_any_call("max_iters", 1000)
        self.runner.message_hub.update_info.assert_any_call(
            "dataset_meta", {'classes': ['cat', 'dog']})

    def test_before_train_iter_updates_lr_and_iter(self):
        """Test if iter count and learning rates are updated before an iteration."""
        self.runner.iter = 101  # Simulate next iteration
        self.hook.before_train_iter(self.runner, batch_idx=1)

        self.runner.message_hub.update_info.assert_called_with("iter", 101)
        self.runner.message_hub.update_scalar.assert_called_once_with("train/lr", 0.01)

    def test_after_train_iter_updates_outputs(self):
        """Test if model outputs are logged after a training iteration."""
        import torch

        outputs = {
            'loss': torch.tensor(0.5),
            'accuracy': 0.9,
            'prediction_map': torch.rand(4, 4)  # Not a scalar
        }
        self.hook.after_train_iter(self.runner, batch_idx=0, outputs=outputs)

        self.runner.message_hub.update_scalar.assert_any_call("train/loss", outputs['loss'])
        self.runner.message_hub.update_scalar.assert_any_call("train/accuracy", outputs['accuracy'])
        self.runner.message_hub.update_info.assert_called_once_with(
            "train/prediction_map", outputs['prediction_map'])

    def test_val_loop_stage_management(self):
        self.runner.message_hub.get_info.return_value = "train"

        self.hook.before_val(self.runner)
        self.runner.message_hub.update_info.assert_called_with("loop_stage", "val")
        self.assertEqual(self.hook.last_loop_stage, "train")

        self.hook.after_val(self.runner)
        self.runner.message_hub.update_info.assert_called_with("loop_stage", "train")

    def test_after_val_epoch_updates_metrics(self):
        import numpy as np
        metrics = {'accuracy': 0.98, 'f1_score': np.float32(0.95)}
        self.hook.after_val_epoch(self.runner, metrics=metrics)

        self.runner.message_hub.update_scalar.assert_any_call("val/accuracy", 0.98)
        self.runner.message_hub.update_scalar.assert_any_call("val/f1_score", np.float32(0.95))

    def test_is_scalar_helper_method(self):
        """Test the _is_scalar helper with various data types."""
        import torch
        import numpy as np

        self.assertTrue(self.hook._is_scalar(10))
        self.assertTrue(self.hook._is_scalar(10.5))
        self.assertTrue(self.hook._is_scalar(np.float32(5.0)))
        self.assertTrue(self.hook._is_scalar(torch.tensor(5)))
        self.assertTrue(self.hook._is_scalar(np.array(5)))

        self.assertFalse(self.hook._is_scalar([1, 2]))
        self.assertFalse(self.hook._is_scalar("string"))
        self.assertFalse(self.hook._is_scalar(torch.tensor([1, 2])))
        self.assertFalse(self.hook._is_scalar(np.array([1, 2])))


if __name__ == '__main__':
    unittest.main()
