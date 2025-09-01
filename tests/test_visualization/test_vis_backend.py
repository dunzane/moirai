# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch, call, ANY
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

from pipeai import Config
from pipeai.visualization import (LocalVisBackend,
                                  TensorboardVisBackend,
                                  SwanlabBackend)
from pipeai.utils import assert_calls_almost_equal


class TestLocalVisBackend(unittest.TestCase):
    """Unit tests for the LocalVisBackend."""

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.vis_backend = LocalVisBackend(save_dir=self.temp_dir)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    @patch('pipeai.visualization.vis_backend.save_config_str')
    def test_add_config_saves_file(self, mock_save_config_str):
        mock_config = MagicMock(spec=Config)
        self.vis_backend.add_config(mock_config)

        expected_config_path = os.path.join(self.temp_dir, 'config.py')
        mock_save_config_str.assert_called_once_with(mock_config, expected_config_path)

    def test_add_image_saves_image_file(self):
        image_name = "test_image"
        step = 1
        dummy_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

        self.vis_backend.add_image(image_name, dummy_image, step)

        expected_img_dir = os.path.join(self.temp_dir, 'vis_image')
        expected_file_path = os.path.join(expected_img_dir, f"{image_name}_{step}.png")

        self.assertTrue(os.path.exists(expected_file_path))
        saved_img = Image.open(expected_file_path)
        self.assertEqual(saved_img.size, (10, 10))

    def test_add_scalar_writes_to_jsonl(self):
        self.vis_backend.add_scalar("loss", 0.5, step=10)
        self.vis_backend.add_scalar("accuracy", torch.tensor(0.95), step=10)

        scalar_file_path = os.path.join(self.temp_dir, 'scalars.json')
        self.assertTrue(os.path.exists(scalar_file_path))

        with open(scalar_file_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            self.assertEqual(data1, {'loss': 0.5, 'step': 10})
            self.assertAlmostEqual(data2['accuracy'], 0.95, places=5)

    def test_add_scalars_writes_to_multiple_files(self):
        scalar_dict = {'metric1': 1.0, 'metric2': 2.0}
        custom_file = 'custom_metrics.json'
        step = 100

        self.vis_backend.add_scalars(scalar_dict, step=step, file_path=custom_file)

        default_scalar_file = os.path.join(self.temp_dir, 'scalars.json')
        self.assertTrue(os.path.exists(default_scalar_file))
        with open(default_scalar_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['metric1'], 1.0)
            self.assertEqual(data['step'], step)

        custom_scalar_file = os.path.join(self.temp_dir, custom_file)
        self.assertTrue(os.path.exists(custom_scalar_file))
        with open(custom_scalar_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['metric2'], 2.0)
            self.assertEqual(data['step'], step)


class TestTensorboardVisBackend(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and fake SummaryWriter for each test."""
        self.temp_dir = tempfile.mkdtemp()

        # fake writer
        self.fake_writer_cls = MagicMock()
        self.fake_writer_instance = self.fake_writer_cls.return_value

        sys.modules['torch.utils.tensorboard'] = types.SimpleNamespace(  # type:ignore
            SummaryWriter=self.fake_writer_cls
        )
        sys.modules['tensorboardX'] = types.SimpleNamespace(  # type:ignore
            SummaryWriter=self.fake_writer_cls
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_add_config_writes_text(self):
        vis_backend = TensorboardVisBackend(save_dir=self.temp_dir)
        mock_config = SimpleNamespace(pretty_text="model = dict(type='ResNet')")

        vis_backend.add_config(mock_config)  # type: ignore

        self.fake_writer_cls.assert_called_once_with(self.temp_dir)
        self.fake_writer_instance.add_text.assert_called_once_with(
            'config', mock_config.pretty_text
        )

    def test_add_image_writes_image(self):
        vis_backend = TensorboardVisBackend(save_dir=self.temp_dir)
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        step = 5

        vis_backend.add_image("test_img", dummy_image, step)

        self.fake_writer_instance.add_image.assert_called_once_with(
            "test_img", dummy_image, step, dataformats='HWC'
        )

    def test_add_scalar_writes_scalar(self):
        vis_backend = TensorboardVisBackend(save_dir=self.temp_dir)
        step = 10

        vis_backend.add_scalar("loss", 0.5, step)
        vis_backend.add_scalar("accuracy", torch.tensor(0.95), step)

        calls = self.fake_writer_instance.add_scalar.call_args_list
        self.assertEqual(calls[0][0][0], "loss")
        self.assertAlmostEqual(calls[0][0][1], 0.5, places=6)
        self.assertEqual(calls[0][0][2], step)

        self.assertEqual(calls[1][0][0], "accuracy")
        self.assertAlmostEqual(calls[1][0][1], 0.95, places=6)
        self.assertEqual(calls[1][0][2], step)

    def test_add_scalars_writes_multiple_scalars(self):
        vis_backend = TensorboardVisBackend(save_dir=self.temp_dir)
        scalar_dict = {'metric1': 1.0, 'metric2': torch.tensor(2.0)}
        step = 100

        vis_backend.add_scalars(scalar_dict, step=step)

        calls = self.fake_writer_instance.add_scalar.call_args_list

        # metric1
        name, value, step_arg = calls[0][0]
        self.assertEqual(name, 'metric1')
        self.assertEqual(step_arg, step)
        self.assertAlmostEqual(value, 1.0, places=6)

        # metric2
        name, value, step_arg = calls[1][0]
        self.assertEqual(name, 'metric2')
        self.assertEqual(step_arg, step)
        self.assertAlmostEqual(value, 2.0, places=6)

    def test_close_closes_writer(self):
        vis_backend = TensorboardVisBackend(save_dir=self.temp_dir)

        vis_backend.add_scalar("dummy", 1.0)
        self.assertIsNotNone(vis_backend._tensorboard)

        vis_backend.close()

        self.fake_writer_instance.close.assert_called_once()
        self.assertIsNone(vis_backend._tensorboard)


class TestSwanlabBackend(unittest.TestCase):
    """Unit tests for the SwanlabBackend."""

    def setUp(self):
        """Set up a temporary directory and mock swanlab in sys.modules."""
        self.temp_dir = tempfile.mkdtemp()

        # 1. Create a comprehensive mock for the entire swanlab module
        self.mock_swanlab = MagicMock()
        # Mock the `init` function and its return value (the run object)
        self.mock_run_instance = self.mock_swanlab.init.return_value
        # Mock the `Image` class within the module
        self.mock_swanlab.Image = MagicMock()

        # 2. Use patch.dict to temporarily insert the mock into sys.modules.
        #    This intercepts the `import swanlab` call inside the hook.
        self.sys_modules_patcher = patch.dict('sys.modules', {'swanlab': self.mock_swanlab})
        self.sys_modules_patcher.start()

    def tearDown(self):
        """Clean up the temporary directory and stop the patcher."""
        # 3. Stop the patcher to clean up sys.modules
        self.sys_modules_patcher.stop()
        shutil.rmtree(self.temp_dir)

    def test_add_config_updates_swanlab_config(self):
        """Test if the config is correctly passed to swanlab's config object."""
        vis_backend = SwanlabBackend(
            save_dir=self.temp_dir, project='test_proj', experiment_name='test_run')
        # Create a Config object that can be cast to a dict
        mock_config = Config({'model': 'ResNet', 'lr': 0.01})

        vis_backend.add_config(mock_config)

        # Assert swanlab.init was called with the correct parameters
        self.mock_swanlab.init.assert_called_once_with(
            project='test_proj', experiment_name='test_run', dir=self.temp_dir)
        # Assert the config was updated on the run object
        self.mock_run_instance.config.update.assert_called_once_with(mock_config)

    def test_add_image_logs_swanlab_image(self):
        """Test if an image is correctly logged as a swanlab.Image."""
        vis_backend = SwanlabBackend(save_dir=self.temp_dir)
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        step = 5

        vis_backend.add_image("test_img", dummy_image, step)

        # Assert swanlab.Image was called with the numpy array
        self.mock_swanlab.Image.assert_called_once_with(dummy_image)
        # Assert the final dictionary was logged
        expected_log = {'test_img': self.mock_swanlab.Image.return_value}
        self.mock_run_instance.log.assert_called_once_with(expected_log, step=step)

    def test_add_scalar_logs_scalar(self):
        """Test if a single scalar value is logged to swanlab."""
        vis_backend = SwanlabBackend(save_dir=self.temp_dir)
        step = 10

        vis_backend.add_scalar("loss", 0.5, step)
        vis_backend.add_scalar("accuracy", torch.tensor(0.95), step)

        # Check calls to the mock run's log method
        expected_calls = [
            call({'loss': 0.5}, step=step),
            call({'accuracy': 0.95}, step=step)
        ]

        actual_calls = self.mock_run_instance.log.call_args_list

        assert_calls_almost_equal(self, actual_calls, expected_calls, places=6)

    def test_add_scalars_logs_dict(self):
        """Test if a dictionary of scalars is logged correctly."""
        vis_backend = SwanlabBackend(save_dir=self.temp_dir)
        scalar_dict = {'metric1': 1.0, 'metric2': 2.0}
        step = 100

        vis_backend.add_scalars(scalar_dict, step=step)

        self.mock_run_instance.log.assert_called_once_with(scalar_dict, step=step)

    def test_close_finishes_run(self):
        """Test if the close method correctly finishes the swanlab run."""
        vis_backend = SwanlabBackend(save_dir=self.temp_dir)

        # Call any method to initialize the run
        vis_backend.add_scalar("dummy", 1.0)
        self.assertIsNotNone(vis_backend._swanlab)

        vis_backend.close()

        self.mock_run_instance.finish.assert_called_once()
        self.assertIsNone(vis_backend._swanlab)


if __name__ == '__main__':
    unittest.main()
