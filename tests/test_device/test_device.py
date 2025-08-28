# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import logging
import os
import unittest
from unittest.mock import patch, Mock, MagicMock


import pipeai.device as pdevice


class TestDevice(unittest.TestCase):
    """Unit tests for device functions in pipeai.device module"""

    def setUp(self):
        """Reset DEVICE_TYPE before each test"""
        pdevice.set_device_type("gpu")  # Default state

    def tearDown(self):
        """Ensure DEVICE_TYPE is reset after each test"""
        pdevice.set_device_type("gpu")

    def test_get_device_type_default(self):
        self.assertEqual(pdevice.get_device_type(), "gpu")

    def test_set_device_type_valid(self):
        for dtype in ["gpu", "mps", "cpu"]:
            pdevice.set_device_type(dtype)
            self.assertEqual(pdevice.get_device_type(), dtype)

    def test_set_device_type_invalid(self):
        for invalid in ["tpu", "invalid", "xyz"]:
            with self.assertRaises(AssertionError):
                pdevice.set_device_type(invalid)

    @patch("torch.cuda.device_count", return_value=4)
    def test_get_device_count_gpu(self, mock_device_count):
        pdevice.set_device_type("gpu")
        self.assertEqual(pdevice.get_device_count(), 4)
        mock_device_count.return_value = 0
        self.assertEqual(pdevice.get_device_count(), 0)

    @patch("torch.backends-", new_callable=Mock)
    def test_get_device_count_mps(self, mock_backends):
        pdevice.set_device_type("mps")

        # MPS available
        mock_backends.mps = Mock()
        mock_backends.mps.is_available.return_value = True
        self.assertEqual(pdevice.get_device_count(), 1)

        # MPS unavailable
        mock_backends.mps.is_available.return_value = False
        self.assertEqual(pdevice.get_device_count(), 0)

        # MPS attribute missing
        del mock_backends.mps
        self.assertEqual(pdevice.get_device_count(), 0)

    def test_get_device_count_cpu(self):
        pdevice.set_device_type("cpu")
        self.assertEqual(pdevice.get_device_count(), 1)

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_set_device_gpu_valid(self, mock_device_count, mock_set_device):
        pdevice.set_device_type("gpu")
        pdevice.set_device(2)
        mock_set_device.assert_called_once_with(2)

    @patch("torch.cuda.device_count", return_value=4)
    def test_set_device_gpu_invalid(self, mock_device_count):
        pdevice.set_device_type("gpu")

        with self.assertRaises(ValueError) as ctx:
            pdevice.set_device(-1)  # negative device ID
        self.assertIn("Invalid GPU device_id", str(ctx.exception))

        # Exceeding max device ID
        mock_device_count.return_value = 2
        with self.assertRaises(ValueError) as ctx:
            pdevice.set_device(3)
        self.assertIn("Invalid GPU device_id", str(ctx.exception))

    def test_set_device_mps(self):
        pdevice.set_device_type("mps")
        pdevice.set_device(0)
        with self.assertRaises(ValueError) as ctx:
            pdevice.set_device(1)
        self.assertIn("only device_id=0 is supported", str(ctx.exception))

    def test_set_device_cpu(self):
        pdevice.set_device_type("cpu")
        pdevice.set_device(0)
        with self.assertRaises(ValueError) as ctx:
            pdevice.set_device(1)
        self.assertIn("only device_id=0 is supported", str(ctx.exception))

    def test_set_device_invalid_type(self):
        with self.assertRaises(TypeError) as ctx:
            pdevice.set_device("0")
        self.assertIn("device_id must be an integer", str(ctx.exception))


class TestSetVisibleDevices(unittest.TestCase):
    """Unit tests for set_visible_devices function"""

    def setUp(self):
        """Set default DEVICE_TYPE before each test"""
        self.original_device_type = pdevice.get_device_type()

    def tearDown(self):
        """Restore DEVICE_TYPE after each test"""
        pdevice.set_device_type(self.original_device_type)

    def test_set_visible_devices_gpu_with_devices(self):
        mock_logger = MagicMock(spec=logging.Logger)
        with patch.dict(os.environ, {}, clear=True):
            pdevice.set_visible_devices("0,1,2", mock_logger)

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2")
            mock_logger.info.assert_called_once_with("Set visible GPU devices to: 0,1,2")

    def test_set_visible_devices_gpu_no_devices(self):
        mock_logger = MagicMock(spec=logging.Logger)
        with patch.dict(os.environ, {}, clear=True):
            pdevice.set_visible_devices(None, mock_logger)
            mock_logger.info.assert_called_once_with("Using all available GPU devices")

    def test_set_visible_devices_mps_with_devices(self):
        mock_logger = MagicMock(spec=logging.Logger)
        pdevice.set_device_type("mps")
        pdevice.set_visible_devices("0", mock_logger)
        mock_logger.warning.assert_called_once_with(
            "Setting visible devices is not applicable for mps device type, ignoring"
        )

    def test_set_visible_devices_cpu_no_devices(self):
        mock_logger = MagicMock(spec=logging.Logger)
        pdevice.set_device_type("cpu")
        pdevice.set_visible_devices(None, mock_logger)
        mock_logger.info.assert_called_once_with("Using cpu device")

    def test_set_visible_devices_invalid_devices_type(self):
        mock_logger = MagicMock(spec=logging.Logger)
        with self.assertRaises(TypeError) as context:
            pdevice.set_visible_devices(123, mock_logger)
        self.assertIn("devices must be a string or None", str(context.exception))

    def test_set_visible_devices_invalid_logger_type(self):
        with self.assertRaises(TypeError) as context:
            pdevice.set_visible_devices("0", "not_a_logger")
        self.assertIn("logger must be logging.Logger", str(context.exception))


class TestGetDeviceInfo(unittest.TestCase):
    """Unit tests for get_device_info function"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        pdevice.DEVICE_TYPE = "gpu"

    def tearDown(self):
        """Clean up after each test method"""
        pdevice.set_device_type("gpu")

    def test_get_device_info_gpu(self):
        pdevice.DEVICE_TYPE = "gpu"
        with patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.current_device", return_value=0), \
                patch("torch.version.cuda", "11.8"):
            info = pdevice.get_device_info()

            expected = {
                "device_type": "gpu",
                "device_count": 2,
                "cuda_available": True,
                "cuda_version": "11.8",
                "current_device": 0
            }
            self.assertEqual(info, expected)

    def test_get_device_info_gpu_unavailable(self):
        pdevice.DEVICE_TYPE = "gpu"
        with patch("torch.cuda.device_count", return_value=0), \
                patch("torch.cuda.is_available", return_value=False):
            info = pdevice.get_device_info()

            expected = {
                "device_type": "gpu",
                "device_count": 0,
                "cuda_available": False,
                "cuda_version": None,
                "current_device": None
            }
            self.assertEqual(info, expected)

    def test_get_device_info_mps(self):
        pdevice.set_device_type("mps")
        with patch("torch.backends-", create=True) as mock_backends:
            mock_backends.mps.is_available.return_value = True
            info = pdevice.get_device_info()

            expected = {
                "device_type": "mps",
                "device_count": 1,
                "mps_available": True
            }
            self.assertEqual(info, expected)

    def test_get_device_info_cpu(self):
        pdevice.set_device_type("cpu")

        info = pdevice.get_device_info()

        expected = {
            "device_type": "cpu",
            "device_count": 1
        }
        self.assertEqual(info, expected)


if __name__ == '__main__':
    unittest.main()
