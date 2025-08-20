import unittest
from unittest.mock import patch

import pipeai.device as pdevice


class TestDeviceAvailable(unittest.TestCase):
    """Unit tests for device availability functions"""

    def test_is_cuda_available(self):
        """Test is_cuda_available returns torch.cuda.is_available"""
        with patch("torch.cuda.is_available", return_value=True):
            self.assertTrue(pdevice.is_cuda_available())
        with patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(pdevice.is_cuda_available())

    def test_is_mps_available(self):
        class MockMPS:
            @staticmethod
            def is_available():
                return True

        class MockMPS2:
            @staticmethod
            def is_available():
                return False

        with patch("torch.backends.mps", new=MockMPS):
            self.assertTrue(pdevice.is_mps_available())
        with patch("torch.backends.mps", new=MockMPS2):
            self.assertFalse(pdevice.is_mps_available())


if __name__ == '__main__':
    unittest.main()
