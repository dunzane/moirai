# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest

import pytest
import torch

import pipeai.device as pdevice


class TestDeviceMemory(unittest.TestCase):
    """Unit tests for device memory functions"""

    def setUp(self) -> None:
        """Reset memory stats before each test"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch, "musa") and torch.musa.is_available():
            torch.musa.empty_cache = getattr(torch.musa, "empty_cache", lambda: None)
            torch.musa.empty_cache()

    def test_get_max_cuda_memory(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")

        # baseline
        baseline = pdevice.get_max_cuda_memory(device)
        assert baseline == 0

        # allocate a tensor (should consume memory)
        x = torch.randn(1024, 1024, device=device)  # ~4MB
        mem_used = pdevice.get_max_cuda_memory(device)
        assert mem_used > baseline

        # allocate more tensors
        y = torch.randn(4096, 4096, device=device)  # ~64MB
        mem_used2 = pdevice.get_max_cuda_memory(device)
        assert mem_used2 >= mem_used

        # cleanup
        del x, y
        torch.cuda.empty_cache()
        # after reset, should be zero again
        reset_val = pdevice.get_max_cuda_memory(device)
        assert reset_val == 0


if __name__ == '__main__':
    unittest.main()
