# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pipeai.visualization import (check_type, tensor2ndarray,
                                  img_from_canvas, check_length,
                                  value2list, wait_continue)


class TestDrawingUtils(unittest.TestCase):
    def test_check_type(self):
        """Test the check_type function for success and failure cases."""
        # Success case: should not raise an error
        check_type("my_list", [1, 2], list)
        check_type("my_string", "hello", (str, int))

        # Failure case: should raise TypeError
        with self.assertRaises(TypeError):
            check_type("my_int", "not_an_int", int)
        with self.assertRaisesRegex(TypeError, "`my_dict` should be <class 'list'>  but got <class 'dict'>"):
            check_type("my_dict", {}, list)

    def test_tensor2ndarray(self):
        """Test the conversion from torch.Tensor to np.ndarray."""
        import torch
        tensor = torch.tensor([[1, 2], [3, 4]])
        ndarray = tensor2ndarray(tensor)
        self.assertIsInstance(ndarray, np.ndarray)
        np.testing.assert_array_equal(ndarray, np.array([[1, 2], [3, 4]]))

        # Test if it handles ndarray input correctly
        original_ndarray = np.array([5, 6])
        self.assertIs(tensor2ndarray(original_ndarray), original_ndarray)

    def test_img_from_canvas(self):
        """Test creating an RGB image from a mock matplotlib canvas."""
        # Create a mock canvas
        mock_canvas = MagicMock()
        height, width = 10, 20
        # Create a dummy RGBA buffer (H, W, 4 channels)
        dummy_rgba_buffer = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        # Configure the mock to return the buffer and its dimensions
        mock_canvas.print_to_buffer.return_value = (dummy_rgba_buffer.tobytes(), (width, height))

        rgb_image = img_from_canvas(mock_canvas)

        self.assertEqual(rgb_image.shape, (height, width, 3))
        self.assertEqual(rgb_image.dtype, np.uint8)
        # Check if the RGB data matches the original buffer's RGB part
        np.testing.assert_array_equal(rgb_image, dummy_rgba_buffer[..., :3])

    def test_check_length(self):
        """Test the check_length function."""
        # Success cases
        check_length("my_list", [1, 2, 3], 3)  # equal
        check_length("my_list", [1, 2, 3, 4], 3)  # greater

        # Failure case
        with self.assertRaises(AssertionError):
            check_length("my_list", [1, 2], 3)

        # Non-list case (should do nothing)
        check_length("my_string", "abc", 5)

    def test_value2list(self):
        """Test the value2list conversion utility."""
        # Case: Convert a valid type
        result = value2list(5, int, 3)
        self.assertEqual(result, [5, 5, 5])

        # Case: Input is already a list (should return unchanged)
        original_list = [1, 2]
        result = value2list(original_list, int, 3)
        self.assertIs(result, original_list)

        # Case: Input is not a valid type (should return unchanged)
        result = value2list("hello", int, 3)
        self.assertEqual(result, "hello")

    def test_wait_continue_inline_backend(self):
        """Test wait_continue's behavior with a non-interactive inline backend."""
        # Create a mock for the entire matplotlib.pyplot module
        mock_plt = MagicMock()
        mock_plt.get_backend.return_value = 'module://ipykernel.pylab.backend_inline'

        # Use patch.dict to temporarily insert the mock into sys.modules.
        # This will intercept the `import matplotlib.pyplot as plt` inside wait_continue.
        with patch.dict('sys.modules', {'matplotlib.pyplot': mock_plt}):
            mock_figure = MagicMock()
            # In an inline backend, it should return 0 immediately
            result = wait_continue(mock_figure)
            self.assertEqual(result, 0)
            mock_figure.canvas.start_event_loop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
