# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import MagicMock, call
from parameterized import parameterized

from pipeai.hooks import NaiveVisualizationHook


class TestNaiveVisualizationHook(unittest.TestCase):
    """Unit tests for the NaiveVisualizationHook."""

    def setUp(self):
        """Set up a mock runner for each test."""
        self.runner = MagicMock()
        self.runner.visualizer = MagicMock()
        self.runner.model = MagicMock()

    def test_before_train_adds_graph(self):
        """Test if `add_graph` is called on the visualizer before training."""
        hook = NaiveVisualizationHook()
        hook.before_train(self.runner)

        self.runner.visualizer.add_graph.assert_called_once_with(
            self.runner.model, None
        )

    @parameterized.expand([
        # interval=3, batch_idx=2 (3rd iter) → trigger visualization
        (3, 2, True),
        # interval=3, batch_idx=3 (4th iter) → NOT trigger
        (3, 3, False),
        # interval=1, batch_idx=0 → always trigger
        (1, 0, True),
    ])
    def test_after_test_iter_visualizes_on_interval(
        self, interval, batch_idx, should_visualize
    ):
        """Test if `add_datasample` is called correctly based on the interval."""
        hook = NaiveVisualizationHook(interval=interval, draw_gt=True, draw_pred=True)

        mock_data_batch = [{"data_sample": "img1"}, {"data_sample": "img2"}]
        mock_outputs = [{"prediction": "pred1"}, {"prediction": "pred2"}]

        hook.after_test_iter(
            self.runner,
            batch_idx=batch_idx,
            data_batch=mock_data_batch,
            outputs=mock_outputs,
        )

        if should_visualize:
            self.assertEqual(self.runner.visualizer.add_datasample.call_count, len(mock_data_batch))

            expected_name = f"sample_{batch_idx}"
            expected_calls = [
                call(expected_name, mock_data_batch[i], mock_data_batch[i],
                     mock_outputs[i], True, True)
                for i in range(len(mock_data_batch))
            ]

            self.runner.visualizer.add_datasample.assert_has_calls(expected_calls)
        else:
            self.runner.visualizer.add_datasample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
