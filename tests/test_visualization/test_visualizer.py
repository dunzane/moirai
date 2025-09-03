# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
from matplotlib import font_manager

from pipeai.visualization import Visualizer
from pipeai.visualization import LocalVisBackend, TensorboardVisBackend, SwanlabBackend


@patch('pipeai.dist.master_only', new=lambda f: f)
class TestVisualizer(unittest.TestCase):
    """Unit tests for the Visualizer class."""

    def setUp(self):
        """Set up mocks for matplotlib and a basic visualizer."""
        self.mock_canvas_cls = MagicMock()
        self.mock_figure_cls = MagicMock()
        self.mock_ax = MagicMock()
        self.mock_fig = MagicMock()
        self.mock_fig.add_subplot.return_value = self.mock_ax
        self.mock_figure_cls.return_value = self.mock_fig

        # Patch dynamic imports
        self.mpl_patcher = patch.dict('sys.modules', {
            'matplotlib.figure': MagicMock(Figure=self.mock_figure_cls),
            'matplotlib.backends.backend_agg': MagicMock(FigureCanvasAgg=self.mock_canvas_cls)
        })
        self.mpl_patcher.start()

        # Dummy image for tests
        self.test_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        # Create visualizer instance for use in tests
        self.visualizer = Visualizer()
        self.visualizer.ax_save = self.mock_ax

    def tearDown(self):
        """Stop patchers."""
        self.mpl_patcher.stop()

    @patch('pipeai.registry.VISBACKENDS.build')
    def test_initialization_builds_backends(self, mock_build):
        """Test building and registering visualizer backends."""
        def fake_build(cfg):
            if cfg['type'] == 'LocalVisBackend':
                return LocalVisBackend(save_dir='tmp')
            if cfg['type'] == 'TensorboardVisBackend':
                return TensorboardVisBackend(save_dir='tmp')
            if cfg['type'] == 'SwanlabBackend':
                return SwanlabBackend(save_dir='tmp')
            raise ValueError(f"Unknown backend type {cfg['type']}")

        mock_build.side_effect = fake_build

        cfgs = [
            {'type': 'LocalVisBackend'},
            {'type': 'TensorboardVisBackend', 'name': 'tb'},
            {'type': 'SwanlabBackend', 'name': 'swan'}
        ]
        visualizer = Visualizer(vis_backends=cfgs, save_dir='tmp')  # type: ignore

        self.assertEqual(mock_build.call_count, 3)
        self.assertIn('LocalVisBackend', visualizer._vis_backends)
        self.assertIn('tb', visualizer._vis_backends)
        self.assertIn('swan', visualizer._vis_backends)
        self.assertIsInstance(visualizer._vis_backends['LocalVisBackend'], LocalVisBackend)
        self.assertIsInstance(visualizer._vis_backends['tb'], TensorboardVisBackend)
        self.assertIsInstance(visualizer._vis_backends['swan'], SwanlabBackend)

    def test_set_image_updates_canvas(self):
        """Test set_image updates internal state and draws on canvas."""
        visualizer = self.visualizer
        visualizer.set_image(self.test_image)

        self.assertEqual(visualizer._width, 150)
        self.assertEqual(visualizer._height, 100)
        self.assertIsNotNone(visualizer._image)
        self.mock_ax.imshow.assert_called_once()
        np.testing.assert_array_equal(self.mock_ax.imshow.call_args[0][0], self.test_image)

    @patch('pipeai.visualization.visualizer.img_from_canvas')
    def test_get_image_retrieves_from_canvas(self, mock_img_from_canvas):
        """Test get_image calls canvas conversion utility."""
        mock_img_from_canvas.return_value = "drawn_image"
        visualizer = self.visualizer
        visualizer.set_image(self.test_image)

        result = visualizer.get_image()
        self.assertEqual(result, "drawn_image")
        mock_img_from_canvas.assert_called_once_with(visualizer.fig_save_canvas)

    @patch('pipeai.visualization.visualizer.color_val_matplotlib')
    def test_draw_texts_calls_ax_text(self, mock_color_val):
        """Test draw_texts calls ax.text with correct arguments."""
        mock_color_val.side_effect = lambda color: color

        visualizer = self.visualizer
        visualizer.set_image(self.test_image)

        texts = ["text1", "text2"]
        positions = np.array([[10, 20], [30, 40]])
        bboxes = [{"facecolor": "white", "alpha": 0.5}] * len(texts)
        font_properties = [font_manager.FontProperties()] * len(texts)

        visualizer.draw_texts(
            texts,
            positions,
            colors='red',
            font_properties=font_properties,
            bboxes=bboxes
        )

        self.assertEqual(self.mock_ax.text.call_count, 2)
        self.mock_ax.text.assert_any_call(
            10, 20, "text1",
            color='red', family=ANY, fontproperties=font_properties[0],
            horizontalalignment=ANY, size=ANY, verticalalignment=ANY,
            bbox=bboxes[0]
        )
        self.mock_ax.text.assert_any_call(
            30, 40, "text2",
            color='red', family=ANY, fontproperties=font_properties[1],
            horizontalalignment=ANY, size=ANY, verticalalignment=ANY,
            bbox=bboxes[1]
        )

    @patch.object(Visualizer, 'draw_polygons')
    def test_draw_bboxes_calls_draw_polygons(self, mock_draw_polygons):
        """Test draw_bboxes converts boxes to polygons and calls draw_polygons."""
        visualizer = Visualizer()
        visualizer.set_image(self.test_image)

        bboxes = np.array([[10, 10, 50, 50]])  # x1, y1, x2, y2
        visualizer.draw_bboxes(bboxes, edge_colors='blue')

        mock_draw_polygons.assert_called_once()
        passed_polygons = mock_draw_polygons.call_args[0][0]
        expected_polygons = [[[10, 10], [50, 10], [50, 50], [10, 50]]]
        np.testing.assert_array_equal(passed_polygons, expected_polygons)
        self.assertEqual(mock_draw_polygons.call_args[1].get('edge_colors'), 'blue')

    @patch('pipeai.registry.VISBACKENDS.build')
    def test_add_scalar_dispatches_to_all_backends(self, mock_build):
        """Test add_scalar calls all registered backends."""
        local_backend = LocalVisBackend(save_dir='tmp')
        tb_backend = TensorboardVisBackend(save_dir='tmp')
        local_backend.add_scalar = MagicMock()
        tb_backend.add_scalar = MagicMock()

        def fake_build(cfg):
            if cfg['type'] == 'LocalVisBackend':
                return local_backend
            if cfg['type'] == 'TensorboardVisBackend':
                return tb_backend
            raise ValueError(f"Unknown backend type {cfg['type']}")

        mock_build.side_effect = fake_build

        cfgs = [
            {'type': 'LocalVisBackend'},
            {'type': 'TensorboardVisBackend', 'name': 'tb'}
        ]
        visualizer = Visualizer(vis_backends=cfgs, save_dir='tmp')  # type: ignore
        visualizer.add_scalar(name='loss', value=0.5, step=10)

        local_backend.add_scalar.assert_called_once_with('loss', 0.5, 10, **{})
        tb_backend.add_scalar.assert_called_once_with('loss', 0.5, 10, **{})


if __name__ == '__main__':
    unittest.main()
