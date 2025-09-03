import inspect
import os.path as osp
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from matplotlib.font_manager import FontProperties

from pipeai import Config
from pipeai.dist import master_only
from pipeai.registry import VISBACKENDS, VISUALIZERS
from pipeai.utils import is_seq_of
from pipeai.visualization import (BaseVisBackend, img_from_canvas, value2list,
                                  tensor2ndarray, wait_continue,color_val_matplotlib)

# Define a type hint for the visualizer backends configuration
VisBackendsType = Union[List[Union[List]], dict, None]


@VISUALIZERS.register_module()
class Visualizer:
    """A Visualizer for managing visualization backends and providing a
    unified drawing interface."""

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: VisBackendsType = None,
                 save_dir: Optional[str] = None,
                 fig_save_cfg: Optional[dict] = None,
                 fig_show_cfg: Optional[dict] = None) -> None:
        super().__init__()

        # === Configurations ===
        self.name = name
        self.fig_save_cfg = fig_save_cfg or {'frameon': False}
        self.fig_show_cfg = fig_show_cfg or {'frameon': False}

        # === Dataset Meta ===
        self._dataset_meta: Optional[dict] = None

        # === Visualization Backends ===
        if save_dir is not None:
            save_dir = osp.join(save_dir, 'vis_data')
        self._vis_backends: Dict[str, BaseVisBackend] = {}
        self._build_vis_backends(vis_backends, save_dir)

        # === Image and Canvas States ===
        self._image: Optional[np.ndarray] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._default_font_size: Optional[int] = None

        # === Matplotlib Figure Manager ===
        self.manager = None
        self.fig_save_canvas, self.fig_save, self.ax_save = (
            self._init_fig(self.fig_save_cfg))
        self.dpi = self.fig_save.get_dpi()

        if image is not None:
            self.set_image(image)

    @property
    @master_only
    def dataset_meta(self) -> Optional[dict]:
        """Get the meta information of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter  # type: ignore
    @master_only
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the meta information of the dataset."""
        self._dataset_meta = dataset_meta

    @master_only
    def show(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key: str = ' ',
             backend: str = 'matplotlib'):
        """Show the drawn image with a specified backend."""
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            is_inline = 'inline' in plt.get_backend()
            img = self.get_image() if drawn_img is None else drawn_img

            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            if is_inline:
                return fig
            return wait_continue(
                fig, timeout=wait_time, continue_key=continue_key)
        else:
            raise ValueError(
                f'Unsupported backend "{backend}", only "matplotlib" is supported.'
            )

    @master_only
    def draw_points(self,
                    positions: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                    marker: Optional[str] = None,
                    sizes: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """Draw single or multiple points on the image."""
        positions = tensor2ndarray(positions)
        if positions.ndim == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, \
            f'The shape of `positions` should be (N, 2), but got {positions.shape}'

        colors = color_val_matplotlib(colors)  # type: ignore
        self.ax_save.scatter(
            positions[:, 0], positions[:, 1], c=colors, s=sizes, marker=marker)
        return self

    @master_only
    def draw_texts(
            self,
            texts: Union[str, List[str]],
            positions: Union[np.ndarray, torch.Tensor],
            font_sizes: Optional[Union[int, List[int]]] = None,
            colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            vertical_alignments: Union[str, List[str]] = 'top',
            horizontal_alignments: Union[str, List[str]] = 'left',
            font_families: Union[str, List[str]] = 'sans-serif',
            bboxes: Optional[Union[dict, List[dict]]] = None,
            font_properties: Optional[Union['FontProperties',
            List['FontProperties']]] = None
    ) -> 'Visualizer':
        """Draw single or multiple text boxes on the image."""
        if isinstance(texts, str):
            texts = [texts]
        num_texts = len(texts)

        positions = tensor2ndarray(positions)
        if positions.ndim == 1:
            positions = positions[None]
        assert positions.shape == (num_texts, 2), \
            f'`positions` should have shape ({num_texts}, 2), but got {positions.shape}'

        if not self._is_position_valid(positions):
            warnings.warn('Warning: The text is out of bounds, '
                          'it may not be visible in the image.', UserWarning)

        # Normalize all style parameters to lists
        font_sizes = value2list(font_sizes or self._default_font_size,
                                (int, float), num_texts)
        colors = color_val_matplotlib(value2list(colors, (str, tuple), num_texts))  # type: ignore
        vertical_alignments = value2list(vertical_alignments, str, num_texts)
        horizontal_alignments = value2list(horizontal_alignments, str,
                                           num_texts)
        font_families = value2list(font_families, str, num_texts)
        bboxes = value2list(bboxes, dict, num_texts)
        font_properties = value2list(font_properties, FontProperties,
                                     num_texts)

        for i in range(num_texts):
            self.ax_save.text(
                positions[i, 0],
                positions[i, 1],
                texts[i],
                size=font_sizes[i],
                bbox=bboxes[i],
                verticalalignment=vertical_alignments[i],
                horizontalalignment=horizontal_alignments[i],
                family=font_families[i],
                fontproperties=font_properties[i],
                color=colors[i])
        return self

    @master_only
    def draw_bboxes(
            self,
            bboxes: Union[np.ndarray, torch.Tensor],
            edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[Union[int, float],
            List[Union[int, float]]] = 2,
            face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
            alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bounding boxes."""
        bboxes = tensor2ndarray(bboxes)
        if bboxes.ndim == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, \
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}'
        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and \
               (bboxes[:, 1] <= bboxes[:, 3]).all()

        if not self._is_position_valid(bboxes.reshape(-1, 2)):
            warnings.warn('Warning: The bbox is out of bounds, '
                          'it may not be visible in the image.', UserWarning)

        # Convert bboxes to polygons
        polygons = np.stack([
            bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
            bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]
        ], axis=-1).reshape(-1, 4, 2)

        return self.draw_polygons(
            polygons.tolist(),
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

    @master_only
    def draw_lines(
            self,
            x_datas: Union[np.ndarray, torch.Tensor],
            y_datas: Union[np.ndarray, torch.Tensor],
            colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[Union[int, float],
            List[Union[int, float]]] = 2
    ) -> 'Visualizer':
        """Draw single or multiple line segments."""
        from matplotlib.collections import LineCollection

        x_datas = tensor2ndarray(x_datas)
        y_datas = tensor2ndarray(y_datas)
        assert x_datas.shape == y_datas.shape, \
            '`x_datas` and `y_datas` must have the same shape'
        if x_datas.ndim == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]
        assert x_datas.shape[-1] == 2, \
            f'The shape of `x_datas` should be (N, 2), but got {x_datas.shape}'

        lines = np.stack([x_datas, y_datas], axis=2)
        if not self._is_position_valid(lines.reshape(-1, 2)):
            warnings.warn('Warning: The line is out of bounds, '
                          'it may not be visible in the image.', UserWarning)

        line_collect = LineCollection(
            lines,
            colors=color_val_matplotlib(colors),  # type: ignore
            linestyles=line_styles,
            linewidths=line_widths)
        self.ax_save.add_collection(line_collect)
        return self

    @master_only
    def draw_circles(
            self,
            center: Union[np.ndarray, torch.Tensor],
            radius: Union[np.ndarray, torch.Tensor],
            edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[Union[int, float],
            List[Union[int, float]]] = 2,
            face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
            alpha: Union[float, int] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple circles."""
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle

        center = tensor2ndarray(center)
        radius = tensor2ndarray(radius)
        if center.ndim == 1:
            center = center[None]
            radius = np.array([radius])
        assert center.shape == (radius.shape[0], 2), \
            'The shape of `center` must be (N, 2) where N is the length of `radius`'

        # Check bounds
        bounds_min = center - radius[:, None]
        bounds_max = center + radius[:, None]
        if not (self._is_position_valid(bounds_min) and self._is_position_valid(bounds_max)):
            warnings.warn('Warning: The circle is out of bounds, '
                          'it may not be visible in the image.', UserWarning)

        circles = [Circle(c, r) for c, r in zip(center, radius)]

        line_widths = value2list(line_widths, (int, float), len(circles))
        capped_widths = [
            min(max(w, 1), self._default_font_size / 4) for w in line_widths
        ]

        patch_collection = PatchCollection(
            circles,
            alpha=alpha,
            facecolors=color_val_matplotlib(face_colors),  # type: ignore
            edgecolors=color_val_matplotlib(edge_colors),  # type: ignore
            linewidths=capped_widths,
            linestyles=line_styles)
        self.ax_save.add_collection(patch_collection)
        return self

    @master_only
    def draw_polygons(
            self,
            polygons: Union[Union[np.ndarray, torch.Tensor],
            List[Union[np.ndarray, torch.Tensor]]],
            edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
            face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
            alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple polygons."""
        from matplotlib.collections import PolyCollection

        if isinstance(polygons, (np.ndarray, torch.Tensor)):
            polygons = [tensor2ndarray(polygons)]
        else:
            polygons = [tensor2ndarray(p) for p in polygons]

        for poly in polygons:
            assert poly.ndim == 2 and poly.shape[1] == 2, \
                f'Each polygon must have shape (M, 2), but got {poly.shape}'
            if not self._is_position_valid(poly):
                warnings.warn('Warning: The polygon is out of bounds, '
                              'it may not be visible in the image.', UserWarning)

        line_widths = value2list(line_widths, (int, float), len(polygons))
        capped_widths = [
            min(max(w, 1), self._default_font_size / 4) for w in line_widths
        ]

        poly_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=color_val_matplotlib(face_colors),  # type: ignore
            linestyles=line_styles,
            edgecolors=color_val_matplotlib(edge_colors),  # type: ignore
            linewidths=capped_widths)
        self.ax_save.add_collection(poly_collection)
        return self

    @master_only
    def get_image(self) -> np.ndarray:
        """Get the drawn image as an RGB numpy array."""
        assert self._image is not None, 'Please set an image using `set_image` first.'
        return img_from_canvas(self.fig_save_canvas)

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw on."""
        assert image is not None
        self._image = image.astype('uint8')
        self._height, self._width = image.shape[:2]
        self._default_font_size = max(
            np.sqrt(self._height * self._width) // 90, 10)

        # Update figure size to match image resolution
        self.fig_save.set_size_inches(
            (self._width + 1e-2) / self.dpi,
            (self._height + 1e-2) / self.dpi)

        # Reset canvas and draw the new image
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            self._image,
            extent=(0, self._width, self._height, 0),
            interpolation='none')

    @master_only
    def get_backend(self, backend_name: str) -> BaseVisBackend:
        """Get a specific visualization backend instance by name."""
        if backend_name not in self._vis_backends:
            raise ValueError(f'Unknown backend: {backend_name}')
        return self._vis_backends[backend_name]

    @master_only
    def add_config(self, config: Config, **kwargs):
        """Record a configuration object to all backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.add_config(config, **kwargs)

    @master_only
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record a model graph to all backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.add_graph(model, data_batch, **kwargs)

    @master_only
    def add_image(self, name: str, image: np.ndarray, step: int = 0) -> None:
        """Record an image to all backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, step)

    @master_only
    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record a single scalar value to all backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalar(name, value, step, **kwargs)

    @master_only
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record a dictionary of scalars to all backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalars(scalar_dict, step, file_path, **kwargs)

    def close(self) -> None:
        """Close all visualization backends."""
        for vis_backend in self._vis_backends.values():
            vis_backend.close()

    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplotlib figure manager for window display."""
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager

        if self.manager is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
        try:
            self.manager.set_window_title(win_name)
        except Exception:
            # Fallback for some matplotlib backends that fail on title set
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
            self.manager.set_window_title(win_name)

    @staticmethod
    def _init_fig(fig_cfg: dict) -> tuple:
        """Initialize a matplotlib figure, canvas, and axes for saving."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(**fig_cfg)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.axis(False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return canvas, fig, ax

    def _build_vis_backends(self, vis_backends: VisBackendsType,
                            save_dir: Optional[str]) -> None:
        """Build and register all specified visualization backends."""
        if vis_backends is None:
            return
        if not isinstance(vis_backends, list):
            vis_backends = [vis_backends]
        if not is_seq_of(vis_backends, (dict, BaseVisBackend)):
            raise TypeError(
                'vis_backends must be a list of dicts or BaseVisBackend instances'
            )

        for vis_backend in vis_backends:
            if isinstance(vis_backend, dict):
                name = vis_backend.pop('name', None)
                vis_backend.setdefault('save_dir', save_dir)
                instance = VISBACKENDS.build(vis_backend)
            else:
                name = None
                instance = vis_backend

            # Use introspection to check if a backend requires `save_dir`
            # but was not initialized with one.
            sig = inspect.signature(instance.__class__.__init__)
            save_dir_arg = sig.parameters.get('save_dir')
            has_no_save_dir = getattr(instance, '_save_dir', None) is None
            if (save_dir_arg is not None and
                    save_dir_arg.default is inspect.Parameter.empty
                    and has_no_save_dir):
                warnings.warn(
                    f'Skipping {instance.__class__.__name__}: missing `save_dir`.'
                )
                continue

            # Use class name as default name if not provided
            backend_name = name or instance.__class__.__name__
            if backend_name in self._vis_backends:
                raise RuntimeError(
                    f'vis_backend name "{backend_name}" already exists.')
            self._vis_backends[backend_name] = instance

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if coordinates are within the image boundaries."""
        return (position.min() >= 0) and \
            (position[..., 0] < self._width).all() and \
            (position[..., 1] < self._height).all()
