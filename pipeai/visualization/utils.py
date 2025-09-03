from typing import (TYPE_CHECKING, Any, List,
                    Optional, Tuple, Type, Union)

import numpy as np
import torch

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg


def check_type(name: str, value: Any,
               valid_type: Union[Type, Tuple[Type, ...]]) -> None:
    """Check whether the type of value is in ``valid_type``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_type (Type, Tuple[Type, ...]): expected type.
    """
    if not isinstance(value, valid_type):
        raise TypeError(f'`{name}` should be {valid_type} '
                        f' but got {type(value)}')


def tensor2ndarray(value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """If the type of value is torch.Tensor, convert the value to np.ndarray.

    Args:
        value (np.ndarray, torch.Tensor): value.

    Returns:
        Any: value.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return value


def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')


def wait_continue(figure, timeout: float = 0, continue_key: str = ' ') -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        figure (matplotlib.figure.Figure): The figure to wait for.
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent
    is_inline = 'inline' in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:

        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ('key_press_event', 'close_event')
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:  # type: ignore
            return 0  # Quit for continue.


def check_type_and_length(name: str, value: Any,
                          valid_type: Union[Type, Tuple[Type, ...]],
                          valid_length: int) -> None:
    """Check whether the type of value is in ``valid_type``. If type of the
    ``value`` is list, check whether its length is equal with or greater than
    ``valid_length``.

    Args:
        value (Any): value.
        legal_type (Type, Tuple[Type, ...]): legal type.
        valid_length (int): expected length.

    Returns:
        List[Any]: value.
    """
    check_type(name, value, valid_type)
    check_length(name, value, valid_length)


def check_length(name: str, value: Any, valid_length: int) -> None:
    """If type of the ``value`` is list, check whether its length is equal with
    or greater than ``valid_length``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_length (int): expected length.
    """
    if isinstance(value, list):
        if len(value) < valid_length:
            raise AssertionError(
                f'The length of {name} must equal with or '
                f'greater than {valid_length}, but got {len(value)}')


def value2list(value: Any, valid_type: Union[Type, Tuple[Type, ...]],
               expand_dim: int) -> List[Any]:
    """If the type of ``value`` is ``valid_type``, convert the value to list
    and expand to ``expand_dim``.

    Args:
        value (Any): value.
        valid_type (Union[Type, Tuple[Type, ...]): valid type.
        expand_dim (int): expand dim.

    Returns:
        List[Any]: value.
    """
    if isinstance(value, valid_type):
        value = [value] * expand_dim
    return value


def color_val_matplotlib(
    colors: Union[str, tuple, List[Union[str, tuple]]]
) -> Union[str, tuple, List[Union[str, tuple]]]:
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
    Returns:
        Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
        floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [
            color_val_matplotlib(color)  # type:ignore
            for color in colors
        ]
        return colors
    else:
        raise TypeError(f'Invalid type for color: {type(colors)}')
