import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np


class HistoryBuffer:
    """Unified storage format for different log types.

    ``HistoryBuffer`` records the history of log for further statistics.

    Examples:
        >>> history_buffer = HistoryBuffer()
        >>> # Update history_buffer.
        >>> history_buffer.update(1)
        >>> history_buffer.update(2)
        >>> history_buffer.min()
        1
        >>> history_buffer.max()
        2
        >>> history_buffer.mean()
        1.5
        >>> history_buffer.statistics('mean') # access method by string.
        1.5

    Args:
        log_history (Sequence): History logs. Defaults to [].
        count_history (Sequence): Counts of history logs. Defaults to [].
        max_length (int): The max length of history logs. Defaults to 1000000.
    """
    _statistics_methods: dict = dict()

    def __init__(self,
                 log_history: Sequence = [],
                 count_history: Sequence = [],
                 max_length: int = 1000000):

        assert len(log_history) == len(count_history), \
            'The lengths of log_history and count_history should be equal'
        if len(log_history) > max_length:
            warnings.warn(f'The length of history buffer({len(log_history)}) '
                          f'exceeds the max_length({max_length}), the first '
                          'few elements will be ignored.')
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)
        self.max_length = max_length

        self._set_default_statistics()

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        """Update the log history with a new value and count.

        If the length of the buffer exceeds ``self.max_length``, the oldest
        elements will be removed to keep the buffer size within the limit.

        Args:
            log_val (int or float): The value to log.
            count (int): The weight or occurrence counts for this value (default: 1).
        """
        if (not isinstance(log_val, (int, float))
                or not isinstance(count, (int, float))):
            raise TypeError(f'log_val must be int or float but got '
                            f'{type(log_val)}, count must be int but got '
                            f'{type(count)}')
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the ``_log_history`` and ``_count_history``."""
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        """Register custom statistics method to ``_statistics_methods``.

        The registered method can be called by ``history_buffer.statistics``
        with corresponding method name and arguments.

        Examples:
            >>> @HistoryBuffer.register_statistics
            >>> def weighted_mean(self, window_size, weight):
            >>>     assert len(weight) == window_size
            >>>     return (self._log_history[-window_size:] *
            >>>             np.array(weight)).sum() / \
            >>>             self._count_history[-window_size:]
            >>> log_buffer = HistoryBuffer([1, 2], [1, 1])
            >>> log_buffer.statistics('weighted_mean', 2, [2, 1])
            2

        Args:
            method (Callable): Custom statistics method.

        Returns:
            Callable: Original custom statistics method.
        """
        method_name = method.__name__
        assert method_name not in cls._statistics_methods, \
            'method_name cannot be registered twice!'
        cls._statistics_methods[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        """Access statistics method by name.

        Args:
            method_name (str): Name of method.

        Returns:
            Any: Depends on corresponding method.
        """
        if method_name not in self._statistics_methods:
            raise KeyError(f'{method_name} has not been registered in '
                           'HistoryBuffer._statistics_methods')
        method = self._statistics_methods[method_name]
        return method(self, *arg, **kwargs)

    def _set_default_statistics(self):
        self._statistics_methods.setdefault('min', HistoryBuffer.min)
        self._statistics_methods.setdefault('max', HistoryBuffer.max)
        self._statistics_methods.setdefault('current', HistoryBuffer.current)
        self._statistics_methods.setdefault('mean', HistoryBuffer.mean)

    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        """the minimum value of the latest ``window_size`` values in log histories."""
        if window_size is None:
            window_size = len(self._log_history)
        else:
            assert isinstance(window_size, int), \
                f'The type of window size should be int, but got {type(window_size)}'
            assert window_size > 0, "window_size must be positive"
            window_size = min(window_size, len(self._log_history))
        return self._log_history[-window_size:].min()

    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        """the maximum value of the latest ``window_size`` values in log histories."""
        if window_size is None:
            window_size = len(self._log_history)
        else:
            assert isinstance(window_size, int), \
                f'The type of window size should be int, but got {type(window_size)}'
            assert window_size > 0, "window_size must be positive"
            window_size = min(window_size, len(self._log_history))
        return self._log_history[-window_size:].max()

    def current(self) -> np.ndarray:
        """the recently updated values in log histories."""
        if len(self._log_history) == 0:
            raise ValueError('HistoryBuffer._log_history is an empty array! '
                             'please call update first')
        return self._log_history[-1]

    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        """the mean value of the latest ``window_size`` values in log histories."""
        if window_size is None:
            window_size = len(self._log_history)
        else:
            assert isinstance(window_size, int), \
                f'The type of window size should be int, but got {type(window_size)}'
            assert window_size > 0, "window_size must be positive"
            window_size = min(window_size, len(self._log_history))
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    def __getstate__(self) -> dict:
        """Make ``_statistics_methods`` can be resumed."""
        self.__dict__.update(statistics_methods=self._statistics_methods)
        return self.__dict__

    def __setstate__(self, state):
        """Try to load ``_statistics_methods`` from state."""
        statistics_methods = state.pop('statistics_methods', {})
        self._set_default_statistics()
        self._statistics_methods.update(statistics_methods)
        self.__dict__.update(state)
