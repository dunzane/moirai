import copy
import numpy as np
import torch
from typing import OrderedDict, Union

from collections import OrderedDict
from typing import Any, Optional, Mapping
from .history_buffer import HistoryBuffer
from .logger import get_logger


class MessageHub:
    """Message hub for components in a pipeline interaction.

    ``MessageHub`` records log information and runtime information.
    - Log information refers to model metrics during training (e.g., loss, learning rate),
      stored as ``HistoryBuffer``.
    - Runtime information refers to iteration counts, meta info, etc., which
      are overwritten on each update.

    Examples:
        >>> from pipeai.logging import HistoryBuffer
        >>> # Create empty MessageHub
        >>> message_hub1 = MessageHub('name')
        >>> # Create MessageHub from existing data
        >>> log_scalars = dict(loss=HistoryBuffer())
        >>> runtime_info = dict(task='task')
        >>> resumed_keys = dict(loss=True)
        >>> message_hub2 = MessageHub(
        >>>     name='name',
        >>>     log_scalars=log_scalars,
        >>>     runtime_info=runtime_info,
        >>>     resumed_keys=resumed_keys
        >>> )

    Args:
        name (str): Name of the MessageHub.
        log_scalars (dict, optional): Mapping from log names to HistoryBuffer instances.
        runtime_info (dict, optional): Mapping from runtime keys to their values.
        resumed_keys (dict, optional): Mapping from keys to bool, indicating
            whether the key should be serialized.
    """

    def __init__(
            self,
            name: str,
            log_scalars: Optional[Mapping[str, Any]] = None,  # 保存训练过程中产生的标量型日志信息（scalars）
            runtime_info: Optional[Mapping[str, Any]] = None,
            resumed_keys: Optional[Mapping[str, bool]] = None  # 控制哪些日志或运行时信息需要在 训练中断后恢复（resume）
    ):
        self.name = name
        self._log_scalars: OrderedDict[str, HistoryBuffer] = self._parse_input('log_scalars', log_scalars)
        self._runtime_info: OrderedDict[str, Any] = self._parse_input('runtime_info', runtime_info)
        self._resumed_keys: OrderedDict[str, bool] = self._parse_input('resumed_keys', resumed_keys)

        for value in self._log_scalars.values():
            if not isinstance(value, HistoryBuffer):
                raise TypeError(f"All values in log_scalars must be HistoryBuffer, got {type(value)}")

        for key in self._resumed_keys.keys():
            if key not in self._log_scalars and key not in self._runtime_info:
                raise KeyError(f"Key '{key}' in resumed_keys must exist in log_scalars or runtime_info")

        self.logger = get_logger("pipeai-message_hub")

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        return self._runtime_info

    def update_scalar(self,
                      key: str,
                      value: Union[int, float, np.ndarray, 'torch.Tensor'],
                      count: int = 1,
                      resumed: bool = True) -> None:
        """Update :attr:_log_scalars.

        Update ``HistoryBuffer`` in :attr:`_log_scalars`. If corresponding key
        ``HistoryBuffer`` has been created, ``value`` and ``count`` is the
        argument of ``HistoryBuffer.update``, Otherwise, ``update_scalar``
        will create an ``HistoryBuffer`` with value and count via the
        constructor of ``HistoryBuffer``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> # create loss `HistoryBuffer` with value=1, count=1
            >>> message_hub.update_scalar('loss', 1)
            >>> # update loss `HistoryBuffer` with value
            >>> message_hub.update_scalar('loss', 3)
            >>> message_hub.update_scalar('loss', 3, resumed=False)
            AssertionError: loss used to be true, but got false now. resumed
            keys cannot be modified repeatedly

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Args:
            key (str): Key of ``HistoryBuffer``.
            value (torch.Tensor or np.ndarray or int or float): Value of log.
            count (torch.Tensor or np.ndarray or int or float): Accumulation
                times of log, defaults to 1. `count` will be used in smooth
                statistics.
            resumed (str): Whether the corresponding ``HistoryBuffer`` could
                be resumed. Defaults to True.
        """
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(count, int), (
            f'The type of count must be int. but got {type(count): {count}}')
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``.If a type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``.Item in
        ``log_dict`` has the same resume option.

        Examples:
            >>> message_hub = MessageHub('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``log_dict``.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = self._get_valid_value(log_val.get('count', 1))
                value = log_val['value']
            else:
                count = 1
                value = log_val
            assert isinstance(count,
                              int), ('The type of count must be int. but got '
                                     f'{type(count): {count}}')
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def update_infos(self, info_dict: dict, resumed: bool = True) -> None:
        """Update runtime information with dictionary.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``info_dict``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info({'iter': 100})

        Args:
            info_dict (str): Runtime information dictionary.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        assert isinstance(info_dict, dict), ('`log_dict` must be a dict!, '
                                             f'but got {type(info_dict)}')
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def pop_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Remove runtime information by key. If the key does not exist, this
        method will return the default value.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: The runtime information if the key exists.
        """
        return self._runtime_info.pop(key, default)

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self._log_scalars:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.__class__.__name__}')

        return self.log_scalars[key]

    def get_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Get runtime information by key.If the key does not exist, this
        method will return default information.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            return default
        else:
            return self._runtime_info[key]

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalars` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, \
                f'{key} used to be {self._resumed_keys[key]}, but got ' \
                '{resumed} now. resumed keys cannot be modified repeatedly.'

    def state_dict(self) -> dict:
        """Returns a dictionary containing log scalars, runtime information and
        resumed keys, which should be resumed.

        The returned ``state_dict`` can be loaded by :meth:`load_state_dict`.

        Returns:
            dict: A dictionary contains ``log_scalars``, ``runtime_info`` and
            ``resumed_keys``.
        """
        saved_scalars = OrderedDict()  # type: ignore
        saved_info = OrderedDict()  # type: ignore

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys)

    def load_state_dict(self, state_dict: Union['MessageHub', dict]) -> None:
        """Loads log scalars, runtime information and resumed keys from
        ``state_dict`` or ``message_hub``.

        If ``state_dict`` is a dictionary returned by :meth:`state_dict`, it
        will only make copies of data which should be resumed from the source
        ``message_hub``.

        If ``state_dict`` is a ``message_hub`` instance, it will make copies of
        all data from the source message_hub. We suggest to load data from
        ``dict`` rather than a ``MessageHub`` instance.

        Args:
            state_dict (dict or MessageHub): A dictionary contains key
                ``log_scalars`` ``runtime_info`` and ``resumed_keys``, or a
                MessageHub instance.
        """
        if isinstance(state_dict, dict):
            for key in ('log_scalars', 'runtime_info', 'resumed_keys'):
                assert key in state_dict, (
                    'The loaded `state_dict` of `MessageHub` must contain '
                    f'key: `{key}`')

            # The old `MessageHub` could save non-HistoryBuffer `log_scalars`,
            # therefore, the loaded `log_scalars` needs to be filtered.
            for key, value in state_dict['log_scalars'].items():
                if not isinstance(value, HistoryBuffer):
                    self.logger.warning(f'{key} in message_hub is not HistoryBuffer, '
                                        f'just skip resuming it.')
                    continue
                self._log_scalars[key] = value

            for key, value in state_dict['runtime_info'].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    self.logger.warning(f'{key} in message_hub cannot be copied, '
                                        f'just return its reference.', )
                    self._runtime_info[key] = value

            for key, value in state_dict['resumed_keys'].items():
                if key not in set(self._log_scalars.keys()) | \
                        set(self._runtime_info.keys()):
                    self.logger.warning(f'resumed key: {key} is not defined in message_hub, '
                                        f'just skip resuming this key.')
                    continue
                elif not value:
                    self.logger.warning(f'Although resumed key: {key} is False, {key} '
                                        'will still be loaded this time. This key will '
                                        'not be saved by the next calling of '
                                        '`MessageHub.state_dict()`')
                self._resumed_keys[key] = value

        # Since some checkpoints saved serialized `message_hub` instance,
        # `load_state_dict` support loading `message_hub` instance for
        # compatibility
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)

    @staticmethod
    def _parse_input(name: str, value: Any) -> OrderedDict[str, Any]:
        """Parse input value into an OrderedDict.

        Args:
            name (str): Name of the input (for error messages).
            value (Any): Input value, should be dict-like or None.

        Returns:
            OrderedDict[str, Any]: Parsed OrderedDict.
        """
        if value is None:
            return OrderedDict()  # type: ignore
        if isinstance(value, Mapping):
            return OrderedDict(value)  # type: ignore
        raise TypeError(f"{name} should be a dict-like object or None, got {type(value)}")

    @staticmethod
    def _get_valid_value(value: Union['torch.Tensor', np.ndarray, np.number, int, float]) \
            -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            value (torch.Tensor or np.ndarray or np.number or int or float):
                value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            # check whether value is torch.Tensor but don't want
            # to import torch in this file
            assert hasattr(value, 'numel') and value.numel() == 1
            value = value.item()
        return value  # type: ignore
