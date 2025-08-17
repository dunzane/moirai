import inspect
from typing import Union, Optional, Any, Dict

from pipeai import Config
from .registry import Registry


def build_from_cfg(cfg: Union[Config, dict],
                   registry: Registry,
                   default_args: Optional[Union[Config, dict]] = None) -> Any:
    from ..logging import get_logger
    logger = get_logger('pipeai-builder')

    if not isinstance(cfg, (Config, dict)):
        raise TypeError(f'cfg should be a dict or Config, got {type(cfg)}')
    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be a Registry object, got {type(registry)}')
    if default_args is not None and not isinstance(default_args, (Config, dict)):
        raise TypeError(f'default_args should be a dict, Config or None, got {type(default_args)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(f'`cfg` or `default_args` must contain the key "type", got {cfg}, {default_args}')

    _cfg = cfg.copy()
    if default_args:
        for k, v in default_args.items():
            _cfg.setdefault(k, v)  # merge default_args

    # switch scope if specified
    scope = _cfg.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as reg:
        obj_type = _cfg.pop('type')

        # resolve the class/function
        if isinstance(obj_type, str):
            obj_cls = reg.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {reg.scope}::{reg.name} registry. '
                    'Please check whether the value of `{obj_type}` is correct or '
                    'it was registered as expected.'
                )
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str or callable, got {type(obj_type)}')

    if inspect.isclass(obj_cls):
        obj = obj_cls(**_cfg)
    elif callable(obj_cls):
        obj = obj_cls(**_cfg)
    else:
        raise TypeError(f'Cannot construct object from {obj_cls}')

    logger.debug(
        f'An instance of `{getattr(obj_cls, "__name__", str(obj_cls))}` '
        f'is built from registry {reg.name} (module {getattr(obj_cls, "__module__", "")})'
    )

    return obj





