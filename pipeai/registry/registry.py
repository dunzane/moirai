import inspect
import sys
from rich.table import Table
from rich.console import Console
from contextlib import contextmanager
from importlib import import_module
from typing import Optional, Callable, List, Dict, Type, Generator, Any, Tuple, Union

from pipeai.config import MODULE_2_PACKAGE
from pipeai.utils import is_seq_of
from .default_scope import SimpleScopeManager
# from .build_functions import build_from_cfg


class Registry:
    """A simple registry for managing classes and functions."""

    def __init__(self,
                 name: str,
                 build_func: Optional[Callable] = None,
                 scope: Optional[str] = None,
                 locations: Optional[List[str]] = None) -> None:
        from pipeai.logging import get_logger
        from .build_functions import build_from_cfg
        self.logger = get_logger(f'pipeai-{__class__.__name__}')

        self._name = name
        self._module_dict: Dict[str, Type] = {}
        self._children: Dict[str, 'Registry'] = {}
        self._locations = locations or []
        self._imported = False
        self._scope = scope if scope is not None else self._infer_scope()
        self._build_func: Callable = build_func or build_from_cfg

    def get(self, key: str) -> Optional[Type]:
        """Retrieve a registered object.

        - If ``key`` exists in the current registry, return it.
        - Otherwise, try to resolve ``key`` as a full import path string.

        Args:
            key (str): The name of the registered item or a fully qualified path.

        Returns:
            Type or None: The corresponding object if found, otherwise None.
        """
        import importlib

        if not isinstance(key, str):
            raise TypeError(
                f'The key argument of `Registry.get` must be a str, got {type(key)}'
            )

        self._import_from_location()
        obj_cls = self._module_dict.get(key)

        if obj_cls is None:
            try:
                module_name, cls_name = key.rsplit(".", 1)
                module = importlib.import_module(module_name)
                obj_cls = getattr(module, cls_name)
            except Exception as e:
                self.logger.debug(
                    f'Failed to import "{key}" when resolving in registry "{self.name}": {e}'
                )
                return None

        cls_name = getattr(obj_cls, '__name__', str(obj_cls))
        self.logger.debug(f'Get class `{cls_name}` from "{self.name}" registry')
        return obj_cls

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance from a configuration dict.

        Args:
            cfg (dict): The configuration dictionary.
            *args: Positional arguments for the build function.
            **kwargs: Keyword arguments for the build function.

        Returns:
            Any: The constructed object.
        """
        return self._build_func(cfg, *args, **kwargs, registry=self)

    def register_module(self,
                        name: Optional[Union[str, List[str]]] = None,
                        force: bool = False,
                        module: Optional[Type] = None) -> Union[type, Callable[[Type], Type]]:
        """Register a module.

        Can be used either as a decorator or a normal method call.

        Args:
            name (str or list of str, optional): The module name(s) to register.
                Defaults to the class/function name if not provided.
            force (bool): Whether to override an existing registration. Defaults to False.
            module (type, optional): The module class/function to register.
                Defaults to None (for decorator usage).

        Returns:
            If used as a decorator, returns a callable that registers the module.
            If used as a direct method call, returns the registered module itself.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, str, or sequence of str, '
                f'but got {type(name)}'
            )

        # Direct registration
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def decorator(cls_or_func: Type) -> Type:
            self._register_module(module=cls_or_func, module_name=name, force=force)
            return cls_or_func

        return decorator

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        """Split scope and key like 'scope.Resnet'."""
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        return None, key

    @contextmanager
    def switch_scope_and_registry(self, scope: Optional[str]) \
            -> Generator['Registry', None, None]:
        """Temporarily switch scope and return this registry.

        Args:
            scope (str, optional): The scope name to switch to.

        Yields:
            Registry: Always yields the current registry instance.
        """
        old_scope = SimpleScopeManager.get_scope()
        SimpleScopeManager.set_scope(scope)

        try:
            yield self
        finally:
            # restore old scope
            SimpleScopeManager.set_scope(old_scope)

    @property
    def name(self) -> str:
        """Return the registry name."""
        return self._name

    @property
    def scope(self) -> str:
        """Return the registry scope."""
        return self._scope

    @property
    def module_dict(self) -> Dict[str, Type]:
        """Return the module dictionary."""
        return self._module_dict

    @property
    def build_func(self) -> Callable:
        return self._build_func

    def _infer_scope(self) -> str:
        """Infer scope from the caller's module name."""
        module = inspect.getmodule(sys._getframe(2))
        if module and module.__name__:
            return module.__name__.split('.')[0]

        default_scope = 'pipeai'
        self.logger.warning(
            f'Scope could not be inferred, set scope as "{default_scope}". '
            'You can silence this warning by passing `scope="your_scope"` '
            'when initializing Registry.'
        )
        return default_scope

    def _import_from_location(self) -> None:
        """Import modules from predefined locations to trigger registration."""
        if self._imported:
            return

        # fallback by official mapping
        if self._scope in MODULE_2_PACKAGE:
            try:
                utils_module = import_module(f"{self.scope}.utils")
                if hasattr(utils_module, "register_all_modules"):
                    utils_module.register_all_modules(False)
                self.logger.debug(
                    f"Fallback import for scope '{self.scope}' succeeded."
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not import fallback '{self.scope}.utils': {e}"
                )

        # import user-provided locations
        for loc in self._locations:
            try:
                import_module(loc)
                self.logger.debug(
                    f"Modules from '{loc}' imported for registry '{self.name}'."
                )
            except ImportError as e:
                self.logger.warning(f"Failed to import '{loc}': {e}")

        self._imported = True

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        """Internal method to register a module in the registry.

        Args:
            module (type): The class/function to register.
            module_name (str or list of str, optional): The name(s) to register under.
                Defaults to module.__name__ if not provided.
            force (bool): Whether to override an existing registration.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        names = [module.__name__] if module_name is None else (
            [module_name] if isinstance(module_name, str) else list(module_name))

        for name in names:
            if not force and name in self._module_dict:
                existed_module = self._module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def __len__(self) -> int:
        """Return the number of registered modules."""
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()
