from typing import Optional, Generator


class SimpleScopeManager:
    """A lightweight global scope manager."""
    _current_scope: Optional[str] = None

    @classmethod
    def get_scope(cls) -> Optional[str]:
        return cls._current_scope

    @classmethod
    def set_scope(cls, scope: Optional[str]):
        cls._current_scope = scope
