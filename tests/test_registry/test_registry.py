# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
import types
import pytest

from pipeai.registry import Registry, build_from_cfg


class TestRegistry(unittest.TestCase):
    """Unit tests for the Registry class"""

    def setUp(self) -> None:
        """Set up a clean registry before each test"""
        self.CATS = Registry('cat')
        assert self.CATS.name == 'cat'
        assert self.CATS.module_dict == {}
        assert self.CATS.build_func is build_from_cfg
        assert len(self.CATS) == 0

    def test_build_func(self) -> None:
        """Custom build_func should override the default build_from_cfg"""

        def custom_build(cfg, registry, default_args=None):
            return "ok"

        DOGS = Registry('dog', build_func=custom_build)
        assert DOGS.build_func is custom_build
        assert DOGS.build({}, default_args={}) == "ok"

    def test_scope_and_switch(self) -> None:
        """Test explicit scope, inferred scope, and temporary scope switching"""

        cats = Registry('cat', scope='cat')
        assert cats.scope == 'cat'

        default_registry = Registry('dog')
        assert default_registry.scope.startswith("test"), \
            f"Expected inferred scope to start with 'test', got {default_registry.scope}"

        with cats.switch_scope_and_registry('tmp_scope') as reg:
            assert reg.scope == 'cat'
            assert reg is cats

    def test_split_scope_key(self):
        """Test splitting keys into scope and name parts"""

        DOGS = Registry('dogs')

        scope, key = DOGS.split_scope_key('BloodHound')
        assert scope is None and key == 'BloodHound'
        scope, key = DOGS.split_scope_key('hound.BloodHound')
        assert scope == 'hound' and key == 'BloodHound'
        scope, key = DOGS.split_scope_key('hound.little_hound.Dachshund')
        assert scope == 'hound' and key == 'little_hound.Dachshund'

    def test_register_module_and_get(self):
        """Test registering modules, retrieving them, and alias support"""

        CATS = Registry('cat')

        @CATS.register_module()
        def miao():
            return "miao"

        assert CATS.get('miao') is miao
        assert 'miao' in CATS

        @CATS.register_module()
        def jump():
            return "jump"

        assert CATS.get('jump') is jump
        assert len(CATS) == 2

        @CATS.register_module(name=["alias1", "alias2"])
        def alias():
            return "alias"

        assert len(CATS) == 4
        assert CATS.get('alias1') is alias
        assert CATS.get('alias2') is alias

    def test_register_module_force(self):
        """Test that force=True allows overwriting existing modules"""

        DUCK = Registry('duck')

        @DUCK.register_module()
        def duck1():
            return "duck1"

        @DUCK.register_module(force=True)
        def duck1():
            return "duck1-new"

        assert DUCK.get("duck1")() == "duck1-new"

    def test_register_module_errors(self):
        """Test that invalid registrations raise appropriate errors"""

        REG = Registry('err')

        with pytest.raises(TypeError):
            REG.register_module(module="string")

        with pytest.raises(TypeError):
            @REG.register_module(name=123)
            def bad():
                pass

        @REG.register_module()
        def foo():
            pass

        with pytest.raises(KeyError):
            @REG.register_module()
            def foo():
                pass

    def test_get_import_path(self):
        """Test get() with fully qualified import path and error handling"""

        MODS = Registry('mods')

        cls = MODS.get("types.SimpleNamespace")
        assert cls is types.SimpleNamespace

        res = MODS.get("types.DoesNotExist")
        assert res is None

        with pytest.raises(TypeError):
            MODS.get(123)

    def test_len_contains_repr(self):
        """Test __len__, __contains__, and __repr__ output formatting"""

        FISH = Registry('fish')

        @FISH.register_module()
        def goldfish():
            pass

        assert len(FISH) == 1
        assert "goldfish" in FISH

        rep = repr(FISH)
        assert "Registry of fish" in rep
        assert "goldfish" in rep
