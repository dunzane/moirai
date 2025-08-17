# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
import pytest
from pipeai.registry import Registry, build_from_cfg


class TestBuildFromCfg(unittest.TestCase):
    """Unit tests for the build_from_cfg function"""

    def setUp(self) -> None:
        """Set up a fresh registry for each test"""
        self.REG = Registry("test")

        @self.REG.register_module()
        class Cat:
            def __init__(self, name="kitty"):
                self.name = name

        @self.REG.register_module()
        def make_dog(name="doggy"):
            return {"dog": name}

    def test_invalid_cfg_type(self):
        """cfg must be a dict or Config"""
        with pytest.raises(TypeError):
            build_from_cfg("not-a-dict", self.REG)

    def test_invalid_registry_type(self):
        """registry must be an instance of Registry"""
        with pytest.raises(TypeError):
            build_from_cfg({}, registry="not-a-registry")

    def test_invalid_default_args_type(self):
        """default_args must be a dict, Config, or None"""
        with pytest.raises(TypeError):
            build_from_cfg({"type": "Cat"}, self.REG, default_args="bad-defaults")

    def test_missing_type_key(self):
        """cfg or default_args must contain 'type'"""
        with pytest.raises(KeyError):
            build_from_cfg({}, self.REG)

        # valid if default_args provides type
        obj = build_from_cfg({}, self.REG, default_args={"type": "Cat"})
        assert isinstance(obj, self.REG.get("Cat"))

    def test_build_class(self):
        """Test building a registered class with extra parameters"""
        obj = build_from_cfg({"type": "Cat", "name": "whiskers"}, self.REG)
        assert obj.name == "whiskers"

    def test_build_function(self):
        """Test building a registered function"""
        obj = build_from_cfg({"type": "make_dog", "name": "spike"}, self.REG)
        assert obj == {"dog": "spike"}

    def test_type_as_callable(self):
        """Test directly passing a callable instead of string"""
        def builder(x=1):
            return {"x": x}

        obj = build_from_cfg({"type": builder, "x": 42}, self.REG)
        assert obj == {"x": 42}

    def test_invalid_type_value(self):
        """'type' must be a str or callable"""
        with pytest.raises(TypeError):
            build_from_cfg({"type": 123}, self.REG)

    def test_unknown_type_str(self):
        """String 'type' must exist in the registry"""
        with pytest.raises(KeyError):
            build_from_cfg({"type": "NonExistent"}, self.REG)

    def test_scope_switching(self):
        """Test that _scope_ is consumed and does not remain in kwargs"""
        obj = build_from_cfg(
            {"type": "Cat", "name": "scoped", "_scope_": "test"},
            self.REG
        )
        assert obj.name == "scoped"


