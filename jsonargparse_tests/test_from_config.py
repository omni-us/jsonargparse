import inspect
import sys
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser, from_config_support
from jsonargparse_tests.conftest import json_or_yaml_dump

# decorator usage tests


def test_decorator_multiple_positional_arguments():
    class Class:
        pass

    with pytest.raises(TypeError, match="from_config_support can only receive a single positional argument"):
        from_config_support(Class, 123)


def test_decorator_non_class_argument():
    with pytest.raises(TypeError, match="from_config_support can only be applied to classes"):
        from_config_support(123)


# defaults override tests


class DefaultsOverrideSelf:
    def __init__(self, param1: str = "default_value", param2: int = 1):
        self.param1 = param1
        self.param2 = param2


@pytest.mark.skipif(sys.version_info < (3, 11), reason="patch.object doesn't work correctly")
def test_defaults_override_self(tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(json_or_yaml_dump({"param1": "overridden_from_path"}))
    DefaultsOverrideSelf.__from_config_defaults__ = config_path

    with patch.object(ArgumentParser, "parse_path", wraps=ArgumentParser.parse_path, autospec=True) as mock:
        from_config_support(DefaultsOverrideSelf)
        assert mock.call_count == 1
        assert mock.mock_calls[0].kwargs["defaults"] is False

    params = inspect.signature(DefaultsOverrideSelf.__init__).parameters
    assert params["param1"].default == "overridden_from_path"

    instance = DefaultsOverrideSelf()
    assert instance.param1 == "overridden_from_path"
    assert instance.param2 == 1


@from_config_support
class DefaultsOverrideParent:
    def __init__(self, parent2: int = 1, parent1: str = "parent_default_value"):
        self.parent1 = parent1
        self.parent2 = parent2


def test_defaults_override_subclass(tmp_cwd, subtests):
    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(json_or_yaml_dump({"parent1": "overridden_parent", "child1": "overridden_child"}))

    class DefaultsOverrideChild(DefaultsOverrideParent):
        __from_config_defaults__ = config_path

        def __init__(self, child2: int = 2, child1: str = "child_default_value", **kwargs):
            super().__init__(**kwargs)
            self.child1 = child1
            self.child2 = child2

    with subtests.test("overridden subclass defaults"):
        params = inspect.signature(DefaultsOverrideChild.__init__).parameters
        assert params["parent1"].default == "overridden_parent"
        assert params["child1"].default == "overridden_child"

        instance = DefaultsOverrideChild()
        assert instance.parent1 == "overridden_parent"
        assert instance.parent2 == 1
        assert instance.child1 == "overridden_child"
        assert instance.child2 == 2

    with subtests.test("shadow override"):
        instance = DefaultsOverrideChild(child1="shadowed_child", parent1="shadowed_parent", child2=3)
        assert instance.parent1 == "shadowed_parent"
        assert instance.parent2 == 1
        assert instance.child1 == "shadowed_child"
        assert instance.child2 == 3

    with subtests.test("parent class unaffected"):
        params = inspect.signature(DefaultsOverrideParent.__init__).parameters
        assert params["parent1"].default == "parent_default_value"

        parent = DefaultsOverrideParent()
        assert parent.parent1 == "parent_default_value"


def test_defaults_override_keyword_only_parameters(tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(json_or_yaml_dump({"parent1": "overridden_parent", "child1": "overridden_child"}))

    @from_config_support
    class DefaultsOverrideKeywordOnlyParent:
        def __init__(self, *, parent1: str = "parent_default_value", parent2: int = 1):
            self.parent1 = parent1
            self.parent2 = parent2

    class DefaultsOverrideKeywordOnlyChild(DefaultsOverrideKeywordOnlyParent):
        __from_config_defaults__ = config_path

        def __init__(self, *, child2: int = 2, child1: str = "child_default_value", **kwargs):
            super().__init__(**kwargs)
            self.child1 = child1
            self.child2 = child2

    params = inspect.signature(DefaultsOverrideKeywordOnlyChild.__init__).parameters
    assert params["parent1"].default == "overridden_parent"
    assert params["child1"].default == "overridden_child"

    instance = DefaultsOverrideKeywordOnlyChild()
    assert instance.parent1 == "overridden_parent"
    assert instance.parent2 == 1
    assert instance.child1 == "overridden_child"
    assert instance.child2 == 2


def test_defaults_override_class_with_init_subclass(tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(json_or_yaml_dump({"parent": "overridden_parent", "child": "overridden_child"}))

    @from_config_support
    class DefaultsOverrideBase:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.original_init_subclass_ran = True

        def __init__(self, parent: str = "default_value"):
            self.parent = parent

    class DefaultsOverrideDerived(DefaultsOverrideBase):
        __from_config_defaults__ = config_path

        def __init__(self, child: str = "default_value", **kwargs):
            super().__init__(**kwargs)
            self.child = child

    params = inspect.signature(DefaultsOverrideDerived.__init__).parameters
    assert params["parent"].default == "overridden_parent"
    assert params["child"].default == "overridden_child"

    instance = DefaultsOverrideDerived()
    assert instance.parent == "overridden_parent"
    assert instance.child == "overridden_child"
    assert instance.original_init_subclass_ran


def test_defaults_override_file_not_found():
    class DefaultsOverrideFileNotFound(DefaultsOverrideParent):
        __from_config_defaults__ = "non_existent_file.yaml"

    assert hasattr(DefaultsOverrideFileNotFound, "from_config")  # make sure decorator applied to subclass
    instance = DefaultsOverrideFileNotFound()
    assert instance.parent1 == "parent_default_value"
    assert instance.parent2 == 1


def test_defaults_override_invalid():
    with pytest.raises(TypeError, match="__from_config_defaults__ must be str, PathLike, or None"):

        @from_config_support
        class DefaultsOverrideInvalid:
            __from_config_defaults__ = 123  # Invalid type


# from_config method tests


def test_without_from_config_method():
    @from_config_support(from_config_method=False)
    class WithoutFromConfigMethod:
        pass

    assert not hasattr(WithoutFromConfigMethod, "from_config")


def test_from_config_method_path(tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(json_or_yaml_dump({"param": "value_from_file"}))

    @from_config_support
    class FromConfigMethodPath:
        def __init__(self, param: str = "default_value"):
            self.param = param

    instance = FromConfigMethodPath.from_config(config_path)
    assert instance.param == "value_from_file"
    assert FromConfigMethodPath.from_config.__func__.__qualname__ == "FromConfigMethodPath.from_config"


def test_from_config_method_dict():
    @from_config_support
    class FromConfigMethodDict:
        def __init__(self, param: str = "default_value"):
            self.param = param

    instance = FromConfigMethodDict.from_config({"param": "value_from_dict"})
    assert instance.param == "value_from_dict"


def test_from_config_method_default():
    @from_config_support(from_config_method_default={"param1": "method_default_value"})
    class FromConfigMethodDefault:
        def __init__(self, param1: str = "default_value", param2: int = 1):
            self.param1 = param1
            self.param2 = param2

    instance = FromConfigMethodDefault.from_config()
    assert instance.param1 == "method_default_value"
    assert instance.param2 == 1


def test_from_config_method_subclass():
    @from_config_support
    class FromConfigMethodParent:
        def __init__(self, parent_param: str = "parent_default"):
            self.parent_param = parent_param

    class FromConfigMethodChild(FromConfigMethodParent):
        def __init__(self, child_param: str = "child_default", **kwargs):
            super().__init__(**kwargs)
            self.child_param = child_param

    instance = FromConfigMethodChild.from_config(
        {"parent_param": "overridden_parent", "child_param": "overridden_child"}
    )
    assert isinstance(instance, FromConfigMethodChild)
    assert instance.parent_param == "overridden_parent"
    assert instance.child_param == "overridden_child"


def test_from_config_method_custom_name():
    @from_config_support(from_config_method_name="custom_name")
    class FromConfigMethodCustomName:
        def __init__(self, param: str = "default_value"):
            self.param = param

    assert hasattr(FromConfigMethodCustomName, "custom_name")
    instance = FromConfigMethodCustomName.custom_name({"param": "custom_name_value"})
    assert instance.param == "custom_name_value"
    assert FromConfigMethodCustomName.custom_name.__func__.__qualname__ == "FromConfigMethodCustomName.custom_name"
