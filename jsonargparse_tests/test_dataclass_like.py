from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union
from unittest.mock import patch

import pytest

from jsonargparse import (
    ArgumentError,
    ArgumentParser,
    Namespace,
    compose_dataclasses,
    lazy_instance,
    set_parsing_settings,
)
from jsonargparse._namespace import NSKeyError
from jsonargparse._optionals import (
    attrs_support,
    docstring_parser_support,
    pydantic_support,
    pydantic_supports_field_init,
    typing_extensions_import,
)
from jsonargparse.typing import PositiveFloat, PositiveInt, final
from jsonargparse_tests.conftest import (
    get_parser_help,
    json_or_yaml_load,
    skip_if_docstring_parser_unavailable,
)

annotated = typing_extensions_import("Annotated")
type_alias_type = typing_extensions_import("TypeAliasType")

# dataclass tests


@dataclasses.dataclass(frozen=True)
class DataClassA:
    """DataClassA description

    Args:
        a1: a1 help
        a2: a2 help
    """

    a1: PositiveInt = PositiveInt(1)
    a2: str = "2"


@dataclasses.dataclass
class DataClassB:
    """DataClassB description

    Args:
        b1: b1 help
        b2: b2 help
    """

    b1: PositiveFloat = PositiveFloat(3.0)
    b2: DataClassA = DataClassA(a2="x")


class MixedClass(int, DataClassA):
    """MixedClass description"""


def test_add_class_arguments(parser, subtests):
    parser.add_class_arguments(DataClassA, "a", default=DataClassA(), help="CustomA title")
    parser.add_class_arguments(DataClassB, "b", default=DataClassB())

    with subtests.test("get_defaults"):
        cfg = parser.get_defaults()
        assert dataclasses.asdict(DataClassA()) == cfg["a"].as_dict()
        assert dataclasses.asdict(DataClassB()) == cfg["b"].as_dict()
        dump = json_or_yaml_load(parser.dump(cfg))
        assert dataclasses.asdict(DataClassA()) == dump["a"]
        assert dataclasses.asdict(DataClassB()) == dump["b"]

    with subtests.test("instantiate_classes"):
        init = parser.instantiate_classes(cfg)
        assert isinstance(init["a"], DataClassA)
        assert isinstance(init["b"], DataClassB)
        assert isinstance(init["b"].b2, DataClassA)

    with subtests.test("parse_args"):
        assert 5 == parser.parse_args(["--b.b2.a1=5"]).b.b2.a1
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--b.b2.a1=x"]))

    with subtests.test("docstrings in help"):
        help_str = get_parser_help(parser)
        if docstring_parser_support:
            assert "CustomA title:" in help_str
            assert "DataClassB description:" in help_str
            assert "b2 help:" in help_str

    with subtests.test("add failures"):
        with pytest.raises(ValueError):
            parser.add_class_arguments(1, "c")
        with pytest.raises(NSKeyError):
            parser.add_class_arguments(DataClassB, "c", default=DataClassB(b2=DataClassB()))
        with pytest.raises(ValueError):
            parser.add_class_arguments(MixedClass, "c")


@dataclasses.dataclass
class NestedDefaultsA:
    x: list = dataclasses.field(default_factory=list)
    v: int = 1


@dataclasses.dataclass
class NestedDefaultsB:
    a: List[NestedDefaultsA]


@dataclasses.dataclass
class NestedDefaultsC:
    field_with_dash: int = 5


@dataclasses.dataclass
class NestedDefaultsD:
    c_with_dash: NestedDefaultsC = dataclasses.field(default_factory=NestedDefaultsC)


def test_add_dataclass_nested_defaults(parser):
    parser.add_class_arguments(NestedDefaultsB, "data")
    cfg = parser.parse_args(["--data.a=[{}]"])
    assert cfg.data == Namespace(a=[Namespace(x=[], v=1)])


class ClassDataAttributes:
    def __init__(
        self,
        a1: DataClassA = DataClassA(),
        a2: DataClassB = DataClassB(),
    ):
        self.a1 = a1
        self.a2 = a2


def test_add_class_with_dataclass_attributes(parser):
    parser.add_class_arguments(ClassDataAttributes, "g")

    cfg = parser.get_defaults()
    assert dataclasses.asdict(DataClassA()) == cfg.g.a1.as_dict()
    assert dataclasses.asdict(DataClassB()) == cfg.g.a2.as_dict()
    dump = json_or_yaml_load(parser.dump(cfg))
    assert dataclasses.asdict(DataClassA()) == dump["g"]["a1"]
    assert dataclasses.asdict(DataClassB()) == dump["g"]["a2"]

    init = parser.instantiate_classes(cfg)
    assert isinstance(init.g.a1, DataClassA)
    assert isinstance(init.g.a2, DataClassB)
    assert isinstance(init.g.a2.b2, DataClassA)


class SubBaseClass:
    def __init__(self, a1: DataClassB = DataClassB()):
        """SubBaseClass description"""
        self.a1 = a1


class RootClass:
    def __init__(self, c1: SubBaseClass):
        """RootClass description"""
        self.c1 = c1


def test_add_class_dataclass_typehint_in_subclass(parser):
    parser.add_class_arguments(RootClass)
    class_path = f'"class_path": "{__name__}.SubBaseClass"'
    init_args = '"init_args": {"a1": {"b2": {"a1": 7}}}'

    cfg = parser.parse_args(["--c1={" + class_path + ", " + init_args + "}"])
    assert cfg.c1.class_path == f"{__name__}.SubBaseClass"
    assert cfg.c1.init_args.a1.b2.a1 == 7
    assert isinstance(cfg.c1.init_args.a1.b2.a1, PositiveInt)

    init = parser.instantiate_classes(cfg)
    assert isinstance(init.c1, SubBaseClass)
    assert isinstance(init.c1.a1, DataClassB)
    assert isinstance(init.c1.a1.b2, DataClassA)
    assert isinstance(init.c1.a1.b1, PositiveFloat)


@dataclasses.dataclass
class OptionalWithDefault:
    param: Optional[str]


def test_add_class_optional_without_default(parser):
    parser.add_class_arguments(OptionalWithDefault)
    assert parser.get_defaults() == Namespace(param=None)
    assert parser.parse_args([]) == Namespace(param=None)
    assert parser.parse_args(["--param=null"]) == Namespace(param=None)


@dataclasses.dataclass
class ListOptionalA:
    x: int


@dataclasses.dataclass
class ListOptionalB:
    a: Optional[ListOptionalA] = None


def test_list_nested_optional_dataclass(parser):
    parser.add_argument("--b", type=List[ListOptionalB])
    cfg = parser.parse_args(['--b=[{"a":{"x":1}}]'])
    assert cfg.b == [Namespace(a=Namespace(x=1))]


@dataclasses.dataclass
class ItemData:
    x: int = 1
    y: str = "one"


def test_list_append_defaults(parser):
    parser.add_argument("--list", type=List[ItemData])
    cfg = parser.parse_args(["--list+={}", '--list+={"x":2,"y":"two"}', '--list+={"x":3}'])
    assert cfg.list == [Namespace(x=1, y="one"), Namespace(x=2, y="two"), Namespace(x=3, y="one")]


def test_add_argument_dataclass_type(parser):
    parser.add_argument("--b", type=DataClassB, default=DataClassB(b1=7.0))
    cfg = parser.get_defaults()
    assert {"b1": 7.0, "b2": {"a1": 1, "a2": "x"}} == cfg.b.as_dict()
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.b, DataClassB)
    assert isinstance(init.b.b2, DataClassA)


def test_add_argument_dataclass_unexpected_keys(parser):
    parser.add_argument("--b", type=DataClassB)
    invalid = {
        "class_path": f"{__name__}.DataClassB",
    }
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args([f"--b={json.dumps(invalid)}"])
    ctx.match("Group 'b' does not accept nested key 'class_path'")


@dataclasses.dataclass
class DataRequiredAttr:
    a1: str
    a2: float = 1.2


def test_add_argument_dataclass_type_required_attr(parser):
    parser.add_argument("--b", type=DataRequiredAttr)
    assert Namespace(a1="v", a2=1.2) == parser.parse_args(["--b.a1=v"]).b
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args([])
    ctx.match('"b.a1" is required')


@dataclasses.dataclass
class DataInitFalse:
    p1: str = dataclasses.field(init=False)


def test_dataclass_field_init_false(parser):
    added = parser.add_class_arguments(DataInitFalse, "data")
    assert added == []
    assert parser.get_defaults() == Namespace()
    cfg = parser.parse_args([])
    assert cfg == Namespace()
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.data, DataInitFalse)


@dataclasses.dataclass
class NestedDataInitFalse:
    x: bool = dataclasses.field(default=False, init=False)


@dataclasses.dataclass
class ParentDataInitFalse:
    y: NestedDataInitFalse = dataclasses.field(default_factory=NestedDataInitFalse)


def test_nested_dataclass_field_init_false(parser):
    parser.add_class_arguments(ParentDataInitFalse, "data")
    assert parser.get_defaults() == Namespace()
    cfg = parser.parse_args([])
    assert cfg == Namespace()
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.data, ParentDataInitFalse)
    assert isinstance(init.data.y, NestedDataInitFalse)
    assert init.data.y.x is False


@dataclasses.dataclass
class DataFieldFactory:
    a1: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
    a2: Dict[str, float] = dataclasses.field(default_factory=lambda: {"a": 1.2, "b": 3.4})


def test_dataclass_field_default_factory(parser):
    parser.add_class_arguments(DataFieldFactory)
    cfg = parser.get_defaults()
    assert [1, 2, 3] == cfg.a1
    assert {"a": 1.2, "b": 3.4} == cfg.a2


class UntypedClass:
    def __init__(self, c1):
        self.c1 = c1


@dataclasses.dataclass
class DataUntypedAttribute:
    a1: UntypedClass
    a2: str = "a2"
    a3: str = "a3"


def test_dataclass_fail_untyped_false(parser):
    parser.add_argument("--data", type=DataUntypedAttribute, fail_untyped=False)
    class_path = f'"class_path": "{__name__}.UntypedClass"'
    init_args = '"init_args": {"c1": 1}'

    cfg = parser.parse_args(["--data.a1={" + class_path + ", " + init_args + "}"])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.data, DataUntypedAttribute)
    assert isinstance(init.data.a1, UntypedClass)
    assert isinstance(init.data.a2, str)
    assert isinstance(init.data.a3, str)


@dataclasses.dataclass
class ComposeA:
    a: int = 1

    def __post_init__(self):
        self.a += 1


@dataclasses.dataclass
class ComposeB:
    b: str = "1"


def test_compose_dataclasses():
    ComposeAB = compose_dataclasses(ComposeA, ComposeB)
    assert 2 == len(dataclasses.fields(ComposeAB))
    assert {"a": 3, "b": "2"} == dataclasses.asdict(ComposeAB(a=2, b="2"))  # pylint: disable=unexpected-keyword-arg


@dataclasses.dataclass
class NestedData:
    name: str = "name"


class MainClass:
    def __init__(self, data: NestedData):
        self.data = data


def test_instantiate_dataclass_within_classes(parser):
    parser.add_class_arguments(MainClass, "class")
    cfg = parser.parse_args([])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init["class"], MainClass)
    assert isinstance(init["class"].data, NestedData)


@dataclasses.dataclass
class RequiredAttr:
    an_int: int


@dataclasses.dataclass
class NestedRequiredAttr:
    b: RequiredAttr


def test_list_nested_dataclass_required_attr(parser):
    parser.add_argument("--a", type=List[NestedRequiredAttr])
    cfg = parser.parse_args(['--a=[{"b": {"an_int": 3}}]'])
    assert cfg == Namespace(a=[Namespace(b=Namespace(an_int=3))])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.a[0].b, RequiredAttr)
    assert isinstance(init.a[0], NestedRequiredAttr)


@dataclasses.dataclass
class WithAttrDocs:
    attr_str: str = "a"
    "attr_str description"
    attr_int: int = 1
    "attr_int description"


@skip_if_docstring_parser_unavailable
@patch.dict("jsonargparse._optionals._docstring_parse_options")
def test_attribute_docstrings(parser):
    set_parsing_settings(docstring_parse_attribute_docstrings=True)
    parser.add_class_arguments(WithAttrDocs)
    help_str = get_parser_help(parser)
    assert "attr_str description (type: str, default: a)" in help_str
    assert "attr_int description (type: int, default: 1)" in help_str


@dataclasses.dataclass
class Data:
    p1: str
    p2: int = 0


parser_optional_data = ArgumentParser(exit_on_error=False)
parser_optional_data.add_argument("--data", type=Optional[Data])


def test_optional_dataclass_type_all_fields():
    cfg = parser_optional_data.parse_args(['--data={"p1": "x", "p2": 1}'])
    assert cfg == Namespace(data=Namespace(p1="x", p2=1))


def test_optional_dataclass_type_single_field():
    cfg = parser_optional_data.parse_args(['--data={"p1": "y"}'])
    assert cfg == Namespace(data=Namespace(p1="y", p2=0))


def test_optional_dataclass_type_invalid_field():
    with pytest.raises(ArgumentError):
        parser_optional_data.parse_args(['--data={"p1": 1}'])


def test_optional_dataclass_type_instantiate():
    cfg = parser_optional_data.parse_args(['--data={"p1": "y", "p2": 2}'])
    init = parser_optional_data.instantiate_classes(cfg)
    assert isinstance(init.data, Data)
    assert init.data.p1 == "y"
    assert init.data.p2 == 2


def test_optional_dataclass_type_dump():
    cfg = parser_optional_data.parse_args(['--data={"p1": "z"}'])
    assert json_or_yaml_load(parser_optional_data.dump(cfg)) == {"data": {"p1": "z", "p2": 0}}


def test_optional_dataclass_type_missing_required_field():
    with pytest.raises(ArgumentError):
        parser_optional_data.parse_args(['--data={"p2": 2}'])


def test_optional_dataclass_type_null_value():
    cfg = parser_optional_data.parse_args(["--data=null"])
    assert cfg == Namespace(data=None)
    assert cfg == parser_optional_data.instantiate_classes(cfg)


@dataclasses.dataclass
class DataWithOptionalB:
    c: int = 3


@dataclasses.dataclass
class DataWithOptionalA:
    b: Optional[DataWithOptionalB] = dataclasses.field(default_factory=DataWithOptionalB)


def data_with_optional(a: DataWithOptionalA):
    pass


def test_dataclass_with_optional_default(parser):
    parser.add_function_arguments(data_with_optional, "data")
    cfg = parser.parse_args([])
    assert cfg.data == Namespace(a=Namespace(b={"c": 3}))
    init = parser.instantiate_classes(cfg)
    assert init.data.a == DataWithOptionalA()


@dataclasses.dataclass
class SingleParamChange:
    p1: int = 0
    p2: int = 0


def test_optional_dataclass_single_param_change(parser):
    parser.add_argument("--config", action="config")
    parser.add_argument("--data", type=Optional[SingleParamChange])
    config = {"data": {"p1": 1}}
    cfg = parser.parse_args([f"--config={json.dumps(config)}", "--data.p2=2"])
    assert cfg.data == Namespace(p1=1, p2=2)


@dataclasses.dataclass
class ModelConfig:
    data: Optional[Dict[str, Any]] = None


def test_dataclass_optional_dict_attribute(parser):
    parser.add_argument("--model", type=Optional[ModelConfig], default=ModelConfig(data={"A": 1, "B": 2}))
    cfg = parser.parse_args(["--model.data.A=4"])
    assert cfg.model["data"] == {"A": 4, "B": 2}
    init = parser.instantiate_classes(cfg)
    assert init.model == ModelConfig(data={"A": 4, "B": 2})


def test_dataclass_in_union_type(parser):
    parser.add_argument("--union", type=Optional[Union[Data, int]])
    cfg = parser.parse_args(["--union=1"])
    assert cfg == Namespace(union=1)
    assert cfg == parser.instantiate_classes(cfg)


def test_dataclass_in_list_type(parser):
    parser.add_argument("--list", type=List[Data])
    cfg = parser.parse_args(['--list=[{"p1": "a"},{"p1": "b"}]'])
    init = parser.instantiate_classes(cfg)
    assert ["a", "b"] == [d.p1 for d in init.list]
    assert isinstance(init.list[0], Data)
    assert isinstance(init.list[1], Data)


T = TypeVar("T", int, float)


@dataclasses.dataclass
class GenericData(Generic[T]):
    g1: T
    g2: Tuple[T, T]
    g3: Union[str, T]
    g4: Dict[str, Union[T, bool]]


def test_generic_dataclass(parser):
    parser.add_argument("--data", type=GenericData[int])
    help_str = get_parser_help(parser).lower()
    assert "--data.g1 g1          (required, type: int)" in help_str
    assert "--data.g2 [item,...]  (required, type: tuple[int, int])" in help_str
    if sys.version_info < (3, 14):
        assert "--data.g3 g3          (required, type: union[str, int])" in help_str
        assert "--data.g4 g4          (required, type: dict[str, union[int, bool]])" in help_str
    else:
        assert "--data.g3 g3          (required, type: str | int)" in help_str
        assert "--data.g4 g4          (required, type: dict[str, int | bool])" in help_str


@dataclasses.dataclass
class SpecificData:
    y: GenericData[float]


def test_nested_generic_dataclass(parser):
    parser.add_class_arguments(SpecificData, "x")
    help_str = get_parser_help(parser).lower()
    assert "--x.y.g1 g1          (required, type: float)" in help_str
    assert "--x.y.g2 [item,...]  (required, type: tuple[float, float])" in help_str
    if sys.version_info < (3, 14):
        assert "--x.y.g3 g3          (required, type: union[str, float])" in help_str
        assert "--x.y.g4 g4          (required, type: dict[str, union[float, bool]])" in help_str
    else:
        assert "--x.y.g3 g3          (required, type: str | float)" in help_str
        assert "--x.y.g4 g4          (required, type: dict[str, float | bool])" in help_str


V = TypeVar("V")


@dataclasses.dataclass(frozen=True)
class GenericChild(Generic[V]):
    value: V


@dataclasses.dataclass(frozen=True)
class GenericBase(Generic[V]):
    children: tuple[GenericChild[V], ...]


@dataclasses.dataclass(frozen=True)
class GenericSubclass(GenericBase[str]):
    children: tuple[GenericChild[str], ...]


def test_generic_dataclass_subclass(parser):
    parser.add_class_arguments(GenericSubclass, "x")
    cfg = parser.parse_args(['--x.children=[{"value": "a"}, {"value": "b"}]'])
    init = parser.instantiate_classes(cfg)
    assert cfg.x.children == (Namespace(value="a"), Namespace(value="b"))
    assert isinstance(init.x, GenericSubclass)
    assert isinstance(init.x.children[0], GenericChild)
    assert isinstance(init.x.children[1], GenericChild)


# union mixture tests


@dataclasses.dataclass
class UnionData:
    data_a: int = 1
    data_b: Optional[str] = None


class UnionClass:
    def __init__(self, prm_1: float, prm_2: bool):
        self.prm_1 = prm_1
        self.prm_2 = prm_2


@pytest.mark.parametrize(
    "union_type",
    [
        Union[UnionData, UnionClass],
        Union[UnionClass, UnionData],
    ],
)
def test_class_path_union_mixture_dataclass_and_class(parser, union_type):
    parser.add_argument("--union", type=union_type, enable_path=True)

    value = {"class_path": f"{__name__}.UnionData", "init_args": {"data_a": 2, "data_b": "x"}}
    cfg = parser.parse_args([f"--union={json.dumps(value)}"])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.union, UnionData)
    assert dataclasses.asdict(init.union) == {"data_a": 2, "data_b": "x"}
    assert json_or_yaml_load(parser.dump(cfg))["union"] == value["init_args"]

    value = {"class_path": f"{__name__}.UnionClass", "init_args": {"prm_1": 1.2, "prm_2": False}}
    cfg = parser.parse_args([f"--union={json.dumps(value)}"])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.union, UnionClass)
    assert init.union.prm_1 == 1.2
    assert json_or_yaml_load(parser.dump(cfg))["union"] == value


# final classes tests


@final
class FinalClass:
    def __init__(self, a1: int = 1, a2: float = 2.3):
        self.a1 = a1
        self.a2 = a2


class NotFinalClass:
    def __init__(self, b1: str = "4", b2: FinalClass = lazy_instance(FinalClass, a2=-3.2)):
        self.b1 = b1
        self.b2 = b2


def test_add_class_final(parser):
    parser.add_class_arguments(NotFinalClass, "b")

    assert parser.get_defaults().b.b2 == Namespace(a1=1, a2=-3.2)
    cfg = parser.parse_args(['--b.b2={"a2": 6.7}'])
    assert cfg.b.b2 == Namespace(a1=1, a2=6.7)
    assert cfg == parser.parse_string(parser.dump(cfg))
    cfg = parser.instantiate_classes(cfg)
    assert isinstance(cfg["b"], NotFinalClass)
    assert isinstance(cfg["b"].b2, FinalClass)

    pytest.raises(ArgumentError, lambda: parser.parse_args(['--b.b2={"bad": "value"}']))
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--b.b2="bad"']))
    pytest.raises(ValueError, lambda: parser.add_subclass_arguments(FinalClass, "a"))
    pytest.raises(ValueError, lambda: parser.add_class_arguments(FinalClass, "a", default=FinalClass()))


if type_alias_type:
    IntOrString = type_alias_type("IntOrString", Union[int, str])

    @dataclasses.dataclass
    class DataClassWithAliasType:
        p1: IntOrString  # type: ignore[valid-type]

    def test_bare_alias_type(parser):
        parser.add_argument("--data", type=IntOrString)
        help_str = get_parser_help(parser)
        help_str_lines = [line for line in help_str.split("\n") if "type: IntOrString" in line]
        assert len(help_str_lines) == 1
        assert "--data DATA" in help_str_lines[0]
        cfg = parser.parse_args(["--data=MyString"])
        assert cfg.data == "MyString"
        cfg = parser.parse_args(["--data=3"])
        assert cfg.data == 3

    def test_dataclass_with_alias_type(parser):
        parser.add_argument("--data", type=DataClassWithAliasType)
        help_str = get_parser_help(parser)
        help_str_lines = [line for line in help_str.split("\n") if "type: IntOrString" in line]
        assert len(help_str_lines) == 1
        assert "--data.p1 P1" in help_str_lines[0]
        cfg = parser.parse_args(["--data.p1=MyString"])
        assert cfg.data.p1 == "MyString"
        cfg = parser.parse_args(["--data.p1=3"])
        assert cfg.data.p1 == 3

    @pytest.mark.skipif(not annotated, reason="Annotated is required")
    def test_annotated_alias_type(parser):
        parser.add_argument("--data", type=annotated[IntOrString, 1])
        help_str = get_parser_help(parser)
        help_str_lines = [line for line in help_str.split("\n") if "type: Annotated[IntOrString, 1]" in line]
        assert len(help_str_lines) == 1
        assert "--data DATA" in help_str_lines[0]
        cfg = parser.parse_args(["--data=MyString"])
        assert cfg.data == "MyString"
        cfg = parser.parse_args(["--data=3"])
        assert cfg.data == 3

    if annotated:

        @dataclasses.dataclass
        class DataClassWithAnnotatedAliasType:
            p1: annotated[IntOrString, 1]  # type: ignore[valid-type]

    @pytest.mark.skipif(not annotated, reason="Annotated is required")
    def test_dataclass_with_annotated_alias_type(parser):
        parser.add_argument("--data", type=DataClassWithAnnotatedAliasType)
        help_str = get_parser_help(parser)
        # The printable field datatype is not uniform across versions.
        help_str_lines = [line for line in help_str.split("\n") if "type:" in line and "IntOrString" in line]
        assert len(help_str_lines) == 1
        assert "--data.p1 P1" in help_str_lines[0]
        cfg = parser.parse_args(["--data.p1=MyString"])
        assert cfg.data.p1 == "MyString"
        cfg = parser.parse_args(["--data.p1=3"])
        assert cfg.data.p1 == 3


# pydantic tests
if annotated and pydantic_support > 1:
    import pydantic

    @pydantic.dataclasses.dataclass(frozen=True)
    class InnerDataClass:
        a2: int = 1

    @pydantic.dataclasses.dataclass(frozen=True)
    class NestedAnnotatedDataClass:
        a1: annotated[InnerDataClass, 1]  # type: ignore[valid-type]

    @pydantic.dataclasses.dataclass(frozen=True)
    class NestedAnnotatedDataClassWithDefault:
        a1: annotated[InnerDataClass, 1] = pydantic.fields.Field(default=InnerDataClass())  # type: ignore[valid-type]

    @pydantic.dataclasses.dataclass(frozen=True)
    class NestedAnnotatedDataClassWithDefaultFactory:
        a1: annotated[InnerDataClass, 1] = pydantic.fields.Field(default_factory=InnerDataClass)  # type: ignore[valid-type]

    def test_pydantic_nested_annotated_dataclass(parser: ArgumentParser):
        parser.add_class_arguments(NestedAnnotatedDataClass, "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass(parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClass, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass_with_default(parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClassWithDefault, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass_with_default_factory(parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClassWithDefaultFactory, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    class PingTask(pydantic.BaseModel):
        type: Literal["ping"] = "ping"
        attr: str = ""

    class PongTask(pydantic.BaseModel):
        type: Literal["pong"] = "pong"

    PingPongTask = annotated[
        Union[PingTask, PongTask],
        pydantic.Field(discriminator="type"),
    ]


length = "length"
if pydantic_support:
    import pydantic

    @pydantic.dataclasses.dataclass
    class PydanticData:
        p1: float = 0.1
        p2: str = "-"

    @pydantic.dataclasses.dataclass
    class PydanticDataNested:
        p3: PydanticData

    if pydantic_supports_field_init:
        from pydantic.dataclasses import dataclass as pydantic_v2_dataclass
        from pydantic.fields import Field as PydanticV2Field

        @pydantic_v2_dataclass
        class PydanticDataFieldInitFalse:
            p1: str = PydanticV2Field("-", init=False)

    @pydantic.dataclasses.dataclass
    class PydanticDataStdlibField:
        p1: str = dataclasses.field(default="-")

    @pydantic.dataclasses.dataclass
    class PydanticDataStdlibFieldWithFactory:
        p1: str = dataclasses.field(default_factory=lambda: "-")

    class PydanticModel(pydantic.BaseModel):
        p1: str
        p2: int = 3

    class PydanticSubModel(PydanticModel):
        p3: float = 0.1

    class PydanticFieldFactory(pydantic.BaseModel):
        p1: List[int] = pydantic.Field(default_factory=lambda: [1, 2])

    class PydanticHelp(pydantic.BaseModel):
        """
        Args:
            p1: p1 help
        """

        p1: str
        p2: int = pydantic.Field(2, description="p2 help")

    if pydantic_support == 1:
        length = "items"

    if annotated and pydantic_support > 1:

        class PydanticAnnotatedField(pydantic.BaseModel):
            p1: annotated[int, pydantic.Field(default=2, ge=1, le=8)]  # type: ignore[valid-type]

    class OptionalPydantic:
        def __init__(self, a: Optional[PydanticModel] = None):
            self.a = a

    class NestedModel(pydantic.BaseModel):
        inputs: List[str]
        outputs: List[str]

    class PydanticNestedDict(pydantic.BaseModel):
        nested: Optional[Dict[str, NestedModel]] = None


def none(x):
    return x


@pytest.mark.skipif(not pydantic_support, reason="pydantic package is required")
class TestPydantic:
    num_models = 0

    def test_dataclass(self, parser):
        parser.add_argument("--data", type=PydanticData)
        defaults = parser.get_defaults()
        assert Namespace(p1=0.1, p2="-") == defaults.data
        cfg = parser.parse_args(["--data.p1=0.2", "--data.p2=x"])
        assert Namespace(p1=0.2, p2="x") == cfg.data

    def test_basemodel(self, parser):
        parser.add_argument("--model", type=PydanticModel, default=PydanticModel(p1="a"))
        cfg = parser.parse_args(["--model.p2=5"])
        assert Namespace(p1="a", p2=5) == cfg.model

    def test_subclass(self, parser):
        parser.add_argument("--model", type=PydanticSubModel, default=PydanticSubModel(p1="a"))
        cfg = parser.parse_args(["--model.p3=0.2"])
        assert Namespace(p1="a", p2=3, p3=0.2) == cfg.model
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.model, PydanticSubModel)

    def test_field_default_factory(self, parser):
        parser.add_argument("--model", type=PydanticFieldFactory)
        cfg1 = parser.parse_args([])
        cfg2 = parser.parse_args([])
        assert cfg1.model.p1 == [1, 2]
        assert cfg1.model.p1 == cfg2.model.p1
        assert cfg1.model.p1 is not cfg2.model.p1

    def test_field_description(self, parser):
        parser.add_argument("--model", type=PydanticHelp)
        help_str = get_parser_help(parser)
        if docstring_parser_support:
            assert "p1 help (required, type: str)" in help_str
        assert "p2 help (type: int, default: 2)" in help_str

    @pytest.mark.skipif(not (annotated and pydantic_support > 1), reason="Annotated is required")
    def test_annotated_field(self, parser):
        parser.add_argument("--model", type=PydanticAnnotatedField)
        cfg = parser.parse_args([])
        assert cfg.model.p1 == 2
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args(["--model.p1=0"])
        ctx.match("model.p1")

    @pytest.mark.skipif(not (annotated and pydantic_support > 1), reason="Annotated is required")
    def test_field_union_discriminator_dot_syntax(self, parser):
        parser.add_argument("--model", type=PingPongTask)
        cfg = parser.parse_args(["--model.type=pong"])
        assert cfg.model == Namespace(type="pong")
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.model, PongTask)
        cfg = parser.parse_args(["--model.type=ping", "--model.attr=abc"])
        assert cfg.model == Namespace(type="ping", attr="abc")
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.model, PingTask)

    @pytest.mark.parametrize(
        ["valid_value", "invalid_value", "cast", "type_str"],
        [
            ("abc", "a", none, "constr(min_length=2, max_length=4)"),
            (2, 0, none, "conint(ge=1)"),
            (-1.0, 1.0, none, "confloat(lt=0.0)"),
            ([1], [], none, f"conlist(int, min_{length}=1)"),
            ([], [3, 4], none, f"conlist(int, max_{length}=1)"),
            ([1], "x", list, f"conset(int, min_{length}=1)"),
            ("http://abc.es/", "-", str, "HttpUrl"),
            ("127.0.0.1", "0", str, "IPvAnyAddress"),
        ],
    )
    def test_pydantic_types(self, valid_value, invalid_value, cast, type_str, monkeypatch):
        pydantic_type = eval(f"pydantic.{type_str}")
        self.num_models += 1
        Model = pydantic.create_model(f"Model{self.num_models}", param=(pydantic_type, ...))
        if pydantic_support == 1:
            monkeypatch.setitem(Model.__init__.__globals__, "pydantic_type", pydantic_type)

        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=Model)
        cfg = parser.parse_args([f"--model.param={valid_value}"])
        assert cast(cfg.model.param) == valid_value
        dump = json_or_yaml_load(parser.dump(cfg))
        assert dump == {"model": {"param": valid_value}}
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args([f"--model.param={invalid_value}"])
        ctx.match("model.param")

    @pytest.mark.skipif(not pydantic_supports_field_init, reason="Field.init is required")
    def test_dataclass_field_init_false(self, parser):
        # Prior to PR #480, this test would produce the following error:
        #
        # TypeError: Parser key "data.p1":
        # Expected a <class 'str'>. Got value: annotation=str
        # required=False default='-' init=False
        parser.add_argument("--data", type=PydanticDataFieldInitFalse)
        help_str = get_parser_help(parser)
        assert "--data.p1" not in help_str
        cfg = parser.parse_args([])
        assert cfg == Namespace()

        init = parser.instantiate_classes(cfg)
        assert init.data.p1 == "-"

    def test_dataclass_stdlib_field(self, parser):
        parser.add_argument("--data", type=PydanticDataStdlibField)
        cfg = parser.parse_args(["--data", "{}"])
        assert cfg.data == Namespace(p1="-")

    def test_dataclass_stdlib_field_init_with_factory(self, parser):
        parser.add_argument("--data", type=PydanticDataStdlibFieldWithFactory)
        cfg = parser.parse_args(["--data", "{}"])
        assert cfg.data == Namespace(p1="-")

    def test_dataclass_nested(self, parser):
        # Prior to PR #480, this test would produce the following error:
        #
        # ValueError: Expected "default" argument to be an instance of
        # "PydanticData" or its kwargs dict, given
        # <dataclasses._MISSING_TYPE object at 0x105624c50>

        parser.add_argument("--data", type=PydanticDataNested)
        cfg = parser.parse_args(["--data", '{"p3": {"p1": 1.0}}'])
        assert cfg.data == Namespace(p3=Namespace(p1=1.0, p2="-"))

    def test_optional_pydantic_model(self, parser):
        parser.add_argument("--b", type=OptionalPydantic)
        parser.add_argument("--cfg", action="config")
        cfg = parser.parse_args([f"--b={__name__}.OptionalPydantic"])
        assert cfg.b.class_path == f"{__name__}.OptionalPydantic"
        assert cfg.b.init_args == Namespace(a=None)
        config = {
            "b": {
                "class_path": f"{__name__}.OptionalPydantic",
                "init_args": {"a": {"p1": "x"}},
            }
        }
        cfg = parser.parse_args([f"--cfg={json.dumps(config)}"])
        assert cfg.b.class_path == f"{__name__}.OptionalPydantic"
        assert cfg.b.init_args == Namespace(a=Namespace(p1="x", p2=3))

    def test_nested_dict(self, parser):
        parser.add_argument("--config", action="config")
        parser.add_argument("--model", type=PydanticNestedDict)
        model = {
            "nested": {
                "key": {
                    "inputs": ["a", "b"],
                    "outputs": ["x", "y"],
                }
            }
        }
        cfg = parser.parse_args(["--model", json.dumps(model)])
        assert cfg.model.nested["key"] == Namespace(inputs=["a", "b"], outputs=["x", "y"])
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.model, PydanticNestedDict)
        assert isinstance(init.model.nested["key"], NestedModel)

    def test_dashes_in_nested_dataclass(self):
        class UnderscoresToDashesParser(ArgumentParser):
            def add_argument(self, *args, **kwargs):
                args = [arg.replace("_", "-") for arg in args]
                return super().add_argument(*args, **kwargs)

        parser = UnderscoresToDashesParser(parse_as_dict=False, default_env=True)
        parser.add_class_arguments(NestedDefaultsD)
        ns = parser.parse_args([])
        cfg = parser.instantiate_classes(ns)
        assert cfg.c_with_dash.field_with_dash == 5


# attrs tests

if attrs_support:
    import attrs

    @attrs.define
    class AttrsData:
        p1: float
        p2: str = "-"

    @attrs.define
    class AttrsSubData(AttrsData):
        p3: int = 3

    @attrs.define
    class AttrsFieldFactory:
        p1: List[str] = attrs.field(factory=lambda: ["one", "two"])

    @attrs.define
    class AttrsFieldInitFalse:
        p1: dict = attrs.field(init=False)

        def __attrs_post_init__(self):
            self.p1 = {}

    @attrs.define
    class AttrsSubField:
        p1: str = "-"
        p2: int = 0

    @attrs.define
    class AttrsWithNestedDefaultDataclass:
        p1: float
        subfield: AttrsSubField = attrs.field(factory=AttrsSubField)

    @attrs.define
    class AttrsWithNestedDataclassNoDefault:
        p1: float
        subfield: AttrsSubField


@pytest.mark.skipif(not attrs_support, reason="attrs package is required")
class TestAttrs:
    def test_define(self, parser):
        parser.add_argument("--data", type=AttrsData)
        defaults = parser.get_defaults()
        assert Namespace(p1=None, p2="-") == defaults.data
        cfg = parser.parse_args(["--data.p1=0.2", "--data.p2=x"])
        assert Namespace(p1=0.2, p2="x") == cfg.data

    def test_subclass(self, parser):
        parser.add_argument("--data", type=AttrsSubData)
        defaults = parser.get_defaults()
        assert Namespace(p1=None, p2="-", p3=3) == defaults.data

    def test_field_factory(self, parser):
        parser.add_argument("--data", type=AttrsFieldFactory)
        cfg1 = parser.parse_args([])
        cfg2 = parser.parse_args([])
        assert cfg1.data.p1 == ["one", "two"]
        assert cfg1.data.p1 == cfg2.data.p1
        assert cfg1.data.p1 is not cfg2.data.p1

    def test_field_init_false(self, parser):
        # Prior to PR #480, this test would produce the following error:
        #
        # TypeError('Validation failed: Key "data.p1" is required but
        # not included in config object or its value is None.')

        parser.add_argument("--data", type=AttrsFieldInitFalse)
        cfg = parser.parse_args([])
        help_str = get_parser_help(parser)
        assert "--data.p1" not in help_str
        assert cfg == Namespace()
        init = parser.instantiate_classes(cfg)
        assert init.data.p1 == {}

    def test_nested_with_default(self, parser):
        parser.add_argument("--data", type=AttrsWithNestedDefaultDataclass)
        cfg = parser.parse_args(["--data.p1=1.23"])
        assert cfg.data == Namespace(p1=1.23, subfield=Namespace(p1="-", p2=0))

    def test_nested_without_default(self, parser):
        parser.add_argument("--data", type=AttrsWithNestedDataclassNoDefault)
        cfg = parser.parse_args(["--data.p1=1.23"])
        assert cfg.data == Namespace(p1=1.23, subfield=Namespace(p1="-", p2=0))
