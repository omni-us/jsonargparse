from __future__ import annotations

import dataclasses
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from unittest.mock import patch

import pytest
import yaml

from jsonargparse import (
    ArgumentError,
    ArgumentParser,
    Namespace,
    compose_dataclasses,
    lazy_instance,
)
from jsonargparse._optionals import (
    attrs_support,
    docstring_parser_support,
    pydantic_support,
    set_docstring_parse_options,
    typing_extensions_import,
)
from jsonargparse.typing import PositiveFloat, PositiveInt, final
from jsonargparse_tests.conftest import (
    get_parser_help,
    skip_if_docstring_parser_unavailable,
)

# dataclass tests


@dataclasses.dataclass(frozen=True)
class DataClassA:
    """DataClassA description

    Args:
        a1: a1 help
        a2: a2 help
    """

    a1: PositiveInt = PositiveInt(1)  # type: ignore
    a2: str = "2"


@dataclasses.dataclass
class DataClassB:
    """DataClassB description

    Args:
        b1: b1 help
        b2: b2 help
    """

    b1: PositiveFloat = PositiveFloat(3.0)  # type: ignore
    b2: DataClassA = DataClassA()


class MixedClass(int, DataClassA):
    """MixedClass description"""


def test_add_dataclass_arguments(parser, subtests):
    parser.add_dataclass_arguments(DataClassA, "a", default=DataClassA(), title="CustomA title")
    parser.add_dataclass_arguments(DataClassB, "b", default=DataClassB())

    with subtests.test("get_defaults"):
        cfg = parser.get_defaults()
        assert dataclasses.asdict(DataClassA()) == cfg["a"].as_dict()
        assert dataclasses.asdict(DataClassB()) == cfg["b"].as_dict()
        dump = yaml.safe_load(parser.dump(cfg))
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
            parser.add_dataclass_arguments(1, "c")
        with pytest.raises(ValueError):
            parser.add_dataclass_arguments(DataClassB, "c", default=DataClassB(b2=DataClassB()))
        with pytest.raises(ValueError):
            parser.add_dataclass_arguments(MixedClass, "c")


@dataclasses.dataclass
class NestedDefaultsA:
    x: list = dataclasses.field(default_factory=list)
    v: int = 1


@dataclasses.dataclass
class NestedDefaultsB:
    a: List[NestedDefaultsA]


def test_add_dataclass_nested_defaults(parser):
    parser.add_dataclass_arguments(NestedDefaultsB, "data")
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
    dump = yaml.safe_load(parser.dump(cfg))
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


def test_add_argument_dataclass_type(parser):
    parser.add_argument("--b", type=DataClassB, default=DataClassB(b1=7.0))
    cfg = parser.get_defaults()
    assert {"b1": 7.0, "b2": {"a1": 1, "a2": "2"}} == cfg.b.as_dict()
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.b, DataClassB)
    assert isinstance(init.b.b2, DataClassA)


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
    p1: str = "-"
    p2: str = dataclasses.field(init=False)


def test_dataclass_field_init_false(parser):
    added = parser.add_dataclass_arguments(DataInitFalse, "d")
    assert added == ["d.p1"]
    assert parser.get_defaults() == Namespace(d=Namespace(p1="-"))


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
class WithAttrDocs:
    attr_str: str = "a"
    "attr_str description"
    attr_int: int = 1
    "attr_int description"


@skip_if_docstring_parser_unavailable
@patch.dict("jsonargparse._optionals._docstring_parse_options")
def test_attribute_docstrings(parser):
    set_docstring_parse_options(attribute_docstrings=True)
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
    assert parser_optional_data.dump(cfg) == "data:\n  p1: z\n  p2: 0\n"


def test_optional_dataclass_type_missing_required_field():
    with pytest.raises(ArgumentError):
        parser_optional_data.parse_args(['--data={"p2": 2}'])


def test_optional_dataclass_type_null_value():
    cfg = parser_optional_data.parse_args(["--data=null"])
    assert cfg == Namespace(data=None)
    assert cfg == parser_optional_data.instantiate_classes(cfg)


@dataclasses.dataclass
class ModelConfig:
    data: Optional[Dict[str, Any]] = None


def test_dataclass_optional_dict_attribute(parser):
    parser.add_argument("--model", type=Optional[ModelConfig], default=ModelConfig(data={"A": 1, "B": 2}))
    cfg = parser.parse_args(["--model.data.A=4"])
    assert cfg.model.data == {"A": 4, "B": 2}


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
    assert "--data.g3 g3          (required, type: union[str, int])" in help_str
    assert "--data.g4 g4          (required, type: dict[str, union[int, bool]])" in help_str


@dataclasses.dataclass
class SpecificData:
    y: GenericData[float]


def test_nested_generic_dataclass(parser):
    parser.add_dataclass_arguments(SpecificData, "x")
    help_str = get_parser_help(parser).lower()
    assert "--x.y.g1 g1          (required, type: float)" in help_str
    assert "--x.y.g2 [item,...]  (required, type: tuple[float, float])" in help_str
    assert "--x.y.g3 g3          (required, type: union[str, float])" in help_str
    assert "--x.y.g4 g4          (required, type: dict[str, union[float, bool]])" in help_str


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


# pydantic tests

annotated = typing_extensions_import("Annotated")
length = "length"
if pydantic_support:
    import pydantic

    @pydantic.dataclasses.dataclass
    class PydanticData:
        p1: float = 0.1
        p2: str = "-"

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
            p1: annotated[int, pydantic.Field(default=2, ge=1, le=8)]  # type: ignore


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
        dump = yaml.safe_load(parser.dump(cfg))
        assert dump == {"model": {"param": valid_value}}
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args([f"--model.param={invalid_value}"])
        ctx.match("model.param")


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
