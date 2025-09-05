from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Literal, Optional, Union

import pytest

from jsonargparse import ArgumentError, ArgumentParser, Namespace
from jsonargparse._optionals import (
    docstring_parser_support,
    pydantic_support,
    pydantic_supports_field_init,
    typing_extensions_import,
)
from jsonargparse_tests.conftest import (
    get_parser_help,
    json_or_yaml_load,
)

if pydantic_support:
    import pydantic

annotated = typing_extensions_import("Annotated")

skip_if_pydantic_v1_on_v2 = pytest.mark.skipif(
    pydantic_support and pydantic is getattr(__import__("pydantic"), "v1", None),
    reason="Not supported for pydantic.v1",
)


@pytest.fixture(autouse=True)
def missing_pydantic():
    if not pydantic_support:
        pytest.skip("pydantic package is required")


@skip_if_pydantic_v1_on_v2
def test_pydantic_secret_str(parser):
    parser.add_argument("--password", type=pydantic.SecretStr)
    cfg = parser.parse_args(["--password=secret"])
    assert isinstance(cfg.password, pydantic.SecretStr)
    assert cfg.password.get_secret_value() == "secret"
    assert "secret" not in parser.dump(cfg)


if annotated and pydantic_support > 1:

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

    class PingTask(pydantic.BaseModel):
        type: Literal["ping"] = "ping"
        attr: str = ""

    class PongTask(pydantic.BaseModel):
        type: Literal["pong"] = "pong"

    PingPongTask = annotated[
        Union[PingTask, PongTask],
        pydantic.Field(discriminator="type"),
    ]

    class PydanticAnnotatedField(pydantic.BaseModel):
        p1: annotated[int, pydantic.Field(default=2, ge=1, le=8)]  # type: ignore[valid-type]


@pytest.mark.skipif(not (annotated and pydantic_support > 1), reason="Annotated is required")
class TestPydantic2Annotated:

    def test_pydantic_nested_annotated_dataclass(self, parser: ArgumentParser):
        parser.add_class_arguments(NestedAnnotatedDataClass, "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass(self, parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClass, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass_with_default(self, parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClassWithDefault, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_pydantic_annotated_nested_annotated_dataclass_with_default_factory(self, parser: ArgumentParser):
        parser.add_class_arguments(annotated[NestedAnnotatedDataClassWithDefaultFactory, 1], "n")
        cfg = parser.parse_args(["--n", "{}"])
        assert cfg.n == Namespace(a1=Namespace(a2=1))

    def test_annotated_field(self, parser):
        parser.add_argument("--model", type=PydanticAnnotatedField)
        cfg = parser.parse_args([])
        assert cfg.model.p1 == 2
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args(["--model.p1=0"])
        ctx.match("model.p1")

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


length = "items" if pydantic_support == 1 else "length"

if pydantic_support:

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


class TestPydanticBasics:
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
    @skip_if_pydantic_v1_on_v2
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
