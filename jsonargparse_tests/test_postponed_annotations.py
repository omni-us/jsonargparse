from __future__ import annotations  # keep

import dataclasses
import os
import sys
from random import Random
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union

import pytest

from jsonargparse import Namespace
from jsonargparse._parameter_resolvers import get_signature_parameters as get_params
from jsonargparse._postponed_annotations import TypeCheckingVisitor, evaluate_postponed_annotations, get_types
from jsonargparse.typing import Path_drw
from jsonargparse_tests.conftest import capture_logs, source_unavailable


def function_pep585_dict(p1: dict[str, int], p2: dict[int, str] = {1: "a"}):
    return p1


def function_pep585_list(p1: list[str], p2: list[float] = [0.1, 2.3]):
    return p1


def function_pep585_set(p1: set[str], p2: set[int] = {1, 2}):
    return p1


def function_pep585_frozenset(p1: frozenset[str], p2: frozenset[int] = frozenset(range(3))):
    return p1


def function_pep585_tuple(p1: tuple[str, float], p2: tuple[int, ...] = (1, 2)):
    return p1


def function_pep585_type(p1: type[Random], p2: type[Random] = Random):
    return p1


@pytest.mark.skipif(sys.version_info >= (3, 9, 0), reason="python<3.9 is required")
@pytest.mark.parametrize(
    ["function", "expected"],
    [
        (function_pep585_dict, {"p1": Dict[str, int], "p2": Dict[int, str]}),
        (function_pep585_list, {"p1": List[str], "p2": List[float]}),
        (function_pep585_set, {"p1": Set[str], "p2": Set[int]}),
        (function_pep585_frozenset, {"p1": FrozenSet[str], "p2": FrozenSet[int]}),
        (function_pep585_tuple, {"p1": Tuple[str, float], "p2": Tuple[int, ...]}),
        (function_pep585_type, {"p1": Type[Random], "p2": Type[Random]}),
    ],
)
def test_get_types_pep585(function, expected):
    types = get_types(function)
    assert types == expected


def function_pep604(p1: str | None, p2: int | float | bool = 1):
    return p1


def test_get_types_pep604():
    types = get_types(function_pep604)
    assert types == {"p1": Union[str, None], "p2": Union[int, float, bool]}


@pytest.mark.skipif(sys.version_info >= (3, 9, 0), reason="python<3.9 is required")
def test_get_types_pep604_source_unavailable(logger):
    with source_unavailable(), pytest.raises(TypeError) as ctx, capture_logs(logger) as logs:
        get_types(function_pep604, logger)
    ctx.match("could not get source code")
    assert "Failed to parse to source code" in logs.getvalue()


class NeedsBackport:
    def __init__(self, p1: list | set):
        self.p1 = p1

    @staticmethod
    def static_method(p1: str | int):
        return p1

    @classmethod
    def class_method(cls, p1: float | None):
        return p1


@pytest.mark.parametrize(
    ["method", "expected"],
    [
        (NeedsBackport.__init__, {"p1": Union[list, set]}),
        (NeedsBackport.static_method, {"p1": Union[str, int]}),
        (NeedsBackport.class_method, {"p1": Union[float, None]}),
    ],
)
def test_get_types_methods(method, expected):
    types = get_types(method)
    assert types == expected


def function_forward_ref(cls: "NeedsBackport", p1: "int"):
    return cls


def test_get_types_forward_ref():
    types = get_types(function_forward_ref)
    assert types == {"cls": NeedsBackport, "p1": int}


def function_undefined_type(p1: not_defined | None, p2: int):  # type: ignore  # noqa: F821
    return p1


def test_get_types_undefined_type():
    types = get_types(function_undefined_type)
    assert types["p2"] is int
    assert isinstance(types["p1"], KeyError)
    assert "not_defined" in str(types["p1"])

    params = get_params(function_undefined_type)
    assert params[0].annotation == "not_defined | None"


def function_all_types_fail(p1: not_defined | None, p2: not_defined):  # type: ignore  # noqa: F821
    return p1


def test_get_types_all_types_fail():
    with pytest.raises(NameError) as ctx:
        get_types(function_all_types_fail)
    ctx.match("not_defined")


def test_evaluate_postponed_annotations_all_types_fail(logger):
    params = get_params(function_all_types_fail)
    with capture_logs(logger) as logs:
        evaluate_postponed_annotations(params, function_all_types_fail, None, logger)
    assert "Unable to evaluate types for " in logs.getvalue()


def function_missing_type(p1, p2: str | int):
    return p1


def test_get_types_missing_type():
    types = get_types(function_missing_type)
    assert types == {"p2": Union[str, int]}


type_checking_template = """
%(typing_import)s

if %(condition)s:
    SUCCESS = True
"""


@pytest.mark.parametrize(
    ["typing_import", "condition"],
    [
        ("from typing import TYPE_CHECKING", "TYPE_CHECKING and COND2 and COND3"),
        ("from typing import TYPE_CHECKING", "COND1 or COND2 or TYPE_CHECKING"),
        ("from typing import TYPE_CHECKING as TC", "TC"),
        ("import typing", "typing.TYPE_CHECKING"),
        ("import typing as t", "t.TYPE_CHECKING"),
    ],
)
def test_type_checking_visitor(typing_import, condition):
    source = type_checking_template % {"typing_import": typing_import, "condition": condition}
    visitor = TypeCheckingVisitor()
    aliases = {}
    visitor.update_aliases(source, __name__, aliases)
    assert aliases.get("SUCCESS") is True


type_checking_failure = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    INVALID += 1
"""


def test_type_checking_visitor_failure(logger):
    visitor = TypeCheckingVisitor()
    with capture_logs(logger) as logs:
        visitor.update_aliases(type_checking_failure, __name__, {}, logger)
    assert "Failed to execute 'TYPE_CHECKING' block" in logs.getvalue()


if TYPE_CHECKING:
    import xml.dom

    class TypeCheckingClass1:
        pass

    class TypeCheckingClass2:
        pass

    type_checking_alias = Union[int, TypeCheckingClass2, List[str]]


def function_type_checking_nested_attr(p1: str, p2: Optional["xml.dom.Node"]):
    return p1


def test_get_types_type_checking_nested_attr():
    types = get_types(function_type_checking_nested_attr)
    from xml.dom import Node

    assert types == {"p1": str, "p2": Optional[Node]}


def function_type_checking_union(p1: Union[bool, TypeCheckingClass1, int], p2: Union[float, "TypeCheckingClass2"]):
    return p1


def test_get_types_type_checking_union():
    types = get_types(function_type_checking_union)
    assert list(types.keys()) == ["p1", "p2"]
    assert str(types["p1"]) == f"typing.Union[bool, {__name__}.TypeCheckingClass1, int]"
    assert str(types["p2"]) == f"typing.Union[float, {__name__}.TypeCheckingClass2]"


def function_type_checking_alias(p1: type_checking_alias, p2: "type_checking_alias"):
    return p1


def test_get_types_type_checking_alias():
    types = get_types(function_type_checking_alias)
    assert list(types.keys()) == ["p1", "p2"]
    assert str(types["p1"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str]]"
    assert str(types["p2"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str]]"


def function_type_checking_optional_alias(p1: type_checking_alias | None, p2: Optional["type_checking_alias"]):
    return p1


def test_get_types_type_checking_optional_alias():
    types = get_types(function_type_checking_optional_alias)
    assert list(types.keys()) == ["p1", "p2"]
    assert str(types["p1"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str], NoneType]"
    assert str(types["p2"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str], NoneType]"


def function_type_checking_list(p1: List[Union["TypeCheckingClass1", TypeCheckingClass2]]):
    return p1


def test_get_types_type_checking_list():
    types = get_types(function_type_checking_list)
    assert list(types.keys()) == ["p1"]
    lst = "typing.List" if sys.version_info < (3, 10) else "list"
    assert str(types["p1"]) == f"{lst}[typing.Union[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]]"


def function_type_checking_tuple(p1: Tuple[TypeCheckingClass1, "TypeCheckingClass2"]):
    return p1


def test_get_types_type_checking_tuple():
    types = get_types(function_type_checking_tuple)
    assert list(types.keys()) == ["p1"]
    tpl = "typing.Tuple" if sys.version_info < (3, 10) else "tuple"
    assert str(types["p1"]) == f"{tpl}[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]"


def function_type_checking_dict(p1: Dict[str, Union[TypeCheckingClass1, "TypeCheckingClass2"]]):
    return p1


def test_get_types_type_checking_dict():
    types = get_types(function_type_checking_dict)
    assert list(types.keys()) == ["p1"]
    dct = "typing.Dict" if sys.version_info < (3, 10) else "dict"
    assert str(types["p1"]) == f"{dct}[str, typing.Union[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]]"


def function_type_checking_undefined_forward_ref(p1: List["Undefined"], p2: bool):  # type: ignore  # noqa: F821
    return p1


def test_get_types_type_checking_undefined_forward_ref(logger):
    with capture_logs(logger) as logs:
        types = get_types(function_type_checking_undefined_forward_ref, logger)
    assert types == {"p1": List["Undefined"], "p2": bool}  # noqa: F821
    assert "Failed to resolve forward refs in " in logs.getvalue()
    assert "NameError: Name 'Undefined' is not defined" in logs.getvalue()


def function_source_unavailable(p1: List["TypeCheckingClass1"]):
    return p1


@pytest.mark.skipif(sys.version_info >= (3, 9, 0), reason="python<3.9 is required")
def test_get_types_source_unavailable():
    with source_unavailable():
        types = get_types(function_source_unavailable)
    assert types == {"p1": List["TypeCheckingClass1"]}


@dataclasses.dataclass
class Data585:
    a: list[int]
    b: str = "x"


def test_get_types_dataclass_pep585(parser):
    types = get_types(Data585)
    list_int = List[int] if sys.version_info < (3, 9) else list[int]
    assert types == {"a": list_int, "b": str}
    parser.add_dataclass_arguments(Data585, "data")
    cfg = parser.parse_args(["--data.a=[1, 2]"])
    assert cfg.data == Namespace(a=[1, 2], b="x")


@dataclasses.dataclass
class DataWithInit585(Data585):
    def __init__(self, b: Path_drw, **kwargs):  # type: ignore
        super().__init__(b=os.fspath(b), **kwargs)


def test_add_dataclass_with_init_pep585(parser, tmp_cwd):
    parser.add_dataclass_arguments(DataWithInit585, "data")
    cfg = parser.parse_args(["--data.a=[1, 2]", "--data.b=."])
    assert cfg.data == Namespace(a=[1, 2], b=Path_drw("."))
