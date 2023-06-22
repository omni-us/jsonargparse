from __future__ import annotations  # keep

import sys
from random import Random
from typing import Dict, FrozenSet, List, Set, Tuple, Type, Union

import pytest

from jsonargparse._backports import get_types
from jsonargparse._parameter_resolvers import get_signature_parameters as get_params
from jsonargparse_tests.conftest import capture_logs, source_unavailable


@pytest.fixture(autouse=True)
def skip_if_python_older_than_3_10():
    if sys.version_info >= (3, 10, 0):
        pytest.skip("python<3.10 is required")


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


def test_get_types_source_unavailable(logger):
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


def function_missing_type(p1, p2: str | int):
    return p1


def test_get_types_missing_type():
    types = get_types(function_missing_type)
    assert types == {"p2": Union[str, int]}
