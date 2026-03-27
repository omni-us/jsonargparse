from __future__ import annotations  # keep

import dataclasses
import importlib.util
import os
import sys
from textwrap import dedent
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, ForwardRef, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import pytest

from jsonargparse import Namespace
from jsonargparse import _postponed_annotations as postponed_annotations
from jsonargparse._optionals import docstring_parser_support
from jsonargparse._parameter_resolvers import get_signature_parameters as get_params
from jsonargparse._postponed_annotations import (
    _TRIGGER_MODULE_CACHE,
    TypeCheckingVisitor,
    _cache_trigger_module_name,
    _collect_string_fwd_ref_names,
    _enrich_globals_for_string_forward_refs,
    evaluate_postponed_annotations,
    get_global_vars,
    get_types,
    type_requires_eval,
)
from jsonargparse.typing import Path_drw
from jsonargparse_tests.conftest import capture_logs, source_unavailable
from jsonargparse_tests.test_dataclasses import DifferentModuleBaseData


def function_pep604(p1: str | None, p2: int | float | bool = 1):
    return p1


def test_get_types_pep604():
    types = get_types(function_pep604)
    assert types == {"p1": Union[str, None], "p2": Union[int, float, bool]}


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
    assert list(types) == ["p1", "p2"]
    if sys.version_info < (3, 14):
        assert str(types["p1"]) == f"typing.Union[bool, {__name__}.TypeCheckingClass1, int]"
        assert str(types["p2"]) == f"typing.Union[float, {__name__}.TypeCheckingClass2]"
    else:
        assert str(types["p1"]) == f"bool | {__name__}.TypeCheckingClass1 | int"
        assert str(types["p2"]) == f"float | {__name__}.TypeCheckingClass2"


def function_type_checking_alias(p1: type_checking_alias, p2: "type_checking_alias"):
    return p1


def test_get_types_type_checking_alias():
    types = get_types(function_type_checking_alias)
    assert list(types) == ["p1", "p2"]
    if sys.version_info < (3, 14):
        assert str(types["p1"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str]]"
        assert str(types["p2"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str]]"
    else:
        assert str(types["p1"]) == f"int | {__name__}.TypeCheckingClass2 | typing.List[str]"
        assert str(types["p2"]) == f"int | {__name__}.TypeCheckingClass2 | typing.List[str]"


def function_type_checking_optional_alias(p1: type_checking_alias | None, p2: Optional["type_checking_alias"]):
    return p1


def test_get_types_type_checking_optional_alias():
    types = get_types(function_type_checking_optional_alias)
    assert list(types) == ["p1", "p2"]
    if sys.version_info < (3, 14):
        assert str(types["p1"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str], NoneType]"
        assert str(types["p2"]) == f"typing.Union[int, {__name__}.TypeCheckingClass2, typing.List[str], NoneType]"
    else:
        assert str(types["p1"]) == f"int | {__name__}.TypeCheckingClass2 | typing.List[str] | None"
        assert str(types["p2"]) == f"int | {__name__}.TypeCheckingClass2 | typing.List[str] | None"


def function_type_checking_list(p1: List[Union["TypeCheckingClass1", TypeCheckingClass2]]):
    return p1


def test_get_types_type_checking_list():
    types = get_types(function_type_checking_list)
    assert list(types) == ["p1"]
    lst = "typing.List"
    if sys.version_info < (3, 14):
        assert str(types["p1"]) == f"{lst}[typing.Union[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]]"
    else:
        assert str(types["p1"]) == f"{lst}[{__name__}.TypeCheckingClass1 | {__name__}.TypeCheckingClass2]"


def function_type_checking_tuple(p1: Tuple[TypeCheckingClass1, "TypeCheckingClass2"]):
    return p1


def test_get_types_type_checking_tuple():
    types = get_types(function_type_checking_tuple)
    assert list(types) == ["p1"]
    tpl = "typing.Tuple"
    assert str(types["p1"]) == f"{tpl}[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]"


def function_type_checking_type(p1: Type["TypeCheckingClass2"]):
    return p1


def test_get_types_type_checking_type():
    types = get_types(function_type_checking_type)
    assert list(types) == ["p1"]
    tpl = "typing.Type"
    assert str(types["p1"]) == f"{tpl}[{__name__}.TypeCheckingClass2]"


def function_type_checking_dict(p1: Dict[str, Union[TypeCheckingClass1, "TypeCheckingClass2"]]):
    return p1


def test_get_types_type_checking_dict():
    types = get_types(function_type_checking_dict)
    assert list(types) == ["p1"]
    dct = "typing.Dict"
    if sys.version_info < (3, 14):
        assert (
            str(types["p1"])
            == f"{dct}[str, typing.Union[{__name__}.TypeCheckingClass1, {__name__}.TypeCheckingClass2]]"
        )
    else:
        assert str(types["p1"]) == f"{dct}[str, {__name__}.TypeCheckingClass1 | {__name__}.TypeCheckingClass2]"


def function_type_checking_undefined_forward_ref(p1: List["Undefined"], p2: bool):  # type: ignore  # noqa: F821
    return p1


def test_get_types_type_checking_undefined_forward_ref(logger):
    with capture_logs(logger) as logs:
        types = get_types(function_type_checking_undefined_forward_ref, logger)
    assert types == {"p1": List["Undefined"], "p2": bool}  # noqa: F821
    assert "Failed to resolve forward refs in " in logs.getvalue()
    assert "NameError: Name 'Undefined' is not defined" in logs.getvalue()


@dataclasses.dataclass
class DataclassForwardRef:
    p1: "int"
    p2: Optional["xml.dom.Node"] = None


def test_get_types_type_checking_dataclass_init_forward_ref():
    import xml.dom

    types = get_types(DataclassForwardRef.__init__)
    assert types == {"p1": int, "p2": Optional[xml.dom.Node], "return": type(None)}


def function_source_unavailable(p1: List["TypeCheckingClass1"]):
    return p1


def test_get_types_source_unavailable(logger):
    with source_unavailable(function_source_unavailable), pytest.raises(NameError) as ctx, capture_logs(logger) as logs:
        get_types(function_source_unavailable, logger)
    ctx.match("'TypeCheckingClass1' is not defined")
    assert "source code not available" in logs.getvalue()


@dataclasses.dataclass
class Data585:
    a: list[int]
    b: str = "x"


def test_get_types_dataclass_pep585(parser):
    types = get_types(Data585)
    assert types == {"a": list[int], "b": str}
    parser.add_class_arguments(Data585, "data")
    cfg = parser.parse_args(["--data.a=[1, 2]"])
    assert cfg.data == Namespace(a=[1, 2], b="x")


@dataclasses.dataclass
class DataWithInit585(Data585):
    def __init__(self, b: Path_drw, **kwargs):
        super().__init__(b=os.fspath(b), **kwargs)


def test_add_dataclass_with_init_pep585(parser, tmp_cwd):
    parser.add_class_arguments(DataWithInit585, "data")
    cfg = parser.parse_args(["--data.a=[1, 2]", "--data.b=."])
    assert cfg.data == Namespace(a=[1, 2], b=Path_drw("."))


@dataclasses.dataclass
class InheritDifferentModule(DifferentModuleBaseData):
    """
    Args:
        extra: an extra string
    """

    extra: str = "default"


def test_get_params_dataclass_inherit_different_module():
    assert "BetweenThreeAndNine" not in globals()
    assert "PositiveInt" not in globals()

    params = get_params(InheritDifferentModule)

    assert [p.name for p in params] == ["count", "numbers", "extra"]
    if docstring_parser_support:
        assert [p.doc for p in params] == ["between 3 and 9", "list of positive ints", "an extra string"]
    assert all(not isinstance(p.annotation, str) for p in params)
    assert not isinstance(params[0].annotation.__args__[0], str)
    assert "BetweenThreeAndNine" in str(params[0].annotation)
    assert not isinstance(params[1].annotation.__args__[0], str)
    assert "PositiveInt" in str(params[1].annotation)


def test_get_global_vars_ignores_type_checking_source_errors(monkeypatch):
    monkeypatch.setattr(
        postponed_annotations.inspect, "getsource", lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    global_vars = get_global_vars(function_type_checking_alias, None)
    assert global_vars["function_type_checking_alias"] is function_type_checking_alias


@pytest.fixture
def fwdref_origin_mod(tmp_path):
    """Module A: defines ForwardReferenced and NamedType = list['ForwardReferenced']."""
    types_module_path = tmp_path / "fwdref_types_module.py"
    types_module_path.write_text(
        dedent(
            """\
            class ForwardReferenced:
                pass
            NamedType = list['ForwardReferenced']
            """
        )
    )
    spec = importlib.util.spec_from_file_location("fwdref_types_module", types_module_path)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"fwdref_types_module": mod}):
        spec.loader.exec_module(mod)
        yield mod


class TestForwardReference:
    def setup_method(self):
        _TRIGGER_MODULE_CACHE.clear()

    @staticmethod
    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_forward_ref_resolved_from_alias_origin_module(self, parser, tmp_path, fwdref_origin_mod):
        """Indirect: ForwardReferenced NOT imported and resolved from alias origin module."""
        indirect_path = tmp_path / "fwdref_indirect_module.py"
        indirect_path.write_text(
            dedent(
                """\
                from fwdref_types_module import NamedType

                class Indirect:
                    def __init__(self, data_type: NamedType):
                        pass
                """
            )
        )
        mod = self._load_module("fwdref_indirect_module", indirect_path)
        with patch.dict(sys.modules, {"fwdref_indirect_module": mod}):
            parser.add_class_arguments(mod.Indirect)
            types = get_types(mod.Indirect.__init__)
            assert not type_requires_eval(types["data_type"])
            assert "ForwardReferenced" in str(types["data_type"])

    def test_forward_ref_resolved_for_aliased_import(self, parser, tmp_path, fwdref_origin_mod):
        """Aliased: alias imported under a different local name and still resolved."""
        aliased_path = tmp_path / "fwdref_aliased_module.py"
        aliased_path.write_text(
            dedent(
                """\
                from fwdref_types_module import NamedType as NT

                class Aliased:
                    def __init__(self, data_type: NT):
                        pass
                """
            )
        )
        mod = self._load_module("fwdref_aliased_module", aliased_path)
        with patch.dict(sys.modules, {"fwdref_aliased_module": mod}):
            parser.add_class_arguments(mod.Aliased)
            types = get_types(mod.Aliased.__init__)
            assert not type_requires_eval(types["data_type"])
            assert "ForwardReferenced" in str(types["data_type"])


class TestEnrichGlobals:
    def setup_method(self):
        _TRIGGER_MODULE_CACHE.clear()

    def test_cache_trigger_module_name_evicts_oldest_trigger(self, monkeypatch):
        """Cache evicts the oldest trigger when inserting beyond the configured size."""
        monkeypatch.setattr(postponed_annotations, "_TRIGGER_MODULE_CACHE_MAXSIZE", 2)

        _cache_trigger_module_name(1, "module_a")
        _cache_trigger_module_name(2, "module_b")
        _cache_trigger_module_name(3, "module_c")

        assert list(_TRIGGER_MODULE_CACHE) == [2, 3]
        assert _TRIGGER_MODULE_CACHE[2] == ["module_b"]
        assert _TRIGGER_MODULE_CACHE[3] == ["module_c"]

    def test_cache_trigger_module_name_deduplicates_module_names(self):
        """Repeated discoveries for the same trigger/module pair are stored once."""
        _cache_trigger_module_name(1, "module_a")
        _cache_trigger_module_name(1, "module_a")
        _cache_trigger_module_name(1, "module_b")

        assert _TRIGGER_MODULE_CACHE[1] == ["module_a", "module_b"]

    def test_resolves_missing_fwd_ref(self, fwdref_origin_mod):
        """Missing forward-ref name is injected from the alias origin module."""
        global_vars = {"NT": fwdref_origin_mod.NamedType}
        _enrich_globals_for_string_forward_refs(global_vars)
        assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_no_overwrite_existing(self, fwdref_origin_mod):
        """Already-present binding is not overwritten by enrichment."""
        sentinel = object()
        global_vars = {"NT": fwdref_origin_mod.NamedType, "ForwardReferenced": sentinel}
        _enrich_globals_for_string_forward_refs(global_vars)
        assert global_vars["ForwardReferenced"] is sentinel

    def test_handles_non_module_sys_entries(self, fwdref_origin_mod):
        """None and non-module entries in sys.modules do not cause errors."""
        with patch.dict(sys.modules, {"_null_sys_entry": None, "_obj_sys_entry": object()}):
            global_vars = {"NT": fwdref_origin_mod.NamedType}
            _enrich_globals_for_string_forward_refs(global_vars)
            assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_collect_string_fwd_ref_names_supports_forwardref_instances(self):
        """Direct ForwardRef objects contribute their root name."""
        names = set()
        _collect_string_fwd_ref_names(ForwardRef("pkg.ForwardReferenced"), names)
        assert names == {"pkg"}

    def test_cached_module_names_are_deduplicated_across_triggers(self, monkeypatch, tmp_path):
        """Duplicate cached module names from multiple trigger aliases are processed once."""
        multi_alias_path = tmp_path / "fwdref_multi_alias_module.py"
        multi_alias_path.write_text(
            dedent(
                """\
                class ForwardReferencedA:
                    pass
                class ForwardReferencedB:
                    pass
                NamedType = list['ForwardReferencedA']
                OtherType = dict[str, 'ForwardReferencedB']
                """
            )
        )
        spec = importlib.util.spec_from_file_location("fwdref_multi_alias_module", multi_alias_path)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fwdref_multi_alias_module": mod}):
            spec.loader.exec_module(mod)
            _TRIGGER_MODULE_CACHE[id(mod.NamedType)] = ["fwdref_multi_alias_module"]
            _TRIGGER_MODULE_CACHE[id(mod.OtherType)] = ["fwdref_multi_alias_module"]

            calls = []
            original = postponed_annotations._update_missing_from_module_vars

            def wrapped(global_vars, missing, mod_vars):
                calls.append(sorted(missing))
                return original(global_vars, missing, mod_vars)

            monkeypatch.setattr(postponed_annotations, "_update_missing_from_module_vars", wrapped)

            global_vars = {"NT": mod.NamedType, "TO": mod.OtherType}
            _enrich_globals_for_string_forward_refs(global_vars)

        assert global_vars["ForwardReferencedA"] is mod.ForwardReferencedA
        assert global_vars["ForwardReferencedB"] is mod.ForwardReferencedB
        assert calls == [["ForwardReferencedA", "ForwardReferencedB"]]

    def test_cached_missing_module_name_falls_back_to_scan(self, fwdref_origin_mod):
        """A stale cache entry does not block the later sys.modules scan."""
        _TRIGGER_MODULE_CACHE[id(fwdref_origin_mod.NamedType)] = ["missing_fwdref_module"]

        global_vars = {"NT": fwdref_origin_mod.NamedType}
        _enrich_globals_for_string_forward_refs(global_vars)

        assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_reuses_cached_modules_before_scanning_sys_modules(self, monkeypatch, fwdref_origin_mod):
        """A warm cache avoids a second full sys.modules scan for the same trigger alias."""
        _enrich_globals_for_string_forward_refs({"NT": fwdref_origin_mod.NamedType})

        class NoScanModules(dict):
            def items(self):
                raise AssertionError("sys.modules should not be scanned when the trigger cache is warm")

        monkeypatch.setattr(
            postponed_annotations,
            "sys",
            SimpleNamespace(modules=NoScanModules({"fwdref_types_module": fwdref_origin_mod})),
        )

        global_vars = {"NT": fwdref_origin_mod.NamedType}
        _enrich_globals_for_string_forward_refs(global_vars)
        assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_ignores_non_module_cached_entries(self, fwdref_origin_mod):
        """Cached entries that are not modules do not break fallback scanning."""
        _TRIGGER_MODULE_CACHE[id(fwdref_origin_mod.NamedType)] = ["_obj_sys_entry"]

        with patch.dict(sys.modules, {"_obj_sys_entry": object()}):
            global_vars = {"NT": fwdref_origin_mod.NamedType}
            _enrich_globals_for_string_forward_refs(global_vars)

        assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_scan_skips_non_modules_before_reaching_origin_module(self, monkeypatch, fwdref_origin_mod):
        """The fallback scan ignores entries without a module dict and keeps searching."""
        fake_sys = SimpleNamespace(modules={"_obj_sys_entry": object(), "fwdref_types_module": fwdref_origin_mod})
        monkeypatch.setattr(postponed_annotations, "sys", fake_sys)

        global_vars = {"NT": fwdref_origin_mod.NamedType}
        _enrich_globals_for_string_forward_refs(global_vars)

        assert global_vars["ForwardReferenced"] is fwdref_origin_mod.ForwardReferenced

    def test_resolves_nested_generic_alias(self, tmp_path):
        """Recursive collection resolves names nested two levels deep (list[list['X']])."""
        nested_path = tmp_path / "fwdref_nested_module.py"
        nested_path.write_text(
            dedent(
                """\
                class Inner:
                    pass
                NestedType = list[list['Inner']]
                """
            )
        )
        spec = importlib.util.spec_from_file_location("fwdref_nested_module", nested_path)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fwdref_nested_module": mod}):
            spec.loader.exec_module(mod)
            global_vars = {"NT": mod.NestedType}
            _enrich_globals_for_string_forward_refs(global_vars)
            assert global_vars["Inner"] is mod.Inner
