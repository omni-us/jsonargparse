from __future__ import annotations

import asyncio
import os
import sys
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Literal, Optional
from unittest.mock import patch

import pytest
import yaml

from jsonargparse import CLI, capture_parser, lazy_instance
from jsonargparse._optionals import docstring_parser_support, ruyaml_support
from jsonargparse.typing import final
from jsonargparse_tests.conftest import skip_if_docstring_parser_unavailable


def get_cli_stdout(*args, **kwargs) -> str:
    out = StringIO()
    with redirect_stdout(out), suppress(SystemExit), patch.dict(os.environ, {"COLUMNS": "150"}):
        CLI(*args, **kwargs)
    return out.getvalue()


# failure cases


@pytest.mark.parametrize("components", [0, [], {"x": 0}])
def test_unexpected_components(components):
    with pytest.raises(ValueError):
        CLI(components)


class ConflictingSubcommandKey:
    def subcommand(self, x: int = 0):
        return x


def test_conflicting_subcommand_key():
    with pytest.raises(ValueError) as ctx:
        CLI(ConflictingSubcommandKey, args=["subcommand", "--x=1"])
    assert ctx.match("subcommand name can't be the same")


# single function tests


def single_function(a1: float):
    """Description of single_function"""
    return a1


def test_single_function_return():
    assert 1.2 == CLI(single_function, args=["1.2"])


def test_single_function_set_defaults():
    def run_cli():
        CLI(single_function, set_defaults={"a1": 3.4})

    parser = capture_parser(run_cli)
    assert 3.4 == parser.get_defaults().a1


def test_single_function_help():
    out = get_cli_stdout(single_function, args=["--help"])
    assert "a1" in out
    if docstring_parser_support:
        assert "Description of single_function" in out
    else:
        assert "function single_function" in out


# callable class tests


class CallableClass:
    """Description of CallableClass"""

    def __call__(self, x: int):
        return x


callable_instance = CallableClass()


def test_callable_instance():
    assert 3 == CLI(callable_instance, as_positional=False, args=["--x=3"])


# multiple functions tests


def cmd1(a1: int):
    """Description of cmd1"""
    return a1


def cmd2(a2: str = "X"):
    return a2


def test_multiple_functions_return():
    assert 5 == CLI([cmd1, cmd2], args=["cmd1", "5"])
    assert "Y" == CLI([cmd1, cmd2], args=["cmd2", "--a2=Y"])


def test_multiple_functions_set_defaults():
    def run_cli():
        CLI([cmd1, cmd2], set_defaults={"cmd2.a2": "Z"})

    parser = capture_parser(run_cli)
    assert "Z" == parser.parse_args(["cmd2"]).cmd2.a2


def test_multiple_functions_main_help():
    out = get_cli_stdout([cmd1, cmd2], args=["--help"])
    assert "{cmd1,cmd2}" in out


def test_multiple_functions_subcommand_help():
    out = get_cli_stdout([cmd1, cmd2], args=["cmd2", "--help"])
    assert "--a2 A2" in out


def conditionalA(foo: int = 1):
    return foo


def conditionalB(bar: int = 2):
    return bar


def conditional_function(fn: "Literal['A', 'B']", *args, **kwargs):
    if fn == "A":
        return conditionalA(*args, **kwargs)
    elif fn == "B":
        return conditionalB(*args, **kwargs)
    raise NotImplementedError(fn)


@pytest.mark.skipif(condition=sys.version_info < (3, 9), reason="python>=3.9 is required")
def test_literal_conditional_function():
    out = get_cli_stdout(conditional_function, args=["--help"])
    assert "Conditional arguments" in out
    assert "--foo FOO             (type: int, default: Conditional<ast-resolver> {1, NOT_ACCEPTED})" in out
    assert "--bar BAR             (type: int, default: Conditional<ast-resolver> {2, NOT_ACCEPTED})" in out


# single class tests


class Class1:
    """Description of Class1"""

    def __init__(self, i1: str):
        self.i1 = i1

    def method1(self, m1: int):
        """Description of method1"""
        return self.i1, m1


def test_single_class_return():
    assert ("0", 2) == CLI(Class1, args=["0", "method1", "2"])
    assert ("3", 4) == CLI(Class1, args=['--config={"i1": "3", "method1": {"m1": 4}}'])
    assert ("5", 6) == CLI(Class1, args=["5", "method1", '--config={"m1": 6}'])


def test_single_class_missing_required_init():
    err = StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit):
        CLI(Class1, args=['--config={"method1": {"m1": 2}}'])
    assert '"i1" is required' in err.getvalue()


def test_single_class_invalid_method_parameter():
    err = StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit):
        CLI(Class1, args=['--config={"i1": "0", "method1": {"m1": "A"}}'])
    assert 'key "m1"' in err.getvalue()


def test_single_class_main_help():
    out = get_cli_stdout(Class1, args=["--help"])
    assert " i1" in out
    if docstring_parser_support:
        assert "Description of Class1" in out
        assert "Description of method1" in out
    else:
        assert "function Class1.method1" in out


def test_single_class_subcommand_help():
    out = get_cli_stdout(Class1, args=["x", "method1", "--help"])
    assert " m1" in out
    if docstring_parser_support:
        assert "Description of method1" in out


@skip_if_docstring_parser_unavailable
def test_single_class_help_docstring_parse_error():
    with patch("docstring_parser.parse") as docstring_parse:
        from docstring_parser import ParseError

        docstring_parse.side_effect = ParseError
        out = get_cli_stdout(Class1, args=["x", "method1", "--help"])
        assert "Description of method1" not in out


def test_single_class_print_config_after_subcommand():
    out = get_cli_stdout(Class1, args=["0", "method1", "2", "--print_config"])
    assert "m1: 2" == out.strip()


def test_single_class_print_config_before_subcommand():
    out = get_cli_stdout(Class1, args=["--print_config", "0", "method1", "2"])
    cfg = yaml.safe_load(out)
    assert cfg == {"i1": "0", "method1": {"m1": 2}}


class MethodWithConfigParam:
    def cmd(self, p1: int = 1, config: dict = {}):
        print(f"p1: {p1}, config: {config}")


def test_method_with_config_parameter():
    out = get_cli_stdout(MethodWithConfigParam, args=["cmd"])
    assert "p1: 1, config: {}" == out.strip()


# function and class tests


class Cmd2:
    def __init__(self, i1: str = "d"):
        """Description of Cmd2"""
        self.i1 = i1

    def method1(self, m1: float):
        return self.i1, m1

    def method2(self, m2: int = 0):
        """Description of method2"""
        return self.i1, m2

    def method3(self):
        return "Cmd2.method3"


def cmd3():
    return "cmd3"


@pytest.mark.parametrize(
    ["expected", "args"],
    [
        (5, ["cmd1", "5"]),
        (("d", 1.2), ["Cmd2", "method1", "1.2"]),
        (("b", 3), ["Cmd2", "--i1=b", "method2", "--m2=3"]),
        (4, ['--config={"cmd1": {"a1": 4}}']),
        (("a", 4.5), ['--config={"Cmd2": {"i1": "a", "method1": {"m1": 4.5}}}']),
        (("c", 6.7), ["Cmd2", "--i1=c", "method1", '--config={"m1": 6.7}']),
        (("d", 8.9), ["Cmd2", '--config={"method1": {"m1": 8.9}}']),
        ("Cmd2.method3", ["Cmd2", "method3"]),
        ("cmd3", ["cmd3"]),
    ],
)
def test_function_and_class_return(expected, args):
    assert expected == CLI([cmd1, Cmd2, cmd3], args=args)


def test_function_and_class_main_help():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["--help"])
    assert "{cmd1,Cmd2,cmd3}" in out
    assert "function cmd3" in out
    if docstring_parser_support:
        assert "Description of cmd1" in out
        assert "Description of Cmd2" in out
    else:
        assert "function cmd1" in out
        assert ".test_cli.Cmd2" in out


def test_function_and_class_subcommand_help():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["Cmd2", "--help"])
    assert "{method1,method2,method3}" in out
    if docstring_parser_support:
        assert "Description of Cmd2:" in out
        assert "Description of method2" in out
    else:
        assert "function Cmd2.method2" in out


def test_function_and_class_subsubcommand_help():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["Cmd2", "method2", "--help"])
    assert "--m2 M2" in out
    if docstring_parser_support:
        assert "Description of method2" in out


def test_function_and_class_print_config_after_subsubcommand():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["Cmd2", "method2", "--print_config"])
    assert "m2: 0\n" == out


def test_function_and_class_print_config_in_between_subcommands():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["Cmd2", "--print_config", "method2"])
    assert "i1: d\nmethod2:\n  m2: 0\n" == out


def test_function_and_class_print_config_before_subcommands():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["--print_config", "Cmd2", "method2"])
    assert "Cmd2:\n  i1: d\n  method2:\n    m2: 0\n" == out


@skip_if_docstring_parser_unavailable
@pytest.mark.skipif(not ruyaml_support, reason="ruyaml not installed")
def test_function_and_class_print_config_comments():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["--print_config=comments", "Cmd2", "method2"])
    assert "# Description of Cmd2" in out
    assert "# Description of method2" in out


def test_function_and_class_method_without_parameters():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["Cmd2", "method3", "--help"])
    assert "--config" not in out


def test_function_and_class_function_without_parameters():
    out = get_cli_stdout([cmd1, Cmd2, cmd3], args=["cmd3", "--help"])
    assert "--config" not in out


# automatic components tests


def test_automatic_components_empty_context():
    def empty_context():
        CLI()

    with patch("inspect.getmodule") as mock_getmodule:
        mock_getmodule.return_value = sys.modules["jsonargparse._core"]
        pytest.raises(ValueError, empty_context)


def test_automatic_components_context_function():
    def non_empty_context_function():
        def function(a1: float):
            return a1

        return CLI(args=["6.7"])

    with patch("inspect.getmodule") as mock_getmodule:
        mock_getmodule.return_value = sys.modules["jsonargparse._core"]
        assert 6.7 == non_empty_context_function()


def test_automatic_components_context_class():
    def non_empty_context_class():
        class ClassX:
            def __init__(self, i1: str):
                self.i1 = i1

            def method(self, m1: int):
                return self.i1, m1

        return CLI(args=["a", "method", "2"])

    with patch("inspect.getmodule") as mock_getmodule:
        mock_getmodule.return_value = sys.modules["jsonargparse._core"]
        assert ("a", 2) == non_empty_context_class()


# class without methods tests


@dataclass
class SettingsClass:
    p1: str
    p2: int = 3


def test_dataclass_without_methods_response():
    settings = CLI(SettingsClass, args=["--p1=x", "--p2=0"], as_positional=False)
    assert isinstance(settings, SettingsClass)
    assert asdict(settings) == {"p1": "x", "p2": 0}


def test_dataclass_without_methods_parser_groups():
    parser = capture_parser(lambda: CLI(SettingsClass, args=[], as_positional=False))
    assert parser.groups == {}


# named components tests


def test_named_components_shallow():
    components = {"cmd1": single_function, "cmd2": callable_instance}
    assert 3.4 == CLI(components, args=["cmd1", "3.4"])
    assert 5 == CLI(components, as_positional=False, args=["cmd2", "--x=5"])
    out = get_cli_stdout(components, args=["--help"])
    if docstring_parser_support:
        assert "Description of single_function" in out
        assert "Description of CallableClass" in out


def test_named_components_deep():
    components = {
        "lv1_a": {
            "_help": "Description of lv1_a",
            "lv2_x": single_function,
            "lv2_y": {"_help": "Description of lv2_y", "lv3_p": callable_instance},
        },
        "lv1_b": {
            "_help": "Description of lv1_b",
            "lv2_z": {
                "_help": "Description of lv2_z",
                "lv3_q": Class1,
            },
        },
    }
    kw = {"as_positional": False}
    out = get_cli_stdout(components, args=["--help"], **kw)
    assert " {lv1_a,lv1_b} ..." in out
    assert "Description of lv1_a" in out
    assert "Description of lv1_b" in out
    out = get_cli_stdout(components, args=["lv1_a", "--help"], **kw)
    assert " {lv2_x,lv2_y} ..." in out
    assert "Description of lv1_a" in out
    if docstring_parser_support:
        assert "Description of single_function" in out
    assert "Description of lv2_y" in out
    out = get_cli_stdout(components, args=["lv1_a", "lv2_x", "--help"], **kw)
    assert " --a1 A1" in out
    if docstring_parser_support:
        assert "Description of single_function" in out
    out = get_cli_stdout(components, args=["lv1_a", "lv2_y", "--help"], **kw)
    assert " {lv3_p} ..." in out
    assert "Description of lv2_y" in out
    if docstring_parser_support:
        assert "Description of CallableClass" in out
    out = get_cli_stdout(components, args=["lv1_a", "lv2_y", "lv3_p", "--help"], **kw)
    assert " --x X" in out
    if docstring_parser_support:
        assert "Description of CallableClass" in out
    out = get_cli_stdout(components, args=["lv1_b", "--help"], **kw)
    assert " {lv2_z} ..." in out
    assert "Description of lv1_b" in out
    assert "Description of lv2_z" in out
    out = get_cli_stdout(components, args=["lv1_b", "lv2_z", "--help"], **kw)
    assert " {lv3_q} ..." in out
    assert "Description of lv2_z" in out
    if docstring_parser_support:
        assert "Description of Class1" in out
    out = get_cli_stdout(components, args=["lv1_b", "lv2_z", "lv3_q", "--help"], **kw)
    assert " {method1} ..." in out
    if docstring_parser_support:
        assert "Description of Class1" in out
        assert "Description of method1" in out
    out = get_cli_stdout(components, args=["lv1_b", "lv2_z", "lv3_q", "method1", "--help"], **kw)
    assert " --m1 M1" in out
    if docstring_parser_support:
        assert "Description of method1" in out

    assert 5.6 == CLI(components, args=["lv1_a", "lv2_x", "--a1=5.6"], **kw)
    assert 7 == CLI(components, args=["lv1_a", "lv2_y", "lv3_p", "--x=7"], **kw)
    assert ("w", 9) == CLI(components, args=["lv1_b", "lv2_z", "lv3_q", "--i1=w", "method1", "--m1=9"], **kw)


# config file tests


class A:
    def __init__(self, p1: str = "a default"):
        self.p1 = p1


class B:
    def __init__(self, a: A = A()):
        self.a = a


class C:
    def __init__(self, a: A = lazy_instance(A), b: Optional[B] = None):
        self.a = a
        self.b = b

    def cmd_a(self):
        print(self.a.p1)

    def cmd_b(self):
        if self.b:
            print(self.b.a.p1)


def test_subclass_type_config_file(tmp_cwd):
    a_yaml = {"class_path": f"{__name__}.A", "init_args": {"p1": "a yaml"}}
    Path("config.yaml").write_text("a: a.yaml\n")
    Path("a.yaml").write_text(yaml.safe_dump(a_yaml))

    out = get_cli_stdout(C, args=["--config=config.yaml", "cmd_a"])
    assert "a yaml\n" == out

    out = get_cli_stdout(C, args=["cmd_a", "--help"])
    assert "--config" not in out

    Path("config.yaml").write_text("a: a.yaml\nb: b.yaml\n")
    Path("b.yaml").write_text(f"class_path: {__name__}.B\ninit_args:\n  a: a.yaml\n")

    out = get_cli_stdout(C, args=["--config=config.yaml", "cmd_b"])
    assert "a yaml\n" == out


@final
class BF:
    def __init__(self, a: A):
        self.a = a


def run_bf(b: BF):
    return b.a.p1


def test_final_and_subclass_type_config_file(tmp_cwd):
    a_yaml = {"class_path": f"{__name__}.A", "init_args": {"p1": "a yaml"}}
    Path("config.yaml").write_text("b: b.yaml\n")
    Path("b.yaml").write_text("a: a.yaml\n")
    Path("a.yaml").write_text(yaml.safe_dump(a_yaml))

    out = CLI(run_bf, args=["--config=config.yaml"])
    assert "a yaml" == out


# async tests


async def run_async(time: float = 0.1):
    await asyncio.sleep(time)
    return "done"


def test_async_function():
    assert "done" == CLI(run_async, args=["--time=0.0"])


class AsyncMethod:
    def __init__(self, time: float = 0.1, require_async: bool = False):
        self.time = time
        if require_async:
            self.loop = asyncio.get_event_loop()

    async def run(self):
        await asyncio.sleep(self.time)
        return "done"


def test_async_method():
    assert "done" == CLI(AsyncMethod, args=["--time=0.0", "run"])


async def run_async_instance(cls: Callable[[], AsyncMethod]):
    return await cls().run()


def test_async_instance():
    config = {
        "cls": {
            "class_path": f"{__name__}.AsyncMethod",
            "init_args": {"time": 0.0, "require_async": True},
        }
    }
    assert "done" == CLI(run_async_instance, args=[f"--config={config}"])
