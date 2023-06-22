from __future__ import annotations

import os
import sys
from contextlib import ExitStack, contextmanager
from enum import Enum
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

from jsonargparse import ActionConfigFile, ActionJsonSchema, ActionYesNo
from jsonargparse._common import parser_context
from jsonargparse.typing import Email, Path_fr, PositiveFloat, PositiveInt
from jsonargparse_tests.conftest import (
    skip_if_jsonschema_unavailable,
    skip_if_not_cpython,
    skip_if_not_posix,
)


@pytest.fixture(autouse=True)
def skip_if_argcomplete_unavailable():
    if not find_spec("argcomplete"):
        pytest.skip("argcomplete package is required")


@contextmanager
def mock_fdopen():
    err = StringIO()
    with patch("os.fdopen", return_value=err):
        yield err


def complete_line(parser, value):
    stack = ExitStack()
    stack.enter_context(parser_context(load_value_mode="yaml"))
    with patch.dict(
        os.environ,
        {
            "_ARGCOMPLETE": "1",
            "_ARGCOMPLETE_SUPPRESS_SPACE": "1",
            "_ARGCOMPLETE_COMP_WORDBREAKS": " \t\n\"'><=;|&(:",
            "COMP_TYPE": str(ord("?")),  # ='63'  str(ord('\t'))='9'
            "COMP_LINE": value,
            "COMP_POINT": str(len(value)),
        },
    ):
        import argcomplete

        out = StringIO()
        with pytest.raises(SystemExit), mock_fdopen() as err:
            argcomplete.autocomplete(parser, exit_method=sys.exit, output_stream=out)
    stack.close()
    return out.getvalue(), err.getvalue()


def test_complete_nested_one_option(parser):
    parser.add_argument("--group1.op")
    out, err = complete_line(parser, "tool.py --group1")
    assert out == "--group1.op"
    assert err == ""


def test_complete_nested_two_options(parser):
    parser.add_argument("--group2.op1")
    parser.add_argument("--group2.op2")
    out, err = complete_line(parser, "tool.py --group2")
    assert out == "--group2.op1\x0b--group2.op2"
    assert err == ""


@skip_if_not_cpython
@pytest.mark.parametrize(
    ["value", "expected"],
    [
        ("--int=a", "value not yet valid, expected type int"),
        ("--int=1", "value already valid, expected type int"),
        ("--float=a", "value not yet valid, expected type float"),
        ("--float=1", "value already valid, expected type float"),
        ("--pint=0", "value not yet valid, expected type PositiveInt"),
        ("--pint=1", "value already valid, expected type PositiveInt"),
        ("--pfloat=0", "value not yet valid, expected type PositiveFloat"),
        ("--pfloat=1", "value already valid, expected type PositiveFloat"),
        ("--email=a", "value not yet valid, expected type Email"),
        ("--email=a@b.c", "value already valid, expected type Email"),
    ],
)
def test_stderr_instruction_simple_types(parser, value, expected):
    parser.add_argument("--int", type=int)
    parser.add_argument("--float", type=float)
    parser.add_argument("--pint", type=PositiveInt)
    parser.add_argument("--pfloat", type=PositiveFloat)
    parser.add_argument("--email", type=Email)
    out, err = complete_line(parser, "tool.py " + value)
    assert out == ""
    assert expected in err


@skip_if_not_posix
def test_action_config_file(parser, tmp_cwd):
    parser.add_argument("--cfg", action=ActionConfigFile)
    Path("file1").touch()
    Path("config.yaml").touch()

    out, err = complete_line(parser, "tool.py --cfg=")
    assert out == "config.yaml\x0bfile1"
    assert err == ""
    out, err = complete_line(parser, "tool.py --cfg=c")
    assert out == "config.yaml"
    assert err == ""


@pytest.mark.parametrize(
    ["value", "expected"],
    [
        ("--op1", "--op1"),
        ("--no_op1", "--no_op1"),
        ("--op2", "--op2"),
        ("--no_op2", "--no_op2"),
        ("--op2=", "true\x0bfalse\x0byes\x0bno"),
        ("--with-op3", "--with-op3"),
        ("--without-op3", "--without-op3"),
    ],
)
def test_action_yes_no(parser, value, expected):
    parser.add_argument("--op1", action=ActionYesNo)
    parser.add_argument("--op2", nargs="?", action=ActionYesNo)
    parser.add_argument("--with-op3", action=ActionYesNo(yes_prefix="with-", no_prefix="without-"))
    out, err = complete_line(parser, "tool.py " + value)
    assert out == expected
    assert err == ""


def test_bool(parser):
    parser.add_argument("--bool", type=bool)
    out, err = complete_line(parser, "tool.py --bool=")
    assert out == "true\x0bfalse"
    assert err == ""
    out, err = complete_line(parser, "tool.py --bool=f")
    assert out == "false"
    assert err == ""


def test_enum(parser):
    class EnumType(Enum):
        abc = 1
        xyz = 2
        abd = 3

    parser.add_argument("--enum", type=EnumType)
    out, err = complete_line(parser, "tool.py --enum=ab")
    assert out == "abc\x0babd"
    assert err == ""


def test_optional_bool(parser):
    parser.add_argument("--bool", type=Optional[bool])
    out, err = complete_line(parser, "tool.py --bool=")
    assert out == "true\x0bfalse\x0bnull"
    assert err == ""


def test_optional_enum(parser):
    class EnumType(Enum):
        A = 1
        B = 2

    parser.add_argument("--enum", type=Optional[EnumType])
    out, err = complete_line(parser, "tool.py --enum=")
    assert out == "A\x0bB\x0bnull"
    assert err == ""


@skip_if_not_cpython
@skip_if_jsonschema_unavailable
def test_action_jsonschema(parser):
    parser.add_argument("--json", action=ActionJsonSchema(schema={"type": "object"}))

    for value, expected in [
        ("--json=1", "value not yet valid"),
        ("--json='{\"a\": 1}'", "value already valid"),
    ]:
        out, err = complete_line(parser, f"tool.py {value}")
        assert out == ""
        assert expected in err

        with patch("os.popen") as popen_mock:
            popen_mock.side_effect = ValueError
            out, err = complete_line(parser, f"tool.py {value}")
            assert out == ""
            assert expected in err

    out, err = complete_line(parser, "tool.py --json ")
    assert "value not yet valid" in out.replace("\xa0", " ").replace("_", " ")
    assert err == ""


def test_list(parser):
    parser.add_argument("--list", type=List[int])

    out, err = complete_line(parser, "tool.py --list='[1, 2, 3]'")
    assert out == ""
    assert "value already valid, expected type List[int]" in err

    out, err = complete_line(parser, "tool.py --list=")
    assert "value not yet valid" in out.replace("\xa0", " ").replace("_", " ")
    assert err == ""


@skip_if_not_posix
def test_optional_path(parser, tmp_cwd):
    parser.add_argument("--path", type=Optional[Path_fr])
    Path("file1").touch()
    Path("file2").touch()

    for value, expected in [
        ("--path=", "null\x0bfile1\x0bfile2"),
        ("--path=n", "null"),
        ("--path=f", "file1\x0bfile2"),
    ]:
        out, err = complete_line(parser, f"tool.py {value}")
        assert out == expected
        assert err == ""
