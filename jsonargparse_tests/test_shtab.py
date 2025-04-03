import re
import subprocess
import tempfile
from enum import Enum
from importlib.util import find_spec
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser, set_parsing_settings
from jsonargparse._completions import norm_name
from jsonargparse._parameter_resolvers import get_signature_parameters
from jsonargparse._typehints import type_to_str
from jsonargparse.typing import Path_drw, Path_fr
from jsonargparse_tests.conftest import capture_logs, get_parse_args_stdout


@pytest.fixture(autouse=True)
def skip_if_no_shtab():
    if not find_spec("shtab"):
        pytest.skip("shtab package is required")


@pytest.fixture(autouse=True)
def skip_if_wsl_message():
    popen = subprocess.Popen(["bash", "-c", "echo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = popen.communicate()
    if "Windows Subsystem for Linux has no installed distributions" in out.decode().replace("\x00", ""):
        pytest.skip(out.decode().replace("\x00", ""))


@pytest.fixture(autouse=True)
def term_env_var():
    with patch.dict("os.environ", {"TERM": "xterm-256color", "COLUMNS": "200"}):
        yield


@pytest.fixture
def parser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False, prog="tool")


def get_shtab_script(parser, shell):
    return get_parse_args_stdout(parser, [f"--print_shtab={shell}"])


def is_positional(dest, parser):
    if parser is not None:
        action = next(a for a in parser._actions if a.dest == dest)
        return action.option_strings == []
    return False


def assert_bash_typehint_completions(subtests, shtab_script, completions):
    parser = None
    if isinstance(shtab_script, ArgumentParser):
        parser = shtab_script
        shtab_script = get_shtab_script(shtab_script, "bash")
    with tempfile.TemporaryDirectory() as tmpdir:
        shtab_script_path = Path(tmpdir) / "comp.sh"
        shtab_script_path.write_text(shtab_script)

        for dest, typehint, word, choices, extra in completions:
            typehint = type_to_str(typehint)
            with subtests.test(f"{word} -> {extra}"):
                sh = f'source {shtab_script_path}; COMP_TYPE=63 _jsonargparse_tool_{norm_name(dest)}_typehint "{word}"'
                popen = subprocess.Popen(["bash", "-c", sh], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = popen.communicate()
                assert list(out.decode().split()) == choices
                if extra is None:
                    assert f"Expected type: {typehint}" in err.decode()
                elif re.match(r"^\d/\d$", extra):
                    assert f"Expected type: {typehint}; {extra} matched choices" in err.decode()
                else:
                    assert f"Expected type: {typehint}; Accepted by subclasses: {extra}" in err.decode()
                if is_positional(dest, parser):
                    assert f"Argument: {dest}; Expected type: {typehint}" in err.decode()


def test_bash_any(parser, subtests):
    parser.add_argument("--any", type=Any)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("any", Any, "", [], None),
        ],
    )


def test_bash_bool(parser, subtests):
    parser.add_argument("--bool", type=bool)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("bool", bool, "", ["true", "false"], "2/2"),
        ],
    )


def test_bash_optional_bool(parser, subtests):
    parser.add_argument("--bool", type=Optional[bool])
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("bool", Optional[bool], "", ["true", "false", "null"], "3/3"),
            ("bool", Optional[bool], "tr", ["true"], "1/3"),
            ("bool", Optional[bool], "x", [], "0/3"),
        ],
    )


def test_bash_argument_group(parser, subtests):
    group = parser.add_argument_group("Group1")
    group.add_argument("--bool", type=bool)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("bool", bool, "", ["true", "false"], "2/2"),
        ],
    )


class AXEnum(Enum):
    ABC = "abc"
    XY = "xy"
    XZ = "xz"


def test_bash_enum(parser, subtests):
    parser.add_argument("--enum", type=AXEnum)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("enum", AXEnum, "", ["ABC", "XY", "XZ"], "3/3"),
            ("enum", AXEnum, "A", ["ABC"], "1/3"),
            ("enum", AXEnum, "X", ["XY", "XZ"], "2/3"),
        ],
    )


def test_bash_optional_enum(parser, subtests):
    parser.add_argument("--enum", type=Optional[AXEnum])
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("enum", Optional[AXEnum], "", ["ABC", "XY", "XZ", "null"], "4/4"),
        ],
    )


def test_bash_literal(parser, subtests):
    typehint = Optional[Literal["one", "two"]]
    parser.add_argument("--literal", type=typehint)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("literal", typehint, "", ["one", "two", "null"], "3/3"),
            ("literal", typehint, "t", ["two"], "1/3"),
        ],
    )


def test_bash_union(parser, subtests):
    typehint = Optional[Union[bool, AXEnum]]
    parser.add_argument("--union", type=typehint)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("union", typehint, "", ["true", "false", "ABC", "XY", "XZ", "null"], "6/6"),
            ("union", typehint, "z", [], "0/6"),
        ],
    )


def test_bash_positional(parser, subtests):
    typehint = Literal["Alice", "Bob"]
    parser.add_argument("name", type=typehint)
    assert_bash_typehint_completions(
        subtests,
        parser,
        [
            ("name", typehint, "", ["Alice", "Bob"], "2/2"),
            ("name", typehint, "Al", ["Alice"], "1/2"),
        ],
    )


def test_shtab_bash_optionals_as_positionals(parser, subtests):
    with patch.dict("jsonargparse._common.parsing_settings"):
        set_parsing_settings(parse_optionals_as_positionals=True)
        parser.prog = "tool"

        parser.add_argument("job", type=str)
        parser.add_argument("--amount", type=int, default=0)
        parser.add_argument("--flag", type=bool, default=False)
        assert_bash_typehint_completions(
            subtests,
            parser,
            [
                ("job", str, "", [], None),
                ("job", str, "easy", [], None),
                ("amount", int, "easy ", [], None),
                ("amount", int, "easy 10", [], None),
                ("flag", bool, "easy 10 x", [], "0/2"),
            ],
        )


def test_bash_config(parser):
    parser.add_argument("--cfg", action="config")
    shtab_script = get_shtab_script(parser, "bash")
    assert "_cfg_COMPGEN=_shtab_compgen_files" in shtab_script


def test_bash_dir(parser):
    parser.add_argument("--path", type=Path_drw)
    shtab_script = get_shtab_script(parser, "bash")
    assert "_path_COMPGEN=_shtab_compgen_dirs" in shtab_script


@pytest.mark.parametrize("path_type", [Path_fr, PathLike, Path, Union[PathLike, str]])
def test_bash_file(parser, path_type):
    parser.add_argument("--path", type=path_type)
    shtab_script = get_shtab_script(parser, "bash")
    assert "_path_COMPGEN=_shtab_compgen_files" in shtab_script


@pytest.mark.parametrize("path_type", [Path_fr, PathLike, Path, Union[Union[PathLike, str], dict]])
def test_bash_optional_file(parser, path_type):
    parser.add_argument("--path", type=Optional[path_type])
    shtab_script = get_shtab_script(parser, "bash")
    assert "_path_COMPGEN=_shtab_compgen_files" in shtab_script


class Base:
    def __init__(self, p1: int):
        pass


def test_bash_class_config(parser):
    parser.add_class_arguments(Base, "class")
    shtab_script = get_shtab_script(parser, "bash")
    assert "_class_COMPGEN=_shtab_compgen_files" in shtab_script


class SubA(Base):
    def __init__(self, p1: int, p2: AXEnum):
        pass


class SubB(Base):
    def __init__(self, p1: int, p3: float):
        pass


def test_bash_subclasses_fail_get_perams(parser, logger):
    def get_params_patch(cls, method, logger):
        if cls == SubB:
            raise Exception("test get params failure")
        return get_signature_parameters(cls, method, logger)

    parser.logger = logger
    parser.add_argument("--cls", type=Base)
    with capture_logs(logger) as logs, patch("jsonargparse._completions.get_signature_parameters", get_params_patch):
        shtab_script = get_shtab_script(parser, "bash")
    assert "'--cls' '--cls.p1' '--cls.p2'" in shtab_script
    assert f"'{__name__}.SubB'" in shtab_script
    assert "'--cls.p3'" not in shtab_script
    assert "test_shtab.SubB': test get params failure" in logs.getvalue()


def test_bash_subclasses_help(parser):
    parser.add_argument("--cls", type=Base)
    shtab_script = get_shtab_script(parser, "bash")
    assert "'--cls.help' '--cls' '--cls.p1' '--cls.p2' '--cls.p3'" in shtab_script
    classes = f"'{__name__}.Base' '{__name__}.SubA' '{__name__}.SubB'"
    assert f"_cls_help_choices=({classes})" in shtab_script


def test_bash_subclasses(parser, subtests):
    parser.add_argument("--cls", type=Base)
    shtab_script = get_shtab_script(parser, "bash")
    classes = f"{__name__}.Base {__name__}.SubA {__name__}.SubB".split()
    assert_bash_typehint_completions(
        subtests,
        shtab_script,
        [
            ("cls", Base, "", classes, "3/3"),
            ("cls", Base, f"{__name__}.S", classes[1:], "2/3"),
            ("cls.p1", int, "1", [], "Base, SubA, SubB"),
            ("cls.p3", float, "", [], "SubB"),
        ],
    )


class Other:
    def __init__(self, o1: bool):
        pass


def test_bash_union_subclasses(parser, subtests):
    parser.add_argument("--cls", type=Union[Base, Other])
    shtab_script = get_shtab_script(parser, "bash")
    assert "'--cls.help' '--cls' '--cls.p1' '--cls.p2' '--cls.p3' '--cls.o1'" in shtab_script
    classes = f"'{__name__}.Base' '{__name__}.SubA' '{__name__}.SubB' '{__name__}.Other'"
    assert f"_cls_help_choices=({classes})" in shtab_script
    assert_bash_typehint_completions(
        subtests,
        shtab_script,
        [
            ("cls.p1", int, "", [], "Base, SubA, SubB"),
        ],
    )


class SupBase:
    def __init__(self, s1: Base):
        pass


class SupA(SupBase):
    def __init__(self, s1: Optional[Base]):
        pass


def test_bash_nested_subclasses(parser, subtests):
    parser.add_argument("--cls", type=SupBase)
    shtab_script = get_shtab_script(parser, "bash")
    assert "'--cls.help' '--cls' '--cls.s1' '--cls.s1.p1' '--cls.s1.p2' '--cls.s1.p3'" in shtab_script
    assert_bash_typehint_completions(
        subtests,
        shtab_script,
        [
            ("cls.s1.p2", AXEnum, "X", ["XY", "XZ"], "SubA; 2/3 matched choices"),
        ],
    )


def test_bash_callable_return_class(parser, subtests):
    parser.add_argument("--cls", type=Callable[[int], Base])
    shtab_script = get_shtab_script(parser, "bash")
    assert "_option_strings=('-h' '--help' '--cls.help' '--cls' '--cls.p2' '--cls.p3')" in shtab_script
    assert "--cls.p1" not in shtab_script
    classes = f"{__name__}.Base {__name__}.SubA {__name__}.SubB".split()
    assert_bash_typehint_completions(
        subtests,
        shtab_script,
        [
            ("cls", Callable[[int], Base], "", classes, "3/3"),
            ("cls.p2", AXEnum, "z", [], "SubA"),
        ],
    )


def test_bash_subcommands(parser, subparser, subtests):
    subparser.add_argument("--enum", type=AXEnum)
    subparser2 = ArgumentParser()
    subparser2.add_argument("--cls", type=Base)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("s1", subparser)
    subcommands.add_subcommand("s2", subparser2)

    help_str = get_parse_args_stdout(parser, ["--help"])
    assert "--print_shtab" in help_str
    help_str = get_parse_args_stdout(parser, ["s1", "--help"])
    assert "--print_shtab" not in help_str

    shtab_script = get_shtab_script(parser, "bash")
    assert "_subparsers=('s1' 's2')" in shtab_script

    assert "_s1_option_strings=('-h' '--help' '--enum')" in shtab_script
    assert "_s2_option_strings=('-h' '--help' '--cls.help' '--cls' '--cls.p1' '--cls.p2' '--cls.p3')" in shtab_script
    classes = f"'{__name__}.Base' '{__name__}.SubA' '{__name__}.SubB'"
    assert f"_s2___cls_help_choices=({classes})" in shtab_script

    assert_bash_typehint_completions(
        subtests,
        shtab_script,
        [
            ("s1.enum", AXEnum, "A", ["ABC"], "1/3"),
            ("s2.cls.p1", int, "1", [], "Base, SubA, SubB"),
        ],
    )


def test_zsh_script(parser):
    parser.add_argument("--enum", type=Optional[AXEnum])
    parser.add_argument("--path", type=PathLike)
    parser.add_argument("--cls", type=Base)
    shtab_script = get_shtab_script(parser, "zsh")
    assert ":enum:(ABC XY XZ null)" in shtab_script
    assert ":path:_files" in shtab_script
    classes = f"{__name__}.Base {__name__}.SubA {__name__}.SubB"
    assert f":cls.help:({classes})" in shtab_script
    assert f":cls:({classes})" in shtab_script
    assert ":cls.p1:" in shtab_script
    assert ":cls.p2:(ABC XY XZ)" in shtab_script
    assert ":cls.p3:" in shtab_script
