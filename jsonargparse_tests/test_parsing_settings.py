import re
from dataclasses import dataclass
from typing import Literal, Optional
from unittest.mock import patch

import pytest

from jsonargparse import ActionYesNo, ArgumentError, Namespace, set_parsing_settings
from jsonargparse._common import get_parsing_setting
from jsonargparse_tests.conftest import capture_logs, get_parse_args_stdout, get_parser_help
from jsonargparse_tests.test_typehints import Optimizer


@pytest.fixture(autouse=True)
def patch_parsing_settings():
    with patch.dict("jsonargparse._common.parsing_settings"):
        yield


def test_get_parsing_setting_failure():
    with pytest.raises(ValueError, match="Unknown parsing setting"):
        get_parsing_setting("unknown_setting")


# validate_defaults


def test_set_validate_defaults_failure():
    with pytest.raises(ValueError, match="validate_defaults must be a boolean"):
        set_parsing_settings(validate_defaults="invalid")


def test_validate_defaults_success(parser):
    set_parsing_settings(validate_defaults=True)

    parser.add_argument("--config", action="config")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--untyped", default=2)


def test_validate_defaults_failure(parser):
    set_parsing_settings(validate_defaults=True)

    with pytest.raises(ValueError, match="Default value is not valid:"):
        parser.add_argument("--num", type=int, default="x")


# parse_optionals_as_positionals


def test_set_parse_optionals_as_positionals_failure():
    with pytest.raises(ValueError, match="parse_optionals_as_positionals must be a boolean"):
        set_parsing_settings(parse_optionals_as_positionals="invalid")


def test_parse_optionals_as_positionals_simple(parser, logger, subtests):
    set_parsing_settings(parse_optionals_as_positionals=True)

    parser.add_argument("p1", type=Optional[Literal["p1"]])
    parser.add_argument("--o1", type=Optional[int])
    parser.add_argument("--flag", default=False, nargs=0, action=ActionYesNo)
    parser.add_argument("--o2", type=Optional[Literal["o2"]])
    parser.add_argument("--o3")

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        assert " p1 [o1 [o2 [o3]]]" in help_str
        assert "extra positionals are parsed as optionals in the order shown above" in help_str

    with subtests.test("no extra positionals"):
        cfg = parser.parse_args(["--o2=o2", "--o1=1", "p1"])
        assert cfg == Namespace(p1="p1", o1=1, o2="o2", o3=None, flag=False)

    with subtests.test("one extra positional"):
        cfg = parser.parse_args(["--o2=o2", "p1", "2"])
        assert cfg == Namespace(p1="p1", o1=2, o2="o2", o3=None, flag=False)

    with subtests.test("two extra positionals"):
        cfg = parser.parse_args(["p1", "3", "o2"])
        assert cfg == Namespace(p1="p1", o1=3, o2="o2", o3=None, flag=False)

    with subtests.test("three extra positionals"):
        cfg = parser.parse_args(["p1", "3", "o2", "v3"])
        assert cfg == Namespace(p1="p1", o1=3, o2="o2", o3="v3", flag=False)

    with subtests.test("extra positional has precedence"):
        cfg = parser.parse_args(["p1", "3", "o2", "--o1=4"])
        assert cfg == Namespace(p1="p1", o1=3, o2="o2", o3=None, flag=False)

    with subtests.test("extra positionals invalid values"):
        with pytest.raises(ArgumentError) as ex:
            parser.parse_args(["p1", "o2", "5"])
        assert re.match('Parser key "o1".*Given value: o2', ex.value.message, re.DOTALL)

        with pytest.raises(ArgumentError) as ex:
            parser.parse_args(["p1", "6", "invalid"])
        assert re.match('Parser key "o2".*Given value: invalid', ex.value.message, re.DOTALL)

    parser.logger = logger
    with subtests.test("unrecognized arguments"):
        with capture_logs(logger) as logs:
            with pytest.raises(ArgumentError, match="Unrecognized arguments: --unk=x"):
                parser.parse_args(["--unk=x"])
        assert "Positional argument p1 missing, aborting _positional_optionals" in logs.getvalue()


def test_parse_optionals_as_positionals_subcommands(parser, subparser, subtests):
    set_parsing_settings(parse_optionals_as_positionals=True)

    subparser.add_argument("p1", type=Optional[Literal["p1"]])
    subparser.add_argument("--o1", type=Optional[int])
    subparser.add_argument("--o2", type=Optional[Literal["o2"]])
    parser.add_argument("--g1")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("subcmd", subparser)

    with subtests.test("help global"):
        help_str = get_parser_help(parser)
        assert " [g1]" not in help_str
        assert "extra positionals are parsed as optionals in the order shown above" not in help_str

    with subtests.test("help subcommand"):
        help_str = get_parse_args_stdout(parser, ["subcmd", "-h"])
        assert " p1 [o1 [o2]]" in help_str
        assert "extra positionals are parsed as optionals in the order shown above" in help_str

    with subtests.test("no extra positionals"):
        cfg = parser.parse_args(["subcmd", "--o2=o2", "--o1=1", "p1"])
        assert cfg.subcmd == Namespace(p1="p1", o1=1, o2="o2")

    with subtests.test("one extra positional"):
        cfg = parser.parse_args(["subcmd", "--o2=o2", "p1", "2"])
        assert cfg.subcmd == Namespace(p1="p1", o1=2, o2="o2")

    with subtests.test("two extra positionals"):
        cfg = parser.parse_args(["subcmd", "p1", "3", "o2"])
        assert cfg.subcmd == Namespace(p1="p1", o1=3, o2="o2")

    with subtests.test("extra positionals invalid values"):
        with pytest.raises(ArgumentError) as ex:
            parser.parse_args(["subcmd", "p1", "o2", "5"])
        assert re.match('Parser key "o1".*Given value: o2', ex.value.message, re.DOTALL)


def test_optionals_as_positionals_usage_wrap(parser):
    set_parsing_settings(parse_optionals_as_positionals=True)

    parser.prog = "long_prog_name"
    parser.add_argument("relatively_long_positional")
    parser.add_argument("--first_long_optional")
    parser.add_argument("--second_long_optional")

    help_str = get_parser_help(parser, columns="80")
    assert "usage: long_prog_name " in help_str
    assert "                      relatively_long_positional" in help_str
    assert "                      [first_long_optional [second_long_optional]]" in help_str


@dataclass
class DataOptions:
    d1: int = 1


def test_optionals_as_positionals_unsupported_arguments(parser):
    set_parsing_settings(parse_optionals_as_positionals=True)

    parser.add_argument("p1", type=Optional[Literal["p1"]])
    parser.add_argument("--o1", type=Optimizer)
    parser.add_argument("--o2", type=Optional[int])
    parser.add_argument("--o3", type=DataOptions)
    parser.add_argument("--o4.n1", type=float)

    help_str = get_parser_help(parser)
    assert " p1 [o2 [o3.d1 [o4.n1]]]" in help_str

    help_str = get_parse_args_stdout(parser, ["--o1.help=Adam"])
    assert "extra positionals are parsed as optionals in the order shown above" not in help_str
