import re
from typing import Literal, Optional
from unittest.mock import patch

import pytest

from jsonargparse import ActionYesNo, ArgumentError, Namespace, set_parsing_settings
from jsonargparse._common import get_parsing_setting
from jsonargparse_tests.conftest import capture_logs, get_parser_help


@pytest.fixture(autouse=True)
def patch_parsing_settings():
    with patch.dict("jsonargparse._common.parsing_settings"):
        yield


def test_get_parsing_setting_failure():
    with pytest.raises(ValueError, match="Unknown parsing setting"):
        get_parsing_setting("unknown_setting")


def test_set_parse_optionals_as_positionals_failure():
    with pytest.raises(ValueError, match="parse_optionals_as_positionals must be a boolean"):
        set_parsing_settings(parse_optionals_as_positionals="invalid")


def test_parse_optionals_as_positionals(parser, logger, subtests):
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
        # TODO: document positionals precedence over optional is expected behavior
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


# TODO: test parse_optionals_as_positionals with subcommands
# TODO: test parse_optionals_as_positionals with inner parsers
# TODO: test parse_optionals_as_positionals with required non-positionals
