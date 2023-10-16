from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from jsonargparse import (
    ActionConfigFile,
    ArgumentError,
    ArgumentParser,
    Namespace,
    strip_meta,
)
from jsonargparse_tests.conftest import get_parse_args_stderr, get_parse_args_stdout, get_parser_help
from jsonargparse_tests.test_subclasses import CustomInstantiationBase, instantiator


@pytest.fixture
def subcommands_parser(parser, subparser, example_parser):
    subparser.add_argument("ap1")
    subparser.add_argument("--ao1", default="ao1_def")
    parser.env_prefix = "APP"
    parser.add_argument("--o1", default="o1_def")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("a", subparser)
    subcommands.add_subcommand("b", example_parser, aliases=["B"])
    return parser


def test_add_subparsers_not_implemented(parser):
    with pytest.raises(NotImplementedError) as ctx:
        parser.add_subparsers()
    ctx.match("add_subcommands method")


def test_add_parser_not_implemented(parser):
    subcommands = parser.add_subcommands()
    with pytest.raises(NotImplementedError) as ctx:
        subcommands.add_parser("")
    ctx.match("add_subcommand method")


def test_subcommands_get_defaults(subcommands_parser):
    cfg = subcommands_parser.get_defaults().as_dict()
    assert cfg == {"o1": "o1_def", "subcommand": None}


def test_subcommands_undefined_subcommand(subcommands_parser):
    pytest.raises(ArgumentError, lambda: subcommands_parser.parse_args(["c"]))


def test_subcommands_not_given_when_few_subcommands(subcommands_parser):
    err = get_parse_args_stderr(subcommands_parser, [])
    assert 'error: expected "subcommand" to be one of {a,b,B}, but it was not provided.' in err


def test_subcommands_not_given_when_many_subcommands(parser, subparser):
    subcommands = parser.add_subcommands()
    for subcommand in range(ord("a"), ord("l") + 1):
        subcommands.add_subcommand(chr(subcommand), subparser)
    err = get_parse_args_stderr(parser, [])
    assert 'error: expected "subcommand" to be one of {a,b,c,d,e, ...}, but it was not provided.' in err


def test_subcommands_missing_required_subargument(subcommands_parser):
    with pytest.raises(ArgumentError) as ctx:
        subcommands_parser.parse_args(["a"])
    ctx.match('"a.ap1" is required')


def test_subcommands_undefined_subargument(subcommands_parser):
    pytest.raises(ArgumentError, lambda: subcommands_parser.parse_args(["b", "--unk"]))


def test_subcommands_parse_args_basics(subcommands_parser):
    cfg = subcommands_parser.parse_args(["--o1", "o1_arg", "a", "ap1_arg"])
    assert cfg["o1"] == "o1_arg"
    assert cfg["subcommand"] == "a"
    assert cfg["a"].as_dict() == {"ap1": "ap1_arg", "ao1": "ao1_def"}
    assert "b" not in cfg

    cfg = subcommands_parser.parse_args(["a", "ap1_arg", "--ao1", "ao1_arg"])
    assert cfg["a"].as_dict() == {"ap1": "ap1_arg", "ao1": "ao1_arg"}

    cfg = subcommands_parser.parse_args(["b"])
    assert cfg["subcommand"] == "b"
    assert "a" not in cfg


def test_main_subcommands_help(subcommands_parser):
    help_str = get_parser_help(subcommands_parser)
    assert help_str.count("{a,b,B}") == 1
    assert "Available subcommands:" in help_str
    assert "b (B)" in help_str


def test_subcommands_parse_args_alias(subcommands_parser):
    cfg = subcommands_parser.parse_args(["B"])
    assert cfg["subcommand"] == "B"
    pytest.raises(ArgumentError, lambda: subcommands_parser.parse_args(["A"]))


def test_subcommands_parse_args_config(subcommands_parser):
    subcommands_parser.add_argument("--cfg", action=ActionConfigFile)
    cfg = subcommands_parser.parse_args(['--cfg={"o1": "o1_arg"}', "a", "ap1_arg"]).as_dict()
    assert cfg == {
        "a": {"ao1": "ao1_def", "ap1": "ap1_arg"},
        "cfg": [None],
        "o1": "o1_arg",
        "subcommand": "a",
    }


def test_subcommands_parse_string_implicit_subcommand(subcommands_parser):
    cfg = subcommands_parser.parse_string('{"a": {"ap1": "ap1_cfg"}}').as_dict()
    assert cfg["subcommand"] == "a"
    assert cfg["a"] == {"ap1": "ap1_cfg", "ao1": "ao1_def"}
    with pytest.raises(ArgumentError) as ctx:
        subcommands_parser.parse_string('{"a": {"ap1": "ap1_cfg", "unk": "unk_cfg"}}')
    ctx.match('No action for key "unk"')


def test_subcommands_parse_string_first_implicit_subcommand(subcommands_parser):
    with warnings.catch_warnings(record=True) as w:
        cfg = subcommands_parser.parse_string('{"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}')
    assert len(w) == 1
    assert 'Subcommand "a" will be used' in str(w[0].message)
    assert cfg.subcommand == "a"
    assert "b" not in cfg


def test_subcommands_parse_string_explicit_subcommand(subcommands_parser):
    cfg = subcommands_parser.parse_string('{"subcommand": "b", "a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}')
    assert cfg.subcommand == "b"
    assert cfg.b.as_dict() == {"bool": True, "nums": {"val1": 2, "val2": 2.0}}
    assert "a" not in cfg


def test_subcommands_parse_args_config_explicit_subcommand_arg(subcommands_parser):
    subcommands_parser.add_argument("--cfg", action=ActionConfigFile)
    cfg = subcommands_parser.parse_args(['--cfg={"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}', "a"])
    assert cfg.as_dict() == {
        "o1": "o1_def",
        "subcommand": "a",
        "cfg": [None],
        "a": {"ap1": "ap1_cfg", "ao1": "ao1_def"},
    }


env = {
    "APP_O1": "o1_env",
    "APP_A__AP1": "ap1_env",
    "APP_A__AO1": "ao1_env",
    "APP_B__NUMS__VAL2": "5.6",
}


@patch.dict(os.environ, env)
def test_subcommands_parse_args_environment(subcommands_parser):
    cfg = subcommands_parser.parse_args(["a"], env=True).as_dict()
    assert cfg["o1"] == "o1_env"
    assert cfg["subcommand"] == "a"
    assert cfg["a"] == {"ap1": "ap1_env", "ao1": "ao1_env"}

    subcommands_parser.default_env = True
    cfg = subcommands_parser.parse_args(["b"]).as_dict()
    assert cfg["subcommand"] == "b"
    assert cfg["b"] == {"bool": True, "nums": {"val1": 1, "val2": 5.6}}


@patch.dict(os.environ, {"APP_SUBCOMMAND": "a", **env})
def test_subcommands_parse_env(subcommands_parser):
    cfg = subcommands_parser.parse_env().as_dict()
    assert cfg["o1"] == "o1_env"
    assert cfg["subcommand"] == "a"
    assert cfg["a"] == {"ap1": "ap1_env", "ao1": "ao1_env"}


def test_subcommands_help_default_env_true(subcommands_parser):
    subcommands_parser.default_env = True
    help_str = get_parser_help(subcommands_parser)
    assert "ENV:   APP_SUBCOMMAND" in help_str


def test_subcommand_required_false(parser, subparser):
    subcommands = parser.add_subcommands(required=False)
    subcommands.add_subcommand("foo", subparser)
    cfg = parser.parse_args([])
    assert cfg == Namespace(subcommand=None)


def test_subcommand_without_options(parser, subparser):
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("foo", subparser)
    cfg = parser.parse_args(["foo"])
    assert cfg.subcommand == "foo"


def test_subcommand_print_config_default_env(subparser):
    subparser.add_argument("--config", action=ActionConfigFile)
    subparser.add_argument("--o", type=int, default=1)

    parser = ArgumentParser(exit_on_error=False, default_env=True)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("a", subparser)

    out = get_parse_args_stdout(parser, ["a", "--print_config"])
    assert yaml.safe_load(out) == {"o": 1}


def test_subcommand_default_config_repeated_keys(parser, subparser, tmp_cwd):
    defaults = tmp_cwd / "defaults.json"
    defaults.write_text('{"test":{"test":"value"}}')
    parser.default_config_files = [defaults]
    subparser.add_argument("--test")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("test", subparser)

    cfg = parser.parse_args([], with_meta=False)
    assert cfg == Namespace(subcommand="test", test=Namespace(test="value"))
    cfg = parser.parse_args(["test", "--test=x"], with_meta=False)
    assert cfg == Namespace(subcommand="test", test=Namespace(test="x"))


def test_subsubcommand_default_config_repeated_keys(parser, subparser, tmp_cwd):
    defaults = tmp_cwd / "defaults.json"
    defaults.write_text('{"test":{"test":{"test":"value"}}}')
    parser.default_config_files = [defaults]
    subsubparser = ArgumentParser()
    subsubparser.add_argument("--test")
    subcommands1 = parser.add_subcommands()
    subcommands1.add_subcommand("test", subparser)
    subcommands2 = subparser.add_subcommands()
    subcommands2.add_subcommand("test", subsubparser)

    cfg = parser.parse_args([], with_meta=False)
    assert cfg.as_dict() == {"subcommand": "test", "test": {"subcommand": "test", "test": {"test": "value"}}}
    cfg = parser.parse_args(["test", "test", "--test=x"], with_meta=False)
    assert cfg.as_dict() == {"subcommand": "test", "test": {"subcommand": "test", "test": {"test": "x"}}}


def test_subcommand_required_arg_in_default_config(parser, subparser, tmp_cwd):
    Path("config.yaml").write_text("output: test\nprepare:\n  media: test\n")
    parser.default_config_files = ["config.yaml"]
    parser.add_argument("--output", required=True)
    subcommands = parser.add_subcommands()
    subparser = ArgumentParser()
    subparser.add_argument("--media", required=True)
    subcommands.add_subcommand("prepare", subparser)
    cfg = parser.parse_args([])
    assert str(cfg.__default_config__) == "config.yaml"
    assert strip_meta(cfg) == Namespace(output="test", prepare=Namespace(media="test"), subcommand="prepare")


def test_subsubcommands_parse_args(subtests):
    parser_s1_a = ArgumentParser(exit_on_error=False)
    parser_s1_a.add_argument("--os1a", default="os1a_def")

    parser_s2_b = ArgumentParser(exit_on_error=False)
    parser_s2_b.add_argument("--os2b", default="os2b_def")

    parser = ArgumentParser(prog="app", exit_on_error=False, default_meta=False)
    subcommands1 = parser.add_subcommands()
    subcommands1.add_subcommand("a", parser_s1_a)

    subcommands2 = parser_s1_a.add_subcommands()
    subcommands2.add_subcommand("b", parser_s2_b)

    with subtests.test("errors"):
        pytest.raises(ArgumentError, lambda: parser.parse_args([]))
        pytest.raises(ArgumentError, lambda: parser.parse_args(["a"]))

    with subtests.test("subsubcommand"):
        cfg = parser.parse_args(["a", "b"]).as_dict()
        assert cfg == {
            "subcommand": "a",
            "a": {"subcommand": "b", "os1a": "os1a_def", "b": {"os2b": "os2b_def"}},
        }

    with subtests.test("sub-optional"):
        cfg = parser.parse_args(["a", "--os1a=os1a_arg", "b"]).as_dict()
        assert cfg == {
            "subcommand": "a",
            "a": {"subcommand": "b", "os1a": "os1a_arg", "b": {"os2b": "os2b_def"}},
        }

    with subtests.test("subsub-optional"):
        cfg = parser.parse_args(["a", "b", "--os2b=os2b_arg"]).as_dict()
        assert cfg == {
            "subcommand": "a",
            "a": {"subcommand": "b", "os1a": "os1a_def", "b": {"os2b": "os2b_arg"}},
        }


def test_subsubcommands_wrong_add_order(parser):
    parser_s1_a = ArgumentParser()
    parser_s2_b = ArgumentParser()

    subcommands2 = parser_s1_a.add_subcommands()
    subcommands2.add_subcommand("b", parser_s2_b)

    subcommands1 = parser.add_subcommands()
    with pytest.raises(ValueError) as ctx:
        subcommands1.add_subcommand("a", parser_s1_a)
    ctx.match("Multiple levels of subcommands must be added in level order")


def test_subcommands_custom_instantiator(parser, subparser, subtests):
    subparser.add_argument("--cls", type=CustomInstantiationBase)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("cmd", subparser)

    with subtests.test("main parser"):
        parser.add_instantiator(instantiator("main parser"), CustomInstantiationBase)
        cfg = parser.parse_args(["cmd", "--cls", "CustomInstantiationBase"])
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.cmd.cls, CustomInstantiationBase)
        assert init.cmd.cls.call == "main parser"

    with subtests.test("subparser"):
        subparser.add_instantiator(instantiator("subparser"), CustomInstantiationBase)
        cfg = parser.parse_args(["cmd", "--cls", "CustomInstantiationBase"])
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.cmd.cls, CustomInstantiationBase)
        assert init.cmd.cls.call == "subparser"
