from __future__ import annotations

import json
import os
import pickle
from calendar import Calendar
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from random import randint, shuffle
from unittest.mock import patch

import pytest
import yaml

from jsonargparse import (
    SUPPRESS,
    ActionConfigFile,
    ActionJsonnet,
    ActionJsonSchema,
    ActionParser,
    ArgumentError,
    ArgumentParser,
    Namespace,
    set_config_read_mode,
    strip_meta,
)
from jsonargparse._formatters import get_env_var
from jsonargparse._optionals import jsonnet_support, jsonschema_support, ruyaml_support
from jsonargparse.typing import Path_fc, Path_fr, path_type
from jsonargparse_tests.conftest import (
    capture_logs,
    get_parse_args_stderr,
    get_parse_args_stdout,
    get_parser_help,
    is_cpython,
    responses_activate,
    skip_if_docstring_parser_unavailable,
    skip_if_fsspec_unavailable,
    skip_if_not_posix,
    skip_if_responses_unavailable,
)


def test_parse_args_simple(parser):
    parser.add_argument("--op", type=int)
    assert parser.parse_args(["--op=1"]) == Namespace(op=1)
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op=1.1"]))


def test_parse_args_nested(parser):
    parser.add_argument("--l1.l2.op", type=float)
    assert parser.parse_args(["--l1.l2.op=2.1"]).l1.l2.op == 2.1
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--l1.l2.op=x"]))


def test_parse_args_unrecognized_arguments(parser):
    err = get_parse_args_stderr(parser, ["--unrecognized"])
    assert "Unrecognized arguments:" in err


def test_parse_args_from_sys_argv(parser):
    parser.add_argument("--op", type=str)
    with patch("sys.argv", ["", "--op", "hello"]):
        assert parser.parse_args() == Namespace(op="hello")


def test_parse_args_invalid_args(parser):
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args([{}])
    ctx.match("expected to be strings")


def test_parse_args_base_namespace(parser):
    parser.add_argument("--op1")
    parser.add_argument("--op2")
    cfg = parser.parse_args(["--op1=abc"], Namespace(op2="xyz"))
    assert cfg == Namespace(op1="abc", op2="xyz")


def test_parse_args_unexpected_kwarg(parser):
    with pytest.raises(ValueError):
        parser.parse_args([], unexpected=True)


def test_parse_args_nargs_plus(parser):
    parser.add_argument("--val", nargs="+", type=int)
    assert [9] == parser.parse_args(["--val", "9"]).val
    assert [3, 6, 2] == parser.parse_args(["--val", "3", "6", "2"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--val"]))


def test_parse_args_nargs_asterisk(parser):
    parser.add_argument("--val", nargs="*", type=float)
    assert [5.2, 1.9] == parser.parse_args(["--val", "5.2", "1.9"]).val
    assert [] == parser.parse_args(["--val"]).val


def test_parse_args_nargs_questionmark(parser):
    parser.add_argument("--val", nargs="?", type=str)
    assert "~" == parser.parse_args(["--val", "~"]).val
    assert None is parser.parse_args(["--val"]).val


def test_parse_args_nargs_number(parser):
    parser.add_argument("--one", nargs=1)
    parser.add_argument("--two", nargs=2)
    assert parser.parse_args(["--one", "-"]) == Namespace(one=["-"], two=None)
    assert parser.parse_args(["--two", "q", "p"]) == Namespace(one=None, two=["q", "p"])
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--two", "-"]))


def test_parse_args_positional_nargs_questionmark(parser):
    parser.add_argument("pos1")
    parser.add_argument("pos2", nargs="?")
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args([])
    ctx.match('"pos1" is required')
    assert parser.parse_args(["v1"]) == Namespace(pos1="v1", pos2=None)
    assert parser.parse_args(["v1", "v2"]) == Namespace(pos1="v1", pos2="v2")


def test_parse_args_positional_nargs_plus(parser):
    parser.add_argument("pos1")
    parser.add_argument("pos2", nargs="+")
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["v1"])
    ctx.match('"pos2" is required')
    assert parser.parse_args(["v1", "v2", "v3"]) == Namespace(pos1="v1", pos2=["v2", "v3"])


def test_parse_args_positional_config(parser):
    parser.add_argument("pos1")
    parser.add_argument("pos2", nargs="+")
    parser.add_argument("--cfg", action=ActionConfigFile)
    cfg = parser.parse_args(["--cfg", '{"pos2": ["v2", "v3"]}', "v1"])
    assert cfg == Namespace(cfg=[None], pos1="v1", pos2=["v2", "v3"])


def test_parse_args_choices(parser):
    parser.add_argument("--ch1", choices="ABC")
    parser.add_argument("--ch2", type=str, choices=["v1", "v2"])
    cfg = parser.parse_args(["--ch1", "C", "--ch2", "v1"])
    assert cfg.as_dict() == {"ch1": "C", "ch2": "v1"}
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--ch1", "D"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--ch2", "v0"]))


def test_parse_args_choices_config(parser):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--ch1", choices="ABC")
    parser.add_argument("--ch2", type=str, choices=["v1", "v2"])
    assert parser.parse_args(["--cfg=ch1: B"]).ch1 == "B"
    assert parser.parse_args(["--cfg=ch2: v2"]).ch2 == "v2"
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg=ch1: D"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg=ch2: v0"]))


def test_parse_args_non_hashable_choice(parser):
    choices = {"A": 1, "B": 2}
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--ch1", choices=choices.keys())
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["--cfg=ch1: [1,2]"])
    ctx.match("not among choices")


def test_parse_args_choices_nargs_plus(parser):
    parser.add_argument("--ch", nargs="+", choices=["A", "B"])
    assert ["A", "B"] == parser.parse_args(["--ch", "A", "B"]).ch
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["--ch", "B", "X"])
    ctx.match("invalid choice")


def test_parse_object_simple(parser):
    parser.add_argument("--op", type=int)
    assert parser.parse_object({"op": 1}) == Namespace(op=1)
    pytest.raises(ArgumentError, lambda: parser.parse_object({"op": 1.1}))
    pytest.raises(ArgumentError, lambda: parser.parse_object({"undefined": True}))


def test_parse_object_nested(parser):
    parser.add_argument("--l1.l2.op", type=float)
    assert parser.parse_object({"l1": {"l2": {"op": 2.1}}}).l1.l2.op == 2.1
    pytest.raises(ArgumentError, lambda: parser.parse_object({"l1": {"l2": {"op": "x"}}}))


def test_env_prefix_from_prog_with_dashes():
    parser = ArgumentParser(prog="cli-name")
    action = parser.add_argument("--arg")
    assert get_env_var(parser, action) == "CLI_NAME_ARG"


def test_parse_env_simple():
    parser = ArgumentParser(prog="app", exit_on_error=False)
    parser.add_argument("--op", type=int)
    with patch.dict(os.environ, {"APP_OP": "1"}):
        assert parser.parse_env() == Namespace(op=1)
    pytest.raises(ArgumentError, lambda: parser.parse_env({"APP_OP": "1.1"}))


def test_parse_env_nested():
    parser = ArgumentParser(prog="app", exit_on_error=False)
    parser.add_argument("--l1.l2.op", type=float)
    assert parser.parse_env({"APP_L1__L2__OP": "2.1"}).l1.l2.op == 2.1
    pytest.raises(ArgumentError, lambda: parser.parse_env({"APP_L1__L2__OP": "x"}))


def test_parse_env_config(parser):
    parser.env_prefix = "app"
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--l1.num", type=int)
    cfg = parser.parse_env({"APP_CFG": '{"l1": {"num": 1}}'})
    assert cfg.cfg == [None]
    assert cfg.l1 == Namespace(num=1)
    pytest.raises(ArgumentError, lambda: parser.parse_env({"APP_CFG": '{"undefined": True}'}))


def test_parse_env_positional_nargs_plus(parser):
    parser.env_prefix = "app"
    parser.add_argument("req", nargs="+")
    assert parser.parse_env({"APP_REQ": "abc"}).req == ["abc"]
    assert parser.parse_env({"APP_REQ": '["abc", "xyz"]'}).req == ["abc", "xyz"]
    assert parser.parse_env({"APP_REQ": '[""","""]'}).req == ['[""","""]']


def test_default_env_property():
    parser = ArgumentParser()
    assert False is parser.default_env
    parser.default_env = True
    assert True is parser.default_env
    parser = ArgumentParser(default_env=True)
    assert True is parser.default_env
    parser.default_env = False
    assert False is parser.default_env
    with pytest.raises(ValueError) as ctx:
        parser.default_env = "invalid"
    ctx.match("default_env expects a boolean")


@patch.dict(os.environ, {"JSONARGPARSE_DEFAULT_ENV": "True"})
def test_default_env_override_true():
    parser = ArgumentParser(default_env=False)
    assert True is parser.default_env
    parser.default_env = False
    assert True is parser.default_env


@patch.dict(os.environ, {"JSONARGPARSE_DEFAULT_ENV": "False"})
def test_default_env_override_false():
    parser = ArgumentParser(default_env=True)
    assert False is parser.default_env
    parser.default_env = True
    assert False is parser.default_env


def test_env_prefix_true():
    parser = ArgumentParser(env_prefix=True, default_env=True, exit_on_error=False)
    parser.add_argument("--test_arg", type=str, required=True)

    with patch.dict(os.environ, {"TEST_ARG": "one"}):
        pytest.raises(ArgumentError, lambda: parser.parse_args([]))

    prefix = os.path.splitext(parser.prog)[0].upper()
    with patch.dict(os.environ, {f"{prefix}_TEST_ARG": "one"}):
        cfg = parser.parse_args([])
    assert "one" == cfg.test_arg


def test_env_prefix_false():
    parser = ArgumentParser(env_prefix=False, default_env=True)
    parser.add_argument("--test_arg", type=str, required=True)
    with patch.dict(os.environ, {"TEST_ARG": "one"}):
        cfg = parser.parse_args([])
    assert "one" == cfg.test_arg


def test_env_properties_set_invalid(parser):
    with pytest.raises(ValueError):
        parser.default_env = "invalid"
    with pytest.raises(ValueError):
        parser.env_prefix = lambda: "invalid"


def test_parse_string_simple(parser):
    parser.add_argument("--op", type=int)
    assert parser.parse_string('{"op": 1}').op == 1


def test_parse_string_simple_errors(parser):
    pytest.raises(ArgumentError, lambda: parser.parse_string('{"op": 1.1}'))
    pytest.raises(ArgumentError, lambda: parser.parse_string('{"undefined": true}'))
    pytest.raises(ArgumentError, lambda: parser.parse_string('"""'))
    pytest.raises(ArgumentError, lambda: parser.parse_string("not a dict"))


def test_parse_string_nested(parser):
    parser.add_argument("--l1.l2.op", type=float)
    assert parser.parse_string('{"l1": {"l2": {"op": 2.1}}}').l1.l2.op == 2.1
    pytest.raises(ArgumentError, lambda: parser.parse_string('{"l1": {"l2": {"op": "x"}}}'))


def test_parse_path_simple(parser, tmp_cwd):
    parser.add_argument("--op", type=int)
    path = Path("config.json")
    path.write_text('{"op": 1}')
    assert parser.parse_path(path) == Namespace(op=1)


def test_parse_path_simple_errors(parser, tmp_cwd):
    parser.add_argument("--op", type=int)
    path = Path("config.json")
    path.write_text('{"op": 1.1}')
    pytest.raises(ArgumentError, lambda: parser.parse_path(path))
    path.write_text('{"undefined": true}')
    pytest.raises(ArgumentError, lambda: parser.parse_path(path))


def test_parse_path_defaults(parser, tmp_cwd):
    parser.add_argument("--op1", type=int, default=1)
    parser.add_argument("--op2", type=float, default=2.3)
    path = Path("config.json")
    path.write_text('{"op1": 2}')
    assert parser.parse_path(path, defaults=True) == Namespace(op1=2, op2=2.3)
    assert parser.parse_path(path, defaults=False) == Namespace(op1=2)
    path.write_text('{"op2": 4.5}')
    assert parser.parse_path(path, defaults=False) == Namespace(op2=4.5)


@skip_if_not_posix
def test_parse_path_file_not_readable(parser, tmp_cwd):
    config_path = Path("config.json")
    config_path.touch()
    config_path.chmod(0)
    pytest.raises(TypeError, lambda: parser.parse_path(config_path))


def test_precedence_of_sources(tmp_cwd, subtests):
    input1_config_file = tmp_cwd / "input1.yaml"
    input2_config_file = tmp_cwd / "input2.yaml"
    default_config_file = tmp_cwd / "default.yaml"

    parser = ArgumentParser(prog="app", default_env=True, default_config_files=[default_config_file])
    parser.add_argument("--op1", default="from parser default")
    parser.add_argument("--op2")
    parser.add_argument("--cfg", action=ActionConfigFile)

    input1_config_file.write_text("op1: from input config file")
    input2_config_file.write_text("op2: unused")

    # parse_env precedence
    with subtests.test("parse_env parser default"):
        assert "from parser default" == parser.parse_env({}).op1
    with subtests.test("parse_env default config file"):
        default_config_file.write_text("op1: from default config file")
        assert "from default config file" == parser.parse_env({}).op1
    with subtests.test("parse_env environment config string"):
        env = {"APP_CFG": '{"op1": "from env config"}'}
        assert "from env config" == parser.parse_env(env).op1
    with subtests.test("parse_env environment variable"):
        env["APP_OP1"] = "from env var"
        assert "from env var" == parser.parse_env(env).op1
    default_config_file.unlink()

    # parse_path precedence
    with subtests.test("parse_path parser default"):
        assert "from parser default" == parser.parse_path(input2_config_file).op1
    with subtests.test("parse_path default config file"):
        default_config_file.write_text("op1: from default config file")
        assert "from default config file" == parser.parse_path(input2_config_file).op1
    env = {"APP_CFG": str(input1_config_file)}
    with subtests.test("parse_path environment config file"), patch.dict(os.environ, env):
        assert "from input config file" == parser.parse_path(input2_config_file).op1
    env["APP_OP1"] = "from env var"
    with subtests.test("parse_path environment variable"), patch.dict(os.environ, env):
        assert "from env var" == parser.parse_path(input2_config_file).op1
    env["APP_CFG"] = str(input2_config_file)
    with subtests.test("parse_path input config file"), patch.dict(os.environ, env):
        assert "from input config file" == parser.parse_path(input1_config_file).op1
    default_config_file.unlink()

    # parse_args precedence
    with subtests.test("parse_args parser default"):
        assert "from parser default" == parser.parse_args([]).op1
    with subtests.test("parse_args default config file"):
        default_config_file.write_text("op1: from default config file")
        assert "from default config file" == parser.parse_args([]).op1
    env = {"APP_CFG": str(input1_config_file)}
    with subtests.test("parse_args environment config file"), patch.dict(os.environ, env):
        assert "from input config file" == parser.parse_args([]).op1
    env["APP_OP1"] = "from env var"
    with subtests.test("parse_args environment variable"), patch.dict(os.environ, env):
        assert "from env var" == parser.parse_args([]).op1
    env["APP_CFG"] = str(input2_config_file)
    with subtests.test("parse_args input argument"), patch.dict(os.environ, env):
        assert "from arg" == parser.parse_args(["--op1", "from arg"]).op1
    with subtests.test("parse_args argument override config"), patch.dict(os.environ, env):
        assert "from arg" == parser.parse_args([f"--cfg={input1_config_file}", "--op1=from arg"]).op1
    with subtests.test("parse_args config override argument"), patch.dict(os.environ, env):
        assert "from input config file" == parser.parse_args(["--op1=from arg", f"--cfg={input1_config_file}"]).op1

    with subtests.test("parse_args config paths"), patch.dict(os.environ, env):
        cfg = parser.parse_args([f"--cfg={input1_config_file}"])
    config_paths = parser.get_config_files(cfg)
    assert str(default_config_file) == str(config_paths[0])
    assert str(input2_config_file) == str(config_paths[1])  # APP_CFG
    assert str(input1_config_file) == str(config_paths[2])  # --cfg


def test_non_positional_required(parser, subtests):
    group = parser.add_argument_group("Group 1")
    group.add_argument("--req1", required=True)
    parser.add_argument("--lev1.req2", required=True)

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        assert "[-h] --req1 REQ1 --lev1.req2 REQ2" in help_str
        assert "--lev1.req2 REQ2  (required)" in help_str

    with subtests.test("parse_env"):
        parser.env_prefix = "APP"
        cfg = parser.parse_env({"APP_REQ1": "val5", "APP_LEV1__REQ2": "val6"})
        assert cfg == Namespace(lev1=Namespace(req2="val6"), req1="val5")
        pytest.raises(ArgumentError, lambda: parser.parse_env({}))

    with subtests.test("parse_string"):
        cfg = parser.parse_string('{"req1":"val3","lev1":{"req2":"val4"}}')
        assert cfg == Namespace(lev1=Namespace(req2="val4"), req1="val3")
        pytest.raises(ArgumentError, lambda: parser.parse_string('{"lev1":{"req2":"val4"}}'))

    with subtests.test("parse_args"):
        cfg = parser.parse_args(["--req1", "val1", "--lev1.req2", "val2"])
        assert cfg == Namespace(lev1=Namespace(req2="val2"), req1="val1")
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--req1", "val1"]))

    with subtests.test("parse_args config"):
        parser.add_argument("--cfg", action=ActionConfigFile)
        cfg = parser.parse_args(["--cfg", '{"req1":"val1","lev1":{"req2":"val2"}}'])
        assert cfg == Namespace(cfg=[None], lev1=Namespace(req2="val2"), req1="val1")


@pytest.fixture
def dump_parser(parser):
    parser.add_argument("--op1", default=123)
    parser.add_argument("--op2", default="abc")
    return parser


def test_dump_complete(dump_parser):
    cfg1 = dump_parser.get_defaults()
    cfg2 = dump_parser.parse_string(dump_parser.dump(cfg1))
    assert cfg1 == cfg2


def test_dump_incomplete(dump_parser):
    dump = dump_parser.dump(Namespace(op1=456))
    assert "op1: 456" == dump.strip()


@pytest.mark.skipif(not is_cpython, reason="requires __setattr__ insertion order")
def test_dump_formats(dump_parser):
    cfg = dump_parser.get_defaults()
    assert dump_parser.dump(cfg) == "op1: 123\nop2: abc\n"
    assert dump_parser.dump(cfg, format="yaml") == dump_parser.dump(cfg)
    assert dump_parser.dump(cfg, format="json") == '{"op1":123,"op2":"abc"}'
    assert dump_parser.dump(cfg, format="json_indented") == '{\n  "op1": 123,\n  "op2": "abc"\n}\n'
    pytest.raises(ValueError, lambda: dump_parser.dump(cfg, format="invalid"))


def test_dump_skip_default_simple(dump_parser):
    assert "{}\n" == dump_parser.dump(dump_parser.get_defaults(), skip_default=True)
    assert "op2: xyz\n" == dump_parser.dump(Namespace(op1=123, op2="xyz"), skip_default=True)


def test_dump_skip_default_nested(parser):
    parser.add_argument("--g1.op1", type=int, default=123)
    parser.add_argument("--g1.op2", type=str, default="abc")
    parser.add_argument("--g2.op1", type=int, default=987)
    parser.add_argument("--g2.op2", type=str, default="xyz")
    assert parser.dump(parser.get_defaults(), skip_default=True) == "{}\n"
    assert parser.dump(parser.parse_args(["--g1.op1=0"]), skip_default=True) == "g1:\n  op1: 0\n"
    assert parser.dump(parser.parse_args(["--g2.op2=pqr"]), skip_default=True) == "g2:\n  op2: pqr\n"


def test_dump_order(parser, subtests):
    args = {}
    for num in range(50):
        args[num] = "".join(chr(randint(97, 122)) for _ in range(8))

    for num in range(len(args)):
        parser.add_argument("--" + args[num], default=num)

    with subtests.test("get_defaults"):
        cfg = parser.get_defaults()
        dump = parser.dump(cfg)
        assert dump == "\n".join(v + ": " + str(n) for n, v in args.items()) + "\n"

    with subtests.test("parse_string"):
        rand = list(range(len(args)))
        shuffle(rand)
        yaml = "\n".join(args[n] + ": " + str(n) for n in rand) + "\n"
        cfg = parser.parse_string(yaml)
        dump = parser.dump(cfg)
        assert dump == "\n".join(v + ": " + str(n) for n, v in args.items()) + "\n"


@pytest.fixture
def parser_schema_jsonnet(parser, example_parser):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--subparser", action=ActionParser(parser=example_parser))
    if jsonschema_support:
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        }
        parser.add_argument(
            "--schema",
            default={"a": 1, "b": 2},
            action=ActionJsonSchema(schema=schema),
        )
    if jsonnet_support:
        parser.add_argument(
            "--jsonnet",
            default={"c": 3, "d": 4},
            action=ActionJsonnet(ext_vars=None),
        )
    expected = parser.parse_args(["--subparser.bool=false", "--subparser.nums.val1=3"])
    subparser_body = yaml.safe_dump(expected.subparser.as_dict())
    schema_body = json.dumps(expected.schema) if jsonschema_support else ""
    jsonnet_body = json.dumps(expected.jsonnet) if jsonnet_support else ""
    return parser, expected, subparser_body, schema_body, jsonnet_body


@skip_if_responses_unavailable
@responses_activate
def test_parse_args_url_config(parser_schema_jsonnet):
    import responses

    set_config_read_mode(urls_enabled=True)
    parser, expected, subparser_body, schema_body, jsonnet_body = parser_schema_jsonnet

    base_url = "http://jsonargparse.com/"
    main_body = f"subparser: {base_url}subparser.yaml\n"
    if jsonschema_support:
        main_body += f"schema: {base_url}schema.yaml\n"
    if jsonnet_support:
        main_body += f"jsonnet: {base_url}jsonnet.yaml\n"

    for name, body in [
        ("main.yaml", main_body),
        ("subparser.yaml", subparser_body),
        ("schema.yaml", schema_body),
        ("jsonnet.yaml", jsonnet_body),
    ]:
        responses.add(responses.GET, base_url + name, status=200, body=body)
        responses.add(responses.HEAD, base_url + name, status=200)

    cfg = parser.parse_args([f"--cfg={base_url}main.yaml"], with_meta=False)
    assert expected.subparser == cfg.subparser
    if jsonschema_support:
        assert expected.schema == cfg.schema
    if jsonnet_support:
        assert expected.jsonnet == cfg.jsonnet

    set_config_read_mode(urls_enabled=False)


def test_save_multifile(parser_schema_jsonnet, subtests, tmp_cwd):
    parser, expected, subparser_body, schema_body, jsonnet_body = parser_schema_jsonnet

    in_dir = tmp_cwd / "input"
    out_dir = tmp_cwd / "output"
    in_dir.mkdir()
    out_dir.mkdir()
    main_file_in = in_dir / "main.yaml"
    subparser_file_in = in_dir / "subparser.yaml"
    schema_file_in = in_dir / "schema.json"
    jsonnet_file_in = in_dir / "jsonnet.json"
    main_file_out = out_dir / "main.yaml"
    subparser_file_out = out_dir / "subparser.yaml"
    schema_file_out = out_dir / "schema.json"
    jsonnet_file_out = out_dir / "jsonnet.json"

    main_body = "subparser: subparser.yaml\n"
    if jsonschema_support:
        main_body += "schema: schema.json\n"
    if jsonnet_support:
        main_body += "jsonnet: jsonnet.json\n"
    main_file_in.write_text(main_body)
    subparser_file_in.write_text(subparser_body)
    if jsonschema_support:
        schema_file_in.write_text(schema_body)
    if jsonnet_support:
        jsonnet_file_in.write_text(jsonnet_body)

    def rm_out_files():
        for file in [main_file_out, subparser_file_out, schema_file_out, jsonnet_file_out]:
            if file.is_file():
                file.unlink()

    with subtests.test("parse_path with metadata"):
        cfg1 = parser.parse_path(main_file_in, with_meta=True)
        assert expected == strip_meta(cfg1)
        assert str(cfg1.subparser["__path__"]) == "subparser.yaml"
        if jsonschema_support:
            assert str(cfg1.schema["__path__"]) == "schema.json"
        if jsonnet_support:
            assert str(cfg1.jsonnet["__path__"]) == "jsonnet.json"

    with subtests.test("save with metadata (multi-file)"):
        parser.save(cfg1, main_file_out)
        assert subparser_file_out.is_file()
        if jsonschema_support:
            assert schema_file_out.is_file()
        if jsonnet_support:
            assert jsonnet_file_out.read_text() == '{"c": 3, "d": 4}'
        cfg2 = parser.parse_path(main_file_out, with_meta=False)
        assert expected == cfg2

    with subtests.test("save without metadata (single-file)"):
        rm_out_files()
        parser.save(cfg1, main_file_out, multifile=False)
        cfg3 = parser.parse_path(main_file_out, with_meta=False)
        assert expected == cfg3

    if jsonschema_support:
        with subtests.test("save jsonschema yaml output"):
            rm_out_files()
            schema_yaml_out = out_dir / "schema.yaml"
            cfg1.schema["__path__"] = Path_fc(schema_yaml_out)
            parser.save(cfg1, main_file_out, multifile=True)
            assert schema_yaml_out.read_text() == "a: 1\nb: 2\n"


def test_save_overwrite(example_parser, tmp_cwd):
    cfg = example_parser.parse_args(["--nums.val1=7"])
    example_parser.save(cfg, "config.yaml")
    with pytest.raises(ValueError) as ctx:
        example_parser.save(cfg, "config.yaml")
    ctx.match("Refusing to overwrite")
    example_parser.save(cfg, "config.yaml", overwrite=True)


def test_save_subconfig_overwrite(parser, example_parser, tmp_cwd):
    Path("subparser.yaml").write_text(example_parser.dump(example_parser.get_defaults()))
    parser.add_argument("--subparser", action=ActionParser(parser=example_parser))
    cfg = parser.get_defaults()
    cfg.subparser.__path__ = Path_fr("subparser.yaml")
    with pytest.raises(ValueError) as ctx:
        parser.save(cfg, "main.yaml")
    ctx.match("Refusing to overwrite")
    parser.save(cfg, "main.yaml", overwrite=True)


def test_save_invalid_format(example_parser, tmp_cwd):
    cfg = example_parser.parse_args(["--nums.val2=-1.2"])
    with pytest.raises(ValueError) as ctx:
        example_parser.save(cfg, "invalid_format.yaml", format="invalid")
    ctx.match("Unknown output format")


def test_save_path_content(parser, tmp_cwd):
    parser.add_argument("--the.path", type=Path_fr)

    Path("pathdir").mkdir()
    Path("outdir").mkdir()
    in_file = Path("pathdir", "file.txt")
    out_yaml = Path("outdir", "saved.yaml")
    out_file = Path("outdir", "file.txt")
    in_file.write_text("file content")

    cfg = parser.parse_args([f"--the.path={in_file}"])
    parser.save_path_content.add("the.path")
    parser.save(cfg, out_yaml)

    assert out_yaml.read_text() == "the:\n  path: file.txt\n"
    assert out_file.read_text() == "file content"


@skip_if_fsspec_unavailable
def test_save_fsspec(example_parser):
    cfg = example_parser.parse_args(["--nums.val1=5"])
    example_parser.save(cfg, "memory://config.yaml", multifile=False)
    path = path_type("sr")("memory://config.yaml")
    assert cfg == example_parser.parse_string(path.get_content())

    with pytest.raises(NotImplementedError) as ctx:
        example_parser.save(cfg, "memory://config.yaml", multifile=True)
    ctx.match("multifile=True not supported")


@pytest.fixture
def print_parser(parser, subparser):
    parser.description = "cli tool"
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--v0", help=SUPPRESS, default="0")
    parser.add_argument("--v1", help="Option v1.", default=1)
    parser.add_argument("--g1.v2", help="Option v2.", default="2")
    subparser.add_argument("--v3")
    parser.add_argument("--g2", action=ActionParser(parser=subparser))
    return parser


def test_print_config_normal(print_parser):
    out = get_parse_args_stdout(print_parser, ["--print_config"])
    assert yaml.safe_load(out) == {"g1": {"v2": "2"}, "g2": {"v3": None}, "v1": 1}


def test_print_config_skip_null(print_parser):
    out = get_parse_args_stdout(print_parser, ["--print_config=skip_null"])
    assert yaml.safe_load(out) == {"g1": {"v2": "2"}, "g2": {}, "v1": 1}


@pytest.mark.skipif(not ruyaml_support, reason="ruyaml package is required")
@skip_if_docstring_parser_unavailable
def test_print_config_comments(print_parser):
    out = get_parse_args_stdout(print_parser, ["--print_config=comments"])
    assert "# cli tool" in out
    assert "# Option v1. (default: 1)" in out
    assert "# Option v2. (default: 2)" in out


def test_print_config_invalid_flag(print_parser):
    with pytest.raises(ArgumentError) as ctx:
        print_parser.parse_args(["--print_config=invalid"])
    ctx.match('Invalid option "invalid"')


def test_print_config_empty_default_config_file(print_parser, tmp_cwd):
    default_config_file = tmp_cwd / "defaults.yaml"
    default_config_file.touch()
    print_parser.default_config_files = [default_config_file]
    out = get_parse_args_stdout(print_parser, ["--print_config"])
    assert yaml.safe_load(out) == {"g1": {"v2": "2"}, "g2": {"v3": None}, "v1": 1}


def test_default_config_files(parser, subtests, tmp_cwd):
    default_config_file = tmp_cwd / "defaults.yaml"
    default_config_file.write_text("op1: from default config file\n")

    parser.default_config_files = [default_config_file]
    parser.add_argument("--op1", default="from parser default")
    parser.add_argument("--op2", default="from parser default")

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        assert "default values below are the ones overridden" in help_str
        assert "from default config file" in help_str
        assert "from parser default" in help_str

    with subtests.test("get_defaults"):
        cfg = parser.get_defaults()
        assert "from default config file" == cfg.op1
        assert "from parser default" == cfg.op2

    with subtests.test("get_default"):
        assert parser.get_default("op1") == "from default config file"
        parser.add_subclass_arguments(Calendar, "cal")
        with pytest.raises(KeyError) as ctx:
            parser.get_default("cal")
        ctx.match("does not specify a default")

    with subtests.test("set invalid"):
        with pytest.raises(ValueError) as ctx:
            parser.default_config_files = False
        assert "default_config_files expects None or List[str | os.PathLike]." == str(ctx.value)


def test_default_config_file_help_message_no_existing(parser, tmp_cwd):
    parser.default_config_files = ["defaults.yaml"]
    help_str = get_parser_help(parser)
    assert "no existing default config file" in help_str


def test_default_config_file_invalid_value(parser, tmp_cwd):
    default_config_file = Path("defaults.yaml")
    default_config_file.write_text("op2: v2\n")

    parser.default_config_files = [default_config_file]
    parser.add_argument("--op1", default="from default")

    with pytest.raises(ArgumentError) as ctx:
        parser.get_default("op1")
    ctx.match("Problem in default config file")

    help_str = get_parser_help(parser)
    assert "tried getting defaults considering default_config_files but failed" in help_str.replace("\n", " ")


@skip_if_not_posix
def test_default_config_file_unreadable(parser, tmp_cwd):
    default_config_file = Path("defaults.yaml")
    default_config_file.write_text("op1: from yaml\n")

    parser.default_config_files = [default_config_file]
    parser.add_argument("--op1", default="from default")

    assert parser.get_default("op1") == "from yaml"
    default_config_file.chmod(0)
    assert parser.get_default("op1") == "from default"


def test_default_config_files_pattern(parser, subtests, tmp_cwd):
    default_configs_pattern = tmp_cwd / "defaults_*.yaml"
    parser.default_config_files = [default_configs_pattern]
    parser.add_argument("--op1", default="from default")
    parser.add_argument("--op2", default="from default")

    with subtests.test("one config"):
        config_1 = tmp_cwd / "defaults_1.yaml"
        config_1.write_text("op1: from yaml 1\nop2: from yaml 1\n")

        cfg = parser.get_defaults()
        assert cfg.op1 == "from yaml 1"
        assert cfg.op2 == "from yaml 1"
        assert str(cfg.__default_config__) == str(config_1)

    with subtests.test("two configs"):
        config_2 = tmp_cwd / "defaults_2.yaml"
        config_2.write_text("op1: from yaml 2\n")

        cfg = parser.get_defaults()
        assert cfg.op1 == "from yaml 2"
        assert cfg.op2 == "from yaml 1"
        assert list(map(str, cfg.__default_config__)) == list(map(str, [config_1, config_2]))

    with subtests.test("three configs"):
        config_0 = tmp_cwd / "defaults_0.yaml"
        config_0.write_text("op2: from yaml 0\n")

        cfg = parser.get_defaults()
        assert cfg.op1 == "from yaml 2"
        assert cfg.op2 == "from yaml 1"
        assert list(map(str, cfg.__default_config__)) == list(map(str, [config_0, config_1, config_2]))

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        assert "defaults_*.yaml" in help_str
        assert "defaults_0.yaml" in help_str
        assert "defaults_1.yaml" in help_str
        assert "defaults_2.yaml" in help_str


def test_named_argument_groups(parser):
    parser.add_argument_group("Group 1", name="group1")
    parser.add_argument_group("Group 2", name="group2")
    assert {"group1", "group2"} == set(parser.groups.keys())
    with pytest.raises(ValueError) as ctx:
        parser.add_argument_group("Bad", name="group1")
    ctx.match("Group with name group1 already exists")


def test_set_get_defaults_single(parser):
    parser.add_argument("--v1")
    parser.set_defaults(v1=1)
    assert parser.get_default("v1") == 1


def test_set_get_defaults_multiple(parser, subparser, subtests):
    parser.add_argument("--v1", default="1")
    parser.add_argument("--g1.v2", default="2")
    subparser.add_argument("--g2.v3", default="3")
    parser.add_argument("--n", action=ActionParser(parser=subparser))

    with subtests.test("set_defaults"):
        parser.set_defaults({"g1.v2": "b", "n.g2.v3": "c"}, v1="a")

    with subtests.test("get_defaults"):
        cfg = parser.get_defaults()
        assert cfg.as_dict() == {"v1": "a", "g1": {"v2": "b"}, "n": {"g2": {"v3": "c"}}}

    with subtests.test("get_default"):
        assert parser.get_default("v1") == cfg.v1
        assert parser.get_default("g1.v2") == cfg.g1.v2
        assert parser.get_default("n.g2.v3") == cfg.n.g2.v3

    with subtests.test("set_defaults undefined key"):
        pytest.raises(KeyError, lambda: parser.set_defaults(v4="d"))

    with subtests.test("get_default undefined key"):
        pytest.raises(KeyError, lambda: parser.get_default("v4"))


def test_add_multiple_config_arguments_error(parser):
    parser.add_argument("--cfg1", action=ActionConfigFile)
    with pytest.raises(ValueError) as ctx:
        parser.add_argument("--cfg2", action=ActionConfigFile)
    ctx.match("only allowed to have a single")


def test_check_config_skip_none(parser):
    parser.add_argument("--op1", type=int)
    parser.add_argument("--op2", type=float)
    cfg = parser.parse_args(["--op2=2.2"])
    parser.check_config(cfg, skip_none=True)
    with pytest.raises(TypeError) as ctx:
        parser.check_config(cfg, skip_none=False)
    ctx.match("Expected a <class 'int'>")


def test_check_config_branch(example_parser):
    cfg = example_parser.get_defaults()
    example_parser.check_config(cfg.nums, branch="nums")
    cfg.nums.val1 = "invalid"
    with pytest.raises(TypeError) as ctx:
        example_parser.check_config(cfg.nums, branch="nums")
    ctx.match("Expected a <class 'int'>")


def test_merge_config(parser):
    for key in [1, 2, 3]:
        parser.add_argument(f"--op{key}", type=int)
    cfg_from = Namespace(op1=1, op2=None)
    cfg_to = Namespace(op1=None, op2=2, op3=3)
    cfg = parser.merge_config(cfg_from, cfg_to)
    assert cfg == Namespace(op1=1, op2=None, op3=3)


def test_strip_unknown(parser, example_parser):
    for key, default in example_parser.get_defaults().items():
        parser.add_argument("--" + key, default=default)
    parser.add_argument("--val", default="val_def")
    parser.add_argument("--lev1.opt4", default="opt3_def")
    parser.add_argument("--nums.val3", type=float, default=1.5)
    cfg = parser.parse_args([])
    cfg.__path__ = "some path"
    stripped = example_parser.strip_unknown(cfg)
    assert set(cfg.keys()) - set(stripped.keys()) == {"val", "nums.val3", "lev1.opt4"}
    assert stripped.pop("__path__") == "some path"


def test_exit_on_error():
    parser = ArgumentParser(exit_on_error=True)
    parser.add_argument("--val", type=int)
    err = StringIO()
    with redirect_stderr(err):
        assert 8 == parser.parse_args(["--val", "8"]).val
        pytest.raises(SystemExit, lambda: parser.parse_args(["--val", "eight"]))
    assert 'Parser key "val":' in err.getvalue()


def test_version_print():
    parser = ArgumentParser(prog="app", version="1.2.3")
    out = get_parse_args_stdout(parser, ["--version"])
    assert out == "app 1.2.3\n"


@patch.dict(os.environ, {"JSONARGPARSE_DEBUG": "true"})
def test_debug_environment_variable(logger):
    parser = ArgumentParser(logger=logger)
    parser.add_argument("--int", type=int)
    with pytest.raises(ArgumentError), capture_logs(logger) as logs:
        parser.parse_args(["--int=invalid"])
    assert "Debug enabled, thus raising exception instead of exit" in logs.getvalue()


def test_parse_known_args_not_implemented(parser):
    pytest.raises(NotImplementedError, lambda: parser.parse_known_args([]))


def test_parse_known_args_not_implemented_without_caller_module(parser):
    """
    Corner case when calling parse_known_args in IPython. The caller module will not exist.
    See https://github.com/omni-us/jsonargparse/pull/179
    """
    with patch("inspect.getmodule", return_value=None):
        pytest.raises(NotImplementedError, lambda: parser.parse_known_args([]))


def test_default_meta_property():
    parser = ArgumentParser()
    assert True is parser.default_meta
    parser.default_meta = False
    assert False is parser.default_meta
    parser = ArgumentParser(default_meta=False)
    assert False is parser.default_meta
    parser.default_meta = True
    assert True is parser.default_meta
    with pytest.raises(ValueError) as ctx:
        parser.default_meta = "invalid"
    ctx.match("default_meta expects a boolean")


def test_pickle_parser(example_parser):
    parser = pickle.loads(pickle.dumps(example_parser))
    assert example_parser.get_defaults() == parser.get_defaults()
