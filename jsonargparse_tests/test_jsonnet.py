from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from jsonargparse import (
    ActionConfigFile,
    ActionJsonnet,
    ActionJsonSchema,
    ArgumentError,
    ArgumentParser,
    strip_meta,
)
from jsonargparse._optionals import jsonnet_support
from jsonargparse_tests.conftest import get_parser_help, skip_if_jsonschema_unavailable


@pytest.fixture(autouse=True)
def skip_if_jsonnet_unavailable():
    if not jsonnet_support:
        pytest.skip("jsonnet package is required")


example_1_jsonnet = """
local make_record(num) = {
    'ref': '#'+(num+1),
    'val': 3*(num/2)+5,
};

{
  'param': 654,
  'records': [make_record(n) for n in std.range(0, 8)],
}
"""

example_2_jsonnet = """
local param = std.extVar('param');

local make_record(num) = {
    'ref': '#'+(num+1),
    'val': 3*(num/2)+5,
};

{
  'param': param,
  'records': [make_record(n) for n in std.range(0, 8)],
}
"""

records_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
            "val": {"type": "number"},
        },
    },
}

example_schema = {
    "type": "object",
    "properties": {
        "param": {"type": "integer"},
        "records": records_schema,
    },
}


# test parser mode jsonnet


@skip_if_jsonschema_unavailable
def test_parser_mode_jsonnet(tmp_path):
    parser = ArgumentParser(parser_mode="jsonnet", exit_on_error=False)
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--param", type=int)
    parser.add_argument("--records", action=ActionJsonSchema(schema=records_schema))

    jsonnet_file = tmp_path / "example.jsonnet"
    jsonnet_file.write_text(example_1_jsonnet)

    cfg = parser.parse_args([f"--cfg={jsonnet_file}"])
    assert 654 == cfg.param
    assert 9 == len(cfg.records)
    assert "#8" == cfg.records[-2]["ref"]
    assert 15.5 == cfg.records[-2]["val"]

    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg", "{}}"]))


def test_parser_mode_jsonnet_import_libsonnet(parser, tmp_cwd):
    parser.parser_mode = "jsonnet"
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--name", type=str, default="Lucky")
    parser.add_argument("--prize", type=int, default=100)

    Path("conf").mkdir()
    Path("conf", "name.libsonnet").write_text('"Mike"')

    config_path = Path("conf", "test.jsonnet")
    config_path.write_text('local name = import "name.libsonnet"; {"name": name, "prize": 80}')

    cfg = parser.parse_args([f"--cfg={config_path}"])
    assert cfg.name == "Mike"
    assert cfg.prize == 80
    assert str(cfg.cfg[0]) == str(config_path)


def test_parser_mode_jsonnet_subconfigs(parser, tmp_cwd):
    class Class:
        def __init__(self, name: str = "Lucky", prize: int = 100):
            pass

    parser.parser_mode = "jsonnet"
    parser.add_class_arguments(Class, "group", sub_configs=True)

    Path("conf").mkdir()
    Path("conf", "name.libsonnet").write_text('"Mike"')
    config_path = Path("conf", "test.jsonnet")
    config_path.write_text('local name = import "name.libsonnet"; {"name": name, "prize": 80}')

    cfg = parser.parse_args([f"--group={config_path}"])
    assert cfg.group.name == "Mike"
    assert cfg.group.prize == 80


# test action jsonnet


@skip_if_jsonschema_unavailable
def test_action_jsonnet(parser):
    parser.add_argument("--input.ext_vars", type=dict)
    parser.add_argument(
        "--input.jsonnet",
        action=ActionJsonnet(ext_vars="input.ext_vars", schema=json.dumps(example_schema)),
    )

    cfg2 = parser.parse_args(["--input.ext_vars", '{"param": 123}', "--input.jsonnet", example_2_jsonnet])
    assert 123 == cfg2.input.jsonnet["param"]
    assert 9 == len(cfg2.input.jsonnet["records"])
    assert "#8" == cfg2.input.jsonnet["records"][-2]["ref"]
    assert 15.5 == cfg2.input.jsonnet["records"][-2]["val"]

    cfg1 = parser.parse_args(["--input.jsonnet", example_1_jsonnet])
    assert cfg1.input.jsonnet["records"] == cfg2.input.jsonnet["records"]

    with pytest.raises(ArgumentError):
        parser.parse_args(["--input.ext_vars", '{"param": "a"}', "--input.jsonnet", example_2_jsonnet])
    with pytest.raises(ArgumentError):
        parser.parse_args(["--input.jsonnet", example_2_jsonnet])


def test_action_jsonnet_save_config_metadata(parser, tmp_path):
    parser.add_argument("--ext_vars", type=dict)
    parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))
    parser.add_argument("--cfg", action=ActionConfigFile)

    jsonnet_file = tmp_path / "example.jsonnet"
    jsonnet_file.write_text(example_2_jsonnet)
    output_yaml = tmp_path / "output" / "main.yaml"
    output_jsonnet = tmp_path / "output" / "example.jsonnet"
    (tmp_path / "output").mkdir()

    # save the config with metadata and verify it is saved as two files
    cfg = parser.parse_args(["--ext_vars", '{"param": 123}', f"--jsonnet={jsonnet_file}"])
    assert str(cfg.jsonnet["__path__"]) == str(jsonnet_file)
    parser.save(cfg, output_yaml)
    assert output_yaml.is_file()
    assert output_jsonnet.is_file()

    # rewrite the config to make sure that ext_vars is after jsonnet
    main_cfg = yaml.safe_load(output_yaml.read_text())
    main_cfg = {k: main_cfg[k] for k in ["jsonnet", "ext_vars"]}
    output_yaml.write_text(yaml.safe_dump(main_cfg, sort_keys=False))

    # parse using saved config and verify result is the same
    cfg2 = parser.parse_args([f"--cfg={output_yaml}"])
    cfg2.cfg = None
    assert strip_meta(cfg) == strip_meta(cfg2)

    # save the config without metadata and verify it is saved as a single file
    output_yaml.unlink()
    output_jsonnet.unlink()
    parser.save(strip_meta(cfg), output_yaml)
    assert output_yaml.is_file()
    assert not output_jsonnet.is_file()

    # parse using saved config and verify result is the same
    cfg3 = parser.parse_args([f"--cfg={output_yaml}"])
    cfg3.cfg = None
    assert strip_meta(cfg) == strip_meta(cfg3)


@skip_if_jsonschema_unavailable
def test_action_jsonnet_in_help(parser):
    parser.add_argument(
        "--jsonnet",
        action=ActionJsonnet(schema=example_schema),
        help="schema: %s",
    )
    help_str = get_parser_help(parser)
    schema = re.sub(
        "^.*schema:([^()]+)[^{}]*$",
        r"\1",
        help_str.replace("\n", " "),
    )
    assert example_schema == json.loads(schema)


def test_action_jsonnet_parse_method():
    parsed = ActionJsonnet().parse(example_2_jsonnet, ext_vars={"param": 123})
    assert 123 == parsed["param"]
    assert 9 == len(parsed["records"])
    assert "#8" == parsed["records"][-2]["ref"]
    assert 15.5 == parsed["records"][-2]["val"]


def test_action_jsonnet_ext_vars_default(parser):
    parser.add_argument("--ext_vars", type=dict, default={"param": 432})
    parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))
    cfg = parser.parse_args(["--jsonnet", example_2_jsonnet])
    assert 432 == cfg.jsonnet["param"]


def test_action_jsonnet_ext_vars_not_defined(parser):
    with pytest.raises(ValueError) as ctx:
        parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))
    ctx.match("No argument found for ext_vars")


def test_action_jsonnet_ext_vars_invalid_type(parser):
    parser.add_argument("--ext_vars", type=list)
    with pytest.raises(ValueError) as ctx:
        parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))
    ctx.match("Type for ext_vars='ext_vars' argument must be dict")


def test_action_jsonnet_ext_vars_invalid_default(parser):
    parser.add_argument("--ext_vars", type=dict, default="none")
    with pytest.raises(ValueError) as ctx:
        parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))
    ctx.match("Default value for the ext_vars='ext_vars' argument must be dict or None")


# other tests


@skip_if_jsonschema_unavailable
def test_action_jsonnet_schema_dict_or_str():
    action1 = ActionJsonnet(schema=example_schema)
    action2 = ActionJsonnet(schema=json.dumps(example_schema))
    assert action1._validator.schema == action2._validator.schema


@skip_if_jsonschema_unavailable
def test_action_jsonnet_init_failures():
    pytest.raises(ValueError, lambda: ActionJsonnet(ext_vars=2))
    pytest.raises(ValueError, lambda: ActionJsonnet(schema="." + json.dumps(example_schema)))
    from jsonschema.exceptions import SchemaError

    pytest.raises(SchemaError, lambda: ActionJsonnet(schema="."))
