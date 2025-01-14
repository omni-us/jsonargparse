from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser, get_loader, set_dumper, set_loader
from jsonargparse._common import parser_context
from jsonargparse._loaders_dumpers import load_value, loaders, yaml_dump
from jsonargparse._optionals import omegaconf_support, pyyaml_available, toml_dump_available, toml_load_available
from jsonargparse_tests.conftest import get_parse_args_stdout, json_or_yaml_dump, json_or_yaml_load, skip_if_no_pyyaml

if pyyaml_available:
    import yaml


@skip_if_no_pyyaml
def test_set_dumper_custom_yaml(parser):
    parser.add_argument("--list", type=List[int])

    def custom_yaml_dump(data) -> str:
        return yaml.safe_dump(data, default_flow_style=True)

    with patch.dict("jsonargparse._loaders_dumpers.dumpers"):
        set_dumper("yaml_custom", custom_yaml_dump)
        cfg = parser.parse_args(["--list=[1,2,3]"])
        dump = parser.dump(cfg, format="yaml_custom")
        assert dump == "{list: [1, 2, 3]}\n"


def test_yaml_implicit_mapping_values_disabled(parser):
    parser.add_argument("--val", type=str)
    assert "{one}" == parser.parse_args(["--val={one}"]).val
    assert "{one,two,three}" == parser.parse_args(["--val={one,two,three}"]).val


class Bar:
    def __init__(self, x: str):
        pass


def test_yaml_implicit_null_disabled(parser):
    parser.add_argument("--bar", type=Bar)
    cfg = parser.parse_args(["--bar=Bar", "--bar.x=Foo:"])
    assert "Foo:" == cfg.bar.init_args.x


def test_invalid_parser_mode():
    pytest.raises(ValueError, lambda: ArgumentParser(parser_mode="invalid"))


def test_get_loader():
    from jsonargparse._loaders_dumpers import jsonnet_load

    assert jsonnet_load is get_loader("jsonnet")


def test_set_loader_parser_mode_subparsers(parser, subparser):
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("sub", subparser)

    with patch.dict("jsonargparse._loaders_dumpers.loaders"):
        set_loader("custom", yaml.safe_load if pyyaml_available else json.loads)
        parser.parser_mode = "custom"
        assert "custom" == parser.parser_mode
        assert "custom" == subparser.parser_mode


@skip_if_no_pyyaml
def test_dump_header_yaml(parser):
    parser.add_argument("--int", type=int, default=1)
    parser.dump_header = ["line 1", "line 2"]
    dump = parser.dump(parser.get_defaults())
    assert dump == "# line 1\n# line 2\nint: 1\n"


def test_dump_header_json(parser):
    parser.add_argument("--int", type=int, default=1)
    parser.dump_header = ["line 1", "line 2"]
    dump = parser.dump(parser.get_defaults(), format="json")
    assert dump == '{"int":1}'


def test_dump_header_invalid(parser):
    with pytest.raises(ValueError):
        parser.dump_header = True


@skip_if_no_pyyaml
def test_load_value_dash():
    with parser_context(load_value_mode="yaml"):
        assert "-" == load_value("-")
        assert " -  " == load_value(" -  ")


@dataclass
class CustomData:
    fn: dict


class CustomContainer:
    def __init__(self, data: CustomData):
        self.data = data


def custom_loader(data):
    if pyyaml_available:
        data = yaml.safe_load(data)
    else:
        data = json.loads(data)
    if isinstance(data, dict) and "fn" in data:
        data["fn"] = {k: custom_loader for k in data["fn"]}
    return data


def custom_dumper(data):
    if "data" in data and "fn" in data["data"]:
        data["data"]["fn"] = {k: "dumped" for k in data["data"]["fn"]}
    return json_or_yaml_dump(data)


def test_nested_parser_mode(parser):
    set_loader("custom", custom_loader)
    set_dumper("custom", custom_dumper)
    parser.parser_mode = "custom"
    parser.add_argument("--custom", type=CustomContainer)
    cfg = parser.parse_args(['--custom.data={"fn": {"key": "value"}}'])
    assert cfg.custom.init_args.data["fn"]["key"] is custom_loader
    dump = json_or_yaml_load(parser.dump(cfg))
    assert dump["custom"]["init_args"]["data"] == {"fn": {"key": "dumped"}}


# toml tests


toml_config = """
root = "-"

[group]
child1 = 1.2
child2 = [ 3.0, 4.5,]
"""


@pytest.mark.skipif(not toml_load_available, reason="tomllib or toml package is required")
def test_toml_parse_args_config(parser, tmp_cwd):
    parser.parser_mode = "toml"
    config_path = Path("config.toml")
    config_path.write_text(toml_config)
    parser.add_argument("--cfg", action="config")
    parser.add_argument("--root", type=str)
    parser.add_argument("--group.child1", type=float)
    parser.add_argument("--group.child2", type=List[float])
    cfg = parser.parse_args([f"--cfg={config_path}"])
    assert cfg.root == "-"
    assert cfg.group.as_dict() == {"child1": 1.2, "child2": [3.0, 4.5]}


@pytest.mark.skipif(not toml_dump_available, reason="toml package is required")
def test_toml_print_config(parser):
    parser.parser_mode = "toml"
    parser.add_argument("--config", action="config")
    parser.add_argument("--root", type=str, default="-")
    parser.add_argument("--group.child1", type=float, default=1.2)
    parser.add_argument("--group.child2", type=List[float], default=[3.0, 4.5])
    out = get_parse_args_stdout(parser, ["--print_config"])
    assert out.strip() == toml_config.strip()


# omegaconf tests


@pytest.mark.skipif(not omegaconf_support, reason="omegaconf package is required")
def test_parser_mode_omegaconf_interpolation():
    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--server.host", type=str)
    parser.add_argument("--server.port", type=int)
    parser.add_argument("--client.url", type=str)
    parser.add_argument("--config", action="config")

    config = {
        "server": {
            "host": "localhost",
            "port": 80,
        },
        "client": {
            "url": "http://${server.host}:${server.port}/",
        },
    }
    cfg = parser.parse_args([f"--config={yaml_dump(config)}"])
    assert cfg.client.url == "http://localhost:80/"
    assert "url: http://localhost:80/" in parser.dump(cfg)


@pytest.mark.skipif(not omegaconf_support, reason="omegaconf package is required")
def test_parser_mode_omegaconf_interpolation_in_subcommands(parser, subparser):
    subparser.add_argument("--config", action="config")
    subparser.add_argument("--source", type=str)
    subparser.add_argument("--target", type=str)

    parser.parser_mode = "omegaconf"
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("sub", subparser)

    config = {
        "source": "hello",
        "target": "${source}",
    }
    cfg = parser.parse_args(["sub", f"--config={yaml_dump(config)}"])
    assert cfg.sub.target == "hello"


@pytest.mark.skipif(
    not (omegaconf_support and "JSONARGPARSE_OMEGACONF_FULL_TEST" in os.environ),
    reason="only for omegaconf as the yaml loader",
)
def test_omegaconf_as_yaml_loader():
    assert loaders["yaml"] is loaders["omegaconf"]
