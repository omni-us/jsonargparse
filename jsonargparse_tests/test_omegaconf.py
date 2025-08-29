from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser, Namespace
from jsonargparse._common import parser_context
from jsonargparse._loaders_dumpers import loaders, yaml_dump
from jsonargparse._optionals import omegaconf_support
from jsonargparse.typing import Path_fr
from jsonargparse_tests.conftest import get_parser_help

if omegaconf_support:
    from omegaconf import OmegaConf

skip_if_omegaconf_unavailable = pytest.mark.skipif(
    not omegaconf_support,
    reason="omegaconf package is required",
)


@pytest.mark.skipif(
    not (omegaconf_support and "JSONARGPARSE_OMEGACONF_FULL_TEST" in os.environ),
    reason="only for omegaconf as the yaml loader",
)
def test_omegaconf_as_yaml_loader():
    assert loaders["yaml"] is loaders["omegaconf"]


@skip_if_omegaconf_unavailable
@pytest.mark.parametrize("mode", ["omegaconf", "omegaconf+"])
def test_omegaconf_interpolation(mode):
    parser = ArgumentParser(parser_mode=mode)
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


@skip_if_omegaconf_unavailable
@pytest.mark.parametrize("mode", ["omegaconf", "omegaconf+"])
def test_omegaconf_interpolation_in_subcommands(mode, parser, subparser):
    subparser.add_argument("--config", action="config")
    subparser.add_argument("--source", type=str)
    subparser.add_argument("--target", type=str)

    parser.parser_mode = mode
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("sub", subparser)

    config = {
        "source": "hello",
        "target": "${source}" if mode == "omegaconf" else "${.source}",
    }
    cfg = parser.parse_args(["sub", f"--config={yaml_dump(config)}"])
    assert cfg.sub.target == "hello"


@dataclass
class Server:
    host: str = "localhost"
    port: int = 80


@dataclass
class Client:
    url: str = "http://example.com:8080"


@skip_if_omegaconf_unavailable
def test_omegaconf_global_interpolation(parser):
    parser.parser_mode = "omegaconf+"
    parser.add_class_arguments(Server, "server")
    parser.add_class_arguments(Client, "client")

    config = {"url": "http://${server.host}:${..server.port}/"}
    cfg = parser.parse_args([f"--client={yaml_dump(config)}"])
    assert cfg.client == Namespace(url="http://localhost:80/")

    cfg = parser.parse_args([f"--client={yaml_dump(config)}", "--server.port=9000"])
    assert cfg.client == Namespace(url="http://localhost:9000/")


@skip_if_omegaconf_unavailable
def test_omegaconf_global_resolver_config(parser):
    OmegaConf.register_new_resolver("increment", lambda x: x + 1)

    parser.parser_mode = "omegaconf+"
    parser.add_argument("--config", action="config")
    parser.add_argument("--value", type=int, default=0)
    parser.add_argument("--incremented", type=int, default=0)

    assert parser.parse_args([]) == Namespace(config=None, value=0, incremented=0)

    config = {"value": 1, "incremented": "${increment:${value}}"}
    cfg = parser.parse_args([f"--config={yaml_dump(config)}", "--value=5"])
    assert cfg == Namespace(value=5, incremented=6)  # currently config is lost

    OmegaConf.clear_resolver("increment")


@skip_if_omegaconf_unavailable
def test_omegaconf_global_resolver_argument(parser):
    def const(expr: str):
        allowed = {"pi": math.pi}
        return eval(expr, {"__builtins__": None}, allowed)

    OmegaConf.register_new_resolver("const", const)

    parser.parser_mode = "omegaconf+"
    parser.add_argument("--value", type=float)
    cfg = parser.parse_args(["--value=${const:3*pi/4}"])
    assert cfg.value == 3 * math.pi / 4

    OmegaConf.clear_resolver("const")


@skip_if_omegaconf_unavailable
@patch.dict(os.environ, {"X": "true"})
def test_omegaconf_global_resolver_default(parser):
    parser.parser_mode = "omegaconf+"
    action = parser.add_argument("--env", type=bool, default="${oc.env:X}")
    assert action.default == "${oc.env:X}"

    help_str = get_parser_help(parser)
    assert "default: ${oc.env:X}" in help_str

    cfg = parser.parse_args([])
    assert cfg.env is True


@dataclass
class Nested:
    path: Path_fr


@skip_if_omegaconf_unavailable
@patch.dict(os.environ, {"X": "Y"})
def test_omegaconf_global_path_preserve_relative(parser, tmp_cwd):
    import yaml

    parser.parser_mode = "omegaconf+"
    parser.add_class_arguments(Nested, "nested")
    parser.add_argument("--env")

    subdir = Path("sub")
    subdir.mkdir()
    (subdir / "file").touch()
    nested = subdir / "nested.json"
    nested.write_text(json.dumps({"path": "file"}))

    cfg = parser.parse_args([f"--nested={nested}", "--env=${oc.env:X}"])
    assert cfg.env == "Y"
    assert cfg.nested.path.relative == "file"
    assert cfg.nested.path.cwd == str(tmp_cwd / subdir)

    with parser_context(path_dump_preserve_relative=True):
        dump = yaml.safe_load(parser.dump(cfg))["nested"]["path"]
    assert dump == {"relative": "file", "cwd": str(tmp_cwd / subdir)}
