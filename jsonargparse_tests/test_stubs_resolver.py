from __future__ import annotations

import ast
import inspect
import sys
from calendar import Calendar, TextCalendar
from contextlib import contextmanager
from email.headerregistry import DateHeader
from importlib.util import find_spec
from ipaddress import ip_network
from random import Random, SystemRandom, uniform
from tarfile import TarFile
from unittest.mock import patch
from uuid import UUID, uuid5

import pytest

from jsonargparse._parameter_resolvers import get_signature_parameters as get_params
from jsonargparse._stubs_resolver import get_arg_type, get_mro_method_parent, get_stubs_resolver
from jsonargparse_tests.conftest import (
    capture_logs,
    get_parser_help,
    skip_if_no_pyyaml,
    skip_if_requests_unavailable,
)

torch_available = bool(find_spec("torch"))


@pytest.fixture(autouse=True)
def skip_if_typeshed_client_unavailable():
    if not find_spec("typeshed_client"):
        pytest.skip("typeshed-client package is required")


@contextmanager
def mock_stubs_missing_types():
    with patch("jsonargparse._parameter_resolvers.add_stub_types"):
        yield


@contextmanager
def mock_stubs_missing_resolver():
    with patch("jsonargparse._parameter_resolvers.get_stubs_resolver") as mock_instance:
        mock_instance.return_value.get_component_imported_info.return_value = None
        yield


@contextmanager
def inspect_signature_failure(mock_obj):
    original_inspect_signature = inspect.signature

    def inspect_signature(obj):
        if obj is mock_obj:
            raise ValueError("inspect_signature failed")
        return original_inspect_signature(obj)

    with patch("inspect.signature", side_effect=inspect_signature) as mock_signature:
        yield
        mock_signature.assert_called_with(mock_obj)


def get_param_types(params):
    return [(p.name, p.annotation) for p in params]


def get_param_names(params):
    return [p.name for p in params]


class WithoutParent:
    pass


@pytest.mark.parametrize(
    ["cls", "method", "expected"],
    [
        (WithoutParent, "__init__", None),
        (SystemRandom, "unknown", None),
        (Random, "__init__", Random),
        (SystemRandom, "__init__", Random),
        (SystemRandom, "uniform", Random),
        (SystemRandom, "getrandbits", SystemRandom),
    ],
)
def test_get_mro_method_parent(cls, method, expected):
    assert get_mro_method_parent(cls, method) is expected


@skip_if_requests_unavailable
def test_stubs_resolver_get_imported_info():
    resolver = get_stubs_resolver()
    imported_info = resolver.get_imported_info("requests.api.get")
    assert imported_info.source_module == ("requests", "api")
    imported_info = resolver.get_imported_info("requests.get")
    assert imported_info.source_module == ("requests", "api")


def test_get_params_class_without_inheritance():
    params = get_params(Calendar)
    assert [("firstweekday", int)] == get_param_types(params)
    with mock_stubs_missing_types():
        params = get_params(Calendar)
    assert [("firstweekday", inspect._empty)] == get_param_types(params)


def test_get_params_class_with_inheritance():
    params = get_params(TextCalendar)
    assert [("firstweekday", int)] == get_param_types(params)
    with mock_stubs_missing_types():
        params = get_params(TextCalendar)
    assert [("firstweekday", inspect._empty)] == get_param_types(params)


def test_get_params_method():
    params = get_params(Random, "randint")
    assert [("a", int), ("b", int)] == get_param_types(params)
    with mock_stubs_missing_types():
        params = get_params(Random, "randint")
    assert [("a", inspect._empty), ("b", inspect._empty)] == get_param_types(params)


def test_get_params_object_instance_method():
    params = get_params(uniform)
    assert [("a", float), ("b", float)] == get_param_types(params)
    with mock_stubs_missing_types():
        params = get_params(uniform)
    assert [("a", inspect._empty), ("b", inspect._empty)] == get_param_types(params)


def test_get_params_conditional_python_version():
    params = get_params(Random, "seed")
    assert ["a", "version"] == get_param_names(params)
    if sys.version_info >= (3, 10):
        assert "int | float | str | bytes | bytearray | None" == str(params[0].annotation)
    else:
        assert "typing.Union[int, float, str, bytes, bytearray, NoneType]" == str(params[0].annotation)
    assert int is params[1].annotation
    with mock_stubs_missing_types():
        params = get_params(Random, "seed")
    assert [("a", inspect._empty), ("version", inspect._empty)] == get_param_types(params)


@patch("jsonargparse._parameter_resolvers.get_stub_types")
def test_get_params_exec_failure(mock_get_stub_types):
    mock_get_stub_types.return_value = None
    params = get_params(Random, "seed")
    assert [("a", inspect._empty), ("version", inspect._empty)] == get_param_types(params)


def test_get_params_classmethod():
    params = get_params(TarFile, "open")
    expected = [
        "name",
        "mode",
        "fileobj",
        "bufsize",
        "format",
        "tarinfo",
        "dereference",
        "ignore_zeros",
        "encoding",
        "errors",
        "pax_headers",
        "debug",
        "errorlevel",
    ]
    if sys.version_info >= (3, 14):
        expected = expected[:4] + ["compresslevel", "preset"] + expected[4:]
    elif sys.version_info >= (3, 12):
        expected = expected[:4] + ["compresslevel"] + expected[4:]
    assert expected == get_param_names(params)[: len(expected)]
    if sys.version_info >= (3, 10):
        assert all(
            p.annotation is not inspect._empty
            for p in params
            if p.name not in {"fileobj", "compresslevel", "stream", "preset"}
        )
    with mock_stubs_missing_types():
        params = get_params(TarFile, "open")
    assert expected == get_param_names(params)[: len(expected)]
    assert all(p.annotation is inspect._empty for p in params)


def test_get_params_staticmethod():
    params = get_params(DateHeader, "value_parser")
    assert [("value", str)] == get_param_types(params)
    with mock_stubs_missing_types():
        params = get_params(DateHeader, "value_parser")
    assert [("value", inspect._empty)] == get_param_types(params)


def test_get_params_function():
    params = get_params(ip_network)
    assert ["address", "strict"] == get_param_names(params)
    if sys.version_info >= (3, 10):
        assert "int | str | bytes | ipaddress.IPv4Address | " in str(params[0].annotation)
    assert bool is params[1].annotation
    with mock_stubs_missing_types():
        params = get_params(ip_network)
    assert [("address", inspect._empty), ("strict", inspect._empty)] == get_param_types(params)


@skip_if_no_pyyaml
def test_get_params_relative_import_from_init():
    import yaml

    params = get_params(yaml.safe_load)
    assert ["stream"] == get_param_names(params)
    assert params[0].annotation is not inspect._empty
    with mock_stubs_missing_types():
        params = get_params(yaml.safe_load)
    assert ["stream"] == get_param_names(params)
    assert params[0].annotation is inspect._empty


def test_get_params_non_unique_alias(logger):
    params = get_params(uuid5)
    name_type = str if sys.version_info < (3, 12) else (str | bytes)
    assert [("namespace", UUID), ("name", name_type)] == get_param_types(params)

    def alias_is_unique(aliases, name, source, value):
        if name == "UUID":
            aliases[name] = ("module", "problem")
        return name != "UUID"

    with patch("jsonargparse._stubs_resolver.alias_is_unique", alias_is_unique):
        with capture_logs(logger) as logs:
            params = get_params(uuid5, logger=logger)
        assert [("namespace", inspect._empty), ("name", name_type)] == get_param_types(params)
        assert "non-unique alias 'UUID': problem (module)" in logs.getvalue()


@skip_if_requests_unavailable
def test_get_params_complex_function_requests_get(parser):
    from requests import get

    with mock_stubs_missing_types():
        params = get_params(get)
    expected = ["url", "params"]
    assert expected == get_param_names(params)
    assert all(p.annotation is inspect._empty for p in params)

    params = get_params(get)
    expected += [
        "data",
        "headers",
        "cookies",
        "files",
        "auth",
        "timeout",
        "allow_redirects",
        "proxies",
        "hooks",
        "stream",
        "verify",
        "cert",
        "json",
    ]
    assert expected == get_param_names(params)
    if sys.version_info >= (3, 10):
        assert all(p.annotation is not inspect._empty for p in params)

    parser.add_function_arguments(get, fail_untyped=False)
    assert ["url", "params"] == list(parser.get_defaults().keys())
    help_str = get_parser_help(parser)
    assert "default: Unknown<stubs-resolver>" in help_str


# stubs only resolver tests


def test_get_params_inspect_signature_failure_function(logger):
    with inspect_signature_failure(ip_network), capture_logs(logger) as logs:
        params = get_params(ip_network, logger=logger)
    assert ["address", "strict"] == get_param_names(params)
    assert params[1].annotation == bool
    if sys.version_info >= (3, 10):
        assert "int | str | bytes | ipaddress.IPv4Address | " in str(params[0].annotation)
    assert "get_parameters_from_ast failed" in logs.getvalue()
    assert "get_parameters_by_assumptions failed" not in logs.getvalue()

    with inspect_signature_failure(ip_network), mock_stubs_missing_resolver(), capture_logs(logger) as logs:
        params = get_params(ip_network, logger=logger)
    assert params == []
    assert "get_parameters_from_ast failed" in logs.getvalue()
    assert "get_parameters_by_assumptions failed" in logs.getvalue()


def test_get_params_inspect_signature_failure_method(logger):
    with inspect_signature_failure(Random.randint), capture_logs(logger) as logs:
        params = get_params(Random, "randint", logger=logger)
    assert [("a", int), ("b", int)] == get_param_types(params)
    assert "get_parameters_from_ast failed" in logs.getvalue()
    assert "get_parameters_by_assumptions failed" not in logs.getvalue()

    with inspect_signature_failure(Random.randint), mock_stubs_missing_resolver(), capture_logs(logger) as logs:
        params = get_params(Random, "randint", logger=logger)
    assert params == []
    assert "get_parameters_from_ast failed" in logs.getvalue()
    assert "get_parameters_by_assumptions failed" in logs.getvalue()


@contextmanager
def get_arg_type_bool_fail():
    original_get_arg_type = get_arg_type

    def mock_get_arg_type(annotation, aliases):
        if isinstance(annotation, ast.Name) and annotation.id == "bool":
            raise RuntimeError("failed to get type")
        return original_get_arg_type(annotation, aliases)

    with patch("jsonargparse._parameter_resolvers.get_arg_type", side_effect=mock_get_arg_type) as mock_signature:
        yield
        mock_signature.assert_called()


def test_get_params_inspect_signature_failure_missing_type(logger):
    with inspect_signature_failure(ip_network), get_arg_type_bool_fail(), capture_logs(logger) as logs:
        params = get_params(ip_network, logger=logger)
    assert ["address", "strict"] == get_param_names(params)
    assert params[1].annotation == inspect._empty
    if sys.version_info >= (3, 10):
        assert "int | str | bytes | ipaddress.IPv4Address | " in str(params[0].annotation)
    assert "get_parameters_from_ast failed" in logs.getvalue()
    assert "get_parameters_by_assumptions failed" not in logs.getvalue()


# pytorch tests


if torch_available:
    import importlib.metadata

    torch_version = tuple(int(v) for v in importlib.metadata.version("torch").split(".", 2)[:2])

    if torch_version < (2, 1) or torch_version >= (2, 4):
        torch_available = False
    else:
        import torch.optim  # pylint: disable=import-error
        import torch.optim.lr_scheduler  # pylint: disable=import-error


@pytest.mark.skipif(not torch_available, reason="only for torch>=2.1,<2.4")
@pytest.mark.parametrize(
    "class_name",
    [
        "Adadelta",
        "Adagrad",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
        "SparseAdam",
    ],
)
def test_get_params_torch_optimizer(class_name):
    cls = getattr(torch.optim, class_name)
    params = get_params(cls)
    assert all(p.annotation is not inspect._empty for p in params)
    with mock_stubs_missing_types():
        params = get_params(cls)
    assert any(p.annotation is inspect._empty for p in params)


@pytest.mark.skipif(not torch_available, reason="only for torch>=2.1,<2.4")
@pytest.mark.parametrize(
    "class_name",
    [
        "_LRScheduler",
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ConstantLR",
        "LinearLR",
        "ExponentialLR",
        "ChainedScheduler",
        "SequentialLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
        "CyclicLR",
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "PolynomialLR",
    ],
)
def test_get_params_torch_lr_scheduler(class_name):
    cls = getattr(torch.optim.lr_scheduler, class_name)
    params = get_params(cls)
    assert all(p.annotation is not inspect._empty for p in params)
    with mock_stubs_missing_types():
        params = get_params(cls)
    assert any(p.annotation is inspect._empty for p in params)
