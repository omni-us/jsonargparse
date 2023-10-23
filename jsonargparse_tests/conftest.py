import logging
import os
import platform
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from functools import wraps
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from typing import Iterator, List
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser
from jsonargparse._optionals import (
    docstring_parser_support,
    fsspec_support,
    jsonschema_support,
    set_docstring_parse_options,
    url_support,
)

if docstring_parser_support:
    from docstring_parser import DocstringStyle

    set_docstring_parse_options(style=DocstringStyle.GOOGLE)


is_cpython = platform.python_implementation() == "CPython"
is_posix = os.name == "posix"

skip_if_not_cpython = pytest.mark.skipif(
    not is_cpython,
    reason="only supported in CPython",
)

skip_if_not_posix = pytest.mark.skipif(
    not is_posix,
    reason="only supported in posix systems",
)


skip_if_jsonschema_unavailable = pytest.mark.skipif(
    not jsonschema_support,
    reason="jsonschema package is required",
)

skip_if_fsspec_unavailable = pytest.mark.skipif(
    not fsspec_support,
    reason="fsspec package is required",
)

skip_if_docstring_parser_unavailable = pytest.mark.skipif(
    not docstring_parser_support,
    reason="docstring-parser package is required",
)

skip_if_requests_unavailable = pytest.mark.skipif(
    not url_support,
    reason="requests package is required",
)

responses_available = bool(find_spec("responses"))

skip_if_responses_unavailable = pytest.mark.skipif(
    not responses_available,
    reason="responses package is required",
)

if responses_available:
    import responses

    responses_activate = responses.activate
else:

    def nothing_decorator(func):
        return func

    responses_activate = nothing_decorator


@pytest.fixture
def parser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def subparser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def example_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="app", exit_on_error=False)
    group_1 = parser.add_argument_group("Group 1", name="group1")
    group_1.add_argument("--bool", type=bool, default=True)
    group_2 = parser.add_argument_group("Group 2")
    group_2.add_argument("--nums.val1", type=int, default=1)
    group_2.add_argument("--nums.val2", type=float, default=2.0)
    return parser


@pytest.fixture
def tmp_cwd(tmpdir) -> Iterator[Path]:
    with tmpdir.as_cwd():
        yield Path(tmpdir)


@pytest.fixture
def file_r(tmp_cwd) -> Iterator[str]:
    filename = "file_r"
    Path(filename).touch()
    yield filename


@pytest.fixture
def logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.level = logging.DEBUG
    logger.parent = None
    logger.handlers = [logging.StreamHandler()]
    return logger


@contextmanager
def capture_logs(logger: logging.Logger) -> Iterator[StringIO]:
    with ExitStack() as stack:
        captured = StringIO()
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stack.enter_context(patch.object(handler, "stream", captured))
        yield captured


@contextmanager
def source_unavailable():
    with patch("inspect.getsource", side_effect=OSError("could not get source code")):
        yield


def get_parser_help(parser: ArgumentParser) -> str:
    out = StringIO()
    with patch.dict(os.environ, {"COLUMNS": "200"}):
        parser.print_help(out)
    return out.getvalue()


def get_parse_args_stdout(parser: ArgumentParser, args: List[str]) -> str:
    out = StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit):
        parser.parse_args(args)
    return out.getvalue()


def get_parse_args_stderr(parser: ArgumentParser, args: List[str]) -> str:
    err = StringIO()
    with patch.object(parser, "exit_on_error", return_value=True):
        with redirect_stderr(err), pytest.raises(SystemExit):
            parser.parse_args(args)
    return err.getvalue()


class BaseClass:
    def __init__(self):
        pass


def wrap_fn(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped_fn
