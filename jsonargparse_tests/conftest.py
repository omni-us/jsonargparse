import logging
import os
import platform
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from typing import Iterator, List
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser
from jsonargparse.optionals import docstring_parser_support, jsonschema_support

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

skip_if_docstring_parser_unavailable = pytest.mark.skipif(
    not docstring_parser_support,
    reason="docstring-parser package is required",
)

responses_available = find_spec("responses") is not None

if responses_available:
    import responses

    responses_activate = responses.activate
else:

    def nothing_decorator(func):
        return func

    responses_activate = nothing_decorator  # type: ignore


@pytest.fixture
def parser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def subparser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def tmp_cwd(tmpdir):
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
def capture_logs(logger: logging.Logger):
    with ExitStack() as stack:
        captured = StringIO()
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stack.enter_context(patch.object(handler, "stream", captured))
        yield captured


def get_parser_help(parser: ArgumentParser) -> str:
    out = StringIO()
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
