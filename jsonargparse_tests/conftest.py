import logging
import os
import platform
from contextlib import ExitStack, contextmanager, redirect_stderr
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from jsonargparse import ArgumentParser

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
    not find_spec("jsonschema"),
    reason="jsonschema package is required",
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


@contextmanager
def capture_logs(logger: logging.Logger):
    with ExitStack() as stack:
        captured = StringIO()
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stack.enter_context(patch.object(handler, "stream", captured))
        yield captured


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull):
            yield


def get_parser_help(parser: ArgumentParser) -> str:
    out = StringIO()
    parser.print_help(out)
    return out.getvalue()
