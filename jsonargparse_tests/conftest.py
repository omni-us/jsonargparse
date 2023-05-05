import os
import platform
from importlib.util import find_spec
from io import StringIO

import pytest

from jsonargparse import ArgumentParser


@pytest.fixture
def parser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def subparser() -> ArgumentParser:
    return ArgumentParser(exit_on_error=False)


@pytest.fixture
def tmp_cwd(tmpdir):
    with tmpdir.as_cwd():
        yield tmpdir


def get_parser_help(parser: ArgumentParser) -> str:
    out = StringIO()
    parser.print_help(out)
    return out.getvalue()


skip_if_not_cpython = pytest.mark.skipif(
    platform.python_implementation() != "CPython",
    reason="only supported in CPython",
)


skip_if_not_posix = pytest.mark.skipif(
    os.name != "posix",
    reason="only supported in posix systems",
)


skip_if_jsonschema_unavailable = pytest.mark.skipif(
    not find_spec("jsonschema"),
    reason="jsonschema package is required",
)
