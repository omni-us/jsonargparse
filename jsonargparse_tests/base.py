import inspect
import os
import platform
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from importlib.util import find_spec
from jsonargparse.optionals import docstring_parser_support, set_docstring_parse_options


responses_available = find_spec('responses') is not None

if responses_available:
    import responses
    responses_activate = responses.activate
else:
    def nothing_decorator(func):
        return func
    responses_activate = nothing_decorator  # type: ignore


if docstring_parser_support:
    from docstring_parser import DocstringStyle
    set_docstring_parse_options(style=DocstringStyle.GOOGLE)


is_posix = os.name == 'posix'
is_cpython = platform.python_implementation() == 'CPython'
os.environ['COLUMNS'] = '150'


@contextmanager
def mock_module(*args):
    __module__ = 'jsonargparse_tests'
    for component in args:
        component.__module__ = __module__
        component.__qualname__ = component.__name__
        if inspect.isclass(component):
            methods = [k for k, v in inspect.getmembers(component) if callable(v) and k[0] != '_']
            for method in [getattr(component, m) for m in methods]:
                method.__module__ = __module__
                method.__qualname__ = component.__name__+'.'+method.__name__
    import jsonargparse_tests
    with unittest.mock.patch.multiple(jsonargparse_tests, create=True, **{c.__name__: c for c in args}):
        yield __module__


@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            yield None


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
        os.chdir(self.tmpdir)


    def tearDown(self):
        os.chdir(self.cwd)
        try:
            shutil.rmtree(self.tmpdir)
        except PermissionError:
            pass
