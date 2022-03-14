import inspect
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr


try:
    import responses
    responses_activate = responses.activate
except ImportError:
    def nothing_decorator(func):
        return func
    responses = False  # type: ignore
    responses_activate = nothing_decorator


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
