import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager


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
    import jsonargparse_tests
    with unittest.mock.patch.multiple(jsonargparse_tests, create=True, **{c.__name__: c for c in args}):
        yield __module__


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
