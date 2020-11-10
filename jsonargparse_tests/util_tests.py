#!/usr/bin/env python3

import os
import stat
import shutil
import pathlib
import logging
import tempfile
import platform
import unittest
from jsonargparse.util import *
from jsonargparse.util import _check_unknown_kwargs, _suppress_stderr

try:
    import responses
    responses_activate = responses.activate
except:
    def nothing_decorator(func):
        return func
    responses = False
    responses_activate = nothing_decorator


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
        os.chdir(self.tmpdir)


    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)


@unittest.skipIf(os.name != 'posix' or platform.python_implementation() != 'CPython',
                 'Path class currently only supported in posix systems and CPython')
class PathTests(TempDirTestCase):

    def setUp(self):
        super().setUp()

        self.file_rw = file_rw = 'file_rw'
        self.file_r = file_r = 'file_r'
        self.file_ = file_ = 'file_'
        self.dir_rwx = dir_rwx = 'dir_rwx'
        self.dir_rx = dir_rx = 'dir_rx'
        self.dir_x = dir_x = 'dir_x'
        self.dir_file_rx = dir_file_rx = os.path.join(dir_x, 'file_rx')

        with open(file_r, 'w') as f:
            f.write('file contents')

        pathlib.Path(file_rw).touch()
        pathlib.Path(file_).touch()
        os.mkdir(dir_rwx)
        os.mkdir(dir_rx)
        os.mkdir(dir_x)
        pathlib.Path(dir_file_rx).touch()

        os.chmod(file_rw, (stat.S_IREAD | stat.S_IWRITE))
        os.chmod(file_r, stat.S_IREAD)
        os.chmod(file_, 0)
        os.chmod(dir_file_rx, (stat.S_IREAD | stat.S_IEXEC))
        os.chmod(dir_rwx, (stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC))
        os.chmod(dir_rx, (stat.S_IREAD | stat.S_IEXEC))
        os.chmod(dir_x, stat.S_IEXEC)


    def tearDown(self):
        os.chmod(self.dir_x, (stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC))
        super().tearDown()


    def test_init(self):
        path1 = Path(self.file_rw, 'frw')
        path2 = Path(path1)
        self.assertEqual(path1.cwd, path2.cwd)
        self.assertEqual(path1.abs_path, path2.abs_path)
        self.assertEqual(path1.path, path2.path)
        self.assertEqual(path1.is_url, path2.is_url)
        self.assertRaises(TypeError, lambda: Path(True))


    def test_cwd(self):
        path = Path('file_rx', mode='fr', cwd=os.path.join(self.tmpdir, 'dir_x'))
        self.assertEqual(path.cwd, Path('file_rx', mode='fr', cwd=[path.cwd]).cwd)


    def test_file_access_mode(self):
        Path(self.file_rw, 'frw')
        Path(self.file_r, 'fr')
        Path(self.file_, 'f')
        Path(self.dir_file_rx, 'fr')
        self.assertRaises(TypeError, lambda: Path(self.file_rw, 'fx'))
        self.assertRaises(TypeError, lambda: Path(self.file_r, 'fw'))
        self.assertRaises(TypeError, lambda: Path(self.file_, 'fr'))
        self.assertRaises(TypeError, lambda: Path(self.dir_file_rx, 'fw'))
        self.assertRaises(TypeError, lambda: Path(self.dir_rx, 'fr'))
        self.assertRaises(TypeError, lambda: Path('file_ne', 'fr'))


    def test_get_contents(self):
        self.assertEqual('file contents', Path(self.file_r, 'fr').get_content())
        self.assertEqual('file contents', Path('file://'+self.tmpdir+'/'+self.file_r, 'fr').get_content())


    def test_dir_access_mode(self):
        Path(self.dir_rwx, 'drwx')
        Path(self.dir_rx, 'drx')
        Path(self.dir_x, 'dx')
        self.assertRaises(TypeError, lambda: Path(self.dir_rx, 'dw'))
        self.assertRaises(TypeError, lambda: Path(self.dir_x, 'dr'))
        self.assertRaises(TypeError, lambda: Path(self.file_r, 'dr'))


    def test_create_mode(self):
        Path(self.file_rw, 'fcrw')
        Path(os.path.join(self.tmpdir, 'file_c'), 'fc')
        Path(self.dir_rwx, 'dcrwx')
        Path(os.path.join(self.tmpdir, 'dir_c'), 'dc')
        self.assertRaises(TypeError, lambda: Path(os.path.join(self.dir_rx, 'file_c'), 'fc'))
        self.assertRaises(TypeError, lambda: Path(os.path.join(self.dir_rx, 'dir_c'), 'dc'))
        self.assertRaises(TypeError, lambda: Path(self.file_rw, 'dc'))
        self.assertRaises(TypeError, lambda: Path(self.dir_rwx, 'fc'))
        self.assertRaises(TypeError, lambda: Path(os.path.join(self.dir_rwx, 'ne', 'file_c'), 'fc'))


    def test_complement_modes(self):
        self.assertRaises(TypeError, lambda: Path(self.file_rw, 'fW'))
        self.assertRaises(TypeError, lambda: Path(self.file_rw, 'fR'))
        self.assertRaises(TypeError, lambda: Path(self.dir_rwx, 'dX'))
        self.assertRaises(TypeError, lambda: Path(self.file_rw, 'F'))
        self.assertRaises(TypeError, lambda: Path(self.dir_rwx, 'D'))


    def test_invalid_modes(self):
        self.assertRaises(ValueError, lambda: Path(self.file_rw, True))
        self.assertRaises(ValueError, lambda: Path(self.file_rw, '≠'))
        self.assertRaises(ValueError, lambda: Path(self.file_rw, 'fd'))
        if url_support:
            self.assertRaises(ValueError, lambda: Path(self.file_rw, 'du'))


    def test_class_hidden_methods(self):
        path = Path(self.file_rw, 'frw')
        self.assertEqual(path(False), self.file_rw)
        self.assertEqual(path(True), os.path.join(self.tmpdir, self.file_rw))
        self.assertEqual(path(), os.path.join(self.tmpdir, self.file_rw))
        self.assertEqual(str(path), os.path.join(self.tmpdir, self.file_rw))
        self.assertTrue(path.__repr__().startswith('Path('))


    @unittest.skipIf(not url_support or not responses, 'validators, requests and responses packages are required')
    @responses_activate
    def test_urls(self):
        existing = 'http://example.com/existing-url'
        existing_body = 'url contents'
        nonexisting = 'http://example.com/non-existing-url'
        responses.add(responses.GET,
                      existing,
                      body=existing_body,
                      status=200)
        responses.add(responses.HEAD,
                      existing,
                      status=200)
        responses.add(responses.HEAD,
                      nonexisting,
                      status=404)
        path = Path(existing, mode='ur')
        self.assertEqual(existing_body, path.get_content())
        self.assertRaises(TypeError, lambda: Path(nonexisting, mode='ur'))


class LoggingPropertyTests(unittest.TestCase):

    def setUp(self):
        class TestClass(LoggerProperty):
            def __init__(self, logger=None):
                self.logger = logger

        self.TestClass = TestClass
        self.log_message = 'Testing log message'


    def test_logger_true(self):
        test = self.TestClass(logger=True)
        self.assertEqual(test.logger.level, logging.WARNING)
        self.assertEqual(test.logger.name, 'TestClass')


    def test_logger_false(self):
        test = self.TestClass(logger=False)
        self.assertEqual(test.logger, null_logger)
        with self.assertRaises(AssertionError), self.assertLogs():
            test.logger.error(self.log_message)


    def test_no_init_logger(self):
        class NoLogger(LoggerProperty):
            pass

        test = NoLogger()
        self.assertEqual(test.logger, null_logger)


    def test_logger_str(self):
        logger = logging.getLogger('test_logger_str')
        test = self.TestClass(logger='test_logger_str')
        self.assertEqual(test.logger, logger)


    def test_logger_object(self):
        logger = logging.getLogger('test_logger_object')
        test = self.TestClass(logger=logger)
        self.assertEqual(test.logger, logger)
        self.assertEqual(test.logger.name, 'test_logger_object')


    def test_logger_name(self):
        test = self.TestClass(logger={'name': 'test_logger_name'})
        self.assertEqual(test.logger.name, 'test_logger_name')


    def test_logger_levels(self):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for num, level in enumerate(levels):
            with self.subTest(level), _suppress_stderr():
                test = self.TestClass(logger={'level': level})
                with self.assertLogs(level=level) as log:
                    getattr(test.logger, level.lower())(self.log_message)
                    self.assertEqual(len(log.output), 1)
                    self.assertIn(self.log_message, log.output[0])
                if num > 0:
                    with self.assertRaises(AssertionError), self.assertLogs():
                        getattr(test.logger, levels[num-1].lower())(self.log_message)


    def test_failure_cases(self):
        self.assertRaises(ValueError, lambda: self.TestClass(logger={'level': 'invalid'}))
        self.assertRaises(ValueError, lambda: self.TestClass(logger=self.TestClass))


class OtherTests(unittest.TestCase):

    def test_check_unknown_kwargs(self):
        _check_unknown_kwargs(kwargs={'a': 0, 'b': 1}, keys={'a', 'b'})
        self.assertRaises(ValueError, lambda: _check_unknown_kwargs(kwargs={'b': 1}, keys={'a'}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
