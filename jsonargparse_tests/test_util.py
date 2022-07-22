#!/usr/bin/env python3

import logging
import os
import stat
import pathlib
import unittest
import zipfile
from jsonargparse import ArgumentParser, LoggerProperty, null_logger, Path
from jsonargparse.optionals import fsspec_support, import_fsspec, reconplogger_support, url_support
from jsonargparse.util import get_import_path, import_object, register_unresolvable_import_paths
from jsonargparse_tests.base import is_posix, mock_module, responses_activate, responses_available, suppress_stderr, TempDirTestCase


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
        self.assertEqual(path1.rel_path, path2.rel_path)
        self.assertEqual(path1.is_url, path2.is_url)
        self.assertRaises(TypeError, lambda: Path(True))


    def test_cwd(self):
        path = Path('file_rx', mode='fr', cwd=os.path.join(self.tmpdir, 'dir_x'))
        self.assertEqual(path.cwd, Path('file_rx', mode='fr', cwd=path.cwd).cwd)


    def test_file_access_mode(self):
        Path(self.file_rw, 'frw')
        Path(self.file_r, 'fr')
        Path(self.file_, 'f')
        Path(self.dir_file_rx, 'fr')
        if is_posix:
            self.assertRaises(TypeError, lambda: Path(self.file_rw, 'fx'))
            self.assertRaises(TypeError, lambda: Path(self.file_, 'fr'))
        self.assertRaises(TypeError, lambda: Path(self.file_r, 'fw'))
        self.assertRaises(TypeError, lambda: Path(self.dir_file_rx, 'fw'))
        self.assertRaises(TypeError, lambda: Path(self.dir_rx, 'fr'))
        self.assertRaises(TypeError, lambda: Path('file_ne', 'fr'))


    def test_get_contents(self):
        self.assertEqual('file contents', Path(self.file_r, 'fr').get_content())
        self.assertEqual('file contents', Path(f'file://{self.tmpdir}/{self.file_r}', 'fr').get_content())
        self.assertEqual('file contents', Path(f'file://{self.tmpdir}/{self.file_r}', 'ur').get_content())


    def test_dir_access_mode(self):
        Path(self.dir_rwx, 'drwx')
        Path(self.dir_rx, 'drx')
        Path(self.dir_x, 'dx')
        if is_posix:
            self.assertRaises(TypeError, lambda: Path(self.dir_rx, 'dw'))
            self.assertRaises(TypeError, lambda: Path(self.dir_x, 'dr'))
        self.assertRaises(TypeError, lambda: Path(self.file_r, 'dr'))


    def test_create_mode(self):
        Path(self.file_rw, 'fcrw')
        Path(os.path.join(self.tmpdir, 'file_c'), 'fc')
        Path(self.dir_rwx, 'dcrwx')
        Path(os.path.join(self.tmpdir, 'dir_c'), 'dc')
        if is_posix:
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
        self.assertEqual(str(path), self.file_rw)
        self.assertTrue(path.__repr__().startswith('Path_frw('))


    def test_tilde_home(self):
        home_env = 'USERPROFILE' if os.name == 'nt' else 'HOME'
        with unittest.mock.patch.dict(os.environ, {home_env: self.tmpdir}):
            home = Path('~', 'dr')
            path = Path(os.path.join('~', self.file_rw), 'frw')
            self.assertEqual(str(home), '~')
            self.assertEqual(str(path), os.path.join('~', self.file_rw))
            self.assertEqual(home(), self.tmpdir)
            self.assertEqual(path(), os.path.join(self.tmpdir, self.file_rw))


    @unittest.skipIf(not url_support or not responses_available, 'validators, requests and responses packages are required')
    @responses_activate
    def test_urls_http(self):
        existing = 'http://example.com/existing-url'
        existing_body = 'url contents'
        nonexisting = 'http://example.com/non-existing-url'
        import responses
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


    @unittest.skipIf(not fsspec_support, 'fsspec package is required')
    def test_fsspec(self):

        def create_zip(zip_path, file_path):
            ziph = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            ziph.write(file_path)
            ziph.close()

        existing = 'existing.txt'
        existing_body = 'existing content'
        nonexisting = 'non-existing.txt'
        with open(existing, 'w') as f:
            f.write(existing_body)
        zip1_path = 'file1.zip'
        zip2_path = 'file2.zip'
        create_zip(zip1_path, existing)
        create_zip(zip2_path, existing)
        os.chmod(zip2_path, 0)

        path = Path('zip://'+existing+'::file://'+zip1_path, mode='sr')
        self.assertEqual(existing_body, path.get_content())

        with self.assertRaises(TypeError):
            Path('zip://'+nonexisting+'::file://'+zip1_path, mode='sr')
        if is_posix:
            with self.assertRaises(TypeError):
                Path('zip://'+existing+'::file://'+zip2_path, mode='sr')
        with self.assertRaises(ValueError):
            Path('zip://'+existing+'::file://'+zip1_path, mode='ds')
        with self.assertRaises(TypeError):
            Path('unsupported://'+existing, mode='sr')

        fsspec = import_fsspec('test_fsspec')

        nonexisting = 'nonexisting.txt'
        path = Path('memory://'+nonexisting, mode='sw')
        with fsspec.open(path(), 'w') as f:
            f.write(existing_body)
        self.assertEqual(existing_body, path.get_content())


class LoggingPropertyTests(unittest.TestCase):

    def setUp(self):
        class TestClass(LoggerProperty):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        self.TestClass = TestClass
        self.log_message = 'Testing log message'


    def test_logger_true(self):
        test = self.TestClass(logger=True)
        self.assertEqual(test.logger.handlers[0].level, logging.WARNING)
        self.assertEqual(test.logger.name, 'plain_logger' if reconplogger_support else 'TestClass')


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


    @unittest.mock.patch.dict(os.environ, {'JSONARGPARSE_DEBUG': ''})
    def test_logger_jsonargparse_debug(self):
        parser = ArgumentParser(logger=False)
        with self.assertLogs(logger=parser.logger, level=logging.DEBUG) as log:
            parser.logger.debug(self.log_message)
            self.assertIn(self.log_message, log.output[0])


    def test_logger_levels(self):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for num, level in enumerate(levels):
            with self.subTest(level), suppress_stderr():
                test = self.TestClass(logger={'level': level})
                with self.assertLogs(logger=test.logger, level=level) as log:
                    getattr(test.logger, level.lower())(self.log_message)
                    self.assertEqual(len(log.output), 1)
                    self.assertIn(self.log_message, log.output[0])
                if num > 0:
                    with self.assertRaises(AssertionError), self.assertLogs():
                        getattr(test.logger, levels[num-1].lower())(self.log_message)


    def test_failure_cases(self):
        self.assertRaises(ValueError, lambda: self.TestClass(logger={'level': 'invalid'}))
        self.assertRaises(ValueError, lambda: self.TestClass(logger=self.TestClass))
        self.assertRaises(ValueError, lambda: self.TestClass(logger={'invalid': 'value'}))


class ImportFunctionsTests(unittest.TestCase):

    def test_import_object_invalid(self):
        self.assertRaises(ValueError, lambda: import_object(True))
        self.assertRaises(ValueError, lambda: import_object('jsonargparse-tests.os'))


    def test_get_import_path(self):
        self.assertEqual(get_import_path(ArgumentParser), 'jsonargparse.ArgumentParser')
        self.assertEqual(get_import_path(ArgumentParser.merge_config), 'jsonargparse.ArgumentParser.merge_config')
        from email.mime.base import MIMEBase
        self.assertEqual(get_import_path(MIMEBase), 'email.mime.base.MIMEBase')


    def test_register_unresolvable_import_paths(self):
        def func():
            pass

        with mock_module(func) as module_name:
            func.__module__ = None
            self.assertRaises(ValueError, lambda: get_import_path(func))
            register_unresolvable_import_paths(__import__(module_name))
            self.assertEqual(get_import_path(func), f'{module_name}.func')


if __name__ == '__main__':
    unittest.main(verbosity=2)
