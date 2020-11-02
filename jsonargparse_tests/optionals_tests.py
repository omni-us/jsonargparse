#!/usr/bin/env python3

import unittest
from jsonargparse.optionals import (jsonschema_support, _import_jsonschema,
                                    jsonnet_support, _import_jsonnet,
                                    url_support, _import_url_validator, _import_requests, _url_validator, _requests,
                                    docstring_parser_support, _import_docstring_parse,
                                    argcomplete_support, _import_argcomplete,
                                    set_url_support, get_config_read_mode)


class JsonSchemaSupportTests(unittest.TestCase):

    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
    def test_jsonschema_support_true(self):
        _import_jsonschema('test_jsonschema_support_true')


    @unittest.skipIf(jsonschema_support, 'jsonschema package should not be installed')
    def test_jsonschema_support_false(self):
        with self.assertRaises(ImportError) as context:
            _import_jsonschema('test_jsonschema_support_false')
            self.assertIn('test_jsonschema_support_false', context.msg)


class JsonnetSupportTests(unittest.TestCase):

    @unittest.skipIf(not jsonnet_support, 'jsonnet package is required')
    def test_jsonnet_support_true(self):
        _import_jsonnet('test_jsonnet_support_true')


    @unittest.skipIf(jsonnet_support, 'jsonnet package should not be installed')
    def test_jsonnet_support_false(self):
        with self.assertRaises(ImportError) as context:
            _import_jsonnet('test_jsonnet_support_false')
            self.assertIn('test_jsonnet_support_false', context.msg)


class UrlSupportTests(unittest.TestCase):

    @unittest.skipIf(not url_support, 'validators and requests packages are required')
    def test_url_support_true(self):
        _import_url_validator('test_url_support_true')
        _import_requests('test_url_support_true')


    @unittest.skipIf(url_support, 'validators and requests packages should not be installed')
    def test_url_support_false(self):
        if _url_validator is None:
            with self.assertRaises(ImportError) as context:
                _import_url_validator('test_url_support_false')
                self.assertIn('test_url_support_false', context.msg)
        if _requests is None:
            with self.assertRaises(ImportError) as context:
                _import_requests('test_url_support_false')
                self.assertIn('test_url_support_false', context.msg)


class DocstringParserSupportTests(unittest.TestCase):

    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_docstring_parser_support_true(self):
        _import_docstring_parse('test_docstring_parser_support_true')


    @unittest.skipIf(docstring_parser_support, 'docstring-parser package should not be installed')
    def test_docstring_parser_support_false(self):
        with self.assertRaises(ImportError) as context:
            _import_docstring_parse('test_docstring_parser_support_false')
            self.assertIn('test_docstring_parser_support_false', context.msg)


class ArgcompleteSupportTests(unittest.TestCase):

    @unittest.skipIf(not argcomplete_support, 'argcomplete package is required')
    def test_argcomplete_support_true(self):
        _import_argcomplete('test_argcomplete_support_true')


    @unittest.skipIf(argcomplete_support, 'argcomplete package should not be installed')
    def test_argcomplete_support_false(self):
        with self.assertRaises(ImportError) as context:
            _import_argcomplete('test_argcomplete_support_false')
            self.assertIn('test_argcomplete_support_false', context.msg)


class ConfigReadModeTests(unittest.TestCase):

    @unittest.skipIf(not url_support, 'validators and requests packages are required')
    def test_url_support_true(self):
        self.assertEqual('fr', get_config_read_mode())
        set_url_support(True)
        self.assertEqual('fur', get_config_read_mode())
        set_url_support(False)
        self.assertEqual('fr', get_config_read_mode())


    @unittest.skipIf(url_support, 'validators and requests packages should not be installed')
    def test_url_support_false(self):
        self.assertEqual('fr', get_config_read_mode())
        with self.assertRaises(ImportError):
            set_url_support(True)
        self.assertEqual('fr', get_config_read_mode())
        set_url_support(False)
        self.assertEqual('fr', get_config_read_mode())


if __name__ == '__main__':
    unittest.main(verbosity=2)
