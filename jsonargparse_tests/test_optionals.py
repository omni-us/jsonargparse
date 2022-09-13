#!/usr/bin/env python3

import unittest
from importlib.util import find_spec
from jsonargparse import get_config_read_mode, set_config_read_mode
from jsonargparse.optionals import (
    argcomplete_support,
    docstring_parser_support,
    fsspec_support,
    get_docstring_parse_options,
    import_argcomplete,
    import_docstring_parser,
    import_fsspec,
    import_jsonnet,
    import_jsonschema,
    import_requests,
    import_ruyaml,
    import_url_validator,
    jsonnet_support,
    jsonschema_support,
    ruyaml_support,
    set_docstring_parse_options,
    url_support,
)


class JsonSchemaSupportTests(unittest.TestCase):

    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
    def test_jsonschema_support_true(self):
        import_jsonschema('test_jsonschema_support_true')


    @unittest.skipIf(jsonschema_support, 'jsonschema package should not be installed')
    def test_jsonschema_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_jsonschema('test_jsonschema_support_false')
            self.assertIn('test_jsonschema_support_false', context.msg)


class JsonnetSupportTests(unittest.TestCase):

    @unittest.skipIf(not jsonnet_support, 'jsonnet package is required')
    def test_jsonnet_support_true(self):
        import_jsonnet('test_jsonnet_support_true')


    @unittest.skipIf(jsonnet_support, 'jsonnet package should not be installed')
    def test_jsonnet_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_jsonnet('test_jsonnet_support_false')
            self.assertIn('test_jsonnet_support_false', context.msg)


class UrlSupportTests(unittest.TestCase):

    @unittest.skipIf(not url_support, 'validators and requests packages are required')
    def test_url_support_true(self):
        import_url_validator('test_url_support_true')
        import_requests('test_url_support_true')


    @unittest.skipIf(url_support, 'validators and requests packages should not be installed')
    def test_url_support_false(self):
        if find_spec('validators') is None:
            with self.assertRaises(ImportError) as context:
                import_url_validator('test_url_support_false')
                self.assertIn('test_url_support_false', context.msg)
        if find_spec('requests') is None:
            with self.assertRaises(ImportError) as context:
                import_requests('test_url_support_false')
                self.assertIn('test_url_support_false', context.msg)


class DocstringParserSupportTests(unittest.TestCase):

    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_docstring_parser_support_true(self):
        import_docstring_parser('test_docstring_parser_support_true')


    @unittest.skipIf(docstring_parser_support, 'docstring-parser package should not be installed')
    def test_docstring_parser_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_docstring_parser('test_docstring_parser_support_false')
            self.assertIn('test_docstring_parser_support_false', context.msg)


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_docstring_parse_options(self):
        from docstring_parser import DocstringStyle
        options = get_docstring_parse_options()
        options['style'] = None
        options = get_docstring_parse_options()

        with self.subTest('style'):
            for style in [DocstringStyle.NUMPYDOC, DocstringStyle.GOOGLE]:
                set_docstring_parse_options(style=style)
                self.assertEqual(options['style'], style)
            with self.assertRaises(ValueError):
                set_docstring_parse_options(style='invalid')

        with self.subTest('attribute_docstrings'):
            self.assertIs(options['attribute_docstrings'], False)
            for attribute_docstrings in [True, False]:
                set_docstring_parse_options(attribute_docstrings=attribute_docstrings)
                self.assertIs(options['attribute_docstrings'], attribute_docstrings)
            with self.assertRaises(ValueError):
                set_docstring_parse_options(attribute_docstrings='invalid')


class ArgcompleteSupportTests(unittest.TestCase):

    @unittest.skipIf(not argcomplete_support, 'argcomplete package is required')
    def test_argcomplete_support_true(self):
        import_argcomplete('test_argcomplete_support_true')


    @unittest.skipIf(argcomplete_support, 'argcomplete package should not be installed')
    def test_argcomplete_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_argcomplete('test_argcomplete_support_false')
            self.assertIn('test_argcomplete_support_false', context.msg)


class FsspecSupportTests(unittest.TestCase):

    @unittest.skipIf(not fsspec_support, 'fsspec package is required')
    def test_fsspec_support_true(self):
        import_fsspec('test_fsspec_support_true')


    @unittest.skipIf(fsspec_support, 'fsspec package should not be installed')
    def test_fsspec_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_fsspec('test_fsspec_support_false')
            self.assertIn('test_fsspec_support_false', context.msg)


class RuyamlSupportTests(unittest.TestCase):

    @unittest.skipIf(not ruyaml_support, 'ruyaml package is required')
    def test_ruyaml_support_true(self):
        import_ruyaml('test_ruyaml_support_true')


    @unittest.skipIf(ruyaml_support, 'ruyaml package should not be installed')
    def test_ruyaml_support_false(self):
        with self.assertRaises(ImportError) as context:
            import_ruyaml('test_ruyaml_support_false')
            self.assertIn('test_ruyaml_support_false', context.msg)


class ConfigReadModeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_config_read_mode()


    @unittest.skipIf(not url_support, 'validators and requests packages are required')
    def test_url_support_true(self):
        self.assertEqual('fr', get_config_read_mode())
        set_config_read_mode(urls_enabled=True)
        self.assertEqual('fur', get_config_read_mode())
        set_config_read_mode(urls_enabled=False)
        self.assertEqual('fr', get_config_read_mode())


    @unittest.skipIf(url_support, 'validators and requests packages should not be installed')
    def test_url_support_false(self):
        self.assertEqual('fr', get_config_read_mode())
        with self.assertRaises(ImportError):
            set_config_read_mode(urls_enabled=True)
        self.assertEqual('fr', get_config_read_mode())
        set_config_read_mode(urls_enabled=False)
        self.assertEqual('fr', get_config_read_mode())


    @unittest.skipIf(not fsspec_support, 'fsspec package is required')
    def test_fsspec_support_true(self):
        self.assertEqual('fr', get_config_read_mode())
        set_config_read_mode(fsspec_enabled=True)
        self.assertEqual('fsr', get_config_read_mode())
        set_config_read_mode(fsspec_enabled=False)
        self.assertEqual('fr', get_config_read_mode())


    @unittest.skipIf(fsspec_support, 'fsspec package should not be installed')
    def test_fsspec_support_false(self):
        self.assertEqual('fr', get_config_read_mode())
        with self.assertRaises(ImportError):
            set_config_read_mode(fsspec_enabled=True)
        self.assertEqual('fr', get_config_read_mode())
        set_config_read_mode(fsspec_enabled=False)
        self.assertEqual('fr', get_config_read_mode())


if __name__ == '__main__':
    unittest.main(verbosity=2)
