#!/usr/bin/env python3

import unittest
import yaml
from importlib.util import find_spec
from typing import List
from jsonargparse import ActionConfigFile, ArgumentParser, set_dumper, set_loader, ParserError
from jsonargparse.loaders_dumpers import load_value, load_value_context, yaml_dump
from jsonargparse.optionals import dump_preserve_order_support


class LoadersTests(unittest.TestCase):

    def test_set_dumper_custom_yaml(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list', type=List[int])

        def custom_yaml_dump(data) -> str:
            return yaml.safe_dump(data, default_flow_style=True)

        with unittest.mock.patch.dict('jsonargparse.loaders_dumpers.dumpers'):
            set_dumper('yaml_custom', custom_yaml_dump)
            cfg = parser.parse_args(['--list=[1,2,3]'])
            dump = parser.dump(cfg, format='yaml_custom')
            self.assertEqual(dump, '{list: [1, 2, 3]}\n')


    def test_set_loader_safe_load_invalid_scientific_notation(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--num', type=float)

        with unittest.mock.patch.dict('jsonargparse.loaders_dumpers.loaders'):
            set_loader('yaml', yaml.safe_load)
            self.assertRaises(ParserError, lambda: parser.parse_args(['--num=1e-3']))

        self.assertEqual(1e-3, parser.parse_args(['--num=1e-3']).num)


    def test_disable_implicit_mapping_values(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', type=str)
        self.assertEqual('{one}', parser.parse_args(['--val={one}']).val)
        self.assertEqual('{one,two,three}', parser.parse_args(['--val={one,two,three}']).val)


    @unittest.skipIf(not dump_preserve_order_support or not find_spec('omegaconf'), 'omegaconf package and CPython required')
    def test_parser_mode_omegaconf(self):
        parser = ArgumentParser(error_handler=None, parser_mode='omegaconf')
        parser.add_argument('--server.host', type=str)
        parser.add_argument('--server.port', type=int)
        parser.add_argument('--client.url', type=str)
        parser.add_argument('--config', action=ActionConfigFile)

        config = {
            'server': {
                'host': 'localhost',
                'port': 80,
            },
            'client': {
                'url': 'http://${server.host}:${server.port}/',
            }
        }

        cfg = parser.parse_args([f'--config={yaml_dump(config)}'])
        self.assertEqual(cfg.client.url, 'http://localhost:80/')
        self.assertIn('url: http://localhost:80/', parser.dump(cfg))


    @unittest.skipIf(not dump_preserve_order_support or not find_spec('omegaconf'), 'omegaconf package and CPython required')
    def test_parser_mode_omegaconf_in_subcommands(self):
        subparser = ArgumentParser()
        subparser.add_argument('--config', action=ActionConfigFile)
        subparser.add_argument('--source', type=str)
        subparser.add_argument('--target', type=str)

        parser = ArgumentParser(error_handler=None, parser_mode='omegaconf')
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('sub', subparser)

        config = {
            'source': 'hello',
            'target': '${source}',
        }
        cfg = parser.parse_args(['sub', f'--config={yaml_dump(config)}'])
        self.assertEqual(cfg.sub.target, 'hello')


    def test_parser_mode_subparsers(self):
        subparser = ArgumentParser()
        parser = ArgumentParser()
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('sub', subparser)

        with unittest.mock.patch.dict('jsonargparse.loaders_dumpers.loaders'):
            set_loader('custom', yaml.safe_load)
            parser.parser_mode = 'custom'
            self.assertEqual('custom', parser.parser_mode)
            self.assertEqual('custom', subparser.parser_mode)


    def test_dump_header_yaml(self):
        parser = ArgumentParser()
        parser.add_argument('--int', type=int, default=1)
        parser.dump_header = ['line 1', 'line 2']
        dump = parser.dump(parser.get_defaults())
        self.assertEqual(dump, '# line 1\n# line 2\nint: 1\n')


    def test_dump_header_json(self):
        parser = ArgumentParser()
        parser.add_argument('--int', type=int, default=1)
        parser.dump_header = ['line 1', 'line 2']
        dump = parser.dump(parser.get_defaults(), format='json')
        self.assertEqual(dump, '{"int":1}')


    def test_dump_header_invalid(self):
        parser = ArgumentParser()
        with self.assertRaises(ValueError):
            parser.dump_header = True


    def test_load_value_dash(self):
        with load_value_context('yaml'):
            self.assertEqual('-', load_value('-'))
            self.assertEqual(' -  ', load_value(' -  '))


if __name__ == '__main__':
    unittest.main(verbosity=2)
