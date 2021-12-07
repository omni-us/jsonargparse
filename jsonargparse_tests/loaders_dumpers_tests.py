#!/usr/bin/env python3

import unittest
import yaml
from importlib.util import find_spec
from typing import List
from jsonargparse import ActionConfigFile, ArgumentParser, set_dumper, set_loader, ParserError
from jsonargparse.loaders_dumpers import yaml_dump
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


    @unittest.skipIf(not dump_preserve_order_support or not find_spec('omegaconf'), 'omegaconf package and CPython required')
    def test_set_loader_omegaconf(self):
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
