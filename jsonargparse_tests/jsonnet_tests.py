#!/usr/bin/env python3

import os
import unittest
from jsonargparse import *
from jsonargparse.optionals import jsonnet_support
from jsonargparse_tests.util_tests import TempDirTestCase


example_jsonnet_1 = '''
local make_record(num) = {
    'ref': '#'+(num+1),
    'val': 3*(num/2)+5,
};

{
  'param': 654,
  'records': [make_record(n) for n in std.range(0, 8)],
}
'''

example_jsonnet_2 = '''
local param = std.extVar('param');

local make_record(num) = {
    'ref': '#'+(num+1),
    'val': 3*(num/2)+5,
};

{
  'param': param,
  'records': [make_record(n) for n in std.range(0, 8)],
}
'''


@unittest.skipIf(not jsonnet_support, 'jsonnet and jsonschema packages are required')
class JsonnetTests(TempDirTestCase):

    def test_parser_mode_jsonnet(self):
        schema = {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'ref': {'type': 'string'},
                    'val': {'type': 'number'},
                },
            },
        }

        parser = ArgumentParser(parser_mode='jsonnet')
        parser.add_argument('--cfg',
            action=ActionConfigFile)
        parser.add_argument('--param',
            type=int)
        parser.add_argument('--records',
            action=ActionJsonSchema(schema=schema))

        jsonnet_file = os.path.join(self.tmpdir, 'example.jsonnet')
        with open(jsonnet_file, 'w') as output_file:
            output_file.write(example_jsonnet_1)

        cfg = parser.parse_args(['--cfg', jsonnet_file])
        self.assertEqual(654, cfg.param)
        self.assertEqual(9, len(cfg.records))
        self.assertEqual('#8', cfg.records[-2].ref)
        self.assertEqual(15.5, cfg.records[-2].val)


    def test_ActionJsonnet(self):
        parser = ArgumentParser()
        parser.add_argument('--input.ext_vars',
            action=ActionJsonnetExtVars())
        parser.add_argument('--input.jsonnet',
            action=ActionJsonnet(ext_vars='input.ext_vars'))

        cfg = parser.parse_args(['--input.ext_vars', '{"param": 123}', '--input.jsonnet', example_jsonnet_2])
        self.assertEqual(123, cfg.input.jsonnet.param)
        self.assertEqual(9, len(cfg.input.jsonnet.records))
        self.assertEqual('#8', cfg.input.jsonnet.records[-2].ref)
        self.assertEqual(15.5, cfg.input.jsonnet.records[-2].val)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--input.jsonnet', example_jsonnet_2]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
