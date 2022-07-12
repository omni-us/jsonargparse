#!/usr/bin/env python3

import json
import os
import re
import unittest
from io import StringIO
from jsonargparse import ActionConfigFile, ActionJsonSchema, ArgumentParser, ParserError
from jsonargparse.optionals import jsonschema_support
from jsonargparse_tests.base import is_posix, TempDirTestCase


schema1 = {
    'type': 'array',
    'items': {'type': 'integer'},
}

schema2 = {
    'type': 'object',
    'properties': {
        'k1': {'type': 'string'},
        'k2': {'type': 'integer'},
        'k3': {
            'type': 'number',
            'default': 17,
        },
    },
    'additionalProperties': False,
}

schema3 = {
    'type': 'object',
    'properties': {
        'n1': {
            'type': 'array',
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'k1': {'type': 'string'},
                    'k2': {'type': 'integer'},
                },
            },
        },
    },
}


@unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
class JsonSchemaTests(TempDirTestCase):

    def test_ActionJsonSchema(self):
        parser = ArgumentParser(prog='app', default_meta=False, error_handler=None)
        parser.add_argument('--op1',
            action=ActionJsonSchema(schema=schema1))
        parser.add_argument('--op2',
            action=ActionJsonSchema(schema=schema2))
        parser.add_argument('--op3',
            action=ActionJsonSchema(schema=schema3))
        parser.add_argument('--cfg',
            action=ActionConfigFile)

        op1_val = [1, 2, 3, 4]
        op2_val = {'k1': 'one', 'k2': 2, 'k3': 3.3}

        self.assertEqual(op1_val, parser.parse_args(['--op1', str(op1_val)]).op1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '[1, "two"]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '[1.5, 2]']))

        self.assertEqual(op2_val, parser.parse_args(['--op2', str(op2_val)]).op2)
        self.assertEqual(17, parser.parse_args(['--op2', '{"k2": 2}']).op2['k3'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k1": 1}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k2": "2"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k4": 4}']))

        op1_file = os.path.join(self.tmpdir, 'op1.json')
        op2_file = os.path.join(self.tmpdir, 'op2.json')
        cfg1_file = os.path.join(self.tmpdir, 'cfg1.yaml')
        cfg3_file = os.path.join(self.tmpdir, 'cfg3.yaml')
        cfg2_str = 'op1:\n  '+str(op1_val)+'\nop2:\n  '+str(op2_val)+'\n'
        with open(op1_file, 'w') as f:
            f.write(str(op1_val))
        with open(op2_file, 'w') as f:
            f.write(str(op2_val))
        with open(cfg1_file, 'w') as f:
            f.write('op1:\n  '+op1_file+'\nop2:\n  '+op2_file+'\n')
        with open(cfg3_file, 'w') as f:
            f.write('op3:\n  n1:\n  - '+str(op2_val)+'\n')

        cfg = parser.parse_path(cfg1_file)
        self.assertEqual(op1_val, cfg['op1'])
        self.assertEqual(op2_val, cfg['op2'])

        cfg = parser.parse_string(cfg2_str)
        self.assertEqual(op1_val, cfg['op1'])
        self.assertEqual(op2_val, cfg['op2'])

        cfg = parser.parse_args(['--cfg', cfg3_file])
        self.assertEqual(op2_val, cfg.op3['n1'][0])
        parser.check_config(cfg, skip_none=True)

        if is_posix:
            os.chmod(op1_file, 0)
            self.assertRaises(ParserError, lambda: parser.parse_path(cfg1_file))


    def test_ActionJsonSchema_failures(self):
        self.assertRaises(ValueError, lambda: ActionJsonSchema())
        self.assertRaises(ValueError, lambda: ActionJsonSchema(schema=':'+json.dumps(schema1)))


    def test_ActionJsonSchema_help(self):
        parser = ArgumentParser()
        parser.add_argument('--op1',
            action=ActionJsonSchema(schema=schema1),
            help='schema: %s')

        out = StringIO()
        parser.print_help(out)

        outval = out.getvalue()
        schema = re.sub('^.*schema:([^()]+)[^{}]*$', r'\1', outval.replace('\n', ' '))
        self.assertEqual(schema1, json.loads(schema))


if __name__ == '__main__':
    unittest.main(verbosity=2)
