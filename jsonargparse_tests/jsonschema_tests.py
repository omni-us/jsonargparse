#!/usr/bin/env python3
# pylint: disable=unexpected-keyword-arg

import re
import json
from io import StringIO
from contextlib import redirect_stdout
from typing import Optional, Union
from jsonargparse_tests.base import *


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

    def test_bool_type(self):
        parser = ArgumentParser(prog='app', default_env=True, error_handler=None)
        parser.add_argument('--val', type=bool)
        self.assertEqual(None,  parser.get_defaults().val)
        self.assertEqual(True,  parser.parse_args(['--val', 'true']).val)
        self.assertEqual(True,  parser.parse_args(['--val', 'TRUE']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'false']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'FALSE']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', '1']))

        os.environ['APP_VAL'] = 'true'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'True'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'false'
        self.assertEqual(False, parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'False'
        self.assertEqual(False, parser.parse_args([]).val)
        os.environ['APP_VAL'] = '2'
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', 'a']))
        del os.environ['APP_VAL']


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

        self.assertEqual(op2_val, vars(parser.parse_args(['--op2', str(op2_val)]).op2))
        self.assertEqual(17, parser.parse_args(['--op2', '{"k2": 2}']).op2.k3)
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

        cfg = namespace_to_dict(parser.parse_path(cfg1_file))
        self.assertEqual(op1_val, cfg['op1'])
        self.assertEqual(op2_val, cfg['op2'])

        cfg = namespace_to_dict(parser.parse_string(cfg2_str))
        self.assertEqual(op1_val, cfg['op1'])
        self.assertEqual(op2_val, cfg['op2'])

        cfg = parser.parse_args(['--cfg', cfg3_file])
        self.assertEqual(op2_val, namespace_to_dict(cfg.op3.n1[0]))
        parser.check_config(cfg, skip_none=True)


    def test_ActionJsonSchema_failures(self):
        self.assertRaises(ValueError, lambda: ActionJsonSchema())
        self.assertRaises(ValueError, lambda: ActionJsonSchema(schema=':'+json.dumps(schema1)))
        self.assertRaises(ValueError, lambda: ActionJsonSchema(schema=schema1, annotation=int))


    def test_ActionJsonSchema_help(self):
        parser = ArgumentParser()
        parser.add_argument('--op1',
            action=ActionJsonSchema(schema=schema1),
            help='schema: %s')

        os.environ['COLUMNS'] = '150'
        out = StringIO()
        with redirect_stdout(out):
            parser.print_help()

        outval = out.getvalue()
        schema = re.sub('^.*schema:([^()]+)[^{}]*$', r'\1', outval.replace('\n', ' '))
        self.assertEqual(schema1, json.loads(schema))


    def test_add_argument_type_hint(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op1', type=Optional[Union[PositiveInt, OpenUnitInterval]])
        self.assertEqual(0.1, parser.parse_args(['--op1', '0.1']).op1)
        self.assertEqual(0.9, parser.parse_args(['--op1', '0.9']).op1)
        self.assertEqual(1, parser.parse_args(['--op1', '1']).op1)
        self.assertEqual(12, parser.parse_args(['--op1', '12']).op1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '0.0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '4.5']))
        parser.add_argument('--op2', type=Optional[Email])
        self.assertEqual('a@b.c', parser.parse_args(['--op2', 'a@b.c']).op2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', 'abc']))


    def test_no_str_strip(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=Optional[str])
        parser.add_argument('--cfg', action=ActionConfigFile)
        self.assertEqual('  ', parser.parse_args(['--op', '  ']).op)
        self.assertEqual('', parser.parse_args(['--op', '']).op)
        self.assertEqual(' abc ', parser.parse_args(['--op= abc ']).op)
        self.assertEqual(' ', parser.parse_args(['--cfg={"op":" "}']).op)


if __name__ == '__main__':
    unittest.main(verbosity=2)
