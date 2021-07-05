#!/usr/bin/env python3

import sys
import json
import yaml
import platform
from io import StringIO
from contextlib import redirect_stdout
from collections import OrderedDict
from random import randint, shuffle
from jsonargparse_tests.base import *
from jsonargparse.optionals import dump_preserve_order_support
from jsonargparse.util import meta_keys, _suppress_stderr


class ParsersTests(TempDirTestCase):

    def test_parse_args(self):
        parser = example_parser()
        self.assertEqual('opt1_arg', parser.parse_args(['--lev1.lev2.opt1', 'opt1_arg']).lev1.lev2.opt1)
        self.assertEqual(9, parser.parse_args(['--nums.val1', '9']).nums.val1)
        self.assertEqual(6.4, parser.parse_args(['--nums.val2', '6.4']).nums.val2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val1', '7.5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val2', 'eight']))
        self.assertEqual(9, vars(parser.parse_args(['--nums.val1', '9'], nested=False))['nums.val1'])


    def test_parse_object(self):
        parser = example_parser()

        cfg = parser.parse_object(yaml.safe_load(example_yaml))
        self.assertEqual('opt1_yaml', cfg.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg.lev1.lev2.opt2)
        self.assertEqual(-1,  cfg.nums.val1)
        self.assertEqual(2.0, cfg.nums.val2)
        self.assertEqual(False, cfg.bools.def_false)
        self.assertEqual(True,  cfg.bools.def_true)

        self.assertRaises(ParserError, lambda: parser.parse_object({'undefined': True}))


    def test_parse_env(self):
        parser = example_parser()
        cfg = parser.parse_env(example_env)
        self.assertEqual('opt1_env', cfg.lev1.lev2.opt1)
        self.assertEqual(0, cfg.nums.val1)
        cfg = parser.parse_env(example_env, defaults=False)
        self.assertFalse(hasattr(cfg, 'bools'))
        self.assertTrue(hasattr(cfg, 'nums'))
        parser.add_argument('--cfg', action=ActionConfigFile)
        env = OrderedDict(example_env)
        env['APP_NUMS__VAL1'] = '"""'
        self.assertRaises(ParserError, lambda: parser.parse_env(env))
        env = OrderedDict(example_env)
        env['APP_CFG'] = '{"nums": {"val1": 1}}'
        self.assertEqual(0, parser.parse_env(env).nums.val1)
        parser.add_argument('req', nargs='+')
        env['APP_REQ'] = 'abc'
        self.assertEqual(['abc'], parser.parse_env(env).req)
        env['APP_REQ'] = '["abc", "xyz"]'
        self.assertEqual(['abc', 'xyz'], parser.parse_env(env).req)
        env['APP_REQ'] = '[""","""]'
        self.assertEqual(['[""","""]'], parser.parse_env(env).req)
        with self.assertRaises(ValueError):
            parser.default_env = 'invalid'
        with self.assertRaises(ValueError):
            parser.env_prefix = lambda: 'invalid'


    def test_default_env(self):
        parser = ArgumentParser()
        self.assertFalse(parser.default_env)
        parser.default_env = True
        self.assertTrue(parser.default_env)
        parser = ArgumentParser(default_env=True)
        self.assertTrue(parser.default_env)
        parser.default_env = False
        self.assertFalse(parser.default_env)


    def test_parse_string(self):
        parser = example_parser()

        cfg1 = parser.parse_string(example_yaml)
        self.assertEqual('opt1_yaml', cfg1.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg1.lev1.lev2.opt2)
        self.assertEqual(-1,  cfg1.nums.val1)
        self.assertEqual(2.0, cfg1.nums.val2)
        self.assertEqual(False, cfg1.bools.def_false)
        self.assertEqual(True,  cfg1.bools.def_true)

        cfg2 = parser.parse_string(example_yaml, defaults=False)
        self.assertFalse(hasattr(cfg2, 'bools'))
        self.assertTrue(hasattr(cfg2, 'nums'))

        self.assertRaises(ParserError, lambda: parser.parse_string('"""'))


    def test_parse_path(self):
        parser = example_parser()
        cfg1 = parser.parse_string(example_yaml)
        cfg2 = parser.parse_string(example_yaml, defaults=False)

        yaml_file = os.path.realpath(os.path.join(self.tmpdir, 'example.yaml'))

        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml)
        self.assertEqual(cfg1, parser.parse_path(yaml_file, defaults=True))
        self.assertEqual(cfg2, parser.parse_path(yaml_file, defaults=False))
        self.assertNotEqual(cfg2, parser.parse_path(yaml_file, defaults=True))
        self.assertNotEqual(cfg1, parser.parse_path(yaml_file, defaults=False))

        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml+'  val2: eight\n')
        self.assertRaises(ParserError, lambda: parser.parse_path(yaml_file))
        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml+'  val3: key_not_defined\n')
        self.assertRaises(ParserError, lambda: parser.parse_path(yaml_file))


    def test_cfg_base(self):
        parser = ArgumentParser()
        parser.add_argument('--op1')
        parser.add_argument('--op2')
        cfg = parser.parse_args(['--op1=abc'], Namespace(op2='xyz'))
        self.assertEqual('abc', cfg.op1)
        self.assertEqual('xyz', cfg.op2)


    def test_precedence_of_sources(self):
        input1_config_file = os.path.realpath(os.path.join(self.tmpdir, 'input1.yaml'))
        input2_config_file = os.path.realpath(os.path.join(self.tmpdir, 'input2.yaml'))
        default_config_file = os.path.realpath(os.path.join(self.tmpdir, 'default.yaml'))

        parser = ArgumentParser(prog='app',
                                default_env=True,
                                default_config_files=[default_config_file])
        parser.add_argument('--op1', default='from parser default')
        parser.add_argument('--op2')
        parser.add_argument('--cfg', action=ActionConfigFile)

        with open(input1_config_file, 'w') as output_file:
            output_file.write('op1: from input config file')
        with open(input2_config_file, 'w') as output_file:
            output_file.write('op2: unused')

        ## check parse_env precedence ##
        self.assertEqual('from parser default', parser.parse_env().op1)
        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from default config file')
        self.assertEqual('from default config file', parser.parse_env().op1)
        env = {'APP_CFG': '{"op1": "from env config"}'}
        self.assertEqual('from env config', parser.parse_env(env).op1)
        env['APP_OP1'] = 'from env var'
        self.assertEqual('from env var', parser.parse_env(env).op1)

        ## check parse_path precedence ##
        os.remove(default_config_file)
        for key in [k for k in ['APP_CFG', 'APP_OP1'] if k in os.environ]:
            del os.environ[key]
        self.assertEqual('from parser default', parser.parse_path(input2_config_file).op1)
        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from default config file')
        self.assertEqual('from default config file', parser.parse_path(input2_config_file).op1)
        os.environ['APP_CFG'] = input1_config_file
        self.assertEqual('from input config file', parser.parse_path(input2_config_file).op1)
        os.environ['APP_OP1'] = 'from env var'
        self.assertEqual('from env var', parser.parse_path(input2_config_file).op1)
        os.environ['APP_CFG'] = input2_config_file
        self.assertEqual('from input config file', parser.parse_path(input1_config_file).op1)

        ## check parse_args precedence ##
        os.remove(default_config_file)
        for key in ['APP_CFG', 'APP_OP1']:
            del os.environ[key]
        self.assertEqual('from parser default', parser.parse_args([]).op1)
        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from default config file')
        self.assertEqual('from default config file', parser.parse_args([]).op1)
        os.environ['APP_CFG'] = input1_config_file
        self.assertEqual('from input config file', parser.parse_args([]).op1)
        os.environ['APP_OP1'] = 'from env var'
        self.assertEqual('from env var', parser.parse_args([]).op1)
        os.environ['APP_CFG'] = input2_config_file
        self.assertEqual('from arg', parser.parse_args(['--op1', 'from arg']).op1)
        self.assertEqual('from arg', parser.parse_args(['--cfg', input1_config_file, '--op1', 'from arg']).op1)
        self.assertEqual('from input config file', parser.parse_args(['--op1', 'from arg', '--cfg', input1_config_file]).op1)

        cfg = parser.parse_args(['--cfg', input1_config_file])
        cfg_list = parser.get_config_files(cfg)
        self.assertEqual(default_config_file, str(cfg_list[0]))
        self.assertEqual(input1_config_file, str(cfg_list[1]))

        for key in ['APP_CFG', 'APP_OP1']:
            del os.environ[key]


    def test_parse_as_dict(self):
        parser = ArgumentParser(parse_as_dict=True, default_meta=False)
        self.assertEqual({}, parser.parse_args([]))
        self.assertEqual({}, parser.parse_env([]))
        self.assertEqual({}, parser.parse_string('{}'))
        self.assertEqual({}, parser.parse_object({}))
        with open('config.json', 'w') as f:
            f.write('{}')
        parser = ArgumentParser(parse_as_dict=True, default_meta=True)


class ArgumentFeaturesTests(unittest.TestCase):

    def test_positionals(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('pos1')
        parser.add_argument('pos2', nargs='?')
        self.assertRaises(ParserError, lambda: parser.parse_args([]))
        self.assertIsNone(parser.parse_args(['v1']).pos2)
        self.assertEqual('v1', parser.parse_args(['v1']).pos1)
        self.assertEqual('v2', parser.parse_args(['v1', 'v2']).pos2)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('pos1')
        parser.add_argument('pos2', nargs='+')
        self.assertRaises(ParserError, lambda: parser.parse_args(['v1']).pos2)
        self.assertEqual(['v2', 'v3'], parser.parse_args(['v1', 'v2', 'v3']).pos2)

        parser.add_argument('--opt')
        parser.add_argument('--cfg',
            action=ActionConfigFile)
        cfg = parser.parse_args(['--cfg', '{"pos2": ["v2", "v3"], "opt": "v4"}', 'v1'])
        self.assertEqual('v1', cfg.pos1)
        self.assertEqual(['v2', 'v3'], cfg.pos2)
        self.assertEqual('v4', cfg.opt)


    def test_required(self):
        parser = ArgumentParser(env_prefix='APP', error_handler=None)
        group = parser.add_argument_group('Group 1')
        group.add_argument('--req1', required=True)
        parser.add_argument('--lev1.req2', required=True)
        cfg = parser.parse_args(['--req1', 'val1', '--lev1.req2', 'val2'])
        self.assertEqual('val1', cfg.req1)
        self.assertEqual('val2', cfg.lev1.req2)
        cfg = parser.parse_string('{"req1":"val3","lev1":{"req2":"val4"}}')
        self.assertEqual('val3', cfg.req1)
        self.assertEqual('val4', cfg.lev1.req2)
        cfg = parser.parse_env({'APP_REQ1': 'val5', 'APP_LEV1__REQ2': 'val6'})
        self.assertEqual('val5', cfg.req1)
        self.assertEqual('val6', cfg.lev1.req2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--req1', 'val1']))
        self.assertRaises(ParserError, lambda: parser.parse_string('{"lev1":{"req2":"val4"}}'))
        self.assertRaises(ParserError, lambda: parser.parse_env({}))

        parser = ArgumentParser(default_env=True)
        parser.add_argument('--req1', required=True)
        parser.add_argument('--cfg', action=ActionConfigFile)
        cfg = parser.parse_args(['--cfg', '{"req1": "val1"}'])
        self.assertEqual('val1', cfg.req1)


    def test_choices(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--ch1',
            choices='ABC')
        parser.add_argument('--ch2',
            choices=['v1', 'v2'])
        cfg = parser.parse_args(['--ch1', 'C', '--ch2', 'v1'])
        self.assertEqual(strip_meta(cfg), {'ch1': 'C', 'ch2': 'v1'})
        self.assertRaises(ParserError, lambda: parser.parse_args(['--ch1', 'D']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--ch2', 'v0']))


    def test_nargs(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', nargs='+', type=int)
        self.assertEqual([9],        parser.parse_args(['--val', '9']).val)
        self.assertEqual([3, 6, 2],  parser.parse_args(['--val', '3', '6', '2']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val']))
        parser = ArgumentParser()
        parser.add_argument('--val', nargs='*', type=float)
        self.assertEqual([5.2, 1.9], parser.parse_args(['--val', '5.2', '1.9']).val)
        self.assertEqual([],         parser.parse_args(['--val']).val)
        parser = ArgumentParser()
        parser.add_argument('--val', nargs='?', type=str)
        self.assertEqual('~',        parser.parse_args(['--val', '~']).val)
        self.assertEqual(None,       parser.parse_args(['--val']).val)
        parser = ArgumentParser()
        parser.add_argument('--val', nargs=2)
        self.assertEqual(['q', 'p'], parser.parse_args(['--val', 'q', 'p']).val)
        parser = ArgumentParser()
        parser.add_argument('--val', nargs=1)
        self.assertEqual(['-'],      parser.parse_args(['--val', '-']).val)


class AdvancedFeaturesTests(unittest.TestCase):

    def test_link_arguments(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--a.v1', default=2)
        parser.add_argument('--a.v2', type=int, default=3)
        parser.add_argument('--b.v2', type=int, default=4)
        def a_prod(a):
            return a['v1'] * a['v2']
        parser.link_arguments('a', 'b.v2', a_prod)

        cfg = parser.parse_args(['--a.v2=-5'])
        self.assertEqual(cfg.b.v2, cfg.a.v1*cfg.a.v2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a.v1=x']))


    def test_subcommands(self):
        parser_a = ArgumentParser(error_handler=None)
        parser_a.add_argument('ap1')
        parser_a.add_argument('--ao1',
            default='ao1_def')

        parser = ArgumentParser(prog='app', error_handler=None)
        parser.add_argument('--o1',
            default='o1_def')
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('a', parser_a)
        subcommands.add_subcommand('b', example_parser(),
            aliases=['B'],
            help='b help')

        self.assertRaises(NotImplementedError, lambda: parser.add_subparsers())
        self.assertRaises(NotImplementedError, lambda: subcommands.add_parser(''))
        self.assertRaises(ParserError, lambda: parser.parse_args(['c']))

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual(cfg, {'o1': 'o1_def', 'subcommand': None})

        parser.add_argument('--cfg', action=ActionConfigFile)
        cfg = namespace_to_dict(parser.parse_args(['--cfg={"o1": "o1_arg"}', 'a', 'ap1_arg']))
        self.assertEqual(cfg, {'a': {'ao1': 'ao1_def', 'ap1': 'ap1_arg'}, 'cfg': [None], 'o1': 'o1_arg', 'subcommand': 'a'})

        cfg = namespace_to_dict(parser.parse_args(['--o1', 'o1_arg', 'a', 'ap1_arg']))
        self.assertEqual(cfg['o1'], 'o1_arg')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(strip_meta(cfg['a']), {'ap1': 'ap1_arg', 'ao1': 'ao1_def'})
        cfg = namespace_to_dict(parser.parse_args(['a', 'ap1_arg', '--ao1', 'ao1_arg'], with_meta=False))
        self.assertEqual(cfg['a'], {'ap1': 'ap1_arg', 'ao1': 'ao1_arg'})
        self.assertRaises(KeyError, lambda: cfg['b'])

        cfg = namespace_to_dict(parser.parse_args(['b', '--lev1.lev2.opt2', 'opt2_arg']))
        cfg_def = namespace_to_dict(example_parser().get_defaults())
        cfg_def['lev1']['lev2']['opt2'] = 'opt2_arg'
        self.assertEqual(cfg['o1'], 'o1_def')
        self.assertEqual(cfg['subcommand'], 'b')
        self.assertEqual(strip_meta(cfg['b']), cfg_def)
        self.assertRaises(KeyError, lambda: cfg['a'])

        parser.parse_args(['B'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['A']))

        self.assertRaises(ParserError, lambda: parser.parse_args())
        self.assertRaises(ParserError, lambda: parser.parse_args(['a']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['b', '--unk']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['c']))

        cfg = namespace_to_dict(parser.parse_string('{"a": {"ap1": "ap1_cfg"}}'))
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(strip_meta(cfg['a']), {'ap1': 'ap1_cfg', 'ao1': 'ao1_def'})
        self.assertRaises(ParserError, lambda: parser.parse_string('{"a": {"ap1": "ap1_cfg", "unk": "unk_cfg"}}'))
        self.assertRaises(ParserError, lambda: parser.parse_string('{"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}'))

        os.environ['APP_O1'] = 'o1_env'
        os.environ['APP_A__AP1'] = 'ap1_env'
        os.environ['APP_A__AO1'] = 'ao1_env'
        os.environ['APP_B__LEV1__LEV2__OPT2'] = 'opt2_env'

        cfg = namespace_to_dict(parser.parse_args(['a'], env=True))
        self.assertEqual(cfg['o1'], 'o1_env')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(strip_meta(cfg['a']), {'ap1': 'ap1_env', 'ao1': 'ao1_env'})
        parser.default_env = True
        cfg = namespace_to_dict(parser.parse_args(['b']))
        cfg_def['lev1']['lev2']['opt2'] = 'opt2_env'
        self.assertEqual(cfg['subcommand'], 'b')
        self.assertEqual(strip_meta(cfg['b']), cfg_def)

        os.environ['APP_SUBCOMMAND'] = 'a'

        cfg = namespace_to_dict(parser.parse_env())
        self.assertEqual(cfg['o1'], 'o1_env')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(strip_meta(cfg['a']), {'ap1': 'ap1_env', 'ao1': 'ao1_env'})

        for key in ['APP_O1', 'APP_A__AP1', 'APP_A__AO1', 'APP_B__LEV1__LEV2__OPT2', 'APP_SUBCOMMAND']:
            del os.environ[key]


    def test_subsubcommands(self):
        parser_s1_a = ArgumentParser(error_handler=None)
        parser_s1_a.add_argument('--os1a',
            default='os1a_def')

        parser_s2_b = ArgumentParser(error_handler=None)
        parser_s2_b.add_argument('--os2b',
            default='os2b_def')

        parser = ArgumentParser(prog='app', error_handler=None, default_meta=False)
        subcommands1 = parser.add_subcommands()
        subcommands1.add_subcommand('a', parser_s1_a)

        subcommands2 = parser_s1_a.add_subcommands()
        subcommands2.add_subcommand('b', parser_s2_b)

        self.assertRaises(ParserError, lambda: parser.parse_args([]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['a']))

        cfg = namespace_to_dict(parser.parse_args(['a', 'b']))
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_def', 'b': {'os2b': 'os2b_def'}}})
        cfg = namespace_to_dict(parser.parse_args(['a', '--os1a=os1a_arg', 'b']))
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_arg', 'b': {'os2b': 'os2b_def'}}})
        cfg = namespace_to_dict(parser.parse_args(['a', 'b', '--os2b=os2b_arg']))
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_def', 'b': {'os2b': 'os2b_arg'}}})


    def test_subsubcommands_bad_order(self):
        parser_s1_a = ArgumentParser()
        parser_s2_b = ArgumentParser()
        parser = ArgumentParser()

        subcommands2 = parser_s1_a.add_subcommands()
        subcommands2.add_subcommand('b', parser_s2_b)

        subcommands1 = parser.add_subcommands()
        self.assertRaises(ValueError, lambda: subcommands1.add_subcommand('a', parser_s1_a))


    @unittest.skipIf(not url_support or not responses, 'validators, requests and responses packages are required')
    @responses_activate
    def test_urls(self):
        set_config_read_mode(urls_enabled=True)
        parser = ArgumentParser()
        parser.add_argument('--cfg',
            action=ActionConfigFile)
        parser.add_argument('--parser',
            action=ActionParser(parser=example_parser()))
        if jsonschema_support:
            schema = {
                'type': 'object',
                'properties': {
                    'a': {'type': 'number'},
                    'b': {'type': 'number'},
                },
            }
            parser.add_argument('--schema',
                default={'a': 1, 'b': 2},
                action=ActionJsonSchema(schema=schema))
        if jsonnet_support:
            parser.add_argument('--jsonnet',
                default={'c': 3, 'd': 4},
                action=ActionJsonnet(ext_vars=None))

        cfg1 = namespace_to_dict(parser.get_defaults())

        base_url = 'http://example.com/'
        main_body = 'parser: '+base_url+'parser.yaml\n'
        if jsonschema_support:
            main_body += 'schema: '+base_url+'schema.yaml\n'
        if jsonnet_support:
            main_body += 'jsonnet: '+base_url+'jsonnet.yaml\n'
        parser_body = example_parser().dump(cfg1['parser'])
        schema_body = jsonnet_body = ''
        if jsonschema_support:
            schema_body = json.dumps(cfg1['schema'])+'\n'
        if jsonnet_support:
            jsonnet_body = json.dumps(cfg1['jsonnet'])+'\n'

        urls = {
            'main.yaml': main_body,
            'parser.yaml': parser_body,
            'schema.yaml': schema_body,
            'jsonnet.yaml': jsonnet_body,
        }

        for name, body in urls.items():
            responses.add(responses.GET,
                          base_url+name,
                          body=body,
                          status=200)
            responses.add(responses.HEAD,
                          base_url+name,
                          status=200)

        cfg2 = parser.parse_args(['--cfg', base_url+'main.yaml'], with_meta=False)
        cfg2 = namespace_to_dict(cfg2)
        self.assertEqual(cfg1['parser'], cfg2['parser'])
        if jsonschema_support:
            self.assertEqual(cfg1['schema'], cfg2['schema'])
        if jsonnet_support:
            self.assertEqual(cfg1['jsonnet'], cfg2['jsonnet'])

        set_config_read_mode(urls_enabled=False)


class OutputTests(TempDirTestCase):

    def test_dump(self):
        parser = example_parser()
        cfg1 = parser.get_defaults()
        cfg2 = parser.parse_string(parser.dump(cfg1))
        self.assertEqual(cfg1, cfg2)
        delattr(cfg2, 'lev1')
        parser.dump(cfg2)


    def test_dump_restricted_string_type(self):
        parser = ArgumentParser()
        parser.add_argument('--str', type=NotEmptyStr)
        cfg = parser.parse_string('str: not-empty')
        self.assertEqual(parser.dump(cfg), 'str: not-empty\n')


    def test_dump_restricted_int_type(self):
        parser = ArgumentParser()
        parser.add_argument('--int', type=PositiveInt)
        cfg = parser.parse_string('int: 1')
        self.assertEqual(parser.dump(cfg), 'int: 1\n')


    def test_dump_restricted_float_type(self):
        parser = ArgumentParser()
        parser.add_argument('--float', type=PositiveFloat)
        cfg = parser.parse_string('float: 1.1')
        self.assertEqual(parser.dump(cfg), 'float: 1.1\n')


    def test_dump_path_type(self):
        parser = ArgumentParser()
        parser.add_argument('--path', type=Path_fc)
        cfg = parser.parse_string('path: path')
        self.assertEqual(parser.dump(cfg), 'path: path\n')

        parser = ArgumentParser()
        parser.add_argument('--paths', nargs='+', type=Path_fc)
        cfg = parser.parse_args(['--paths', 'path1', 'path2'])
        self.assertEqual(parser.dump(cfg), 'paths:\n- path1\n- path2\n')


    def test_dump_formats(self):
        parser = ArgumentParser()
        parser.add_argument('--op1', default=123)
        parser.add_argument('--op2', default='abc')
        cfg = parser.get_defaults()
        self.assertEqual(parser.dump(cfg), 'op1: 123\nop2: abc\n')
        self.assertEqual(parser.dump(cfg, format='yaml'), parser.dump(cfg))
        self.assertEqual(parser.dump(cfg, format='json'), '{"op1":123,"op2":"abc"}')
        self.assertEqual(parser.dump(cfg, format='json_indented'), '{\n  "op1": 123,\n  "op2": "abc"\n}\n')
        self.assertRaises(ValueError, lambda: parser.dump(cfg, format='invalid'))


    @unittest.skipIf(not dump_preserve_order_support,
                     'Dump preserve order only supported in python>=3.6 and CPython')
    def test_dump_order(self):
        args = {}
        for num in range(50):
            args[num] = ''.join(chr(randint(97, 122)) for n in range(8))

        parser = ArgumentParser()
        for num in range(len(args)):
            parser.add_argument('--'+args[num], default=num)

        cfg = parser.get_defaults()
        dump = parser.dump(cfg)
        self.assertEqual(dump, '\n'.join(v+': '+str(n) for n, v in args.items())+'\n')

        rand = list(range(len(args)))
        shuffle(rand)
        yaml = '\n'.join(args[n]+': '+str(n) for n in rand)+'\n'
        cfg = parser.parse_string(yaml)
        dump = parser.dump(cfg)
        self.assertEqual(dump, '\n'.join(v+': '+str(n) for n, v in args.items())+'\n')


    def test_save(self):
        parser = ArgumentParser()
        parser.add_argument('--parser',
            action=ActionParser(parser=example_parser()))
        if jsonschema_support:
            schema = {
                'type': 'object',
                'properties': {
                    'a': {'type': 'number'},
                    'b': {'type': 'number'},
                },
            }
            parser.add_argument('--schema',
                default={'a': 1, 'b': 2},
                action=ActionJsonSchema(schema=schema))
        if jsonnet_support:
            parser.add_argument('--jsonnet',
                default={'c': 3, 'd': 4},
                action=ActionJsonnet(ext_vars=None))

        indir = os.path.join(self.tmpdir, 'input')
        outdir = os.path.join(self.tmpdir, 'output')
        os.mkdir(outdir)
        os.mkdir(indir)
        main_file = os.path.join(indir, 'main.yaml')
        parser_file = os.path.join(indir, 'parser.yaml')
        schema_file = os.path.join(indir, 'schema.json')
        jsonnet_file = os.path.join(indir, 'jsonnet.json')

        cfg1 = parser.get_defaults()

        with open(main_file, 'w') as output_file:
            output_file.write('parser: parser.yaml\n')
            if jsonschema_support:
                output_file.write('schema: schema.json\n')
            if jsonnet_support:
                output_file.write('jsonnet: jsonnet.json\n')
        with open(parser_file, 'w') as output_file:
            output_file.write(example_parser().dump(cfg1.parser))
        if jsonschema_support:
            with open(schema_file, 'w') as output_file:
                output_file.write(json.dumps(namespace_to_dict(cfg1.schema))+'\n')
        if jsonnet_support:
            with open(jsonnet_file, 'w') as output_file:
                output_file.write(json.dumps(namespace_to_dict(cfg1.jsonnet))+'\n')

        cfg2 = parser.parse_path(main_file, with_meta=True)
        self.assertEqual(namespace_to_dict(cfg1), strip_meta(cfg2))
        self.assertEqual(str(cfg2.parser.__path__), 'parser.yaml')
        if jsonschema_support:
            self.assertEqual(str(cfg2.schema.__path__), 'schema.json')
        if jsonnet_support:
            self.assertEqual(str(cfg2.jsonnet.__path__), 'jsonnet.json')

        parser.save(cfg2, os.path.join(outdir, 'main.yaml'))
        self.assertTrue(os.path.isfile(os.path.join(outdir, 'parser.yaml')))
        if jsonschema_support:
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'schema.json')))
        if jsonnet_support:
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'jsonnet.json')))

        cfg3 = parser.parse_path(os.path.join(outdir, 'main.yaml'), with_meta=False)
        self.assertEqual(namespace_to_dict(cfg1), namespace_to_dict(cfg3))

        parser.save(cfg2, os.path.join(outdir, 'main.yaml'), multifile=False, overwrite=True)
        cfg4 = parser.parse_path(os.path.join(outdir, 'main.yaml'), with_meta=False)
        self.assertEqual(namespace_to_dict(cfg1), namespace_to_dict(cfg4))

        if jsonschema_support:
            cfg2.schema.__path__ = Path(os.path.join(indir, 'schema.yaml'), mode='fc')
            parser.save(cfg2, os.path.join(outdir, 'main.yaml'), overwrite=True)
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'schema.yaml')))


    def test_save_path_content(self):
        parser = ArgumentParser()
        parser.add_argument('--the.path', type=Path_fr)

        os.mkdir('pathdir')
        os.mkdir('outdir')
        file_txt = os.path.join('pathdir', 'file.txt')
        out_yaml = os.path.join('outdir', 'saved.yaml')
        out_file = os.path.join('outdir', 'file.txt')

        with open(file_txt, 'w') as output_file:
            output_file.write('file content')

        cfg = parser.parse_args(['--the.path', file_txt])
        parser.save_path_content.add('the.path')
        parser.save(cfg, out_yaml)

        self.assertTrue(os.path.isfile(out_yaml))
        self.assertTrue(os.path.isfile(out_file))
        with open(out_yaml) as input_file:
            self.assertEqual(input_file.read(), 'the:\n  path: file.txt\n')
        with open(out_file) as input_file:
            self.assertEqual(input_file.read(), 'file content')


    def test_save_failures(self):
        parser = ArgumentParser()
        with open('existing.yaml', 'w') as output_file:
            output_file.write('should not be overritten\n')
        cfg = parser.get_defaults()
        self.assertRaises(ValueError, lambda: parser.save(cfg, 'existing.yaml'))
        self.assertRaises(ValueError, lambda: parser.save(cfg, 'invalid_format.yaml', format='invalid'))

        parser.add_argument('--parser',
            action=ActionParser(parser=example_parser()))
        cfg = parser.get_defaults()
        with open('parser.yaml', 'w') as output_file:
            output_file.write(example_parser().dump(cfg.parser))
        cfg.parser.__path__ = Path('parser.yaml')
        self.assertRaises(ValueError, lambda: parser.save(cfg, 'main.yaml'))


    def test_print_config(self):
        parser = ArgumentParser(error_handler=None, description='cli tool')
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--v0', help=SUPPRESS, default='0')
        parser.add_argument('--v1', help='Option v1.', default=1)
        parser.add_argument('--g1.v2', help='Option v2.', default='2')
        parser2 = ArgumentParser()
        parser2.add_argument('--v3')
        parser.add_argument('--g2', action=ActionParser(parser=parser2))

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--print_config'])

        outval = yaml.safe_load(out.getvalue())
        self.assertEqual(outval, {'g1': {'v2': '2'}, 'g2': {'v3': None}, 'v1': 1})

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--print_config=skip_null'])
        outval = yaml.safe_load(out.getvalue())
        self.assertEqual(outval, {'g1': {'v2': '2'}, 'v1': 1})

        self.assertRaises(ParserError, lambda: parser.parse_args(['--print_config=bad']))

        if docstring_parser_support and ruyaml_support:
            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args(['--print_config=comments'])
            self.assertIn('# cli tool', out.getvalue())
            self.assertIn('# Option v1. (default: 1)', out.getvalue())
            self.assertIn('# Option v2. (default: 2)', out.getvalue())


class ConfigFilesTests(TempDirTestCase):

    def test_default_config_files(self):
        default_config_file = os.path.realpath(os.path.join(self.tmpdir, 'example.yaml'))
        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from default config file\n')

        parser = ArgumentParser(prog='app', default_config_files=[default_config_file])
        parser.add_argument('--op1', default='from parser default')
        parser.add_argument('--op2', default='from parser default')

        cfg = parser.get_defaults()
        self.assertEqual('from default config file', cfg.op1)
        self.assertEqual('from parser default', cfg.op2)

        with self.assertRaises(ValueError):
            parser.default_config_files = False


    def test_get_default_with_default_config_file(self):
        default_config_file = os.path.realpath(os.path.join(self.tmpdir, 'defaults.yaml'))
        parser = ArgumentParser(default_config_files=[default_config_file], error_handler=None)
        parser.add_argument('--op1', default='from default')

        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from yaml\n')

        self.assertEqual(parser.get_default('op1'), 'from yaml')

        with open(default_config_file, 'w') as output_file:
            output_file.write('op2: v2\n')
        self.assertRaises(ParserError, lambda: parser.get_default('op1'))

        out = StringIO()
        parser.print_help(out)
        outval = ' '.join(out.getvalue().split())
        self.assertIn('tried getting defaults considering default_config_files but failed', outval)

        if os.name == 'posix' and platform.python_implementation() == 'CPython':
            os.chmod(default_config_file, 0)
            self.assertEqual(parser.get_default('op1'), 'from default')


    def test_get_default_with_multiple_default_config_files(self):
        default_configs_pattern = os.path.realpath(os.path.join(self.tmpdir, 'defaults_*.yaml'))
        parser = ArgumentParser(default_config_files=[default_configs_pattern], error_handler=None)
        parser.add_argument('--op1', default='from default')
        parser.add_argument('--op2', default='from default')

        config_1 = os.path.realpath(os.path.join(self.tmpdir, 'defaults_1.yaml'))
        with open(config_1, 'w') as output_file:
            output_file.write('op1: from yaml 1\nop2: from yaml 1\n')

        cfg = parser.get_defaults()
        self.assertEqual(cfg.op1, 'from yaml 1')
        self.assertEqual(cfg.op2, 'from yaml 1')
        self.assertEqual(str(cfg.__default_config__), config_1)

        config_2 = os.path.realpath(os.path.join(self.tmpdir, 'defaults_2.yaml'))
        with open(config_2, 'w') as output_file:
            output_file.write('op1: from yaml 2\n')

        cfg = parser.get_defaults()
        self.assertEqual(cfg.op1, 'from yaml 2')
        self.assertEqual(cfg.op2, 'from yaml 1')
        self.assertIsInstance(cfg.__default_config__, list)
        self.assertEqual([str(v) for v in cfg.__default_config__], [config_1, config_2])

        config_0 = os.path.realpath(os.path.join(self.tmpdir, 'defaults_0.yaml'))
        with open(config_0, 'w') as output_file:
            output_file.write('op2: from yaml 0\n')

        cfg = parser.get_defaults()
        self.assertEqual(cfg.op1, 'from yaml 2')
        self.assertEqual(cfg.op2, 'from yaml 1')
        self.assertIsInstance(cfg.__default_config__, list)
        self.assertEqual([str(v) for v in cfg.__default_config__], [config_0, config_1, config_2])

        out = StringIO()
        parser.print_help(out)
        self.assertIn('defaults_0.yaml', out.getvalue())
        self.assertIn('defaults_1.yaml', out.getvalue())
        self.assertIn('defaults_2.yaml', out.getvalue())


    def test_ActionConfigFile(self):
        os.mkdir(os.path.join(self.tmpdir, 'subdir'))
        rel_yaml_file = os.path.join('subdir', 'config.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(self.tmpdir, rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('val: yaml\n')

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--val')

        cfg = parser.parse_args(['--cfg', abs_yaml_file, '--cfg', rel_yaml_file, '--cfg', 'val: arg'])
        self.assertEqual(3, len(cfg.cfg))
        self.assertEqual('arg', cfg.val)
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[0]()))
        self.assertEqual(abs_yaml_file, os.path.realpath(str(cfg.cfg[0])))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[1]()))
        self.assertEqual(rel_yaml_file, str(cfg.cfg[1]))
        self.assertEqual(None, cfg.cfg[2])

        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', '{"k":"v"}']))


    def test_ActionConfigFile_failures(self):
        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ValueError, lambda: parser.add_argument('--cfg', default='config.yaml', action=ActionConfigFile))
        self.assertRaises(ValueError, lambda: parser.add_argument('--nested.cfg', action=ActionConfigFile))

        parser.add_argument('--cfg', action=ActionConfigFile)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', '"""']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg=not-exist']))


class OtherTests(unittest.TestCase):

    def test_set_get_defaults(self):
        parser = ArgumentParser(default_meta=False)
        parser.add_argument('--v1', default='1')
        parser.add_argument('--g1.v2', default='2')
        nested_parser = ArgumentParser()
        nested_parser.add_argument('--g2.v3', default='3')
        parser.add_argument('--n', action=ActionParser(parser=nested_parser))
        parser.set_defaults({'g1.v2': 'b', 'n.g2.v3': 'c'}, v1='a')
        cfg = parser.get_defaults()
        self.assertEqual(namespace_to_dict(cfg), {'v1': 'a', 'g1': {'v2': 'b'}, 'n': {'g2': {'v3': 'c'}}})
        self.assertEqual(parser.get_default('v1'), cfg.v1)
        self.assertEqual(parser.get_default('g1.v2'), cfg.g1.v2)
        self.assertEqual(parser.get_default('n.g2.v3'), cfg.n.g2.v3)

        self.assertRaises(KeyError, lambda: parser.set_defaults(v4='d'))
        self.assertRaises(KeyError, lambda: parser.get_default('v4'))

        parser = ArgumentParser()
        parser.add_argument('--v1')
        parser.set_defaults(v1=1)
        self.assertEqual(parser.get_default('v1'), 1)


    def test_named_groups(self):
        parser = example_parser()
        self.assertEqual({'group1', 'group2'}, set(parser.groups.keys()))
        self.assertRaises(ValueError, lambda: parser.add_argument_group('Bad', name='group1'))


    def test_strip_unknown(self):
        base_parser = example_parser()
        ext_parser = example_parser()
        ext_parser.add_argument('--val', default='val_def')
        ext_parser.add_argument('--lev1.lev2.opt3', default='opt3_def')
        ext_parser.add_argument('--lev1.opt4', default='opt3_def')
        ext_parser.add_argument('--nums.val3', type=float, default=1.5)
        cfg = ext_parser.parse_args([])
        cfg.__path__ = 'some path'
        cfg = base_parser.strip_unknown(cfg)
        self.assertEqual(cfg.__path__, 'some path')
        base_parser.check_config(cfg, skip_none=False)


    def test_merge_config(self):
        cfg_from = Namespace(op1=1, op2=None)
        cfg_to = Namespace(op1=None, op2=2, op3=3)
        cfg = ArgumentParser.merge_config(cfg_from, cfg_to)
        self.assertEqual(cfg, Namespace(op1=1, op2=None, op3=3))


    def test_check_config_branch(self):
        parser = example_parser()
        cfg = parser.get_defaults()
        parser.check_config(cfg.lev1, branch='lev1')


    def test_usage_and_exit_error_handler(self):
        with _suppress_stderr():
            parser = ArgumentParser()
            parser.add_argument('--val', type=int)
            self.assertEqual(8, parser.parse_args(['--val', '8']).val)
            self.assertRaises(SystemExit, lambda: parser.parse_args(['--val', 'eight']))


    def test_error_handler_property(self):
        parser = ArgumentParser()
        self.assertEqual(parser.error_handler, usage_and_exit_error_handler)

        def custom_error_handler(self, message):
            print('custom_error_handler')
            self.exit(2)

        parser.error_handler = custom_error_handler
        self.assertEqual(parser.error_handler, custom_error_handler)

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--invalid'])
        self.assertEqual(out.getvalue(), 'custom_error_handler\n')

        with self.assertRaises(ValueError):
            parser.error_handler = 'invalid'


    def test_version_print(self):
        parser = ArgumentParser(prog='app', version='1.2.3')
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--version'])
        self.assertEqual(out.getvalue(), 'app 1.2.3\n')


    def test_meta_key_failures(self):
        parser = ArgumentParser()
        for meta_key in meta_keys:
            self.assertRaises(ValueError, lambda: parser.add_argument(meta_key))
        self.assertEqual(parser.default_meta, True)
        with self.assertRaises(ValueError):
            parser.default_meta = 'invalid'


    def test_invalid_parser_mode(self):
        self.assertRaises(ValueError, lambda: ArgumentParser(parser_mode='invalid'))


    def test_parse_known_args(self):
        parser = ArgumentParser()
        self.assertRaises(NotImplementedError, lambda: parser.parse_known_args([]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
