#!/usr/bin/env python3

import os
import json
import yaml
import shutil
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout
from collections import OrderedDict
from jsonargparse import *
from jsonargparse.util import _suppress_stderr
from jsonargparse.optionals import url_support, jsonschema_support, jsonnet_support
from jsonargparse_tests.util_tests import responses, responses_activate, TempDirTestCase
from jsonargparse_tests.examples import example_parser, example_yaml, example_env


class ParsersTests(TempDirTestCase):

    def test_parse_args(self):
        parser = example_parser()
        self.assertEqual('opt1_arg', parser.parse_args(['--lev1.lev2.opt1', 'opt1_arg']).lev1.lev2.opt1)
        self.assertEqual(9, parser.parse_args(['--nums.val1', '9']).nums.val1)
        self.assertEqual(6.4, parser.parse_args(['--nums.val2', '6.4']).nums.val2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val1', '7.5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val2', 'eight']))


    def test_parse_object(self):
        parser = example_parser()

        cfg = parser.parse_object(yaml.safe_load(example_yaml))
        self.assertEqual('opt1_yaml', cfg.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg.lev1.lev2.opt2)
        self.assertEqual(-1,  cfg.nums.val1)
        self.assertEqual(2.0, cfg.nums.val2)
        self.assertEqual(False, cfg.bools.def_false)
        self.assertEqual(True,  cfg.bools.def_true)


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
        env['APP_CFG'] = '{"nums": {"val1": 1}}'
        self.assertEqual(0, parser.parse_env(env).nums.val1)
        parser.add_argument('req', nargs='+')
        env['APP_REQ'] = 'abc'
        self.assertEqual(['abc'], parser.parse_env(env).req)
        env['APP_REQ'] = '["abc", "xyz"]'
        self.assertEqual(['abc', 'xyz'], parser.parse_env(env).req)


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
        self.assertTrue(hasattr(parser.parse_path(yaml_file, with_meta=True), '__cwd__'))
        self.assertFalse(hasattr(parser.parse_path(yaml_file), '__cwd__'))

        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml+'  val2: eight\n')
        self.assertRaises(ParserError, lambda: parser.parse_path(yaml_file))
        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml+'  val3: key_not_defined\n')
        self.assertRaises(ParserError, lambda: parser.parse_path(yaml_file))


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
        self.assertEqual(input1_config_file, cfg_list[0](absolute=False))

        for key in ['APP_CFG', 'APP_OP1']:
            del os.environ[key]


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


    def test_bool_type(self):
        parser = ArgumentParser(prog='app', default_env=True, error_handler=None)
        parser.add_argument('--val', type=bool)
        self.assertEqual(False, parser.get_defaults().val)
        self.assertEqual(True,  parser.parse_args(['--val', 'true']).val)
        self.assertEqual(True,  parser.parse_args(['--val', 'yes']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'false']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'no']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', '1']))
        self.assertRaises(ValueError, lambda: parser.add_argument('--val2', type=bool, nargs='+'))

        os.environ['APP_VAL'] = 'true'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'yes'
        self.assertEqual(True,  parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'false'
        self.assertEqual(False, parser.parse_args([]).val)
        os.environ['APP_VAL'] = 'no'
        self.assertEqual(False, parser.parse_args([]).val)
        del os.environ['APP_VAL']


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

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual(cfg, {'o1': 'o1_def', 'subcommand': None})

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


    @unittest.skipIf(not url_support or not responses, 'validators, requests and responses packages are required')
    @responses_activate
    def test_urls(self):
        set_url_support(True)
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

        set_url_support(False)


class OutputTests(TempDirTestCase):

    def test_dump(self):
        parser = example_parser()
        cfg1 = parser.get_defaults()
        cfg2 = parser.parse_string(parser.dump(cfg1))
        self.assertEqual(cfg1, cfg2)
        delattr(cfg2, 'lev1')
        parser.dump(cfg2)


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
        schema_file = os.path.join(indir, 'schema.yaml')
        jsonnet_file = os.path.join(indir, 'jsonnet.yaml')

        cfg1 = parser.get_defaults()

        with open(main_file, 'w') as output_file:
            output_file.write('parser: parser.yaml\n')
            if jsonschema_support:
                output_file.write('schema: schema.yaml\n')
            if jsonnet_support:
                output_file.write('jsonnet: jsonnet.yaml\n')
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
        self.assertEqual(cfg2.__path__(), main_file)
        self.assertEqual(cfg2.parser.__path__(absolute=False), 'parser.yaml')
        if jsonschema_support:
            self.assertEqual(cfg2.schema.__path__(absolute=False), 'schema.yaml')
        if jsonnet_support:
            self.assertEqual(cfg2.jsonnet.__path__(absolute=False), 'jsonnet.yaml')

        parser.save(cfg2, os.path.join(outdir, 'main.yaml'))
        self.assertTrue(os.path.isfile(os.path.join(outdir, 'parser.yaml')))
        if jsonschema_support:
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'schema.yaml')))
        if jsonnet_support:
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'jsonnet.yaml')))

        cfg3 = parser.parse_path(os.path.join(outdir, 'main.yaml'), with_meta=False)
        self.assertEqual(namespace_to_dict(cfg1), namespace_to_dict(cfg3))

        self.assertRaises(ValueError, lambda: parser.save(cfg2, os.path.join(outdir, 'main.yaml')))

        parser.save(cfg2, os.path.join(outdir, 'main.yaml'), multifile=False, overwrite=True)
        cfg4 = parser.parse_path(os.path.join(outdir, 'main.yaml'), with_meta=False)
        self.assertEqual(namespace_to_dict(cfg1), namespace_to_dict(cfg4))


    def test_print_config(self):
        parser = ArgumentParser()
        parser.add_argument('--v0', help=SUPPRESS, default='0')
        parser.add_argument('--v1', help='Option v1.', default=1)
        parser.add_argument('--g1.v2', help='Option v2.', default='2')
        parser2 = ArgumentParser()
        parser2.add_argument('--v3')
        parser.add_argument('--g2', action=ActionParser(parser=parser2))

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--print-config'])

        outval = yaml.safe_load(out.getvalue())
        self.assertEqual(outval, {'g1': {'v2': '2'}, 'g2': {'v3': None}, 'v1': 1})


    def test_default_help_formatter(self):
        parser = ArgumentParser(prog='app', default_env=True)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--v1', help='Option v1.', default='1', required=True)
        parser.add_argument('--g1.v2', help='Option v2.', default='2')
        parser2 = ArgumentParser()
        parser2.add_argument('--v3')
        parser.add_argument('--g2', action=ActionParser(parser=parser2))

        os.environ['COLUMNS'] = '150'
        out = StringIO()
        with redirect_stdout(out):
            parser.print_help()

        outval = out.getvalue()
        self.assertIn('--print-config', outval)
        self.assertIn('--cfg CFG', outval)
        self.assertIn('APP_CFG', outval)
        self.assertIn('--v1 V1', outval)
        self.assertIn('APP_V1', outval)
        self.assertIn('--g1.v2 V2', outval)
        self.assertIn('APP_G1__V2', outval)
        self.assertIn('Option v1. (required, default: 1)', outval)
        self.assertIn('Option v2. (default: 2)', outval)
        self.assertIn('--g2.help', outval)


class ConfigFilesTests(unittest.TestCase):

    def test_default_config_files(self):
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
        default_config_file = os.path.realpath(os.path.join(tmpdir, 'example.yaml'))
        with open(default_config_file, 'w') as output_file:
            output_file.write('op1: from default config file\n')

        parser = ArgumentParser(prog='app', default_config_files=[default_config_file])
        parser.add_argument('--op1', default='from parser default')
        parser.add_argument('--op2', default='from parser default')

        cfg = parser.get_defaults()
        self.assertEqual('from default config file', cfg.op1)
        self.assertEqual('from parser default', cfg.op2)

        shutil.rmtree(tmpdir)


    def test_ActionConfigFile_and_ActionPath(self):
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
        os.mkdir(os.path.join(tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(tmpdir, 'example', rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('file: '+rel_yaml_file+'\ndir: '+tmpdir+'\n')

        parser = ArgumentParser(prog='app', error_handler=None)
        parser.add_argument('--cfg',
            action=ActionConfigFile)
        parser.add_argument('--file',
            action=ActionPath(mode='fr'))
        parser.add_argument('--dir',
            action=ActionPath(mode='drw'))
        parser.add_argument('--files',
            nargs='+',
            action=ActionPath(mode='fr'))

        cfg = parser.parse_args(['--cfg', abs_yaml_file])
        self.assertEqual(tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[0](absolute=False)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[0](absolute=True)))
        self.assertEqual(rel_yaml_file, cfg.file(absolute=False))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', abs_yaml_file+'~']))

        cfg = parser.parse_args(['--cfg', 'file: '+abs_yaml_file+'\ndir: '+tmpdir+'\n'])
        self.assertEqual(tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(None, cfg.cfg[0])
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', '{"k":"v"}']))

        cfg = parser.parse_args(['--file', abs_yaml_file, '--dir', tmpdir])
        self.assertEqual(tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dir', abs_yaml_file]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--file', tmpdir]))

        cfg = parser.parse_args(['--files', abs_yaml_file, abs_yaml_file])
        self.assertTrue(isinstance(cfg.files, list))
        self.assertEqual(2, len(cfg.files))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.files[-1](absolute=True)))

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionPath))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', action=ActionPath()))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionPath(mode='+')))

        shutil.rmtree(tmpdir)


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


    def test_named_groups(self):
        parser = example_parser()
        self.assertEqual({'group1', 'group2'}, set(parser.groups.keys()))
        self.assertRaises(ValueError, lambda: parser.add_argument_group('Bad', name='group1'))


    def test_strip_unknown(self):
        base_parser = example_parser()
        ext_parser = example_parser()
        ext_parser.add_argument('--val')
        ext_parser.add_argument('--lev1.lev2.opt3', default='opt3_def')
        ext_parser.add_argument('--lev1.opt4', default='opt3_def')
        ext_parser.add_argument('--nums.val3', type=float, default=1.5)
        cfg = ext_parser.parse_args([])
        cfg = base_parser.strip_unknown(cfg)
        base_parser.check_config(cfg, skip_none=False)


    def test_usage_and_exit_error_handler(self):
        with _suppress_stderr():
            parser = ArgumentParser(prog='app', error_handler='usage_and_exit_error_handler')
            parser.add_argument('--val', type=int)
            self.assertEqual(8, parser.parse_args(['--val', '8']).val)
            self.assertRaises(SystemExit, lambda: parser.parse_args(['--val', 'eight']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
