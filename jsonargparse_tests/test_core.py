#!/usr/bin/env python3

import json
import os
import pickle
import sys
import unittest
import unittest.mock
import warnings
import yaml
from io import StringIO
from calendar import Calendar
from contextlib import redirect_stderr, redirect_stdout
from collections import OrderedDict
from random import randint, shuffle
from typing import Optional
from jsonargparse import (
    ActionConfigFile,
    ActionJsonnet,
    ActionJsonSchema,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    Namespace,
    ParserError,
    Path,
    set_config_read_mode,
    SUPPRESS,
    strip_meta,
    usage_and_exit_error_handler,
)
from jsonargparse.namespace import meta_keys
from jsonargparse.optionals import (
    docstring_parser_support,
    dump_preserve_order_support,
    fsspec_support,
    jsonnet_support,
    jsonschema_support,
    ruyaml_support,
    url_support,
)
from jsonargparse.typing import NotEmptyStr, Path_fc, Path_fr, PositiveFloat, PositiveInt
from jsonargparse.util import CaptureParserException, capture_parser, DebugException, null_logger
from jsonargparse_tests.base import is_posix, responses_activate, responses_available, TempDirTestCase


def example_parser():
    """Creates a simple parser for doing tests."""
    parser = ArgumentParser(prog='app', default_meta=False, error_handler=None)

    group_one = parser.add_argument_group('Group 1', name='group1')
    group_one.add_argument('--bools.def_false',
        default=False,
        nargs='?',
        action=ActionYesNo)
    group_one.add_argument('--bools.def_true',
        default=True,
        nargs='?',
        action=ActionYesNo)

    group_two = parser.add_argument_group('Group 2', name='group2')
    group_two.add_argument('--lev1.lev2.opt1',
        default='opt1_def')
    group_two.add_argument('--lev1.lev2.opt2',
        default='opt2_def')

    group_three = parser.add_argument_group('Group 3')
    group_three.add_argument('--nums.val1',
        type=int,
        default=1)
    group_three.add_argument('--nums.val2',
        type=float,
        default=2.0)

    return parser


example_yaml = '''
lev1:
  lev2:
    opt1: opt1_yaml
    opt2: opt2_yaml

nums:
  val1: -1
'''

example_env = {
    'APP_LEV1__LEV2__OPT1': 'opt1_env',
    'APP_NUMS__VAL1': '0'
}


class ParsersTests(TempDirTestCase):

    def test_parse_args(self):
        parser = example_parser()
        self.assertEqual('opt1_arg', parser.parse_args(['--lev1.lev2.opt1', 'opt1_arg']).lev1.lev2.opt1)
        self.assertEqual(9, parser.parse_args(['--nums.val1', '9']).nums.val1)
        self.assertEqual(6.4, parser.parse_args(['--nums.val2', '6.4']).nums.val2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val1', '7.5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val2', 'eight']))
        self.assertEqual(9, parser.parse_args(['--nums.val1', '9'])['nums.val1'])


    def test_parse_object(self):
        parser = example_parser()

        base = Namespace(**{'nums.val2': 3.4})
        cfg = parser.parse_object(yaml.safe_load(example_yaml), cfg_base=base)
        self.assertEqual('opt1_yaml', cfg.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg.lev1.lev2.opt2)
        self.assertEqual(-1,  cfg.nums.val1)
        self.assertEqual(3.4, cfg.nums.val2)
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
        with unittest.mock.patch.dict(os.environ, {'JSONARGPARSE_DEFAULT_ENV': 'True'}):
            parser = ArgumentParser()
            self.assertTrue(parser.default_env)
            parser.default_env = False
            self.assertTrue(parser.default_env)
        with unittest.mock.patch.dict(os.environ, {'JSONARGPARSE_DEFAULT_ENV': 'False'}):
            parser = ArgumentParser(default_env=True)
            self.assertFalse(parser.default_env)
            parser.default_env = True
            self.assertFalse(parser.default_env)


    def test_env_prefix(self):
        parser = ArgumentParser(env_prefix=True, default_env=True, error_handler=None)
        parser.add_argument("--test_arg", type=str, required=True, help="Test argument")
        with self.assertRaises(ParserError):
            with unittest.mock.patch.dict(os.environ, {'TEST_ARG': 'one'}):
                parser.parse_args([])
        prefix = os.path.splitext(parser.prog)[0].upper()
        with unittest.mock.patch.dict(os.environ, {f'{prefix}_TEST_ARG': 'one'}):
            cfg = parser.parse_args([])
            self.assertEqual('one', cfg.test_arg)

        parser = ArgumentParser(env_prefix=False, default_env=True)
        parser.add_argument("--test_arg", type=str, required=True, help="Test argument")
        with unittest.mock.patch.dict(os.environ, {'TEST_ARG': 'one'}):
            cfg = parser.parse_args([])
            self.assertEqual('one', cfg.test_arg)


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
        self.assertEqual(input2_config_file, str(cfg_list[1]))  # From os.environ['APP_CFG']
        self.assertEqual(input1_config_file, str(cfg_list[2]))

        for key in ['APP_CFG', 'APP_OP1']:
            del os.environ[key]


    def test_parse_unexpected_kwargs(self):
        with self.assertRaises(ValueError):
            ArgumentParser().parse_args([], unexpected=True)


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

        out = StringIO()
        parser.print_help(out)
        self.assertIn('[-h] --req1 REQ1 --lev1.req2 REQ2', out.getvalue())

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
        self.assertEqual(cfg.as_dict(), {'ch1': 'C', 'ch2': 'v1'})
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
        self.assertRaises(NotImplementedError, lambda: subcommands.add_parser(''))
        self.assertRaises(ParserError, lambda: parser.parse_args(['c']))

        cfg = parser.get_defaults().as_dict()
        self.assertEqual(cfg, {'o1': 'o1_def', 'subcommand': None})

        parser.add_argument('--cfg', action=ActionConfigFile)
        cfg = parser.parse_args(['--cfg={"o1": "o1_arg"}', 'a', 'ap1_arg']).as_dict()
        self.assertEqual(cfg, {'a': {'ao1': 'ao1_def', 'ap1': 'ap1_arg'}, 'cfg': [None], 'o1': 'o1_arg', 'subcommand': 'a'})

        cfg = parser.parse_args(['--o1', 'o1_arg', 'a', 'ap1_arg'])
        self.assertEqual(cfg['o1'], 'o1_arg')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(cfg['a'].as_dict(), {'ap1': 'ap1_arg', 'ao1': 'ao1_def'})
        cfg = parser.parse_args(['a', 'ap1_arg', '--ao1', 'ao1_arg'])
        self.assertEqual(cfg['a'].as_dict(), {'ap1': 'ap1_arg', 'ao1': 'ao1_arg'})
        self.assertRaises(KeyError, lambda: cfg['b'])

        cfg = parser.parse_args(['b', '--lev1.lev2.opt2', 'opt2_arg']).as_dict()
        cfg_def = example_parser().get_defaults().as_dict()
        cfg_def['lev1']['lev2']['opt2'] = 'opt2_arg'
        self.assertEqual(cfg['o1'], 'o1_def')
        self.assertEqual(cfg['subcommand'], 'b')
        self.assertEqual(cfg['b'], cfg_def)
        self.assertRaises(KeyError, lambda: cfg['a'])

        parser.parse_args(['B'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['A']))

        self.assertRaises(ParserError, lambda: parser.parse_args())
        self.assertRaises(ParserError, lambda: parser.parse_args(['a']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['b', '--unk']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['c']))

        cfg = parser.parse_string('{"a": {"ap1": "ap1_cfg"}}').as_dict()
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(cfg['a'], {'ap1': 'ap1_cfg', 'ao1': 'ao1_def'})
        self.assertRaises(ParserError, lambda: parser.parse_string('{"a": {"ap1": "ap1_cfg", "unk": "unk_cfg"}}'))

        with warnings.catch_warnings(record=True) as w:
            cfg = parser.parse_string('{"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}')
            self.assertEqual(cfg.subcommand, 'a')
            self.assertFalse(hasattr(cfg, 'b'))
            self.assertEqual(len(w), 1)
            self.assertIn('Subcommand "a" will be used', str(w[0].message))

        cfg = parser.parse_string('{"subcommand": "b", "a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}')
        self.assertFalse(hasattr(cfg, 'a'))

        cfg = parser.parse_args(['--cfg={"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}', 'a'])
        cfg = cfg.as_dict()
        self.assertEqual(cfg, {'o1': 'o1_def', 'subcommand': 'a', 'cfg': [None], 'a': {'ap1': 'ap1_cfg', 'ao1': 'ao1_def'}})
        cfg = parser.parse_args(['--cfg={"a": {"ap1": "ap1_cfg"}, "b": {"nums": {"val1": 2}}}', 'b'])
        self.assertFalse(hasattr(cfg, 'a'))
        self.assertTrue(hasattr(cfg, 'b'))

        os.environ['APP_O1'] = 'o1_env'
        os.environ['APP_A__AP1'] = 'ap1_env'
        os.environ['APP_A__AO1'] = 'ao1_env'
        os.environ['APP_B__LEV1__LEV2__OPT2'] = 'opt2_env'

        cfg = parser.parse_args(['a'], env=True).as_dict()
        self.assertEqual(cfg['o1'], 'o1_env')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(cfg['a'], {'ap1': 'ap1_env', 'ao1': 'ao1_env'})
        parser.default_env = True
        cfg = parser.parse_args(['b']).as_dict()
        cfg_def['lev1']['lev2']['opt2'] = 'opt2_env'
        self.assertEqual(cfg['subcommand'], 'b')
        self.assertEqual(cfg['b'], cfg_def)

        os.environ['APP_SUBCOMMAND'] = 'a'

        cfg = parser.parse_env().as_dict()
        self.assertEqual(cfg['o1'], 'o1_env')
        self.assertEqual(cfg['subcommand'], 'a')
        self.assertEqual(cfg['a'], {'ap1': 'ap1_env', 'ao1': 'ao1_env'})

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

        cfg = parser.parse_args(['a', 'b']).as_dict()
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_def', 'b': {'os2b': 'os2b_def'}}})
        cfg = parser.parse_args(['a', '--os1a=os1a_arg', 'b']).as_dict()
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_arg', 'b': {'os2b': 'os2b_def'}}})
        cfg = parser.parse_args(['a', 'b', '--os2b=os2b_arg']).as_dict()
        self.assertEqual(cfg, {'subcommand': 'a', 'a': {'subcommand': 'b', 'os1a': 'os1a_def', 'b': {'os2b': 'os2b_arg'}}})


    def test_subsubcommands_bad_order(self):
        parser_s1_a = ArgumentParser()
        parser_s2_b = ArgumentParser()
        parser = ArgumentParser()

        subcommands2 = parser_s1_a.add_subcommands()
        subcommands2.add_subcommand('b', parser_s2_b)

        subcommands1 = parser.add_subcommands()
        self.assertRaises(ValueError, lambda: subcommands1.add_subcommand('a', parser_s1_a))


    def test_optional_subcommand(self):
        parser = ArgumentParser(error_handler=None)
        subcommands = parser.add_subcommands(required=False)
        subparser = ArgumentParser()
        subcommands.add_subcommand('foo', subparser)
        cfg = parser.parse_args([])
        self.assertEqual(cfg, Namespace(subcommand=None))


    def test_subcommand_without_options(self):
        parser = ArgumentParser()
        subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        subcommands.add_subcommand('foo', subparser)
        cfg = parser.parse_args(['foo'])
        self.assertEqual(cfg.subcommand, 'foo')


    def test_subcommand_print_config_default_env_issue_126(self):
        subparser = ArgumentParser()
        subparser.add_argument('--config', action=ActionConfigFile)
        subparser.add_argument('--o', type=int, default=1)

        parser = ArgumentParser(error_handler=None, default_env=True)
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('a', subparser)

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['a', '--print_config'])
        self.assertEqual(yaml.safe_load(out.getvalue()), {'o': 1})


    @unittest.skipIf(not (url_support and responses_available), 'validators, requests and responses packages are required')
    @responses_activate
    def test_urls(self):
        set_config_read_mode(urls_enabled=True)
        parser = ArgumentParser(error_handler=None)
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

        cfg1 = parser.get_defaults()

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

        import responses
        for name, body in urls.items():
            responses.add(responses.GET,
                          base_url+name,
                          body=body,
                          status=200)
            responses.add(responses.HEAD,
                          base_url+name,
                          status=200)

        cfg2 = parser.parse_args(['--cfg', base_url+'main.yaml'], with_meta=False)
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


    def test_dump_skip_default(self):
        parser = ArgumentParser()
        parser.add_argument('--op1', default=123)
        parser.add_argument('--op2', default='abc')
        self.assertEqual(parser.dump(parser.get_defaults(), skip_default=True), '{}\n')
        self.assertEqual(parser.dump(Namespace(op1=123, op2='xyz'), skip_default=True), 'op2: xyz\n')


    def test_dump_skip_default_nested(self):
        parser = ArgumentParser()
        parser.add_argument('--g1.op1', type=int, default=123)
        parser.add_argument('--g1.op2', type=str, default='abc')
        parser.add_argument('--g2.op1', type=int, default=987)
        parser.add_argument('--g2.op2', type=str, default='xyz')
        self.assertEqual(parser.dump(parser.get_defaults(), skip_default=True), '{}\n')
        self.assertEqual(parser.dump(parser.parse_args(['--g1.op1=0']), skip_default=True), 'g1:\n  op1: 0\n')
        self.assertEqual(parser.dump(parser.parse_args(['--g2.op2=pqr']), skip_default=True), 'g2:\n  op2: pqr\n')


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
        parser = ArgumentParser(error_handler=None)
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
        main_file_in = os.path.join(indir, 'main.yaml')
        parser_file_in = os.path.join(indir, 'parser.yaml')
        schema_file_in = os.path.join(indir, 'schema.json')
        jsonnet_file_in = os.path.join(indir, 'jsonnet.json')
        main_file_out = os.path.join(outdir, 'main.yaml')
        parser_file_out = os.path.join(outdir, 'parser.yaml')
        schema_file_out = os.path.join(outdir, 'schema.json')
        jsonnet_file_out = os.path.join(outdir, 'jsonnet.json')
        cfg1 = parser.get_defaults()

        with open(main_file_in, 'w') as output_file:
            output_file.write('parser: parser.yaml\n')
            if jsonschema_support:
                output_file.write('schema: schema.json\n')
            if jsonnet_support:
                output_file.write('jsonnet: jsonnet.json\n')
        with open(parser_file_in, 'w') as output_file:
            output_file.write(example_parser().dump(cfg1.parser))
        if jsonschema_support:
            with open(schema_file_in, 'w') as output_file:
                output_file.write(json.dumps(cfg1.schema)+'\n')
        if jsonnet_support:
            with open(jsonnet_file_in, 'w') as output_file:
                output_file.write(json.dumps(cfg1.jsonnet)+'\n')

        cfg2 = parser.parse_path(main_file_in, with_meta=True)
        self.assertEqual(cfg1.as_dict(), strip_meta(cfg2).as_dict())
        self.assertEqual(str(cfg2.parser['__path__']), 'parser.yaml')
        if jsonschema_support:
            self.assertEqual(str(cfg2.schema['__path__']), 'schema.json')
        if jsonnet_support:
            self.assertEqual(str(cfg2.jsonnet['__path__']), 'jsonnet.json')

        parser.save(cfg2, main_file_out)
        self.assertTrue(os.path.isfile(parser_file_out))
        if jsonschema_support:
            self.assertTrue(os.path.isfile(schema_file_out))
        if jsonnet_support:
            self.assertTrue(os.path.isfile(jsonnet_file_out))

        for file in [main_file_out, parser_file_out, schema_file_out, jsonnet_file_out]:
            if os.path.isfile(file):
                os.remove(file)
        parser.save(cfg2, main_file_out)
        self.assertTrue(os.path.isfile(parser_file_out))
        if jsonschema_support:
            self.assertTrue(os.path.isfile(schema_file_out))
        if jsonnet_support:
            self.assertTrue(os.path.isfile(jsonnet_file_out))

        cfg3 = parser.parse_path(main_file_out, with_meta=False)
        self.assertEqual(cfg1.as_dict(), cfg3.as_dict())

        parser.save(cfg2, main_file_out, multifile=False, overwrite=True)
        cfg4 = parser.parse_path(main_file_out, with_meta=False)
        self.assertEqual(cfg1, cfg4)

        if jsonschema_support:
            cfg2.schema['__path__'] = Path(os.path.join(indir, 'schema.yaml'), mode='fc')
            parser.save(cfg2, main_file_out, overwrite=True)
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


    @unittest.skipIf(not fsspec_support, 'fsspec package is required')
    def test_save_fsspec(self):
        parser = example_parser()
        cfg = parser.get_defaults()
        parser.save(cfg, 'memory://config.yaml', multifile=False)
        path = Path('memory://config.yaml', mode='sr')
        self.assertEqual(cfg, parser.parse_string(path.get_content()))
        self.assertRaises(NotImplementedError, lambda: parser.save(cfg, 'memory://config.yaml', multifile=True))


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
        self.assertEqual(outval, {'g1': {'v2': '2'}, 'g2': {}, 'v1': 1})

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

        parser.add_subclass_arguments(Calendar, 'cal')
        self.assertRaises(KeyError, lambda: parser.get_default('cal'))

        with open(default_config_file, 'w') as output_file:
            output_file.write('op2: v2\n')
        self.assertRaises(ParserError, lambda: parser.get_default('op1'))

        out = StringIO()
        parser.print_help(out)
        outval = ' '.join(out.getvalue().split())
        self.assertIn('tried getting defaults considering default_config_files but failed', outval)

        if is_posix:
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
        self.assertEqual(cfg.as_dict(), {'v1': 'a', 'g1': {'v2': 'b'}, 'n': {'g2': {'v3': 'c'}}})
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
        parser = ArgumentParser()
        for key in [1, 2, 3]:
            parser.add_argument(f'--op{key}', type=Optional[int])
        cfg_from = Namespace(op1=1, op2=None)
        cfg_to = Namespace(op1=None, op2=2, op3=3)
        cfg = parser.merge_config(cfg_from, cfg_to)
        self.assertEqual(cfg, Namespace(op1=1, op2=None, op3=3))


    def test_check_config_branch(self):
        parser = example_parser()
        cfg = parser.get_defaults()
        parser.check_config(cfg.lev1, branch='lev1')


    def test_usage_and_exit_error_handler(self):
        err = StringIO()
        with redirect_stderr(err):
            parser = ArgumentParser()
            parser.add_argument('--val', type=int)
            self.assertEqual(8, parser.parse_args(['--val', '8']).val)
            self.assertRaises(SystemExit, lambda: parser.parse_args(['--val', 'eight']))
        self.assertIn('Parser key "val":', err.getvalue())


    @unittest.mock.patch.dict(os.environ, {'JSONARGPARSE_DEBUG': ''})
    def test_debug_usage_and_exit_error_handler(self):
        parser = ArgumentParser(logger=null_logger)
        parser.add_argument('--int', type=int)
        err = StringIO()
        with redirect_stderr(err), self.assertRaises(DebugException):
            parser.parse_args(['--int=invalid'])


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


    def test_parse_args_invalid_args(self):
        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ParserError, lambda: parser.parse_args([{}]))


    @unittest.skipIf(sys.version_info[:2] == (3, 6), 'loggers not pickleable in python 3.6')
    def test_pickle_parser(self):
        parser1 = example_parser()
        parser2 = pickle.loads(pickle.dumps(parser1))
        self.assertEqual(parser1.get_defaults(), parser2.get_defaults())


    def test_unrecognized_arguments(self):
        parser = ArgumentParser()
        err = StringIO()
        with redirect_stderr(err):
            self.assertRaises(SystemExit, lambda: parser.parse_args(['--unrecognized']))
        self.assertIn('Unrecognized arguments:', err.getvalue())


    def test_capture_parser(self):
        def parse_args(args=[]):
            parser = ArgumentParser()
            parser.add_argument('--int', type=int, default=1)
            return parser.parse_args(args)

        parser = capture_parser(parse_args, ['--int=2'])
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual(parser.get_defaults(), Namespace(int=1))

        with self.assertRaises(CaptureParserException):
            capture_parser(lambda: None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
