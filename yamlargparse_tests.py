#!/usr/bin/env python
"""Unit tests for the yamlargparse module."""

import os
import sys
import shutil
import tempfile
import pathlib
import unittest
from yamlargparse import *


def example_parser():
    """Creates a simple parser for doing tests."""
    parser = ArgumentParser(prog='app')

    group_one = parser.add_argument_group('Group 1', name='group1')
    group_one.add_argument('--bools.def_false',
        default=False,
        action=ActionYesNo)
    group_one.add_argument('--bools.def_true',
        default=True,
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


class YamlargparseTests(unittest.TestCase):
    """Tests for yamlargparse."""

    def test_groups(self):
        """Test storage of named groups."""
        parser = example_parser()
        self.assertEqual(['group1', 'group2'], list(sorted(parser.groups.keys())))


    def test_parse_args(self):
        parser = example_parser()
        self.assertEqual('opt1_arg', parser.parse_args(['--lev1.lev2.opt1', 'opt1_arg']).lev1.lev2.opt1)
        self.assertEqual(9, parser.parse_args(['--nums.val1', '9']).nums.val1)
        self.assertEqual(6.4, parser.parse_args(['--nums.val2', '6.4']).nums.val2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val1', '7.5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--nums.val2', 'eight']))


    def test_usage_and_exit_error_handler(self):
        parser = ArgumentParser(prog='app', error_handler=usage_and_exit_error_handler)
        parser.add_argument('--val', type=int)
        self.assertEqual(8, parser.parse_args(['--val', '8']).val)
        sys.stderr.write('\n')
        self.assertRaises(SystemExit, lambda: parser.parse_args(['--val', 'eight']))


    def test_yes_no_action(self):
        """Test the correct functioning of ActionYesNo."""
        parser = example_parser()
        defaults = parser.get_defaults()
        self.assertEqual(False, defaults.bools.def_false)
        self.assertEqual(True,  defaults.bools.def_true)
        self.assertEqual(True,  parser.parse_args(['--bools.def_false']).bools.def_false)
        self.assertEqual(False, parser.parse_args(['--no_bools.def_false']).bools.def_false)
        self.assertEqual(True,  parser.parse_args(['--bools.def_true']).bools.def_true)
        self.assertEqual(False, parser.parse_args(['--no_bools.def_true']).bools.def_true)


    def test_parse_yaml(self):
        """Test the parsing and checking of yaml."""
        parser = example_parser()

        cfg1 = parser.parse_yaml_string(example_yaml)
        self.assertEqual('opt1_yaml', cfg1.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg1.lev1.lev2.opt2)
        self.assertEqual(-1,   cfg1.nums.val1)
        self.assertEqual(2.0, cfg1.nums.val2)
        self.assertEqual(False, cfg1.bools.def_false)
        self.assertEqual(True,  cfg1.bools.def_true)

        cfg2 = parser.parse_yaml_string(example_yaml, defaults=False)
        self.assertFalse(hasattr(cfg2, 'bools'))
        self.assertTrue(hasattr(cfg2, 'nums'))

        tmpdir = tempfile.mkdtemp(prefix='_yamlargparse_tests_')
        yaml_file = os.path.realpath(os.path.join(tmpdir, 'example.yaml'))

        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml)
        self.assertEqual(cfg1, parser.parse_yaml_path(yaml_file, defaults=True))
        self.assertEqual(cfg2, parser.parse_yaml_path(yaml_file, defaults=False))
        self.assertNotEqual(cfg2, parser.parse_yaml_path(yaml_file, defaults=True))
        self.assertNotEqual(cfg1, parser.parse_yaml_path(yaml_file, defaults=False))

        with open(yaml_file, 'w') as output_file:
            output_file.write(example_yaml+'  val2: eight\n')
        self.assertRaises(ParserError, lambda: parser.parse_yaml_path(yaml_file))

        shutil.rmtree(tmpdir)


    def test_parse_env(self):
        """Test the parsing of environment variables."""
        parser = example_parser()
        cfg = parser.parse_env(env=example_env)
        self.assertEqual('opt1_env', cfg.lev1.lev2.opt1)
        self.assertEqual(0, cfg.nums.val1)
        cfg = parser.parse_env(env=example_env, defaults=False)
        self.assertFalse(hasattr(cfg, 'bools'))
        self.assertTrue(hasattr(cfg, 'nums'))


    def test_default_config_files(self):
        """Test the use of default_config_files."""
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_yamlargparse_tests_'))
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


    def test_configfile_filepath(self):
        """Test the use of ActionConfigFile and ActionPath."""
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_yamlargparse_tests_'))
        os.mkdir(os.path.join(tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(tmpdir, 'example', rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('file: '+rel_yaml_file+'\ndir: '+tmpdir+'\n')

        parser = ArgumentParser(prog='app')
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
        self.assertRaises(KeyError, lambda: parser.parse_args(['--cfg', '{"k":"v"}']))

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


    def test_filepathlist(self):
        """Test the use of ActionPathList."""
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_yamlargparse_tests_'))
        pathlib.Path(os.path.join(tmpdir, 'file1')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file2')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file3')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file4')).touch()
        list_file = os.path.join(tmpdir, 'files.lst')
        with open(list_file, 'w') as output_file:
            output_file.write('file1\nfile2\nfile3\nfile4\n')

        parser = ArgumentParser(prog='app')
        parser.add_argument('--list',
            action=ActionPathList(mode='r', rel='list'))
        parser.add_argument('--list_cwd',
            action=ActionPathList(mode='r', rel='cwd'))

        cfg = parser.parse_args(['--list', list_file])
        self.assertEqual(4, len(cfg.list))
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [x(absolute=False) for x in cfg.list])

        cwd = os.getcwd()
        os.chdir(tmpdir)
        cfg = parser.parse_args(['--list_cwd', list_file])
        self.assertEqual(4, len(cfg.list_cwd))
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [x(absolute=False) for x in cfg.list_cwd])
        os.chdir(cwd)

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionPathList))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', action=ActionPathList(mode='r'), nargs='+'))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionPathList(mode='r', rel='.')))

        shutil.rmtree(tmpdir)


    def test_actionparser(self):
        """Test the use of ActionParser."""
        yaml1_str = 'root:\n  child: from single\n'
        yaml2_str = 'root:\n  example3.yaml\n'
        yaml3_str = 'child: from example3\n'

        parser2 = ArgumentParser()
        parser2.add_argument('--child')
        parser1 = ArgumentParser(prog='app')
        parser1.add_argument('--root',
            action=ActionParser(parser=parser2))
        parser1.add_argument('--cfg',
            action=ActionConfigFile)

        tmpdir = tempfile.mkdtemp(prefix='_yamlargparse_tests_')
        os.mkdir(os.path.join(tmpdir, 'example'))
        yaml1_file = os.path.join(tmpdir, 'example1.yaml')
        yaml2_file = os.path.join(tmpdir, 'example2.yaml')
        yaml3_file = os.path.join(tmpdir, 'example3.yaml')
        with open(yaml1_file, 'w') as output_file:
            output_file.write(yaml1_str)
        with open(yaml2_file, 'w') as output_file:
            output_file.write(yaml2_str)
        with open(yaml3_file, 'w') as output_file:
            output_file.write(yaml3_str)

        self.assertEqual('from single', parser1.parse_args(['--cfg', yaml1_file]).root.child)
        self.assertEqual('from example3', parser1.parse_args(['--cfg', yaml2_file]).root.child)
        self.assertEqual('from single', parser1.parse_yaml_string(yaml1_str).root.child)
        self.assertEqual('from example3', parser1.parse_yaml_path(yaml2_file).root.child)

        self.assertRaises(ValueError, lambda: parser1.add_argument('--op1', action=ActionParser))
        self.assertRaises(ValueError, lambda: parser1.add_argument('--op2', action=ActionParser()))

        shutil.rmtree(tmpdir)


    @unittest.skipIf(isinstance(jsonvalidator, Exception), 'jsonschema package is required :: '+str(jsonvalidator))
    def test_jsonschema(self):
        """Test the use of ActionJsonSchema."""

        schema1 = {
            "type": "array",
            "items": { "type": "integer" }
        }

        schema2 = {
            "type": "object",
            "properties": {
                "k1": { "type": "string" },
                "k2": { "type": "integer" },
                "k3": { "type": "number" }
            },
            "additionalProperties": False
        }

        parser = ArgumentParser(prog='app')
        parser.add_argument('--op1',
            action=ActionJsonSchema(schema=schema1))
        parser.add_argument('--op2',
            action=ActionJsonSchema(schema=schema2))

        self.assertEqual([1, 2, 3, 4], parser.parse_args(['--op1', '[1, 2, 3, 4]']).op1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '[1, "two"]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op1', '[1.5, 2]']))

        self.assertEqual({"k1": "one", "k2": 2, "k3": 3.3}, vars(parser.parse_args(['--op2', '{"k1": "one", "k2": 2, "k3": 3.3}']).op2))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k1": 1}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k2": "2"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', '{"k4": 4}']))


    def test_operators(self):
        """Test the use of ActionOperators."""
        parser = ArgumentParser(prog='app')
        parser.add_argument('--le0',
            action=ActionOperators(expr=('<', 0)))
        parser.add_argument('--gt1.a.le4',
            action=ActionOperators(expr=[('>', 1.0), ('<=', 4.0)], join='and', type=float))
        parser.add_argument('--lt5.o.ge10.o.eq7',
            action=ActionOperators(expr=[('<', 5), ('>=', 10), ('==', 7)], join='or', type=int))
        def int_or_off(x): return x if x == 'off' else int(x)
        parser.add_argument('--gt0.o.off',
            action=ActionOperators(expr=[('>', 0), ('==', 'off')], join='or', type=int_or_off))
        parser.add_argument('--ge0',
            nargs=3,
            action=ActionOperators(expr=('>=', 0)))

        self.assertEqual(1.5, parser.parse_args(['--gt1.a.le4', '1.5']).gt1.a.le4)
        self.assertEqual(4.0, parser.parse_args(['--gt1.a.le4', '4.0']).gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt1.a.le4', '1.0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt1.a.le4', '5.5']))

        self.assertEqual(1.5, parser.parse_yaml_string('gt1:\n  a:\n    le4: 1.5').gt1.a.le4)
        self.assertEqual(4.0, parser.parse_yaml_string('gt1:\n  a:\n    le4: 4.0').gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_yaml_string('gt1:\n  a:\n    le4: 1.0'))
        self.assertRaises(ParserError, lambda: parser.parse_yaml_string('gt1:\n  a:\n    le4: 5.5'))

        self.assertEqual(1.5, parser.parse_env(env={'APP_GT1__A__LE4': '1.5'}).gt1.a.le4)
        self.assertEqual(4.0, parser.parse_env(env={'APP_GT1__A__LE4': '4.0'}).gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_env(env={'APP_GT1__A__LE4': '1.0'}))
        self.assertRaises(ParserError, lambda: parser.parse_env(env={'APP_GT1__A__LE4': '5.5'}))

        self.assertEqual(2, parser.parse_args(['--lt5.o.ge10.o.eq7', '2']).lt5.o.ge10.o.eq7)
        self.assertEqual(7, parser.parse_args(['--lt5.o.ge10.o.eq7', '7']).lt5.o.ge10.o.eq7)
        self.assertEqual(10, parser.parse_args(['--lt5.o.ge10.o.eq7', '10']).lt5.o.ge10.o.eq7)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--lt5.o.ge10.o.eq7', '5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--lt5.o.ge10.o.eq7', '8']))

        self.assertEqual(9, parser.parse_args(['--gt0.o.off', '9']).gt0.o.off)
        self.assertEqual('off', parser.parse_args(['--gt0.o.off', 'off']).gt0.o.off)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt0.o.off', 'on']))

        self.assertEqual([0, 1, 2], parser.parse_args(['--ge0', '0', '1', '2']).ge0)

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionOperators))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', action=ActionOperators()))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionOperators(expr='<')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
