#!/usr/bin/env python
"""Unit tests for the yamlargparse module."""

import os
import sys
import shutil
import tempfile
import unittest
from yamlargparse import ArgumentParser, ActionYesNo, ActionConfigFile, ActionFilePath


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

    parser.add_argument('--cfg',
        action=ActionConfigFile)
    parser.add_argument('--file',
        action=ActionFilePath(mode='r'))

    return parser


example_yaml = '''
lev1:
  lev2:
    opt1: opt1_yaml
    opt2: opt2_yaml

nums:
  val1: -1
  val2: -2.0
'''

example_env = {
    'APP_LEV1__LEV2__OPT1': 'opt1_env',
    'APP_NUMS__VAL1': 0.0
}


class YamlargparseTests(unittest.TestCase):
    """Tests for yamlargparse."""

    def test_groups(self):
        """Test storage of named groups."""
        parser = example_parser()
        self.assertEqual(['group1', 'group2'], list(parser.groups.keys()))

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
        cfg = parser.parse_yaml_from_string(example_yaml)
        self.assertEqual('opt1_yaml', cfg.lev1.lev2.opt1)
        self.assertEqual('opt2_yaml', cfg.lev1.lev2.opt2)
        self.assertEqual(-1,   cfg.nums.val1)
        self.assertEqual(-2.0, cfg.nums.val2)
        self.assertEqual(False, cfg.bools.def_false)
        self.assertEqual(True,  cfg.bools.def_true)
        cfg = parser.parse_yaml_from_string(example_yaml, defaults=False)
        self.assertFalse(hasattr(cfg, 'bools'))
        self.assertTrue(hasattr(cfg, 'nums'))

    def test_parse_env(self):
        """Test the parsing of environment variables."""
        parser = example_parser()
        cfg = parser.parse_env(env=example_env)
        self.assertEqual('opt1_env', cfg.lev1.lev2.opt1)
        self.assertEqual(0.0, cfg.nums.val1)
        cfg = parser.parse_env(env=example_env, defaults=False)
        self.assertFalse(hasattr(cfg, 'bools'))
        self.assertTrue(hasattr(cfg, 'nums'))

    def test_configfile_filepath(self):
        """Test the use of ActionConfigFile and ActionFilePath."""
        tmpdir = tempfile.mkdtemp(prefix='_yamlargparse_tests_')
        os.mkdir(os.path.join(tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.join(tmpdir, 'example', rel_yaml_file)
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write(example_yaml + '\nfile: '+rel_yaml_file+'\n')
        parser = example_parser()
        cfg = parser.parse_args(['--cfg', abs_yaml_file])
        self.assertEqual(abs_yaml_file, cfg.cfg[0](absolute=False))
        self.assertEqual(abs_yaml_file, cfg.cfg[0](absolute=True))
        self.assertEqual(rel_yaml_file, cfg.file(absolute=False))
        self.assertEqual(abs_yaml_file, cfg.file(absolute=True))
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover(__name__, pattern='yamlargparse_tests.py')
    run_tests = unittest.TextTestRunner(verbosity=2).run(tests)
    sys.exit(not run_tests.wasSuccessful())
