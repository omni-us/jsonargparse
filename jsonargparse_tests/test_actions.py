#!/usr/bin/env python3

import json
import os
import pathlib
import unittest
from io import StringIO
from jsonargparse import ActionConfigFile, ActionParser, ActionPathList, ActionYesNo, ArgumentParser, ParserError, strip_meta
from jsonargparse_tests.base import TempDirTestCase
from jsonargparse_tests.test_core import example_parser


class SimpleActionsTests(unittest.TestCase):

    def test_ActionYesNo(self):
        parser = example_parser()
        defaults = parser.get_defaults()
        self.assertEqual(False, defaults.bools.def_false)
        self.assertEqual(True,  defaults.bools.def_true)
        self.assertEqual(True,  parser.parse_args(['--bools.def_false']).bools.def_false)
        self.assertEqual(False, parser.parse_args(['--no_bools.def_false']).bools.def_false)
        self.assertEqual(True,  parser.parse_args(['--bools.def_true']).bools.def_true)
        self.assertEqual(False, parser.parse_args(['--no_bools.def_true']).bools.def_true)
        self.assertEqual(True,  parser.parse_args(['--bools.def_false=true']).bools.def_false)
        self.assertEqual(False, parser.parse_args(['--bools.def_false=false']).bools.def_false)
        self.assertEqual(True,  parser.parse_args(['--bools.def_false=yes']).bools.def_false)
        self.assertEqual(False, parser.parse_args(['--bools.def_false=no']).bools.def_false)
        self.assertEqual(True,  parser.parse_args(['--no_bools.def_true=no']).bools.def_true)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--bools.def_true nope']))

        parser = ArgumentParser()
        parser.add_argument('--val', action=ActionYesNo)
        self.assertEqual(True,  parser.parse_args(['--val']).val)
        self.assertEqual(False, parser.parse_args(['--no_val']).val)
        parser = ArgumentParser()
        parser.add_argument('--with-val', action=ActionYesNo(yes_prefix='with-', no_prefix='without-'))
        self.assertEqual(True,  parser.parse_args(['--with-val']).with_val)
        self.assertEqual(False, parser.parse_args(['--without-val']).with_val)
        parser = ArgumentParser()
        self.assertRaises(ValueError, lambda: parser.add_argument('--val', action=ActionYesNo(yes_prefix='yes_')))
        self.assertRaises(ValueError, lambda: parser.add_argument('pos', action=ActionYesNo))
        self.assertRaises(ValueError, lambda: parser.add_argument('--val', nargs='?', action=ActionYesNo(no_prefix=None)))


    def test_ActionYesNo_parse_env(self):
        parser = example_parser()
        self.assertEqual(True,  parser.parse_env({'APP_BOOLS__DEF_FALSE': 'true'}).bools.def_false)
        self.assertEqual(True,  parser.parse_env({'APP_BOOLS__DEF_FALSE': 'yes'}).bools.def_false)
        self.assertEqual(False, parser.parse_env({'APP_BOOLS__DEF_TRUE': 'false'}).bools.def_true)
        self.assertEqual(False, parser.parse_env({'APP_BOOLS__DEF_TRUE': 'no'}).bools.def_true)

        parser = ArgumentParser(default_env=True, env_prefix='APP')
        parser.add_argument('--op', action=ActionYesNo, default=False)
        self.assertEqual(True, parser.parse_env({'APP_OP': 'true'}).op)


    def test_ActionYesNo_old_bool(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', nargs=1, action=ActionYesNo(no_prefix=None))
        self.assertEqual(False, parser.get_defaults().val)
        self.assertEqual(True,  parser.parse_args(['--val', 'true']).val)
        self.assertEqual(True,  parser.parse_args(['--val', 'yes']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'false']).val)
        self.assertEqual(False, parser.parse_args(['--val', 'no']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--val', '1']))


class ActionPathTests(TempDirTestCase):

    def test_ActionPathList(self):
        tmpdir = os.path.join(self.tmpdir, 'subdir')
        os.mkdir(tmpdir)
        pathlib.Path(os.path.join(tmpdir, 'file1')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file2')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file3')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file4')).touch()
        pathlib.Path(os.path.join(tmpdir, 'file5')).touch()
        list_file = os.path.join(tmpdir, 'files.lst')
        list_file2 = os.path.join(tmpdir, 'files2.lst')
        list_file3 = os.path.join(tmpdir, 'files3.lst')
        list_file4 = os.path.join(tmpdir, 'files4.lst')
        with open(list_file, 'w') as output_file:
            output_file.write('file1\nfile2\nfile3\nfile4\n')
        with open(list_file2, 'w') as output_file:
            output_file.write('file5\n')
        pathlib.Path(list_file3).touch()
        with open(list_file4, 'w') as output_file:
            output_file.write('file1\nfile2\nfile6\n')

        parser = ArgumentParser(prog='app', error_handler=None)
        parser.add_argument('--list',
            nargs='+',
            action=ActionPathList(mode='fr', rel='list'))
        parser.add_argument('--list_cwd',
            action=ActionPathList(mode='fr', rel='cwd'))

        cfg = parser.parse_args(['--list', list_file])
        self.assertEqual(4, len(cfg.list))
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [str(x) for x in cfg.list])

        cfg = parser.parse_args(['--list', list_file, list_file2])
        self.assertEqual(5, len(cfg.list))
        self.assertEqual(['file1', 'file2', 'file3', 'file4', 'file5'], [str(x) for x in cfg.list])

        self.assertEqual(0, len(parser.parse_args(['--list', list_file3]).list))

        cwd = os.getcwd()
        os.chdir(tmpdir)
        cfg = parser.parse_args(['--list_cwd', list_file])
        self.assertEqual(4, len(cfg.list_cwd))
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [str(x) for x in cfg.list_cwd])
        os.chdir(cwd)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--list']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list', list_file4]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list', 'no-such-file']))

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionPathList))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', action=ActionPathList(mode='fr'), nargs='*'))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionPathList(mode='fr', rel='.')))


class ActionParserTests(TempDirTestCase):

    def test_ActionParser(self):
        parser_lv3 = ArgumentParser(prog='lv3', default_env=False)
        parser_lv3.add_argument('--opt3',
            default='opt3_def')

        parser_lv2 = ArgumentParser(prog='lv2', default_env=False)
        parser_lv2.add_argument('--opt2',
            default='opt2_def')
        parser_lv2.add_argument('--inner3',
            action=ActionParser(parser=parser_lv3))

        parser = ArgumentParser(prog='lv1', default_env=True, error_handler=None)
        parser.add_argument('--opt1',
            default='opt1_def')
        parser.add_argument('--inner2',
            action=ActionParser(parser=parser_lv2))

        tmpdir = os.path.join(self.tmpdir, 'subdir')
        os.mkdir(tmpdir)
        yaml_main_file = os.path.join(tmpdir, 'main.yaml')
        yaml_inner2_file = os.path.join(tmpdir, 'inner2.yaml')
        yaml_inner3_file = os.path.join(tmpdir, 'inner3.yaml')

        with open(yaml_main_file, 'w') as output_file:
            output_file.write('opt1: opt1_yaml\ninner2: inner2.yaml\n')
        with open(yaml_inner2_file, 'w') as output_file:
            output_file.write('opt2: opt2_yaml\ninner3: inner3.yaml\n')
        with open(yaml_inner3_file, 'w') as output_file:
            output_file.write('opt3: opt3_yaml\n')

        ## Check defaults
        cfg = parser.get_defaults()
        self.assertEqual('opt1_def', cfg.opt1)
        self.assertEqual('opt2_def', cfg.inner2.opt2)
        self.assertEqual('opt3_def', cfg.inner2.inner3.opt3)

        ## Check ActionParser with parse_path
        expected = {'opt1': 'opt1_yaml', 'inner2': {'opt2': 'opt2_yaml', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_path(yaml_main_file)
        self.assertEqual(str(cfg.inner2.__path__), 'inner2.yaml')
        self.assertEqual(str(cfg.inner2.inner3.__path__), 'inner3.yaml')
        self.assertEqual(expected, strip_meta(cfg).as_dict())
        with open(yaml_main_file, 'w') as output_file:
            output_file.write(parser.dump(cfg))
        cfg2 = parser.parse_path(yaml_main_file, with_meta=False)
        self.assertEqual(expected, cfg2.as_dict())

        ## Check ActionParser inner environment variables
        self.assertEqual('opt2_env', parser.parse_env({'LV1_INNER2__OPT2': 'opt2_env'}).inner2.opt2)
        self.assertEqual('opt3_env', parser.parse_env({'LV1_INNER2__INNER3__OPT3': 'opt3_env'}).inner2.inner3.opt3)
        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_def', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_env({'LV1_INNER2__INNER3': yaml_inner3_file}, with_meta=False)
        self.assertEqual(expected, cfg.as_dict())
        parser.parse_env({'LV1_INNER2': yaml_inner2_file})
        self.assertEqual('opt2_yaml', parser.parse_env({'LV1_INNER2': yaml_inner2_file}).inner2.opt2)

        ## Check ActionParser as argument path
        expected = {'opt1': 'opt1_arg', 'inner2': {'opt2': 'opt2_yaml', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--opt1', 'opt1_arg', '--inner2', yaml_inner2_file], with_meta=False)
        self.assertEqual(expected, cfg.as_dict())

        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_arg', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--inner2.opt2', 'opt2_arg', '--inner2.inner3', yaml_inner3_file], with_meta=False)
        self.assertEqual(expected, cfg.as_dict())

        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_def', 'inner3': {'opt3': 'opt3_arg'}}}
        cfg = parser.parse_args(['--inner2.inner3', yaml_inner3_file, '--inner2.inner3.opt3', 'opt3_arg'], with_meta=False)
        self.assertEqual(expected, cfg.as_dict())

        ## Check ActionParser as argument string
        expected = {'opt2': 'opt2_str', 'inner3': {'opt3': 'opt3_str'}}
        cfg = parser.parse_args(['--inner2', json.dumps(expected)], with_meta=False)
        self.assertEqual(expected, cfg.as_dict()['inner2'])

        expected = {'opt3': 'opt3_str'}
        cfg = parser.parse_args(['--inner2.inner3', json.dumps(expected)], with_meta=False)
        self.assertEqual(expected, cfg.as_dict()['inner2']['inner3'])

        ## Check ActionParser with ActionConfigFile
        parser.add_argument('--cfg',
            action=ActionConfigFile)

        expected = {'opt1': 'opt1_yaml', 'inner2': {'opt2': 'opt2_yaml', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--cfg', yaml_main_file], with_meta=False)
        delattr(cfg, 'cfg')
        self.assertEqual(expected, cfg.as_dict())

        cfg = parser.parse_args(['--cfg', yaml_main_file, '--inner2.opt2', 'opt2_arg', '--inner2.inner3.opt3', 'opt3_arg'])
        self.assertEqual('opt2_arg', cfg.inner2.opt2)
        self.assertEqual('opt3_arg', cfg.inner2.inner3.opt3)


    def test_ActionParser_required(self):
        p1 = ArgumentParser()
        p1.add_argument('--op1', required=True)
        p2 = ArgumentParser(error_handler=None)
        p2.add_argument('--op2', action=ActionParser(parser=p1))
        p2.parse_args(['--op2.op1=1'])
        self.assertRaises(ParserError, lambda: p2.parse_args([]))


    def test_ActionParser_failures(self):
        parser_lv2 = ArgumentParser()
        parser_lv2.add_argument('--op')
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--inner', action=ActionParser(parser=parser_lv2))
        self.assertRaises(Exception, lambda: parser.add_argument('--bad', action=ActionParser))
        self.assertRaises(ValueError, lambda: parser.add_argument('--bad', action=ActionParser(parser=parser)))
        self.assertRaises(ValueError, lambda: parser.add_argument('--bad', type=str, action=ActionParser(ArgumentParser())))
        self.assertRaises(ValueError, lambda: parser.add_argument('-b', '--bad', action=ActionParser(ArgumentParser())))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--inner=1']))
        self.assertRaises(ValueError, lambda: ActionParser())
        self.assertRaises(ValueError, lambda: ActionParser(parser=object))


    def test_ActionParser_conflict(self):
        parser_lv2 = ArgumentParser()
        parser_lv2.add_argument('--op')
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--inner.op')
        self.assertRaises(ValueError, lambda: parser.add_argument('--inner', action=ActionParser(parser_lv2)))


    def test_ActionParser_nested_dash_names(self):
        p1 = ArgumentParser(error_handler=None)
        p1.add_argument('--op1-like')

        p2 = ArgumentParser(error_handler=None)
        p2.add_argument('--op2-like', action=ActionParser(parser=p1))

        self.assertEqual(p2.parse_args(['--op2-like.op1-like=a']).op2_like.op1_like, 'a')

        p3 = ArgumentParser(error_handler=None)
        p3.add_argument('--op3', action=ActionParser(parser=p2))

        self.assertEqual(p3.parse_args(['--op3.op2-like.op1-like=b']).op3.op2_like.op1_like, 'b')


    def test_ActionParser_action_groups(self):
        def get_parser_lv2():
            parser_lv2 = ArgumentParser(description='parser_lv2 description')
            parser_lv2.add_argument('--a1', help='lv2_a1 help')
            group_lv2 = parser_lv2.add_argument_group(description='group_lv2 description')
            group_lv2.add_argument('--a2', help='lv2_a2 help')
            return parser_lv2

        parser_lv2 = get_parser_lv2()
        parser = ArgumentParser()
        parser.add_argument('--lv2', action=ActionParser(parser_lv2))

        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertIn('parser_lv2 description', outval)
        self.assertIn('group_lv2 description', outval)
        self.assertIn('--lv2.a1 A1', outval)

        parser_lv2 = get_parser_lv2()
        parser = ArgumentParser()
        parser.add_argument(
            '--lv2',
            title='ActionParser title',
            description='ActionParser description',
            action=ActionParser(parser_lv2),
        )

        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertIn('ActionParser title', outval)
        self.assertIn('ActionParser description', outval)


if __name__ == '__main__':
    unittest.main(verbosity=2)
