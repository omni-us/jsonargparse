#!/usr/bin/env python3

import enum
import json
import shutil
import tempfile
import pathlib
import unittest
from io import StringIO
from jsonargparse import *
from jsonargparse_tests.examples import example_parser


class ActionsTests(unittest.TestCase):

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

        self.assertEqual(True,  parser.parse_env({'APP_BOOLS__DEF_FALSE': 'true'}).bools.def_false)
        self.assertEqual(True,  parser.parse_env({'APP_BOOLS__DEF_FALSE': 'yes'}).bools.def_false)
        self.assertEqual(False, parser.parse_env({'APP_BOOLS__DEF_TRUE': 'false'}).bools.def_true)
        self.assertEqual(False, parser.parse_env({'APP_BOOLS__DEF_TRUE': 'no'}).bools.def_true)

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


    def test_ActionPathList(self):
        tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
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
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [x(absolute=False) for x in cfg.list])

        cfg = parser.parse_args(['--list', list_file, list_file2])
        self.assertEqual(5, len(cfg.list))
        self.assertEqual(['file1', 'file2', 'file3', 'file4', 'file5'], [x(absolute=False) for x in cfg.list])

        self.assertEqual(0, len(parser.parse_args(['--list', list_file3]).list))

        cwd = os.getcwd()
        os.chdir(tmpdir)
        cfg = parser.parse_args(['--list_cwd', list_file])
        self.assertEqual(4, len(cfg.list_cwd))
        self.assertEqual(['file1', 'file2', 'file3', 'file4'], [x(absolute=False) for x in cfg.list_cwd])
        os.chdir(cwd)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--list']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list', list_file4]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list', 'no-such-file']))

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionPathList))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', action=ActionPathList(mode='fr'), nargs='*'))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionPathList(mode='fr', rel='.')))

        shutil.rmtree(tmpdir)


    def test_ActionEnum(self):

        class MyEnum(enum.Enum):
            A = 1
            B = 2
            C = 3

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--enum',
            action=ActionEnum(enum=MyEnum),
            default=MyEnum.C,
            help='MyEnum')

        for val in ['A', 'B', 'C']:
            self.assertEqual(MyEnum[val], parser.parse_args(['--enum='+val]).enum)
        for val in ['X', 'b', 2]:
            self.assertRaises(ParserError, lambda: parser.parse_args(['--enum='+str(val)]))

        cfg = parser.parse_args(['--enum=C'], with_meta=False)
        self.assertEqual('enum: C\n', parser.dump(cfg))

        os.environ['COLUMNS'] = '150'
        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('MyEnum (default: C)', help_str.getvalue())

        def func(a1: MyEnum = MyEnum['A']):
            return a1

        parser = ArgumentParser()
        parser.add_function_arguments(func)
        self.assertEqual(MyEnum['A'], parser.get_defaults().a1)
        self.assertEqual(MyEnum['B'], parser.parse_args(['--a1=B']).a1)

        self.assertRaises(ValueError, lambda: ActionEnum())
        self.assertRaises(ValueError, lambda: ActionEnum(enum=object))


    def test_ActionOperators(self):
        parser = ArgumentParser(prog='app', error_handler=None)
        parser.add_argument('--le0',
            action=ActionOperators(expr=('<', 0)))
        parser.add_argument('--gt1.a.le4',
            action=ActionOperators(expr=[('>', 1.0), ('<=', 4.0)], join='and', type=float))
        parser.add_argument('--lt5.o.ge10.o.eq7',
            action=ActionOperators(expr=[('<', 5), ('>=', 10), ('==', 7)], join='or', type=int))
        def int_or_off(x): return x if x == 'off' else int(x)
        parser.add_argument('--ge0',
            nargs=3,
            action=ActionOperators(expr=('>=', 0)))

        self.assertEqual(1.5, parser.parse_args(['--gt1.a.le4', '1.5']).gt1.a.le4)
        self.assertEqual(4.0, parser.parse_args(['--gt1.a.le4', '4.0']).gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt1.a.le4', '1.0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt1.a.le4', '5.5']))

        self.assertEqual(1.5, parser.parse_string('gt1:\n  a:\n    le4: 1.5').gt1.a.le4)
        self.assertEqual(4.0, parser.parse_string('gt1:\n  a:\n    le4: 4.0').gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_string('gt1:\n  a:\n    le4: 1.0'))
        self.assertRaises(ParserError, lambda: parser.parse_string('gt1:\n  a:\n    le4: 5.5'))

        self.assertEqual(1.5, parser.parse_env({'APP_GT1__A__LE4': '1.5'}).gt1.a.le4)
        self.assertEqual(4.0, parser.parse_env({'APP_GT1__A__LE4': '4.0'}).gt1.a.le4)
        self.assertRaises(ParserError, lambda: parser.parse_env({'APP_GT1__A__LE4': '1.0'}))
        self.assertRaises(ParserError, lambda: parser.parse_env({'APP_GT1__A__LE4': '5.5'}))

        self.assertEqual(2, parser.parse_args(['--lt5.o.ge10.o.eq7', '2']).lt5.o.ge10.o.eq7)
        self.assertEqual(7, parser.parse_args(['--lt5.o.ge10.o.eq7', '7']).lt5.o.ge10.o.eq7)
        self.assertEqual(10, parser.parse_args(['--lt5.o.ge10.o.eq7', '10']).lt5.o.ge10.o.eq7)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--lt5.o.ge10.o.eq7', '5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--lt5.o.ge10.o.eq7', '8']))

        self.assertEqual([0, 1, 2], parser.parse_args(['--ge0', '0', '1', '2']).ge0)

        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionOperators))
        action = ActionOperators(expr=('<', 0))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op2', type=float, action=action))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', nargs=0, action=action))
        self.assertRaises(ValueError, lambda: ActionOperators())
        self.assertRaises(ValueError, lambda: ActionOperators(expr='<'))
        self.assertRaises(ValueError, lambda: ActionOperators(expr=[('<', 5), ('>=', 10)], join='xor'))


    def test_ActionParser(self):
        parser_lv3 = ArgumentParser(prog='lv3', default_env=False)
        parser_lv3.add_argument('--opt3',
            default='opt3_def')

        parser_lv2 = ArgumentParser(prog='lv2', default_env=False)
        parser_lv2.add_argument('--opt2',
            default='opt2_def')
        parser_lv2.add_argument('--inner3',
            action=ActionParser(parser=parser_lv3))

        parser = ArgumentParser(prog='lv1', default_env=True)
        parser.add_argument('--opt1',
            default='opt1_def')
        parser.add_argument('--inner2',
            action=ActionParser(parser=parser_lv2))

        tmpdir = tempfile.mkdtemp(prefix='_jsonargparse_test_')
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
        cfg = parser.parse_path(yaml_main_file, with_meta=False)
        with open(yaml_main_file, 'w') as output_file:
            output_file.write(parser.dump(cfg))
        cfg2 = parser.parse_path(yaml_main_file, with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg2))

        ## Check ActionParser inner environment variables
        self.assertEqual('opt2_env', parser.parse_env({'LV1_INNER2__OPT2': 'opt2_env'}).inner2.opt2)
        self.assertEqual('opt3_env', parser.parse_env({'LV1_INNER2__INNER3__OPT3': 'opt3_env'}).inner2.inner3.opt3)
        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_def', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_env({'LV1_INNER2__INNER3': yaml_inner3_file}, with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg))
        self.assertEqual('opt2_yaml', parser.parse_env({'LV1_INNER2': yaml_inner2_file}).inner2.opt2)

        ## Check ActionParser as argument path
        expected = {'opt1': 'opt1_arg', 'inner2': {'opt2': 'opt2_yaml', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--opt1', 'opt1_arg', '--inner2', yaml_inner2_file], with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg))

        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_arg', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--inner2.opt2', 'opt2_arg', '--inner2.inner3', yaml_inner3_file], with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg))

        expected = {'opt1': 'opt1_def', 'inner2': {'opt2': 'opt2_def', 'inner3': {'opt3': 'opt3_arg'}}}
        cfg = parser.parse_args(['--inner2.inner3', yaml_inner3_file, '--inner2.inner3.opt3', 'opt3_arg'], with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg))

        ## Check ActionParser as argument string
        expected = {'opt2': 'opt2_str', 'inner3': {'opt3': 'opt3_str'}}
        cfg = parser.parse_args(['--inner2', json.dumps(expected)], with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg)['inner2'])

        expected = {'opt3': 'opt3_str'}
        cfg = parser.parse_args(['--inner2.inner3', json.dumps(expected)], with_meta=False)
        self.assertEqual(expected, namespace_to_dict(cfg)['inner2']['inner3'])

        ## Check ActionParser with ActionConfigFile
        parser.add_argument('--cfg',
            action=ActionConfigFile)

        expected = {'opt1': 'opt1_yaml', 'inner2': {'opt2': 'opt2_yaml', 'inner3': {'opt3': 'opt3_yaml'}}}
        cfg = parser.parse_args(['--cfg', yaml_main_file], with_meta=False)
        delattr(cfg, 'cfg')
        self.assertEqual(expected, namespace_to_dict(cfg))

        cfg = parser.parse_args(['--cfg', yaml_main_file, '--inner2.opt2', 'opt2_arg', '--inner2.inner3.opt3', 'opt3_arg'])
        self.assertEqual('opt2_arg', cfg.inner2.opt2)
        self.assertEqual('opt3_arg', cfg.inner2.inner3.opt3)

        ## Check invalid initializations
        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', action=ActionParser))
        self.assertRaises(ValueError, lambda: ActionParser())
        self.assertRaises(ValueError, lambda: ActionParser(parser=object))

        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
