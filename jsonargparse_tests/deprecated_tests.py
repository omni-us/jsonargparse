#!/usr/bin/env python3

from enum import Enum
from io import StringIO
from jsonargparse_tests.base import *
from jsonargparse import ArgumentParser


class DeprecatedTests(unittest.TestCase):

    def test_ActionEnum(self):

        class MyEnum(Enum):
            A = 1
            B = 2
            C = 3

        parser = ArgumentParser(error_handler=None)
        action = ActionEnum(enum=MyEnum)
        parser.add_argument('--enum',
            action=action,
            default=MyEnum.C,
            help='Description')

        for val in ['A', 'B', 'C']:
            self.assertEqual(MyEnum[val], parser.parse_args(['--enum='+val]).enum)
        for val in ['X', 'b', 2]:
            self.assertRaises(ParserError, lambda: parser.parse_args(['--enum='+str(val)]))

        cfg = parser.parse_args(['--enum=C'], with_meta=False)
        self.assertEqual('enum: C\n', parser.dump(cfg))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('Description (type: MyEnum, default: C)', help_str.getvalue())

        def func(a1: MyEnum = MyEnum['A']):
            return a1

        parser = ArgumentParser()
        parser.add_function_arguments(func)
        self.assertEqual(MyEnum['A'], parser.get_defaults().a1)
        self.assertEqual(MyEnum['B'], parser.parse_args(['--a1=B']).a1)

        self.assertRaises(ValueError, lambda: ActionEnum())
        self.assertRaises(ValueError, lambda: ActionEnum(enum=object))
        self.assertRaises(ValueError, lambda: parser.add_argument('--bad1', type=MyEnum, action=True))
        self.assertRaises(ValueError, lambda: parser.add_argument('--bad2', type=float, action=action))


    def test_ActionOperators(self):
        parser = ArgumentParser(prog='app', error_handler=None)
        parser.add_argument('--le0',
            action=ActionOperators(expr=('<', 0)))
        parser.add_argument('--gt1.a.le4',
            action=ActionOperators(expr=[('>', 1.0), ('<=', 4.0)], join='and', type=float))
        parser.add_argument('--lt5.o.ge10.o.eq7',
            action=ActionOperators(expr=[('<', 5), ('>=', 10), ('==', 7)], join='or', type=int))
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


    @unittest.skipIf(not url_support, 'validators and requests packages are required')
    def test_url_support_true(self):
        self.assertEqual('fr', get_config_read_mode())
        set_url_support(True)
        self.assertEqual('fur', get_config_read_mode())
        set_url_support(False)
        self.assertEqual('fr', get_config_read_mode())


    @unittest.skipIf(url_support, 'validators and requests packages should not be installed')
    def test_url_support_false(self):
        self.assertEqual('fr', get_config_read_mode())
        with self.assertRaises(ImportError):
            set_url_support(True)
        self.assertEqual('fr', get_config_read_mode())
        set_url_support(False)
        self.assertEqual('fr', get_config_read_mode())


class DeprecatedTempDirTests(TempDirTestCase):

    def test_ActionPath(self):
        os.mkdir(os.path.join(self.tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(self.tmpdir, 'example', rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('file: '+rel_yaml_file+'\ndir: '+self.tmpdir+'\n')

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--file', action=ActionPath(mode='fr'))
        parser.add_argument('--dir', action=ActionPath(mode='drw'))
        parser.add_argument('--files', nargs='+', action=ActionPath(mode='fr'))

        cfg = parser.parse_args(['--cfg', abs_yaml_file])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[0](absolute=False)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.cfg[0](absolute=True)))
        self.assertEqual(rel_yaml_file, cfg.file(absolute=False))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', abs_yaml_file+'~']))

        cfg = parser.parse_args(['--cfg', 'file: '+abs_yaml_file+'\ndir: '+self.tmpdir+'\n'])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(None, cfg.cfg[0])
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', '{"k":"v"}']))

        cfg = parser.parse_args(['--file', abs_yaml_file, '--dir', self.tmpdir])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir(absolute=True)))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file(absolute=True)))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dir', abs_yaml_file]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--file', self.tmpdir]))

        cfg = parser.parse_args(['--files', abs_yaml_file, abs_yaml_file])
        self.assertTrue(isinstance(cfg.files, list))
        self.assertEqual(2, len(cfg.files))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.files[-1](absolute=True)))

        self.assertRaises(TypeError, lambda: parser.add_argument('--op1', action=ActionPath))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op3', action=ActionPath(mode='+')))
        self.assertRaises(ValueError, lambda: parser.add_argument('--op4', type=str, action=ActionPath(mode='fr')))


    def test_ActionPath_skip_check(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--file', action=ActionPath(mode='fr', skip_check=True))
        cfg = parser.parse_args(['--file=not-exist'])
        self.assertIsInstance(cfg.file, Path)
        self.assertEqual(str(cfg.file), 'not-exist')
        self.assertEqual(parser.dump(cfg), 'file: not-exist\n')
        self.assertTrue(repr(cfg.file).startswith('Path_fr_skip_check'))


    def test_ActionPath_dump(self):
        parser = ArgumentParser()
        parser.add_argument('--path', action=ActionPath(mode='fc'))
        cfg = parser.parse_string('path: path')
        self.assertEqual(parser.dump(cfg), 'path: path\n')

        parser = ArgumentParser()
        parser.add_argument('--paths', nargs='+', action=ActionPath(mode='fc'))
        cfg = parser.parse_args(['--paths', 'path1', 'path2'])
        self.assertEqual(parser.dump(cfg), 'paths:\n- path1\n- path2\n')


    def test_ActionPath_nargs_questionmark(self):
        parser = ArgumentParser()
        parser.add_argument('val', type=int)
        parser.add_argument('path', nargs='?', action=ActionPath(mode='fc'))
        self.assertIsNone(parser.parse_args(['1']).path)
        self.assertIsNotNone(parser.parse_args(['2', 'file']).path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
