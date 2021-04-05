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


if __name__ == '__main__':
    unittest.main(verbosity=2)
