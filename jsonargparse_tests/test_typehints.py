#!/usr/bin/env python3

import json
import os
import pathlib
import random
import re
import sys
import time
import unittest
import uuid
import warnings
import yaml
from calendar import Calendar, HTMLCalendar, TextCalendar
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from enum import Enum
from gzip import GzipFile
from io import StringIO
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from jsonargparse import ActionConfigFile, ArgumentParser, CLI, lazy_instance, Namespace, ParserError, Path
from jsonargparse.typehints import ActionTypeHint, is_optional, Literal
from jsonargparse.typing import (
    Email,
    final,
    NotEmptyStr,
    OpenUnitInterval,
    Path_drw,
    Path_fc,
    Path_fr,
    path_type,
    PositiveInt,
    register_type,
    restricted_number_type,
)
from jsonargparse_tests.base import mock_module, TempDirTestCase


class TypeHintsTests(unittest.TestCase):

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
        self.assertIsNone(parser.parse_args(['--op2=null']).op2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op2', 'abc']))


    def test_type_hint_action_failure(self):
        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ValueError, lambda: parser.add_argument('--op1', type=Optional[bool], action=True))


    def test_bool(self):
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


    def test_no_str_strip(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=Optional[str])
        parser.add_argument('--cfg', action=ActionConfigFile)
        self.assertEqual('  ', parser.parse_args(['--op', '  ']).op)
        self.assertEqual('', parser.parse_args(['--op', '']).op)
        self.assertEqual(' abc ', parser.parse_args(['--op= abc ']).op)
        self.assertEqual('xyz: ', parser.parse_args(['--op=xyz: ']).op)
        self.assertEqual(' ', parser.parse_args(['--cfg={"op":" "}']).op)
        self.assertIsNone(parser.parse_args(['--op=null']).op)


    def test_str_not_timestamp_issue_135(self):
        parser = ArgumentParser()
        parser.add_argument('foo', type=str)
        self.assertEqual('2022-04-12', parser.parse_args(['2022-04-12']).foo)
        self.assertEqual('2022-04-32', parser.parse_args(['2022-04-32']).foo)


    def test_float_scientific_notation(self):
        parser = ArgumentParser()
        parser.add_argument('--num', type=float)
        self.assertEqual(1e-3, parser.parse_args(['--num=1e-3']).num)


    def test_str_with_number_value(self):
        class Class:
            def __init__(self, val: str = '-'):
                pass

        with mock_module(Class):
            parser = ArgumentParser()
            parser.add_argument('--val', type=str)
            parser.add_argument('--cls', type=Class, default=lazy_instance(Class))

            for value in ['1', '02', '3.40', '5.7e-8']:
                with self.subTest(value):
                    self.assertEqual(value, parser.parse_args([f'--val={value}']).val)
                    self.assertEqual(value, parser.parse_args([f'--cls.val={value}']).cls.init_args.val)


    def test_list(self):
        for list_type in [Iterable, List, Sequence]:
            with self.subTest(str(list_type)):
                parser = ArgumentParser()
                parser.add_argument('--list', type=list_type[int])
                cfg = parser.parse_args(['--list=[1, 2]'])
                self.assertEqual([1, 2], cfg.list)


    def test_enum(self):
        class MyEnum(Enum):
            A = 1
            B = 2
            C = 3

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--enum', type=MyEnum, default=MyEnum.C, help='Description')

        for val in ['A', 'B', 'C']:
            self.assertEqual(MyEnum[val], parser.parse_args(['--enum='+val]).enum)
        for val in ['X', 'b', 2]:
            self.assertRaises(ParserError, lambda: parser.parse_args(['--enum='+str(val)]))

        cfg = parser.parse_args(['--enum=C'], with_meta=False)
        self.assertEqual('enum: C\n', parser.dump(cfg))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('Description (type: MyEnum, default: C)', help_str.getvalue())


    def test_list_enum(self):
        class MyEnum(Enum):
            ab = 0
            xy = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list', type=List[MyEnum])
        self.assertEqual([MyEnum.xy, MyEnum.ab], parser.parse_args(['--list=["xy", "ab"]']).list)


    def test_list_union(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list1', type=List[Union[float, str, type(None)]])
        parser.add_argument('--list2', type=List[Union[int, MyEnum]])
        self.assertEqual([1.2, 'ab'], parser.parse_args(['--list1=[1.2, "ab"]']).list1)
        self.assertEqual([3, MyEnum.ab], parser.parse_args(['--list2=[3, "ab"]']).list2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--list1={"a":1, "b":"2"}']))


    def test_dict(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--dict', type=dict)
        self.assertEqual({}, parser.parse_args(['--dict={}'])['dict'])
        self.assertEqual({'a': 1, 'b': '2'}, parser.parse_args(['--dict={"a":1, "b":"2"}'])['dict'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dict=1']))


    def test_dict_items(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--dict', type=Dict[str, int])
        cfg = parser.parse_args(['--dict.one=1', '--dict.two=2'])
        self.assertEqual(cfg.dict, {'one': 1, 'two': 2})


    def test_dict_union(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--dict1', type=Dict[int, Optional[Union[float, MyEnum]]])
        parser.add_argument('--dict2', type=Dict[str, Union[bool, Path_fc]])
        cfg = parser.parse_args(['--dict1={"2":4.5, "6":"ab"}', '--dict2={"a":true, "b":"f"}'])
        self.assertEqual({2: 4.5, 6: MyEnum.ab}, cfg['dict1'])
        self.assertEqual({'a': True, 'b': 'f'}, cfg['dict2'])
        self.assertIsInstance(cfg['dict2']['b'], Path)
        self.assertEqual({5: None}, parser.parse_args(['--dict1={"5":null}'])['dict1'])
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dict1=["a", "b"]']))
        cfg = yaml.safe_load(parser.dump(cfg))
        self.assertEqual({'dict1': {'2': 4.5, '6': 'ab'}, 'dict2': {'a': True, 'b': 'f'}}, cfg)


    def test_set(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--set', type=Set[int])
        self.assertEqual({1, 2}, parser.parse_args(['--set=[1, 2]']).set)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--set=["a", "b"]']))


    def test_tuple(self):
        class MyEnum(Enum):
            ab = 1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Union[int, MyEnum], Path_fc, NotEmptyStr])
        cfg = parser.parse_args(['--tuple=[2, "a", "b"]'])
        self.assertEqual((2, 'a', 'b'), cfg.tuple)
        self.assertIsInstance(cfg.tuple[1], Path)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a", "b", 5]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a"]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple={"a":1, "b":"2"}']))
        out = StringIO()
        parser.print_help(out)
        self.assertIn('--tuple [ITEM,...]  (type: Tuple[Union[int, MyEnum], Path_fc, NotEmptyStr], default: null)', out.getvalue())


    def test_tuple_untyped(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=tuple)
        cfg = parser.parse_args(['--tuple=[1, "a", True]'])
        self.assertEqual((1, 'a', True), cfg.tuple)
        out = StringIO()
        parser.print_help(out)
        self.assertIn('--tuple [ITEM,...]  (type: tuple, default: null)', out.getvalue())


    def test_nested_tuples(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], Tuple[int, float]]])
        cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
        self.assertEqual((('foo', 'bar'), ((1, 2.02), (3, 3.09))), cfg.tuple)


    def test_list_tuple(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--list', type=List[Tuple[int, float]])
        cfg = parser.parse_args(['--list=[[1, 2.02], [3, 3.09]]'])
        self.assertEqual([(1, 2.02), (3, 3.09)], cfg.list)


    def test_list_str_positional(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('list', type=List[str])
        cfg = parser.parse_args(['["a", "b"]'])
        self.assertEqual(cfg.list, ['a', 'b'])


    def test_tuple_ellipsis(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[float, ...])
        self.assertEqual((1.2,), parser.parse_args(['--tuple=[1.2]']).tuple)
        self.assertEqual((1.2, 3.4), parser.parse_args(['--tuple=[1.2, 3.4]']).tuple)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[]']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--tuple=[2, "a"]']))

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--tuple', type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], ...]])
        cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
        self.assertEqual((('foo', 'bar'), ((1, 2.02), (3, 3.09))), cfg.tuple)


    def test_complex_number(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--complex', type=complex)
        cfg = parser.parse_args(['--complex=(2+3j)'])
        self.assertEqual(cfg.complex, 2+3j)
        self.assertEqual(parser.dump(cfg), 'complex: (2+3j)\n')


    def test_list_append(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', type=Union[int, str, List[int]])
        self.assertEqual(0, parser.parse_args(['--val=0']).val)
        self.assertEqual([0], parser.parse_args(['--val+=0']).val)
        self.assertEqual([1, 2, 3], parser.parse_args(['--val=1', '--val+=2', '--val+=3']).val)
        self.assertEqual([1, 2, 3], parser.parse_args(['--val=[1,2]', '--val+=3']).val)
        self.assertEqual([1], parser.parse_args(['--val=a', '--val+=1']).val)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(3, parser.parse_args(['--val=[1,2]', '--val=3']).val)
            self.assertIn('Replacing list value "[1, 2]" with "3"', str(w[0].message))


    def test_list_append_config(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--val', type=List[int], default=[1, 2])
        self.assertEqual([3, 4], parser.parse_args(['--cfg', 'val: [3, 4]']).val)
        self.assertEqual([1, 2, 3], parser.parse_args(['--cfg', 'val+: 3']).val)
        self.assertEqual([1, 2, 3, 4], parser.parse_args(['--cfg', 'val+: [3, 4]']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cfg', 'val+: a']))


    def test_list_append_subclass_init_args(self):
        class Class:
            def __init__(self, p1: int = 0, p2: int = 0):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', type=Union[Class, List[Class]])

        with mock_module(Class) as module:
            cfg = parser.parse_args([f'--val+={module}.Class', '--val.p1=1', '--val.p2=2', '--val.p1=3'])
            self.assertEqual(cfg.val, [Namespace(class_path=f'{module}.Class', init_args=Namespace(p1=3, p2=2))])
            cfg = parser.parse_args([f'--val+=Class', '--val.p2=2', '--val.p1=1'])
            self.assertEqual(cfg.val, [Namespace(class_path=f'{module}.Class', init_args=Namespace(p1=1, p2=2))])


    def test_list_append_subclass_nonclass_default(self):
        @final
        class Class:
            def __init__(self, cal: Union[Calendar, Iterable[Calendar], bool] = True):
                self.cal = cal

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('cls', type=Class)

        cfg = parser.parse_args(['--cls.cal=calendar.TextCalendar', '--cls.cal.firstweekday=2'])
        self.assertNotIsInstance(cfg.cls.cal, list)
        self.assertEqual('calendar.TextCalendar', cfg.cls.cal.class_path)
        self.assertEqual(2, cfg.cls.cal.init_args.firstweekday)

        cfg = parser.parse_args(['--cls.cal=calendar.TextCalendar', '--cls.cal+=calendar.HTMLCalendar'])
        self.assertIsInstance(cfg.cls.cal, list)
        self.assertEqual(['TextCalendar', 'HTMLCalendar'], [c.class_path.split('.')[1] for c in cfg.cls.cal])


    def test_list_append_subcommand_subclass(self):
        class A:
            def __init__(self, cals: Union[Calendar, List[Calendar]] = None):
                self.cals = cals

        parser = ArgumentParser(error_handler=None)
        subparser = ArgumentParser()
        subparser.add_class_arguments(A, 'a')
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('cmd', subparser)
        cfg = parser.parse_args([
            'cmd',
            '--a.cals+=Calendar',
            '--a.cals.firstweekday=3',
            '--a.cals+=TextCalendar',
            '--a.cals.firstweekday=1',
        ])
        self.assertEqual(['calendar.Calendar', 'calendar.TextCalendar'], [x.class_path for x in cfg.cmd.a.cals])
        self.assertEqual([3, 1], [x.init_args.firstweekday for x in cfg.cmd.a.cals])
        cfg = parser.parse_args(['cmd', f'--a={json.dumps(cfg.cmd.a.as_dict())}', '--a.cals.firstweekday=4'])
        self.assertEqual(Namespace(firstweekday=4), cfg.cmd.a.cals[-1].init_args)


    def test_restricted_number_type(self):
        limit_val = random.randint(100, 10000)
        larger_than = restricted_number_type(f'larger_than_{limit_val}', int, ('>', limit_val))

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--val', type=larger_than, default=limit_val+1, help='Description')

        self.assertEqual(limit_val+1, parser.parse_args([f'--val={limit_val+1}']).val)
        self.assertRaises(ParserError, lambda: parser.parse_args([f'--val={limit_val-1}']))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn(f'Description (type: larger_than_{limit_val}, default: {limit_val+1})', help_str.getvalue())


    def test_type_Any(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--any', type=Any)
        self.assertEqual('abc', parser.parse_args(['--any=abc'])['any'])
        self.assertEqual(123, parser.parse_args(['--any=123'])['any'])
        self.assertEqual(5.6, parser.parse_args(['--any=5.6'])['any'])
        self.assertEqual([7, 8], parser.parse_args(['--any=[7, 8]'])['any'])
        self.assertEqual({"a":0, "b":1}, parser.parse_args(['--any={"a":0, "b":1}'])['any'])
        self.assertTrue(parser.parse_args(['--any=True'])['any'])
        self.assertFalse(parser.parse_args(['--any=False'])['any'])
        self.assertIsNone(parser.parse_args(['--any=null'])['any'])
        self.assertEqual(' ', parser.parse_args(['--any= '])['any'])
        self.assertEqual(' xyz ', parser.parse_args(['--any= xyz '])['any'])
        self.assertEqual('[[[', parser.parse_args(['--any=[[['])['any'])


    def test_type_any_subclasses(self):
        class Class:
            def __init__(self, cal1: Calendar, cal2: Any):
                self.cal1 = cal1
                self.cal2 = cal2

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--any', type=Any)

        with mock_module(Class) as module:
            value = {
                'class_path': f'{module}.Class',
                'init_args': {
                    'cal1': {
                        'class_path': 'calendar.TextCalendar',
                        'init_args': {'firstweekday': 1},
                    },
                    'cal2': {
                        'class_path': 'calendar.HTMLCalendar',
                        'init_args': {'firstweekday': 2},
                    },
                },
            }

            cfg = parser.parse_args([f'--any={value}'])
            init = parser.instantiate_classes(cfg)
            self.assertIsInstance(init.any, Class)
            self.assertIsInstance(init.any.cal1, TextCalendar)
            self.assertIsInstance(init.any.cal2, HTMLCalendar)
            self.assertEqual(init.any.cal1.firstweekday, 1)
            self.assertEqual(init.any.cal2.firstweekday, 2)

            value['init_args']['cal2']['class_path'] = 'does.not.exist'
            cfg = parser.parse_args([f'--any={value}'])
            self.assertIsInstance(cfg.any.init_args.cal1, Namespace)
            self.assertIsInstance(cfg.any.init_args.cal2, dict)
            init = parser.instantiate_classes(cfg)
            self.assertIsInstance(init.any, Class)
            self.assertIsInstance(init.any.cal1, TextCalendar)
            self.assertIsInstance(init.any.cal2, dict)
            self.assertEqual(init.any.cal1.firstweekday, 1)
            self.assertEqual(init.any.cal2['init_args']['firstweekday'], 2)

            value['init_args']['cal1']['class_path'] = 'does.not.exist'
            cfg = parser.parse_args([f'--any={value}'])
            self.assertIsInstance(cfg.any, dict)


    def test_type_any_list_of_subclasses(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--any', type=Any)

        value = [
            {
                'class_path': 'calendar.TextCalendar',
                'init_args': {'firstweekday': 1},
            },
            {
                'class_path': 'calendar.HTMLCalendar',
                'init_args': {'firstweekday': 2},
            },
        ]

        cfg = parser.parse_args([f'--any={value}'])
        init = parser.instantiate_classes(cfg)
        self.assertIsInstance(init.any, list)
        self.assertEqual(len(init.any), 2)
        self.assertIsInstance(init.any[0], TextCalendar)
        self.assertIsInstance(init.any[1], HTMLCalendar)
        self.assertEqual(init.any[0].firstweekday, 1)
        self.assertEqual(init.any[1].firstweekday, 2)


    def test_type_any_dict_of_subclasses(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--any', type=Any)

        value = {
            'k1': {
                'class_path': 'calendar.TextCalendar',
                'init_args': {'firstweekday': 1},
            },
            'k2': {
                'class_path': 'calendar.HTMLCalendar',
                'init_args': {'firstweekday': 2},
            },
        }

        cfg = parser.parse_args([f'--any={value}'])
        init = parser.instantiate_classes(cfg)
        self.assertIsInstance(init.any, dict)
        self.assertEqual(len(init.any), 2)
        self.assertIsInstance(init.any['k1'], TextCalendar)
        self.assertIsInstance(init.any['k2'], HTMLCalendar)
        self.assertEqual(init.any['k1'].firstweekday, 1)
        self.assertEqual(init.any['k2'].firstweekday, 2)


    def test_union_subtypes_order(self):
        for subtypes, arg, expected in [
            ((bool, str), '=true', True),
            ((str, bool), '=true', 'true'),
            ((int, str), '=1', 1),
            ((str, int), '=2', '2'),
            ((float, int), '=3', 3.0),
            ((int, float), '=4', 4),
            ((int, List[int]), '=5', 5),
            ((List[int], int), '=6', 6),
            ((int, List[int]), '+=7', [7]),
            ((List[int], int), '+=8', [8]),
        ]:
            with self.subTest(f'{subtypes}, {arg}, {expected}'):
                parser = ArgumentParser()
                parser.add_argument('--val', type=Union[subtypes])
                val = parser.parse_args([f'--val{arg}']).val
                self.assertIsInstance(val, type(expected))
                self.assertEqual(val, expected)


    @unittest.skipIf(not Literal, 'Literal introduced in python 3.8 or backported in typing_extensions')
    def test_Literal(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--str', type=Literal['a', 'b', None])
        parser.add_argument('--int', type=Literal[3, 4])
        parser.add_argument('--true', type=Literal[True])
        parser.add_argument('--false', type=Literal[False])
        self.assertEqual('a', parser.parse_args(['--str=a']).str)
        self.assertEqual('b', parser.parse_args(['--str=b']).str)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--str=x']))
        self.assertIsNone(parser.parse_args(['--str=null']).str)
        self.assertEqual(4, parser.parse_args(['--int=4']).int)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--int=5']))
        self.assertIs(True, parser.parse_args(['--true=true']).true)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--true=false']))
        self.assertIs(False, parser.parse_args(['--false=false']).false)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--false=true']))
        out = StringIO()
        parser.print_help(out)
        for value in ['--str {a,b,null}', '--int {3,4}', '--true True', '--false False']:
            self.assertIn(value, out.getvalue())


    def test_nested_mapping_without_args(self):
        parser = ArgumentParser()
        parser.add_argument('--map', type=Mapping[str, Union[int, Mapping]])
        self.assertEqual(parser.parse_args(['--map={"a": 1}']).map, {"a": 1})
        self.assertEqual(parser.parse_args(['--map={"b": {"c": 2}}']).map, {"b": {"c": 2}})


    def _test_typehint_non_parameterized_types(self, type):
        parser = ArgumentParser(error_handler=None)
        ActionTypeHint.is_supported_typehint(type, full=True)
        parser.add_argument('--type', type=type)
        cfg = parser.parse_args(['--type=uuid.UUID'])
        self.assertEqual(cfg.type, uuid.UUID)
        self.assertEqual(parser.dump(cfg), 'type: uuid.UUID\n')


    def _test_typehint_parameterized_types(self, type):
        parser = ArgumentParser(error_handler=None)
        ActionTypeHint.is_supported_typehint(type, full=True)
        parser.add_argument('--cal', type=type[Calendar])
        cfg = parser.parse_args(['--cal=calendar.Calendar'])
        self.assertEqual(cfg.cal, Calendar)
        self.assertEqual(parser.dump(cfg), 'cal: calendar.Calendar\n')
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal=uuid.UUID']))


    def test_typehint_Type(self):
        self._test_typehint_non_parameterized_types(type=Type)
        self._test_typehint_parameterized_types(type=Type)


    def test_typehint_non_parameterized_type(self):
        self._test_typehint_non_parameterized_types(type=type)


    @unittest.skipIf(sys.version_info[:2] < (3, 9), '[] support for builtins introduced in python 3.9')
    def test_typehint_parametrized_type(self):
        self._test_typehint_parameterized_types(type=type)


    def test_uuid(self):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--uuid', type=uuid.UUID)
        parser.add_argument('--uuids', type=List[uuid.UUID])
        cfg = parser.parse_args(['--uuid='+str(id1), '--uuids=["'+str(id1)+'", "'+str(id2)+'"]'])
        self.assertEqual(cfg.uuid, id1)
        self.assertEqual(cfg.uuids, [id1, id2])
        self.assertEqual('uuid: '+str(id1)+'\nuuids:\n- '+str(id1)+'\n- '+str(id2)+'\n', parser.dump(cfg))


    @unittest.skipIf(sys.version_info[:2] < (3, 10), 'new union syntax introduced in python 3.10')
    def test_union_new_syntax(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--str', type=eval('int | None'))
        self.assertEqual(123, parser.parse_args(['--str=123']).str)
        self.assertIsNone(parser.parse_args(['--str=null']).str)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--str=abc']))


    def test_Callable_with_function_path(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--callable', type=Callable, default=time.time)
        parser.add_argument('--list', type=List[Callable])

        cfg = parser.get_defaults()
        self.assertEqual(time.time, cfg.callable)
        self.assertEqual(parser.dump(cfg), 'callable: time.time\n')
        cfg = parser.parse_args(['--callable=random.randint'])
        self.assertEqual(random.randint, cfg.callable)
        cfg = parser.parse_args(['--callable=jsonargparse.CLI'])
        self.assertEqual(CLI, cfg.callable)
        self.assertEqual(parser.dump(cfg), 'callable: jsonargparse.CLI\n')
        self.assertEqual([CLI], parser.parse_args(['--list=[jsonargparse.CLI]']).list)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--callable=jsonargparse.not_exist']))

        out = StringIO()
        parser.print_help(out)
        self.assertIn('(type: Callable, default: time.time)', out.getvalue())


    def test_Callable_with_class_path(self):
        class MyFunc1:
            def __init__(self, p1: int = 1):
                self.p1 = p1
            def __call__(self):
                return self.p1

        class MyFunc2(MyFunc1):
            pass

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--callable', type=Callable)

        with mock_module(MyFunc1, MyFunc2) as module:
            value = {'class_path': f'{module}.MyFunc2', 'init_args': {'p1': 1}}
            cfg = parser.parse_args([f'--callable={module}.MyFunc2'])
            self.assertEqual(cfg.callable.as_dict(), value)
            value = {'class_path': f'{module}.MyFunc1', 'init_args': {'p1': 2}}
            cfg = parser.parse_args([f'--callable={json.dumps(value)}'])
            self.assertEqual(cfg.callable.as_dict(), value)
            self.assertEqual(yaml.safe_load(parser.dump(cfg))['callable'], value)
            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.callable, MyFunc1)
            self.assertEqual(cfg_init.callable(), 2)

            self.assertRaises(ParserError, lambda: parser.parse_args(['--callable={}']))
            self.assertRaises(ParserError, lambda: parser.parse_args(['--callable=jsonargparse.SUPPRESS']))
            self.assertRaises(ParserError, lambda: parser.parse_args(['--callable=calendar.Calendar']))
            value = {'class_path': f'{module}.MyFunc1', 'key': 'val'}
            self.assertRaises(ParserError, lambda: parser.parse_args([f'--callable={json.dumps(value)}']))


    def test_callable_with_class_path_short_init_args(self):
        class MyCallable:
            def __init__(self, name: str):
                self.name = name
            def __call__(self):
                return self.name

        parser = ArgumentParser()
        parser.add_argument('--call', type=Callable)

        with mock_module(MyCallable) as module:
            cfg = parser.parse_args([f'--call={module}.MyCallable', '--call.name=Bob'])
            self.assertEqual(cfg.call.class_path, f'{module}.MyCallable')
            self.assertEqual(cfg.call.init_args, Namespace(name='Bob'))
            init = parser.instantiate_classes(cfg)
            self.assertEqual(init.call(), 'Bob')


    def test_union_callable_with_class_path_short_init_args(self):
        class MyCallable:
            def __init__(self, name: str):
                self.name = name
            def __call__(self):
                return self.name

        parser = ArgumentParser()
        parser.add_argument('--call', type=Union[Callable, None])

        with mock_module(MyCallable) as module:
            cfg = parser.parse_args([f'--call={module}.MyCallable', '--call.name=Bob'])
            self.assertEqual(cfg.call.class_path, f'{module}.MyCallable')
            self.assertEqual(cfg.call.init_args, Namespace(name='Bob'))
            init = parser.instantiate_classes(cfg)
            self.assertEqual(init.call(), 'Bob')


    def test_typed_Callable_with_function_path(self):
        def my_func_1(p: int) -> str:
            return str(p)

        def my_func_2(p: str) -> int:
            return int(p)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--callable', type=Callable[[int], str])

        with mock_module(my_func_1, my_func_2) as module:
            cfg = parser.parse_args([f'--callable={module}.my_func_1'])
            self.assertEqual(my_func_1, cfg.callable)
            cfg = parser.parse_args([f'--callable={module}.my_func_2'])
            self.assertEqual(my_func_2, cfg.callable)  # Currently callable types are ignored


    def test_typed_callable_with_return_type_class(self):
        class Optimizer:
            def __init__(self, params: List[float], lr: float = 1e-3):
                self.params = params
                self.lr = lr

        class SGD(Optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class Adam(Optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        value = {
            'class_path': 'Adam',
            'init_args': {
                'lr': 0.01,
            }
        }

        with mock_module(Optimizer, SGD, Adam) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--optimizer', type=Callable[[List[float]], Optimizer], default=SGD)

            cfg = parser.get_defaults()
            init = parser.instantiate_classes(cfg)
            optim = init.optimizer([0.1, 2,3])
            self.assertIsInstance(optim, SGD)
            self.assertEqual(optim.params, [0.1, 2,3])
            self.assertEqual(optim.lr, 1e-3)

            cfg = parser.parse_args(['--optimizer', str(value)])
            self.assertEqual(cfg.optimizer.class_path, f'{module}.Adam')
            self.assertEqual(cfg.optimizer.init_args, Namespace(lr=0.01))
            init = parser.instantiate_classes(cfg)
            optim = init.optimizer([4.5, 6.7])
            self.assertIsInstance(optim, Adam)
            self.assertEqual(optim.params, [4.5, 6.7])
            self.assertEqual(optim.lr, 0.01)

            help_str = StringIO()
            parser.print_help(help_str)
            for name in ['Optimizer', 'SGD', 'Adam']:
                self.assertIn(f'{module}.{name}', help_str.getvalue())


    def test_typed_callable_with_return_type_union_of_classes(self):
        class Optimizer:
            pass

        class StepLR:
            def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
                self.optimizer = optimizer
                self.last_epoch = last_epoch

        class ReduceLROnPlateau:
            def __init__(self, optimizer: Optimizer, monitor: str):
                self.optimizer = optimizer
                self.monitor = monitor

        optim = Optimizer()
        value = {
            'class_path': 'ReduceLROnPlateau',
            'init_args': {
                'monitor': 'loss',
            }
        }

        with mock_module(StepLR, ReduceLROnPlateau) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--scheduler', type=Callable[[Optimizer], Union[StepLR, ReduceLROnPlateau]], default=StepLR)

            cfg = parser.get_defaults()
            init = parser.instantiate_classes(cfg)
            sched = init.scheduler(optim)
            self.assertIsInstance(sched, StepLR)
            self.assertIs(sched.optimizer, optim)
            self.assertEqual(sched.last_epoch, -1)

            cfg = parser.parse_args(['--scheduler', str(value)])
            self.assertEqual(cfg.scheduler.class_path, f'{module}.ReduceLROnPlateau')
            self.assertEqual(cfg.scheduler.init_args, Namespace(monitor='loss'))
            init = parser.instantiate_classes(cfg)
            sched = init.scheduler(optim)
            self.assertIsInstance(sched, ReduceLROnPlateau)
            self.assertIs(sched.optimizer, optim)
            self.assertEqual(sched.monitor, 'loss')

            help_str = StringIO()
            parser.print_help(help_str)
            for name in ['StepLR', 'ReduceLROnPlateau']:
                self.assertIn(f'{module}.{name}', help_str.getvalue())


    def test_class_type(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=Optional[List[Calendar]])

        class_path = '"class_path": "calendar.Calendar"'
        expected = [{'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 0}}]
        cfg = parser.parse_args(['--op=[{'+class_path+'}]'])
        self.assertEqual(cfg.as_dict()['op'], expected)
        cfg = parser.parse_args(['--op=["calendar.Calendar"]'])
        self.assertEqual(cfg.as_dict()['op'], expected)
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'][0], Calendar)

        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse.ArgumentParser"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse.NotExist"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{"class_path": "jsonargparse0.IncorrectModule"}]'])
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[1]'])

        init_args = '"init_args": {"bad_arg": True}'
        with self.assertRaises(ParserError):
            parser.parse_args(['--op=[{'+class_path+', '+init_args+'}]'])

        init_args = '"init_args": {"firstweekday": 3}'
        cfg = parser.parse_args(['--op=[{'+class_path+', '+init_args+'}]'])
        self.assertEqual(cfg['op'][0]['init_args'].as_dict(), {'firstweekday': 3})
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'][0], Calendar)
        self.assertEqual(3, cfg['op'][0].firstweekday)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--n.op', type=Optional[Calendar])
        cfg = parser.parse_args(['--n.op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['n']['op'], Calendar)
        self.assertEqual(3, cfg['n']['op'].firstweekday)

        parser = ArgumentParser()
        parser.add_argument('--op', type=Calendar)
        cfg = parser.parse_args(['--op={'+class_path+', '+init_args+'}'])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['op'], Calendar)
        self.assertEqual(3, cfg['op'].firstweekday)

        cfg = parser.instantiate_classes(parser.parse_args([]))
        self.assertIsNone(cfg['op'])


    def test_class_type_without_defaults(self):
        class MyCal(Calendar):
            def __init__(self, p1: int = 1, p2: str = '2'):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=MyCal)

        with mock_module(MyCal) as module:
            cfg = parser.parse_args([f'--op.class_path={module}.MyCal', '--op.init_args.p1=3'], defaults=False)
            self.assertEqual(cfg.op, Namespace(class_path=f'{module}.MyCal', init_args=Namespace(p1=3)))
            cfg = parser.parse_args(['--op.class_path', f'{module}.MyCal', '--op.init_args.p1', '3'], defaults=False)
            self.assertEqual(cfg.op, Namespace(class_path=f'{module}.MyCal', init_args=Namespace(p1=3)))


    def test_class_type_required_params(self):
        class MyCal(Calendar):
            def __init__(self, p1: int, p2: str):
                pass

        with mock_module(MyCal) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--op', type=MyCal, default=lazy_instance(MyCal))

            cfg = parser.get_defaults()
            self.assertEqual(cfg.op.class_path, f'{module}.MyCal')
            self.assertEqual(cfg.op.init_args, Namespace(p1=None, p2=None))
            self.assertRaises(ParserError, lambda: parser.parse_args([f'--op={module}.MyCal']))


    def test_class_type_subclass_given_by_name_issue_84(self):
        class LocalCalendar(Calendar):
            pass

        parser = ArgumentParser()
        parser.add_argument('--op', type=Union[Calendar, GzipFile, None])
        cfg = parser.parse_args(['--op=TextCalendar'])
        self.assertEqual(cfg.op.class_path, 'calendar.TextCalendar')

        out = StringIO()
        parser.print_help(out)
        for class_path in ['calendar.Calendar', 'calendar.TextCalendar', 'gzip.GzipFile']:
            self.assertIn(class_path, out.getvalue())
        self.assertNotIn('LocalCalendar', out.getvalue())

        class HTMLCalendar(Calendar):
            pass

        with mock_module(HTMLCalendar) as module:
            err = StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit):
                parser.parse_args(['--op.help=HTMLCalendar'])
            self.assertIn('Give the full class path to avoid ambiguity', err.getvalue())
            self.assertIn(f'{module}.HTMLCalendar', err.getvalue())


    def test_class_type_set_defaults_class_name(self):
        parser = ArgumentParser()
        parser.add_argument('--cal', type=Calendar)
        parser.set_defaults({
            'cal': {
                'class_path': 'TextCalendar',
                'init_args': {
                    'firstweekday': 1,
                }
            }
        })
        cal = parser.get_default('cal').as_dict()
        self.assertEqual(cal, {'class_path': 'calendar.TextCalendar', 'init_args': {'firstweekday': 1}})


    def test_class_type_subclass_short_init_args(self):
        parser = ArgumentParser()
        parser.add_argument('--op', type=Calendar)
        cfg = parser.parse_args(['--op=TextCalendar', '--op.firstweekday=2'])
        self.assertEqual(cfg.op.class_path, 'calendar.TextCalendar')
        self.assertEqual(cfg.op.init_args, Namespace(firstweekday=2))


    def test_class_type_invalid_class_name_then_init_args(self):
        parser = ArgumentParser()
        parser.add_argument('--cal', type=Calendar)
        err = StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            parser.parse_args(['--cal=NotCalendarSubclass', '--cal.firstweekday=2'])
        #self.assertIn('NotCalendarSubclass', err.getvalue())  # Need new way to show NotCalendarSubclass


    def test_class_type_config_merge_init_args(self):
        class MyCal(Calendar):
            def __init__(self, param_a: int = 1, param_b: str = 'x', **kwargs):
                super().__init__(**kwargs)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--cal', type=Calendar)

        with mock_module(MyCal) as module:
            config1 = {
                'cal': {
                    'class_path': f'{module}.MyCal',
                    'init_args': {
                        'firstweekday': 2,
                        'param_b': 'y',
                    }
                }
            }
            config2 = deepcopy(config1)
            config2['cal']['init_args'] = {
                'param_a': 2,
                'firstweekday': 3,
            }
            expected = deepcopy(config1['cal'])
            expected['init_args'].update(config2['cal']['init_args'])

            cfg = parser.parse_args([f'--cfg={yaml.safe_dump(config1)}', f'--cfg={yaml.safe_dump(config2)}'])
            self.assertEqual(cfg.cal.as_dict(), expected)


    def test_init_args_without_class_path(self):
        parser = ArgumentParser()
        parser.add_argument('--config', action=ActionConfigFile)
        parser.add_argument('--cal', type=Calendar)

        config = """cal:
          class_path: TextCalendar
          init_args:
            firstweekday: 2
        """
        cal = """init_args:
            firstweekday: 3
        """

        cfg = parser.parse_args([f'--config={config}', f'--cal={cal}'])
        self.assertEqual(cfg.cal.init_args, Namespace(firstweekday=3))

        cfg = parser.parse_args([f'--config={config}', f'--cal={cfg.cal.init_args.as_dict()}'])
        self.assertEqual(cfg.cal.init_args, Namespace(firstweekday=3))


    def test_class_type_subclass_nested_init_args(self):
        class Class:
            def __init__(self, cal: Calendar, p1: int = 0):
                self.cal = cal

        for full in ['init_args.', '']:
            with self.subTest('full' if full else 'short'), mock_module(Class) as module:
                parser = ArgumentParser()
                parser.add_argument('--op', type=Class)
                cfg = parser.parse_args([
                    f'--op={module}.Class',
                    f'--op.{full}p1=1',
                    f'--op.{full}cal=calendar.TextCalendar',
                    f'--op.{full}cal.{full}firstweekday=2',
                ])
                self.assertEqual(cfg.op.class_path, f'{module}.Class')
                self.assertEqual(cfg.op.init_args.p1, 1)
                self.assertEqual(cfg.op.init_args.cal.class_path, 'calendar.TextCalendar')
                self.assertEqual(cfg.op.init_args.cal.init_args, Namespace(firstweekday=2))


    def test_class_type_in_union_with_str(self):
        parser = ArgumentParser()
        parser.add_argument('--op', type=Optional[Union[str, Calendar]])
        cfg = parser.parse_args(['--op=value'])
        self.assertEqual(cfg.op, 'value')
        cfg = parser.parse_args([
            '--op=TextCalendar',
            '--op.firstweekday=1',
            '--op.firstweekday=2',
        ])
        self.assertEqual(cfg.op, Namespace(class_path='calendar.TextCalendar', init_args=Namespace(firstweekday=2)))


    def test_class_type_dict_default_nested_init_args(self):
        class Data:
            def __init__(self, p1: int = 1, p2: str = 'x', p3: bool = False):
                pass

        with mock_module(Data) as module:
            parser = ArgumentParser()
            parser.add_argument('--data', type=Data)
            parser.set_defaults({'data': {'class_path': f'{module}.Data'}})
            cfg = parser.parse_args([
                f'--data.init_args.p1=2',
                f'--data.init_args.p2=y',
                f'--data.init_args.p3=true',
            ])
            self.assertEqual(cfg.data.init_args, Namespace(p1=2, p2='y', p3=True))


    def test_class_type_subclass_in_union_help(self):
        parser = ArgumentParser()
        parser.add_argument('--op', type=Union[str, Mapping[str, int], Calendar])

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--help'])
        self.assertIn('Show the help for the given subclass of Calendar', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args([f'--op.help=TextCalendar'])
        self.assertIn('--op.init_args.firstweekday', out.getvalue())


    def test_class_type_subclass_nested_help(self):
        class Class:
            def __init__(self, cal: Calendar, p1: int = 0):
                self.cal = cal

        parser = ArgumentParser()
        parser.add_argument('--op', type=Class)

        for pattern in [r'[\s=]', r'\s']:
            with self.subTest('" "' if '=' in pattern else '"="'), mock_module(Class) as module:
                out = StringIO()
                args = re.split(pattern, f'--op.help={module}.Class --op.init_args.cal.help=TextCalendar')
                with redirect_stdout(out), self.assertRaises(SystemExit):
                    parser.parse_args(args)
                self.assertIn('--op.init_args.cal.init_args.firstweekday', out.getvalue())

        with self.subTest('invalid'), mock_module(Class) as module:
            err = StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit):
                parser.parse_args([f'--op.help={module}.Class', '--op.init_args.p1=1'])
            self.assertIn('Expected a nested --*.help option', err.getvalue())


    def test_class_type_unresolved_parameters(self):
        class Class:
            def __init__(self, p1: int = 1, p2: str = '2', **kwargs):
                self.kwargs = kwargs

        with mock_module(Class) as module:
            config = f"""cls:
              class_path: {module}.Class
              init_args:
                  p1: 5
              dict_kwargs:
                  p2: '6'
                  p3: 7.0
                  p4: x
            """
            expected = Namespace(
                class_path=f'{module}.Class',
                init_args=Namespace(p1=5, p2='6'),
                dict_kwargs={'p3': 7.0, 'p4': 'x'},
            )

            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--config', action=ActionConfigFile)
            parser.add_argument('--cls', type=Class)

            cfg = parser.parse_args([f'--config={config}'])
            self.assertEqual(cfg.cls, expected)
            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.cls, Class)
            self.assertEqual(cfg_init.cls.kwargs, expected.dict_kwargs)

            cfg = parser.parse_args(['--cls=Class', '--cls.dict_kwargs.p4=-', '--cls.dict_kwargs.p3=7.0', '--cls.dict_kwargs.p4=x'])
            self.assertEqual(cfg.cls.dict_kwargs, expected.dict_kwargs)

            with self.assertRaises(ParserError):
                parser.parse_args(['--cls=Class', '--cls.dict_kwargs=1'])

            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args([f'--config={config}', '--print_config'])
            data = yaml.safe_load(out.getvalue())['cls']
            self.assertEqual(data, expected.as_dict())


    def test_class_type_unresolved_name_clash(self):
        class Class:
            def __init__(self, dict_kwargs: int = 1, **kwargs):
                self.kwargs = kwargs

        with mock_module(Class) as module:
            parser = ArgumentParser()
            parser.add_argument('--cls', type=Class)
            args = [f'--cls={module}.Class', '--cls.dict_kwargs=2']
            cfg = parser.parse_args(args)
            self.assertEqual(cfg.cls.init_args.as_dict(), {'dict_kwargs': 2})
            args.append('--cls.dict_kwargs.p1=3')
            cfg = parser.parse_args(args)
            self.assertEqual(cfg.cls.init_args.as_dict(), {'dict_kwargs': 2})
            self.assertEqual(cfg.cls.dict_kwargs, {'p1': 3})


    def test_invalid_init_args_in_yaml(self):
        config = """cal:
            class_path: calendar.Calendar
            init_args:
        """
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--config', action=ActionConfigFile)
        parser.add_argument('--cal', type=Calendar)
        self.assertRaises(ParserError, lambda: parser.parse_args([f'--config={config}']))


    def test_help_known_subclasses_class(self):
        parser = ArgumentParser()
        parser.add_argument('--cal', type=Calendar)
        out = StringIO()
        parser.print_help(out)
        self.assertIn('known subclasses: calendar.Calendar,', out.getvalue())


    def test_help_known_subclasses_type(self):
        parser = ArgumentParser()
        parser.add_argument('--cal', type=Type[Calendar])
        out = StringIO()
        parser.print_help(out)
        self.assertIn('known subclasses: calendar.Calendar,', out.getvalue())


    def test_class_type_required(self):
        parser = ArgumentParser()
        parser.add_argument('cal', type=Calendar)

        cfg = parser.parse_args(['TextCalendar'])
        self.assertEqual(cfg.cal.class_path, 'calendar.TextCalendar')

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn("required, type: <class 'Calendar'>", help_str.getvalue())
        self.assertIn('--cal.help', help_str.getvalue())


    def test_typehint_serialize_list(self):
        parser = ArgumentParser()
        action = parser.add_argument('--list', type=Union[PositiveInt, List[PositiveInt]])
        self.assertEqual([1, 2], action.serialize([PositiveInt(1), PositiveInt(2)]))
        self.assertRaises(ValueError, lambda: action.serialize([1, -2]))


    def test_typehint_serialize_enum(self):

        class MyEnum(Enum):
            a = 1
            b = 2

        parser = ArgumentParser()
        action = parser.add_argument('--enum', type=Optional[MyEnum])
        self.assertEqual('b', action.serialize(MyEnum.b))
        self.assertRaises(ValueError, lambda: action.serialize('x'))


    def test_unsupported_type(self):
        for typehint in [
            lambda: None,
            'unsupported',
            Optional['unsupported'],
            Tuple[int, 'unsupported'],
            Union['unsupported1', 'unsupported2'],
        ]:
            with self.subTest(typehint):
                with self.assertRaises(ValueError):
                    ActionTypeHint(typehint=typehint)


    def test_union_partially_unsupported_type(self):
        parser = ArgumentParser(logger={'level': 'DEBUG'})
        with self.assertLogs(logger=parser.logger, level='DEBUG') as log:
            parser.add_argument('--union', type=Union[int, str, 'unsupported'])
            self.assertEqual(1, len(log.output))
            self.assertIn('Discarding unsupported subtypes', log.output[0])


    def test_nargs_questionmark(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('p1')
        parser.add_argument('p2', nargs='?', type=OpenUnitInterval)
        self.assertIsNone(parser.parse_args(['a']).p2)
        self.assertEqual(0.5, parser.parse_args(['a', '0.5']).p2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['a', 'b']))


    def test_register_type(self):

        def serializer(v):
            return v.isoformat()

        def deserializer(v):
            return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')

        register_type(datetime, serializer, deserializer)
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--datetime', type=datetime)
        cfg = parser.parse_args(['--datetime=2008-09-03T20:56:35'])
        self.assertEqual(cfg.datetime, datetime(2008, 9, 3, 20, 56, 35))
        self.assertEqual(parser.dump(cfg), "datetime: '2008-09-03T20:56:35'\n")
        self.assertRaises(ValueError, lambda: register_type(datetime))
        register_type(uuid.UUID)


    def test_lazy_instance_invalid_kwargs(self):
        class MyClass:
            def __init__(self, param: int = 1):
                pass

        self.assertRaises(ValueError, lambda: lazy_instance(MyClass, param='bad'))


    def test_dump_skip_default(self):
        class MyCalendar(Calendar):
            def __init__(self, *args, param: str = '0', **kwargs):
                super().__init__(*args, **kwargs)

        with mock_module(MyCalendar) as module:
            parser = ArgumentParser()
            parser.add_argument('--g1.op1', default=1)
            parser.add_argument('--g1.op2', default='abc')
            parser.add_argument('--g2.op1', type=Callable, default=deepcopy)
            parser.add_argument('--g2.op2', type=Calendar, default=lazy_instance(Calendar, firstweekday=2))

            cfg = parser.get_defaults()
            dump = parser.dump(cfg, skip_default=True)
            self.assertEqual(dump, '{}\n')

            cfg.g2.op2.class_path = f'{module}.MyCalendar'
            dump = parser.dump(cfg, skip_default=True)
            self.assertEqual(dump, f'g2:\n  op2:\n    class_path: {module}.MyCalendar\n    init_args:\n      firstweekday: 2\n')

            cfg.g2.op2.init_args.firstweekday = 0
            dump = parser.dump(cfg, skip_default=True)
            self.assertEqual(dump, f'g2:\n  op2:\n    class_path: {module}.MyCalendar\n')

            parser.link_arguments('g1.op1', 'g2.op2.init_args.firstweekday')
            parser.link_arguments('g1.op2', 'g2.op2.init_args.param')
            del cfg['g2.op2.init_args']
            dump = parser.dump(cfg, skip_default=True)
            self.assertEqual(dump, f'g2:\n  op2:\n    class_path: {module}.MyCalendar\n')


class TypeHintsTmpdirTests(TempDirTestCase):

    def test_path(self):
        os.mkdir(os.path.join(self.tmpdir, 'example'))
        rel_yaml_file = os.path.join('..', 'example', 'example.yaml')
        abs_yaml_file = os.path.realpath(os.path.join(self.tmpdir, 'example', rel_yaml_file))
        with open(abs_yaml_file, 'w') as output_file:
            output_file.write('file: '+rel_yaml_file+'\ndir: '+self.tmpdir+'\n')

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_argument('--file', type=Path_fr)
        parser.add_argument('--dir', type=Path_drw)
        parser.add_argument('--files', nargs='+', type=Path_fr)

        cfg = parser.parse_args(['--cfg', abs_yaml_file])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(rel_yaml_file, str(cfg.file))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))

        cfg = parser.parse_args(['--cfg', 'file: '+abs_yaml_file+'\ndir: '+self.tmpdir+'\n'])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))

        cfg = parser.parse_args(['--file', abs_yaml_file, '--dir', self.tmpdir])
        self.assertEqual(self.tmpdir, os.path.realpath(cfg.dir()))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.file()))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--dir', abs_yaml_file]))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--file', self.tmpdir]))

        cfg = parser.parse_args(['--files', abs_yaml_file, abs_yaml_file])
        self.assertTrue(isinstance(cfg.files, list))
        self.assertEqual(2, len(cfg.files))
        self.assertEqual(abs_yaml_file, os.path.realpath(cfg.files[-1]()))


    def test_list_path(self):
        parser = ArgumentParser()
        parser.add_argument('--paths', type=List[Path_fc])
        cfg = parser.parse_args(['--paths=["file1", "file2"]'])
        self.assertEqual(['file1', 'file2'], cfg.paths)
        self.assertIsInstance(cfg.paths[0], Path)
        self.assertIsInstance(cfg.paths[1], Path)


    def test_optional_path(self):
        pathlib.Path('file_fr').touch()
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--path', type=Optional[Path_fr])
        self.assertIsNone(parser.parse_args(['--path=null']).path)
        cfg = parser.parse_args(['--path=file_fr'])
        self.assertEqual('file_fr', cfg.path)
        self.assertIsInstance(cfg.path, Path)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--path=not_exist']))


    def test_enable_path(self):
        data = {'a': 1, 'b': 2, 'c': [3, 4]}
        cal = {'class_path': 'calendar.Calendar'}
        with open('data.yaml', 'w') as f:
            json.dump(data, f)
        with open('cal.yaml', 'w') as f:
            json.dump(cal, f)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--data', type=Dict[str, Any], enable_path=True)
        parser.add_argument('--cal', type=Calendar, enable_path=True)
        cfg = parser.parse_args(['--data=data.yaml'])
        self.assertEqual('data.yaml', str(cfg['data'].pop('__path__')))
        self.assertEqual(data, cfg['data'])
        cfg = parser.instantiate_classes(parser.parse_args(['--cal=cal.yaml']))
        self.assertIsInstance(cfg['cal'], Calendar)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--data=does-not-exist.yaml']))


    def test_default_path_unregistered_type(self):
        parser = ArgumentParser()
        parser.add_argument('--path',
                            type=path_type('drw', skip_check=True),
                            default=Path('test', mode='drw', skip_check=True))
        cfg = parser.parse_args([])
        self.assertEqual('path: test\n', parser.dump(cfg))
        out = StringIO()
        parser.print_help(out)
        self.assertIn('(type: Path_drw_skip_check, default: test)', out.getvalue())


    def test_path_like_within_subclass(self):
        class Data:
            def __init__(self, path: Optional[os.PathLike] = None):
                pass

        data_path = pathlib.Path('data.json')
        data_path.write_text('{"a": 1}')

        parser = ArgumentParser()
        parser.add_argument('--data', type=Data, enable_path=True)

        with mock_module(Data) as module:
            cfg = parser.parse_args([f'--data={module}.Data', f'--data.path={data_path}'])
            self.assertEqual(cfg.data.class_path, f'{module}.Data')
            self.assertEqual(cfg.data.init_args, Namespace(path=str(data_path)))


    def test_list_append_default_config_files(self):
        config_path = pathlib.Path(self.tmpdir, 'config.yaml')
        parser = ArgumentParser(default_config_files=[str(config_path)])
        parser.add_argument('--nums', type=List[int], default=[0])

        with self.subTest('replace in default config'):
            config_path.write_text('nums: [1]\n')
            cfg = parser.parse_args(['--nums+=2'])
            self.assertEqual(cfg.nums, [1, 2])
            cfg = parser.parse_args(['--nums+=[2, 3]'])
            self.assertEqual(cfg.nums, [1, 2, 3])

        with self.subTest('append in default config'):
            config_path.write_text('nums+: [1]\n')
            cfg = parser.get_defaults()
            self.assertEqual(cfg.nums, [0, 1])
            cfg = parser.parse_args(['--nums+=2'])
            self.assertEqual(cfg.nums, [0, 1, 2])
            cfg = parser.parse_args(['--nums+=[2, 3]'])
            self.assertEqual(cfg.nums, [0, 1, 2, 3])

        with self.subTest('two default config appends'):
            config_path2 = pathlib.Path(self.tmpdir, 'config2.yaml')
            config_path2.write_text('nums+: [2]\n')
            parser.default_config_files += [str(config_path2)]
            cfg = parser.get_defaults()
            self.assertEqual(cfg.nums, [0, 1, 2])


    def test_list_append_default_config_files_subcommand(self):
        config_path = pathlib.Path(self.tmpdir, 'config.yaml')
        parser = ArgumentParser(default_config_files=[str(config_path)])
        subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        subparser.add_argument('--nums', type=List[int], default=[0])
        subcommands.add_subcommand('sub', subparser)
        config_path.write_text('sub:\n  nums: [1]\n')
        cfg = parser.parse_args(['sub', '--nums+=2'])
        self.assertEqual(cfg.sub.nums, [1, 2])


    def test_class_type_with_default_config_files(self):
        config = {
            'class_path': 'calendar.Calendar',
            'init_args': {'firstweekday': 3},
        }
        config_path = os.path.join(self.tmpdir, 'config.yaml')
        with open(config_path, 'w') as f:
            json.dump({'data': {'cal': config}}, f)

        class MyClass:
            def __init__(self, cal: Optional[Calendar] = None, val: int = 2):
                self.cal = cal

        parser = ArgumentParser(error_handler=None, default_config_files=[config_path])
        parser.add_argument('--op', default='from default')
        parser.add_class_arguments(MyClass, 'data')

        cfg = parser.get_defaults()
        self.assertEqual(config_path, str(cfg['__default_config__']))
        self.assertEqual(cfg.data.cal.as_dict(), config)
        dump = parser.dump(cfg)
        self.assertIn('class_path: calendar.Calendar\n', dump)
        self.assertIn('firstweekday: 3\n', dump)

        cfg = parser.parse_args([])
        self.assertEqual(cfg.data.cal.as_dict(), config)
        cfg = parser.parse_args(['--data.cal.class_path=calendar.Calendar'], defaults=False)
        self.assertEqual(cfg.data.cal, Namespace(class_path='calendar.Calendar'))


    def test_class_path_override_config_with_defaults(self):
        class Base:
            def __init__(self, b: int = 1):
                pass

        class Subclass1(Base):
            def __init__(self, s1: str = '-'):
                pass

        class Subclass2(Base):
            def __init__(self, s2: str = '-'):
                pass

        with mock_module(Base, Subclass1, Subclass2) as module:
            parser = ArgumentParser()
            parser.add_argument('--cfg', action=ActionConfigFile)
            parser.add_argument('--s', type=Base, default=lazy_instance(Subclass1, s1='v1'))

            config = {'s': {'class_path': 'Subclass2', 'init_args': {'s2': 'v2'}}}
            with warnings.catch_warnings(record=True) as w:
                cfg = parser.parse_args([f'--cfg={config}'])
                self.assertIn("discarding init_args: {'s1': 'v1'}", str(w[0].message))
            self.assertEqual(cfg.s.class_path, f'{module}.Subclass2')
            self.assertEqual(cfg.s.init_args, Namespace(s2='v2'))


    def test_class_path_override_with_default_config_files(self):

        class MyCalendar(Calendar):
            def __init__(self, *args, param: str = '0', **kwargs):
                super().__init__(*args, **kwargs)

        with mock_module(MyCalendar) as module:
            config = {
                'class_path': f'{module}.MyCalendar',
                'init_args': {'firstweekday': 2, 'param': '1'},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')
            with open(config_path, 'w') as f:
                json.dump({'cal': config}, f)

            parser = ArgumentParser(error_handler=None, default_config_files=[config_path])
            parser.add_argument('--cal', type=Optional[Calendar])

            cfg = parser.instantiate_classes(parser.get_defaults())
            self.assertIsInstance(cfg['cal'], MyCalendar)

            with warnings.catch_warnings(record=True) as w:
                cfg = parser.parse_args(['--cal={"class_path": "calendar.Calendar", "init_args": {"firstweekday": 3}}'])
                self.assertIn("discarding init_args: {'param': '1'}", str(w[0].message))
            self.assertEqual(cfg.cal.init_args, Namespace(firstweekday=3))
            self.assertEqual(type(parser.instantiate_classes(cfg)['cal']), Calendar)


    def test_mapping_class_typehint(self):
        class A:
            pass

        class B:
            def __init__(
                self,
                class_map: Mapping[str, A],
                int_list: List[int],
            ):
                self.class_map = class_map
                self.int_list = int_list

        with mock_module(A, B) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(B, 'b')

            config = {
                'b': {
                    'class_map': {
                        'one': {'class_path': f'{module}.A'},
                    },
                    'int_list': [1],
                },
            }

            cfg = parser.parse_object(config)
            self.assertEqual(cfg.b.class_map, {'one': Namespace(class_path=f'{module}.A')})
            self.assertEqual(cfg.b.int_list, [1])

            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.b, B)
            self.assertIsInstance(cfg_init.b.class_map, dict)
            self.assertIsInstance(cfg_init.b.class_map['one'], A)

            config['b']['int_list'] = config['b']['class_map']
            self.assertRaises(ParserError, lambda: parser.parse_object(config))


    def test_subcommand_with_subclass_default_override_lightning_issue_10859(self):

        class Arch:
            def __init__(self, a: int = 1):
                pass

        class ArchB(Arch):
            def __init__(self, a: int = 2, b: int = 3):
                pass

        class ArchC(Arch):
            def __init__(self, a: int = 4, c: int = 5):
                pass

        parser = ArgumentParser(error_handler=None)
        parser_subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        subparser.add_argument('--arch', type=Arch)

        with mock_module(Arch, ArchB, ArchC) as module:
            default = {'class_path': f'{module}.ArchB'}
            value = {'class_path': f'{module}.ArchC', 'init_args': {'a': 10, 'c': 11}}

            subparser.set_defaults(arch=default)
            parser_subcommands.add_subcommand('fit', subparser)

            with warnings.catch_warnings(record=True) as w:
                cfg = parser.parse_args(['fit', f'--arch={json.dumps(value)}'])
                self.assertIn("discarding init_args: {'b': 3}", str(w[0].message))
            self.assertEqual(cfg.fit.arch.as_dict(), value)


class OtherTests(unittest.TestCase):

    def test_is_optional(self):
        class MyEnum(Enum):
            A = 1

        params = [
            (Optional[bool],             bool, True),
            (Union[type(None), bool],    bool, True),
            (Dict[bool, type(None)],     bool, False),
            (Optional[Path_fr],          Path, True),
            (Union[type(None), Path_fr], Path, True),
            (Dict[Path_fr, type(None)],  Path, False),
            (Optional[MyEnum],           Enum, True),
            (Union[type(None), MyEnum],  Enum, True),
            (Dict[MyEnum, type(None)],   Enum, False),
        ]

        for typehint, ref_type, expected in params:
            with self.subTest(str(typehint)):
                self.assertEqual(expected, is_optional(typehint, ref_type))


if __name__ == '__main__':
    unittest.main(verbosity=2)
