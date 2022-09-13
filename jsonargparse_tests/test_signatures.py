#!/usr/bin/env python3

import dataclasses
import json
import os
import unittest
import warnings
import yaml
from calendar import Calendar, January  # type: ignore
from contextlib import redirect_stderr, redirect_stdout
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
    compose_dataclasses,
    lazy_instance,
    Namespace,
    ParserError,
    strip_meta,
)
from jsonargparse.actions import _find_action
from jsonargparse.optionals import docstring_parser_support, set_docstring_parse_options
from jsonargparse.typing import final, OpenUnitInterval, PositiveFloat, PositiveInt
from jsonargparse_tests.base import mock_module, suppress_stderr, TempDirTestCase


class SignaturesTests(unittest.TestCase):

    def test_add_class_arguments(self):

        class Class0:
            def __init__(self,
                         c0_a0: Optional[str] = '0'):
                pass

        class Class1(Class0):
            def __init__(self,
                         c1_a1: str,
                         c1_a2: Any = 2.0,
                         c1_a3 = None,
                         c1_a4: int = 4,
                         c1_a5: str = '5'):
                """Class1 short description

                Args:
                    c1_a2: c1_a2 description
                    c1_a3: c1_a3 description
                """
                super().__init__()
                self.c1_a1 = c1_a1

            def __call__(self):
                return self.c1_a1

        class Class2(Class1):
            def __init__(self,
                         c2_a0,
                         c3_a4,
                         *args,
                         **kwargs):
                super().__init__(c3_a4, *args, **kwargs)

        class Class3(Class2):
            def __init__(self,
                         c3_a0: Any,
                         c3_a1 = '1',
                         c3_a2: float = 2.0,
                         c3_a3: bool = False,
                         c3_a4: Optional[str] = None,
                         c3_a5: Union[int, float, str, List[int], Dict[str, float]] = 5,
                         c3_a6: Optional[Class1] = None,
                         c3_a7: Tuple[str, int, float] = ('7', 7, 7.0),
                         c3_a8: Tuple[str, Class1] = None,
                         c1_a5: str = 'five',
                         **kwargs):
                """Class3 short description

                Args:
                    c3_a0: c3_a0 description
                    c3_a1: c3_a1 description
                    c3_a2: c3_a2 description
                    c3_a4: c3_a4 description
                    c3_a5: c3_a5 description
                """
                super().__init__(None, c3_a4, **kwargs)

        ## Test without nesting ##
        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(Class3)

        self.assertRaises(ValueError, lambda: parser.add_class_arguments('Class3'))

        self.assertIn('Class3', parser.groups)

        for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a3', 'c3_a4', 'c3_a5', 'c3_a6', 'c3_a7', 'c3_a8', 'c1_a2', 'c1_a4', 'c1_a5']:
            self.assertIsNotNone(_find_action(parser, key), key+' should be in parser but is not')
        for key in ['c2_a0', 'c1_a1', 'c1_a3', 'c0_a0']:
            self.assertIsNone(_find_action(parser, key), key+' should not be in parser but is')

        cfg = parser.parse_args(['--c3_a0=0', '--c3_a3=true', '--c3_a4=a'], with_meta=False)
        self.assertEqual(cfg.as_dict(), {
            'c1_a2': 2.0,
            'c1_a4': 4,
            'c1_a5': 'five',
            'c3_a0': 0,
            'c3_a1': '1',
            'c3_a2': 2.0,
            'c3_a3': True,
            'c3_a4': 'a',
            'c3_a5': 5,
            'c3_a6': None,
            'c3_a7': ('7', 7, 7.0),
            'c3_a8': None,
        })
        self.assertEqual([1, 2], parser.parse_args(['--c3_a0=0', '--c3_a5=[1,2]']).c3_a5)
        self.assertEqual({'k': 5.0}, parser.parse_args(['--c3_a0=0', '--c3_a5={"k": 5.0}']).c3_a5)
        self.assertEqual(('3', 3, 3.0), parser.parse_args(['--c3_a0=0', '--c3_a7=["3", 3, 3.0]']).c3_a7)
        self.assertEqual('a', Class3(**cfg.as_dict())())

        self.assertRaises(ParserError, lambda: parser.parse_args([]))  # c3_a0 is required
        self.assertRaises(ParserError, lambda: parser.parse_args(['--c3_a0=0', '--c3_a7=["3", "3", 3.0]']))  # tuple[1] is int

        if docstring_parser_support:
            self.assertEqual('Class3 short description', parser.groups['Class3'].title)
            for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a4', 'c3_a5', 'c1_a2']:
                self.assertEqual(key+' description', _find_action(parser, key).help)
            for key in ['c3_a3', 'c3_a7', 'c1_a4']:
                self.assertNotEqual(key+' description', _find_action(parser, key).help)

        ## Test nested and as_group=False ##
        parser = ArgumentParser()
        added_args = parser.add_class_arguments(Class3, 'g', as_group=False)

        self.assertNotIn('g', parser.groups)
        self.assertEqual(12, len(added_args))
        self.assertTrue(all(a.startswith('g.') for a in added_args))

        for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a3', 'c3_a4', 'c3_a5', 'c3_a6', 'c3_a7', 'c3_a8', 'c1_a2', 'c1_a4', 'c1_a5']:
            self.assertIsNotNone(_find_action(parser, 'g.'+key), key+' should be in parser but is not')
        for key in ['c2_a0', 'c1_a1', 'c1_a3', 'c0_a0']:
            self.assertIsNone(_find_action(parser, 'g.'+key), key+' should not be in parser but is')

        ## Test default group title ##
        parser = ArgumentParser()
        parser.add_class_arguments(Class0)
        self.assertEqual(str(Class0), parser.groups['Class0'].title)

        ## Test positional without type ##
        self.assertRaises(ValueError, lambda: parser.add_class_arguments(Class2))


    def test_add_class_without_valid_args(self):
        class NoValidArgs:
            def __init__(self, a0=None):
                pass

        parser = ArgumentParser()
        self.assertEqual([], parser.add_class_arguments(NoValidArgs))


    def test_add_function_with_dict_int_keys_arg(self):
        def func(a1: Union[int, Dict[int, int]] = 1):
            pass

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)
        parser.get_defaults()
        self.assertEqual({2: 7, 4: 9}, parser.parse_args(['--a1={"2": 7, "4": 9}']).a1)


    def test_add_class_implemented_with_new(self):

        class ClassA:
            def __new__(cls, a1: int = 1, a2: float = 2.3):
                obj = object.__new__(cls)
                obj.a1 = a1  # type: ignore
                obj.a2 = a2  # type: ignore
                return obj

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassA, 'a')

        cfg = parser.parse_args(['--a.a1=4'])
        self.assertEqual(cfg.a, Namespace(a1=4, a2=2.3))


    def test_add_method_arguments(self):

        class MyClass:
            def mymethod(self,
                         a1 = '1',
                         a2: float = 2.0,
                         a3: bool = False,
                         a4 = None):
                """mymethod short description

                Args:
                    a1: a1 description
                    a2: a2 description
                    a4: a4 description
                """
                return a1

            @staticmethod
            def mystaticmethod(a1: str,
                               a2: float = 2.0,
                               a3 = None):
                return a1

        parser = ArgumentParser()
        added_args1 = parser.add_method_arguments(MyClass, 'mymethod', 'm')
        added_args2 = parser.add_method_arguments(MyClass, 'mystaticmethod', 's')

        self.assertRaises(ValueError, lambda: parser.add_method_arguments('MyClass', 'mymethod'))
        self.assertRaises(ValueError, lambda: parser.add_method_arguments(MyClass, 'mymethod3'))

        self.assertIn('m', parser.groups)
        self.assertIn('s', parser.groups)
        self.assertEqual(added_args1, ['m.a1', 'm.a2', 'm.a3'])
        self.assertEqual(added_args2, ['s.a1', 's.a2'])

        for key in ['m.a1', 'm.a2', 'm.a3', 's.a1', 's.a2']:
            self.assertIsNotNone(_find_action(parser, key), key+' should be in parser but is not')
        for key in ['m.a4', 's.a3']:
            self.assertIsNone(_find_action(parser, key), key+' should not be in parser but is')

        cfg = parser.parse_args(['--m.a1=x', '--s.a1=y'], with_meta=False).as_dict()
        self.assertEqual(cfg, {'m': {'a1': 'x', 'a2': 2.0, 'a3': False}, 's': {'a1': 'y', 'a2': 2.0}})
        self.assertEqual('x', MyClass().mymethod(**cfg['m']))
        self.assertEqual('y', MyClass.mystaticmethod(**cfg['s']))

        if docstring_parser_support:
            self.assertEqual('mymethod short description', parser.groups['m'].title)
            self.assertEqual(str(MyClass.mystaticmethod), parser.groups['s'].title)
            for key in ['m.a1', 'm.a2']:
                self.assertEqual(key.split('.')[1]+' description', _find_action(parser, key).help)
            for key in ['m.a3', 's.a1', 's.a2']:
                self.assertNotEqual(key.split('.')[1]+' description', _find_action(parser, key).help)


    def test_add_method_arguments_parent_classes(self):
        class MyClass:
            def mymethod(self, p1: str = '1'):
                return p1

        class MySubClass(MyClass):
            def mymethod(self, *args, p2: int = 2, **kwargs):
                p1 = super().mymethod(**kwargs)
                return p1, p2

        parser = ArgumentParser()
        added_args = parser.add_method_arguments(MySubClass, 'mymethod', 'm')

        self.assertIn('m', parser.groups)
        self.assertEqual(set(added_args), {'m.p1', 'm.p2'})


    def test_add_function_arguments(self):

        def func(a1 = '1',
                 a2: float = 2.0,
                 a3: bool = False,
                 a4 = None):
            """func short description

            Args:
                a1: a1 description
                a2: a2 description
                a4: a4 description
            """
            return a1

        parser = ArgumentParser()
        parser.add_function_arguments(func)

        self.assertRaises(ValueError, lambda: parser.add_function_arguments('func'))

        self.assertIn('func', parser.groups)

        for key in ['a1', 'a2', 'a3']:
            self.assertIsNotNone(_find_action(parser, key), key+' should be in parser but is not')
        self.assertIsNone(_find_action(parser, 'a4'), 'a4 should not be in parser but is')

        cfg = parser.parse_args(['--a1=x'], with_meta=False).as_dict()
        self.assertEqual(cfg, {'a1': 'x', 'a2': 2.0, 'a3': False})
        self.assertEqual('x', func(**cfg))

        if docstring_parser_support:
            self.assertEqual('func short description', parser.groups['func'].title)
            for key in ['a1', 'a2']:
                self.assertEqual(key+' description', _find_action(parser, key).help)


    def test_add_subclass_arguments(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(Calendar, 'cal')

        cal = {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 1}}
        cfg = parser.parse_args(['--cal='+json.dumps(cal)])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        cal['init_args']['firstweekday'] = 2
        cfg = parser.parse_args(['--cal.class_path=calendar.Calendar', '--cal.init_args.firstweekday=2'])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        cal['init_args']['firstweekday'] = 3
        cfg = parser.parse_args(['--cal.class_path', 'calendar.Calendar', '--cal.init_args.firstweekday', '3'])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        cal['init_args']['firstweekday'] = 4
        cfg = parser.parse_args(['--cal.class_path=calendar.Calendar', '--cal.init_args.firstweekday=4', '--cal.class_path=calendar.Calendar'])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal={"class_path":"not.exist.Class"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal={"class_path":"calendar.January"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal.help=calendar.January']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal.help=calendar.does_not_exist']))
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(January, 'jan'))

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--cal.help=calendar.Calendar'])
        self.assertIn('--cal.init_args.firstweekday', out.getvalue())

        # lazy_instance
        class MyCalendar(Calendar):
            init_called = False
            getfirst = Calendar.getfirstweekday
            def __init__(self, *args, **kwargs):
                self.init_called = True
                super().__init__(*args, **kwargs)

        lazy_calendar = lazy_instance(MyCalendar, firstweekday=3)
        self.assertFalse(lazy_calendar.init_called, '__init__ was already called but supposed to be lazy')
        self.assertEqual(lazy_calendar.getfirstweekday(), 3)
        self.assertTrue(lazy_calendar.init_called)

        cal['init_args']['firstweekday'] = 4
        lazy_calendar = lazy_instance(Calendar, firstweekday=4)
        parser.set_defaults({'cal': lazy_calendar})
        cfg = parser.parse_string(parser.dump(parser.parse_args([])))
        self.assertEqual(cfg['cal'].as_dict(), cal)
        self.assertEqual(lazy_calendar.getfirstweekday(), 4)

        parser.add_argument('--config', action=ActionConfigFile)
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--print_config'])
        self.assertIn('class_path: calendar.Calendar', out.getvalue())

        out = StringIO()
        parser.print_help(out)
        self.assertIn("'init_args': {'firstweekday': 4}", out.getvalue())

        # defaults
        parser.set_defaults({'cal': Calendar(firstweekday=4)})
        cfg = parser.parse_args([])
        self.assertIsInstance(cfg['cal'], Calendar)
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['cal'], Calendar)
        with warnings.catch_warnings(record=True) as w:
            dump = parser.dump(cfg)
            self.assertIn('Not possible to serialize an instance of', str(w[0].message))
        self.assertIn('cal: <calendar.Calendar object at ', dump)


    def test_subclass_help(self):
        class MyCal(Calendar):
            def __init__(self, *args, param, **kwargs):
                self.param = param
                super().__init__(*args, **kwargs)

        with mock_module(MyCal) as module:
            args = [f'--cal.help={module}.MyCal']

            parser = ArgumentParser()
            parser.add_subclass_arguments(Calendar, 'cal', skip={'param'})

            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args(args)
            self.assertIn('--cal.init_args.firstweekday', out.getvalue())
            self.assertNotIn('param', out.getvalue())

            parser = ArgumentParser(error_handler=None)
            parser.add_subclass_arguments(Calendar, 'cal')
            self.assertRaises(ValueError, lambda: parser.parse_args(args))


    def test_add_subclass_discard_init_args(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(Calendar, 'cal')

        class CalA(Calendar):
            def __init__(self, pa: str = 'a', pc: str = '', **kwds):
                super().__init__(**kwds)

        class CalB(Calendar):
            def __init__(self, pb: str = 'b', pc: int = 4, **kwds):
                super().__init__(**kwds)

        with mock_module(CalA, CalB) as module, warnings.catch_warnings(record=True) as w:
            cfg = parser.parse_args([
                f'--cal.class_path={module}.CalA',
                '--cal.init_args.pa=A',
                '--cal.init_args.pc=X',
                '--cal.init_args.firstweekday=3',
                f'--cal.class_path={module}.CalB',
                '--cal.init_args.pb=B',
            ])
            self.assertEqual(cfg.cal.class_path, f'{module}.CalB')
            self.assertEqual(cfg.cal.init_args, Namespace(pb='B', pc=4, firstweekday=3))
            self.assertIn("discarding init_args: {'pa': 'A', 'pc': 'X'}", str(w[0].message))


    def test_class_path_override_with_mixed_type(self):
        class MyCalendar(Calendar):
            def __init__(self, *args, param: int = 0, **kwargs):
                super().__init__(*args, **kwargs)

        class Main:
            def __init__(self, cal: Union[Calendar, bool] = lazy_instance(MyCalendar, param=1)):
                self.cal = cal

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(Main, 'main')

        with warnings.catch_warnings(record=True) as w:
            parser.parse_args(['--main.cal=Calendar'])
            self.assertIn("discarding init_args: {'param': 1}", str(w[0].message))


    def test_add_subclass_init_args_without_class_path(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(Calendar, 'cal1')
        parser.add_subclass_arguments(Calendar, 'cal2', default=lazy_instance(Calendar))
        parser.add_subclass_arguments(Calendar, 'cal3', default=lazy_instance(Calendar, firstweekday=2))

        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal1.init_args.firstweekday=4']))
        cfg = parser.parse_args(['--cal2.init_args.firstweekday=4', '--cal3.init_args.firstweekday=5'])
        self.assertEqual(cfg.cal2.init_args, Namespace(firstweekday=4))
        self.assertEqual(cfg.cal3.init_args, Namespace(firstweekday=5))


    def test_add_subclass_merge_init_args_in_full_config(self):
        class ModelBaseClass():
            def __init__(self, batch_size: int=0):
                self.batch_size = batch_size

        class ModelClass(ModelBaseClass):
            def __init__(self, image_size: int=0, **kwargs):
                super().__init__(**kwargs)
                self.image_size = image_size

        with mock_module(ModelBaseClass, ModelClass) as module:
            parser = ArgumentParser(error_handler=None)
            parser.add_subclass_arguments(ModelBaseClass, 'model')
            parser.add_argument('--config', action=ActionConfigFile)

            model = yaml.safe_dump({
                'class_path': 'ModelClass',
                'init_args': {
                    'image_size': 10
                }
            })
            config = yaml.safe_dump({
                'model': {
                    'init_args': {
                        'batch_size': 5
                    }
                }
            })

            cfg = parser.parse_args([f'--model={model}', f'--config={config}'])
            self.assertEqual(cfg.model.class_path, f'{module}.ModelClass')
            self.assertEqual(cfg.model.init_args, Namespace(batch_size=5, image_size=10))


    def test_add_subclass_init_args_in_subcommand(self):
        parser = ArgumentParser(error_handler=None)
        subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        subparser.add_subclass_arguments(Calendar, 'cal', default=lazy_instance(Calendar))
        subcommands.add_subcommand('cmd', subparser)

        cfg = parser.parse_args(['cmd', '--cal.init_args.firstweekday=4'])
        self.assertEqual(cfg.cmd.cal.init_args, Namespace(firstweekday=4))


    def test_add_subclass_arguments_tuple(self):

        class ClassA:
            def __init__(self, a1: int = 1, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB:
            def __init__(self, b1: float = 4.5, b2: int = 6):
                self.b1 = b1
                self.b2 = b2

        with mock_module(ClassA, ClassB) as module:
            class_path_a = f'{module}.ClassA'
            class_path_b = f'{module}.ClassB'

            parser = ArgumentParser(error_handler=None)
            parser.add_subclass_arguments((ClassA, ClassB), 'c')

            cfg = parser.parse_args(['--c={"class_path": "'+class_path_a+'", "init_args": {"a1": -1}}'])
            self.assertEqual(cfg.c.init_args.a1, -1)
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c'], ClassA)

            cfg = parser.parse_args(['--c={"class_path": "'+class_path_b+'", "init_args": {"b1": -4.5}}'])
            self.assertEqual(cfg.c.init_args.b1, -4.5)
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c'], ClassB)

            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args(['--c.help='+class_path_b])

            self.assertIn('--c.init_args.b1', out.getvalue())


    def test_required_group(self):
        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(Calendar, None, required=True))
        parser.add_subclass_arguments(Calendar, 'cal', required=True)
        self.assertRaises(ParserError, lambda: parser.parse_args([]))
        out = StringIO()
        parser.print_help(out)
        self.assertIn('[-h] [--cal.help CLASS_PATH_OR_NAME] --cal ', out.getvalue())


    def test_not_required_group(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(Calendar, 'cal', required=False)
        cfg = parser.parse_args([])
        self.assertEqual(cfg, Namespace())
        cfg_init = parser.instantiate_classes(cfg)
        self.assertEqual(cfg_init, Namespace())


    def test_invalid_type(self):

        def func(a1: None):
            return a1

        parser = ArgumentParser(error_handler=None)
        self.assertRaises(ValueError, lambda: parser.add_function_arguments(func))


    def test_optional_enum(self):

        class MyEnum(Enum):
            A = 1
            B = 2
            C = 3

        def func(a1: Optional[MyEnum] = None):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)
        self.assertEqual(MyEnum.B, parser.parse_args(['--a1=B']).a1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a1=D']))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('--a1 {A,B,C,null}', help_str.getvalue())

        class MyEnum2(str, Enum):
            A = 'A'
            B = 'B'

        def func2(a1: Optional[MyEnum2] = None):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func2)
        self.assertEqual(MyEnum2.B, parser.parse_args(['--a1=B']).a1)
        self.assertEqual('B', parser.parse_args(['--a1=B']).a1)


    def test_type_any_serialize(self):

        class MyEnum(str, Enum):
            A = 'a'
            B = 'b'

        def func(a1: Any = MyEnum.B):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)
        cfg = parser.parse_args([])
        self.assertEqual('a1: B\n', parser.dump(cfg))


    def test_skip(self):

        def func(a1 = '1',
                 a2: float = 2.0,
                 a3: bool = False,
                 a4: int = 4):
            return a1

        parser = ArgumentParser()
        parser.add_function_arguments(func, skip={'a2', 'a4'})

        for key in ['a1', 'a3']:
            self.assertIsNotNone(_find_action(parser, key), key+' should be in parser but is not')
        for key in ['a2', 'a4']:
            self.assertIsNone(_find_action(parser, key), key+' should not be in parser but is')


    def test_skip_within_subclass_type(self):

        class Class1:
            def __init__(self, a1: int = 1, a2: float = 2.3, a3: str = '4'):
                self.a1 = a1
                self.a2 = a2
                self.a3 = a3

        class Class2:
            def __init__(self, c1: Class1, c2: int = 5, c3: float = 6.7):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(Class2, skip={'c1.init_args.a2', 'c2'})

        with mock_module(Class1) as module:
            cfg = parser.parse_args([f'--c1={module}.Class1'])
            self.assertEqual(cfg.c1.init_args, Namespace(a1=1, a3='4'))


    def test_skip_in_add_subclass_arguments(self):

        class ClassA:
            def __init__(self, a1: int = 1, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB(ClassA):
            def __init__(self, b1: float = 4.5, b2: int = 6, **kwargs):
                super().__init__(**kwargs)
                self.b1 = b1
                self.b2 = b2

        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(ClassA, 'c', skip={'a1', 'b2'})

        with mock_module(ClassA, ClassB) as module:
            class_path_a = f'{module}.ClassA'
            class_path_b = f'{module}.ClassB'

            cfg = parser.parse_args(['--c='+class_path_a])
            self.assertEqual(cfg.c.init_args, Namespace(a2=2.3))
            cfg = parser.parse_args(['--c='+class_path_b])
            self.assertEqual(cfg.c.init_args, Namespace(a2=2.3, b1=4.5))


    def test_final_class(self):

        @final
        class ClassA:
            def __init__(self, a1: int = 1, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB:
            def __init__(self, b1: str = '4', b2: ClassA = lazy_instance(ClassA, a2=-3.2)):
                self.b1 = b1
                self.b2 = b2

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassB, 'b')

        self.assertEqual(parser.get_defaults().b.b2, Namespace(a1=1, a2=-3.2))
        cfg = parser.parse_args(['--b.b2={"a2": 6.7}'])
        self.assertEqual(cfg.b.b2, Namespace(a1=1, a2=6.7))
        self.assertEqual(cfg, parser.parse_string(parser.dump(cfg)))
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['b'], ClassB)
        self.assertIsInstance(cfg['b'].b2, ClassA)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2={"bad": "value"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2="bad"']))
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(ClassA, 'a'))
        self.assertRaises(ValueError, lambda: parser.add_class_arguments(ClassA, 'a', default=ClassA()))


    def test_basic_subtypes(self):

        def func(a1: PositiveFloat = PositiveFloat(1),
                 a2: Optional[Union[PositiveInt, OpenUnitInterval]] = 0.5):
            return a1, a2

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)

        self.assertEqual(1.0, parser.parse_args(['--a1=1']).a1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a1=-1']))

        self.assertEqual(0.7, parser.parse_args(['--a2=0.7']).a2)
        self.assertEqual(5, parser.parse_args(['--a2=5']).a2)
        self.assertEqual(None, parser.parse_args(['--a2=null']).a2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a2=0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a2=1.5']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a2=-1']))


    def test_dict_int_str_type(self):
        class Foo:
            def __init__(self, d: Dict[int, str]):
                self.d = d

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(Foo)
        parser.add_argument('--config', action=ActionConfigFile)
        cfg = {'d': {1: 'val1', 2: 'val2'}}
        self.assertEqual(cfg['d'], parser.parse_args(['--config', str(cfg)]).d)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--config={"d": {"a": "b"}}']))


    def test_logger_debug(self):

        with suppress_stderr():

            class Class1:
                def __init__(self,
                            c1_a1: float,
                            c1_a2: int = 1):
                    pass

            class Class2(Class1):
                def __init__(self,
                            *args,
                            c2_a1: int = 2,
                            c2_a2: float = 0.2,
                            **kwargs):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(logger=parser.logger, level='DEBUG') as log:
                parser.add_class_arguments(Class2, skip={'c2_a2'})
                self.assertEqual(1, len(log.output))
                self.assertIn('"c2_a2" from "Class2.__init__"', log.output[0])
                self.assertIn('Parameter requested to be skipped', log.output[0])


    def test_instantiate_classes(self):
        class Class1:
            def __init__(self, a1: Optional[int] = 1, a2: Optional[float] = 2.3):
                self.a1 = a1
                self.a2 = a2

        class Class2:
            def __init__(self, c1: Optional[Class1]):
                self.c1 = c1

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(Class2)

        with mock_module(Class1) as module:
            class_path = f'"class_path": "{module}.Class1"'
            init_args = '"init_args": {"a1": 7}'
            cfg = parser.parse_args(['--c1={'+class_path+', '+init_args+'}'])
            self.assertEqual(cfg.c1.class_path, f'{module}.Class1')
            self.assertEqual(cfg.c1.init_args, Namespace(a1=7, a2=2.3))
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c1'], Class1)
            self.assertEqual(7, cfg['c1'].a1)
            self.assertEqual(2.3, cfg['c1'].a2)

            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(Class2, 'c2')

            cfg = parser.parse_args(['--c2={"c1": {'+class_path+', '+init_args+'}}'])
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c2'], Class2)
            self.assertIsInstance(cfg['c2'].c1, Class1)

            class EmptyInitClass:
                pass

            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(EmptyInitClass, 'e')
            cfg = parser.parse_args([])
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['e'], EmptyInitClass)


    def test_instantiate_classes_subcommand(self):
        class Foo:
            def __init__(self, a: int = 1):
                self.a = a

        parser = ArgumentParser()
        subcommands = parser.add_subcommands()
        subparser = ArgumentParser()
        key = "foo"
        subparser.add_class_arguments(Foo, key)
        subcommand = "cmd"
        subcommands.add_subcommand(subcommand, subparser)

        config = parser.parse_args([subcommand])
        config_init = parser.instantiate_classes(config)
        self.assertIsInstance(config_init[subcommand][key], Foo)


    def test_implicit_optional(self):

        def func(a1: int = None):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)

        self.assertIsNone(parser.parse_args(['--a1=null']).a1)


    def test_fail_untyped_false(self):

        def func(a1, a2=None):
            return a1

        parser = ArgumentParser(error_handler=None)
        added_args = parser.add_function_arguments(func, fail_untyped=False)

        self.assertEqual(['a1', 'a2'], added_args)
        self.assertEqual(Namespace(a1=None, a2=None), parser.parse_args([]))


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_docstring_parse_fail(self):

        class Class1:
            def __init__(self, a1: int = 1):
                """
                Args:
                    a1: a1 description
                """

        with unittest.mock.patch('docstring_parser.parse') as docstring_parse:
            from docstring_parser import ParseError
            docstring_parse.side_effect = ParseError
            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(Class1)

            help_str = StringIO()
            parser.print_help(help_str)
            self.assertIn('--a1 A1', help_str.getvalue())
            self.assertNotIn('a1 description', help_str.getvalue())


    def test_print_config(self):
        class MyClass:
            def __init__(
                self,
                a1: Calendar,
                a2: int = 7,
            ):
                pass

        parser = ArgumentParser()
        parser.add_argument('--config', action=ActionConfigFile)
        parser.add_class_arguments(MyClass, 'g')

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--g.a1=calendar.Calendar', '--print_config'])

        outval = yaml.safe_load(out.getvalue())
        self.assertEqual(outval['g'], {'a1': {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 0}}, 'a2': 7})

        err = StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            parser.parse_args(['--g.a1=calendar.Calendar', '--g.a1.invalid=1', '--print_config'])

        self.assertIn('No action for destination key "invalid"', err.getvalue())


    def test_print_config_subclass_required_param_issue_115(self):
        class Class(object):
            def __init__(self, arg1: float):
                pass

        class BaseClass(object):
            def __init__(self):
                pass

        class SubClass(BaseClass):
            def __init__(self, arg1: int, arg2: int = 1):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_argument("--config", action=ActionConfigFile)
        parser.add_class_arguments(Class, 'class')
        parser.add_subclass_arguments(BaseClass, 'subclass')

        with mock_module(BaseClass, SubClass) as module:
            args = [f'--subclass={module}.SubClass', '--print_config']
            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args(args)
            expected = f'class:\n  arg1: null\nsubclass:\n  class_path: {module}.SubClass\n  init_args:\n    arg1: null\n    arg2: 1\n'
            self.assertEqual(out.getvalue(), expected)


    def test_class_from_function(self):

        def get_calendar() -> Calendar:
            return Calendar()

        class Foo:
            @classmethod
            def get_foo(cls) -> 'Foo':
                return cls()

        def closure_get_foo():
            def get_foo() -> 'Foo':
                return Foo()
            return get_foo

        for function, class_type in [
            (get_calendar, Calendar),
            (Foo.get_foo, Foo),
            (closure_get_foo(), Foo),
        ]:
            with self.subTest(str((function, class_type))):
                cls = class_from_function(function)
                self.assertTrue(issubclass(cls, class_type))
                self.assertIsInstance(cls(), class_type)


    def test_invalid_class_from_function(self):

        def get_unknown() -> 'Unknown':  # type: ignore
            return None

        self.assertRaises(ValueError, lambda: class_from_function(get_unknown))


    def test_add_class_from_function_arguments(self):

        def get_calendar(a1: str, a2: int = 2) -> Calendar:
            """Returns instance of Calendar"""
            cal = Calendar()
            cal.a1 = a1  # type: ignore
            cal.a2 = a2  # type: ignore
            return cal

        get_calendar_class = class_from_function(get_calendar)

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(get_calendar_class, 'a')

        if docstring_parser_support:
            help_str = StringIO()
            parser.print_help(help_str)
            self.assertIn('Returns instance of Calendar', help_str.getvalue())

        cfg = parser.parse_args(['--a.a1=v', '--a.a2=3'])
        self.assertEqual(cfg.a, Namespace(a1='v', a2=3))
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['a'], Calendar)
        self.assertEqual(cfg['a'].a1, 'v')
        self.assertEqual(cfg['a'].a2, 3)


    def test_dict_type_nested_in_two_level_subclasses(self):

        class Module:
            pass

        class Network(Module):
            def __init__(self, sub_network: Module, some_dict: Dict[str, Any] = {}):
                pass

        class Model:
            def __init__(self, encoder: Module):
                pass

        with mock_module(Module, Network, Model) as module:

            config = f"""model:
              encoder:
                class_path: {module}.Network
                init_args:
                  some_dict:
                    a: 1
                  sub_network:
                    class_path: {module}.Network
                    init_args:
                      some_dict:
                        b: 2
                      sub_network:
                        class_path: {module}.Module
            """

            parser = ArgumentParser(error_handler=None)
            parser.add_argument('--config', action=ActionConfigFile)
            parser.add_class_arguments(Model, 'model')

            cfg = parser.parse_args([f'--config={config}'])
            self.assertEqual(cfg.model.encoder.init_args.some_dict, {'a': 1})
            self.assertEqual(cfg.model.encoder.init_args.sub_network.init_args.some_dict, {'b': 2})
            self.assertEqual(cfg.model.as_dict(), yaml.safe_load(config)['model'])


class SignaturesConfigTests(TempDirTestCase):

    def test_add_function_arguments_config(self):

        def func(a1 = '1',
                 a2: float = 2.0,
                 a3: bool = False):
            return a1

        parser = ArgumentParser(error_handler=None, default_meta=False)
        parser.add_function_arguments(func, 'func')

        cfg_path = 'config.yaml'
        with open(cfg_path, 'w') as f:
            f.write(yaml.dump({'a1': 'one', 'a3': True}))

        cfg = parser.parse_args(['--func', cfg_path])
        self.assertEqual(cfg.func, Namespace(a1='one', a2=2.0, a3=True))

        cfg = parser.parse_args(['--func={"a1": "ONE"}'])
        self.assertEqual(cfg.func, Namespace(a1='ONE', a2=2.0, a3=False))

        self.assertRaises(ParserError, lambda: parser.parse_args(['--func="""']))


    def test_config_within_config(self):

        def func(a1 = '1',
                 a2: float = 2.0,
                 a3: bool = False):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_function_arguments(func, 'func')

        cfg_path = 'subdir/config.yaml'
        subcfg_path = 'subsubdir/func_config.yaml'
        os.mkdir('subdir')
        os.mkdir('subdir/subsubdir')
        with open(cfg_path, 'w') as f:
            f.write('func: '+subcfg_path+'\n')
        with open(os.path.join('subdir', subcfg_path), 'w') as f:
            f.write(yaml.dump({'a1': 'one', 'a3': True}))

        cfg = parser.parse_args(['--cfg', cfg_path])
        self.assertEqual(str(cfg.func.__path__), subcfg_path)
        self.assertEqual(strip_meta(cfg.func), Namespace(a1='one', a2=2.0, a3=True))


    def test_add_subclass_arguments_with_config(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--cfg', action=ActionConfigFile)
        parser.add_subclass_arguments(Calendar, 'cal')

        cfg_path = 'config.yaml'
        cal = {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 1}}
        with open(cfg_path, 'w') as f:
            f.write(yaml.dump({'cal': cal}))

        cfg = parser.parse_args(['--cfg='+cfg_path])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        cal['init_args']['firstweekday'] = 2
        cfg = parser.parse_args(['--cfg='+cfg_path, '--cal.init_args.firstweekday=2'])
        self.assertEqual(cfg['cal'].as_dict(), cal)

        parser = ArgumentParser(error_handler=None, default_config_files=['config.yaml'])
        parser.add_subclass_arguments(Calendar, 'cal')

        cfg = parser.parse_args(['--cal.init_args.firstweekday=2'])
        self.assertEqual(cfg['cal'].as_dict(), cal)


    def test_add_class_arguments_with_config_not_found(self):
        class A:
            def __init__(self, param: int):
                self.param = param

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(A, 'a')
        try:
            parser.parse_args(['--a=does_not_exist.yaml'])
        except ParserError as ex:
            self.assertIn('Unable to load config "does_not_exist.yaml"', str(ex))
        else:
            raise ValueError('Expected ParserError to be raised')


    def test_add_subclass_arguments_with_multifile_save(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(Calendar, 'cal')

        cal_cfg_path = 'cal.yaml'
        with open(cal_cfg_path, 'w') as f:
            f.write(yaml.dump({'class_path': 'calendar.Calendar'}))

        cfg = parser.parse_args(['--cal='+cal_cfg_path])
        os.mkdir('out')
        out_main_cfg = os.path.join('out', 'config.yaml')
        parser.save(cfg, out_main_cfg, multifile=True)

        with open(out_main_cfg) as f:
            self.assertEqual('cal: cal.yaml', f.read().strip())
        with open(os.path.join('out', 'cal.yaml')) as f:
            cal = yaml.safe_load(f.read())
            self.assertEqual({'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 0}}, cal)


    def test_subclass_required_param_with_default_config_files(self):

        class SubModule:
            def __init__(self, p1: int, p2: int = 2, p3: int = 3):
                pass

        class Model:
            def __init__(self, sub_module: SubModule):
                pass

        with mock_module(SubModule, Model) as module:

            defaults = f"""model:
              sub_module:
                class_path: {module}.SubModule
                init_args:
                  p1: 4
                  p2: 5
            """
            expected = yaml.safe_load(defaults.replace('p2: 5', 'p2: 7'))['model']
            expected['sub_module']['init_args']['p3'] = 3

            with open('defaults.yaml', 'w') as f:
                f.write(defaults)

            parser = ArgumentParser(error_handler=None, default_config_files=['defaults.yaml'])
            parser.add_class_arguments(Model, 'model')

            cfg = parser.parse_args(['--model.sub_module.init_args.p2=7'])
            self.assertEqual(cfg.model.as_dict(), expected)


    def test_parent_parser_default_config_files_lightning_issue_11622(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11622

        with open('default.yaml', 'w') as f:
            f.write('fit:\n  model:\n    foo: 123')

        class Foo:
            def __init__(self, foo: int):
                self.foo = foo

        parser = ArgumentParser(default_config_files=["default.yaml"], error_handler=None)
        parser.add_argument('--config', action=ActionConfigFile)
        subcommands = parser.add_subcommands()

        subparser = ArgumentParser()
        subparser.add_class_arguments(Foo, nested_key="model")
        subcommands.add_subcommand("fit", subparser)

        subparser = ArgumentParser()
        subparser.add_class_arguments(Foo, nested_key="model")
        subcommands.add_subcommand("test", subparser)

        cfg = parser.parse_args(['fit'])
        self.assertEqual(cfg.fit.model.foo, 123)


class DataclassesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @dataclasses.dataclass(frozen=True)
        class MyDataClassA:
            """MyDataClassA description

            Args:
                a1: a1 help
                a2: a2 help
            """
            a1: PositiveInt = PositiveInt(1)
            a2: str = '2'

        @dataclasses.dataclass
        class MyDataClassB:
            """MyDataClassB description

            Args:
                b1: b1 help
                b2: b2 help
            """
            b1: PositiveFloat = PositiveFloat(3.0)
            b2: MyDataClassA = MyDataClassA()

        cls.MyDataClassA = MyDataClassA
        cls.MyDataClassB = MyDataClassB


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_add_dataclass_arguments(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_dataclass_arguments(self.MyDataClassA, 'a', default=self.MyDataClassA(), title='CustomA title')
        parser.add_dataclass_arguments(self.MyDataClassB, 'b', default=self.MyDataClassB())

        cfg = parser.get_defaults()
        self.assertEqual(dataclasses.asdict(self.MyDataClassA()), cfg['a'].as_dict())
        self.assertEqual(dataclasses.asdict(self.MyDataClassB()), cfg['b'].as_dict())
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dataclasses.asdict(self.MyDataClassA()), dump['a'])
        self.assertEqual(dataclasses.asdict(self.MyDataClassB()), dump['b'])

        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['a'], self.MyDataClassA)
        self.assertIsInstance(cfg['b'], self.MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, self.MyDataClassA)

        self.assertEqual(5, parser.parse_args(['--b.b2.a1=5']).b.b2.a1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2.a1=x']))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('CustomA title:', help_str.getvalue())
        self.assertIn('MyDataClassB description:', help_str.getvalue())
        self.assertIn('b2 help:', help_str.getvalue())

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(1, 'c')

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(self.MyDataClassB, 'c', default=self.MyDataClassB(b2=self.MyDataClassB()))

        class MyClass(int, self.MyDataClassA):
            """MyClass description"""

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(MyClass, 'c')


    def test_dataclass_typehint(self):

        class MyClass:
            def __init__(
                self,
                a1: self.MyDataClassA = self.MyDataClassA(),  # type: ignore
                a2: self.MyDataClassB = self.MyDataClassB(),  # type: ignore
            ):
                self.a1 = a1
                self.a2 = a2

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass, 'g')

        cfg = parser.get_defaults()
        self.assertEqual(dataclasses.asdict(self.MyDataClassA()), cfg['g']['a1'].as_dict())
        self.assertEqual(dataclasses.asdict(self.MyDataClassB()), cfg['g']['a2'].as_dict())
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dataclasses.asdict(self.MyDataClassA()), dump['g']['a1'])
        self.assertEqual(dataclasses.asdict(self.MyDataClassB()), dump['g']['a2'])

        cfg_init = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg_init.g.a1, self.MyDataClassA)
        self.assertIsInstance(cfg_init.g.a2, self.MyDataClassB)
        self.assertIsInstance(cfg_init.g.a2.b2, self.MyDataClassA)


    def test_dataclass_typehint_in_subclass(self):

        class MyClass1:
            def __init__(self, a1: self.MyDataClassB = self.MyDataClassB()):  # type: ignore
                """MyClass1 description"""
                self.a1 = a1

        class MyClass2:
            def __init__(self, c1: MyClass1):
                """MyClass2 description"""
                self.c1 = c1

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(MyClass2)

        with mock_module(MyClass1, MyClass2) as module:
            class_path = f'"class_path": "{module}.MyClass1"'
            init_args = '"init_args": {"a1": {"b2": {"a1": 7}}}'
            cfg = parser.parse_args(['--c1={'+class_path+', '+init_args+'}'])
            self.assertEqual(cfg.c1.class_path, f'{module}.MyClass1')
            self.assertEqual(cfg.c1.init_args.a1.b2.a1, 7)
            self.assertIsInstance(cfg.c1.init_args.a1.b2.a1, PositiveInt)
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c1'], MyClass1)
            self.assertIsInstance(cfg['c1'].a1, self.MyDataClassB)
            self.assertIsInstance(cfg['c1'].a1.b2, self.MyDataClassA)
            self.assertIsInstance(cfg['c1'].a1.b1, PositiveFloat)


    def test_dataclass_add_argument_type(self):
        parser = ArgumentParser()
        parser.add_argument('--b', type=self.MyDataClassB, default=self.MyDataClassB(b1=7.0))

        cfg = parser.get_defaults()
        self.assertEqual({'b1': 7.0, 'b2': {'a1': 1, 'a2': '2'}}, cfg['b'].as_dict())

        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['b'], self.MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, self.MyDataClassA)


    def test_dataclass_add_argument_type_some_required(self):

        @dataclasses.dataclass
        class MyDataClass:
            a1: str
            a2: float = 1.2

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--b', type=MyDataClass)

        self.assertEqual(Namespace(a1='v', a2=1.2), parser.parse_args(['--b.a1=v']).b)
        self.assertRaises(ParserError, lambda: parser.parse_args([]))


    def test_dataclass_field_default_factory(self):

        @dataclasses.dataclass
        class MyDataClass:
            a1: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
            a2: Dict[str, float] = dataclasses.field(default_factory=lambda: {'a': 1.2, 'b': 3.4})

        parser = ArgumentParser()
        parser.add_class_arguments(MyDataClass)

        cfg = parser.get_defaults().as_dict()
        self.assertEqual([1, 2, 3], cfg['a1'])
        self.assertEqual({'a': 1.2, 'b': 3.4}, cfg['a2'])


    def test_compose_dataclasses(self):

        @dataclasses.dataclass
        class MyDataClassA:
            a: int = 1

            def __post_init__(self):
                self.a += 1

        @dataclasses.dataclass
        class MyDataClassB:
            b: str = '1'

        MyDataClassAB = compose_dataclasses(MyDataClassA, MyDataClassB)
        self.assertEqual(2, len(dataclasses.fields(MyDataClassAB)))
        self.assertEqual({'a': 3, 'b': '2'}, dataclasses.asdict(MyDataClassAB(a=2, b='2')))  # pylint: disable=unexpected-keyword-arg


    def test_instantiate_classes_dataclasses_lightning_issue_9207(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/9207

        @dataclasses.dataclass
        class MyDataClass:
            name: str = 'name'

        class MyClass:
            def __init__(self, data: MyDataClass):
                self.data = data

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass, 'class')
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['class'], MyClass)
        self.assertIsInstance(cfg['class'].data, MyDataClass)


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_attribute_docstrings(self):

        @dataclasses.dataclass
        class WithAttrDocs:
            attr_str: str = 'a'
            "attr_str description"
            attr_int: int = 1
            "attr_int description"

        set_docstring_parse_options(attribute_docstrings=True)

        parser = ArgumentParser()
        parser.add_class_arguments(WithAttrDocs)
        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('attr_str description (type: str, default: a)', help_str.getvalue())
        self.assertIn('attr_int description (type: int, default: 1)', help_str.getvalue())

        set_docstring_parse_options(attribute_docstrings=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
