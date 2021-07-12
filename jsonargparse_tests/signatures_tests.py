#!/usr/bin/env python3

import calendar
import json
import platform
import yaml
from enum import Enum
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, Union
from jsonargparse_tests.base import *
from jsonargparse.actions import _find_action
from jsonargparse.util import _suppress_stderr


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
                    c1_a3: c1_a3 description
                """
                super().__init__()
                self.c1_a1 = c1_a1

            def __call__(self):
                return self.c1_a1

        class Class2(Class1):
            """Class2 short description

            Args:
                c1_a2: c1_a2 description
            """
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
        self.assertEqual(namespace_to_dict(cfg), {'c1_a2': 2.0,
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
                                                  'c3_a8': None})
        self.assertEqual([1, 2], parser.parse_args(['--c3_a0=0', '--c3_a5=[1,2]']).c3_a5)
        self.assertEqual({'k': 5.0}, namespace_to_dict(parser.parse_args(['--c3_a0=0', '--c3_a5={"k": 5.0}']).c3_a5))
        self.assertEqual(('3', 3, 3.0), parser.parse_args(['--c3_a0=0', '--c3_a7=["3", 3, 3.0]']).c3_a7)
        self.assertEqual('a', Class3(**namespace_to_dict(cfg))())

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

        ## Test no arguments added ##
        class NoValidArgs:
            def __init__(self, a0=None):
                pass

        self.assertEqual([], parser.add_class_arguments(NoValidArgs))

        def func(a1: Union[int, Dict[int, int]] = 1):
            pass

        parser = ArgumentParser()
        parser.add_function_arguments(func)
        parser.get_defaults()
        cfg = parser.parse_args(['--a1={"2": 7, "4": 9}'])
        self.assertEqual({2: 7, 4: 9}, parser.parse_args(['--a1={"2": 7, "4": 9}']).a1)


    def test_add_class_implemented_with_new(self):

        class ClassA:
            def __new__(cls, a1: int = 1, a2: float = 2.3):
                obj = object.__new__(cls)
                obj.a1 = a1
                obj.a2 = a2
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

        cfg = namespace_to_dict(parser.parse_args(['--m.a1=x', '--s.a1=y'], with_meta=False))
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

        cfg = namespace_to_dict(parser.parse_args(['--a1=x'], with_meta=False))
        self.assertEqual(cfg, {'a1': 'x', 'a2': 2.0, 'a3': False})
        self.assertEqual('x', func(**cfg))

        if docstring_parser_support:
            self.assertEqual('func short description', parser.groups['func'].title)
            for key in ['a1', 'a2']:
                self.assertEqual(key+' description', _find_action(parser, key).help)


    def test_add_subclass_arguments(self):
        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
        parser.add_subclass_arguments(calendar.Calendar, 'cal')

        cal = {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 1}}
        cfg = parser.parse_args(['--cal='+json.dumps(cal)])
        self.assertEqual(cfg['cal'], cal)

        cal['init_args']['firstweekday'] = 2
        cfg = parser.parse_args(['--cal.class_path=calendar.Calendar', '--cal.init_args.firstweekday=2'])
        self.assertEqual(cfg['cal'], cal)

        cal['init_args']['firstweekday'] = 3
        cfg = parser.parse_args(['--cal.class_path', 'calendar.Calendar', '--cal.init_args.firstweekday', '3'])
        self.assertEqual(cfg['cal'], cal)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal={"class_path":"not.exist.Class"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal={"class_path":"calendar.January"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--cal.help=calendar.January']))
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(calendar.January, 'jan'))

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            parser.parse_args(['--cal.help=calendar.Calendar'])
        self.assertIn('--cal.init_args.firstweekday', out.getvalue())

        if platform.python_implementation() == 'CPython':
            cal['init_args']['firstweekday'] = 4
            lazy_calendar = lazy_instance(calendar.Calendar, firstweekday=4)
            parser.set_defaults({'cal': lazy_calendar})
            cfg = parser.parse_string(parser.dump(parser.parse_args([])))
            self.assertEqual(cfg['cal'], cal)
            self.assertEqual(lazy_calendar.getfirstweekday(), 4)

            parser.add_argument('--config', action=ActionConfigFile)
            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                parser.parse_args(['--print_config'])
            self.assertIn('class_path: calendar.Calendar', out.getvalue())

        parser.set_defaults({'cal': calendar.Calendar(firstweekday=4)})
        cfg = parser.parse_args([])
        self.assertIsInstance(cfg['cal'], calendar.Calendar)
        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['cal'], calendar.Calendar)
        dump = parser.dump(cfg)
        self.assertIn('cal: <calendar.Calendar object at ', dump)


    def test_add_subclass_arguments_tuple(self):

        class ClassA:
            def __init__(self, a1: int = 1, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB:
            def __init__(self, b1: float = 4.5, b2: int = 6):
                self.b1 = b1
                self.b2 = b2

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'ClassA', ClassA)
        setattr(signatures_tests, 'ClassB', ClassB)
        class_path_a = 'jsonargparse_tests.signatures_tests.ClassA'
        class_path_b = 'jsonargparse_tests.signatures_tests.ClassB'

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
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(calendar.Calendar, None, required=True))
        parser.add_subclass_arguments(calendar.Calendar, 'cal', required=True)
        self.assertRaises(ParserError, lambda: parser.parse_args([]))


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

        parser = ArgumentParser(error_handler=None, parse_as_dict=True)
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

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'Class1', Class1)
        class_path = 'jsonargparse_tests.signatures_tests.Class1'

        cfg = parser.parse_args(['--c1='+class_path])
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

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'ClassA', ClassA)
        setattr(signatures_tests, 'ClassB', ClassB)
        class_path_a = 'jsonargparse_tests.signatures_tests.ClassA'
        class_path_b = 'jsonargparse_tests.signatures_tests.ClassB'

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
            def __init__(self, b1: str = '4', b2: ClassA = None):
                self.b1 = b1
                self.b2 = b2

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassB, 'b')
        cfg = parser.parse_args(['--b.b2={"a2": 6.7}'])
        self.assertEqual(cfg.b.b2, Namespace(a1=1, a2=6.7))
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['b'], ClassB)
        self.assertIsInstance(cfg['b'].b2, ClassA)

        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2={"bad": "value"}']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2="bad"']))
        self.assertRaises(ValueError, lambda: parser.add_subclass_arguments(ClassA, 'a'))


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


    def test_logger_debug(self):

        with _suppress_stderr():

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
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class2, skip={'c2_a2'})
                self.assertEqual(1, len(log.output))
                self.assertIn('"c2_a2" from "Class2"', log.output[0])
                self.assertIn('Parameter requested to be skipped', log.output[0])

            class Class3(Class1):
                def __init__(self, *args):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class3)
                self.assertEqual(1, len(log.output))
                self.assertIn('"c1_a2" from "Class1"', log.output[0])
                self.assertIn('Keyword parameter but **kwargs not propagated', log.output[0])

            class Class4(Class1):
                def __init__(self, **kwargs):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class4)
                self.assertEqual(1, len(log.output))
                self.assertIn('"c1_a1" from "Class1"', log.output[0])
                self.assertIn('Positional parameter but *args not propagated', log.output[0])


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

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'Class1', Class1)

        class_path = '"class_path": "jsonargparse_tests.signatures_tests.Class1"'
        init_args = '"init_args": {"a1": 7}'
        cfg = parser.parse_args(['--c1={'+class_path+', '+init_args+'}'])
        self.assertEqual(cfg.c1.class_path, 'jsonargparse_tests.signatures_tests.Class1')
        self.assertEqual(cfg.c1.init_args, Namespace(a1=7, a2=2.3))
        cfg = parser.instantiate_subclasses(cfg)
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
                pass

        with mock.patch('docstring_parser.parse') as docstring_parse:
            docstring_parse.side_effect = ValueError
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
                a1: calendar.Calendar,
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


    def test_link_arguments(self):

        class ClassA:
            def __init__(self, v1: int = 2, v2: int = 3):
                pass

        class ClassB:
            def __init__(self, v1: int = -1, v2: int = 4, v3: int = 2):
                """ClassB title
                Args:
                    v1: b v1 help
                """

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassA, 'a')
        parser.add_class_arguments(ClassB, 'b')
        parser.link_arguments('a.v2', 'b.v1')
        def add(*args):
            return sum(args)
        parser.link_arguments(('a.v1', 'a.v2'), 'b.v2', add)

        cfg = parser.parse_args([])
        self.assertEqual(cfg.b.v1, cfg.a.v2)
        self.assertEqual(cfg.b.v2, cfg.a.v1+cfg.a.v2)
        cfg = parser.parse_args(['--a.v1=11', '--a.v2=7'])
        self.assertEqual(7, cfg.b.v1)
        self.assertEqual(11+7, cfg.b.v2)
        self.assertEqual(Namespace(), parser.parse_args([], defaults=False))

        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.v1=5']))
        self.assertRaises(ValueError, lambda: parser.link_arguments('a.v2', 'b.v1'))
        self.assertRaises(ValueError, lambda: parser.link_arguments('x', 'b.v2'))
        self.assertRaises(ValueError, lambda: parser.link_arguments('a.v1', 'x'))
        self.assertRaises(ValueError, lambda: parser.link_arguments(('a.v1', 'a.v2'), 'b.v3'))
        self.assertRaises(ValueError, lambda: parser.link_arguments('a.v1', 'b.v2', apply_on='bad'))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('Linked arguments', help_str.getvalue())
        self.assertIn('b.v1 <-- a.v2', help_str.getvalue())
        self.assertIn('b.v2 <-- add(a.v1, a.v2)', help_str.getvalue())
        if docstring_parser_support:
            self.assertIn('b v1 help', help_str.getvalue())


    def test_link_arguments_subclasses(self):
        class ClassA:
            def __init__(
                self,
                v1: Union[int, str] = 1,
                v2: Union[int, str] = 2,
            ):
                pass

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'ClassA', ClassA)

        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(ClassA, 'a')
        parser.add_subclass_arguments(calendar.Calendar, 'c')

        def add(v1, v2):
            return v1 + v2
        parser.link_arguments(('a.init_args.v1', 'a.init_args.v2'), 'c.init_args.firstweekday', add)

        a_value = {
            'class_path': 'jsonargparse_tests.signatures_tests.ClassA',
            'init_args': {'v2': 3},
        }

        cfg = parser.parse_args(['--a='+json.dumps(a_value), '--c=calendar.Calendar'])
        self.assertEqual(cfg.c.init_args.firstweekday, 4)
        self.assertEqual(cfg.c.init_args.firstweekday, cfg.a.init_args.v1+cfg.a.init_args.v2)

        self.assertRaises(ValueError, lambda: parser.link_arguments('a.init_args.v1', 'c'))

        a_value['init_args'] = {'v1': 'a', 'v2': 'b'}
        with self.assertRaises(ParserError):
            parser.parse_args(['--a='+json.dumps(a_value), '--c=calendar.Calendar'])


    def test_link_arguments_apply_on_instantiate(self):

        class ClassA:
            def __init__(self, a1: int, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB:
            def __init__(self, b1: float = 4.5, b2: int = 6, b3: str = '7'):
                self.b1 = b1
                self.b2 = b2
                self.b3 = b3

        class ClassC:
            def __init__(self, c1: int = 7, c2: str = '8'):
                self.c1 = c1
                self.c2 = c2

        from jsonargparse_tests import signatures_tests
        setattr(signatures_tests, 'ClassA', ClassA)
        setattr(signatures_tests, 'ClassB', ClassB)

        def make_parser_1():
            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(ClassA, 'a')
            parser.add_class_arguments(ClassB, 'b')
            parser.add_class_arguments(ClassC, 'c')
            parser.add_argument('--d', type=int, default=-1)
            parser.link_arguments('b.b2', 'a.a1', apply_on='instantiate')
            parser.link_arguments('c.c1', 'b.b1', apply_on='instantiate')
            return parser

        def get_b2(obj_b):
            return obj_b.b2

        def make_parser_2():
            parser = ArgumentParser(error_handler=None)
            parser.add_subclass_arguments(ClassA, 'a')
            parser.add_subclass_arguments(ClassB, 'b')
            parser.link_arguments('b', 'a.init_args.a1', get_b2, apply_on='instantiate')
            return parser

        # Link object attribute
        parser = make_parser_1()
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertEqual(cfg['a'].a1, 6)

        # Link all group arguments
        parser = make_parser_1()
        parser.link_arguments('b.b1', 'a.a2', apply_on='instantiate')
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertEqual(cfg['a'].a1, 6)
        self.assertEqual(cfg['a'].a2, 7)

        # Links with cycle
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.a2', 'b.b3', apply_on='instantiate')
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.a2', 'c.c2', apply_on='instantiate')

        # Not subclass action or a class group
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('d', 'c.c2', apply_on='instantiate')

        # Link subclass and compute function
        parser = make_parser_2()
        cfg = parser.parse_args([
            '--a=jsonargparse_tests.signatures_tests.ClassA',
            '--b=jsonargparse_tests.signatures_tests.ClassB',
        ])
        cfg = parser.instantiate_classes(cfg)
        self.assertEqual(cfg['a'].a1, 6)

        # Unsupported multi-source
        parser = make_parser_2()
        with self.assertRaises(ValueError):
            parser.link_arguments(('b.b2', 'c.c1'), 'b.b3', compute_fn=lambda x, y: x, apply_on='instantiate')

        # Source must be subclass action or class group
        parser = make_parser_2()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.b.c', 'b.b3', apply_on='instantiate')

        # Object link without compute function
        parser = make_parser_2()
        with self.assertRaises(ValueError):
            parser.link_arguments('b', 'a.init_args.a2', apply_on='instantiate')

        # Unsupported deeper levels
        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassA, 'g.a')
        parser.add_class_arguments(ClassB, 'g.b')
        with self.assertRaises(ValueError):
            parser.link_arguments('g.b.b2', 'g.a.a1', apply_on='instantiate')


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
        self.assertEqual(strip_meta(cfg.func), {'a1': 'one', 'a2': 2.0, 'a3': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
