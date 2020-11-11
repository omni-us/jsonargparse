#!/usr/bin/env python3

import enum
import unittest
from typing import Dict, List, Tuple, Optional, Union, Any
from jsonargparse import *
from jsonargparse.typing import PositiveInt, PositiveFloat, OpenUnitInterval
from jsonargparse.actions import _find_action
from jsonargparse.optionals import jsonschema_support, docstring_parser_support
from jsonargparse.util import _suppress_stderr


class SignaturesTests(unittest.TestCase):

    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
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
                         c1_a4: int = 4):
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

        for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a3', 'c3_a4', 'c3_a5', 'c3_a7', 'c1_a2', 'c1_a4']:
            self.assertIsNotNone(_find_action(parser, key), key+' should be in parser but is not')
        for key in ['c3_a6', 'c2_a0', 'c1_a1', 'c1_a3', 'c0_a0']:
            self.assertIsNone(_find_action(parser, key), key+' should not be in parser but is')

        cfg = parser.parse_args(['--c3_a0=0', '--c3_a3=true', '--c3_a4=a'], with_meta=False)
        self.assertEqual(namespace_to_dict(cfg), {'c1_a2': 2.0,
                                                  'c1_a4': 4,
                                                  'c3_a0': 0,
                                                  'c3_a1': '1',
                                                  'c3_a2': 2.0,
                                                  'c3_a3': True,
                                                  'c3_a4': 'a',
                                                  'c3_a5': 5,
                                                  'c3_a7': ('7', 7, 7.0)})
        self.assertEqual([1, 2], parser.parse_args(['--c3_a0=0', '--c3_a5=[1,2]']).c3_a5)
        self.assertEqual({'k': 5.0}, namespace_to_dict(parser.parse_args(['--c3_a0=0', '--c3_a5={"k": 5.0}']).c3_a5))
        self.assertEqual(('3', 3, 3.0), parser.parse_args(['--c3_a0=0', '--c3_a7=["3", 3, 3.0]']).c3_a7)
        self.assertEqual('a', Class3(**namespace_to_dict(cfg))())

        self.assertRaises(ParserError, lambda: parser.parse_args([]))  # c3_a0 is required
        self.assertRaises(ParserError, lambda: parser.parse_args(['--c3_a0=0', '--c3_a4=4.0']))  # c3_a4 is str or None
        self.assertRaises(ParserError, lambda: parser.parse_args(['--c3_a0=0', '--c3_a7=["3", "3", 3.0]']))  # tuple[1] is int

        if docstring_parser_support:
            self.assertEqual('Class3 short description', parser.groups['Class3'].title)
            for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a4', 'c3_a5', 'c1_a2']:
                self.assertEqual(key+' description', _find_action(parser, key).help)
            for key in ['c3_a3', 'c3_a7', 'c1_a4']:
                self.assertIsNone(_find_action(parser, key).help, 'expected help for '+key+' to be None')

        ## Test nested and as_group=False ##
        parser = ArgumentParser()
        parser.add_class_arguments(Class3, 'g', as_group=False)

        self.assertNotIn('g', parser.groups)

        for key in ['c3_a0', 'c3_a1', 'c3_a2', 'c3_a3', 'c3_a4', 'c3_a5', 'c3_a7', 'c1_a2', 'c1_a4']:
            self.assertIsNotNone(_find_action(parser, 'g.'+key), key+' should be in parser but is not')
        for key in ['c3_a6', 'c2_a0', 'c1_a1', 'c1_a3', 'c0_a0']:
            self.assertIsNone(_find_action(parser, 'g.'+key), key+' should not be in parser but is')

        ## Test default group title ##
        parser = ArgumentParser()
        parser.add_class_arguments(Class0)
        self.assertEqual(str(Class0), parser.groups['Class0'].title)

        ## Test positional without type ##
        self.assertRaises(ValueError, lambda: parser.add_class_arguments(Class2))

        ## Test no arguments added ##
        class NoValidArgs:
            def __init__(self, a0=None, a1: Optional[Class1] = None):
                pass

        self.assertEqual(0, parser.add_class_arguments(NoValidArgs))

        def func(a1: Union[int, Dict[int, int]] = 1):
            pass

        parser = ArgumentParser()
        parser.add_function_arguments(func)
        parser.get_defaults()
        cfg = parser.parse_args(['--a1={"2": 7, "4": 9}'])
        self.assertEqual({2: 7, 4: 9}, parser.parse_args(['--a1={"2": 7, "4": 9}']).a1)


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
        parser.add_method_arguments(MyClass, 'mymethod', 'm')
        parser.add_method_arguments(MyClass, 'mystaticmethod', 's')

        self.assertRaises(ValueError, lambda: parser.add_method_arguments('MyClass', 'mymethod'))
        self.assertRaises(ValueError, lambda: parser.add_method_arguments(MyClass, 'mymethod3'))

        self.assertIn('m', parser.groups)
        self.assertIn('s', parser.groups)

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
                self.assertIsNone(_find_action(parser, key).help, 'expected help for '+key+' to be None')


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
            self.assertIsNone(_find_action(parser, 'a3').help, 'expected help for a3 to be None')


    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
    def test_optional_enum(self):

        class MyEnum(enum.Enum):
            A = 1
            B = 2
            C = 3

        def func(a1: Optional[MyEnum] = None):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func)
        self.assertEqual(MyEnum.B, parser.parse_args(['--a1=B']).a1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a1=D']))

        class MyEnum2(str, enum.Enum):
            A = 'A'
            B = 'B'

        def func2(a1: Optional[MyEnum2] = None):
            return a1

        parser = ArgumentParser(error_handler=None)
        parser.add_function_arguments(func2)
        self.assertEqual(MyEnum2.B, parser.parse_args(['--a1=B']).a1)
        self.assertEqual('B', parser.parse_args(['--a1=B']).a1)


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


    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
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


    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
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
                            c2_a3: Optional[Class1] = None,
                            **kwargs):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class2, skip={'c2_a2'})
                self.assertEqual(2, len(log.output))
                self.assertIn('"c2_a2" from "Class2"', log.output[0])
                self.assertIn('Argument requested to be skipped', log.output[0])
                self.assertIn('"c2_a3" from "Class2"', log.output[1])
                self.assertIn('Unable to generate schema', log.output[1])

            class Class3(Class1):
                def __init__(self, *args):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class3)
                self.assertEqual(1, len(log.output))
                self.assertIn('"c1_a2" from "Class1"', log.output[0])
                self.assertIn('Keyword argument but **kwargs not propagated', log.output[0])

            class Class4(Class1):
                def __init__(self, **kwargs):
                    pass

            parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
            with self.assertLogs(level='DEBUG') as log:
                parser.add_class_arguments(Class4)
                self.assertEqual(1, len(log.output))
                self.assertIn('"c1_a1" from "Class1"', log.output[0])
                self.assertIn('Positional argument but *args not propagated', log.output[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
