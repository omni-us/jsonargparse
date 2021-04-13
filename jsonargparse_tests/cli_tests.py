#!/usr/bin/env python3

from jsonargparse_tests.base import *


class CLITests(unittest.TestCase):

    def test_single_function_cli(self):
        def function(a1: float):
            return a1

        self.assertEqual(1.2, CLI(function, args=['1.2']))
        parser = CLI(function, return_parser=True, set_defaults={'a1': 3.4})
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual(3.4, parser.get_defaults().a1)


    def test_multiple_functions_cli(self):
        def cmd1(a1: int):
            return a1

        def cmd2(a2: str = 'X'):
            return a2

        functions = [cmd1, cmd2]
        self.assertEqual(5, CLI(functions, args=['cmd1', '5']))
        self.assertEqual('Y', CLI(functions, args=['cmd2', '--a2=Y']))
        parser = CLI(functions, return_parser=True, set_defaults={'cmd2.a2': 'Z'})
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual('Z', parser.parse_args(['cmd2'])['cmd2']['a2'])


    def test_single_class_cli(self):
        class Class1:
            def __init__(self, i1: str):
                self.i1 = i1
            def method1(self, m1: int):
                return self.i1, m1

        self.assertEqual(('0', 2), CLI(Class1, args=['0', 'method1', '2']))


    def test_function_and_class_cli(self):
        def cmd1(a1: int):
            return a1

        class Cmd2:
            def __init__(self, i1: str = 'd'):
                self.i1 = i1
            def method1(self, m1: float):
                return self.i1, m1
            def method2(self, m1: int = 0):
                return self.i1, m1

        components = [cmd1, Cmd2]
        self.assertEqual(5, CLI(components, args=['cmd1', '5']))
        self.assertEqual(('d', 1.2), CLI(components, args=['Cmd2', 'method1', '1.2']))
        self.assertEqual(('b', 3), CLI(components, args=['Cmd2', '--i1=b', 'method2', '--m1=3']))


    def test_empty_context(self):
        def empty_context():
            CLI()

        self.assertRaises(ValueError, empty_context)


    def test_non_empty_context(self):
        def non_empty_context_1():
            def function(a1: float):
                return a1

            return CLI(args=['6.7'])

        def non_empty_context_2():
            class Class1:
                def __init__(self, i1: str):
                    self.i1 = i1
                def method1(self, m1: int):
                    return self.i1, m1

            return CLI(args=['a', 'method1', '2'])

        self.assertEqual(6.7, non_empty_context_1())
        self.assertEqual(('a', 2), non_empty_context_2())


if __name__ == '__main__':
    unittest.main(verbosity=2)
