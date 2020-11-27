#!/usr/bin/env python3

from jsonargparse_tests.base import *


class CLITests(unittest.TestCase):

    def test_single_function_cli(self):
        def function(a1: float):
            return a1

        self.assertEqual(1.2, CLI(function, args=['1.2']))


    def test_multiple_functions_cli(self):
        def cmd1(a1: int):
            return a1

        def cmd2(a2: str = 'X'):
            return a2

        functions = [cmd1, cmd2]
        self.assertEqual(5, CLI(functions, args=['cmd1', '5']))
        self.assertEqual('Y', CLI(functions, args=['cmd2', '--a2=Y']))


    def test_empty_context(self):
        def empty_context():
            CLI()

        self.assertRaises(ValueError, empty_context)


    def test_non_empty_context(self):
        def non_empty_context():
            def function(a1: float):
                return a1

            return CLI(args=['6.7'])

        self.assertEqual(6.7, non_empty_context())


if __name__ == '__main__':
    unittest.main(verbosity=2)
