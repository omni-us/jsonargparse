#!/usr/bin/env python3

import yaml
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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
            """Description of Class1"""
            def __init__(self, i1: str):
                self.i1 = i1
            def method1(self, m1: int):
                """Description of method1"""
                return self.i1, m1

        self.assertEqual(('0', 2), CLI(Class1, args=['0', 'method1', '2']))
        self.assertEqual(('3', 4), CLI(Class1, args=['--config={"i1": "3", "method1": {"m1": 4}}']))
        self.assertEqual(('5', 6), CLI(Class1, args=['5', 'method1', '--config={"m1": 6}']))

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            CLI(Class1, args=['--config={"method1": {"m1": 2}}'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            CLI(Class1, args=['--config={"i1": "0", "method1": {"m1": "A"}}'])

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(Class1, args=['--help'])

        self.assertIn('i1', out.getvalue())
        if docstring_parser_support:
            self.assertIn('Description of Class1', out.getvalue())
            self.assertIn('Description of method1', out.getvalue())
        else:
            self.assertIn('.<locals>.Class1.method1', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(Class1, args=['x', 'method1', '--help'])

        self.assertIn('m1', out.getvalue())
        if docstring_parser_support:
            self.assertIn('Description of method1:', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(Class1, args=['0', 'method1', '2', '--print_config'])
        self.assertEqual('m1: 2\n', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(Class1, args=['--print_config', '0', 'method1', '2'])
        cfg = yaml.safe_load(out.getvalue())
        self.assertEqual(cfg, {'i1': '0', 'method1': {'m1': 2}})


    def test_function_and_class_cli(self):
        def cmd1(a1: int):
            """Description of cmd1"""
            return a1

        class Cmd2:
            def __init__(self, i1: str = 'd'):
                """Description of Cmd2"""
                self.i1 = i1
            def method1(self, m1: float):
                return self.i1, m1
            def method2(self, m2: int = 0):
                """Description of method2"""
                return self.i1, m2

        components = [cmd1, Cmd2]
        self.assertEqual(5, CLI(components, args=['cmd1', '5']))
        self.assertEqual(('d', 1.2), CLI(components, args=['Cmd2', 'method1', '1.2']))
        self.assertEqual(('b', 3), CLI(components, args=['Cmd2', '--i1=b', 'method2', '--m2=3']))
        self.assertEqual(4, CLI(components, args=['--config={"cmd1": {"a1": 4}}']))
        self.assertEqual(('a', 4.5), CLI(components, args=['--config={"Cmd2": {"i1": "a", "method1": {"m1": 4.5}}}']))
        self.assertEqual(('c', 6.7), CLI(components, args=['Cmd2', '--i1=c', 'method1', '--config={"m1": 6.7}']))
        self.assertEqual(('d', 8.9), CLI(components, args=['Cmd2', '--config={"method1": {"m1": 8.9}}']))

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['--help'])

        if docstring_parser_support:
            self.assertIn('Description of cmd1', out.getvalue())
            self.assertIn('Description of Cmd2', out.getvalue())
        else:
            self.assertIn('.<locals>.cmd1', out.getvalue())
            self.assertIn('.<locals>.Cmd2', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['Cmd2', '--help'])

        if docstring_parser_support:
            self.assertIn('Description of Cmd2:', out.getvalue())
            self.assertIn('Description of method2', out.getvalue())
        else:
            self.assertIn('.<locals>.Cmd2.method2', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['Cmd2', 'method2', '--help'])

        if docstring_parser_support:
            self.assertIn('Description of method2:', out.getvalue())
            self.assertIn('--m2 M2', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['Cmd2', 'method2', '--print_config'])
        self.assertEqual('m2: 0\n', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['Cmd2', '--print_config', 'method2'])
        self.assertEqual('i1: d\nmethod2:\n  m2: 0\n', out.getvalue())

        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            CLI(components, args=['--print_config', 'Cmd2', 'method2'])
        self.assertEqual('Cmd2:\n  i1: d\n  method2:\n    m2: 0\n', out.getvalue())

        if docstring_parser_support and ruyaml_support:
            out = StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit):
                CLI(components, args=['--print_config=comments', 'Cmd2', 'method2'])
            self.assertIn('# Description of Cmd2', out.getvalue())
            self.assertIn('# Description of method2', out.getvalue())


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


class CLITempDirTests(TempDirTestCase):

    def test_subclass_type_config_file(self):
        a_yaml = {
            'class_path': 'jsonargparse_tests.A',
            'init_args': {'p1': 'a yaml'}
        }

        with open('config.yaml', 'w') as f:
            f.write('a: a.yaml\n')
        with open('a.yaml', 'w') as f:
            f.write(yaml.safe_dump(a_yaml))

        class A:
            def __init__(self, p1: str = 'a default'):
                self.p1 = p1

        class B:
            def __init__(self, a: A = A()):
                self.a = a

        class C:
            def __init__(self, a: A = A(), b: B = None):
                self.a = a
                self.b = b
            def cmd_a(self):
                print(self.a.p1)
            def cmd_b(self):
                print(self.b.a.p1)

        import jsonargparse_tests
        setattr(jsonargparse_tests, 'A', A)

        out = StringIO()
        with redirect_stdout(out):
            CLI(C, args=['--config=config.yaml', 'cmd_a'])
        self.assertEqual('a yaml\n', out.getvalue())

        #with open('config.yaml', 'w') as f:
        #    f.write('a: a.yaml\nb: b.yaml\n')
        #with open('b.yaml', 'w') as f:
        #    f.write('a: a.yaml\n')

        #out = StringIO()
        #with redirect_stdout(out):
        #    CLI(C, args=['--config=config.yaml', 'cmd_b'])
        #self.assertEqual('a yaml\n', out.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
