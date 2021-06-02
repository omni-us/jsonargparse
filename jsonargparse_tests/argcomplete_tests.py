#!/usr/bin/env python3

import sys
import pathlib
import platform
from enum import Enum
from typing import List, Optional
from io import BytesIO, StringIO
from contextlib import redirect_stdout, redirect_stderr
from jsonargparse_tests.base import *


@unittest.skipIf(not argcomplete_support, 'argcomplete package is required')
class ArgcompleteTests(TempDirTestCase):

    @classmethod
    def setUpClass(cls):
        cls.orig_environ = os.environ.copy()
        cls.argcomplete = import_argcomplete('ArgcompleteTests')


    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls.orig_environ)


    def setUp(self):
        super().setUp()
        self.tearDownClass()
        os.environ['_ARGCOMPLETE'] = '1'
        os.environ['_ARGCOMPLETE_SUPPRESS_SPACE'] = '1'
        os.environ['_ARGCOMPLETE_COMP_WORDBREAKS'] = " \t\n\"'><=;|&(:"
        os.environ['COMP_TYPE'] = str(ord('?'))   # ='63'  str(ord('\t'))='9'
        self.parser = ArgumentParser(error_handler=lambda x: x.exit(2))


    def test_complete_nested_one_option(self):
        self.parser.add_argument('--group1.op')

        os.environ['COMP_LINE'] = 'tool.py --group1'
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'--group1.op')


    def test_complete_nested_two_options(self):
        self.parser.add_argument('--group2.op1')
        self.parser.add_argument('--group2.op2')

        os.environ['COMP_LINE'] = 'tool.py --group2'
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'--group2.op1\x0b--group2.op2')


    @unittest.skipIf(platform.python_implementation() != 'CPython', 'only CPython supported')
    def test_TypeCastCompleterMethod(self):
        self.parser.add_argument('--int', type=int)
        self.parser.add_argument('--float', type=float)
        self.parser.add_argument('--pint', type=PositiveInt)
        self.parser.add_argument('--pfloat', type=PositiveFloat)
        self.parser.add_argument('--email', type=Email)

        for arg, expected in [('--int=a',       'value not yet valid, expected type int'),
                              ('--int=1',       'value already valid, expected type int'),
                              ('--float=a',     'value not yet valid, expected type float'),
                              ('--float=1',     'value already valid, expected type float'),
                              ('--pint=0',      'value not yet valid, expected type PositiveInt'),
                              ('--pint=1',      'value already valid, expected type PositiveInt'),
                              ('--pfloat=0',    'value not yet valid, expected type PositiveFloat'),
                              ('--pfloat=1',    'value already valid, expected type PositiveFloat'),
                              ('--email=a',     'value not yet valid, expected type Email'),
                              ('--email=a@b.c', 'value already valid, expected type Email')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out, err = BytesIO(), StringIO()
                with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(out.getvalue(), b'')
                self.assertIn(expected, err.getvalue())


    @unittest.skipIf(os.name != 'posix', 'Path class currently only supported in posix systems')
    def test_ActionConfigFile(self):
        self.parser.add_argument('--cfg', action=ActionConfigFile)
        pathlib.Path('file1').touch()
        pathlib.Path('config.yaml').touch()

        for arg, expected in [('--cfg=',  b'config.yaml\x0bfile1'),
                              ('--cfg=c', b'config.yaml')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out = BytesIO()
                with redirect_stdout(out), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(expected, out.getvalue())


    def test_ActionYesNo(self):
        self.parser.add_argument('--op1', action=ActionYesNo)
        self.parser.add_argument('--op2', nargs='?', action=ActionYesNo)
        self.parser.add_argument('--with-op3', action=ActionYesNo(yes_prefix='with-', no_prefix='without-'))

        for arg, expected in [('--op1', b'--op1'),
                              ('--no_op1', b'--no_op1'),
                              ('--op2', b'--op2'),
                              ('--no_op2', b'--no_op2'),
                              ('--op2=', b'true\x0bfalse\x0byes\x0bno'),
                              ('--with-op3', b'--with-op3'),
                              ('--without-op3', b'--without-op3')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out = BytesIO()
                with redirect_stdout(out), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(expected, out.getvalue())


    def test_ActionEnum(self):
        class MyEnum(Enum):
            abc = 1
            xyz = 2
            abd = 3

        self.parser.add_argument('--enum', type=MyEnum)

        os.environ['COMP_LINE'] = 'tool.py --enum=ab'
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'abc\x0babd')


    def test_optional(self):
        class MyEnum(Enum):
            A = 1
            B = 2

        self.parser.add_argument('--enum', type=Optional[MyEnum])
        self.parser.add_argument('--bool', type=Optional[bool])

        for arg, expected in [('--enum=', b'A\x0bB\x0bnull'),
                              ('--bool=', b'true\x0bfalse\x0bnull')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out = BytesIO()
                with redirect_stdout(out), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(expected, out.getvalue())


    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
    @unittest.skipIf(platform.python_implementation() != 'CPython', 'only CPython supported')
    def test_json(self):
        self.parser.add_argument('--json', action=ActionJsonSchema(schema={'type': 'object'}))

        for arg, expected in [('--json=1',            'value not yet valid'),
                              ("--json='{\"a\": 1}'", 'value already valid')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out, err = BytesIO(), StringIO()
                with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(out.getvalue(), b'')
                self.assertIn(expected, err.getvalue())

                with mock.patch('os.popen') as popen_mock:
                    popen_mock.side_effect = ValueError
                    with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
                        self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                    self.assertEqual(out.getvalue(), b'')
                    self.assertIn(expected, err.getvalue())

        os.environ['COMP_LINE'] = 'tool.py --json='
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(err.getvalue(), '')
        self.assertIn('value not yet valid', out.getvalue().decode('utf-8').replace('\xa0', ' ').replace('_', ' '))


    def test_list(self):
        self.parser.add_argument('--list', type=List[int])

        os.environ['COMP_LINE'] = "tool.py --list='[1, 2, 3]'"
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'')
        self.assertIn('value already valid, expected type List[int]', err.getvalue())

        os.environ['COMP_LINE'] = 'tool.py --list='
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(err.getvalue(), '')
        self.assertIn('value not yet valid', out.getvalue().decode('utf-8').replace('\xa0', ' ').replace('_', ' '))


    def test_bool(self):
        self.parser.add_argument('--bool', type=bool)
        os.environ['COMP_LINE'] = 'tool.py --bool='
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(b'true\x0bfalse', out.getvalue())


    @unittest.skipIf(os.name != 'posix', 'Path class currently only supported in posix systems')
    def test_optional_path(self):
        self.parser.add_argument('--path', type=Optional[Path_fr])
        pathlib.Path('file1').touch()
        pathlib.Path('file2').touch()

        for arg, expected in [('--path=',  b'null\x0bfile1\x0bfile2'),
                              ('--path=n', b'null'),
                              ('--path=f', b'file1\x0bfile2')]:
            os.environ['COMP_LINE'] = 'tool.py '+arg
            os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

            with self.subTest(os.environ['COMP_LINE']):
                out = BytesIO()
                with redirect_stdout(out), self.assertRaises(SystemExit):
                    self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
                self.assertEqual(expected, out.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
