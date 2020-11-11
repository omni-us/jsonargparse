#!/usr/bin/env python3

import os
import sys
import enum
import unittest
from typing import List
from io import BytesIO, StringIO
from contextlib import redirect_stdout, redirect_stderr
from jsonargparse import *
from jsonargparse.optionals import argcomplete_support, _import_argcomplete, jsonschema_support


@unittest.skipIf(not argcomplete_support, 'argcomplete package is required')
class ArgcompleteTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.orig_environ = os.environ.copy()
        self.argcomplete = _import_argcomplete('ArgcompleteTests')


    @classmethod
    def tearDownClass(self):
        os.environ.clear()
        os.environ.update(self.orig_environ)


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


    def test_ActionYesNo(self):
        self.parser.add_argument('--op1', type=bool)
        self.parser.add_argument('--op2', action=ActionYesNo)
        self.parser.add_argument('--with-op3', action=ActionYesNo(yes_prefix='with-', no_prefix='without-'))

        for arg, expected in [('--op1=', b'true\x0bfalse'),
                              ('--op2', b'--op2'),
                              ('--no_op2', b'--no_op2'),
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
        class MyEnum(enum.Enum):
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


    @unittest.skipIf(not jsonschema_support, 'jsonschema package is required')
    def test_ActionJsonSchema(self):
        self.parser.add_argument('--json', action=ActionJsonSchema(schema={'type': 'object'}))
        self.parser.add_argument('--list', type=List[int])

        os.environ['COMP_LINE'] = "tool.py --json=1"
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'')
        self.assertIn('value not yet valid', err.getvalue())

        os.environ['COMP_LINE'] = "tool.py --json='{\"a\": 1}'"
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'')
        self.assertIn('value already valid', err.getvalue())

        os.environ['COMP_LINE'] = "tool.py --list='[1, 2, 3]'"
        os.environ['COMP_POINT'] = str(len(os.environ['COMP_LINE']))

        out, err = BytesIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'')
        self.assertIn('value already valid, expected type List[int]', err.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
