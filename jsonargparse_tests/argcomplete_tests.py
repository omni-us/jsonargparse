#!/usr/bin/env python3

import os
import sys
import unittest
from io import BytesIO
from contextlib import redirect_stdout
from jsonargparse import *
from jsonargparse.optionals import argcomplete_support, _import_argcomplete


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
        os.environ['COMP_TYPE'] = '9'
        self.parser = ArgumentParser()


    def test_complete_nested_one_option(self):
        os.environ['COMP_POINT'] = '16'
        os.environ['COMP_LINE'] = 'tool.py --group1'
        self.parser.add_argument('--group1.op')

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'--group1.op')


    def test_complete_nested_two_options(self):
        os.environ['COMP_POINT'] = '16'
        os.environ['COMP_LINE'] = 'tool.py --group2'
        self.parser.add_argument('--group2.op1')
        self.parser.add_argument('--group2.op2')

        out = BytesIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            self.argcomplete.autocomplete(self.parser, exit_method=sys.exit, output_stream=sys.stdout)
        self.assertEqual(out.getvalue(), b'--group2.op1\x0b--group2.op2')


if __name__ == '__main__':
    unittest.main(verbosity=2)
