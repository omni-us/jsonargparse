import os
import shutil
import tempfile
import unittest
from contextlib import suppress

from jsonargparse.optionals import docstring_parser_support, set_docstring_parse_options

if docstring_parser_support:
    from docstring_parser import DocstringStyle

    set_docstring_parse_options(style=DocstringStyle.GOOGLE)


os.environ["COLUMNS"] = "150"


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix="_jsonargparse_test_"))
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.cwd)
        with suppress(PermissionError):
            shutil.rmtree(self.tmpdir)
