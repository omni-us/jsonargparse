import os
import shutil
import tempfile
import unittest
from unittest import mock
from jsonargparse import *
from jsonargparse.typing import *
from jsonargparse.optionals import (
    jsonschema_support, import_jsonschema,
    jsonnet_support, import_jsonnet,
    url_support, import_url_validator, import_requests,
    docstring_parser_support, import_docstring_parse,
    argcomplete_support, import_argcomplete,
    dataclasses_support, import_dataclasses,
    fsspec_support, import_fsspec,
    get_config_read_mode,
    ruyaml_support, import_ruyaml,
    ModuleNotFound,
)


try:
    import responses
    responses_activate = responses.activate
except (ImportError, ModuleNotFound):
    def nothing_decorator(func):
        return func
    responses = False
    responses_activate = nothing_decorator


os.environ['COLUMNS'] = '150'


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix='_jsonargparse_test_'))
        os.chdir(self.tmpdir)


    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)


def example_parser():
    """Creates a simple parser for doing tests."""
    parser = ArgumentParser(prog='app', default_meta=False, error_handler=None)

    group_one = parser.add_argument_group('Group 1', name='group1')
    group_one.add_argument('--bools.def_false',
        default=False,
        nargs='?',
        action=ActionYesNo)
    group_one.add_argument('--bools.def_true',
        default=True,
        nargs='?',
        action=ActionYesNo)

    group_two = parser.add_argument_group('Group 2', name='group2')
    group_two.add_argument('--lev1.lev2.opt1',
        default='opt1_def')
    group_two.add_argument('--lev1.lev2.opt2',
        default='opt2_def')

    group_three = parser.add_argument_group('Group 3')
    group_three.add_argument('--nums.val1',
        type=int,
        default=1)
    group_three.add_argument('--nums.val2',
        type=float,
        default=2.0)

    return parser


example_yaml = '''
lev1:
  lev2:
    opt1: opt1_yaml
    opt2: opt2_yaml

nums:
  val1: -1
'''

example_env = {
    'APP_LEV1__LEV2__OPT1': 'opt1_env',
    'APP_NUMS__VAL1': '0'
}
