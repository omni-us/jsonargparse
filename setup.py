#!/usr/bin/env python3

from setuptools import setup, Command
import re


NAME_TESTS = next(filter(lambda x: x.startswith('test_suite = '), open('setup.cfg').readlines())).strip().split()[-1]
LONG_DESCRIPTION = re.sub(':class:|:func:|:ref:|:py:meth:', '', open('README.rst').read())
CMDCLASS = {}


## test_coverage target ##
class CoverageCommand(Command):
    description = 'run test coverage and generate html report'
    user_options = []  # type: ignore
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        __import__(NAME_TESTS+'.__main__').__main__.run_test_coverage()

CMDCLASS['test_coverage'] = CoverageCommand


## build_sphinx target ##
try:
    from sphinx.setup_command import BuildDoc
    CMDCLASS['build_sphinx'] = BuildDoc  # type: ignore
except:
    print('warning: sphinx package not found, build_sphinx target will not be available.')


## Run setuptools setup ##
setup(long_description=LONG_DESCRIPTION,
      cmdclass=CMDCLASS)
