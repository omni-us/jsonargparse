#!/usr/bin/env python
"""Setup file for yamlargparse package."""

from setuptools import setup, find_packages, Command
import subprocess
from glob import glob
import re

try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    BuildDoc = False
    print('warning: sphinx not found, build_sphinx target will not be available.')

from yamlargparse import __version__

NAME = 'yamlargparse'
DESCRIPTION = 'Parsing of command line options, yaml config files and/or environment variables based on argparse.'
LONG_DESCRIPTION = open('README.rst').read()


class CoverageCommand(Command):
    """Custom command to print test coverage report."""
    description = 'print test coverage report'
    user_options = []

    def initialize_options(self):
        """init options"""
        pass

    def finalize_options(self):
        """finalize options"""
        pass

    def run(self):
        """run commands"""
        subprocess.check_call(['python', '-m', 'coverage', 'run', '--source', NAME, 'setup.py', 'test'])
        subprocess.check_call(['python', '-m', 'coverage', 'report', '-m'])


CMDCLASS = {'test_coverage': CoverageCommand}
if BuildDoc:
    CMDCLASS['build_sphinx'] = BuildDoc


def get_runtime_requirements():
    """Returns a list of required packages filtered to include only the ones necessary at runtime."""
    with open('requirements.txt') as f:
        requirements = f.readlines()
    requirements = [x.strip() for x in requirements]
    regex = re.compile(r'^coverage|^Sphinx', re.IGNORECASE)
    return [x for x in requirements if not regex.match(x)]


setup(name=NAME,
      version=__version__,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author='Mauricio Villegas',
      author_email='mauricio@omnius.com',
      url='https://omni-us.github.io/yamlargparse/html/',
      license='MIT',
      py_modules=[NAME, NAME+'_tests'],
      install_requires=get_runtime_requirements(),
      test_suite=NAME+'_tests',
      cmdclass=CMDCLASS,
      command_options={
          'build_sphinx': {
              'project': ('setup.py', NAME),
              'version': ('setup.py', __version__),
              'release': ('setup.py', __version__),
              'build_dir': ('setup.py', 'docs/_build'),
              'source_dir': ('setup.py', 'docs')}})
