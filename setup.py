#!/usr/bin/env python3
"""Setup file for jsonargparse package."""

from setuptools import setup, Command
import re


NAME = next(filter(lambda x: x.startswith('name = '), open('setup.cfg').readlines())).strip().split()[-1]
NAME_TEST = NAME+'_test'
LONG_DESCRIPTION = re.sub(':class:|:func:|:ref:', '', open('README.rst').read())
CMDCLASS = {}


## test_coverage target ##
try:
    import coverage

    class CoverageCommand(Command):
        description = 'print test coverage report'
        user_options = []  # type: ignore
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self):
            cov = coverage.Coverage()
            cov.start()
            __import__(NAME_TEST).run_tests()
            cov.stop()
            cov.save()
            cov.report(show_missing=True, include=[NAME+'*'], omit=['*_test*'])

    CMDCLASS['test_coverage'] = CoverageCommand

except Exception:
    print('warning: coverage package not found, test_coverage target will not be available.')


## build_sphinx target ##
try:
    from sphinx.setup_command import BuildDoc
    CMDCLASS['build_sphinx'] = BuildDoc

except Exception:
    print('warning: sphinx package not found, build_sphinx target will not be available.')


## Run setuptools setup ##
setup(version=__import__(NAME).__version__,
      long_description=LONG_DESCRIPTION,
      py_modules=[NAME, NAME_TEST],
      test_suite=NAME_TEST,
      cmdclass=CMDCLASS)
