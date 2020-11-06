"""Run all unit tests in package."""

import os
import sys
import unittest


testing_package = os.path.basename(os.path.dirname(os.path.realpath(__file__)))


def run_tests():
    pattern = '*_tests.py' if sys.version_info.minor < 6 else '*_tests*.py'
    tests = unittest.defaultTestLoader.discover(testing_package, pattern=pattern)
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        sys.exit(True)


def run_test_coverage():
    try:
        import coverage
    except:
        print('error: coverage package not found, run_test_coverage requires it.')
        sys.exit(True)
    package_source = os.path.dirname(__file__.replace('_tests', ''))
    cov = coverage.Coverage(source=[package_source])
    cov.start()
    run_tests()
    cov.stop()
    cov.save()
    cov.report()
    if 'xml' in sys.argv:
        outfile = sys.argv[sys.argv.index('xml')+1]
        cov.xml_report(outfile=outfile)
        print('\nSaved coverage report to '+outfile+'.')
    else:
        cov.html_report(directory='htmlcov')
        print('\nSaved html coverage report to htmlcov directory.')


if __name__ == '__main__':
    if 'coverage' in sys.argv:
        run_test_coverage()
    else:
        run_tests()
