"""Run all unit tests in package."""

import os
import sys
import warnings
from pathlib import Path

import pytest


def run_tests():
    filter_action = "default"
    warnings.simplefilter(filter_action)
    os.environ["PYTHONWARNINGS"] = filter_action
    testing_package = str(Path(__file__).parent)
    exit_code = pytest.main(["-v", "-s", f"--rootdir={testing_package}", "--pyargs", testing_package])
    if exit_code != 0:
        sys.exit(True)


def run_test_coverage():
    try:
        import coverage
    except ImportError:
        print("error: coverage package not found, run_test_coverage requires it.")
        sys.exit(True)
    package_source = os.path.dirname(__file__.replace("_tests", ""))
    cov = coverage.Coverage(source=[package_source])
    cov.start()
    run_tests()
    cov.stop()
    cov.save()
    cov.report()
    if "xml" in sys.argv:
        outfile = sys.argv[sys.argv.index("xml") + 1]
        cov.xml_report(outfile=outfile)
        print(f"\nSaved coverage report to {outfile}.")
    else:
        cov.html_report(directory="htmlcov")
        print("\nSaved html coverage report to htmlcov directory.")


if __name__ == "__main__":
    if "coverage" in sys.argv:
        run_test_coverage()
    else:
        run_tests()
