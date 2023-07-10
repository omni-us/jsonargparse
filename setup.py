#!/usr/bin/env python3

import re
from pathlib import Path

from setuptools import setup

## Use README.rst for the package long description ##
LONG_DESCRIPTION = re.sub(
    ":class:|:func:|:ref:|:py:meth:|:py:mod:|py:attr:| *# doctest:.*",
    "",
    Path("README.rst").read_text(),
)
LONG_DESCRIPTION = re.sub("([+|][- ]{12})[- ]{5}", r"\1", LONG_DESCRIPTION)

LONG_DESCRIPTION_LINES = []
skip_line = False
lines = LONG_DESCRIPTION.split("\n")
for num, line in enumerate(lines):
    if any(line.startswith(".. " + x) for x in ["doctest:: :hide:", "testsetup::", "testcleanup::"]) or (
        line.startswith(".. testcode::") and ":hide:" in lines[num + 1]
    ):
        skip_line = True
    elif skip_line and line != "" and not line.startswith(" "):
        skip_line = False
    if not skip_line:
        LONG_DESCRIPTION_LINES.append(line)
LONG_DESCRIPTION = "\n".join(LONG_DESCRIPTION_LINES)
LONG_DESCRIPTION = re.sub("(testcode::|doctest::).*", "code-block:: python", LONG_DESCRIPTION)


## Run setuptools setup ##
setup(
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
)
