import sys
from importlib.util import find_spec

if sys.version_info[:2] >= (3, 8):
    from typing import final
elif find_spec('typing_extensions'):
    from typing_extensions import final
