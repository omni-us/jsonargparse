import sys
from importlib.util import find_spec

if sys.version_info[:2] >= (3, 8):
    pass
elif find_spec('typing_extensions'):
    pass
