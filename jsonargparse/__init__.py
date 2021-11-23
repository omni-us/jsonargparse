from argparse import (
    OPTIONAL,
    REMAINDER,
    SUPPRESS,
    PARSER,
    ONE_OR_MORE,
    ZERO_OR_MORE,
)
from .actions import *
from .cli import *
from .core import *
from .deprecated import *
from .formatters import *
from .jsonnet import *
from .jsonschema import *
from .namespace import *
from .optionals import *
from .signatures import *
from .typehints import *
from .util import *


_all_ = [
    'OPTIONAL',
    'REMAINDER',
    'SUPPRESS',
    'PARSER',
    'ONE_OR_MORE',
    'ZERO_OR_MORE',
]

for module in ['cli',
               'core',
               'signatures',
               'typehints',
               'jsonschema',
               'jsonnet',
               'actions',
               'namespace',
               'formatters',
               'optionals',
               'util',
               'deprecated']:
    _all_.extend(getattr(__import__('jsonargparse.'+module, fromlist=['__all__']), '__all__'))

locals()['__all__'] = _all_   # Workaround because mypy does not handle dynamic __all__


__version__ = '4.0.3'
