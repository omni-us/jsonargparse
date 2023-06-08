from argparse import (
    ONE_OR_MORE,
    OPTIONAL,
    PARSER,
    REMAINDER,
    SUPPRESS,
    ZERO_OR_MORE,
    ArgumentError,
)

from .actions import *  # noqa: F403
from .cli import *  # noqa: F403
from .core import *  # noqa: F403
from .deprecated import *  # noqa: F403
from .formatters import *  # noqa: F403
from .jsonnet import *  # noqa: F403
from .jsonschema import *  # noqa: F403
from .link_arguments import *  # noqa: F403
from .loaders_dumpers import *  # noqa: F403
from .namespace import *  # noqa: F403
from .optionals import *  # noqa: F403
from .signatures import *  # noqa: F403
from .typehints import *  # noqa: F403
from .util import *  # noqa: F403

__all__ = [
    "ArgumentError",
    "OPTIONAL",
    "REMAINDER",
    "SUPPRESS",
    "PARSER",
    "ONE_OR_MORE",
    "ZERO_OR_MORE",
]


from . import (
    actions,
    cli,
    core,
    deprecated,
    formatters,
    jsonnet,
    jsonschema,
    link_arguments,
    loaders_dumpers,
    namespace,
    optionals,
    signatures,
    typehints,
    util,
)

__all__ += cli.__all__
__all__ += core.__all__
__all__ += signatures.__all__
__all__ += typehints.__all__
__all__ += link_arguments.__all__
__all__ += jsonschema.__all__
__all__ += jsonnet.__all__
__all__ += actions.__all__
__all__ += namespace.__all__
__all__ += formatters.__all__
__all__ += optionals.__all__
__all__ += loaders_dumpers.__all__
__all__ += util.__all__
__all__ += deprecated.__all__


__version__ = "4.21.2"
