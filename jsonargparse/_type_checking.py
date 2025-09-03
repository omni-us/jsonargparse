from typing import TYPE_CHECKING

__all__ = [
    "ArgumentGroup",
    "ActionsContainer",
    "ArgumentParser",
    "docstring_parser",
    "ruamelCommentedMap",
]

if TYPE_CHECKING:  # pragma: no cover
    import docstring_parser
    from ruamel.yaml.comments import CommentedMap as ruamelCommentedMap

    from ._core import ActionsContainer, ArgumentGroup, ArgumentParser
else:
    globals().update({k: None for k in __all__})
