from typing import TYPE_CHECKING

__all__ = [
    "ArgumentGroup",
    "ActionsContainer",
    "ArgumentParser",
    "docstring_parser",
    "ruyamlCommentedMap",
]

if TYPE_CHECKING:  # pragma: no cover
    import docstring_parser
    from ruyaml.comments import CommentedMap as ruyamlCommentedMap

    from ._core import ActionsContainer, ArgumentGroup, ArgumentParser
else:
    globals().update({k: None for k in __all__})
