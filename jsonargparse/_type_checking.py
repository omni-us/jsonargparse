from typing import TYPE_CHECKING

__all__ = [
    "_ArgumentGroup",
    "ArgumentParser",
    "ruyamlCommentedMap",
]

if TYPE_CHECKING:  # pragma: no cover
    from ruyaml.comments import CommentedMap as ruyamlCommentedMap

    from ._core import ArgumentParser, _ArgumentGroup
else:
    globals().update({k: None for k in __all__})
