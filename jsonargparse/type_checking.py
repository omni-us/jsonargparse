from typing import TYPE_CHECKING

__all__ = [
    '_ArgumentGroup',
    'ArgumentParser',
    'ruyamlCommentedMap',
]

if TYPE_CHECKING:  # pragma: no cover
    from .core import ArgumentParser, _ArgumentGroup
    from ruyaml.comments import CommentedMap as ruyamlCommentedMap
else:
    globals().update({k: None for k in __all__})
