from typing import TYPE_CHECKING

__all__ = [
    '_ArgumentGroup',
    'ArgumentParser',
    'ruyamlCommentedMap',
]

ArgumentParser = __import__('.core', fromlist=['ArgumentParser']) if TYPE_CHECKING else None
_ArgumentGroup = __import__('.core', fromlist=['_ArgumentGroup']) if TYPE_CHECKING else None

ruyamlCommentedMap = __import__('ruyaml.comments', fromlist=['CommentedMap']) if TYPE_CHECKING else None
