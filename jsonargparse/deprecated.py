"""Deprecated code."""

from enum import Enum
from .optionals import get_config_read_mode, set_config_read_mode
from .typehints import ActionTypeHint
from .typing import path_type, restricted_number_type, registered_types
from .util import _issubclass


__all__ = [
    'ActionEnum',
    'ActionOperators',
    'ActionPath',
    'set_url_support',
]


class ActionEnum:
    """DEPRECATED: An action based on an Enum that maps to-from strings and enum values.

    Enums now should be given directly as a type.
    """

    def __init__(self, **kwargs):
        if 'enum' in kwargs:
            if not _issubclass(kwargs['enum'], Enum):
                raise ValueError('Expected enum to be an subclass of Enum.')
            self._type = kwargs['enum']
        else:
            raise ValueError('Expected enum keyword argument.')

    def __call__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('ActionEnum does not allow type given to add_argument.')
        return ActionTypeHint(typehint=self._type)(**kwargs)


class ActionOperators:
    """DEPRECATED: Action to restrict a value with comparison operators.

    The new alternative is explained in :ref:`restricted-numbers`.
    """

    def __init__(self, **kwargs):
        if 'expr' in kwargs:
            restrictions = [kwargs['expr']] if isinstance(kwargs['expr'], tuple) else kwargs['expr']
            register_key = (tuple(sorted(restrictions)), kwargs.get('type', int), kwargs.get('join', 'and'))
            if register_key in registered_types:
                self._type = registered_types[register_key]
            else:
                self._type = restricted_number_type(None, kwargs.get('type', int), kwargs['expr'], kwargs.get('join', 'and'))
        else:
            raise ValueError('Expected expr keyword argument.')

    def __call__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('ActionOperators does not allow type given to add_argument.')
        return ActionTypeHint(typehint=self._type)(**kwargs)


class ActionPath:
    """DEPRECATED: Action to check and store a path.

    Paths now should be given directly as a type.
    """

    def __init__(
        self,
        mode: str,
        skip_check: bool = False,
    ):
        self._type = path_type(mode, skip_check=skip_check)

    def __call__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('ActionPath does not allow type given to add_argument.')
        return ActionTypeHint(typehint=self._type)(**kwargs)


def set_url_support(enabled:bool):
    """DEPRECATED: Enables/disables URL support for config read mode.

    Optional config read modes should now be set using function set_config_read_mode.
    """
    set_config_read_mode(
        urls_enabled=enabled,
        fsspec_enabled=True if 's' in get_config_read_mode() else False,
    )
