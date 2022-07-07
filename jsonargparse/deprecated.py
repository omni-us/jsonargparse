"""Deprecated code."""

import functools
import inspect
import os
from enum import Enum
from typing import Any, Dict, Set
from .namespace import Namespace
from .optionals import get_config_read_mode, set_config_read_mode
from .typehints import ActionTypeHint
from .typing import path_type, restricted_number_type, registered_types
from .util import is_subclass, warning


__all__ = [
    'ActionEnum',
    'ActionOperators',
    'ActionPath',
    'set_url_support',
]


shown_deprecation_warnings: Set[Any] = set()


class JsonargparseDeprecationWarning(DeprecationWarning):
    pass


def deprecation_warning(component, message):
    env_var = os.environ.get('JSONARGPARSE_DEPRECATION_WARNINGS', '').lower()
    show_warnings = env_var != 'off'
    all_warnings = env_var == 'all'
    if show_warnings and (component not in shown_deprecation_warnings or all_warnings):
        if len(shown_deprecation_warnings) == 0 and not all_warnings:
            warning(
                """
                By default only one JsonargparseDeprecationWarning per type is shown. To see
                all warnings set environment variable JSONARGPARSE_DEPRECATION_WARNINGS=all
                and to disable the warnings set JSONARGPARSE_DEPRECATION_WARNINGS=off.
                """,
                JsonargparseDeprecationWarning,
                stacklevel=1,
            )
        warning(message, JsonargparseDeprecationWarning, stacklevel=3)
        shown_deprecation_warnings.add(component)


def deprecated(message):

    def deprecated_decorator(component):
        warning = '\n\n.. warning::\n    ' + message + '\n'
        component.__doc__ = ('' if component.__doc__ is None else component.__doc__) + warning

        if inspect.isclass(component):
            @functools.wraps(component.__init__)
            def init_wrap(self, *args, **kwargs):
                deprecation_warning(component, message)
                self._original_init(*args, **kwargs)

            component._original_init = component.__init__
            component.__init__ = init_wrap
            decorated = component

        else:
            @functools.wraps(component)
            def decorated(*args, **kwargs):
                deprecation_warning(component, message)
                return component(*args, **kwargs)

        return decorated

    return deprecated_decorator


def parse_as_dict_patch():
    """Adds parse_as_dict support to ArgumentParser as a patch.

    This is a temporal backward compatible support for parse_as_dict to have
    cleaner code in v4.0.0 and warn users about the deprecation and future
    removal.
    """
    from .core import ArgumentParser
    assert not hasattr(ArgumentParser, '_unpatched_init')

    message = """
    ``parse_as_dict`` parameter was deprecated in v4.0.0 and will be removed in
    v5.0.0. After removal, the parse_*, dump, save and instantiate_classes
    methods will only return Namespace and/or accept Namespace objects. If
    needed for some use case, config objects can be converted to a nested dict
    using the Namespace.as_dict method.
    """

    # Patch __init__
    def patched_init(self, *args, parse_as_dict: bool = False, **kwargs):
        self._parse_as_dict = parse_as_dict
        if parse_as_dict:
            deprecation_warning(patched_init, message)
        self._unpatched_init(*args, **kwargs)

    ArgumentParser._unpatched_init = ArgumentParser.__init__
    ArgumentParser.__init__ = patched_init

    from typing import Union

    # Patch parse methods
    def patch_parse_method(method_name):
        unpatched_method_name = '_unpatched_'+method_name

        def patched_parse(self, *args, _skip_check: bool = False, **kwargs) -> Union[Namespace, Dict[str, Any]]:
            parse_method = getattr(self, unpatched_method_name)
            cfg = parse_method(*args, _skip_check=_skip_check, **kwargs)
            return cfg.as_dict() if self._parse_as_dict and not _skip_check else cfg

        setattr(ArgumentParser, unpatched_method_name, getattr(ArgumentParser, method_name))
        setattr(ArgumentParser, method_name, patched_parse)

    patch_parse_method('parse_args')
    patch_parse_method('parse_object')
    patch_parse_method('parse_env')
    patch_parse_method('parse_string')

    # Patch instantiate_classes
    def patched_instantiate_classes(self, cfg: Union[Namespace, Dict[str, Any]], **kwargs) -> Union[Namespace, Dict[str, Any]]:
        if isinstance(cfg, dict):
            cfg = self._apply_actions(cfg)
        cfg = self._unpatched_instantiate_classes(cfg, **kwargs)
        return cfg.as_dict() if self._parse_as_dict else cfg

    ArgumentParser._unpatched_instantiate_classes = ArgumentParser.instantiate_classes
    ArgumentParser.instantiate_classes = patched_instantiate_classes

    # Patch dump
    def patched_dump(self, cfg: Union[Namespace, Dict[str, Any]], *args, **kwargs) -> str:
        if isinstance(cfg, dict):
            cfg = self.parse_object(cfg, _skip_check=True)
        return self._unpatched_dump(cfg, *args, **kwargs)

    ArgumentParser._unpatched_dump = ArgumentParser.dump
    ArgumentParser.dump = patched_dump

    # Patch save
    def patched_save(self, cfg: Union[Namespace, Dict[str, Any]], *args, multifile: bool = True, **kwargs) -> None:
        if multifile and isinstance(cfg, dict):
            cfg = self.parse_object(cfg, _skip_check=True)
        return self._unpatched_save(cfg, *args, multifile=multifile, **kwargs)

    ArgumentParser._unpatched_save = ArgumentParser.save
    ArgumentParser.save = patched_save


def instantiate_subclasses_patch():
    from .core import ArgumentParser

    @deprecated("""
        instantiate_subclasses was deprecated in v4.0.0 and will be removed in v5.0.0.
    """)
    def instantiate_subclasses(self, cfg: Namespace) -> Namespace:
        """Calls instantiate_classes with instantiate_groups=False.

        Args:
            cfg: The configuration object to use.

        Returns:
            A configuration object with all subclasses instantiated.
        """
        return self.instantiate_classes(cfg, instantiate_groups=False)

    ArgumentParser.instantiate_subclasses = instantiate_subclasses


@deprecated("""
    ActionEnum was deprecated in v3.9.0 and will be removed in v5.0.0. Enums now
    should be given directly as a type as explained in :ref:`enums`.
""")
class ActionEnum:
    """An action based on an Enum that maps to-from strings and enum values."""

    def __init__(self, **kwargs):
        if 'enum' in kwargs:
            if not is_subclass(kwargs['enum'], Enum):
                raise ValueError('Expected enum to be an subclass of Enum.')
            self._type = kwargs['enum']
        else:
            raise ValueError('Expected enum keyword argument.')

    def __call__(self, *args, **kwargs):
        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated("""
    ActionOperators was deprecated in v3.0.0 and will be removed in v5.0.0. Now
    types should be used as explained in :ref:`restricted-numbers`.
""")
class ActionOperators:
    """Action to restrict a value with comparison operators."""

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
        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated("""
    ActionPath was deprecated in v3.11.0 and will be removed in v5.0.0. Paths
    now should be given directly as a type as explained in :ref:`parsing-paths`.
""")
class ActionPath:
    """Action to check and store a path."""

    def __init__(
        self,
        mode: str,
        skip_check: bool = False,
    ):
        self._type = path_type(mode, skip_check=skip_check)

    def __call__(self, *args, **kwargs):
        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated("""
    set_url_support was deprecated in v3.12.0 and will be removed in v5.0.0.
    Optional config read modes should now be set using function
    set_config_read_mode.
""")
def set_url_support(enabled:bool):
    """Enables/disables URL support for config read mode."""
    set_config_read_mode(
        urls_enabled=enabled,
        fsspec_enabled=True if 's' in get_config_read_mode() else False,
    )


cli_return_parser_message = """
    The return_parser parameter was deprecated in v4.5.0 and will be removed in
    v5.0.0. Instead of this use function capture_parser.
"""

logger_property_none_message = """
    Setting the logger property to None was deprecated in v4.10.0 and will raise
    an exception in v5.0.0. Use False instead.
"""

env_prefix_property_none_message = """
    Setting the env_prefix property to None was deprecated in v4.11.0 and will raise
    an exception in v5.0.0. Use True instead.
"""
