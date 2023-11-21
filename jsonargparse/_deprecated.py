"""Deprecated code."""

import functools
import inspect
import os
import sys
from argparse import Action, ArgumentError
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Set

from ._namespace import Namespace
from ._optionals import FilesCompleterMethod
from ._type_checking import ArgumentParser

__all__ = [
    "ActionEnum",
    "ActionJsonnetExtVars",
    "ActionOperators",
    "ActionPath",
    "ActionPathList",
    "ParserError",
    "set_url_support",
    "usage_and_exit_error_handler",
]


shown_deprecation_warnings: Set[Any] = set()


class JsonargparseDeprecationWarning(DeprecationWarning):
    pass


def deprecation_warning(component, message, stacklevel=1):
    env_var = os.environ.get("JSONARGPARSE_DEPRECATION_WARNINGS", "").lower()
    show_warnings = env_var != "off"
    all_warnings = env_var == "all"
    if show_warnings and (component not in shown_deprecation_warnings or all_warnings):
        from ._util import warning

        if len(shown_deprecation_warnings) == 0 and not all_warnings:
            warning(
                """
                By default only one JsonargparseDeprecationWarning per type is shown. To see
                all warnings set environment variable JSONARGPARSE_DEPRECATION_WARNINGS=all
                and to disable the warnings set JSONARGPARSE_DEPRECATION_WARNINGS=off.
                """,
                JsonargparseDeprecationWarning,
                stacklevel=stacklevel + 2,
            )
        warning(message, JsonargparseDeprecationWarning, stacklevel=stacklevel + 2)
        shown_deprecation_warnings.add(component)


def deprecated(message):
    def deprecated_decorator(component):
        warning = "\n\n.. warning::\n    " + message + "\n"
        component.__doc__ = ("" if component.__doc__ is None else component.__doc__) + warning

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
    from ._core import ArgumentParser

    assert not hasattr(ArgumentParser, "_unpatched_init")

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
        unpatched_method_name = "_unpatched_" + method_name

        def patched_parse(self, *args, _skip_check: bool = False, **kwargs) -> Union[Namespace, Dict[str, Any]]:
            parse_method = getattr(self, unpatched_method_name)
            cfg = parse_method(*args, _skip_check=_skip_check, **kwargs)
            return cfg.as_dict() if self._parse_as_dict and not _skip_check else cfg

        setattr(ArgumentParser, unpatched_method_name, getattr(ArgumentParser, method_name))
        setattr(ArgumentParser, method_name, patched_parse)

    patch_parse_method("parse_args")
    patch_parse_method("parse_object")
    patch_parse_method("parse_env")
    patch_parse_method("parse_string")

    # Patch instantiate_classes
    def patched_instantiate_classes(
        self, cfg: Union[Namespace, Dict[str, Any]], **kwargs
    ) -> Union[Namespace, Dict[str, Any]]:
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


@deprecated(
    """
    instantiate_subclasses was deprecated in v4.0.0 and will be removed in v5.0.0.
"""
)
def instantiate_subclasses(self, cfg: Namespace) -> Namespace:
    """Calls instantiate_classes with instantiate_groups=False.

    Args:
        cfg: The configuration object to use.

    Returns:
        A configuration object with all subclasses instantiated.
    """
    return self.instantiate_classes(cfg, instantiate_groups=False)


def instantiate_subclasses_patch():
    from ._core import ArgumentParser

    ArgumentParser.instantiate_subclasses = instantiate_subclasses


@deprecated(
    """
    ActionEnum was deprecated in v3.9.0 and will be removed in v5.0.0. Enums now
    should be given directly as a type as explained in :ref:`enums`.
"""
)
class ActionEnum:
    """An action based on an Enum that maps to-from strings and enum values."""

    def __init__(self, **kwargs):
        if "enum" in kwargs:
            from ._common import is_subclass

            if not is_subclass(kwargs["enum"], Enum):
                raise ValueError("Expected enum to be an subclass of Enum.")
            self._type = kwargs["enum"]
        else:
            raise ValueError("Expected enum keyword argument.")

    def __call__(self, *args, **kwargs):
        from ._typehints import ActionTypeHint

        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated(
    """
    ActionOperators was deprecated in v3.0.0 and will be removed in v5.0.0. Now
    types should be used as explained in :ref:`restricted-numbers`.
"""
)
class ActionOperators:
    """Action to restrict a value with comparison operators."""

    def __init__(self, **kwargs):
        if "expr" in kwargs:
            restrictions = [kwargs["expr"]] if isinstance(kwargs["expr"], tuple) else kwargs["expr"]
            register_key = (tuple(sorted(restrictions)), kwargs.get("type", int), kwargs.get("join", "and"))
            from .typing import registered_types, restricted_number_type

            if register_key in registered_types:
                self._type = registered_types[register_key]
            else:
                self._type = restricted_number_type(
                    None, kwargs.get("type", int), kwargs["expr"], kwargs.get("join", "and")
                )
        else:
            raise ValueError("Expected expr keyword argument.")

    def __call__(self, *args, **kwargs):
        from ._typehints import ActionTypeHint

        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated(
    """
    ActionPath was deprecated in v3.11.0 and will be removed in v5.0.0. Paths
    now should be given directly as a type as explained in :ref:`parsing-paths`.
"""
)
class ActionPath:
    """Action to check and store a path."""

    def __init__(
        self,
        mode: str,
        skip_check: bool = False,
    ):
        from .typing import path_type

        self._type = path_type(mode, skip_check=skip_check)

    def __call__(self, *args, **kwargs):
        from ._typehints import ActionTypeHint

        return ActionTypeHint(typehint=self._type)(**kwargs)


@deprecated(
    """
    ActionPathList was deprecated in v4.20.0 and will be removed in v5.0.0. Instead
    use as type ``List[<path_type>]`` with ``enable_path=True``.
"""
)
class ActionPathList(Action, FilesCompleterMethod):
    """Action to check and store a list of file paths read from a plain text file or stream."""

    def __init__(self, mode: Optional[str] = None, rel: str = "cwd", **kwargs):
        """Initializer for ActionPathList instance.

        Args:
            mode: The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument (uppercase means
                not), e.g. ActionPathList(mode='fr').
            rel: Whether relative paths are with respect to current working directory 'cwd' or the list's parent
                directory 'list'.

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if mode is not None:
            from .typing import path_type

            self._type = path_type(mode)
            self._rel = rel
            if self._rel not in {"cwd", "list"}:
                raise ValueError(f'rel must be either "cwd" or "list", got {self._rel}.')
        elif "_type" not in kwargs:
            raise ValueError("Expected mode keyword argument.")
        else:
            self._type = kwargs.pop("_type")
            self._rel = kwargs.pop("_rel")
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a PathList and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid PathList.
        """
        if len(args) == 0:
            if "nargs" in kwargs and kwargs["nargs"] not in {"+", 1}:
                raise ValueError('ActionPathList only supports nargs of 1 or "+".')
            kwargs["_type"] = self._type
            kwargs["_rel"] = self._rel
            return ActionPathList(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))
        return None

    def _check_type(self, value, cfg=None):
        if value == []:
            return value
        from ._actions import _is_action_value_list

        islist = _is_action_value_list(self)
        if not islist and not isinstance(value, list):
            value = [value]
        if isinstance(value, list) and all(not isinstance(v, self._type) for v in value):
            path_list_files = value
            value = []
            for path_list_file in path_list_files:
                try:
                    with sys.stdin if path_list_file == "-" else open(path_list_file) as f:
                        path_list = [x.strip() for x in f.readlines()]
                except FileNotFoundError as ex:
                    raise TypeError(f"Problems reading path list: {path_list_file} :: {ex}") from ex
                cwd = os.getcwd()
                if self._rel == "list" and path_list_file != "-":
                    os.chdir(os.path.abspath(os.path.join(path_list_file, os.pardir)))
                try:
                    for num, val in enumerate(path_list):
                        try:
                            path_list[num] = self._type(val)
                        except TypeError as ex:
                            raise TypeError(f"Path number {num+1} in list {path_list_file}, {ex}") from ex
                finally:
                    os.chdir(cwd)
                value += path_list
        return value


@deprecated(
    """
    set_url_support was deprecated in v3.12.0 and will be removed in v5.0.0.
    Optional config read modes should now be set using function
    set_config_read_mode.
"""
)
def set_url_support(enabled: bool):
    """Enables/disables URL support for config read mode."""
    from ._optionals import get_config_read_mode, set_config_read_mode

    set_config_read_mode(
        urls_enabled=enabled,
        fsspec_enabled=True if "s" in get_config_read_mode() else False,
    )


cli_return_parser_message = """
    The return_parser parameter was deprecated in v4.5.0 and will be removed in
    v5.0.0. Instead of this use function capture_parser.
"""


def deprecation_warning_cli_return_parser():
    deprecation_warning("CLI.__init__.return_parser", cli_return_parser_message, stacklevel=2)


logger_property_none_message = """
    Setting the logger property to None was deprecated in v4.10.0 and will raise
    an exception in v5.0.0. Use False instead.
"""

env_prefix_property_none_message = """
    Setting the env_prefix property to None was deprecated in v4.11.0 and will raise
    an exception in v5.0.0. Use True instead.
"""


path_skip_check_message = """
    The skip_check parameter of Path was deprecated in v4.20.0 and will be
    removed in v5.0.0. There is no reason to use a Path type if its checks are
    disabled. Instead use a type such as str or os.PathLike.
"""


def path_skip_check_deprecation(stacklevel=2):
    deprecation_warning("Path.__init__", path_skip_check_message, stacklevel=stacklevel)


path_immutable_attrs_message = """
    Path objects are not meant to be mutable. To make this more explicit,
    attributes have been renamed and changed into properties without setters.
    Please update your code to use the new property names and don't modify path
    attributes. The changes are: ``rel_path`` -> ``relative`` and ``abs_path``
    -> ``absolute``, ``cwd`` no name change, ``skip_check`` will be removed.
"""


class PathDeprecations:
    @property
    def rel_path(self):
        deprecation_warning("Path attr get", path_immutable_attrs_message)
        return self._relative

    @rel_path.setter
    def rel_path(self, rel_path):
        deprecation_warning("Path attr set", path_immutable_attrs_message)
        self._relative = rel_path

    @property
    def abs_path(self):
        deprecation_warning("Path attr get", path_immutable_attrs_message)
        return self._absolute

    @abs_path.setter
    def abs_path(self, abs_path):
        deprecation_warning("Path attr set", path_immutable_attrs_message)
        self._absolute = abs_path

    @property
    def cwd(self):
        return self._cwd

    @cwd.setter
    def cwd(self, cwd):
        deprecation_warning("Path attr set", path_immutable_attrs_message)
        self._cwd = cwd

    def _deprecated_kwargs(self, kwargs):
        from ._util import get_private_kwargs

        self._skip_check = get_private_kwargs(kwargs, skip_check=False)
        if self._skip_check:
            path_skip_check_deprecation()

    def _repr_skip_check(self, name):
        if self._skip_check:
            name += "_skip_check"
        return name

    @property
    def skip_check(self):
        return self._skip_check

    @skip_check.setter
    def skip_check(self, skip_check):
        deprecation_warning("Path attr set", path_immutable_attrs_message)
        self._skip_check = skip_check


class DebugException(Exception):
    pass


@deprecated(
    """
    usage_and_exit_error_handler was deprecated in v4.20.0 and will be removed
    in v5.0.0. With the removal of error_handler, there is no longer a need for
    this function.
"""
)
def usage_and_exit_error_handler(parser: "ArgumentParser", message: str) -> None:
    """Prints the usage and exits with error code 2 (same behavior as argparse).

    Args:
        parser: The parser object.
        message: The message describing the error being handled.
    """
    parser.print_usage(sys.stderr)
    args = {"prog": parser.prog, "message": message}
    sys.stderr.write("%(prog)s: error: %(message)s\n" % args)
    parser.exit(2)


error_handler_message = """
    ArgumentParser's error_handler was deprecated in v4.20.0 and will be removed
    in v5.0.0. Instead use the new exit_on_error parameter from argparse.
"""


def deprecation_warning_error_handler(stacklevel):
    deprecation_warning("ArgumentParser.error_handler", error_handler_message, stacklevel=stacklevel)


class ParserDeprecations:
    def __init__(self, *args, error_handler=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_handler = error_handler

    @property
    def error_handler(self) -> Optional[Callable[["ArgumentParser", str], None]]:
        """Property for the error_handler function that is called when there are parsing errors.

        :getter: Returns the current error_handler function.
        :setter: Sets a new error_handler function (Callable[self, message:str] or None).

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._error_handler

    @error_handler.setter
    def error_handler(self, error_handler):
        if error_handler is not False:
            stacklevel = 2
            stack = inspect.stack()[1]
            if stack.filename.endswith("jsonargparse/_deprecated.py"):
                stacklevel = 5
            deprecation_warning_error_handler(stacklevel)
        if callable(error_handler) or error_handler in {None, False}:
            self._error_handler = error_handler
        else:
            raise ValueError("error_handler can be either a Callable or None.")


ParserError = ArgumentError


def deprecated_module(module_name, mappings=None):
    module_path = f"jsonargparse.{module_name}"
    module = ModuleType(module_path, f"deprecated {module_path}")
    sys.modules[module_path] = module

    @deprecated(
        f"""
        Only use the public API as described in
        https://jsonargparse.readthedocs.io/en/stable/#api-reference. Importing
        from {module_path} is kept only to avoid breaking code that does not
        correctly use the public API. It will no longer be available from v5.0.0.
    """
    )
    def __getattr__(name):
        new_module = f"_{module_name}"
        if mappings and name in mappings:
            new_module, name = mappings[name]
        return getattr(import_module(f"jsonargparse.{new_module}"), name)

    module.__getattr__ = __getattr__
    module.__dict__["__file__"] = str(Path(__file__).parent / f"{module_name}.py")
    module.__dict__["__path__"] = module_path


deprecated_module("actions")
deprecated_module("cli")
deprecated_module("core")
deprecated_module("deprecated")
deprecated_module("formatters")
deprecated_module("jsonnet")
deprecated_module("jsonschema")
deprecated_module("link_arguments")
deprecated_module("loaders_dumpers")
deprecated_module("namespace")
deprecated_module("parameter_resolvers")
deprecated_module("signatures")
deprecated_module("typehints")
deprecated_module("util")
deprecated_module(
    "optionals",
    {
        "import_docstring_parse": ("_optionals", "import_docstring_parser"),
    },
)


@deprecated(
    """
    ActionJsonnetExtVars was deprecated in v4.24.0 and will be removed in
    v5.0.0. Instead use ``type=dict``.
"""
)
class ActionJsonnetExtVars:
    """Action to add argument to provide ext_vars for jsonnet parsing."""

    def __call__(self, *args, **kwargs):
        from ._typehints import ActionTypeHint

        action = ActionTypeHint(typehint=dict)(**kwargs)
        action.jsonnet_ext_vars = True
        return action
