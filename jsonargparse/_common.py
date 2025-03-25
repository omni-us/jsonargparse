import argparse
import dataclasses
import inspect
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (  # type: ignore[attr-defined]
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    _GenericAlias,
)

from ._namespace import Namespace
from ._optionals import (
    capture_typing_extension_shadows,
    get_alias_target,
    get_annotated_base_type,
    import_reconplogger,
    is_alias_type,
    is_annotated,
    reconplogger_support,
    typing_extensions_import,
)
from ._type_checking import ArgumentParser

__all__ = [
    "LoggerProperty",
    "null_logger",
    "set_parsing_settings",
]

ClassType = TypeVar("ClassType")

_UnpackGenericAlias = typing_extensions_import("_UnpackAlias")

unpack_meta_types = set()
if _UnpackGenericAlias:
    unpack_meta_types.add(_UnpackGenericAlias)
capture_typing_extension_shadows(_UnpackGenericAlias, "_UnpackGenericAlias", unpack_meta_types)


class InstantiatorCallable(Protocol):
    def __call__(self, class_type: Type[ClassType], *args, **kwargs) -> ClassType:
        pass  # pragma: no cover


InstantiatorsDictType = Dict[Tuple[type, bool], InstantiatorCallable]


parent_parser: ContextVar[Optional["ArgumentParser"]] = ContextVar("parent_parser", default=None)
parser_capture: ContextVar[bool] = ContextVar("parser_capture", default=False)
defaults_cache: ContextVar[Optional[Namespace]] = ContextVar("defaults_cache", default=None)
lenient_check: ContextVar[Union[bool, str]] = ContextVar("lenient_check", default=False)
load_value_mode: ContextVar[Optional[str]] = ContextVar("load_value_mode", default=None)
class_instantiators: ContextVar[Optional[InstantiatorsDictType]] = ContextVar("class_instantiators", default=None)
nested_links: ContextVar[List[dict]] = ContextVar("nested_links", default=[])


parser_context_vars = dict(
    parent_parser=parent_parser,
    parser_capture=parser_capture,
    defaults_cache=defaults_cache,
    lenient_check=lenient_check,
    load_value_mode=load_value_mode,
    class_instantiators=class_instantiators,
    nested_links=nested_links,
)


@contextmanager
def parser_context(**kwargs):
    context_var_tokens = []
    for name, value in kwargs.items():
        context_var = parser_context_vars[name]
        token = context_var.set(value)
        context_var_tokens.append((context_var, token))
    try:
        yield
    finally:
        for context_var, token in context_var_tokens:
            context_var.reset(token)


parsing_settings = dict(
    parse_optionals_as_positionals=False,
)


def set_parsing_settings(*, parse_optionals_as_positionals: Optional[bool] = None) -> None:
    """
    Modify settings that affect the parsing behavior.

    Args:
        parse_optionals_as_positionals: [EXPERIMENTAL] If True, the parser will
            take extra positional command line arguments as values for optional
            arguments. This means that optional arguments can be given by name
            --key=value as usual, but also as positional. The extra positionals
            are applied to optionals in the order that they were added to the
            parser.
    """
    if isinstance(parse_optionals_as_positionals, bool):
        parsing_settings["parse_optionals_as_positionals"] = parse_optionals_as_positionals
    elif parse_optionals_as_positionals is not None:
        raise ValueError(f"parse_optionals_as_positionals must be a boolean, but got {parse_optionals_as_positionals}.")


def get_parsing_setting(name: str):
    if name not in parsing_settings:
        raise ValueError(f"Unknown parsing setting {name}.")
    return parsing_settings[name]


def get_optionals_as_positionals_actions(parser, include_positionals=False):
    from jsonargparse._actions import ActionConfigFile, _ActionConfigLoad, filter_default_actions
    from jsonargparse._completions import ShtabAction
    from jsonargparse._typehints import ActionTypeHint

    actions = []
    for action in filter_default_actions(parser._actions):
        if isinstance(action, (_ActionConfigLoad, ActionConfigFile, ShtabAction)):
            continue
        if ActionTypeHint.is_subclass_typehint(action, all_subtypes=False):
            continue
        if action.nargs not in {1, None}:
            continue
        if not include_positionals and action.option_strings == []:
            continue
        actions.append(action)

    return actions


def supports_optionals_as_positionals(parser):
    return (
        get_parsing_setting("parse_optionals_as_positionals")
        and not parser._subcommands_action
        and not getattr(parser, "_inner_parser", False)
    )


def is_subclass(cls, class_or_tuple) -> bool:
    """Extension of issubclass that supports non-class arguments."""
    try:
        return inspect.isclass(cls) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def is_final_class(cls) -> bool:
    """Checks whether a class is final, i.e. decorated with ``typing.final``."""
    return getattr(cls, "__final__", False)


def is_generic_class(cls) -> bool:
    return isinstance(cls, _GenericAlias) and getattr(cls, "__module__", "") != "typing"


def is_unpack_typehint(cls) -> bool:
    return any(isinstance(cls, unpack_type) for unpack_type in unpack_meta_types)


def get_generic_origin(cls):
    return cls.__origin__ if is_generic_class(cls) else cls


def get_unaliased_type(cls):
    new_cls = cls
    while True:
        cur_cls = new_cls
        if is_annotated(new_cls):
            new_cls = get_annotated_base_type(new_cls)
        if is_alias_type(new_cls):
            new_cls = get_alias_target(new_cls)
        if new_cls == cur_cls:
            break
    return cur_cls


def is_dataclass_like(cls) -> bool:
    if is_generic_class(cls):
        return is_dataclass_like(cls.__origin__)
    if not inspect.isclass(cls) or cls is object:
        return False
    if is_final_class(cls):
        return True
    classes = [c for c in inspect.getmro(cls) if c not in {object, Generic}]
    all_dataclasses = all(dataclasses.is_dataclass(c) for c in classes)

    if not all_dataclasses:
        from ._optionals import attrs_support, is_pydantic_model

        if is_pydantic_model(cls):
            return True

        if attrs_support:
            import attrs

            if attrs.has(cls):
                return True

    return all_dataclasses


def default_class_instantiator(class_type: Type[ClassType], *args, **kwargs) -> ClassType:
    return class_type(*args, **kwargs)


class ClassInstantiator:
    def __init__(self, instantiators: InstantiatorsDictType) -> None:
        self.instantiators = instantiators

    def __call__(self, class_type: Type[ClassType], *args, **kwargs) -> ClassType:
        for (cls, subclasses), instantiator in self.instantiators.items():
            if class_type is cls or (subclasses and is_subclass(class_type, cls)):
                return instantiator(class_type, *args, **kwargs)
        return default_class_instantiator(class_type, *args, **kwargs)


def get_class_instantiator() -> InstantiatorCallable:
    instantiators = class_instantiators.get()
    if not instantiators:
        return default_class_instantiator
    return ClassInstantiator(instantiators)


# logging

logging_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
null_logger = logging.getLogger("jsonargparse_null_logger")
null_logger.addHandler(logging.NullHandler())
null_logger.parent = None


def setup_default_logger(data, level, caller):
    name = caller
    if isinstance(data, str):
        name = data
    elif isinstance(data, dict) and "name" in data:
        name = data["name"]
    logger = logging.getLogger(name)
    logger.parent = None
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    level = getattr(logging, level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def parse_logger(logger: Union[bool, str, dict, logging.Logger], caller):
    if not isinstance(logger, (bool, str, dict, logging.Logger)):
        raise ValueError(f"Expected logger to be an instance of (bool, str, dict, logging.Logger), but got {logger}.")
    if isinstance(logger, dict) and len(set(logger.keys()) - {"name", "level"}) > 0:
        value = {k: v for k, v in logger.items() if k not in {"name", "level"}}
        raise ValueError(f"Unexpected data to configure logger: {value}.")
    if logger is False:
        return null_logger
    level = "WARNING"
    if isinstance(logger, dict) and "level" in logger:
        level = logger["level"]
    if level not in logging_levels:
        raise ValueError(f"Got logger level {level!r} but must be one of {logging_levels}.")
    if (logger is True or (isinstance(logger, dict) and "name" not in logger)) and reconplogger_support:
        kwargs = {"level": "DEBUG", "reload": True} if debug_mode_active() else {}
        logger = import_reconplogger("parse_logger").logger_setup(**kwargs)
    if not isinstance(logger, logging.Logger):
        logger = setup_default_logger(logger, level, caller)
    return logger


class LoggerProperty:
    """Class designed to be inherited by other classes to add a logger property."""

    def __init__(self, *args, logger: Union[bool, str, dict, logging.Logger] = False, **kwargs):
        """Initializer for LoggerProperty class."""
        self.logger = logger  # type: ignore[assignment]
        super().__init__(*args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        """The logger property for the class.

        :getter: Returns the current logger.
        :setter: Sets the given logging.Logger as logger or sets the default logger
                 if given True/str(logger name)/dict(name, level), or disables logging
                 if given False.

        Raises:
            ValueError: If an invalid logger value is given.
        """
        return self._logger

    @logger.setter
    def logger(self, logger: Union[bool, str, dict, logging.Logger]):
        if logger is None:
            from ._deprecated import deprecation_warning, logger_property_none_message

            deprecation_warning((LoggerProperty.logger, None), logger_property_none_message, stacklevel=2)
            logger = False
        if not logger and debug_mode_active():
            logger = {"level": "DEBUG"}
        self._logger = parse_logger(logger, type(self).__name__)


def debug_mode_active() -> bool:
    return os.getenv("JSONARGPARSE_DEBUG", "").lower() not in {"", "false", "no", "0"}


if debug_mode_active():
    os.environ["LOGGER_LEVEL"] = "DEBUG"  # pragma: no cover


# base classes


class Action(LoggerProperty, argparse.Action):
    """Base for jsonargparse Action classes."""

    def _check_type_(self, value, **kwargs):
        if not hasattr(self, "_check_type_kwargs"):
            self._check_type_kwargs = set(inspect.signature(self._check_type).parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in self._check_type_kwargs}
        return self._check_type(value, **kwargs)
