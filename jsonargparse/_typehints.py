"""Action to support type hints."""

import inspect
import os
import re
import sys
from argparse import ArgumentError
from collections import OrderedDict, abc, defaultdict
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from functools import partial
from importlib import import_module
from types import FunctionType, MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from ._actions import (
    Action,
    ActionConfigFile,
    _ActionHelpClassPath,
    _ActionPrintConfig,
    _find_action,
    _find_parent_action,
    _is_action_value_list,
    parent_parsers_context,
    parse_kwargs,
    remove_actions,
)
from ._common import (
    get_class_instantiator,
    get_unaliased_type,
    is_dataclass_like,
    is_subclass,
    nested_links,
    parent_parser,
    parser_context,
)
from ._loaders_dumpers import (
    get_loader_exceptions,
    json_or_yaml_load,
    json_or_yaml_loader_exceptions,
    load_value,
)
from ._namespace import Namespace
from ._optionals import (
    capture_typing_extension_shadows,
    get_alias_target,
    is_alias_type,
    is_annotated,
    is_annotated_validator,
    typing_extensions_import,
    validate_annotated,
)
from ._util import (
    ClassType,
    NestedArg,
    NoneType,
    Path,
    PathError,
    change_to_path_dir,
    get_import_path,
    get_typehint_origin,
    import_object,
    indent_text,
    iter_to_set_str,
    object_path_serializer,
    parse_value_or_config,
    warning,
)
from .typing import get_registered_type, is_pydantic_type

__all__ = ["lazy_instance"]


NotRequired = typing_extensions_import("NotRequired")
Required = typing_extensions_import("Required")
_TypedDictMeta = typing_extensions_import("_TypedDictMeta")
Unpack = typing_extensions_import("Unpack")


def _capture_typing_extension_shadows(name: str, *collections) -> None:
    """
    Ensure different origins for types in typing_extensions are captured.
    """
    current_module = sys.modules[__name__]
    typehint = getattr(current_module, name)
    return capture_typing_extension_shadows(typehint, name, *collections)


root_types = {
    str,
    int,
    float,
    bool,
    Any,
    Literal,
    Type,
    type,
    Union,
    List,
    list,
    Iterable,
    Sequence,
    MutableSequence,
    abc.Iterable,
    abc.Sequence,
    abc.MutableSequence,
    Tuple,
    tuple,
    Set,
    set,
    frozenset,
    MutableSet,
    abc.MutableSet,
    Dict,
    dict,
    Mapping,
    MutableMapping,
    abc.Mapping,
    abc.MutableMapping,
    OrderedDict,
    Callable,
    abc.Callable,
    NotRequired,
    Required,
    Unpack,
}

leaf_types = {
    str,
    int,
    float,
    bool,
    NoneType,
}

leaf_or_root_types = leaf_types.union(root_types)

tuple_set_origin_types = {Tuple, tuple, Set, set, frozenset, MutableSet, abc.Set, abc.MutableSet}
sequence_origin_types = {
    List,
    list,
    Iterable,
    Sequence,
    MutableSequence,
    abc.Iterable,
    abc.Sequence,
    abc.MutableSequence,
}
mapping_origin_types = {
    Dict,
    dict,
    Mapping,
    MappingProxyType,
    MutableMapping,
    abc.Mapping,
    abc.MutableMapping,
    OrderedDict,
}
sequence_or_mapping_origin_types = sequence_origin_types.union(mapping_origin_types)
callable_origin_types = {Callable, abc.Callable}

literal_types = {Literal}
_capture_typing_extension_shadows("Literal", root_types, literal_types)

not_required_types = {NotRequired}
_capture_typing_extension_shadows("NotRequired", root_types, not_required_types)

required_types = {Required}
_capture_typing_extension_shadows("Required", root_types, required_types)
not_required_required_types = not_required_types.union(required_types)

typed_dict_types = {TypedDict}
_capture_typing_extension_shadows("TypedDict", typed_dict_types)

typed_dict_meta_types = {_TypedDictMeta}
_capture_typing_extension_shadows("_TypedDictMeta", typed_dict_meta_types)

unpack_types = {Unpack}
_capture_typing_extension_shadows("Unpack", unpack_types)

subclass_arg_parser: ContextVar = ContextVar("subclass_arg_parser")
allow_default_instance: ContextVar = ContextVar("allow_default_instance", default=False)
sub_defaults: ContextVar = ContextVar("sub_defaults", default=False)


def get_parse_optional_num_return() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--test")
    arg_parsed = parser._parse_optional("--test=x")
    return len(arg_parsed)


parse_optional_num_return = get_parse_optional_num_return()


class ActionTypeHint(Action):
    """Action to parse a type hint."""

    def __init__(self, typehint: Optional[Type] = None, enable_path: bool = False, **kwargs):
        """Initializer for ActionTypeHint instance.

        Args:
            typehint: The type hint to use for parsing.
            enable_path: Whether to try to load parsed value from path.

        Raises:
            ValueError: If a parameter is invalid.
        """
        if typehint is not None:
            if not self.is_supported_typehint(typehint, full=True):
                raise ValueError(f"Unsupported type hint {typehint}.")
            if get_typehint_origin(typehint) == Union:
                subtype_supported = [
                    subtype is NoneType or self.is_supported_typehint(subtype, full=True)
                    for subtype in typehint.__args__
                ]
                if sum(subtype_supported) < len(subtype_supported):
                    discard = {typehint.__args__[n] for n, s in enumerate(subtype_supported) if not s}
                    kwargs["logger"].debug(f"Discarding unsupported subtypes {discard} from {typehint}")
                    subtypes = tuple(t for t, s in zip(typehint.__args__, subtype_supported) if s)
                    typehint = Union[subtypes]  # type: ignore[assignment]
            self._typehint = typehint
            self._enable_path = False if is_pathlike(typehint) else enable_path
        elif "_typehint" not in kwargs:
            raise ValueError("Expected typehint keyword argument.")
        else:
            self._typehint = kwargs.pop("_typehint")
            self._enable_path = kwargs.pop("_enable_path")
            self.sub_add_kwargs: dict = {}
            if "metavar" not in kwargs:
                kwargs["metavar"] = typehint_metavar(self._typehint)
            super().__init__(**kwargs)
            self._supports_append = self.supports_append(self._typehint)
            self.default = self.normalize_default(self.default)

    def normalize_default(self, default):
        is_subclass_type = self.is_subclass_typehint(self._typehint, all_subtypes=False)
        if isinstance(default, LazyInitBaseClass):
            default = default.lazy_get_init_data()
        elif is_dataclass_like(default.__class__):
            from ._signatures import dataclass_to_dict

            default = dataclass_to_dict(default)
        elif is_subclass_type and isinstance(default, dict) and "class_path" in default:
            default = subclass_spec_as_namespace(default)
            default.class_path = normalize_import_path(default.class_path, self._typehint)
        elif is_enum_type(self._typehint) and isinstance(default, Enum):
            default = default.name
        elif is_callable_type(self._typehint) and callable(default) and not inspect.isclass(default):
            default = get_import_path(default)
        elif is_subclass_type and not allow_default_instance.get():
            from ._parameter_resolvers import UnknownDefault

            default_type = type(default)
            if not is_subclass(default_type, UnknownDefault) and self.is_subclass_typehint(default_type):
                raise ValueError("Subclass types require as default either a dict with class_path or a lazy instance.")
        elif ActionTypeHint.is_return_subclass_typehint(self._typehint) and inspect.isclass(default):
            default = {"class_path": get_import_path(default)}
        return default

    @staticmethod
    def prepare_add_argument(args, kwargs, enable_path, container, logger, sub_add_kwargs=None):
        if kwargs.get("action") is not None:
            raise ValueError("Providing both type and action not allowed.")
        typehint = kwargs.pop("type")
        if args[0].startswith("--") and ActionTypeHint.supports_append(typehint):
            args = tuple(list(args) + [args[0] + "+"])
        if ActionTypeHint.is_subclass_typehint(
            typehint, all_subtypes=False
        ) or ActionTypeHint.is_return_subclass_typehint(typehint):
            help_option = f"--{args[0]}.help" if args[0][0] != "-" else f"{args[0]}.help"
            help_action = container.add_argument(help_option, action=_ActionHelpClassPath(typehint=typehint))
            if sub_add_kwargs:
                help_action.sub_add_kwargs = sub_add_kwargs
        kwargs["action"] = ActionTypeHint(typehint=typehint, enable_path=enable_path, logger=logger)
        return args

    @staticmethod
    def is_supported_typehint(typehint, full=False):
        """Whether the given type hint is supported."""
        typehint = get_unaliased_type(typehint)

        if is_subclass(typehint, Namespace):
            raise ValueError("jsonargparse.Namespace is only intended for parsing results and not supported as a type.")

        supported = (
            typehint in root_types
            or get_typehint_origin(typehint) in root_types
            or get_registered_type(typehint) is not None
            or is_subclass(typehint, Enum)
            or is_dataclass_like(typehint)
            or ActionTypeHint.is_subclass_typehint(typehint)
        )
        if full and supported:
            typehint_origin = get_typehint_origin(typehint) or typehint
            if typehint not in root_types and typehint_origin in root_types and typehint_origin not in literal_types:
                num_supported_args = 0
                subtypes = getattr(typehint, "__args__", [])
                subtypes = [s for s in subtypes if s is not NoneType]
                for subtype in subtypes:
                    if (
                        subtype == Ellipsis
                        or (typehint_origin == type and isinstance(subtype, TypeVar))
                        or subtype in leaf_types
                        or ActionTypeHint.is_supported_typehint(subtype, full=True)
                    ):
                        num_supported_args += 1
                    elif typehint_origin != Union:
                        return False
                if typehint_origin == Union and subtypes and num_supported_args == 0:
                    return False
        return supported

    @staticmethod
    def is_subclass_typehint(typehint, all_subtypes=True, also_lists=False):
        typehint = typehint_from_action(typehint)
        if typehint is None:
            return False
        typehint = get_unaliased_type(typehint)
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin == Union or (also_lists and typehint_origin in sequence_origin_types):
            subtypes = [a for a in typehint.__args__ if a != NoneType]
            test = all if all_subtypes else any
            k = {"also_lists": also_lists}
            return test(ActionTypeHint.is_subclass_typehint(s, **k) for s in subtypes)
        return is_single_subclass_typehint(typehint, typehint_origin)

    @staticmethod
    def is_return_subclass_typehint(typehint):
        typehint = get_unaliased_type(get_optional_arg(get_unaliased_type(typehint)))
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin in callable_origin_types:
            return_type = get_callable_return_type(typehint)
            if ActionTypeHint.is_subclass_typehint(return_type):
                return True
        return False

    @staticmethod
    def is_mapping_typehint(typehint):
        typehint = get_unaliased_type(typehint)
        typehint_origin = get_typehint_origin(typehint) or typehint
        if (
            typehint in mapping_origin_types
            or typehint_origin in mapping_origin_types
            or is_optional(typehint, tuple(mapping_origin_types))
        ):
            return True
        return False

    @staticmethod
    def is_callable_typehint(typehint):
        typehint = typehint_from_action(typehint)
        typehint_origin = get_typehint_origin(get_optional_arg(get_unaliased_type(typehint)))
        return typehint_origin in callable_origin_types

    def is_init_arg_mapping_typehint(self, key, cfg):
        result = False
        class_path = cfg.get(f"{self.dest}.class_path")
        if (
            isinstance(class_path, str)
            and key.startswith(f"{self.dest}.init_args.")
            and self.is_subclass_typehint(self)
        ):
            sub_add_kwargs = dict(self.sub_add_kwargs)
            sub_add_kwargs.pop("linked_targets", None)
            parser = ActionTypeHint.get_class_parser(class_path, sub_add_kwargs=sub_add_kwargs)
            key = re.sub(f"^{self.dest}.init_args.", "", key)
            typehint = getattr(_find_action(parser, key), "_typehint", None)
            result = self.is_mapping_typehint(typehint)
        return result

    @staticmethod
    def parse_argv_item(arg_string):
        parser = subclass_arg_parser.get()
        action = None
        sep = None
        if arg_string.startswith("--"):
            arg_base, explicit_arg = (arg_string, None)
            if "=" in arg_string:
                arg_base, sep, explicit_arg = arg_string.partition("=")
            if "." in arg_base and arg_base not in parser._option_string_actions:
                action = _find_parent_action(parser, arg_base[2:])

        typehint = typehint_from_action(action)
        if typehint:
            if parse_optional_num_return == 4:
                return action, arg_base, sep, explicit_arg
            elif parse_optional_num_return == 1:
                return [(action, arg_base, sep, explicit_arg)]
            return action, arg_base, explicit_arg
        return None

    @staticmethod
    def discard_init_args_on_class_path_change(parser_or_action, prev_cfg, cfg):
        if isinstance(prev_cfg, dict):
            return
        keys = list(prev_cfg.keys(branches=True))
        num = 0
        while num < len(keys):
            key = keys[num]
            prev_val = prev_cfg.get(key)
            val = cfg.get(key)
            if is_subclass_spec(prev_val) and is_subclass_spec(val):
                action = parser_or_action
                if not isinstance(parser_or_action, ActionTypeHint):
                    action = _find_action(parser_or_action, key)
                if isinstance(action, ActionTypeHint):
                    discard_init_args_on_class_path_change(action, prev_val, val)
                    prev_sub_cfg = prev_val.get("init_args")
                    if prev_sub_cfg:
                        sub_add_kwargs = getattr(action, "sub_add_kwargs", {})
                        subparser = ActionTypeHint.get_class_parser(val["class_path"], sub_add_kwargs)
                        sub_cfg = val.get("init_args", Namespace())
                        ActionTypeHint.discard_init_args_on_class_path_change(subparser, prev_sub_cfg, sub_cfg)
                    keys = keys[: num + 1] + [k for k in keys[num + 1 :] if not k.startswith(key + ".")]
            num += 1

    @staticmethod
    @contextmanager
    def subclass_arg_context(parser):
        subclass_arg_parser.set(parser)
        yield

    @staticmethod
    @contextmanager
    def allow_default_instance_context():
        token = allow_default_instance.set(True)
        try:
            yield
        finally:
            allow_default_instance.reset(token)

    @staticmethod
    @contextmanager
    def sub_defaults_context():
        t = sub_defaults.set(True)
        try:
            yield
        finally:
            sub_defaults.reset(t)

    @staticmethod
    def add_sub_defaults(parser, cfg):
        def skip_sub_defaults_apply(v):
            return not (
                isinstance(v, (str, Namespace))
                or is_subclass_spec(v)
                or (isinstance(v, list) and any(is_subclass_spec(e) for e in v))
                or (isinstance(v, dict) and any(is_subclass_spec(e) for e in v.values()))
            )

        with ActionTypeHint.sub_defaults_context(), parent_parsers_context(None, None):
            parser._apply_actions(cfg, skip_fn=skip_sub_defaults_apply)

    @staticmethod
    def supports_append(action):
        typehint = typehint_from_action(action)
        typehint_origin = get_typehint_origin(typehint)
        return typehint and (
            typehint_origin in sequence_origin_types
            or (
                typehint_origin == Union
                and any(get_typehint_origin(x) in sequence_origin_types for x in typehint.__args__)
            )
        )

    def serialize(self, value, dump_kwargs=None):
        sub_add_kwargs = getattr(self, "sub_add_kwargs", {})
        with dump_kwargs_context(dump_kwargs):
            if _is_action_value_list(self):
                return [
                    adapt_typehints(
                        v,
                        self._typehint,
                        default=self.default,
                        serialize=True,
                        sub_add_kwargs=sub_add_kwargs,
                        logger=self.logger,
                    )
                    for v in value
                ]
            return adapt_typehints(
                value,
                self._typehint,
                default=self.default,
                serialize=True,
                sub_add_kwargs=sub_add_kwargs,
                logger=self.logger,
            )

    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding type hint.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs["_typehint"] = self._typehint
            kwargs["_enable_path"] = self._enable_path
            if "nargs" in kwargs and kwargs["nargs"] == 0:
                raise ValueError("ActionTypeHint does not allow nargs=0.")
            return ActionTypeHint(**kwargs)
        cfg, val, opt_str = args[1:]
        if not (self.nargs == "?" and val is None):
            if isinstance(opt_str, str) and opt_str.startswith(f"--{self.dest}."):
                if opt_str.startswith(f"--{self.dest}.init_args."):
                    sub_opt = opt_str[len(f"--{self.dest}.init_args.") :]
                else:
                    sub_opt = opt_str[len(f"--{self.dest}.") :]
                val = NestedArg(key=sub_opt, val=val)
            append = opt_str == f"--{self.dest}+"
            val = self._check_type_(val, append=append, cfg=cfg)
            if is_subclass_spec(val):
                prev_val = cfg.get(self.dest)
                if is_subclass_spec(prev_val) and "init_args" in prev_val:
                    ActionTypeHint.discard_init_args_on_class_path_change(
                        self,
                        prev_val.init_args,
                        val.get("init_args"),
                    )
        cfg.update(val, self.dest)
        return None

    def _check_type(self, value, append=False, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                orig_val = val
                enable_path = self._enable_path and not isinstance(val, NestedArg)
                try:
                    val, config_path = parse_value_or_config(val, enable_path=enable_path)
                except get_loader_exceptions():
                    config_path = None
                path_meta = val.pop("__path__", None) if isinstance(val, dict) else None

                prev_val = cfg.get(self.dest) if cfg else None
                if prev_val is None and not sub_defaults.get() and is_subclass_spec(self.default):
                    prev_val = Namespace(class_path=self.default["class_path"])

                kwargs = {
                    "sub_add_kwargs": getattr(self, "sub_add_kwargs", {}),
                    "prev_val": prev_val,
                    "orig_val": orig_val,
                    "append": append,
                    "enable_path": enable_path,
                    "logger": self.logger,
                }
                try:
                    with change_to_path_dir(config_path):
                        val = adapt_typehints(val, self._typehint, **kwargs)
                except ValueError as ex:
                    assert ex  # needed due to ruff bug that removes " as ex"
                    if orig_val == "-" and isinstance(getattr(ex, "parent", None), PathError):
                        raise ex
                    try:
                        if isinstance(orig_val, str):
                            with change_to_path_dir(config_path):
                                val = adapt_typehints(orig_val, self._typehint, default=self.default, **kwargs)
                            ex = None
                    except ValueError:
                        if self._enable_path and config_path is None and isinstance(orig_val, str):
                            msg = f"\n- Expected a config path but {orig_val} either not accessible or invalid\n- "
                            raise type(ex)(msg + str(ex)) from ex
                    if ex:
                        raise ex

                if path_meta is not None:
                    val["__path__"] = path_meta
                if isinstance(val, (Namespace, dict)) and config_path is not None:
                    val["__path__"] = config_path
                value[num] = val
            except (TypeError, ValueError) as ex:
                if self._is_valid_string(val):
                    value[num] = val
                else:
                    elem = "" if not islist else f" element {num+1}"
                    error = indent_text(str(ex))
                    raise TypeError(f'Parser key "{self.dest}"{elem}:\n{error}') from ex
        return value if islist else value[0]

    def _is_valid_string(self, value):
        typehint = self._typehint
        return isinstance(value, str) and (
            typehint is str or (get_typehint_origin(typehint) == Union and str in typehint.__args__)
        )

    def instantiate_classes(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        sub_add_kwargs = getattr(self, "sub_add_kwargs", {})
        for num, val in enumerate(value):
            value[num] = adapt_typehints(
                val,
                self._typehint,
                default=self.default,
                instantiate_classes=True,
                sub_add_kwargs=sub_add_kwargs,
                logger=self.logger,
            )
        return value if islist else value[0]

    @staticmethod
    def get_class_parser(val_class, sub_add_kwargs=None, skip_args=0):
        if isinstance(val_class, str):
            val_class = import_object(val_class)
        kwargs = dict(sub_add_kwargs) if sub_add_kwargs else {}
        if skip_args:
            kwargs.setdefault("skip", set()).add(skip_args)
        if is_subclass_spec(kwargs.get("default")):
            kwargs["default"] = kwargs["default"].get("init_args")
        parser = parent_parser.get()
        parser = type(parser)(exit_on_error=False, logger=parser.logger, parser_mode=parser.parser_mode)
        remove_actions(parser, (ActionConfigFile, _ActionPrintConfig))
        if inspect.isclass(val_class) or inspect.isclass(get_typehint_origin(val_class)):
            parser.add_class_arguments(val_class, **kwargs)
        else:
            kwargs = {k: v for k, v in kwargs.items() if k != "instantiate"}
            parser.add_function_arguments(val_class, **kwargs)

        if "linked_targets" in kwargs and parser.required_args:
            for key in kwargs["linked_targets"]:
                if key in parser.required_args:
                    parser.required_args.remove(key)

        for link_kwargs in nested_links.get():
            parser.link_arguments(**link_kwargs)

        parser._inner_parser = True

        return parser

    def extra_help(self):
        extra = ""
        typehint = get_optional_arg(self._typehint)
        if self.is_subclass_typehint(typehint, all_subtypes=False) or get_typehint_origin(
            typehint
        ) in callable_origin_types.union({Type, type}):
            if self.is_callable_typehint(typehint) and getattr(typehint, "__args__", None):
                typehint = get_callable_return_type(get_optional_arg(typehint))
            class_paths = get_all_subclass_paths(typehint)
            if class_paths:
                extra = ", known subclasses: " + ", ".join(class_paths)
        return extra

    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        from ._completions import argcomplete_warn_redraw_prompt, get_files_completer

        if self.choices:
            return [str(c) for c in self.choices]
        elif self._typehint == bool:
            return ["true", "false"]
        elif is_optional(self._typehint, bool):
            return ["true", "false", "null"]
        elif is_subclass(self._typehint, Enum):
            enum = self._typehint
            return list(enum.__members__.keys())
        elif is_optional(self._typehint, Enum):
            enum = get_optional_arg(self._typehint)
            return list(enum.__members__.keys()) + ["null"]
        elif is_optional(self._typehint, Path):
            files_completer = get_files_completer()
            return ["null"] + sorted(files_completer(prefix, **kwargs))
        elif chr(int(os.environ["COMP_TYPE"])) == "?":
            try:
                if prefix.strip() == "":
                    raise ValueError()
                self._check_type(prefix)
                msg = "value already valid, "
            except (TypeError, ValueError) + get_loader_exceptions():
                msg = "value not yet valid, "
            msg += "expected type " + type_to_str(self._typehint)
            return argcomplete_warn_redraw_prompt(prefix, msg)


def is_pathlike(typehint) -> bool:
    if get_typehint_origin(typehint) == Union:
        return any(is_pathlike(t) for t in typehint.__args__)
    return is_subclass(typehint, os.PathLike)


def raise_unexpected_value(message: str, val: Any = inspect._empty, exception: Optional[Exception] = None) -> NoReturn:
    if val is not inspect._empty:
        message += f". Got value: {val}"
    raise ValueError(message) from exception


def raise_union_unexpected_value(subtypes, val: Any, exceptions: List[Exception]) -> NoReturn:
    str_exceptions = [indent_text(str(e), first_line=False) for e in exceptions]
    errors = indent_text("- " + "\n- ".join(str_exceptions))
    errors = errors.replace(f". Got value: {val}", "").replace(f" {val} ", " ")
    raise ValueError(
        f"Does not validate against any of the Union subtypes\nSubtypes: {subtypes}"
        f"\nErrors:\n{errors}\nGiven value type: {type(val)}\nGiven value: {val}"
    ) from exceptions[0]


def resolve_forward_ref(ref):
    if not isinstance(ref, ForwardRef) or not ref.__forward_module__:
        return ref

    aliases = __builtins__.copy()
    aliases.update(vars(import_module(ref.__forward_module__)))
    return aliases.get(ref.__forward_arg__, ref)


def adapt_typehints(
    val,
    typehint,
    serialize=False,
    instantiate_classes=False,
    prev_val=None,
    orig_val=None,
    append=False,
    list_item=False,
    enable_path=False,
    sub_add_kwargs=None,
    default=None,
    logger=None,
):
    if type(val) in {str, bool, int, float} and val == default:
        return val

    adapt_kwargs = {
        "serialize": serialize,
        "instantiate_classes": instantiate_classes,
        "prev_val": prev_val,
        "orig_val": orig_val,
        "append": append,
        "enable_path": enable_path,
        "sub_add_kwargs": sub_add_kwargs or {},
        "logger": logger,
    }
    subtypehints = getattr(typehint, "__args__", None)
    typehint_origin = get_typehint_origin(typehint) or typehint

    # Any
    if typehint == Any:
        type_val = type(val)
        if get_registered_type(type_val) or is_subclass(type_val, Enum):
            val = adapt_typehints(val, type_val, **adapt_kwargs)
        elif isinstance(val, str):
            with suppress(*get_loader_exceptions()):
                val, _ = parse_value_or_config(val, enable_path=False, simple_types=True)
        val = adapt_classes_any(val, serialize, instantiate_classes, sub_add_kwargs)

    # Literal
    elif typehint_origin in literal_types:
        if val not in subtypehints and isinstance(val, str):
            subtypes = Union[tuple({type(v) for v in subtypehints if type(v) is not str})]
            val = adapt_typehints(val, subtypes, **adapt_kwargs)
        if val not in subtypehints:
            raise_unexpected_value(f"Expected a {typehint}", val)

    # Basic types
    elif typehint in leaf_types:
        if isinstance(val, str) and typehint is not str:
            with suppress(*json_or_yaml_loader_exceptions):
                val = json_or_yaml_load(val)
        if typehint is float and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)
        if not isinstance(val, typehint) or (typehint in (int, float) and isinstance(val, bool)):
            raise_unexpected_value(f"Expected a {typehint}", val)

    # Annotated
    elif is_annotated(typehint):
        if not serialize and is_annotated_validator(typehint):
            try:
                val = validate_annotated(val, typehint)
            except Exception as ex:
                raise_unexpected_value(str(ex), val, ex)
        else:
            val = adapt_typehints(val, typehint_origin, **adapt_kwargs)

    # Registered types
    elif get_registered_type(typehint):
        registered_type = get_registered_type(typehint)
        if serialize:
            val = registered_type.serializer(val)
        elif not serialize and not registered_type.is_value_of_type(val):
            val = registered_type.deserializer(val)

    # Enum
    elif is_subclass(typehint, Enum):
        if serialize:
            if isinstance(val, typehint):
                val = val.name
        elif not isinstance(val, typehint):
            try:
                val = typehint[val]
            except KeyError as ex:
                raise_unexpected_value(
                    f"Expected a member of {typehint}: {iter_to_set_str(typehint.__members__)}", val, ex
                )

    # Type
    elif typehint in {Type, type} or typehint_origin in {Type, type}:
        if serialize:
            val = object_path_serializer(val)
        elif not serialize and not isinstance(val, type):
            path = val
            val = import_object(val)
            if (typehint in {Type, type} and not isinstance(val, type)) or (
                typehint not in {Type, type} and not is_subclass(val, subtypehints[0])
            ):
                raise_unexpected_value(f"Expected an import path corresponding to a {typehint}", path)

    # Union
    elif typehint_origin == Union:
        vals = []
        sorted_subtypes = sort_subtypes_for_union(subtypehints, val, append)
        for subtype in sorted_subtypes:
            try:
                vals.append(adapt_typehints(val, subtype, **adapt_kwargs))
                break
            except Exception as ex:
                if subtype is str and not isinstance(val, str) and isinstance(orig_val, str):
                    vals.append(orig_val)
                    continue
                vals.append(ex)
        if all(isinstance(v, Exception) for v in vals):
            raise_union_unexpected_value(sorted_subtypes, val, vals)
        val = vals[-1]

    # Tuple or Set
    elif typehint_origin in tuple_set_origin_types:
        if not isinstance(val, (list, tuple, set)):
            raise_unexpected_value(f"Expected a {typehint_origin}", val)
        val = list(val)
        if subtypehints is not None:
            is_tuple = typehint_origin in {Tuple, tuple}
            is_ellipsis = is_ellipsis_tuple(typehint)
            if is_tuple and not is_ellipsis and len(val) != len(subtypehints):
                raise_unexpected_value(f"Expected a tuple with {len(subtypehints)} elements", val)
            for n, v in enumerate(val):
                subtypehint = subtypehints[0 if is_ellipsis or not is_tuple else n]
                val[n] = adapt_typehints(v, subtypehint, **adapt_kwargs)
        if not serialize:
            val = tuple(val) if typehint_origin in {Tuple, tuple} else set(val)

    # List, Iterable or Sequence
    elif typehint_origin in sequence_origin_types:
        if append:
            adapt_kwargs.pop("prev_val")
            if prev_val is None:
                prev_val = []
            elif not isinstance(prev_val, list):
                try:
                    prev_val = [adapt_typehints(prev_val, subtypehints[0], **adapt_kwargs)]
                except Exception:
                    prev_val = []
            val_is_list = isinstance(val, list)
            val = prev_val + (val if val_is_list else [val])
            prev_val = prev_val + [None] * (len(val) - len(prev_val) if val_is_list else 1)
        list_path = None
        if enable_path and type(val) is str:
            with suppress(TypeError):
                from ._optionals import _get_config_read_mode

                list_path = Path(val, mode=_get_config_read_mode())
                val = list_path.get_content().splitlines()
        if isinstance(val, NestedArg) and subtypehints is not None:
            val = (prev_val[:-1] if isinstance(prev_val, list) else []) + [val]
        elif isinstance(val, Iterable) and not isinstance(val, (list, str)) and type(val) not in mapping_origin_types:
            val = list(val)
        elif not isinstance(val, list):
            raise_unexpected_value(f"Expected a {typehint_origin}", val)
        if subtypehints is not None:
            for n, v in enumerate(val):
                if isinstance(prev_val, list) and len(prev_val) == len(val):
                    adapt_kwargs_n = {**deepcopy(adapt_kwargs), "prev_val": prev_val[n]}
                else:
                    adapt_kwargs_n = deepcopy(adapt_kwargs)
                with change_to_path_dir(list_path):
                    val[n] = adapt_typehints(v, subtypehints[0], list_item=True, **adapt_kwargs_n)

    # Dict, Mapping
    elif typehint_origin in mapping_origin_types:
        if isinstance(val, NestedArg):
            if isinstance(prev_val, dict):
                val = {**prev_val, val.key: val.val}
            else:
                val = {val.key: val.val}
        elif isinstance(val, MappingProxyType):
            val = dict(val)
        elif not isinstance(val, dict):
            raise_unexpected_value(f"Expected a {typehint_origin}", val)
        if subtypehints is not None:
            if subtypehints[0] == int:
                cast = str if serialize else int
                val = {cast(k): v for k, v in val.items()}
            for k, v in val.items():
                if "linked_targets" in adapt_kwargs["sub_add_kwargs"]:
                    kwargs = deepcopy(adapt_kwargs)
                    sub_add_kwargs = kwargs["sub_add_kwargs"]
                    sub_add_kwargs["linked_targets"] = {
                        t[len(k + ".") :] for t in sub_add_kwargs["linked_targets"] if t.startswith(k + ".")
                    }
                    sub_add_kwargs["linked_targets"] = {
                        t[len("init_args.") :] if t.startswith("init_args.") else t
                        for t in sub_add_kwargs["linked_targets"]
                    }
                else:
                    kwargs = adapt_kwargs.copy()
                if kwargs.get("prev_val"):
                    if isinstance(kwargs["prev_val"], dict):
                        kwargs["prev_val"] = kwargs["prev_val"].get(k)
                    else:
                        kwargs["prev_val"] = None
                val[k] = adapt_typehints(v, subtypehints[1], **kwargs)
        if type(typehint) in typed_dict_meta_types:
            if hasattr(typehint, "__required_keys__"):
                required_keys = set(typehint.__required_keys__)
                # The standard library TypedDict below Python 3.11 does not store runtime
                # information about optional and required keys when using Required or NotRequired.
                # Thus, capture explicitly Required keys
                required_keys.update(
                    {k for k, v in typehint.__annotations__.items() if get_typehint_origin(v) in required_types}
                )
                # And remove explicitly NotRequired keys
                required_keys.difference_update(
                    {k for k, v in typehint.__annotations__.items() if get_typehint_origin(v) in not_required_types}
                )
            # The standard library TypedDict in Python 3.8 does not store runtime information
            # about which (if any) keys are optional. See https://bugs.python.org/issue38834.
            # Thus, fall back to totality and explicitly Required keys
            elif typehint.__total__:
                required_keys = {
                    k for k, v in typehint.__annotations__.items() if get_typehint_origin(v) not in not_required_types
                }
            else:
                required_keys = {
                    k for k, v in typehint.__annotations__.items() if get_typehint_origin(v) in required_types
                }
            missing_keys = required_keys - val.keys()
            if missing_keys:
                raise_unexpected_value(f"Missing required keys: {missing_keys}", val)
            extra_keys = val.keys() - typehint.__annotations__.keys()
            if extra_keys:
                raise_unexpected_value(f"Unexpected keys: {extra_keys}", val)
            for k, v in val.items():
                subtypehint = resolve_forward_ref(typehint.__annotations__[k])
                val[k] = adapt_typehints(v, subtypehint, **adapt_kwargs)
        if typehint_origin is MappingProxyType and not serialize:
            val = MappingProxyType(val)
        elif typehint_origin is OrderedDict:
            val = dict(val) if serialize else OrderedDict(val)

    # TypedDict NotRequired and Required
    elif typehint_origin in not_required_required_types:
        assert len(subtypehints) == 1, "(Not)Required requires a single type argument"
        val = adapt_typehints(val, subtypehints[0], **adapt_kwargs)

    # Callable
    elif typehint_origin in callable_origin_types or typehint in callable_origin_types:
        if serialize:
            if is_subclass_spec(val):
                val, _, num_partial_args = adapt_partial_callable_class(typehint, val)
                val = adapt_class_type(val, True, False, sub_add_kwargs, skip_args=num_partial_args)
            else:
                val = object_path_serializer(val)
        else:
            try:
                val_input = val
                if isinstance(val, str):
                    class_path = val
                    return_type = get_callable_return_type(typehint)
                    if "." not in val and return_type:
                        class_path = resolve_class_path_by_name(return_type, val)
                    val_obj = import_object(class_path)
                    if inspect.isclass(val_obj):
                        val = Namespace(class_path=class_path)
                    elif callable(val_obj):
                        val = val_obj
                    else:
                        raise ImportError(f"Unexpected import object {val_obj}")
                if isinstance(val, (dict, Namespace, NestedArg)):
                    if prev_val is None:
                        return_type = get_callable_return_type(typehint)
                        if return_type and not inspect.isabstract(return_type):
                            with suppress(ValueError):
                                prev_val = Namespace(class_path=get_import_path(return_type))
                    val = subclass_spec_as_namespace(val, prev_val)
                    if not is_subclass_spec(val):
                        raise ImportError(
                            f"Dict must include a class_path and optionally init_args, but got {val_input}"
                        )
                    val, partial_classes, num_partial_args = adapt_partial_callable_class(typehint, val)
                    val_class = import_object(val["class_path"])
                    if inspect.isclass(val_class) and not (partial_classes or callable_instances(val_class)):
                        raise ImportError(
                            f"Expected '{val['class_path']}' to be a class that instantiates into callable "
                            f"or a subclass of {partial_classes}."
                        )
                    val["class_path"] = get_import_path(val_class)
                    val = adapt_class_type(
                        val,
                        False,
                        instantiate_classes,
                        sub_add_kwargs,
                        skip_args=num_partial_args,
                        partial_classes=partial_classes,
                        prev_val=prev_val,
                    )
            except (ImportError, AttributeError, ArgumentError) as ex:
                raise_unexpected_value(f"Type {typehint} expects a function or a callable class: {ex}", val, ex)

    # Dataclass-like
    elif is_dataclass_like(typehint):
        if isinstance(prev_val, (dict, Namespace)):
            assert isinstance(sub_add_kwargs, dict)
            sub_add_kwargs["default"] = prev_val
        parser = ActionTypeHint.get_class_parser(typehint, sub_add_kwargs=sub_add_kwargs)
        if instantiate_classes:
            init_args = parser.instantiate_classes(val)
            return typehint(**init_args)
        if serialize:
            val = load_value(parser.dump(val, **dump_kwargs.get()))
        elif isinstance(val, (dict, Namespace)):
            if is_subclass_spec(val) and get_import_path(typehint) == val.get("class_path"):
                val = val.get("init_args")
            val = parser.parse_object(val, defaults=sub_defaults.get() or list_item)
        elif isinstance(val, NestedArg):
            prev_val = prev_val if isinstance(prev_val, Namespace) else None
            val = parser.parse_args([f"--{val.key}={val.val}"], namespace=prev_val)
        else:
            raise_unexpected_value(f"Type {typehint} expects a dict or Namespace", val)

    # Subclass
    elif not hasattr(typehint, "__origin__") and inspect.isclass(typehint):
        if is_instance_or_supports_protocol(val, typehint):
            if serialize:
                val = serialize_class_instance(val)
            return val
        if serialize and isinstance(val, str):
            return val

        prev_implicit_defaults = False
        if prev_val is None and not inspect.isabstract(typehint) and not is_protocol(typehint):
            with suppress(ValueError):
                prev_val = Namespace(class_path=get_import_path(typehint))  # implicit class_path
                if parse_kwargs.get().get("defaults") is True:
                    prev_implicit_defaults = True

        val_input = val
        val = subclass_spec_as_namespace(val, prev_val)
        if not is_subclass_spec(val):
            msg = "Does not implement protocol" if is_protocol(typehint) else "Not a valid subclass of"
            raise_unexpected_value(
                f"{msg} {typehint.__name__}. Got value: {val_input}\n"
                "Subclass types expect one of:\n"
                "- a class path (str)\n"
                "- a dict with class_path entry\n"
                "- a dict without class_path but with init_args entry (class path given previously)"
            )

        try:
            val_class = import_object(resolve_class_path_by_name(typehint, val["class_path"]))
            if is_instance_or_supports_protocol(val_class, typehint):
                return val_class  # importable instance
            if is_protocol(val_class):
                raise_unexpected_value(f"Expected an instantiatable class, but {val['class_path']} is a protocol")
            not_subclass = False
            if not is_subclass_or_implements_protocol(val_class, typehint):
                not_subclass = True
                if not inspect.isclass(val_class) and callable(val_class):
                    from ._postponed_annotations import get_return_type

                    return_type = get_return_type(val_class, logger)
                    if is_subclass_or_implements_protocol(return_type, typehint):
                        not_subclass = False
            elif prev_implicit_defaults:
                inner_parser = ActionTypeHint.get_class_parser(typehint, sub_add_kwargs)
                prev_val.init_args = inner_parser.get_defaults()
            if not_subclass:
                msg = "implement protocol" if is_protocol(typehint) else "correspond to a subclass of"
                raise_unexpected_value(f"Import path {val['class_path']} does not {msg} {typehint.__name__}")
            val["class_path"] = get_import_path(val_class)
            val = adapt_class_type(val, serialize, instantiate_classes, sub_add_kwargs, prev_val=prev_val)
        except (ImportError, AttributeError, AssertionError, ArgumentError) as ex:
            class_path = val if isinstance(val, str) else val["class_path"]
            error = indent_text(str(ex))
            raise_unexpected_value(f"Problem with given class_path {class_path!r}:\n{error}", exception=ex)

    # TypeAliasType -- 3.12 `type x = y` or manually via typing_extensions
    elif is_alias_type(typehint):
        return adapt_typehints(val, get_alias_target(typehint), **adapt_kwargs)

    else:
        if str(typehint) == "+VT_co":
            return val  # required for typing.Mapping in python 3.8
        raise RuntimeError(f"The code should never reach here: typehint={typehint}")  # pragma: no cover

    return val


protocol_irrelevant_dunder_methods = {
    "__init__",
    "__new__",
    "__del__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "__subclasshook__",
}


def implements_protocol(value, protocol) -> bool:
    from jsonargparse._parameter_resolvers import get_signature_parameters
    from jsonargparse._postponed_annotations import get_return_type

    if not inspect.isclass(value) or value is object or not is_protocol(protocol):
        return False
    members = 0
    for name, _ in inspect.getmembers(protocol, predicate=inspect.isfunction):
        is_dunder = name.startswith("__") and name.endswith("__")
        if (not is_dunder and name.startswith("_")) or (is_dunder and name in protocol_irrelevant_dunder_methods):
            continue
        if not hasattr(value, name):
            return False
        members += 1
        try:
            value_params = get_signature_parameters(value, name)
        except ValueError:
            return False
        proto_params = get_signature_parameters(protocol, name)
        if [(p.name, p.annotation) for p in proto_params] != [(p.name, p.annotation) for p in value_params]:
            return False
        proto_return = get_return_type(inspect.getattr_static(protocol, name))
        value_return = get_return_type(inspect.getattr_static(value, name))
        if proto_return != value_return:
            return False
    return True if members else False


def is_protocol(class_type) -> bool:
    return getattr(class_type, "_is_protocol", False)


def is_subclass_or_implements_protocol(value, class_type) -> bool:
    if is_protocol(class_type):
        return implements_protocol(value, class_type)
    return is_subclass(value, class_type)


def is_instance_or_supports_protocol(value, class_type):
    if is_protocol(class_type):
        return is_subclass_or_implements_protocol(value.__class__, class_type)
    return isinstance(value, class_type)


def is_subclass_spec(val):
    is_class = isinstance(val, (dict, Namespace)) and "class_path" in val
    if is_class:
        keys = getattr(val, "__dict__", val).keys()
        is_class = len(set(keys) - {"class_path", "init_args", "dict_kwargs", "__path__"}) == 0
    return is_class


def subclass_spec_as_namespace(val, prev_val=None):
    if not isinstance(val, (str, dict, Namespace, NestedArg)):
        return None
    if isinstance(val, str):
        return Namespace(class_path=val)
    if isinstance(val, NestedArg):
        key, val = val
        if "." not in key:
            root_key = key
        else:
            if key.startswith("dict_kwargs."):
                root_key = "dict_kwargs"
                key = key[len("dict_kwargs.") :]
                val = {key: val}
            else:
                root_key = "init_args"
                val = NestedArg(key=key, val=val)
        val = Namespace({root_key: val})
        if isinstance(prev_val, str):
            prev_val = Namespace(class_path=prev_val)
    if isinstance(val, dict):
        val = Namespace(val)
    if "init_args" in val and isinstance(val["init_args"], dict):
        val["init_args"] = Namespace(val["init_args"])
    if not is_subclass_spec(val) and isinstance(prev_val, (Namespace, dict)) and "class_path" in prev_val:
        if "init_args" in val or "dict_kwargs" in val:
            val["class_path"] = prev_val["class_path"]
        else:
            val = Namespace(class_path=prev_val["class_path"], init_args=val)
    return val


def get_callable_return_type(typehint):
    return_type = None
    args = getattr(typehint, "__args__", None)
    if isinstance(args, tuple) and len(args) > 0:
        return_type = args[-1]
    return return_type


def is_single_subclass_typehint(typehint, typehint_origin):
    return (
        inspect.isclass(typehint)
        and typehint not in leaf_or_root_types
        and not get_registered_type(typehint)
        and not is_pydantic_type(typehint)
        and not is_dataclass_like(typehint)
        and typehint_origin is None
        and not is_subclass(typehint, (Path, Enum))
    )


def yield_subclass_types(typehint, also_lists=False, callable_return=False):
    typehint = typehint_from_action(typehint)
    if typehint is None:
        return
    typehint = get_unaliased_type(get_optional_arg(get_unaliased_type(typehint)))
    typehint_origin = get_typehint_origin(typehint)
    if callable_return and typehint_origin in callable_origin_types:
        return_type = get_callable_return_type(typehint)
        if return_type:
            k = {"also_lists": also_lists, "callable_return": callable_return}
            yield from yield_subclass_types(return_type, **k)
    elif typehint_origin == Union or (also_lists and typehint_origin in sequence_origin_types):
        k = {"also_lists": also_lists, "callable_return": callable_return}
        for subtype in typehint.__args__:
            yield from yield_subclass_types(subtype, **k)
    if is_single_subclass_typehint(typehint, typehint_origin):
        yield typehint


def get_subclass_types(typehint, also_lists=False, callable_return=False):
    types = tuple(yield_subclass_types(typehint, also_lists=also_lists, callable_return=callable_return))
    return types or None


def get_subclass_names(typehint, callable_return=False):
    return tuple(t.__name__ for t in yield_subclass_types(typehint, callable_return=callable_return))


def adapt_partial_callable_class(callable_type, subclass_spec):
    partial_classes = False
    num_partial_args = 0
    return_type = get_callable_return_type(callable_type)
    if return_type:
        subclass_types = get_subclass_types(return_type)
        class_type = import_object(resolve_class_path_by_name(return_type, subclass_spec.class_path))
        if subclass_types and is_subclass(class_type, subclass_types):
            subclass_spec = subclass_spec.clone()
            subclass_spec["class_path"] = get_import_path(class_type)
            partial_classes = True
            num_partial_args = len(callable_type.__args__) - 1
    return subclass_spec, partial_classes, num_partial_args


def get_all_subclass_paths(cls: Type) -> List[str]:
    subclass_list = []

    def is_local(cl):
        return ".<locals>." in getattr(cl, "__qualname__", ".<locals>.")

    def is_private(class_path):
        return "._" in class_path

    def add_subclasses(cl):
        if hasattr(cl, "__args__") and get_typehint_origin(cl) in sequence_origin_types.union({Union}):
            for arg in cl.__args__:
                add_subclasses(arg)
            return
        try:
            class_path = get_import_path(cl)
        except (ImportError, AttributeError) as err:  # Attribute is added in case of dot notation imports
            warning(f"Hit failing import with following error: {err}")
            return
        if is_local(cl) or is_subclass(cl, LazyInitBaseClass):
            return
        if not (inspect.isabstract(cl) or is_private(class_path) or is_protocol(cl)):
            if class_path in subclass_list:
                return
            subclass_list.append(class_path)
        for subclass in cl.__subclasses__() if hasattr(cl, "__subclasses__") else []:
            add_subclasses(subclass)

    if get_typehint_origin(cls) in callable_origin_types:
        if len(getattr(cls, "__args__", [])) < 2:
            return subclass_list
        cls = cls.__args__[-1]

    if get_typehint_origin(cls) in {Union, Type, type}:
        for arg in cls.__args__:
            if ActionTypeHint.is_subclass_typehint(arg, also_lists=True) and arg not in {object, type}:
                add_subclasses(arg)
    else:
        add_subclasses(cls)

    return subclass_list


def resolve_class_path_by_name(cls: Type, name: str) -> str:
    class_path = name
    if "." not in class_path:
        subclass_dict = defaultdict(list)
        for subclass in get_all_subclass_paths(cls):
            subclass_name = subclass.rsplit(".", 1)[1]
            subclass_dict[subclass_name].append(subclass)
        if name in subclass_dict:
            name_subclasses = subclass_dict[name]
            if len(name_subclasses) > 1:
                raise ValueError(
                    f"Multiple subclasses with name {name}. Give the full class path to "
                    f'avoid ambiguity: {", ".join(name_subclasses)}.'
                )
            class_path = name_subclasses[0]
    return class_path


def normalize_import_path(class_path, typehint):
    if "." not in class_path:
        class_path = resolve_class_path_by_name(typehint, class_path)
    return get_import_path(import_object(class_path))


dump_kwargs: ContextVar = ContextVar("dump_kwargs", default={})


@contextmanager
def dump_kwargs_context(kwargs):
    dump_kwargs.set(kwargs if kwargs else {})
    yield


def discard_init_args_on_class_path_change(parser_or_action, prev_val, value):
    if prev_val and "init_args" in prev_val and prev_val["class_path"] != value["class_path"]:
        parser = parser_or_action
        if isinstance(parser_or_action, ActionTypeHint):
            sub_add_kwargs = getattr(parser_or_action, "sub_add_kwargs", {})
            parser = ActionTypeHint.get_class_parser(value["class_path"], sub_add_kwargs)
        del_args = {}
        prev_val = subclass_spec_as_namespace(prev_val)
        for key, val in list(prev_val.init_args.__dict__.items()):
            action = _find_action(parser, key)
            if action:
                with parser_context(lenient_check=False, load_value_mode=parser.parser_mode):
                    try:
                        parser._check_value_key(action, val, key, Namespace())
                    except Exception:
                        action = None
            if not action:
                del_args[key] = prev_val.init_args.pop(key)
        if del_args:
            parser_or_action.logger.debug(
                f"Due to class_path change from {prev_val['class_path']!r} to {value['class_path']!r}, "
                f"discarding init_args: {del_args}."
            )


def adapt_class_type(
    value, serialize, instantiate_classes, sub_add_kwargs, prev_val=None, skip_args=0, partial_classes=False
):
    prev_val = subclass_spec_as_namespace(prev_val)
    value = subclass_spec_as_namespace(value)
    val_class = import_object(value.class_path)
    parser = ActionTypeHint.get_class_parser(val_class, sub_add_kwargs, skip_args=skip_args)

    # No need to re-create the linked arg but just "inform" the corresponding parser actions that it exists upstream.
    for target in sub_add_kwargs.get("linked_targets", []):
        split_index = target.find(".")
        if split_index != -1:
            split = ".init_args." if target[split_index:].startswith(".init_args.") else "."

            parent_key, key = target.split(split, maxsplit=1)

            try:
                action = next(a for a in parser._actions if a.dest == parent_key)
            except StopIteration:
                continue

            sub_add_kwargs = getattr(action, "sub_add_kwargs")
            sub_add_kwargs.setdefault("linked_targets", set())
            sub_add_kwargs["linked_targets"].add(key)

    discard_init_args_on_class_path_change(parser, prev_val, value)

    dict_kwargs = value.pop("dict_kwargs", {})
    init_args = value.get("init_args", Namespace())

    if instantiate_classes:
        init_args = parser.instantiate_classes(init_args)
        if not sub_add_kwargs.get("instantiate", True):
            if init_args:
                value["init_args"] = init_args
            return value

        instantiator_fn = get_class_instantiator()

        if partial_classes:
            return partial(
                instantiator_fn,
                val_class,
                **{**init_args, **dict_kwargs},
            )
        return instantiator_fn(val_class, **{**init_args, **dict_kwargs})

    prev_init_args = prev_val.get("init_args") if isinstance(prev_val, Namespace) else None

    if isinstance(init_args, NestedArg):
        value["init_args"] = parser.parse_args(
            [f"--{init_args.key}={init_args.val}"],
            namespace=prev_init_args,
            defaults=sub_defaults.get(),
        )
        return value

    if serialize:
        if init_args:
            value["init_args"] = load_value(parser.dump(init_args, **dump_kwargs.get()))
    else:
        if isinstance(dict_kwargs, dict):
            for key in list(dict_kwargs.keys()):
                if _find_action(parser, key):
                    init_args[key] = dict_kwargs.pop(key)
        elif dict_kwargs:
            init_args["dict_kwargs"] = dict_kwargs
            dict_kwargs = None
        init_args = parser.parse_object(init_args, cfg_base=prev_init_args, defaults=sub_defaults.get())
        if init_args:
            value["init_args"] = init_args
    if dict_kwargs:
        if prev_val and prev_val.get("class_path") == value["class_path"] and prev_val.get("dict_kwargs"):
            dict_kwargs = {**prev_val.get("dict_kwargs"), **dict_kwargs}
        value["dict_kwargs"] = {}
        for key, val in dict_kwargs.items():
            if isinstance(val, str):
                with suppress(get_loader_exceptions()):
                    val = load_value(val, simple_types=True)
            value["dict_kwargs"][key] = val
    return value


def adapt_classes_any(val, serialize, instantiate_classes, sub_add_kwargs):
    if is_subclass_spec(val):
        orig_val = val
        val = subclass_spec_as_namespace(val)
        init_args = val.get("init_args")
        if init_args and not instantiate_classes:
            for subkey, subval in init_args.__dict__.items():
                init_args[subkey] = adapt_classes_any(subval, serialize, instantiate_classes, sub_add_kwargs)
            val["init_args"] = init_args
        try:
            val = adapt_class_type(val, serialize, instantiate_classes, sub_add_kwargs)
        except Exception:
            return orig_val
    elif isinstance(val, list):
        for num, subval in enumerate(val):
            val[num] = adapt_classes_any(subval, serialize, instantiate_classes, sub_add_kwargs)
    elif isinstance(val, dict):
        for key, subval in val.items():
            val[key] = adapt_classes_any(subval, serialize, instantiate_classes, sub_add_kwargs)
    return val


def sort_subtypes_for_union(subtypes, val, append):
    if len(subtypes) > 1:
        if isinstance(val, str):
            key_fn = lambda x: (
                x != NoneType,
                get_typehint_origin(x) not in sequence_or_mapping_origin_types,
            )
        else:
            key_fn = lambda x: x != NoneType
        subtypes = sorted(subtypes, key=key_fn)
        if append:
            subtypes = sorted(subtypes, key=lambda x: get_typehint_origin(x) not in sequence_origin_types)
    return subtypes


def is_ellipsis_tuple(typehint):
    return typehint.__origin__ in {Tuple, tuple} and len(typehint.__args__) > 1 and typehint.__args__[1] == Ellipsis


def is_optional(annotation, ref_type=None):
    """Checks whether a type annotation is an optional for one type class."""
    return (
        get_typehint_origin(annotation) == Union
        and len(annotation.__args__) == 2
        and any(NoneType == a for a in annotation.__args__)
        and (ref_type is None or all(is_subclass(a, ref_type) for a in annotation.__args__ if a != NoneType))
    )


def get_optional_arg(annotation, ref_type=None):
    if is_optional(annotation, ref_type):
        annotation = next(a for a in annotation.__args__ if a != NoneType)
    return annotation


def is_enum_type(annotation):
    return is_subclass(annotation, Enum) or (
        get_typehint_origin(annotation) == Union and any(is_subclass(a, Enum) for a in annotation.__args__)
    )


def is_callable_type(annotation):
    def is_callable(a):
        return (get_typehint_origin(a) or a) in callable_origin_types or a in callable_origin_types

    return is_callable(annotation) or (
        get_typehint_origin(annotation) == Union and any(is_callable(a) for a in annotation.__args__)
    )


def typehint_from_action(action_or_typehint):
    if isinstance(action_or_typehint, Action):
        action_or_typehint = getattr(action_or_typehint, "_typehint", None)
    return action_or_typehint


def type_to_str(obj):
    if obj in {bool, tuple} or is_subclass(obj, (int, float, str, Path, Enum)):
        return obj.__name__
    return re.sub(r"[A-Za-z0-9_<>.]+\.", "", str(obj)).replace("NoneType", "null")


def literal_to_str(val):
    return "null" if val is None else str(val)


def typehint_metavar(typehint):
    """Generates a metavar for some types."""
    metavar = None
    typehint_origin = get_typehint_origin(typehint) or typehint
    if typehint == bool:
        metavar = "{true,false}"
    elif is_optional(typehint, bool):
        metavar = "{true,false,null}"
    elif typehint_origin in literal_types:
        args = typehint.__args__
        metavar = iter_to_set_str(literal_to_str(a) for a in args)
    elif is_subclass(typehint, Enum):
        enum = typehint
        metavar = iter_to_set_str(enum.__members__)
    elif is_optional(typehint, Enum):
        enum = typehint.__args__[0]
        metavar = iter_to_set_str(list(enum.__members__.keys()) + ["null"])
    elif typehint_origin in tuple_set_origin_types:
        metavar = "[ITEM,...]"
    return metavar


def serialize_class_instance(val):
    with suppress(Exception):
        import_path = get_import_path(val)
        if import_path and import_object(import_path) is val:
            return import_path
    val = f"Unable to serialize instance {val}"
    warning(val)
    return val


def callable_instances(cls: Type):
    # https://stackoverflow.com/a/71568161/2732151
    return isinstance(getattr(cls, "__call__", None), FunctionType)


def check_lazy_kwargs(class_type: Type, lazy_kwargs: dict):
    if lazy_kwargs:
        from ._core import ArgumentParser

        parser = ArgumentParser(exit_on_error=False)
        parser.add_class_arguments(class_type)
        try:
            parser.parse_object(lazy_kwargs)
        except ArgumentError as ex:
            raise ValueError(str(ex)) from ex


class LazyInitBaseClass:
    def __init__(self, class_type: Type, lazy_kwargs: dict):
        assert not issubclass(class_type, LazyInitBaseClass)
        check_lazy_kwargs(class_type, lazy_kwargs)
        self._lazy = type(self)
        self._lazy_class_type = class_type
        self._lazy_kwargs = lazy_kwargs
        self._lazy_methods = {}
        seen_methods: dict = {}
        for name, member in inspect.getmembers(class_type, predicate=inspect.isfunction):
            method = getattr(self, name)
            if not inspect.ismethod(method) or name == "__init__":
                continue
            assert name not in self.__dict__
            self._lazy_methods[name] = method
            if id(member) in seen_methods:
                self.__dict__[name] = seen_methods[id(member)]
            else:
                lazy_method = partial(self._lazy_init_then_call_method, name)
                self.__dict__[name] = lazy_method
                if name == "__call__":
                    self._lazy.__call__ = lazy_method  # type: ignore[method-assign]
                seen_methods[id(member)] = lazy_method

    def _lazy_init(self):
        for name in self._lazy_methods:
            if name == "__call__":
                self._lazy.__call__ = self._lazy_methods[name]
            del self.__dict__[name]
        super().__init__(**self._lazy_kwargs)

    def _lazy_init_then_call_method(self, method_name, *args, **kwargs):
        self._lazy_init()
        return self._lazy_methods[method_name](*args, **kwargs)

    def lazy_get_init_args(self) -> Namespace:
        return Namespace(self._lazy_kwargs)

    def lazy_get_init_data(self):
        init_args = self.lazy_get_init_args()
        init = Namespace(class_path=get_import_path(self._lazy_class_type))
        if len(self._lazy_kwargs) > 0:
            init["init_args"] = init_args
        return init


def lazy_instance(class_type: Type[ClassType], **kwargs) -> ClassType:
    """Instantiates a lazy instance of the given type.

    By lazy it is meant that the __init__ is delayed unit the first time that a
    method of the instance is called. It also provides a `lazy_get_init_data` method
    useful for serializing.

    Args:
        class_type: The class to instantiate.
        **kwargs: Any keyword arguments to use for instantiation.
    """
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    class_name = f"LazyInstance_{class_type.__name__}"
    if hasattr(caller_module, class_name):
        lazy_init_class = getattr(caller_module, class_name)
        assert is_subclass(lazy_init_class, LazyInitBaseClass) and is_subclass(lazy_init_class, class_type)
    else:
        lazy_init_class = type(
            class_name,
            (LazyInitBaseClass, class_type),
            {"__doc__": f"Class for lazy instances of {class_type}"},
        )
        if caller_module is not None:
            lazy_init_class.__module__ = getattr(caller_module, "__name__", __name__)
            setattr(caller_module, lazy_init_class.__qualname__, lazy_init_class)
    return lazy_init_class(class_type, kwargs)
