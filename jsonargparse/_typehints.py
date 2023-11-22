"""Action to support type hints."""

import inspect
import os
import re
from argparse import ArgumentError
from collections import abc, defaultdict
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from functools import partial
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
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
    remove_actions,
)
from ._common import get_class_instantiator, is_dataclass_like, is_subclass, nested_links, parent_parser, parser_context
from ._loaders_dumpers import (
    get_loader_exceptions,
    load_value,
    pyyaml_exceptions,
    yaml_load,
)
from ._namespace import Namespace
from ._optionals import (
    argcomplete_warn_redraw_prompt,
    get_files_completer,
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
    change_to_path_dir,
    get_import_path,
    get_typehint_origin,
    import_object,
    indent_text,
    iter_to_set_str,
    object_path_serializer,
    parse_value_or_config,
    read_stdin,
    warning,
)
from .typing import get_registered_type, is_pydantic_type

__all__ = ["lazy_instance"]


Literal = typing_extensions_import("Literal")


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
    Callable,
    abc.Callable,
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
mapping_origin_types = {Dict, dict, Mapping, MutableMapping, abc.Mapping, abc.MutableMapping}
callable_origin_types = {Callable, abc.Callable}

literal_types = {Literal}
if getattr(Literal, "__module__", None) == "typing_extensions" and hasattr(__import__("typing"), "Literal"):
    root_types.add(__import__("typing").Literal)
    literal_types.add(__import__("typing").Literal)

subclass_arg_parser: ContextVar = ContextVar("subclass_arg_parser")
allow_default_instance: ContextVar = ContextVar("allow_default_instance", default=False)
sub_defaults: ContextVar = ContextVar("sub_defaults", default=False)


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
                self.is_supported_typehint(typehint, full=True)
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
                    typehint = Union[subtypes]  # type: ignore
            self._typehint = typehint
            self._enable_path = False if is_optional(typehint, Path) else enable_path
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
        return default

    @staticmethod
    def prepare_add_argument(args, kwargs, enable_path, container, logger, sub_add_kwargs=None):
        if "action" in kwargs:
            raise ValueError("Providing both type and action not allowed.")
        typehint = kwargs.pop("type")
        if args[0].startswith("--") and ActionTypeHint.supports_append(typehint):
            args = tuple(list(args) + [args[0] + "+"])
        if ActionTypeHint.is_subclass_typehint(typehint, all_subtypes=False):
            help_option = f"--{args[0]}.help" if args[0][0] != "-" else f"{args[0]}.help"
            help_action = container.add_argument(help_option, action=_ActionHelpClassPath(baseclass=typehint))
            if sub_add_kwargs:
                help_action.sub_add_kwargs = sub_add_kwargs
        kwargs["action"] = ActionTypeHint(typehint=typehint, enable_path=enable_path, logger=logger)
        return args

    @staticmethod
    def is_supported_typehint(typehint, full=False):
        """Whether the given type hint is supported."""
        if is_annotated(typehint):
            typehint = get_typehint_origin(typehint)
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
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin == Union or (also_lists and typehint_origin in sequence_origin_types):
            subtypes = [a for a in typehint.__args__ if a != NoneType]
            test = all if all_subtypes else any
            return test(ActionTypeHint.is_subclass_typehint(s) for s in subtypes)
        return (
            inspect.isclass(typehint)
            and typehint not in leaf_or_root_types
            and not get_registered_type(typehint)
            and not is_pydantic_type(typehint)
            and not is_dataclass_like(typehint)
            and typehint_origin is None
            and not is_subclass(typehint, (Path, Enum))
        )

    @staticmethod
    def is_return_subclass_typehint(typehint):
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin in callable_origin_types:
            return_type = get_callable_return_type(typehint)
            if ActionTypeHint.is_subclass_typehint(return_type):
                return True
        return False

    @staticmethod
    def is_mapping_typehint(typehint):
        typehint_origin = get_typehint_origin(typehint) or typehint
        if (
            typehint in mapping_origin_types
            or typehint_origin in mapping_origin_types
            or is_optional(typehint, tuple(mapping_origin_types))
        ):
            return True
        return False

    @staticmethod
    def is_callable_typehint(typehint, all_subtypes=True):
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin == Union:
            subtypes = [a for a in typehint.__args__ if a != NoneType]
            test = all if all_subtypes else any
            return test(ActionTypeHint.is_callable_typehint(s) for s in subtypes)
        return typehint_origin in callable_origin_types or typehint in callable_origin_types

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
        if arg_string.startswith("--"):
            arg_base, explicit_arg = (arg_string, None)
            if "=" in arg_string:
                arg_base, explicit_arg = arg_string.split("=", 1)
            if "." in arg_base and arg_base not in parser._option_string_actions:
                action = _find_parent_action(parser, arg_base[2:])

        typehint = typehint_from_action(action)
        if typehint:
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

        with ActionTypeHint.sub_defaults_context():
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

    @staticmethod
    def apply_appends(parser, cfg):
        for key in [k for k in cfg.keys() if k.endswith("+")]:
            action = _find_action(parser, key[:-1])
            if ActionTypeHint.supports_append(action):
                with parser_context(load_value_mode=parser.parser_mode):
                    val = action._check_type(cfg[key], append=True, cfg=cfg)
                cfg[key[:-1]] = val
                cfg.pop(key)

    def serialize(self, value, dump_kwargs=None):
        sub_add_kwargs = getattr(self, "sub_add_kwargs", {})
        with dump_kwargs_context(dump_kwargs):
            if _is_action_value_list(self):
                return [
                    adapt_typehints(v, self._typehint, serialize=True, sub_add_kwargs=sub_add_kwargs) for v in value
                ]
            return adapt_typehints(value, self._typehint, serialize=True, sub_add_kwargs=sub_add_kwargs)

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
            val = self._check_type(val, append=append, cfg=cfg)
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
                    "append": append,
                    "enable_path": enable_path,
                }
                try:
                    with change_to_path_dir(config_path):
                        val = adapt_typehints(val, self._typehint, **kwargs)
                except ValueError as ex:
                    assert ex  # needed due to ruff bug that removes " as ex"
                    try:
                        if isinstance(orig_val, str):
                            with change_to_path_dir(config_path):
                                val = adapt_typehints(orig_val, self._typehint, **kwargs)
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
                elem = "" if not islist else f" element {num+1}"
                error = indent_text(str(ex))
                raise TypeError(f'Parser key "{self.dest}"{elem}:\n{error}') from ex
        return value if islist else value[0]

    def instantiate_classes(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        sub_add_kwargs = getattr(self, "sub_add_kwargs", {})
        for num, val in enumerate(value):
            value[num] = adapt_typehints(val, self._typehint, instantiate_classes=True, sub_add_kwargs=sub_add_kwargs)
        return value if islist else value[0]

    @staticmethod
    def get_class_parser(val_class, sub_add_kwargs=None, skip_args=0):
        if isinstance(val_class, str):
            val_class = import_object(val_class)
        kwargs = dict(sub_add_kwargs) if sub_add_kwargs else {}
        if skip_args:
            kwargs.setdefault("skip", set()).add(skip_args)
        parser = parent_parser.get()
        parser = type(parser)(exit_on_error=False, logger=parser.logger)
        remove_actions(parser, (ActionConfigFile, _ActionPrintConfig))
        parser.add_class_arguments(val_class, **kwargs)

        if "linked_targets" in kwargs and parser.required_args:
            for key in kwargs["linked_targets"]:
                if key in parser.required_args:
                    parser.required_args.remove(key)

        for link_kwargs in nested_links.get():
            parser.link_arguments(**link_kwargs)

        return parser

    def extra_help(self):
        extra = ""
        if self.is_subclass_typehint(self, all_subtypes=False) or get_typehint_origin(
            self._typehint
        ) in callable_origin_types.union({Type, type}):
            class_paths = get_all_subclass_paths(self._typehint)
            if class_paths:
                extra = ", known subclasses: " + ", ".join(class_paths)
        return extra

    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        if self._typehint == bool:
            return ["true", "false"]
        elif is_optional(self._typehint, bool):
            return ["true", "false", "null"]
        elif is_subclass(self._typehint, Enum):
            enum = self._typehint
            return list(enum.__members__.keys())
        elif is_optional(self._typehint, Enum):
            enum = self._typehint.__args__[0]
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


def raise_unexpected_value(message: str, val: Any = inspect._empty, exception: Optional[Exception] = None) -> NoReturn:
    if val is not inspect._empty:
        message += f". Got value: {val}"
    raise ValueError(message) from exception


def raise_union_unexpected_value(uniontype, val: Any, exceptions: List[Exception]) -> NoReturn:
    str_exceptions = [indent_text(str(e), first_line=False) for e in exceptions]
    errors = indent_text("- " + "\n- ".join(str_exceptions))
    errors = errors.replace(f". Got value: {val}", "").replace(f" {val} ", " ")
    subtypes = uniontype.__args__
    raise ValueError(
        f"Does not validate against any of the Union subtypes\nSubtypes: {subtypes}"
        f"\nErrors:\n{errors}\nGiven value type: {type(val)}\nGiven value: {val}"
    ) from exceptions[0]


def adapt_typehints(
    val,
    typehint,
    serialize=False,
    instantiate_classes=False,
    prev_val=None,
    append=False,
    list_item=False,
    enable_path=False,
    sub_add_kwargs=None,
):
    adapt_kwargs = {
        "serialize": serialize,
        "instantiate_classes": instantiate_classes,
        "prev_val": prev_val,
        "append": append,
        "enable_path": enable_path,
        "sub_add_kwargs": sub_add_kwargs or {},
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
            subtypes = Union[tuple({type(v) for v in subtypehints})]
            val = adapt_typehints(val, subtypes, **adapt_kwargs)
        if val not in subtypehints:
            raise_unexpected_value(f"Expected a {typehint}", val)

    # Basic types
    elif typehint in leaf_types:
        if isinstance(val, str) and typehint is not str:
            with suppress(*pyyaml_exceptions):
                val = yaml_load(val)
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
        subtypehints = sort_subtypes_for_append(subtypehints) if append else subtypehints
        for subtypehint in subtypehints:
            try:
                vals.append(adapt_typehints(val, subtypehint, **adapt_kwargs))
                break
            except Exception as ex:
                vals.append(ex)
        if all(isinstance(v, Exception) for v in vals):
            raise_union_unexpected_value(typehint, val, vals)
        val = [v for v in vals if not isinstance(v, Exception)][0]

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
            if is_tuple and len(val) == 0:
                raise_unexpected_value("Expected a non-empty tuple", val)
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
        if enable_path and type(val) is str:
            if val == "-":
                val = read_stdin().splitlines()
            else:
                with suppress(TypeError):
                    from ._optionals import get_config_read_mode

                    list_path = Path(val, mode=get_config_read_mode())
                    val = list_path.get_content().splitlines()
        if isinstance(val, NestedArg) and subtypehints is not None:
            val = (prev_val[:-1] if isinstance(prev_val, list) else []) + [val]
        elif isinstance(val, Iterable) and not isinstance(val, (list, str)) and type(val) not in mapping_origin_types:
            val = list(val)
        elif not isinstance(val, list):
            raise_unexpected_value(f"Expected a {typehint_origin}", val)
        if subtypehints is not None:
            adapt_kwargs_n = adapt_kwargs
            for n, v in enumerate(val):
                if isinstance(prev_val, list) and len(prev_val) == len(val):
                    adapt_kwargs_n = {**adapt_kwargs, "prev_val": prev_val[n]}
                val[n] = adapt_typehints(v, subtypehints[0], list_item=True, **adapt_kwargs_n)

    # Dict, Mapping
    elif typehint_origin in mapping_origin_types:
        if isinstance(val, NestedArg):
            if isinstance(prev_val, dict):
                val = {**prev_val, val.key: val.val}
            else:
                val = {val.key: val.val}
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
                    kwargs = adapt_kwargs
                val[k] = adapt_typehints(v, subtypehints[1], **kwargs)

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
                    val = adapt_class_type(val, False, instantiate_classes, sub_add_kwargs, skip_args=num_partial_args)
            except (ImportError, AttributeError, ArgumentError) as ex:
                raise_unexpected_value(f"Type {typehint} expects a function or a callable class: {ex}", val, ex)

    # Dataclass-like
    elif is_dataclass_like(typehint):
        if is_dataclass_like(type(prev_val)) and is_subclass(type(prev_val), typehint):
            from ._signatures import dataclass_to_dict

            assert isinstance(sub_add_kwargs, dict)
            sub_add_kwargs["default"] = lazy_instance(type(prev_val), **dataclass_to_dict(prev_val))
        parser = ActionTypeHint.get_class_parser(typehint, sub_add_kwargs=sub_add_kwargs)
        if instantiate_classes:
            init_args = parser.instantiate_classes(val)
            return typehint(**init_args)
        if serialize:
            val = load_value(parser.dump(val, **dump_kwargs.get()))
        elif isinstance(val, (dict, Namespace)):
            val = parser.parse_object(val, defaults=sub_defaults.get() or list_item)
        elif isinstance(val, NestedArg):
            val = parser.parse_args([f"--{val.key}={val.val}"])
        else:
            raise_unexpected_value(f"Type {typehint} expects a dict or Namespace", val)

    # Subclass
    elif not hasattr(typehint, "__origin__") and inspect.isclass(typehint):
        if isinstance(val, typehint):
            if serialize:
                val = serialize_class_instance(val)
            return val
        if serialize and isinstance(val, str):
            return val

        val_input = val
        val = subclass_spec_as_namespace(val, prev_val)
        if not is_subclass_spec(val):
            raise_unexpected_value(
                f"Not a valid subclass of {typehint.__name__}. Got value: {val_input}\n"
                "Subclass types expect one of:\n"
                "- a class path (str)\n"
                "- a dict with class_path entry\n"
                "- a dict without class_path but with init_args entry (class path given previously)"
            )

        try:
            val_class = import_object(resolve_class_path_by_name(typehint, val["class_path"]))
            if not is_subclass(val_class, typehint):
                raise_unexpected_value(
                    f'Import path {val["class_path"]} does not correspond to a subclass of {typehint}'
                )
            val["class_path"] = get_import_path(val_class)
            val = adapt_class_type(val, serialize, instantiate_classes, sub_add_kwargs, prev_val=prev_val)
        except (ImportError, AttributeError, AssertionError, ArgumentError) as ex:
            class_path = val if isinstance(val, str) else val["class_path"]
            error = indent_text(str(ex))
            raise_unexpected_value(f"Problem with given class_path {class_path!r}:\n{error}", exception=ex)

    return val


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
    if len(getattr(typehint, "__args__", None) or []) > 1:
        return_type = typehint.__args__[-1]
    return return_type


def get_subclass_types(typehint, callable_return=True):
    subclass_types = None
    if callable_return and ActionTypeHint.is_callable_typehint(typehint, all_subtypes=False) and typehint.__args__:
        typehint = typehint.__args__[-1]
    if ActionTypeHint.is_subclass_typehint(typehint, all_subtypes=False):
        if get_typehint_origin(typehint) == Union:
            subclass_types = tuple(t for t in typehint.__args__ if ActionTypeHint.is_subclass_typehint(t))
        else:
            subclass_types = (typehint,)
    return subclass_types


def adapt_partial_callable_class(callable_type, subclass_spec):
    partial_classes = False
    num_partial_args = 0
    return_type = get_callable_return_type(callable_type)
    if return_type:
        subclass_types = get_subclass_types(return_type, callable_return=False)
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
        try:
            class_path = get_import_path(cl)
        except (ImportError, AttributeError) as err:  # Attribute is added in case of dot notation imports
            warning(f"Hit failing import with following error: {err}")
            return
        if is_local(cl) or issubclass(cl, LazyInitBaseClass):
            return
        if not (inspect.isabstract(cl) or is_private(class_path)):
            subclass_list.append(class_path)
        for subclass in cl.__subclasses__() if hasattr(cl, "__subclasses__") else []:
            add_subclasses(subclass)

    if get_typehint_origin(cls) in callable_origin_types:
        if len(getattr(cls, "__args__", [])) < 2:
            return subclass_list
        cls = cls.__args__[-1]

    if get_typehint_origin(cls) in {Union, Type, type}:
        for arg in cls.__args__:
            if ActionTypeHint.is_subclass_typehint(arg) and arg not in {object, type}:
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


def adapt_class_type(value, serialize, instantiate_classes, sub_add_kwargs, prev_val=None, skip_args=0):
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

            action = next(a for a in parser._actions if a.dest == parent_key)

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

        if skip_args:

            def partial_instance(*args):
                return instantiator_fn(val_class, *args, **{**init_args, **dict_kwargs})

            return partial_instance
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
        value["dict_kwargs"] = {
            k: load_value(v, simple_types=True) if isinstance(v, str) else v for k, v in dict_kwargs.items()
        }
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


def sort_subtypes_for_append(subtypes):
    if subtypes and len(subtypes) > 1:
        subtypes = sorted(subtypes, key=lambda x: get_typehint_origin(x) not in sequence_origin_types)
    return subtypes


def is_ellipsis_tuple(typehint):
    return typehint.__origin__ in {Tuple, tuple} and len(typehint.__args__) > 1 and typehint.__args__[1] == Ellipsis


def is_optional(annotation, ref_type=object):
    """Checks whether a type annotation is an optional for one type class."""
    return (
        get_typehint_origin(annotation) == Union
        and len(annotation.__args__) == 2
        and any(NoneType == a for a in annotation.__args__)
        and any(is_subclass(a, ref_type) for a in annotation.__args__)
    )


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
                self.__dict__[name] = partial(self._lazy_init_then_call_method, name)
                seen_methods[id(member)] = self.__dict__[name]

    def _lazy_init(self):
        for name in self._lazy_methods:
            del self.__dict__[name]
        super().__init__(**self._lazy_kwargs)

    def _lazy_init_then_call_method(self, method_name, *args, **kwargs):
        self._lazy_init()
        return getattr(self, method_name)(*args, **kwargs)

    def lazy_get_init_args(self) -> Namespace:
        return Namespace(self._lazy_kwargs)

    def lazy_get_init_data(self):
        init_args = self.lazy_get_init_args()
        if is_dataclass_like(self._lazy_class_type):
            return init_args
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
    lazy_init_class = type(
        "LazyInstance_" + class_type.__name__,
        (LazyInitBaseClass, class_type),
        {"__doc__": f"Class for lazy instances of {class_type}"},
    )
    return lazy_init_class(class_type, kwargs)
