"""Action to support type hints."""

import inspect
import os
import re
import sys
import warnings
from argparse import Action
from collections import abc, defaultdict
from contextlib import contextmanager
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
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .actions import _ActionHelpClassPath, _find_action, _find_parent_action, _is_action_value_list
from .loaders_dumpers import get_loader_exceptions, load_value, load_value_context
from .namespace import Namespace
from .typing import is_final_class, registered_types
from .optionals import (
    argcomplete_warn_redraw_prompt,
    get_files_completer,
    typing_extensions_import,
)
from .util import (
    change_to_path_dir,
    ClassType,
    get_import_path,
    import_object,
    indent_text,
    is_subclass,
    iter_to_set_str,
    lenient_check,
    lenient_check_context,
    NestedArg,
    NoneType,
    object_path_serializer,
    ParserError,
    parse_value_or_config,
    Path,
    warning,
)


__all__ = ['lazy_instance']


Literal = False if sys.version_info[:2] == (3, 6) else typing_extensions_import('Literal')


root_types = {
    str,
    int,
    float,
    bool,
    Any,
    Literal,
    Type, type,
    Union,
    List, list, Iterable, Sequence, MutableSequence, abc.Iterable, abc.Sequence, abc.MutableSequence,
    Tuple, tuple,
    Set, set, frozenset, MutableSet, abc.MutableSet,
    Dict, dict, Mapping, MutableMapping, abc.Mapping, abc.MutableMapping,
    Callable, abc.Callable,
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
sequence_origin_types = {List, list, Iterable, Sequence, MutableSequence, abc.Iterable, abc.Sequence,
                         abc.MutableSequence}
mapping_origin_types = {Dict, dict, Mapping, MutableMapping, abc.Mapping, abc.MutableMapping}
callable_origin_types = {Callable, abc.Callable}


subclass_arg_parser: ContextVar = ContextVar('subclass_arg_parser')
sub_defaults: ContextVar = ContextVar('sub_defaults', default=False)


class ActionTypeHint(Action):
    """Action to parse a type hint."""

    def __init__(
        self,
        typehint: Type = None,
        enable_path: bool = False,
        **kwargs
    ):
        """Initializer for ActionTypeHint instance.

        Args:
            typehint: The type hint to use for parsing.
            enable_path: Whether to try to load parsed value from path.

        Raises:
            ValueError: If a parameter is invalid.
        """
        if typehint is not None:
            if not self.is_supported_typehint(typehint, full=True):
                raise ValueError(f'Unsupported type hint {typehint}.')
            self._typehint = typehint
            self._enable_path = False if is_optional(typehint, Path) else enable_path
        elif '_typehint' not in kwargs:
            raise ValueError('Expected typehint keyword argument.')
        else:
            self._typehint = kwargs.pop('_typehint')
            self._enable_path = kwargs.pop('_enable_path')
            if 'metavar' not in kwargs:
                kwargs['metavar'] = typehint_metavar(self._typehint)
            super().__init__(**kwargs)
            self._supports_append = self.supports_append(self._typehint)
            self.normalize_default()


    def normalize_default(self):
        default = self.default
        if isinstance(default, LazyInitBaseClass):
            self.default = default.lazy_get_init_data()
        elif self.is_subclass_typehint(self._typehint, all_subtypes=False) and isinstance(default, dict) and 'class_path' in default:
            self.default = subclass_spec_as_namespace(default)
        elif is_enum_type(self._typehint) and isinstance(default, Enum):
            self.default = default.name
        elif is_callable_type(self._typehint) and callable(default) and not inspect.isclass(default):
            self.default = get_import_path(default)


    @staticmethod
    def prepare_add_argument(args, kwargs, enable_path, container, sub_add_kwargs=None):
        if 'action' in kwargs:
            raise ValueError('Providing both type and action not allowed.')
        typehint = kwargs.pop('type')
        if args[0].startswith('--') and ActionTypeHint.supports_append(typehint):
            args = tuple(list(args)+[args[0]+'+'])
        if ActionTypeHint.is_subclass_typehint(typehint):
            help_action = container.add_argument(args[0]+'.help', action=_ActionHelpClassPath(baseclass=typehint))
            if sub_add_kwargs:
                help_action.sub_add_kwargs = sub_add_kwargs
        kwargs['action'] = ActionTypeHint(typehint=typehint, enable_path=enable_path)
        return args


    @staticmethod
    def is_supported_typehint(typehint, full=False):
        """Whether the given type hint is supported."""
        supported = \
            typehint in root_types or \
            get_typehint_origin(typehint) in root_types or \
            typehint in registered_types or \
            is_subclass(typehint, Enum) or \
            ActionTypeHint.is_subclass_typehint(typehint)
        if full and supported:
            typehint_origin = get_typehint_origin(typehint) or typehint
            if typehint not in root_types and typehint_origin in root_types and typehint_origin != Literal:
                for typehint in getattr(typehint, '__args__', []):
                    if not (
                        typehint == Ellipsis or
                        (typehint_origin == type and isinstance(typehint, TypeVar)) or
                        typehint in leaf_types or
                        ActionTypeHint.is_supported_typehint(typehint, full=True)
                    ):
                        return False
        return supported


    @staticmethod
    def is_subclass_typehint(typehint, all_subtypes=True):
        typehint = typehint_from_action(typehint)
        if typehint is None:
            return False
        typehint_origin = get_typehint_origin(typehint)
        if typehint_origin == Union:
            subtypes = [a for a in typehint.__args__ if a != NoneType]
            test = all if all_subtypes else any
            return test(ActionTypeHint.is_subclass_typehint(s) for s in subtypes)
        return inspect.isclass(typehint) and \
            typehint not in leaf_or_root_types and \
            typehint not in registered_types and \
            typehint_origin is None and \
            not is_subclass(typehint, (Path, Enum))


    @staticmethod
    def is_mapping_typehint(typehint):
        typehint_origin = get_typehint_origin(typehint) or typehint
        if typehint in mapping_origin_types or typehint_origin in mapping_origin_types or is_optional(typehint, tuple(mapping_origin_types)):
            return True
        return False


    def is_init_arg_mapping_typehint(self, key, cfg):
        result = False
        class_path = cfg.get(f'{self.dest}.class_path')
        if isinstance(class_path, str) and key.startswith(f'{self.dest}.init_args.') and self.is_subclass_typehint(self):
            sub_add_kwargs = dict(self.sub_add_kwargs)
            sub_add_kwargs.pop('linked_targets', None)
            parser = ActionTypeHint.get_class_parser(class_path, sub_add_kwargs=sub_add_kwargs)
            key = re.sub(f'^{self.dest}.init_args.', '', key)
            typehint = getattr(_find_action(parser, key), '_typehint', None)
            result = self.is_mapping_typehint(typehint)
        return result


    @staticmethod
    def parse_argv_item(arg_string):
        parser = subclass_arg_parser.get()
        action = None
        if arg_string.startswith('--'):
            arg_base, explicit_arg = (arg_string, None)
            if '=' in arg_string:
                arg_base, explicit_arg = arg_string.split('=', 1)
            if '.' in arg_base and arg_base not in parser._option_string_actions:
                action = _find_parent_action(parser, arg_base[2:])

        typehint = typehint_from_action(action)
        if (
            ActionTypeHint.is_subclass_typehint(typehint, all_subtypes=False) or
            ActionTypeHint.is_mapping_typehint(typehint)
        ):
            return action, arg_base, explicit_arg
        elif parser._subcommands_action and arg_string in parser._subcommands_action._name_parser_map:
            subparser = parser._subcommands_action._name_parser_map[arg_string]
            subclass_arg_parser.set(subparser)


    @staticmethod
    def discard_init_args_on_class_path_change(parser, cfg_to, cfg_from):
        for action in [a for a in parser._actions if ActionTypeHint.is_subclass_typehint(a, all_subtypes=False)]:
            val_to = cfg_to.get(action.dest)
            val_from = cfg_from.get(action.dest)
            if is_subclass_spec(val_to) and is_subclass_spec(val_from):
                discard_init_args_on_class_path_change(action, val_to, val_from)


    @staticmethod
    @contextmanager
    def subclass_arg_context(parser):
        subclass_arg_parser.set(parser)
        yield


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
        with ActionTypeHint.sub_defaults_context():
            parser._apply_actions(cfg)


    @staticmethod
    def supports_append(action):
        typehint = typehint_from_action(action)
        typehint_origin = get_typehint_origin(typehint)
        return typehint and (
            typehint_origin in sequence_origin_types or
            (
                typehint_origin == Union and
                any(get_typehint_origin(x) in sequence_origin_types for x in typehint.__args__)
            )
        )

    @staticmethod
    def apply_appends(parser, cfg):
        for key in [k for k in cfg.keys() if k.endswith('+')]:
            action = _find_action(parser, key[:-1])
            if ActionTypeHint.supports_append(action):
                with load_value_context(parser.parser_mode):
                    val = action._check_type(cfg[key], append=True, cfg=cfg)
                cfg[key[:-1]] = val
                cfg.pop(key)


    def serialize(self, value, dump_kwargs=None):
        sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
        with dump_kwargs_context(dump_kwargs):
            if _is_action_value_list(self):
                return [adapt_typehints(v, self._typehint, serialize=True, sub_add_kwargs=sub_add_kwargs) for v in value]
            return adapt_typehints(value, self._typehint, serialize=True, sub_add_kwargs=sub_add_kwargs)


    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding type hint.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_typehint'] = self._typehint
            kwargs['_enable_path'] = self._enable_path
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise ValueError('ActionTypeHint does not allow nargs=0.')
            return ActionTypeHint(**kwargs)
        if self.nargs == '?' and args[2] is None:
            val = None
        else:
            parser, cfg, val, opt_str = args
            if isinstance(opt_str, str) and opt_str.startswith(f'--{self.dest}.'):
                sub_opt = opt_str[len(f'--{self.dest}.'):]
                val = NestedArg(key=sub_opt, val=val)
                if self.dest not in cfg:
                    try:
                        default = parser.get_default(self.dest)
                        cfg = deepcopy(cfg)
                        cfg[self.dest] = default
                    except KeyError:
                        pass
            append = opt_str == f'--{self.dest}+'
            val = self._check_type(val, append=append, cfg=cfg)
        args[1].update(val, self.dest)


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
                path_meta = val.pop('__path__', None) if isinstance(val, dict) else None
                kwargs = {
                    'sub_add_kwargs': getattr(self, 'sub_add_kwargs', {}),
                    'prev_val': cfg.get(self.dest) if cfg else None,
                    'append': append,
                }
                try:
                    with change_to_path_dir(config_path):
                        val = adapt_typehints(val, self._typehint, **kwargs)
                except ValueError as ex:
                    val_is_int_float_or_none = isinstance(val, (int, float)) or val is None
                    if lenient_check.get():
                        value[num] = orig_val if val_is_int_float_or_none else val
                        continue
                    if val_is_int_float_or_none and config_path is None:
                        val = adapt_typehints(orig_val, self._typehint, **kwargs)
                    else:
                        if self._enable_path and config_path is None and isinstance(orig_val, str):
                            msg = f'\n- Expected a config path but "{orig_val}" either not accessible or invalid.\n- '
                            raise type(ex)(msg+str(ex)) from ex
                        raise ex

                if not append and self._supports_append:
                    prev_val = kwargs.get('prev_val')
                    if isinstance(prev_val, list) and not_append_diff(prev_val, val) and get_typehint_origin(self._typehint) == Union:
                        warnings.warn(f'Replacing list value "{prev_val}" with "{val}". To append to a list use "{self.dest}+".')

                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, (Namespace, dict)) and config_path is not None:
                    val['__path__'] = config_path
                value[num] = val
            except (TypeError, ValueError) as ex:
                elem = '' if not islist else f' element {num+1}'
                raise TypeError(f'Parser key "{self.dest}"{elem}: {ex}') from ex
        return value if islist else value[0]


    def instantiate_classes(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
        for num, val in enumerate(value):
            value[num] = adapt_typehints(val, self._typehint, instantiate_classes=True, sub_add_kwargs=sub_add_kwargs)
        return value if islist else value[0]


    @staticmethod
    def get_class_parser(val_class, sub_add_kwargs=None):
        from .core import ArgumentParser
        if isinstance(val_class, str):
            val_class = import_object(val_class)
        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(val_class, **(sub_add_kwargs or {}))
        return parser


    def extra_help(self):
        extra = ''
        if (
            self.is_subclass_typehint(self, all_subtypes=False) or
            get_typehint_origin(self._typehint) in {Type, type}
        ):
            class_paths = get_all_subclass_paths(self._typehint)
            if class_paths:
                extra = ', known subclasses: '+', '.join(class_paths)
        return extra


    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        if self._typehint == bool:
            return ['true', 'false']
        elif is_optional(self._typehint, bool):
            return ['true', 'false', 'null']
        elif is_subclass(self._typehint, Enum):
            enum = self._typehint
            return list(enum.__members__.keys())
        elif is_optional(self._typehint, Enum):
            enum = self._typehint.__args__[0]
            return list(enum.__members__.keys())+['null']
        elif is_optional(self._typehint, Path):
            files_completer = get_files_completer()
            return ['null'] + sorted(files_completer(prefix, **kwargs))
        elif chr(int(os.environ['COMP_TYPE'])) == '?':
            try:
                if prefix.strip() == '':
                    raise ValueError()
                self._check_type(prefix)
                msg = 'value already valid, '
            except (TypeError, ValueError) + get_loader_exceptions():
                msg = 'value not yet valid, '
            msg += 'expected type '+type_to_str(self._typehint)
            return argcomplete_warn_redraw_prompt(prefix, msg)


def adapt_typehints(val, typehint, serialize=False, instantiate_classes=False, prev_val=None, append=False, sub_add_kwargs=None):

    adapt_kwargs = {
        'serialize': serialize,
        'instantiate_classes': instantiate_classes,
        'prev_val': prev_val,
        'append': append,
        'sub_add_kwargs': sub_add_kwargs or {},
    }
    subtypehints = get_typehint_subtypes(typehint, append=append)
    typehint_origin = get_typehint_origin(typehint) or typehint

    # Any
    if typehint == Any:
        type_val = type(val)
        if type_val in registered_types or is_subclass(type_val, Enum):
            val = adapt_typehints(val, type_val, **adapt_kwargs)
        elif isinstance(val, str):
            try:
                val, _ = parse_value_or_config(val, enable_path=False)
            except get_loader_exceptions():
                pass

    # Literal
    elif typehint_origin == Literal:
        if val not in subtypehints:
            raise ValueError(f'Expected a {typehint} but got "{val}"')

    # Basic types
    elif typehint in leaf_types:
        if not isinstance(val, typehint):
            if typehint is float and isinstance(val, int):
                val = float(val)
            else:
                raise ValueError(f'Expected a {typehint} but got "{val}"')

    # Registered types
    elif typehint in registered_types:
        registered_type = registered_types[typehint]
        if serialize:
            if registered_type.is_value_of_type(val):
                val = registered_type.serializer(val)
            else:
                registered_type.deserializer(val)
        elif not serialize and not registered_type.is_value_of_type(val):
            val = registered_type.deserializer(val)

    # Enum
    elif is_subclass(typehint, Enum):
        if serialize:
            if isinstance(val, typehint):
                val = val.name
            else:
                if val not in typehint.__members__:
                    raise ValueError(f'Value "{val}" is not a valid member name for the enum "{typehint}"')
        elif not serialize and not isinstance(val, typehint):
            val = typehint[val]

    # Type
    elif typehint in {Type, type} or typehint_origin in {Type, type}:
        if serialize:
            val = object_path_serializer(val)
        elif not serialize and not isinstance(val, type):
            path = val
            val = import_object(val)
            if (typehint in {Type, type} and not isinstance(val, type)) or \
               (typehint not in {Type, type} and not is_subclass(val, subtypehints[0])):
                raise ValueError(f'Value "{path}" is not a {typehint}.')

    # Union
    elif typehint_origin == Union:
        vals = []
        for subtypehint in subtypehints:
            try:
                vals.append(adapt_typehints(val, subtypehint, **adapt_kwargs))
                break
            except Exception as ex:
                vals.append(ex)
        if all(isinstance(v, Exception) for v in vals):
            e = indent_text('\n- '.join(str(v) for v in ['']+vals))
            raise ValueError(f'Value "{val}" does not validate against any of the types in {typehint}:{e}')
        val = [v for v in vals if not isinstance(v, Exception)][0]

    # Tuple or Set
    elif typehint_origin in tuple_set_origin_types:
        if not isinstance(val, (list, tuple, set)):
            raise ValueError(f'Expected a {typehint_origin} but got "{val}"')
        val = list(val)
        if subtypehints is not None:
            is_tuple = typehint_origin in {Tuple, tuple}
            is_ellipsis = is_ellipsis_tuple(typehint)
            if is_tuple and not is_ellipsis and len(val) != len(subtypehints):
                raise ValueError(f'Expected a tuple with {len(subtypehints)} elements but got "{val}"')
            for n, v in enumerate(val):
                subtypehint = subtypehints[0 if is_ellipsis else n]
                val[n] = adapt_typehints(v, subtypehint, **adapt_kwargs)
            if is_tuple and len(val) == 0:
                raise ValueError('Expected a non-empty tuple')
        if not serialize:
            val = tuple(val) if typehint_origin in {Tuple, tuple} else set(val)

    # List, Iterable or Sequence
    elif typehint_origin in sequence_origin_types:
        if append:
            if prev_val is None:
                prev_val = []
            elif not isinstance(prev_val, list):
                try:
                    prev_val = [adapt_typehints(prev_val, subtypehints[0], **adapt_kwargs)]
                except Exception:
                    prev_val = []
            val = prev_val + (val if isinstance(val, list) else [val])
            prev_val = prev_val + [None] * (len(val) if isinstance(val, list) else 1)
        if isinstance(val, NestedArg) and subtypehints is not None:
            val = (prev_val[:-1] if isinstance(prev_val, list) else []) + [val]
        elif not isinstance(val, list):
            raise ValueError(f'Expected a List but got "{val}"')
        if subtypehints is not None:
            for n, v in enumerate(val):
                adapt_kwargs_n = {**adapt_kwargs, 'prev_val': prev_val[n]} if isinstance(prev_val, list) else adapt_kwargs
                val[n] = adapt_typehints(v, subtypehints[0], **adapt_kwargs_n)

    # Dict, Mapping
    elif typehint_origin in mapping_origin_types:
        if isinstance(val, NestedArg):
            if isinstance(prev_val, dict):
                val = {**prev_val, val.key: val.val}
            else:
                val = {val.key: val.val}
        elif not isinstance(val, dict):
            raise ValueError(f'Expected a Dict but got "{val}"')
        if subtypehints is not None:
            if subtypehints[0] == int:
                cast = str if serialize else int
                val = {cast(k): v for k, v in val.items()}
            for k, v in val.items():
                if "linked_targets" in adapt_kwargs["sub_add_kwargs"]:
                    kwargs = deepcopy(adapt_kwargs)
                    sub_add_kwargs = kwargs["sub_add_kwargs"]
                    sub_add_kwargs["linked_targets"] = {t[len(k + "."):] for t in sub_add_kwargs["linked_targets"]
                                                        if t.startswith(k + ".")}
                    sub_add_kwargs["linked_targets"] = {t[len("init_args."):] if t.startswith("init_args.") else t
                                                        for t in sub_add_kwargs["linked_targets"]}
                else:
                    kwargs = adapt_kwargs
                val[k] = adapt_typehints(v, subtypehints[1], **kwargs)

    # Callable
    elif typehint_origin in callable_origin_types or typehint in callable_origin_types:
        if serialize:
            if is_subclass_spec(val):
                val = adapt_class_type(val, True, False, sub_add_kwargs)
            else:
                val = object_path_serializer(val)
        else:
            try:
                if isinstance(val, str):
                    val_obj = import_object(val)
                    if inspect.isclass(val_obj):
                        val = {'class_path': val}
                    elif callable(val_obj):
                        val = val_obj
                    else:
                        raise ImportError(f'Unexpected import object {val_obj}')
                if isinstance(val, (dict, Namespace)):
                    if not is_subclass_spec(val):
                        raise ImportError(f'Dict must include a class_path and optionally init_args, but got {val}')
                    val_class = import_object(val['class_path'])
                    if not (inspect.isclass(val_class) and callable_instances(val_class)):
                        raise ImportError(f'{val["class_path"]!r} is not a callable class.')
                    val['class_path'] = get_import_path(val_class)
                    val = adapt_class_type(val, False, instantiate_classes, sub_add_kwargs)
            except (ImportError, AttributeError, ParserError) as ex:
                raise ValueError(f'Type {typehint} expects a function or a callable class: {ex}')

    # Subclass
    elif not hasattr(typehint, '__origin__') and inspect.isclass(typehint):
        if isinstance(val, typehint):
            if serialize:
                val = serialize_class_instance(val)
            return val
        if serialize and isinstance(val, str):
            return val

        val_input = val
        val = subclass_spec_as_namespace(val, prev_val)
        if not is_subclass_spec(val):
            raise ValueError(
                f'Type {typehint} expects: a class path (str); or a dict with a class_path entry; '
                f'or a dict with init_args (if class path given previously). Got "{val_input}".'
            )

        try:
            val_class = import_object(resolve_class_path_by_name(typehint, val['class_path']))
            if not is_subclass(val_class, typehint):
                raise ValueError(f'"{val["class_path"]}" is not a subclass of {typehint}')
            val['class_path'] = get_import_path(val_class)
            val = adapt_class_type(val, serialize, instantiate_classes, sub_add_kwargs, prev_val=prev_val)
        except (ImportError, AttributeError, AssertionError, ParserError) as ex:
            class_path = val if isinstance(val, str) else val['class_path']
            e = indent_text(f'\n- {ex}')
            raise ValueError(f'Problem with given class_path "{class_path}":{e}') from ex

    return val


def is_subclass_spec(val):
    is_class = isinstance(val, (dict, Namespace)) and 'class_path' in val
    if is_class:
        keys = getattr(val, '__dict__', val).keys()
        is_class = len(set(keys)-{'class_path', 'init_args', 'dict_kwargs', '__path__'}) == 0
    return is_class


def subclass_spec_as_namespace(val, prev_val=None):
    if isinstance(val, str):
        return Namespace(class_path=val)
    if isinstance(val, NestedArg):
        key, val = val
        if '.' not in key:
            root_key = key
        else:
            if key.startswith('init_args.'):
                root_key = 'init_args'
                key = key[len('init_args.'):]
                val = Namespace({key: val})
            elif key.startswith('dict_kwargs.'):
                root_key = 'dict_kwargs'
                key = key[len('dict_kwargs.'):]
                val = {key: val}
            else:
                root_key = 'init_args'
                val = NestedArg(key=key, val=val)
        val = Namespace({root_key: val})
        if isinstance(prev_val, str):
            prev_val = Namespace(class_path=prev_val)
    if isinstance(val, dict):
        val = Namespace(val)
    if 'init_args' in val and isinstance(val['init_args'], dict):
        val['init_args'] = Namespace(val['init_args'])
    if not is_subclass_spec(val) and isinstance(prev_val, (Namespace, dict)) and 'class_path' in prev_val:
        if 'init_args' in val or 'dict_kwargs' in val:
            val['class_path'] = prev_val['class_path']
        else:
            val = Namespace(class_path=prev_val['class_path'], init_args=val)
    return val


def get_all_subclass_paths(cls: Type) -> List[str]:
    subclass_list = []

    def is_local(cl):
        return '.<locals>.' in getattr(cl, '__qualname__', '.<locals>.')

    def is_private(class_path):
        return '._' in class_path

    def add_subclasses(cl):
        class_path = get_import_path(cl)
        if is_local(cl) or issubclass(cl, LazyInitBaseClass):
            return
        if not (inspect.isabstract(cl) or is_private(class_path)):
            subclass_list.append(class_path)
        for subclass in cl.__subclasses__() if hasattr(cl, '__subclasses__') else []:
            add_subclasses(subclass)

    if get_typehint_origin(cls) in {Union, Type, type}:
        for arg in cls.__args__:
            if ActionTypeHint.is_subclass_typehint(arg) and arg not in {object, type}:
                add_subclasses(arg)
    else:
        add_subclasses(cls)

    return subclass_list


def resolve_class_path_by_name(cls: Type, name: str) -> str:
    class_path = name
    if '.' not in class_path:
        subclass_dict = defaultdict(list)
        for subclass in get_all_subclass_paths(cls):
            subclass_name = subclass.rsplit('.', 1)[1]
            subclass_dict[subclass_name].append(subclass)
        if name in subclass_dict:
            name_subclasses = subclass_dict[name]
            if len(name_subclasses) > 1:
                raise ValueError(
                    f'Multiple subclasses with name {name}. Give the full class path to '
                    f'avoid ambiguity: {", ".join(name_subclasses)}.'
                )
            class_path = name_subclasses[0]
    return class_path


dump_kwargs: ContextVar = ContextVar('dump_kwargs', default={})


@contextmanager
def dump_kwargs_context(kwargs):
    dump_kwargs.set(kwargs if kwargs else {})
    yield


def discard_init_args_on_class_path_change(parser_or_action, prev_val, value):
    if prev_val and 'init_args' in prev_val and prev_val['class_path'] != value.class_path:
        parser = parser_or_action
        if isinstance(parser_or_action, ActionTypeHint):
            sub_add_kwargs = getattr(parser_or_action, 'sub_add_kwargs', {})
            parser = ActionTypeHint.get_class_parser(value.class_path, sub_add_kwargs)
        prev_val = subclass_spec_as_namespace(prev_val)
        del_args = {}
        for key, val in list(prev_val.init_args.__dict__.items()):
            action = _find_action(parser, key)
            if action:
                with lenient_check_context(lenient=False):
                    try:
                        parser._check_value_key(action, val, key, Namespace())
                    except Exception:
                        action = None
            if not action:
                del_args[key] = prev_val.init_args.pop(key)
        if del_args:
            warnings.warn(
                f'Due to class_path change from {prev_val.class_path!r} to {value.class_path!r}, '
                f'discarding init_args: {del_args}.'
            )


def adapt_class_type(value, serialize, instantiate_classes, sub_add_kwargs, prev_val=None):
    value = subclass_spec_as_namespace(value)
    val_class = import_object(value.class_path)
    parser = ActionTypeHint.get_class_parser(val_class, sub_add_kwargs)

    # No need to re-create the linked arg but just "inform" the corresponding parser actions that it exists upstream.
    for target in sub_add_kwargs.get('linked_targets', []):
        split_index = target.find(".")
        if split_index != -1:
            split = ".init_args." if target[split_index:].startswith(".init_args.") else "."

            parent_key, key = target.split(split, maxsplit=1)

            action = next(a for a in parser._actions if a.dest == parent_key)

            sub_add_kwargs = getattr(action, 'sub_add_kwargs')
            sub_add_kwargs.setdefault('linked_targets', set())
            sub_add_kwargs['linked_targets'].add(key)

            break

    discard_init_args_on_class_path_change(parser, prev_val, value)

    dict_kwargs = value.pop('dict_kwargs', {})
    init_args = value.get('init_args', Namespace())

    if instantiate_classes:
        init_args = parser.instantiate_classes(init_args)
        if not sub_add_kwargs.get('instantiate', True):
            if init_args:
                value['init_args'] = init_args
            return value
        return val_class(**{**init_args, **dict_kwargs})

    if isinstance(init_args, NestedArg):
        value['init_args'] = parser.parse_args(
            [f'--{init_args.key}={init_args.val}'],
            namespace=prev_val.init_args.clone(),
            defaults=sub_defaults.get(),
        )
        return value

    if serialize:
        if init_args:
            value['init_args'] = load_value(parser.dump(init_args, **dump_kwargs.get()))
    else:
        if isinstance(dict_kwargs, dict):
            for key in list(dict_kwargs.keys()):
                if _find_action(parser, key):
                    init_args[key] = dict_kwargs.pop(key)
        elif dict_kwargs:
            init_args['dict_kwargs'] = dict_kwargs
            dict_kwargs = None
        init_args = parser.parse_object(init_args, defaults=sub_defaults.get())
        if init_args:
            value['init_args'] = init_args
    if dict_kwargs:
        if isinstance(prev_val, Namespace) and prev_val.get('class_path') == value['class_path'] and prev_val.get('dict_kwargs'):
            dict_kwargs.update(prev_val.get('dict_kwargs'))
        value['dict_kwargs'] = dict_kwargs
    return value


def not_append_diff(val1, val2):
    if isinstance(val1, list) and isinstance(val2, list):
        val1 = [x.get('class_path') if is_subclass_spec(x) else x for x in val1]
        val2 = [x.get('class_path') if is_subclass_spec(x) else x for x in val2]
    return val1 != val2


def get_typehint_subtypes(typehint, append):
    subtypes = getattr(typehint, '__args__', None)
    if append and subtypes:
        subtypes = sorted(subtypes, key=lambda x: get_typehint_origin(x) not in sequence_origin_types)
    return subtypes


def get_typehint_origin(typehint):
    if not hasattr(typehint, '__origin__') and get_import_path(typehint.__class__) == 'types.UnionType':
        return Union
    return getattr(typehint, '__origin__', None)


def is_ellipsis_tuple(typehint):
    return typehint.__origin__ in {Tuple, tuple} and len(typehint.__args__) > 1 and typehint.__args__[1] == Ellipsis


def is_optional(annotation, ref_type):
    """Checks whether a type annotation is an optional for one type class."""
    return get_typehint_origin(annotation) == Union and \
        len(annotation.__args__) == 2 and \
        any(NoneType == a for a in annotation.__args__) and \
        any(is_subclass(a, ref_type) for a in annotation.__args__)


def is_enum_type(annotation):
    return is_subclass(annotation, Enum) or \
        (get_typehint_origin(annotation) == Union and
         any(is_subclass(a, Enum) for a in annotation.__args__))


def is_callable_type(annotation):
    def is_callable(a):
        return (get_typehint_origin(a) or a) in callable_origin_types or a in callable_origin_types
    return is_callable(annotation) or \
        (get_typehint_origin(annotation) == Union and
         any(is_callable(a) for a in annotation.__args__))


def typehint_from_action(action_or_typehint):
    if isinstance(action_or_typehint, Action):
        action_or_typehint = getattr(action_or_typehint, '_typehint', None)
    return action_or_typehint


def type_to_str(obj):
    if obj in {bool, tuple} or is_subclass(obj, (int, float, str, Enum)):
        return obj.__name__
    elif obj is not None:
        return re.sub(r'[A-Za-z0-9_<>.]+\.', '', str(obj)).replace('NoneType', 'null')


def literal_to_str(val):
    return 'null' if val is None else str(val)


def typehint_metavar(typehint):
    """Generates a metavar for some types."""
    metavar = None
    typehint_origin = get_typehint_origin(typehint) or typehint
    if typehint == bool:
        metavar = '{true,false}'
    elif is_optional(typehint, bool):
        metavar = '{true,false,null}'
    elif typehint_origin == Literal:
        args = typehint.__args__
        metavar = iter_to_set_str(literal_to_str(a) for a in args)
    elif is_subclass(typehint, Enum):
        enum = typehint
        metavar = iter_to_set_str(enum.__members__)
    elif is_optional(typehint, Enum):
        enum = typehint.__args__[0]
        metavar = iter_to_set_str(list(enum.__members__.keys())+['null'])
    elif typehint_origin in tuple_set_origin_types:
        metavar = '[ITEM,...]'
    return metavar


def serialize_class_instance(val):
    type_val = type(val)
    val = str(val)
    warning(f"""
        Not possible to serialize an instance of {type_val}. It will be
        represented as the string {val}. If this was set as a default, consider
        using lazy_instance.
    """)
    return val


def callable_instances(cls: Type):
    # https://stackoverflow.com/a/71568161/2732151
    return isinstance(getattr(cls, '__call__', None), FunctionType)


def check_lazy_kwargs(class_type: Type, lazy_kwargs: dict):
    if lazy_kwargs:
        from .core import ArgumentParser
        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(class_type)
        try:
            parser.parse_object(lazy_kwargs)
        except ParserError as ex:
            raise ValueError(str(ex)) from ex


class LazyInitBaseClass:

    def __init__(self, class_type: Type, lazy_kwargs: dict):
        check_lazy_kwargs(class_type, lazy_kwargs)
        self._lazy_class_type = class_type
        self._lazy_class_path = get_import_path(class_type)
        self._lazy_kwargs = lazy_kwargs
        self._lazy_methods = {}
        seen_methods: Dict = {}
        for name, _ in inspect.getmembers(class_type, predicate=inspect.isfunction):
            method = getattr(self, name)
            if not inspect.ismethod(method) or \
               name in {'__init__', '_lazy_init', '_lazy_init_then_call_method', 'lazy_get_init_data'}:
                continue
            assert name not in self.__dict__
            self._lazy_methods[name] = method
            if method in seen_methods:
                self.__dict__[name] = seen_methods[method]
            else:
                self.__dict__[name] = partial(self._lazy_init_then_call_method, name)
                seen_methods[method] = self.__dict__[name]

    def _lazy_init(self):
        for name in self._lazy_methods.keys():
            del self.__dict__[name]
        super().__init__(**self._lazy_kwargs)

    def _lazy_init_then_call_method(self, method_name, *args, **kwargs):
        self._lazy_init()
        return getattr(self, method_name)(*args, **kwargs)

    def lazy_get_init_data(self):
        init_args = Namespace(self._lazy_kwargs)
        if is_final_class(self._lazy_class_type):
            return init_args
        init = Namespace(class_path=self._lazy_class_path)
        if len(self._lazy_kwargs) > 0:
            init['init_args'] = init_args
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
        'LazyInstance_'+class_type.__name__,
        (LazyInitBaseClass, class_type),
        {'__doc__': f'Class for lazy instances of {class_type}'},
    )
    return lazy_init_class(class_type, kwargs)
