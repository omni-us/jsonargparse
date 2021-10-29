"""Action to support type hints."""

import copy
import inspect
import os
import re
import typing
import warnings
import yaml
from argparse import Action
from collections.abc import Iterable as abcIterable
from collections.abc import Mapping as abcMapping
from collections.abc import MutableMapping as abcMutableMapping
from collections.abc import Set as abcSet
from collections.abc import MutableSet as abcMutableSet
from collections.abc import Sequence as abcSequence
from collections.abc import MutableSequence as abcMutableSequence
from enum import Enum
from functools import partial
from typing import (
    Any,
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

from .actions import _is_action_value_list
from .namespace import Namespace
from .typing import get_import_path, is_final_class, object_path_serializer, registered_types
from .optionals import (
    argcomplete_warn_redraw_prompt,
    files_completer,
)
from .util import (
    change_to_path_dir,
    import_object,
    ParserError,
    Path,
    NoneType,
    yamlParserError,
    yamlScannerError,
    indent_text,
    _issubclass,
    _load_config,
)


__all__ = ['ActionTypeHint', 'lazy_instance']


Literal = getattr(typing, 'Literal', False)


root_types = {
    bool,
    Any,
    Literal,
    Type, type,
    Union,
    List, list, Iterable, Sequence, MutableSequence, abcIterable, abcSequence, abcMutableSequence,
    Tuple, tuple,
    Set, set, frozenset, MutableSet, abcMutableSet,
    Dict, dict, Mapping, MutableMapping, abcMapping, abcMutableMapping,
}

leaf_types = {
    str,
    int,
    float,
    bool,
    NoneType,
}

not_subclass_types: Set = set(k for k in registered_types.keys() if not isinstance(k, tuple))
not_subclass_types = not_subclass_types.union(leaf_types).union(root_types)

tuple_set_origin_types = {Tuple, tuple, Set, set, frozenset, MutableSet, abcSet, abcMutableSet}
sequence_origin_types = {List, list, Iterable, Sequence, MutableSequence, abcIterable, abcSequence,
                         abcMutableSequence}
mapping_origin_types = {Dict, dict, Mapping, MutableMapping, abcMapping, abcMutableMapping}


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


    @staticmethod
    def is_supported_typehint(typehint, full=False):
        """Whether the given type hint is supported."""
        supported = \
            typehint in root_types or \
            getattr(typehint, '__origin__', None) in root_types or \
            typehint in registered_types or \
            _issubclass(typehint, Enum) or \
            ActionTypeHint.is_class_typehint(typehint)
        if full and supported:
            typehint_origin = getattr(typehint, '__origin__', typehint)
            if typehint_origin in root_types and typehint_origin != Literal:
                for typehint in getattr(typehint, '__args__', []):
                    if typehint == Ellipsis or (typehint_origin == type and isinstance(typehint, TypeVar)):
                        continue
                    if not (typehint in leaf_types or ActionTypeHint.is_supported_typehint(typehint, full=True)):
                        return False
        return supported


    @staticmethod
    def is_class_typehint(typehint, only_subclasses=False):
        if isinstance(typehint, Action):
            typehint = getattr(typehint, '_typehint', None)
        typehint_origin = getattr(typehint, '__origin__', None)
        if typehint_origin == Union:
            subtypes = [a for a in typehint.__args__ if a != NoneType]
            return all(ActionTypeHint.is_class_typehint(s, only_subclasses) for s in subtypes)
        if only_subclasses and is_final_class(typehint):
            return False
        return inspect.isclass(typehint) and typehint not in not_subclass_types and typehint_origin is None


    @staticmethod
    def is_subclass_typehint(typehint):
        return ActionTypeHint.is_class_typehint(typehint, True)


    @staticmethod
    def is_mapping_class_typehint(typehint, only_subclasses=False):
        if isinstance(typehint, Action):
            typehint = getattr(typehint, '_typehint', None)
        typehint_origin = getattr(typehint, '__origin__', None)
        if typehint_origin not in mapping_origin_types:
            return False
        return ActionTypeHint.is_class_typehint(getattr(typehint, '__args__')[1], only_subclasses=only_subclasses)


    @staticmethod
    def parse_subclass_arg(parser, arg_string):
        if '.class_path' in arg_string or '.init_args.' in arg_string:
            if '.class_path' in arg_string:
                arg_base, explicit_arg = arg_string.rsplit('.class_path', 1)
            else:
                arg_base, init_arg = arg_string.rsplit('.init_args.', 1)
                match = re.match(r'([\w_]+)(|=.*)$', init_arg)
                if match:
                    explicit_arg = match.groups()[1]
            action = parser._option_string_actions.get(arg_base)
            if action:
                if explicit_arg:
                    arg_string = arg_string[:-len(explicit_arg)]
                    explicit_arg = explicit_arg[1:]
                else:
                    explicit_arg = None
                return action, arg_string, explicit_arg


    def serialize(self, value):
        sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
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
            cfg, val, opt_str = args[1:]
            if isinstance(opt_str, str):
                if opt_str.endswith('.class_path'):
                    cfg_dest = cfg.get(self.dest, Namespace())
                    if cfg_dest.get('class_path') == val:
                        return
                    elif cfg_dest.get('init_args') is not None and cfg_dest.get('init_args') != Namespace():
                        warnings.warn(
                            f'Argument {opt_str}={val} implies discarding init_args {cfg_dest.get("init_args").as_dict()} '
                            f'defined for class_path {cfg_dest.get("class_path")}'
                        )
                    val = Namespace(class_path=val)
                elif '.init_args.' in opt_str:
                    match = re.match(r'.+\.init_args\.([^.]+)$', opt_str)
                    cfg_dest = cfg.get(self.dest, Namespace())
                    init_args = cfg_dest.get('init_args', Namespace())
                    init_args[match.groups()[0]] = val
                    val = cfg_dest
            val = self._check_type(val)
        if 'cfg_dest' in locals():
            args[1].update(val, self.dest)
        else:
            setattr(args[1], self.dest, val)


    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                orig_val = val
                try:
                    val, config_path = _load_config(val, enable_path=self._enable_path)
                except (yamlParserError, yamlScannerError):
                    config_path = None
                path_meta = val.pop('__path__') if isinstance(val, dict) and '__path__' in val else None
                sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
                try:
                    with change_to_path_dir(config_path):
                        val = adapt_typehints(val, self._typehint, sub_add_kwargs=sub_add_kwargs)
                except ValueError as ex:
                    if isinstance(val, (int, float)) and config_path is None:
                        val = adapt_typehints(orig_val, self._typehint, sub_add_kwargs=sub_add_kwargs)
                    else:
                        if self._enable_path and config_path is None and isinstance(orig_val, str):
                            msg = f'\n- Expected a config path but "{orig_val}" either not accessible or invalid.\n- '
                            raise type(ex)(msg+str(ex)) from ex
                        raise ex
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, (Namespace, dict)) and config_path is not None:
                    val['__path__'] = config_path
                value[num] = val
            except (TypeError, ValueError) as ex:
                elem = '' if not islist else f' element {num+1}'
                raise TypeError(f'Parser key "{self.dest}"{elem}: {ex}') from ex
        return value if islist else value[0]


    def instantiate_classes(self, val):
        sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
        return adapt_typehints(val, self._typehint, instantiate_classes=True, sub_add_kwargs=sub_add_kwargs)


    @staticmethod
    def get_class_parser(val_class, sub_add_kwargs=None):
        if isinstance(val_class, str):
            val_class = import_object(val_class)
        from .core import ArgumentParser
        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(val_class, **(sub_add_kwargs or {}))
        return parser


    def completer(self, prefix, **kwargs):
        """Used by argcomplete, validates value and shows expected type."""
        if self._typehint == bool:
            return ['true', 'false']
        elif is_optional(self._typehint, bool):
            return ['true', 'false', 'null']
        elif _issubclass(self._typehint, Enum):
            enum = self._typehint
            return list(enum.__members__.keys())
        elif is_optional(self._typehint, Enum):
            enum = self._typehint.__args__[0]
            return list(enum.__members__.keys())+['null']
        elif is_optional(self._typehint, Path):
            return ['null'] + sorted(files_completer(prefix, **kwargs))
        elif chr(int(os.environ['COMP_TYPE'])) == '?':
            try:
                if prefix.strip() == '':
                    raise ValueError()
                self._check_type(prefix)
                msg = 'value already valid, '
            except (TypeError, ValueError, yamlParserError, yamlScannerError):
                msg = 'value not yet valid, '
            msg += 'expected type '+type_to_str(self._typehint)
            return argcomplete_warn_redraw_prompt(prefix, msg)


def adapt_typehints(val, typehint, serialize=False, instantiate_classes=False, sub_add_kwargs=None):

    adapt_kwargs = {
        'serialize': serialize,
        'instantiate_classes': instantiate_classes,
        'sub_add_kwargs': sub_add_kwargs or {},
    }
    subtypehints = getattr(typehint, '__args__', None)
    typehint_origin = getattr(typehint, '__origin__', typehint)

    # Any
    if typehint == Any:
        type_val = type(val)
        if type_val in registered_types or _issubclass(type_val, Enum):
            val = adapt_typehints(val, type_val, **adapt_kwargs)
        elif isinstance(val, str):
            try:
                val, _ = _load_config(val, enable_path=False)
            except (yamlParserError, yamlScannerError):
                pass

    # Literal
    elif typehint_origin == Literal:
        if val not in subtypehints:
            raise ValueError(f'Expected a {typehint} but got "{val}"')

    # Basic types
    elif typehint in leaf_types:
        if not isinstance(val, typehint):
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
    elif _issubclass(typehint, Enum):
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
               (typehint not in {Type, type} and not _issubclass(val, subtypehints[0])):
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
        if not isinstance(val, list):
            raise ValueError(f'Expected a List but got "{val}"')
        if subtypehints is not None:
            for n, v in enumerate(val):
                val[n] = adapt_typehints(v, subtypehints[0], **adapt_kwargs)

    # Dict, Mapping
    elif typehint_origin in mapping_origin_types:
        if isinstance(val, Namespace):
            val = val.as_dict()
        if not isinstance(val, dict):
            raise ValueError(f'Expected a Dict but got "{val}"')
        if subtypehints is not None:
            if subtypehints[0] == int:
                cast = str if serialize else int
                val = {cast(k): v for k, v in val.items()}
            for k, v in val.items():
                if "linked_targets" in adapt_kwargs["sub_add_kwargs"]:
                    kwargs = copy.deepcopy(adapt_kwargs)
                    sub_add_kwargs = kwargs["sub_add_kwargs"]
                    sub_add_kwargs["linked_targets"] = {t[len(k + "."):] for t in sub_add_kwargs["linked_targets"]
                                                        if t.startswith(k + ".")}
                    sub_add_kwargs["linked_targets"] = {t[len("init_args."):] if t.startswith("init_args.") else t
                                                        for t in sub_add_kwargs["linked_targets"]}
                else:
                    kwargs = adapt_kwargs
                val[k] = adapt_typehints(v, subtypehints[1], **kwargs)

    # Final class
    elif is_final_class(typehint):
        if isinstance(val, dict):
            val = Namespace(val)
        if not isinstance(val, Namespace):
            raise ValueError(f'Expected a Dict/Namespace but got "{val}"')
        val = adapt_class_type(typehint, val, serialize, instantiate_classes, sub_add_kwargs)

    # Subclass
    elif not hasattr(typehint, '__origin__') and inspect.isclass(typehint):
        if isinstance(val, typehint):
            if serialize:
                val = str(val)
                warnings.warn(
                    f'Not possible to serialize an instance of {typehint.__name__}. It will be represented as the '
                    f'string {val}. If this was set as a default, consider setting using lazy_instance.'
                )
            return val
        if serialize and isinstance(val, str):
            return val
        if not (isinstance(val, str) or is_class_object(val)):
            raise ValueError(f'Type {typehint} expects an str or a Dict/Namespace with a class_path entry but got "{val}"')
        try:
            if isinstance(val, str):
                val = Namespace(class_path=val)
            elif isinstance(val, dict):
                val = Namespace(val)
            val_class = import_object(val['class_path'])
            if isinstance(val.get('init_args'), dict):
                val['init_args'] = Namespace(val['init_args'])
            if not _issubclass(val_class, typehint):
                raise ValueError(f'"{val["class_path"]}" is not a subclass of {typehint.__name__}')
            init_args = val.get('init_args', Namespace())
            adapted = adapt_class_type(val_class, init_args, serialize, instantiate_classes, sub_add_kwargs)
            if instantiate_classes and sub_add_kwargs.get('instantiate', True):
                val = adapted
            elif adapted is not None:
                val['init_args'] = adapted
        except (ImportError, AttributeError, AssertionError, ParserError) as ex:
            class_path = val if isinstance(val, str) else val['class_path']
            e = indent_text(f'\n- {ex}')
            raise ValueError(f'Problem with given class_path "{class_path}":{e}') from ex

    return val


def is_class_object(val):
    return isinstance(val, (dict, Namespace)) and 'class_path' in val


def adapt_class_type(val_class, init_args, serialize, instantiate_classes, sub_add_kwargs):
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

    if instantiate_classes:
        init_args = parser.instantiate_subclasses(init_args)
        if not sub_add_kwargs.get('instantiate', True):
            return init_args
        return val_class(**init_args)
    if serialize:
        init_args = None if init_args == Namespace() else yaml.safe_load(parser.dump(init_args))
    else:
        init_args = parser.parse_object(init_args)
    return init_args


def is_ellipsis_tuple(typehint):
    return typehint.__origin__ in {Tuple, tuple} and len(typehint.__args__) > 1 and typehint.__args__[1] == Ellipsis


def is_optional(annotation, ref_type):
    """Checks whether a type annotation is an optional for one type class."""
    return hasattr(annotation, '__origin__') and \
        annotation.__origin__ == Union and \
        len(annotation.__args__) == 2 and \
        any(NoneType == a for a in annotation.__args__) and \
        any(_issubclass(a, ref_type) for a in annotation.__args__)


def type_to_str(obj):
    if _issubclass(obj, (bool, int, float, str, Enum)):
        return obj.__name__
    elif obj is not None:
        return re.sub(r'[a-z_.]+\.', '', str(obj)).replace('NoneType', 'null')


def typehint_metavar(typehint):
    """Generates a metavar for some types."""
    metavar = None
    if typehint == bool:
        metavar = '{true,false}'
    elif is_optional(typehint, bool):
        metavar = '{true,false,null}'
    elif _issubclass(typehint, Enum):
        enum = typehint
        metavar = '{'+','.join(list(enum.__members__.keys()))+'}'
    elif is_optional(typehint, Enum):
        enum = typehint.__args__[0]
        metavar = '{'+','.join(list(enum.__members__.keys())+['null'])+'}'
    return metavar


class LazyInitBaseClass:

    def __init__(self, class_type: Type, lazy_kwargs: dict):
        self._lazy_class_type = class_type
        self._lazy_class_path = get_import_path(class_type)
        self._lazy_kwargs = lazy_kwargs
        self._lazy_methods = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name in {'__init__', '_lazy_init', '_lazy_init_then_call_method', 'lazy_get_init_data'}:
                continue
            self._lazy_methods[name] = method
            setattr(self, name, partial(self._lazy_init_then_call_method, method_name=name))

    def _lazy_init(self):
        for name, method in self._lazy_methods.items():
            setattr(self, name, method)
        super().__init__(**self._lazy_kwargs)

    def _lazy_init_then_call_method(self, method_name, *args, **kwargs):
        self._lazy_init()
        return getattr(self, method_name)(*args, **kwargs)

    def lazy_get_init_data(self):
        init = Namespace(class_path=self._lazy_class_path)
        if len(self._lazy_kwargs) > 0:
            init['init_args'] = Namespace(self._lazy_kwargs)
        return init


ClassType = TypeVar('ClassType')


def lazy_instance(class_type: Type[ClassType], **kwargs) -> ClassType:
    """Instantiates a lazy instance of the given type.

    By lazy it is meant that the __init__ is delayed unit the first time that a
    method of the instance is called. It also provides a `lazy_get_init_data` method
    useful for serializing.

    Note: Only supported in CPython.

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
