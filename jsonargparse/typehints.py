"""Action to support type hints."""

import inspect
import os
import re
import warnings
import yaml
from argparse import Action, Namespace
from collections.abc import Iterable as abcIterable
from collections.abc import Sequence as abcSequence
from enum import Enum
from functools import partial
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, Type, TypeVar, Union

try:
    from typing import Literal  # type: ignore
except ImportError:
    Literal = False

from .actions import _is_action_value_list
from .typing import get_import_path, is_final_class, object_path_serializer, registered_types
from .optionals import (
    argcomplete_warn_redraw_prompt,
    files_completer,
    ModuleNotFound,
)
from .util import (
    change_to_path_dir,
    import_object,
    ParserError,
    Path,
    namespace_to_dict,
    NoneType,
    yamlParserError,
    yamlScannerError,
    _issubclass,
    _load_config,
)


__all__ = ['ActionTypeHint', 'lazy_instance']


root_types = {
    bool,
    Any,
    Literal,
    Type, type,
    Union,
    List, list, Iterable, Sequence, abcIterable, abcSequence,
    Tuple, tuple,
    Set, set,
    Dict, dict,
}

leaf_types = {
    str,
    int,
    float,
    bool,
    NoneType,
}

not_subclass_types = set(k for k in registered_types.keys() if not isinstance(k, tuple))  # type: Set
not_subclass_types = not_subclass_types.union(leaf_types).union(root_types)


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
                raise ValueError('Unsupported type hint '+str(typehint)+'.')
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
                    val = {'class_path': val}
                elif '.init_args.' in opt_str:
                    match = re.match(r'.+\.init_args\.([^.]+)$', opt_str)
                    if match and isinstance(getattr(cfg, self.dest, None), dict):
                        init_args = getattr(cfg, self.dest).get('init_args', {})
                        init_args[match.groups()[0]] = val
                        val = getattr(cfg, self.dest)
            val = self._check_type(val)
        setattr(args[1], self.dest, val)


    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        for num, val in enumerate(value):
            try:
                orig_val = val
                try:
                    val, config_path = _load_config(val, enable_path=self._enable_path, flat_namespace=False)
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
                        raise ex
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, dict) and config_path is not None:
                    val['__path__'] = config_path
                value[num] = val
            except (TypeError, ValueError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex)) from ex
        return value if islist else value[0]


    def _instantiate_classes(self, val):
        sub_add_kwargs = getattr(self, 'sub_add_kwargs', {})
        return adapt_typehints(val, self._typehint, instantiate_classes=True, sub_add_kwargs=sub_add_kwargs)


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
                val = _load_config(val, enable_path=False, flat_namespace=False)[0]
            except (yamlParserError, yamlScannerError):
                pass

    # Literal
    elif typehint_origin == Literal:
        if val not in subtypehints:
            raise ValueError('Expected a '+str(typehint)+' but got "'+str(val)+'"')

    # Basic types
    elif typehint in leaf_types:
        if not isinstance(val, typehint):
            raise ValueError('Expected a '+str(typehint)+' but got "'+str(val)+'"')

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
                typehint[val]
        elif not serialize and not isinstance(val, typehint):
            val = typehint[val]

    # Type
    elif typehint in {Type, type} or typehint_origin in {Type, type}:
        if serialize:
            val = object_path_serializer(val)
        elif not serialize and not isinstance(val, Type):
            path = val
            val = import_object(val)
            if (typehint in {Type, type} and not isinstance(val, (Type, type))) or \
               (typehint != Type and typehint_origin in {Type, type} and not _issubclass(val, subtypehints[0])):
                raise ValueError('Value "'+str(path)+'" is not a '+str(typehint)+'.')

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
            raise ValueError('Value "'+str(val)+'" does not validate against any of the types in '+str(typehint)+':'+e)
        val = [v for v in vals if not isinstance(v, Exception)][0]

    # Tuple or Set
    elif typehint_origin in {Tuple, tuple, Set, set}:
        if not isinstance(val, (list, tuple, set)):
            raise ValueError('Expected a '+str(typehint_origin)+' but got "'+str(val)+'"')
        val = list(val)
        if subtypehints is not None:
            is_tuple = typehint_origin in {Tuple, tuple}
            is_ellipsis = is_ellipsis_tuple(typehint)
            if is_tuple and not is_ellipsis and len(val) != len(subtypehints):
                raise ValueError('Expected a tuple with '+str(len(subtypehints))+' elements but got "'+str(val)+'"')
            for n, v in enumerate(val):
                subtypehint = subtypehints[0 if is_ellipsis else n]
                val[n] = adapt_typehints(v, subtypehint, **adapt_kwargs)
            if is_tuple and len(val) == 0:
                raise ValueError('Expected a non-empty tuple')
        if not serialize:
            val = tuple(val) if typehint_origin in {Tuple, tuple} else set(val)

    # List, Iterable or Sequence
    elif typehint_origin in {List, list, Iterable, Sequence, abcIterable, abcSequence}:
        if not isinstance(val, list):
            raise ValueError('Expected a List but got "'+str(val)+'"')
        if subtypehints is not None:
            for n, v in enumerate(val):
                val[n] = adapt_typehints(v, subtypehints[0], **adapt_kwargs)

    # Dict
    elif typehint_origin in {Dict, dict}:
        if not isinstance(val, dict):
            raise ValueError('Expected a Dict but got "'+str(val)+'"')
        if subtypehints is not None:
            if subtypehints[0] == int:
                cast = str if serialize else int
                val = {cast(k): v for k, v in val.items()}
            for k, v in val.items():
                val[k] = adapt_typehints(v, subtypehints[1], **adapt_kwargs)

    # Final class
    elif is_final_class(typehint):
        if not isinstance(val, dict):
            raise ValueError('Expected a Dict but got "'+str(val)+'"')
        val = adapt_class_type(typehint, val, serialize, instantiate_classes, sub_add_kwargs)

    # Subclass
    elif not hasattr(typehint, '__origin__') and inspect.isclass(typehint):
        if isinstance(val, typehint):
            if isinstance(val, LazyInitBaseClass):
                val = val.get_init_dict()
            elif serialize:
                val = str(val)
                warnings.warn(
                    'Not possible to serialize an instance of ' + str(typehint.__name__) + '. It will be represented as the '
                    'string ' + val + '. If this was set as a default, consider setting a dict or using lazy_instance.'
                )
            return val
        if not (isinstance(val, str) or (isinstance(val, dict) and 'class_path' in val) or (isinstance(val, Namespace) and hasattr(val, 'class_path'))):
            raise ValueError('Expected an str or a Dict with a class_path entry but got "'+str(val)+'"')
        try:
            if isinstance(val, Namespace):
                val = namespace_to_dict(val)
            if isinstance(val, str):
                val_class = import_object(val)
                val = {'class_path': val}
            else:
                val_class = import_object(val['class_path'])
            if not _issubclass(val_class, typehint):
                raise ValueError('"'+val['class_path']+'" is not a subclass of '+typehint.__name__)
            init_args = val.get('init_args', {})
            adapted = adapt_class_type(val_class, init_args, serialize, instantiate_classes, sub_add_kwargs)
            if instantiate_classes and sub_add_kwargs.get('instantiate', True):
                val = adapted
            else:
                val['init_args'] = adapted
        except (ImportError, ModuleNotFound, AttributeError, AssertionError, ParserError) as ex:
            class_path = val if isinstance(val, str) else val['class_path']
            e = indent_text('\n- '+str(ex))
            raise ValueError('Problem with given class_path "'+class_path+'":'+e) from ex

    return val


def adapt_class_type(val_class, init_args, serialize, instantiate_classes, sub_add_kwargs):
    from .core import ArgumentParser
    parser = ArgumentParser(error_handler=None, parse_as_dict=True)
    parser.add_class_arguments(val_class, **sub_add_kwargs)
    if instantiate_classes:
        init_args = parser.instantiate_subclasses(init_args)
        if not sub_add_kwargs.get('instantiate', True):
            return init_args
        return val_class(**init_args)
    if serialize:
        init_args = yaml.safe_load(parser.dump(init_args))
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


def indent_text(text):
    return text.replace('\n', '\n  ')


class LazyInitBaseClass:

    def __init__(self, class_type: Type, lazy_kwargs: dict):
        self._lazy_class_type = class_type
        self._lazy_class_path = get_import_path(class_type)
        self._lazy_kwargs = lazy_kwargs
        self._lazy_methods = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name in {'__init__', '_lazy_init', '_lazy_init_then_call_method', 'get_init_dict'}:
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

    def get_init_dict(self):
        init_dict = {'class_path': self._lazy_class_path}
        if len(self._lazy_kwargs) > 0:
            init_dict['init_args'] = self._lazy_kwargs
        return init_dict


ClassType = TypeVar('ClassType')


def lazy_instance(class_type: Type[ClassType], **kwargs) -> ClassType:
    """Instantiates a lazy instance of the given type.

    By lazy it is meant that the init is delayed unit the first time that a
    method of the instance is called. It also provides a `get_init_dict` method
    useful for serializing.

    Note: Only supported in CPython.

    Args:
        class_type: The class to instantiate.
        **kwargs: Any keyword arguments to use for instantiation.
    """
    class LazyInitClass(LazyInitBaseClass, class_type):  # type: ignore
        pass

    return LazyInitClass(class_type, kwargs)
