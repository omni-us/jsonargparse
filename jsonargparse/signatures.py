"""Methods to add arguments based on class/method/function signatures."""

import dataclasses
import inspect
import re
from argparse import SUPPRESS
from typing import Any, Callable, List, Optional, Set, Tuple, Type, Union

from .actions import _ActionConfigLoad
from .optionals import get_doc_short_description
from .parameter_resolvers import get_signature_parameters, ParamData
from .typehints import ActionTypeHint, is_optional, LazyInitBaseClass
from .typing import is_final_class
from .util import get_import_path, is_subclass, iter_to_set_str, LoggerProperty

__all__ = [
    'compose_dataclasses',
    'SignatureArguments',
]


kinds = inspect._ParameterKind
inspect_empty = inspect._empty


class SignatureArguments(LoggerProperty):
    """Methods to add arguments based on signatures to an ArgumentParser instance."""

    def add_class_arguments(
        self,
        theclass: Type,
        nested_key: Optional[str] = None,
        as_group: bool = True,
        as_positional: bool = False,
        default: Optional[LazyInitBaseClass] = None,
        skip: Optional[Set[str]] = None,
        instantiate: bool = True,
        fail_untyped: bool = True,
        sub_configs: bool = False,
        linked_targets: Optional[Set[str]] = None,
    ) -> List[str]:
        """Adds arguments from a class based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            theclass: Class from which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            default: Default value used to override parameter defaults. Must be lazy_instance.
            skip: Names of parameters that should be skipped.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When not given a class.
            ValueError: When there are required parameters without at least one valid type.
        """
        if not inspect.isclass(theclass):
            raise ValueError(f'Expected "theclass" parameter to be a class type, got: {theclass}.')
        if default and not (isinstance(default, LazyInitBaseClass) and isinstance(default, theclass)):
            raise ValueError(f'Expected "default" parameter to be a lazy instance of the class, got: {default}.')

        added_args = self._add_signature_arguments(
            theclass,
            None,
            nested_key,
            as_group,
            as_positional,
            skip,
            fail_untyped,
            sub_configs=sub_configs,
            instantiate=instantiate,
            linked_targets=linked_targets,
        )

        if default:
            skip = skip or set()
            prefix = nested_key+'.' if nested_key else ''
            defaults = default.lazy_get_init_data().as_dict()
            if defaults:
                defaults = {prefix+k: v for k, v in defaults.items() if k not in skip}
                self.set_defaults(**defaults)  # type: ignore

        return added_args


    def add_method_arguments(
        self,
        theclass: Type,
        themethod: str,
        nested_key: Optional[str] = None,
        as_group: bool = True,
        as_positional: bool = False,
        skip: Optional[Set[str]] = None,
        fail_untyped: bool = True,
        sub_configs: bool = False,
    ) -> List[str]:
        """Adds arguments from a class based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            theclass: Class which includes the method.
            themethod: Name of the method for which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            skip: Names of parameters that should be skipped.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When not given a class or the name of a method of the class.
            ValueError: When there are required parameters without at least one valid type.
        """
        if not inspect.isclass(theclass):
            raise ValueError('Expected "theclass" argument to be a class object.')
        if not hasattr(theclass, themethod) or not callable(getattr(theclass, themethod)):
            raise ValueError('Expected "themethod" argument to be a callable member of the class.')

        return self._add_signature_arguments(
            theclass,
            themethod,
            nested_key,
            as_group,
            as_positional,
            skip,
            fail_untyped,
            sub_configs=sub_configs,
        )


    def add_function_arguments(
        self,
        function: Callable,
        nested_key: Optional[str] = None,
        as_group: bool = True,
        as_positional: bool = False,
        skip: Optional[Set[str]] = None,
        fail_untyped: bool = True,
        sub_configs: bool = False,
    ) -> List[str]:
        """Adds arguments from a function based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            function: Function from which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            skip: Names of parameters that should be skipped.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When not given a callable.
            ValueError: When there are required parameters without at least one valid type.
        """
        if not callable(function):
            raise ValueError('Expected "function" argument to be a callable object.')

        return self._add_signature_arguments(
            function,
            None,
            nested_key,
            as_group,
            as_positional,
            skip,
            fail_untyped,
            sub_configs=sub_configs,
        )


    def _add_signature_arguments(
        self,
        function_or_class,
        method_name,
        nested_key: Optional[str],
        as_group: bool = True,
        as_positional: bool = False,
        skip: Optional[Set[str]] = None,
        fail_untyped: bool = True,
        sub_configs: bool = False,
        instantiate: bool = True,
        linked_targets: Optional[Set[str]] = None,
    ) -> List[str]:
        """Adds arguments from parameters of objects based on signatures and docstrings.

        Args:
            function_or_class: Object from which to add arguments.
            method_name: Class method from which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            skip: Names of parameters that should be skipped.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When there are required parameters without at least one valid type.
        """
        params = get_signature_parameters(function_or_class, method_name, logger=self.logger)

        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, logger=self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        group = self._create_group_if_requested(component, nested_key, as_group, doc_group, instantiate=instantiate)

        ## Add parameter arguments ##
        added_args: List[str] = []
        for param in params:
            self._add_signature_parameter(
                group,
                nested_key,
                param,
                added_args,
                skip,
                fail_untyped=fail_untyped,
                sub_configs=sub_configs,
                linked_targets=linked_targets,
                as_positional=as_positional,
            )

        return added_args


    def _add_signature_parameter(
        self,
        group,
        nested_key: Optional[str],
        param,
        added_args: List[str],
        skip: Optional[Set[str]] = None,
        fail_untyped: bool = True,
        as_positional: bool = False,
        sub_configs: bool = False,
        instantiate: bool = True,
        linked_targets: Optional[Set[str]] = None,
        default: Any = inspect_empty,
        **kwargs
    ):
        name = param.name
        kind = param.kind
        annotation = param.annotation
        if default == inspect_empty:
            default = param.default
        is_required = default == inspect_empty
        src = (param.parent.__name__+'.' if param.parent else '')
        src += iter_to_set_str(x.__name__ for x in (param.component if isinstance(param.component, tuple) else [param.component]))
        skip_message = f'Skipping parameter "{name}" from "{src}" because of: '
        if not fail_untyped and annotation == inspect_empty:
            annotation = Any
            default = None if is_required else default
            is_required = False
        if is_required and linked_targets is not None and name in linked_targets:
            default = None
            is_required = False
        if kind in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD} or \
           (not is_required and name[0] == '_') or \
           (annotation == inspect_empty and not is_required and default is None):
            return
        elif skip and name in skip:
            self.logger.debug(skip_message+'Parameter requested to be skipped.')
            return
        if is_factory_class(default):
            default = param.parent.__dataclass_fields__[name].default_factory()
        if annotation == inspect_empty and not is_required:
            annotation = type(default)
        if 'help' not in kwargs:
            kwargs['help'] = param.doc
        if not is_required:
            kwargs['default'] = default
            if default is None and not is_optional(annotation, object):
                annotation = Optional[annotation]
        elif not as_positional:
            kwargs['required'] = True
        is_subclass_typehint = False
        is_final_class_typehint = is_final_class(annotation)
        dest = (nested_key+'.' if nested_key else '') + name
        args = [dest if is_required and as_positional else '--'+dest]
        if annotation in {str, int, float, bool} or \
           is_subclass(annotation, (str, int, float)) or \
           is_final_class_typehint or \
           is_pure_dataclass(annotation):
            kwargs['type'] = annotation
        elif annotation != inspect_empty:
            try:
                is_subclass_typehint = ActionTypeHint.is_subclass_typehint(annotation)
                kwargs['type'] = annotation
                sub_add_kwargs = None
                if is_subclass_typehint:
                    prefix = name + '.init_args.'
                    subclass_skip = {s[len(prefix):] for s in skip or [] if s.startswith(prefix)}
                    sub_add_kwargs = {'fail_untyped': fail_untyped, 'skip': subclass_skip}
                args = ActionTypeHint.prepare_add_argument(
                    args=args,
                    kwargs=kwargs,
                    enable_path=is_subclass_typehint and sub_configs,
                    container=group,
                    sub_add_kwargs=sub_add_kwargs,
                )
            except ValueError as ex:
                self.logger.debug(skip_message+str(ex))
        if 'type' in kwargs or 'action' in kwargs:
            sub_add_kwargs = {
                'fail_untyped': fail_untyped,
                'sub_configs': sub_configs,
                'instantiate': instantiate,
            }
            if is_final_class_typehint:
                kwargs.update(sub_add_kwargs)
            action = group.add_argument(*args, **kwargs)
            action.sub_add_kwargs = sub_add_kwargs
            if is_subclass_typehint and len(subclass_skip) > 0:
                action.sub_add_kwargs['skip'] = subclass_skip
            added_args.append(dest)
        elif is_required and fail_untyped:
            raise ValueError(f'Required parameter without a type for "{src}" parameter "{name}".')


    def add_dataclass_arguments(
        self,
        theclass: Type,
        nested_key: str,
        default: Union[Type, dict] = None,
        as_group: bool = True,
        **kwargs
    ) -> List[str]:
        """Adds arguments from a dataclass based on its field types and docstrings.

        Args:
            theclass: Class from which to add arguments.
            nested_key: Key for nested namespace.
            default: Value for defaults. Must be instance of or kwargs for theclass.
            as_group: Whether arguments should be added to a new argument group.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When not given a dataclass.
            ValueError: When default is not instance of or kwargs for theclass.
        """
        if not is_pure_dataclass(theclass):
            raise ValueError(f'Expected "theclass" argument to be a pure dataclass, given {theclass}')

        doc_group = get_doc_short_description(theclass, logger=self.logger)
        for key in ['help', 'title']:
            if key in kwargs and kwargs[key] is not None:
                doc_group = kwargs.pop(key)
        group = self._create_group_if_requested(theclass, nested_key, as_group, doc_group, config_load_type=theclass)

        defaults = {}
        if default is not None:
            if isinstance(default, dict):
                try:
                    default = theclass(**default)
                except TypeError:
                    pass
            if not isinstance(default, theclass):
                raise ValueError(f'Expected "default" argument to be an instance of "{theclass.__name__}" or its kwargs dict, given {default}')
            defaults = dataclasses.asdict(default)

        added_args: List[str] = []
        params = {p.name: p for p in get_signature_parameters(theclass, None, logger=self.logger)}
        for field in dataclasses.fields(theclass):
            self._add_signature_parameter(
                group,
                nested_key,
                params[field.name],
                added_args,
                default=defaults.get(field.name, inspect_empty),
            )

        return added_args


    def add_subclass_arguments(
        self,
        baseclass: Union[Type, Tuple[Type, ...]],
        nested_key: str,
        as_group: bool = True,
        skip: Optional[Set[str]] = None,
        instantiate: bool = True,
        required: bool = False,
        metavar: str = 'CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE',
        help: str = 'One or more arguments specifying "class_path" and "init_args" for any subclass of %(baseclass_name)s.',
        **kwargs
    ):
        """Adds arguments to allow specifying any subclass of the given base class.

        This adds an argument that requires a dictionary with a "class_path"
        entry which must be a import dot notation expression. Optionally any
        init arguments for the class can be given in the "init_args" entry.
        Since subclasses can have different init arguments, the help does not
        show the details of the arguments of the base class. Instead a help
        argument is added that will print the details for a given class path.

        Args:
            baseclass: Base class or classes to use to check subclasses.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            skip: Names of parameters that should be skipped.
            required: Whether the argument group is required.
            metavar: Variable string to show in the argument's help.
            help: Description of argument to show in the help.
            **kwargs: Additional parameters like in add_class_arguments.

        Raises:
            ValueError: When given an invalid base class.
        """
        if is_final_class(baseclass):
            raise ValueError("Not allowed for classes that are final.")
        if type(baseclass) is not tuple:
            baseclass = (baseclass,)  # type: ignore
        if not all(inspect.isclass(c) for c in baseclass):
            raise ValueError('Expected "baseclass" argument to be a class or a tuple of classes.')

        doc_group = get_doc_short_description(baseclass[0], logger=self.logger)
        group = self._create_group_if_requested(
            baseclass,
            nested_key,
            as_group,
            doc_group,
            config_load=False,
            required=required,
            instantiate=False,
        )

        added_args: List[str] = []
        if skip is not None:
            skip = {nested_key+'.init_args.'+s for s in skip}
        param = ParamData(name=nested_key, annotation=Union[baseclass], component=baseclass)
        str_baseclass = iter_to_set_str(get_import_path(x) for x in baseclass)
        kwargs.update({
            'metavar': metavar,
            'help': (help % {'baseclass_name': str_baseclass}),
        })
        if 'default' not in kwargs:
            kwargs['default'] = SUPPRESS
        self._add_signature_parameter(
            group,
            None,
            param,
            added_args,
            skip,
            sub_configs=True,
            instantiate=instantiate,
            **kwargs
        )


    def _create_group_if_requested(self, obj, nested_key, as_group, doc_group, config_load=True, config_load_type=None, required=False, instantiate=True):
        if required:
            if nested_key is None:
                raise ValueError('A nested_key is mandatory to make required.')
            self.required_args.add(nested_key)

        group = self
        if as_group:
            if doc_group is None:
                doc_group = str(obj)
            name = obj.__name__ if nested_key is None else nested_key
            group = self.add_argument_group(strip_title(doc_group), name=name)
            if config_load and nested_key is not None:
                group.add_argument('--'+nested_key, action=_ActionConfigLoad(basetype=config_load_type))
            if inspect.isclass(obj) and nested_key is not None and instantiate:
                group.dest = nested_key
                group.group_class = obj
                group.instantiate_class = group_instantiate_class
        return group


def group_instantiate_class(group, cfg):
    try:
        value, parent, key = cfg.get_value_and_parent(group.dest)
    except KeyError:
        value = {}
        parent = cfg
        key = group.dest
        assert '.' not in key
    parent[key] = group.group_class(**value)


def strip_title(value):
    if value is not None:
        return re.sub(r'\.$', '', value.strip())


def is_factory_class(value):
    return value.__class__ == dataclasses._HAS_DEFAULT_FACTORY_CLASS


def is_pure_dataclass(value):
    if not inspect.isclass(value):
        return False
    classes = [c for c in inspect.getmro(value) if c != object]
    return all(dataclasses.is_dataclass(c) for c in classes)


def compose_dataclasses(*args):
    """Returns a pure dataclass inheriting all given dataclasses and properly handling __post_init__."""

    @dataclasses.dataclass
    class ComposedDataclass(*args):
        def __post_init__(self):
            for arg in args:
                if hasattr(arg, '__post_init__'):
                    arg.__post_init__(self)

    return ComposedDataclass
