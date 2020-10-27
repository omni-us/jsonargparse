"""Add arguments based on signatures."""

import enum
import inspect

from .actions import ActionEnum
from .jsonschema import ActionJsonSchema
from .optionals import docstring_parser_support, _import_docstring_parse


class SignatureArguments:
    """Methods to add arguments based on signatures to an ArgumentParser instance."""

    def add_class_arguments(self, theclass, nested_key=None, as_group=True):
        """Adds arguments from a class based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            theclass (class): Class from which to add arguments.
            nested_key (str or None): Key for nested namespace.
            as_group (bool): Whether arguments should be added to a new argument group.

        Returns:
            int: Number of arguments added.

        Raises:
            ValueError: When not given a class.
            ValueError: When there are positional arguments without at least one valid type.
        """
        if not inspect.isclass(theclass):
            raise ValueError('Expected a class object.')

        def docs_func(base):
            return [base.__init__.__doc__, base.__doc__]

        return self._add_signature_arguments(inspect.getmro(theclass), nested_key, as_group, docs_func)


    def add_method_arguments(self, theclass, themethod, nested_key=None, as_group=True):
        """Adds arguments from a class based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            theclass (class): Class which includes the method.
            themethod (str): Name of the method for which to add arguments.
            nested_key (str or None): Key for nested namespace.
            as_group (bool): Whether arguments should be added to a new argument group.

        Returns:
            int: Number of arguments added.

        Raises:
            ValueError: When not given a class or the name of a method of the class.
            ValueError: When there are positional arguments without at least one valid type.
        """
        if not inspect.isclass(theclass):
            raise ValueError('Expected a class object.')
        if not hasattr(theclass, themethod) or not callable(getattr(theclass, themethod)):
            raise ValueError('Expected the method to a callable member of the class.')

        def docs_func(base):
            return [base.__doc__]

        skip_first = False if isinstance(theclass.__dict__[themethod], staticmethod) else True
        themethod = getattr(theclass, themethod)

        return self._add_signature_arguments([themethod], nested_key, as_group, docs_func, skip_first=skip_first)


    def add_function_arguments(self, function, nested_key=None, as_group=True):
        """Adds arguments from a function based on its type hints and docstrings.

        Note: Keyword arguments without at least one valid type are ignored.

        Args:
            function (callable): Function from which to add arguments.
            nested_key (str or None): Key for nested namespace.
            as_group (bool): Whether arguments should be added to a new argument group.

        Returns:
            int: Number of arguments added.

        Raises:
            ValueError: When not given a callable.
            ValueError: When there are positional arguments without at least one valid type.
        """
        if not callable(function):
            raise ValueError('Expected a callable object.')

        def docs_func(base):
            return [base.__doc__]

        return self._add_signature_arguments([function], nested_key, as_group, docs_func)


    def _add_signature_arguments(self, objects, nested_key, as_group, docs_func, skip_first=False):
        """Adds arguments from arguments of objects based on signatures and docstrings.

        Args:
            objects (tuple or list): Objects from which to add signatures.
            nested_key (str or None): Key for nested namespace.
            as_group (bool): Whether arguments should be added to a new argument group.
            docs_func (callable): Function that returns docstrings for a given object.
            skip_first (bool): Whether to skip first argument, i.e., skip self of class methods.

        Returns:
            int: Number of arguments added.

        Raises:
            ValueError: When there are positional arguments without at least one valid type.
        """
        kinds = inspect._ParameterKind

        def update_has_args_kwargs(base, has_args=True, has_kwargs=True):
            params = list(inspect.signature(base).parameters.values())
            has_args &= any(p._kind == kinds.VAR_POSITIONAL for p in params)
            has_kwargs &= any(p._kind == kinds.VAR_KEYWORD for p in params)
            return has_args, has_kwargs

        ## Determine propagation of arguments ##
        add_types = [(True, True)]
        has_args, has_kwargs = update_has_args_kwargs(objects[0])
        for num in range(1, len(objects)):
            if not (has_args or has_kwargs):
                objects = objects[:num]
                break
            add_types.append((has_args, has_kwargs))
            has_args, has_kwargs = update_has_args_kwargs(objects[num], has_args, has_kwargs)

        ## Gather docstrings ##
        doc_group = None
        doc_params = {}
        if docstring_parser_support:
            docstring_parse = _import_docstring_parse('_add_signature_arguments')
            for base in objects:
                for doc in docs_func(base):
                    docstring = docstring_parse(doc)
                    if docstring.short_description and not doc_group:
                        doc_group = docstring.short_description
                    for param in docstring.params:
                        if param.arg_name not in doc_params:
                            doc_params[param.arg_name] = param.description

        ## Create group if requested ##
        group = self
        if as_group:
            if doc_group is None:
                doc_group = str(objects[0])
            name = objects[0].__name__ if nested_key is None else nested_key
            group = self.add_argument_group(doc_group, name=name)

        ## Add objects arguments ##
        num_added = 0
        for obj, (add_args, add_kwargs) in zip(objects, add_types):
            for num, param in enumerate(inspect.signature(obj).parameters.values()):
                annotation = param.annotation
                default = param.default
                is_positional = default == inspect._empty
                if param._kind in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD} or \
                   (is_positional and not add_args) or \
                   (not is_positional and not add_kwargs) or \
                   (is_positional and skip_first and num == 0):
                    continue
                if annotation == inspect._empty and not is_positional:
                    annotation = type(default)
                kwargs = {'help': doc_params.get(param.name)}
                if is_positional:
                    kwargs['required'] = True
                else:
                    kwargs['default'] = default
                if annotation in {str, int, float, bool}:
                    kwargs['type'] = annotation
                elif inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
                    kwargs['action'] = ActionEnum(enum=annotation)
                else:
                    try:
                        kwargs['action'] = ActionJsonSchema(annotation=annotation)
                    except:
                        pass
                if 'type' in kwargs or 'action' in kwargs:
                    arg = '--' + (nested_key+'.' if nested_key else '') + param.name
                    group.add_argument(arg, **kwargs)
                    num_added += 1
                elif is_positional:
                    raise ValueError('Positional argument without a type for '+obj.__name__+' argument '+param.name+'.')

        return num_added
