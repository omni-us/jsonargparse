import inspect
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

from ._core import ArgumentParser

__all__ = ["from_config_support"]

T = TypeVar("T")


def _parse_class_kwargs_from_config(cls: Type[T], config: Union[str, PathLike, dict]) -> dict:
    """Parse the init kwargs for `cls` from a config file or dict."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(cls)
    if isinstance(config, dict):
        cfg = parser.parse_object(config, defaults=False)
    else:
        cfg = parser.parse_path(config, defaults=False)
    return parser.instantiate_classes(cfg).as_dict()


def _override_init_defaults_from_config(cls: Type[T]) -> None:
    """Override __init__ defaults for `cls` based on a config file."""
    config = getattr(cls, "__from_config_defaults__", None)
    if not isinstance(config, (str, PathLike, type(None))):
        raise TypeError("__from_config_defaults__ must be str, PathLike, or None")
    if not (isinstance(config, (str, PathLike)) and Path(config).is_file()):
        return

    defaults = _parse_class_kwargs_from_config(cls, config)

    # Override defaults for parameters in cls.__init__
    params = inspect.signature(cls.__init__).parameters
    for name, default in defaults.copy().items():
        param = params.get(name)
        if param and param.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            defaults.pop(name)
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                cls.__init__.__kwdefaults__[name] = default  # type: ignore[index]
            else:
                index = list(params).index(name) - 1
                aux = cls.__init__.__defaults__ or ()
                cls.__init__.__defaults__ = aux[:index] + (default,) + aux[index + 1 :]

    # Gather defaults for parameters in parent classes' __init__
    override_parent_params = []
    for base in inspect.getmro(cls)[1:]:
        if not defaults:
            break

        params = inspect.signature(base.__init__).parameters  # type: ignore[misc]
        for name, default in defaults.copy().items():
            if name in params:
                defaults.pop(name)
                new_param = inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=params[name].annotation,
                )
                override_parent_params.append(new_param)

    # Override defaults for parameters in parent classes' __init__ via a wrapper
    if override_parent_params:
        original_init = cls.__init__
        original_sig = inspect.signature(cls.__init__)
        parameters = list(original_sig.parameters.values())

        # Find and pop the **kwargs parameter, if it exists
        kwargs_param = None
        if parameters and parameters[-1].kind == inspect.Parameter.VAR_KEYWORD:
            kwargs_param = parameters.pop()

        # Add new parameters
        for param in reversed(override_parent_params):
            parameters.append(param)

        # Add **kwargs back at the end
        if kwargs_param:
            parameters.append(kwargs_param)

        # Create and set __init__ wrapper with new signature
        parent_defaults = {p.name: p.default for p in override_parent_params}

        @wraps(original_init)
        def wrapper(*args, **kwargs):
            for name, default in parent_defaults.items():
                if name not in kwargs:
                    kwargs[name] = default
            return original_init(*args, **kwargs)

        wrapper.__signature__ = original_sig.replace(parameters=parameters)  # type: ignore[attr-defined]
        cls.__init__ = wrapper  # type: ignore[method-assign]


def from_config_support(
    *args,
    from_config_method: bool = True,
    from_config_method_name: str = "from_config",
    from_config_method_default: Optional[Union[str, PathLike, dict]] = None,
):
    """Class decorator that adds config support to a base class.

    This decorator does two things:

        1. Adds support for overriding __init__ defaults by defining a
           `__from_config_defaults__` class attribute pointing to a config file
           path. The overriding of defaults happens on decorator application or
           on class creation for subclasses. Inspecting the signature will
           give the overridden defaults.

        2. Adds a @classmethod, by default named `from_config`, that
           instantiates the class based on a config file or dict.

    The decorator can be used without parentheses, e.g.

        @from_config_support
        class MyClass:
            ...

    Use parentheses to customize the behavior, e.g.

        @from_config_support(from_config_method=False)
        class MyClass:
            ...

    Args:
        from_config_method: Whether to add the `from_config` classmethod.
        from_config_method_name: Name of the `from_config` classmethod.
        from_config_method_default: Default value for the `config` parameter.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if not inspect.isclass(cls):
            raise TypeError("from_config_support can only be applied to classes")

        # 1. Add the from_config classmethod to the base class
        if from_config_method:

            def from_config(cls: Type[T], config: Union[str, PathLike, dict]) -> T:
                """Instantiate current class based on a config file or dict.

                Args:
                    config: Path to a config file or a dict with config values.
                """
                kwargs = _parse_class_kwargs_from_config(cls, config)
                return cls(**kwargs)

            if from_config_method_default is not None:
                from_config.__defaults__ = (from_config_method_default,)
                from_config.__name__ = from_config_method_name

            from_config.__module__ = cls.__module__
            from_config.__qualname__ = f"{cls.__name__}.{from_config_method_name}"
            setattr(cls, from_config_method_name, classmethod(from_config))

        # 2. Override defaults for the decorated class itself
        _override_init_defaults_from_config(cls)

        # 3. Get the original __init_subclass__ defined on `cls`, if any.
        #    Check __dict__ so that parent's method isn't grabbed.
        original_init_subclass = cls.__dict__.get("__init_subclass__")

        # 4. Create the new __init_subclass__
        def new_init_subclass(cls_sub, **kwargs):
            """This method will be called on every subclass of `cls`."""
            # A. Override defaults for the subclass
            _override_init_defaults_from_config(cls_sub)

            # B. Call the original __init_subclass__ if this class defined one
            if original_init_subclass:
                # Call the original function (it's a classmethod object)
                original_init_subclass.__func__(cls_sub, **kwargs)
            else:
                # This class (cls) didn't have one, so just call up the MRO to the *next* class.
                # super(cls, cls_sub) finds the next __init_subclass__ in the MRO *after* `cls`.
                super(cls, cls_sub).__init_subclass__(**kwargs)

        # 5. Attach the new method to the class
        cls.__init_subclass__ = classmethod(new_init_subclass)  # type: ignore[assignment]

        return cls

    # Handle decorator usage without parentheses
    if len(args) > 0:
        if len(args) == 1:
            return decorator(args[0])
        raise TypeError("from_config_support can only receive a single positional argument")

    return decorator
