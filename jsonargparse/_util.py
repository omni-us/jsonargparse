"""Collection of general functions and classes."""

import inspect
import os
import textwrap
import warnings
from argparse import ArgumentError
from collections import namedtuple
from functools import wraps
from importlib import import_module
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

from ._common import (
    ClassType,
    get_generic_origin,
    is_subclass,
    parser_capture,
    parser_context,
)
from ._loaders_dumpers import json_compact_dump, load_value
from ._optionals import _get_config_read_mode
from ._paths import Path
from ._type_checking import ArgumentParser

__all__ = [
    "capture_parser",
    "class_from_function",
    "register_unresolvable_import_paths",
]


NoneType = type(None)


default_config_option_help = "Path to a configuration file."


def argument_error(message: str) -> ArgumentError:
    return ArgumentError(None, message)


class JsonargparseWarning(UserWarning):
    pass


def warning(message, category=JsonargparseWarning, stacklevel=1):
    message = textwrap.fill(textwrap.dedent(message), 110).strip()
    warnings.warn(
        "\n" + textwrap.indent(message, "    ") + "\n",
        category=category,
        stacklevel=stacklevel + 1,
    )


class CaptureParserException(Exception):
    def __init__(self, parser: Optional[ArgumentParser]):
        self.parser = parser
        super().__init__("" if parser else "No parse_args call to capture the parser.")


def capture_parser(function: Callable, *args, **kwargs) -> ArgumentParser:
    """Returns the parser object used within the execution of a function.

    The function execution is stopped on the start of the call to parse_args. No
    parsing is done or execution of instructions after the parse_args.

    Args:
        function: A callable that internally creates a parser and calls parse_args.
        *args: Positional arguments used to run the function.
        **kwargs: Keyword arguments used to run the function.

    Raises:
        CaptureParserException: If the function does not call parse_args.
    """
    try:
        with parser_context(parser_capture=True):
            function(*args, **kwargs)
    except CaptureParserException as ex:
        return ex.parser  # type: ignore[return-value]
    raise CaptureParserException(None)


def return_parser_if_captured(parser: ArgumentParser):
    if parser_capture.get():
        raise CaptureParserException(parser)


def identity(value):
    return value


NestedArg = namedtuple("NestedArg", "key val")


def parse_value_or_config(
    value: Any, enable_path: bool = True, simple_types: bool = False
) -> Tuple[Any, Optional[Path]]:
    """Parses yaml/json config in a string or a path"""
    nested_arg: Union[bool, NestedArg] = False
    if isinstance(value, NestedArg):
        nested_arg = value
        value = nested_arg.val
    cfg_path = None
    if enable_path and type(value) is str and value != "-":
        try:
            cfg_path = Path(value, mode=_get_config_read_mode())
        except TypeError:
            pass
        else:
            with cfg_path.relative_path_context():
                value = load_value(cfg_path.get_content(), simple_types=simple_types)
    if type(value) is str and value.strip() != "":
        parsed_val = load_value(value, simple_types=simple_types)
        if type(parsed_val) is not str:
            value = parsed_val
    if isinstance(value, dict) and cfg_path is not None:
        value["__path__"] = cfg_path
    if nested_arg:
        value = NestedArg(key=nested_arg.key, val=value)  # type: ignore[union-attr]
    return value, cfg_path


def import_object(name: str):
    """Returns an object in a module given its dot import path."""
    if not isinstance(name, str) or "." not in name:
        raise ValueError(f"Expected a dot import path string: {name}")
    if not all(x.isidentifier() for x in name.split(".")):
        raise ValueError(f"Unexpected import path format: {name}")
    name_module, name_object = name.rsplit(".", 1)
    try:
        parent = __import__(name_module, fromlist=[name_object])
    except ModuleNotFoundError as ex:
        if "." not in name_module:
            raise ex
        name_module, name_object1 = name_module.rsplit(".", 1)
        parent = getattr(__import__(name_module, fromlist=[name_object1]), name_object1)
    return getattr(parent, name_object)


unresolvable_import_paths = {}


def register_unresolvable_import_paths(*modules: ModuleType):
    """Saves import paths of module objects for which its import path is unresolvable from the object alone.

    Objects with unresolvable import paths have the __module__ attribute set to None.
    """
    for module in modules:
        for val in vars(module).values():
            if (
                getattr(val, "__module__", None) is None
                and getattr(val, "__name__", None)
                and type(val) in {BuiltinFunctionType, FunctionType, Type}
            ):
                unresolvable_import_paths[val] = f"{module.__name__}.{val.__name__}"


def get_module_var_path(module_path: str, value: Any) -> Optional[str]:
    module = import_module(module_path)
    for name, var in vars(module).items():
        if var is value:
            return module_path + "." + name
    return None


def get_import_path(value: Any) -> Optional[str]:
    """Returns the shortest dot import path for the given object."""
    path = None
    value = get_generic_origin(value)
    if hasattr(value, "__self__") and inspect.isclass(value.__self__) and inspect.ismethod(value):
        module_path = getattr(value.__self__, "__module__", None)
        qualname = f"{value.__self__.__name__}.{value.__name__}"
    else:
        module_path = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", "")

    if module_path is None:
        path = unresolvable_import_paths.get(value)
        if path:
            module_path, _ = path.rsplit(".", 1)
    elif (not qualname and not inspect.isclass(value)) or (
        inspect.ismethod(value) and not inspect.isclass(value.__self__)
    ):
        path = get_module_var_path(module_path, value)
    elif qualname:
        path = module_path + "." + qualname

    if not path:
        raise ValueError(f"Not possible to determine the import path for object {value}.")

    if qualname and module_path and ("." in qualname or "." in module_path):
        module_parts = module_path.split(".")
        for num in range(len(module_parts)):
            module_path = ".".join(module_parts[: num + 1])
            module = import_module(module_path)
            if "." in qualname:
                obj_name, attr = qualname.rsplit(".", 1)
                obj = getattr(module, obj_name, None)
                if getattr(module, attr, None) is value:
                    path = module_path + "." + attr
                    break
                elif getattr(obj, attr, None) == value:
                    path = module_path + "." + qualname
                    break
            elif getattr(module, qualname, None) is value:
                path = module_path + "." + qualname
                break
    return path


def object_path_serializer(value):
    try:
        path = get_import_path(value)
        reimported = import_object(path)
        if value is not reimported:
            raise ValueError
        return path
    except Exception as ex:
        raise ValueError(f"Only possible to serialize an importable object, given {value}: {ex}") from ex


def get_typehint_origin(typehint):
    if not hasattr(typehint, "__origin__"):
        typehint_class = get_import_path(typehint.__class__)
        if typehint_class == "types.UnionType":
            return Union
        if typehint_class in {"typing._TypedDictMeta", "typing_extensions._TypedDictMeta"}:
            return dict
    return getattr(typehint, "__origin__", None)


def hash_item(item):
    try:
        if isinstance(item, (dict, list)):
            item_hash = hash(json_compact_dump(item))
        else:
            item_hash = hash(item)
    except Exception:
        item_hash = hash(repr(item))
    return item_hash


def unique(iterable):
    unique_items = []
    seen = set()
    for item in iterable:
        key = hash_item(item)
        if key not in seen:
            unique_items.append(item)
            seen.add(key)
    return unique_items


def iter_to_set_str(val, sep=","):
    val = unique(val)
    if len(val) == 1:
        return str(val[0])
    return "{" + sep.join(str(x) for x in val) + "}"


def indent_text(text: str, first_line: bool = True) -> str:
    if first_line:
        return textwrap.indent(text, "  ")
    lines = text.splitlines()
    if len(lines) == 1:
        return text
    return lines[0] + os.linesep + textwrap.indent(os.linesep.join(lines[1:]), "  ")


def get_private_kwargs(data, **kwargs):
    extracted = [data.pop(name, default) for name, default in kwargs.items()]
    if data:
        raise ValueError(f"Unexpected keyword parameters: {set(data)}")
    return extracted[0] if len(extracted) == 1 else extracted


class ClassFromFunctionBase:
    wrapped_function: Callable


def class_from_function(
    func: Callable[..., ClassType],
    func_return: Optional[Type[ClassType]] = None,
    name: Optional[str] = None,
) -> Type[ClassType]:
    """Creates a dynamic class which if instantiated is equivalent to calling func.

    Args:
        func: A function that returns an instance of a class.
        func_return: The return type of the function. Required if func does not have a return type annotation.
        name: The name of the class. Defaults to function name suffixed with "_class".
    """
    if func_return is None:
        func_return = inspect.signature(func).return_annotation
    if func_return is inspect.Signature.empty:
        raise ValueError(f"{func} does not have a return type annotation")
    if isinstance(func_return, str):
        try:
            func_return = get_type_hints(func)["return"]
        except Exception as ex:
            func_return = inspect.signature(func).return_annotation
            raise ValueError(f"Unable to dereference {func_return}, the return type of {func}: {ex}") from ex

    if not name:
        name = func.__qualname__.replace(".", "__") + "_class"

    caller_module = inspect.getmodule(inspect.stack()[1][0]) or inspect.getmodule(class_from_function)
    assert caller_module
    if hasattr(caller_module, name):
        cls = getattr(caller_module, name)
        mro = inspect.getmro(cls) if inspect.isclass(cls) else ()
        if (
            len(mro) > 1
            and mro[1] is func_return
            and is_subclass(cls, ClassFromFunctionBase)
            and cls.wrapped_function is func
            and cls.__name__ == name
        ):
            return cls
        raise ValueError(f"{caller_module.__name__} already defines {name!r}, please use a different name")

    @wraps(func)
    def __new__(cls, *args, **kwargs):
        return func(*args, **kwargs)

    class ClassFromFunction(func_return, ClassFromFunctionBase):  # type: ignore[valid-type,misc]
        pass

    setattr(caller_module, name, ClassFromFunction)
    ClassFromFunction.wrapped_function = func
    ClassFromFunction.__new__ = __new__  # type: ignore[method-assign]
    ClassFromFunction.__doc__ = func.__doc__
    ClassFromFunction.__module__ = caller_module.__name__
    ClassFromFunction.__name__ = name
    ClassFromFunction.__qualname__ = name
    return ClassFromFunction


def get_argument_group_class(parser):
    import ast

    from ._core import ActionsContainer, ArgumentGroup

    if parser.__class__.add_argument != ActionsContainer.add_argument:
        try:
            add_argument = parser.__class__.add_argument
            source = inspect.getsource(add_argument)
            source = "class _ArgumentGroupAutoSubclass(ArgumentGroup):\n" + source
            class_ast = ast.parse(source)
            code = compile(class_ast, filename="<ast>", mode="exec")
            namespace = {**add_argument.__globals__, "ArgumentGroup": ArgumentGroup}
            exec(code, namespace)
            group_class = namespace["_ArgumentGroupAutoSubclass"]
            group_class.__module__ = parser.__class__.__module__
            add_argument.__globals__[group_class.__name__] = group_class
            return group_class
        except Exception as ex:
            parser.logger.debug(
                f"Failed to create ArgumentGroup subclass based on {parser.__class__.__name__}: {ex}", exc_info=ex
            )
    return ArgumentGroup
