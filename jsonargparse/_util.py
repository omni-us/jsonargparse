"""Collection of general functions and classes."""

import inspect
import os
import re
import stat
import sys
import textwrap
import warnings
from argparse import ArgumentError
from collections import Counter, namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from io import StringIO
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import (
    IO,
    Any,
    Callable,
    Iterator,
    List,
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
from ._deprecated import PathDeprecations
from ._loaders_dumpers import json_compact_dump, load_value
from ._optionals import (
    _get_config_read_mode,
    fsspec_support,
    import_fsspec,
    import_requests,
    url_support,
)
from ._type_checking import ArgumentParser

__all__ = [
    "capture_parser",
    "class_from_function",
    "Path",
    "register_unresolvable_import_paths",
]


NoneType = type(None)


default_config_option_help = "Path to a configuration file."


@dataclass
class UrlData:
    scheme: str
    url_path: str


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
) -> Tuple[Any, Optional["Path"]]:
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


class CachedStdin(StringIO):
    """Used to allow reading sys.stdin multiple times."""


def get_cached_stdin() -> CachedStdin:
    if not isinstance(sys.stdin, CachedStdin):
        sys.stdin = CachedStdin(sys.stdin.read())
    return sys.stdin


def read_cached_stdin() -> str:
    stdin = get_cached_stdin()
    value = stdin.read()
    stdin.seek(0)
    return value


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


current_path_dir: ContextVar[Optional[str]] = ContextVar("current_path_dir", default=None)


@contextmanager
def change_to_path_dir(path: Optional["Path"]) -> Iterator[Optional[str]]:
    """A context manager for running code in the directory of a path."""
    path_dir = current_path_dir.get()
    chdir: Union[bool, str] = False
    if path is not None:
        if path._url_data and (path.is_url or path.is_fsspec):
            scheme = path._url_data.scheme
            path_dir = path._url_data.url_path
        else:
            scheme = ""
            path_dir = path.absolute
            chdir = True
        if "d" not in path.mode:
            path_dir = os.path.dirname(path_dir)
        path_dir = scheme + path_dir

    token = current_path_dir.set(path_dir)
    if chdir and path_dir:
        chdir = os.getcwd()
        path_dir = os.path.abspath(path_dir)
        os.chdir(path_dir)

    try:
        yield path_dir
    finally:
        current_path_dir.reset(token)
        if chdir:
            os.chdir(chdir)


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
        raise ValueError(f"Unexpected keyword parameters: {set(data.keys())}")
    return extracted[0] if len(extracted) == 1 else extracted


def known_to_fsspec(path: str) -> bool:
    import_fsspec("known_to_fsspec")
    from fsspec.registry import known_implementations

    for protocol in known_implementations:
        if path.startswith(protocol + "://") or path.startswith(protocol + "::"):
            return True
    return False


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
            if isinstance(func_return, __import__("typing").ForwardRef):
                func_return = func_return._evaluate(func.__globals__, {})
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


def parse_url(url: str) -> Optional[UrlData]:
    index = url.rfind("://")
    if index <= 0:
        return None
    return UrlData(
        scheme=url[: index + 3],
        url_path=url[index + 3 :],
    )


def is_absolute_path(path: str) -> bool:
    if path.find("://") > 0:
        return True
    return os.path.isabs(path)


def resolve_relative_path(path: str) -> str:
    parts = path.split("/")
    resolved: List[str] = []
    for part in parts:
        if part == "..":
            resolved.pop()
        elif part != ".":
            resolved.append(part)
    return "/".join(resolved)


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


class PathError(TypeError):
    """Exception raised for errors in the Path class."""


class Path(PathDeprecations):
    """Stores a (possibly relative) path and the corresponding absolute path.

    The absolute path can be obtained without having to remember the working
    directory (or parent remote path) from when the object was created.

    When a Path instance is created, it is checked that: the path exists,
    whether it is a file or directory and whether it has the required access
    permissions (f=file, d=directory, r=readable, w=writeable, x=executable,
    c=creatable, u=url, s=fsspec or in uppercase meaning not, i.e., F=not-file,
    D=not-directory, R=not-readable, W=not-writeable and X=not-executable).

    The creatable flag "c" can be given one or two times. If give once, the
    parent directory must exist and be writeable. If given twice, the parent
    directory does not have to exist, but should be allowed to create.

    An instance of Path class can also refer to the standard input or output.
    To do that, path must be set with the value "-"; it is a common practice.
    Then, getting the content or opening it will automatically be done on
    standard input or output.
    """

    _url_data: Optional[UrlData]
    _file_scheme = re.compile("^file:///?")

    def __init__(
        self,
        path: Union[str, os.PathLike, "Path"],
        mode: str = "fr",
        cwd: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        """Initializer for Path instance.

        Args:
            path: The path to check and store.
            mode: The required type and access permissions among [fdrwxcuFDRWX].
            cwd: Working directory for relative paths. If None, os.getcwd() is used.

        Raises:
            ValueError: If the provided mode is invalid.
            PathError: If the path does not exist or does not agree with the mode.
        """
        self._deprecated_kwargs(kwargs)
        self._check_mode(mode)
        self._std_io = False

        is_url = False
        is_fsspec = False
        if isinstance(path, Path):
            self._std_io = path._std_io
            is_url = path.is_url
            is_fsspec = path.is_fsspec
            url_data = path._url_data
            cwd = path.cwd
            abs_path = path.absolute
            path = path.relative
        elif isinstance(path, (str, os.PathLike)):
            if path == "-":
                self._std_io = True
            path = os.fspath(path)
            cwd = os.fspath(cwd) if cwd else None
            abs_path = os.path.expanduser(path)
            if self._file_scheme.match(abs_path):
                abs_path = self._file_scheme.sub("" if os.name == "nt" else "/", abs_path)
            is_absolute = is_absolute_path(abs_path)
            url_data = parse_url(abs_path)
            cwd_url_data = parse_url(cwd or current_path_dir.get() or os.getcwd())
            if ("u" in mode or "s" in mode) and (url_data or (cwd_url_data and not is_absolute)):
                if cwd_url_data and not is_absolute:
                    abs_path = resolve_relative_path(cwd_url_data.url_path + "/" + path)
                    abs_path = cwd_url_data.scheme + abs_path
                    url_data = parse_url(abs_path)
                if cwd is None:
                    cwd = current_path_dir.get() or os.getcwd()
                if "u" in mode and url_support:
                    is_url = True
                elif "s" in mode and fsspec_support and known_to_fsspec(abs_path):
                    is_fsspec = True
            else:
                if cwd is None:
                    cwd = os.getcwd()
                abs_path = abs_path if is_absolute else os.path.join(cwd, abs_path)
                url_data = None
        else:
            raise PathError("Expected path to be a string, os.PathLike or a Path object.")

        if not self._skip_check and is_url:
            if "r" in mode:
                requests = import_requests("Path with URL support")
                try:
                    requests.head(abs_path).raise_for_status()
                except requests.HTTPError as ex:
                    raise PathError(f"{abs_path} HEAD not accessible :: {ex}") from ex
        elif not self._skip_check and is_fsspec:
            fsspec_mode = "".join(c for c in mode if c in {"r", "w"})
            if fsspec_mode:
                fsspec = import_fsspec("Path")
                try:
                    handle = fsspec.open(abs_path, fsspec_mode)
                    handle.open()
                    handle.close()
                except (FileNotFoundError, KeyError) as ex:
                    raise PathError(f"Path does not exist: {abs_path!r}") from ex
                except PermissionError as ex:
                    raise PathError(f"Path exists but no permission to access: {abs_path!r}") from ex
        elif not self._skip_check and not self._std_io:
            ptype = "Directory" if "d" in mode else "File"
            if "c" in mode:
                pdir = os.path.realpath(os.path.join(abs_path, ".."))
                if not os.path.isdir(pdir) and mode.count("c") == 2:
                    ppdir = None
                    while not os.path.isdir(pdir) and pdir != ppdir:
                        ppdir = pdir
                        pdir = os.path.realpath(os.path.join(pdir, ".."))
                if not os.path.isdir(pdir):
                    raise PathError(f"{ptype} is not creatable since parent directory does not exist: {abs_path!r}")
                if not os.access(pdir, os.W_OK):
                    raise PathError(f"{ptype} is not creatable since parent directory not writeable: {abs_path!r}")
                if "d" in mode and os.access(abs_path, os.F_OK) and not os.path.isdir(abs_path):
                    raise PathError(f"{ptype} is not creatable since path already exists: {abs_path!r}")
                if "f" in mode and os.access(abs_path, os.F_OK) and not os.path.isfile(abs_path):
                    raise PathError(f"{ptype} is not creatable since path already exists: {abs_path!r}")
            elif "d" in mode or "f" in mode:
                if not os.access(abs_path, os.F_OK):
                    raise PathError(f"{ptype} does not exist: {abs_path!r}")
                if "d" in mode and not os.path.isdir(abs_path):
                    raise PathError(f"Path is not a directory: {abs_path!r}")
                if "f" in mode and not (os.path.isfile(abs_path) or stat.S_ISFIFO(os.stat(abs_path).st_mode)):
                    raise PathError(f"Path is not a file: {abs_path!r}")

            if "r" in mode and not os.access(abs_path, os.R_OK):
                raise PathError(f"{ptype} is not readable: {abs_path!r}")
            if "w" in mode and not os.access(abs_path, os.W_OK):
                raise PathError(f"{ptype} is not writeable: {abs_path!r}")
            if "x" in mode and not os.access(abs_path, os.X_OK):
                raise PathError(f"{ptype} is not executable: {abs_path!r}")
            if "D" in mode and os.path.isdir(abs_path):
                raise PathError(f"Path is a directory: {abs_path!r}")
            if "F" in mode and (os.path.isfile(abs_path) or stat.S_ISFIFO(os.stat(abs_path).st_mode)):
                raise PathError(f"Path is a file: {abs_path!r}")
            if "R" in mode and os.access(abs_path, os.R_OK):
                raise PathError(f"{ptype} is readable: {abs_path!r}")
            if "W" in mode and os.access(abs_path, os.W_OK):
                raise PathError(f"{ptype} is writeable: {abs_path!r}")
            if "X" in mode and os.access(abs_path, os.X_OK):
                raise PathError(f"{ptype} is executable: {abs_path!r}")

        self._relative = path
        self._absolute = abs_path
        self._cwd = cwd
        self._mode = mode
        self._is_url = is_url
        self._is_fsspec = is_fsspec
        self._url_data = url_data

    @property
    def relative(self) -> str:
        """Returns the relative representation of the path (how the path was given on instance creation)."""
        return self._relative

    @property
    def absolute(self) -> str:
        """Returns the absolute representation of the path."""
        return self._absolute

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_url(self) -> bool:
        return self._is_url

    @property
    def is_fsspec(self) -> bool:
        return self._is_fsspec

    def __str__(self):
        return self._relative

    def __repr__(self):
        name = "Path_" + self._mode
        name = self._repr_skip_check(name)
        cwd = ""
        if self._relative != self._absolute:
            cwd = ", cwd=" + self._cwd
        return f"{name}({self._relative}{cwd})"

    def __fspath__(self) -> str:
        return self._absolute

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Path):
            return self._absolute == other._absolute
        elif isinstance(other, str):
            return str(self) == other
        return False

    def __call__(self, absolute: bool = True) -> str:
        """Returns the path as a string.

        Args:
            absolute: If false returns the original path given, otherwise the corresponding absolute path.
        """
        return self._absolute if absolute else self._relative

    def get_content(self, mode: str = "r") -> str:
        """Returns the contents of the file or the remote path."""
        if self._std_io:
            return read_cached_stdin()
        elif self._is_url:
            assert mode == "r"
            requests = import_requests("Path.get_content")
            response = requests.get(self._absolute)
            response.raise_for_status()
            return response.text
        elif self._is_fsspec:
            fsspec = import_fsspec("Path.get_content")
            with fsspec.open(self._absolute, mode) as handle:
                with handle as input_file:
                    return input_file.read()
        else:
            with open(self._absolute, mode) as input_file:
                return input_file.read()

    @contextmanager
    def open(self, mode: str = "r") -> Iterator[IO]:
        """Return an opened file object for the path."""
        if self._std_io:
            if "r" in mode:
                yield get_cached_stdin()
            elif "w" in mode:
                yield sys.stdout
        elif self._is_url:
            yield StringIO(self.get_content())
        elif self._is_fsspec:
            fsspec = import_fsspec("Path.open")
            with fsspec.open(self._absolute, mode) as handle:
                yield handle
        else:
            with open(self._absolute, mode) as handle:
                yield handle

    @contextmanager
    def relative_path_context(self) -> Iterator[str]:
        """Context manager to use this path's parent (directory or URL) for relative paths defined within."""
        with change_to_path_dir(self) as path_dir:
            assert isinstance(path_dir, str)
            yield path_dir

    @staticmethod
    def _check_mode(mode: str):
        if not isinstance(mode, str):
            raise ValueError("Expected mode to be a string.")
        if len(set(mode) - set("fdrwxcusFDRWX")) > 0:
            raise ValueError("Expected mode to only include [fdrwxcusFDRWX] flags.")
        for flag, count in Counter(mode).items():
            if count > (2 if flag == "c" else 1):
                raise ValueError(f'Too many occurrences ({count}) for flag "{flag}".')
        if "f" in mode and "d" in mode:
            raise ValueError('Both modes "f" and "d" not possible.')
        if "u" in mode and "d" in mode:
            raise ValueError('Both modes "d" and "u" not possible.')
        if "s" in mode and "d" in mode:
            raise ValueError('Both modes "d" and "s" not possible.')
