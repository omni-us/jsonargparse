"""Collection of general functions and classes."""

import inspect
import logging
import os
import re
import stat
import sys
import warnings
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

from .loaders_dumpers import load_value
from .optionals import (
    fsspec_support,
    get_config_read_mode,
    import_fsspec,
    import_reconplogger,
    import_requests,
    import_url_validator,
    reconplogger_support,
    url_support,
)
from .type_checking import ArgumentParser


__all__ = [
    'capture_parser',
    'class_from_function',
    'LoggerProperty',
    'null_logger',
    'ParserError',
    'Path',
    'register_unresolvable_import_paths',
    'usage_and_exit_error_handler',
]


logging_levels = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'}
null_logger = logging.getLogger('jsonargparse_null_logger')
null_logger.addHandler(logging.NullHandler())
null_logger.parent = None


NoneType = type(None)


default_config_option_help = 'Path to a configuration file.'


class ParserError(Exception):
    """Error raised when parsing a value fails."""


class DebugException(Exception):
    pass


class JsonargparseWarning(UserWarning):
    pass


def warning(message, category=JsonargparseWarning, stacklevel=1):
    warnings.warn(
        re.sub('\n\n+', '\n\n', re.sub('\n +', '\n  ', message)),
        category=category,
        stacklevel=stacklevel+1,
    )


class CaptureParserException(Exception):
    def __init__(self, parser: Optional[ArgumentParser]):
        self.parser = parser
        super().__init__('' if parser else 'No parse_args call to capture the parser.')


parser_captured: ContextVar = ContextVar('parser_captured', default=False)


@contextmanager
def capture_parser_context():
    t = parser_captured.set(True)
    try:
        yield
    finally:
        parser_captured.reset(t)


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
        with capture_parser_context():
            function(*args, **kwargs)
    except CaptureParserException as ex:
        return ex.parser  # type: ignore
    raise CaptureParserException(None)


def return_parser_if_captured(parser: ArgumentParser):
    if parser_captured.get():
        raise CaptureParserException(parser)


def identity(value):
    return value


NestedArg = namedtuple('NestedArg', 'key val')


def parse_value_or_config(value: Any, enable_path: bool = True) -> Tuple[Any, Optional['Path']]:
    """Parses yaml/json config in a string or a path"""
    nested_arg: Union[bool, NestedArg] = False
    if isinstance(value, NestedArg):
        nested_arg = value
        value = nested_arg.val
    cfg_path = None
    if enable_path and type(value) is str:
        try:
            cfg_path = Path(value, mode=get_config_read_mode())
        except TypeError:
            pass
        else:
            with change_to_path_dir(cfg_path):
                value = load_value(cfg_path.get_content())
    if type(value) is str and value.strip() != '':
        parsed_val = load_value(value)
        if type(parsed_val) is not str:
            value = parsed_val
    if isinstance(value, dict) and cfg_path is not None:
        value['__path__'] = cfg_path
    if nested_arg:
        value = NestedArg(key=nested_arg.key, val=value)  # type: ignore
    return value, cfg_path


def usage_and_exit_error_handler(parser: 'ArgumentParser', message: str) -> None:
    """Error handler that prints the usage and exits with error code 2 (same behavior as argparse).

    If the JSONARGPARSE_DEBUG environment variable is set, instead of exit, a
    DebugException is raised.

    Args:
        parser: The parser object.
        message: The message describing the error being handled.
    """
    parser.print_usage(sys.stderr)
    args = {'prog': parser.prog, 'message': message}
    sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
    if 'JSONARGPARSE_DEBUG' in os.environ:
        raise DebugException('jsonargparse debug enabled, thus raising exception instead of exit.')
    else:
        parser.exit(2)


def is_subclass(cls, class_or_tuple):
    """Extension of issubclass that supports non-class argument."""
    return inspect.isclass(cls) and issubclass(cls, class_or_tuple)


def import_object(name: str):
    """Returns an object in a module given its dot import path."""
    if not isinstance(name, str) or '.' not in name:
        raise ValueError(f'Expected a dot import path string: {name}')
    if not all(x.isidentifier() for x in name.split('.')):
        raise ValueError(f'Unexpected import path format: {name}')
    name_module, name_object = name.rsplit('.', 1)
    try:
        parent = __import__(name_module, fromlist=[name_object])
    except ModuleNotFoundError as ex:
        if '.' not in name_module:
            raise ex
        name_module, name_object1 = name_module.rsplit('.', 1)
        parent = getattr(__import__(name_module, fromlist=[name_object1]), name_object1)
    return getattr(parent, name_object)


def import_module_leaf(name: str):
    """Similar to __import__(name) but returns the leaf module instead of the root."""
    if '.' in name:
        name_parent, name_leaf = name.rsplit('.', 1)
        parent = __import__(name_parent, fromlist=[name_leaf])
        module = getattr(parent, name_leaf)
    else:
        module = __import__(name)
    return module


unresolvable_import_paths = {}


def register_unresolvable_import_paths(*modules: ModuleType):
    """Saves import paths of module objects for which its import path is unresolvable from the object alone.

    Objects with unresolvable import paths have the __module__ attribute set to None.
    """
    for module in modules:
        for val in vars(module).values():
            if getattr(val, '__module__', None) is None and \
               getattr(val, '__name__', None) and \
               type(val) in {BuiltinFunctionType, FunctionType, Type}:
                unresolvable_import_paths[val] = f'{module.__name__}.{val.__name__}'


def get_import_path(value):
    """Returns the shortest dot import path for the given object."""
    module_path = getattr(value, '__module__', None)
    if module_path is None:
        path = unresolvable_import_paths.get(value)
        if not path:
            raise ValueError(f'Not possible to determine the import path for object {value}.')
        module_path = path.rsplit('.', 1)[0]
    else:
        path = module_path + '.' + value.__qualname__
    if '.' in module_path:
        module_parts = module_path.split('.')
        for num in range(len(module_parts)):
            module_path = '.'.join(module_parts[:num+1])
            module = import_module_leaf(module_path)
            if '.' in value.__qualname__:
                obj_name, attr = value.__qualname__.rsplit('.', 1)
                obj = getattr(module, obj_name, None)
                if getattr(obj, attr, None) is value:
                    path = module_path + '.' + value.__qualname__
                    break
            elif getattr(module, value.__qualname__, None) is value:
                path = module_path + '.' + value.__qualname__
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
        raise ValueError(f'Only possible to serialize an importable object, given {value}: {ex}') from ex


lenient_check: ContextVar = ContextVar('lenient_check', default=False)


@contextmanager
def lenient_check_context(caller=None, lenient=True):
    t = lenient_check.set(False if caller == 'argcomplete' else lenient)
    try:
        yield
    finally:
        lenient_check.reset(t)


@contextmanager
def change_to_path_dir(path: Optional['Path']):
    """A context manager for running code in the directory of a path."""
    chdir = path is not None and not (path.is_url or path.is_fsspec)
    if chdir:
        cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(str(path))))
    try:
        yield None
    finally:
        if chdir:
            os.chdir(cwd)


def iter_to_set_str(val, sep=','):
    val = list(val)
    if len(val) == 1:
        return str(val[0])
    return '{'+sep.join(str(x) for x in val)+'}'


def indent_text(text: str) -> str:
    return text.replace('\n', '\n  ')


def get_private_kwargs(kwargs, items):
    extracted = [kwargs.pop(name, default) for name, default in items.items()]
    if kwargs:
        raise ValueError(f'Unexpected keyword parameters: {set(kwargs.keys())}')
    return extracted[0] if len(extracted) == 1 else extracted


def known_to_fsspec(path: str) -> bool:
    import_fsspec('known_to_fsspec')
    from fsspec.registry import known_implementations
    for protocol in known_implementations.keys():
        if path.startswith(protocol+'://') or path.startswith(protocol+'::'):
            return True
    return False


class ClassFromFunctionBase:
    wrapped_function: Callable


ClassType = TypeVar('ClassType')


def class_from_function(func: Callable[..., ClassType]) -> Type[ClassType]:
    """Creates a dynamic class which if instantiated is equivalent to calling func.

    Args:
        func: A function that returns an instance of a class. It must have a return type annotation.
    """
    func_return = inspect.signature(func).return_annotation
    if isinstance(func_return, str):
        caller_frame = inspect.currentframe().f_back  # type: ignore
        func_return = caller_frame.f_locals.get(func_return) or caller_frame.f_globals.get(func_return)  # type: ignore
        if func_return is None:
            raise ValueError(f'Unable to dereference {func_return} the return type of {func}.')

    @wraps(func)
    def __new__(cls, *args, **kwargs):
        return func(*args, **kwargs)

    class ClassFromFunction(func_return, ClassFromFunctionBase):  # type: ignore
        pass

    ClassFromFunction.wrapped_function = func
    ClassFromFunction.__new__ = __new__  # type: ignore
    ClassFromFunction.__doc__ = func.__doc__
    ClassFromFunction.__name__ = func.__name__
    return ClassFromFunction


class Path:
    """Stores a (possibly relative) path and the corresponding absolute path.

    When a Path instance is created it is checked that: the path exists, whether
    it is a file or directory and whether has the required access permissions
    (f=file, d=directory, r=readable, w=writeable, x=executable, c=creatable,
    u=url, s=fsspec or in uppercase meaning not, i.e., F=not-file,
    D=not-directory, R=not-readable, W=not-writeable and X=not-executable). The
    absolute path can be obtained without having to remember the working
    directory from when the object was created.
    """

    file_scheme = re.compile('^file:///?')

    def __init__(
        self,
        path: Union[str, 'Path'],
        mode: str = 'fr',
        cwd: Optional[str] = None,
        skip_check: bool = False,
    ):
        """Initializer for Path instance.

        Args:
            path: The path to check and store.
            mode: The required type and access permissions among [fdrwxcuFDRWX].
            cwd: Working directory for relative paths. If None, then os.getcwd() is used.
            skip_check: Whether to skip path checks.

        Raises:
            ValueError: If the provided mode is invalid.
            TypeError: If the path does not exist or does not agree with the mode.
        """
        self._check_mode(mode)
        if cwd is None:
            cwd = os.getcwd()

        is_url = False
        is_fsspec = False
        if isinstance(path, Path):
            is_url = path.is_url
            is_fsspec = path.is_fsspec
            cwd = path.cwd  # type: ignore
            abs_path = path.abs_path  # type: ignore
            path = path.rel_path  # type: ignore
        elif isinstance(path, str):
            abs_path = os.path.expanduser(path)
            if self.file_scheme.match(abs_path):
                abs_path = self.file_scheme.sub('' if os.name == 'nt' else '/', abs_path)
            if 'u' in mode and url_support and import_url_validator('Path')(abs_path):
                is_url = True
            elif 's' in mode and fsspec_support and known_to_fsspec(abs_path):
                is_fsspec = True
            elif 'f' in mode or 'd' in mode:
                abs_path = abs_path if os.path.isabs(abs_path) else os.path.join(cwd, abs_path)
        else:
            raise TypeError('Expected path to be a string or a Path object.')

        if not skip_check and is_url:
            if 'r' in mode:
                requests = import_requests('Path with URL support')
                try:
                    requests.head(abs_path).raise_for_status()
                except requests.HTTPError as ex:
                    raise TypeError(f'{abs_path} HEAD not accessible :: {ex}') from ex
        elif not skip_check and is_fsspec:
            fsspec_mode = ''.join(c for c in mode if c in {'r','w'})
            if fsspec_mode:
                fsspec = import_fsspec('Path')
                try:
                    handle = fsspec.open(abs_path, fsspec_mode)
                    handle.open()
                    handle.close()
                except (FileNotFoundError, KeyError):
                    raise TypeError('Path does not exist: '+abs_path)
                except PermissionError:
                    raise TypeError('Path exists but no permission to access: '+abs_path)
        elif not skip_check:
            ptype = 'Directory' if 'd' in mode else 'File'
            if 'c' in mode:
                pdir = os.path.realpath(os.path.join(abs_path, '..'))
                if not os.path.isdir(pdir):
                    raise TypeError(ptype+' is not creatable since parent directory does not exist: '+abs_path)
                if not os.access(pdir, os.W_OK):
                    raise TypeError(ptype+' is not creatable since parent directory not writeable: '+abs_path)
                if 'd' in mode and os.access(abs_path, os.F_OK) and not os.path.isdir(abs_path):
                    raise TypeError(ptype+' is not creatable since path already exists: '+abs_path)
                if 'f' in mode and os.access(abs_path, os.F_OK) and not os.path.isfile(abs_path):
                    raise TypeError(ptype+' is not creatable since path already exists: '+abs_path)
            else:
                if not os.access(abs_path, os.F_OK):
                    raise TypeError(ptype+' does not exist: '+abs_path)
                if 'd' in mode and not os.path.isdir(abs_path):
                    raise TypeError('Path is not a directory: '+abs_path)
                if 'f' in mode and not (os.path.isfile(abs_path) or stat.S_ISFIFO(os.stat(abs_path).st_mode)):
                    raise TypeError('Path is not a file: '+abs_path)
            if 'r' in mode and not os.access(abs_path, os.R_OK):
                raise TypeError(ptype+' is not readable: '+abs_path)
            if 'w' in mode and not os.access(abs_path, os.W_OK):
                raise TypeError(ptype+' is not writeable: '+abs_path)
            if 'x' in mode and not os.access(abs_path, os.X_OK):
                raise TypeError(ptype+' is not executable: '+abs_path)
            if 'D' in mode and os.path.isdir(abs_path):
                raise TypeError('Path is a directory: '+abs_path)
            if 'F' in mode and (os.path.isfile(abs_path) or stat.S_ISFIFO(os.stat(abs_path).st_mode)):
                raise TypeError('Path is a file: '+abs_path)
            if 'R' in mode and os.access(abs_path, os.R_OK):
                raise TypeError(ptype+' is readable: '+abs_path)
            if 'W' in mode and os.access(abs_path, os.W_OK):
                raise TypeError(ptype+' is writeable: '+abs_path)
            if 'X' in mode and os.access(abs_path, os.X_OK):
                raise TypeError(ptype+' is executable: '+abs_path)

        self.rel_path = path
        self.abs_path = abs_path
        self.cwd = cwd
        self.mode = mode
        self.is_url: bool = is_url
        self.is_fsspec: bool = is_fsspec
        self.skip_check = skip_check

    def __str__(self):
        return self.rel_path

    def __repr__(self):
        cwd = '' if self.rel_path == self.abs_path else ', cwd='+self.cwd
        name = 'Path_'+self.mode
        if self.skip_check:
            name += '_skip_check'
        return name+'('+self.rel_path+cwd+')'

    def __call__(self, absolute:bool=True) -> str:
        """Returns the path as a string.

        Args:
            absolute: If false returns the original path given, otherwise the corresponding absolute path.
        """
        return self.abs_path if absolute else self.rel_path

    def get_content(self, mode:str='r') -> str:
        """Returns the contents of the file or the response of a GET request to the URL."""
        if self.is_url:
            requests = import_requests('Path with URL support')
            response = requests.get(self.abs_path)
            response.raise_for_status()
            return response.text
        elif self.is_fsspec:
            fsspec = import_fsspec('Path')
            with fsspec.open(self.abs_path, mode) as handle:
                with handle as input_file:
                    return input_file.read()
        else:
            with open(self.abs_path, mode) as input_file:
                return input_file.read()

    @staticmethod
    def _check_mode(mode:str):
        if not isinstance(mode, str):
            raise ValueError('Expected mode to be a string.')
        if len(set(mode)-set('fdrwxcusFDRWX')) > 0:
            raise ValueError('Expected mode to only include [fdrwxcusFDRWX] flags.')
        if 'f' in mode and 'd' in mode:
            raise ValueError('Both modes "f" and "d" not possible.')
        if 'u' in mode and 'd' in mode:
            raise ValueError('Both modes "d" and "u" not possible.')
        if 's' in mode and 'd' in mode:
            raise ValueError('Both modes "d" and "s" not possible.')


def setup_default_logger(data, level, caller):
    name = caller
    if isinstance(data, str):
        name = data
    elif isinstance(data, dict) and 'name' in data:
        name = data['name']
    logger = logging.getLogger(name)
    logger.parent = None
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    level = getattr(logging, level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def parse_logger(logger: Union[bool, str, dict, logging.Logger], caller):
    if not isinstance(logger, (bool, str, dict, logging.Logger)):
        raise ValueError(f'Expected logger to be an instance of (bool, str, dict, logging.Logger), but got {logger}.')
    if isinstance(logger, dict) and len(set(logger.keys())-{'name', 'level'}) > 0:
        value = {k: v for k, v in logger.items() if k not in {'name', 'level'}}
        raise ValueError(f'Unexpected data to configure logger: {value}.')
    if logger is False:
        return null_logger
    level = 'WARNING'
    if isinstance(logger, dict) and 'level' in logger:
        level = logger['level']
    if level not in logging_levels:
        raise ValueError(f'Got logger level {level!r} but must be one of {logging_levels}.')
    if (logger is True or (isinstance(logger, dict) and 'name' not in logger)) and reconplogger_support:
        logger = import_reconplogger('parse_logger').logger_setup(level=level)
    if not isinstance(logger, logging.Logger):
        logger = setup_default_logger(logger, level, caller)
    return logger


class LoggerProperty:
    """Class designed to be inherited by other classes to add a logger property."""

    def __init__(self, *args, logger: Union[bool, str, dict, logging.Logger] = False, **kwargs):
        """Initializer for LoggerProperty class."""
        self.logger = logger
        super().__init__(*args, **kwargs)


    @property
    def logger(self):
        """The logger property for the class.

        :getter: Returns the current logger.
        :setter: Sets the given logging.Logger as logger or sets the default logger
                 if given True/str(logger name)/dict(name, level), or disables logging
                 if given False.

        Raises:
            ValueError: If an invalid logger value is given.
        """
        return self._logger

    @logger.setter
    def logger(self, logger: Union[bool, str, dict, logging.Logger]):
        if logger is None:
            from .deprecated import deprecation_warning, logger_property_none_message
            deprecation_warning((LoggerProperty.logger, None), logger_property_none_message)
            logger = False
        if not logger and 'JSONARGPARSE_DEBUG' in os.environ:
            logger = {'level': 'DEBUG'}
        self._logger = parse_logger(logger, type(self).__name__)


if 'JSONARGPARSE_DEBUG' in os.environ:
    os.environ['LOGGER_LEVEL'] = 'DEBUG'  # pragma: no cover
