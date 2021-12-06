"""Collection of general functions and classes."""

import inspect
import logging
import os
import re
import stat
import sys
import warnings
from argparse import Action
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr
from contextvars import ContextVar
from typing import Any, Optional, Tuple, Union

from .loaders_dumpers import load_value
from .optionals import (
    url_support,
    import_requests,
    import_url_validator,
    fsspec_support,
    import_fsspec,
    get_config_read_mode,
)
from .type_checking import ArgumentParser


__all__ = [
    'ParserError',
    'null_logger',
    'usage_and_exit_error_handler',
    'Path',
    'LoggerProperty',
]


null_logger = logging.Logger('jsonargparse_null_logger')
null_logger.addHandler(logging.NullHandler())


NoneType = type(None)


class ParserError(Exception):
    """Error raised when parsing a value fails."""
    pass


class JsonargparseWarning(UserWarning):
    pass


def warning(message, category=JsonargparseWarning, stacklevel=1):
    warnings.warn(
        re.sub('\n\n+', '\n\n', re.sub('\n +', '\n  ', message)),
        category=category,
        stacklevel=stacklevel+1,
    )


def _parse_value_or_config(value: Any, enable_path: bool = True) -> Tuple[Any, Optional['Path']]:
    """Parses yaml/json config in a string or a path"""
    cfg_path = None
    if isinstance(value, str) and value.strip() != '':
        parsed_val = load_value(value)
        if not isinstance(parsed_val, str):
            value = parsed_val
    if enable_path and isinstance(value, str):
        try:
            cfg_path = Path(value, mode=get_config_read_mode())
        except TypeError:
            pass
        else:
            value = load_value(cfg_path.get_content())
    if isinstance(value, dict) and cfg_path is not None:
        value['__path__'] = cfg_path
    return value, cfg_path


def usage_and_exit_error_handler(parser: 'ArgumentParser', message: str) -> None:
    """Error handler that prints the usage and exits with error code 2 (same behavior as argparse).

    Args:
        parser: The parser object.
        message: The message describing the error being handled.
    """
    parser.print_usage(sys.stderr)
    args = {'prog': parser.prog, 'message': message}
    sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
    parser.exit(2)


def _get_env_var(parser: 'ArgumentParser', action: 'Action') -> str:
    """Returns the environment variable for a given parser and action."""
    if hasattr(parser, '_parser'):
        parser = parser._parser
    env_var = (parser._env_prefix+'_' if parser._env_prefix else '') + action.dest
    env_var = env_var.replace('.', '__').upper()
    return env_var


def _issubclass(cls, class_or_tuple):
    """Extension of issubclass that supports non-class argument."""
    return inspect.isclass(cls) and issubclass(cls, class_or_tuple)


def import_object(name: str):
    """Returns an object in a module given its dot import path."""
    if not isinstance(name, str) or '.' not in name:
        raise ValueError('Expected a dot import path string')
    name_module, name_object = name.rsplit('.', 1)
    module = __import__(name_module, fromlist=[name_object])
    return getattr(module, name_object)


lenient_check: ContextVar = ContextVar('lenient_check', default=False)


@contextmanager
def _lenient_check_context(caller=None):
    t = lenient_check.set(False if caller == 'argcomplete' else True)
    try:
        yield
    finally:
        lenient_check.reset(t)


@contextmanager
def _suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            yield None


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


def indent_text(text: str) -> str:
    return text.replace('\n', '\n  ')


def known_to_fsspec(path: str) -> bool:
    import_fsspec('known_to_fsspec')
    from fsspec.registry import known_implementations
    for protocol in known_implementations.keys():
        if path.startswith(protocol+'://') or path.startswith(protocol+'::'):
            return True
    return False


class DirectedGraph:
    def __init__(self):
        self.nodes = []
        self.edges_dict = defaultdict(list)

    def add_edge(self, source, target):
        for node in [source, target]:
            if node not in self.nodes:
                self.nodes.append(node)
        self.edges_dict[self.nodes.index(source)].append(self.nodes.index(target))

    def get_topological_order(self):
        exploring = [False]*len(self.nodes)
        visited = [False]*len(self.nodes)
        order = []
        for source in range(len(self.nodes)):
            if not visited[source]:
                self.topological_sort(source, exploring, visited, order)
        return [self.nodes[n] for n in order]

    def topological_sort(self, source, exploring, visited, order):
        exploring[source] = True
        for target in self.edges_dict[source]:
            if exploring[target]:
                raise ValueError(f'Graph has cycles, found while checking {self.nodes[source]} --> '+self.nodes[target])
            elif not visited[target]:
                self.topological_sort(target, exploring, visited, order)
        visited[source] = True
        exploring[source] = False
        order.insert(0, source)


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
            if re.match('^file:///?', abs_path):
                abs_path = re.sub('^file:///?', '/', abs_path)
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
                except FileNotFoundError:
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


class LoggerProperty:
    """Class designed to be inherited by other classes to add a logger property."""

    def __init__(self):
        """Initializer for LoggerProperty class."""
        if not hasattr(self, '_logger'):
            self.logger = None


    @property
    def logger(self):
        """The logger property for the class.

        :getter: Returns the current logger.
        :setter: Sets the given logging.Logger as logger or sets the default logger
                 if given True/str(logger name)/dict(name, level), or disables logging
                 if given False/None.

        Raises:
            ValueError: If an invalid logger value is given.
        """
        return self._logger


    @logger.setter
    def logger(self, logger):
        if logger is None or (isinstance(logger, bool) and not logger):
            self._logger = null_logger
        elif isinstance(logger, (bool, str, dict)) and logger:
            levels = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'}
            level = logging.WARNING
            if isinstance(logger, dict) and 'level' in logger:
                if logger['level'] not in levels:
                    raise ValueError(f'Logger level must be one of {levels}.')
                level = getattr(logging, logger['level'])
            if isinstance(logger, bool) or (isinstance(logger, dict) and 'name' not in logger):
                try:
                    import reconplogger
                    logger = reconplogger.logger_setup(level=level)
                except (ImportError, ValueError):
                    pass
            if not isinstance(logger, logging.Logger):
                name = type(self).__name__
                if isinstance(logger, str):
                    name = logger
                elif isinstance(logger, dict) and 'name' in logger:
                    name = logger['name']
                logger = logging.getLogger(name)
                if len(logger.handlers) == 0:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    logger.addHandler(handler)
                logger.setLevel(level)
            self._logger = logger
        elif not isinstance(logger, logging.Logger):
            raise ValueError('Expected logger to be an instance of logging.Logger or bool or str or dict or None.')
        else:
            self._logger = logger
