"""Collection of general functions and classes."""

import os
import re
import sys
import stat
import yaml
import inspect
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager, redirect_stderr
from argparse import Namespace
from yaml.parser import ParserError as yamlParserError
from yaml.scanner import ScannerError as yamlScannerError

from .optionals import (
    ModuleNotFound,
    url_support,
    import_requests,
    import_url_validator,
    fsspec_support,
    import_fsspec,
    get_config_read_mode,
)


__all__ = [
    'ParserError',
    'null_logger',
    'dict_to_namespace',
    'namespace_to_dict',
    'meta_keys',
    'strip_meta',
    'usage_and_exit_error_handler',
    'Path',
    'LoggerProperty',
]


null_logger = logging.Logger('jsonargparse_null_logger')
null_logger.addHandler(logging.NullHandler())

meta_keys = {'__default_config__', '__path__', '__orig__'}

empty_namespace = Namespace()

NoneType = type(None)


class ParserError(Exception):
    """Error raised when parsing a value fails."""
    pass


def _load_config(value, enable_path=True, flat_namespace=True):
    """Parses yaml config in a string or a path"""
    cfg_path = None
    if isinstance(value, str) and value.strip() != '':
        parsed_val = yaml.safe_load(value)
        if not isinstance(parsed_val, str):
            value = parsed_val
    if enable_path and isinstance(value, str):
        try:
            cfg_path = Path(value, mode=get_config_read_mode())
        except TypeError:
            pass
        else:
            value = yaml.safe_load(cfg_path.get_content())

    if flat_namespace and isinstance(value, dict):
        value = _dict_to_flat_namespace(value)
        if cfg_path is not None:
            setattr(value, '__path__', cfg_path)
        return value

    return value, cfg_path


def get_key_value_from_flat_dict(cfg, key):
    value = cfg.get(key)
    if value is not None:
        return value
    value = {k[len(key)+1:]: v for k, v in cfg.items() if k.startswith(key+'.')}
    return _flat_namespace_to_dict(Namespace(**value))


def update_key_value_in_flat_dict(cfg, key, value):
    if isinstance(value, dict):
        value = vars(_dict_to_flat_namespace(value))
        if key in cfg:
            del cfg[key]
        cfg.update({key+'.'+k: v for k, v in value.items()})
    else:
        cfg[key] = value


def _get_key_value(cfg, key, parent=False):
    """Gets the value for a given key in a config object (dict or argparse.Namespace)."""
    def key_in_cfg(cfg, key):
        if (isinstance(cfg, Namespace) and hasattr(cfg, key)) or \
           (isinstance(cfg, dict) and key in cfg):
            return True
        return False

    c = cfg
    k = key
    while '.' in k and not key_in_cfg(c, k):
        kp, k = k.split('.', 1)
        c = c[kp] if isinstance(c, dict) else getattr(c, kp)

    v = c[k] if isinstance(c, dict) else getattr(c, k)
    return (v, c, k) if parent else v


def _flat_namespace_to_dict(cfg_ns:Namespace) -> Dict[str, Any]:
    """Converts a flat namespace into a nested dictionary.

    Args:
        cfg_ns: The configuration to process.

    Returns:
        The nested configuration dictionary.
    """
    def raise_conflicting_base(base):
        raise ParserError('Conflicting namespace base: '+base)

    nested_keys = {k.rsplit('.', 1)[0]+'.' for k in vars(cfg_ns).keys() if '.' in k}
    skip_keys = {k for k, v in vars(cfg_ns).items() if v is None and k+'.' in nested_keys}

    cfg_ns = deepcopy(cfg_ns)
    cfg_dict = {}
    for k, v in vars(cfg_ns).items():
        if k in skip_keys:
            continue
        ksplit = k.split('.')
        if len(ksplit) == 1:
            if k in cfg_dict:
                raise_conflicting_base(k)
            #elif isinstance(v, list) and any([isinstance(x, Namespace) for x in v]):
            #    cfg_dict[k] = [namespace_to_dict(x) for x in v]
            #elif isinstance(v, Namespace):
            #    cfg_dict[k] = vars(v)  # type: ignore
            elif not (v is None and k in cfg_dict):
                cfg_dict[k] = v
        else:
            kdict = cfg_dict
            for num, kk in enumerate(ksplit[:len(ksplit)-1]):
                if kk not in kdict or kdict[kk] is None or kdict[kk] == empty_namespace:
                    kdict[kk] = {}
                elif not isinstance(kdict[kk], dict):
                    raise_conflicting_base('.'.join(ksplit[:num+1])+', expected dict but is '+str(kdict[kk]))
                kdict = kdict[kk]
            if ksplit[-1] in kdict and kdict[ksplit[-1]] is not None:
                raise_conflicting_base(k)
            #if isinstance(v, list) and any([isinstance(x, Namespace) for x in v]):
            #    kdict[ksplit[-1]] = [namespace_to_dict(x) for x in v]
            #elif not (v is None and ksplit[-1] in kdict):
            if not (v is None and ksplit[-1] in kdict):
                kdict[ksplit[-1]] = v
    return cfg_dict


def _dict_to_flat_namespace(cfg_dict:Dict[str, Any]) -> Namespace:
    """Converts a nested dictionary into a flat namespace.

    Args:
        cfg_dict: The configuration to process.

    Returns:
        The configuration namespace.
    """
    cfg_dict = deepcopy(cfg_dict)
    cfg_ns = {}

    def flatten_dict(cfg, base=None):
        for key, val in cfg.items():
            kbase = key if base is None else base+'.'+key
            if isinstance(val, dict) and val != {} and all(isinstance(k, str) for k in val.keys()):
                flatten_dict(val, kbase)
            else:
                cfg_ns[kbase] = val

    flatten_dict(cfg_dict)

    return Namespace(**cfg_ns)


def dict_to_namespace(cfg_dict:Dict[str, Any]) -> Namespace:
    """Converts a nested dictionary into a nested namespace.

    Args:
        cfg_dict: The configuration to process.

    Returns:
        The nested configuration namespace.
    """
    cfg_dict = deepcopy(cfg_dict)
    def expand_dict(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict) and all(isinstance(k, str) for k in v.keys()):
                cfg[k] = expand_dict(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, dict) and all(isinstance(k, str) for k in vv.keys()):
                        cfg[k][nn] = expand_dict(vv)
        return Namespace(**cfg)
    return expand_dict(cfg_dict)


def namespace_to_dict(cfg_ns:Namespace) -> Dict[str, Any]:
    """Converts a nested namespace into a nested dictionary.

    Args:
        cfg_ns: The configuration to process.

    Returns:
        The nested configuration dictionary.
    """
    cfg_ns = deepcopy(cfg_ns)
    def expand_namespace(cfg):
        if not isinstance(cfg, dict):
            cfg = dict(vars(cfg))
        for k, v in cfg.items():
            if isinstance(v, Namespace):
                cfg[k] = expand_namespace(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, Namespace):
                        cfg[k][nn] = expand_namespace(vv)
        return cfg
    return expand_namespace(cfg_ns)


def strip_meta(cfg:Union[Namespace, Dict[str, Any]]) -> Dict[str, Any]:
    """Removes all metadata keys from a configuration object.

    Args:
        cfg: The configuration object to strip.

    Returns:
        A copy of the configuration object without any metadata keys.
    """
    cfg = deepcopy(cfg)
    if not isinstance(cfg, dict):
        cfg = namespace_to_dict(cfg)

    def strip_keys(cfg, base=None):
        del_keys = []
        for key, val in cfg.items():
            kbase = str(key) if base is None else base+'.'+str(key)
            if isinstance(val, dict):
                strip_keys(val, kbase)
            elif key in meta_keys:
                del_keys.append(key)
        for key in del_keys:
            del cfg[key]

    strip_keys(cfg)
    return cfg


def usage_and_exit_error_handler(self, message:str):
    """Error handler to get the same behavior as in argparse.

    Args:
        self (ArgumentParser): The ArgumentParser object.
        message: The message describing the error being handled.
    """
    self.print_usage(sys.stderr)
    args = {'prog': self.prog, 'message': message}
    sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
    self.exit(2)


def _get_env_var(parser, action) -> str:
    """Returns the environment variable for a given parser and action."""
    if hasattr(parser, '_parser'):
        parser = parser._parser
    env_var = (parser._env_prefix+'_' if parser._env_prefix else '') + action.dest
    env_var = env_var.replace('.', '__').upper()
    return env_var


def _issubclass(cls, class_or_tuple):
    """Extension of issubclass that supports non-class argument."""
    return inspect.isclass(cls) and issubclass(cls, class_or_tuple)


def import_object(name):
    """Returns an object in a module given its dot import path."""
    if not isinstance(name, str):
        raise ValueError('Expected a dot import path string')
    name_module, name_object = name.rsplit('.', 1)
    module = __import__(name_module, fromlist=[name_object])
    return getattr(module, name_object)


@contextmanager
def _suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            yield None


@contextmanager
def change_to_path_dir(path):
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


def known_to_fsspec(path):
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
                raise ValueError('Graph has cycles, found while checking '+self.nodes[source]+' --> '+self.nodes[target])
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
                    raise TypeError(abs_path+' HEAD not accessible :: '+str(ex)) from ex
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
        self.is_url = is_url  # type: bool
        self.is_fsspec = is_fsspec  # type: bool
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
                    raise ValueError('Logger level must be one of '+str(levels)+'.')
                level = getattr(logging, logger['level'])
            if isinstance(logger, bool) or (isinstance(logger, dict) and 'name' not in logger):
                try:
                    import reconplogger
                    logger = reconplogger.logger_setup(level=level)
                except (ImportError, ModuleNotFound, ValueError):
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
