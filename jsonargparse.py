import os
import re
import sys
import stat
import glob
import json
import yaml
import logging
import operator
import argparse
import traceback
import importlib.util
from argparse import Action, OPTIONAL, REMAINDER, SUPPRESS, PARSER, ONE_OR_MORE, ZERO_OR_MORE
from argparse import ArgumentError, _UNRECOGNIZED_ARGS_ATTR
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, List, Dict, Set, Union

try:
    from contextlib import contextmanager, redirect_stderr
except:
    from contextlib2 import contextmanager, redirect_stderr  # type: ignore


jsonschema = jsonvalidator = importlib.util.find_spec('jsonschema')
_jsonnet = importlib.util.find_spec('_jsonnet')
url_validator = importlib.util.find_spec('validators')
requests = importlib.util.find_spec('requests')

jsonschema_support = False if jsonschema is None else True
jsonnet_support = False if any(x is None for x in [_jsonnet, jsonschema]) else True
url_support = False if any(x is None for x in [url_validator, requests]) else True


def import_jsonschema(importer):
    global jsonschema, jsonvalidator
    try:
        import jsonschema
        from jsonschema import Draft7Validator as jsonvalidator
    except Exception as ex:
        raise ImportError('jsonschema package is required by '+importer+' :: '+str(ex))


def import_jsonnet(importer):
    global _jsonnet
    try:
        import _jsonnet
    except Exception as ex:
        raise ImportError('jsonnet package is required by '+importer+' :: '+str(ex))


def import_url_validator(importer):
    global url_validator
    try:
        from validators.url import url as url_validator
    except Exception as ex:
        raise ImportError('validators package is required by '+importer+' :: '+str(ex))


def import_requests(importer):
    global requests
    try:
        import requests
    except Exception as ex:
        raise ImportError('requests package is required by '+importer+' :: '+str(ex))


__version__ = '2.32.1'


meta_keys = {'__cwd__', '__path__'}
config_read_mode = 'fr'


def set_url_support(enabled):
    """Enables/disables URL support for config read mode."""
    if enabled and not url_support:
        pkg = ['validators', 'requests']
        missing = {pkg[n] for n, x in enumerate([url_validator, requests]) if x is None}
        raise ImportError('Missing packages for URL support: '+str(missing))
    global config_read_mode
    config_read_mode = 'fur' if enabled else 'fr'


class ParserError(Exception):
    """Error raised when parsing a value fails."""
    pass


class DefaultHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help message formatter with namespace key, env var and default values in argument help.

    This class is an extension of `argparse.ArgumentDefaultsHelpFormatter
    <https://docs.python.org/3/library/argparse.html#argparse.ArgumentDefaultsHelpFormatter>`_.
    The main difference is that optional arguments are preceded by 'ARG:', the
    nested namespace key in dot notation is included preceded by 'NSKEY:', and
    if the ArgumentParser's default_env=True, the environment variable name is
    included preceded by 'ENV:'.
    """

    _env_prefix = None
    _default_env = True
    _conf_file = True

    def _format_action_invocation(self, action):
        if action.option_strings == [] or action.default == SUPPRESS or (not self._conf_file and not self._default_env):
            return super()._format_action_invocation(action)
        extr = ''
        if not isinstance(action, ActionConfigFile):
            extr += '\n  NSKEY: ' + action.dest
        if self._default_env:
            extr += '\n  ENV:   ' + _get_env_var(self, action)
        if isinstance(action, ActionParser):
            extr += '\n                        For more details run command with --'+action.dest+'.help.'
        return 'ARG:   ' + super()._format_action_invocation(action) + extr


null_logger = logging.Logger('null')
null_logger.addHandler(logging.NullHandler())


class LoggerProperty:
    """Class designed to be inherited by other classes to add a logger property."""

    def __init__(self):
        """Initializer for LoggerProperty class."""
        if not hasattr(self, '_logger'):
            self.logger = None


    @property
    def logger(self):
        """The current logger."""
        return self._logger


    @logger.setter
    def logger(self, logger):
        """Sets a new logger.

        Args:
            logger (logging.Logger or bool or str or dict or None): A logger
                object to use, or True/str(logger name)/dict(name, level) to use the
                default logger, or False/None to disable logging.

        Raises:
            ValueError: If an invalid logger value is given.
        """
        if logger is None or (isinstance(logger, bool) and not logger):
            self._logger = null_logger
        elif isinstance(logger, (bool, str, dict)) and logger:
            levels = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'}
            level = logging.INFO
            if isinstance(logger, dict) and 'level' in logger:
                if logger['level'] not in levels:
                    raise ValueError('Logger level must be one of '+str(levels)+'.')
                level = getattr(logging, logger['level'])
            name = type(self).__name__
            if isinstance(logger, str):
                name = logger
            elif isinstance(logger, dict) and 'name' in logger:
                name = logger['name']
            logger = logging.getLogger(name)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(level)
            self._logger = logger
        elif not isinstance(logger, logging.Logger):
            raise ValueError('Expected logger to be an instance of logging.Logger or bool or str or dict or None.')
        else:
            self._logger = logger


class _ActionsContainer(argparse._ActionsContainer):
    """Extension of argparse._ActionsContainer to support additional functionalities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('action', 'parsers', ActionSubCommands)


    def add_argument(self, *args, **kwargs):
        """Adds an argument to the parser or argument group.

        All the arguments from `argparse.ArgumentParser.add_argument
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
        are supported.
        """
        if 'type' in kwargs and kwargs['type'] == bool:
            if 'nargs' in kwargs:
                raise ValueError('Argument with type=bool does not support nargs.')
            kwargs['nargs'] = 1
            kwargs['action'] = ActionYesNo(no_prefix=None)
        action = super().add_argument(*args, **kwargs)
        for key in meta_keys:
            if key in action.dest:
                raise ValueError('Argument with destination name "'+key+'" not allowed.')
        parser = self.parser if hasattr(self, 'parser') else self  # pylint: disable=no-member
        if action.required:
            parser.required_args.add(action.dest)  # pylint: disable=no-member
            action.required = False
        if isinstance(action, ActionConfigFile) and parser.formatter_class == DefaultHelpFormatter:  # pylint: disable=no-member
            setattr(parser.formatter_class, '_conf_file', True)  # pylint: disable=no-member
        elif isinstance(action, ActionParser):
            _set_inner_parser_prefix(self, action.dest, action)
        return action


class _ArgumentGroup(_ActionsContainer, argparse._ArgumentGroup):
    """Extension of argparse._ArgumentGroup to support additional functionalities."""
    parser = None  # type: Union[ArgumentParser, None]


class ArgumentParser(_ActionsContainer, argparse.ArgumentParser, LoggerProperty):
    """Parser for command line, yaml/jsonnet files and environment variables."""

    groups = None  # type: Dict[str, argparse._ArgumentGroup]


    def __init__(self,
                 *args,
                 env_prefix=None,
                 error_handler=None,
                 formatter_class='default',
                 logger=None,
                 version=None,
                 parser_mode='yaml',
                 default_config_files:List[str]=[],
                 default_env:bool=False,
                 default_meta:bool=True,
                 **kwargs):
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            env_prefix (str): Prefix for environment variables.
            error_handler (Callable): Handler for parsing errors (default=None). For same behavior as argparse use :func:`usage_and_exit_error_handler`.
            formatter_class (argparse.HelpFormatter or str): Class for printing help messages or one of {"default", "default_argparse"}.
            logger (logging.Logger or bool or int or str or None): A logger to use or True/int(log level)/str(logger name)
                                                                   to use the default logger or False/None to disable logging.
            version (str): Program version string to add --version argument.
            parser_mode (str): Mode for parsing configuration files, either "yaml" or "jsonnet".
            default_config_files (list[str]): List of strings defining default config file locations. For example: :code:`['~/.config/myapp/*.yaml']`.
            default_env (bool): Set the default value on whether to parse environment variables.
            default_meta (bool): Set the default value on whether to include metadata in config objects.
        """
        if isinstance(formatter_class, str) and formatter_class not in {'default', 'default_argparse'}:
            raise ValueError('The only accepted values for formatter_class are {"default", "default_argparse"} or a HelpFormatter class.')
        if formatter_class == 'default':
            formatter_class = DefaultHelpFormatter
        elif formatter_class == 'default_argparse':
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
        kwargs['formatter_class'] = formatter_class
        if formatter_class == DefaultHelpFormatter:
            setattr(formatter_class, '_conf_file', False)
        if self.groups is None:
            self.groups = {}
        super().__init__(*args, **kwargs)
        self.required_args = set()  # type: Set[str]
        self._stderr = sys.stderr
        self._default_config_files = default_config_files
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.parser_mode = parser_mode
        self.logger = logger
        self.error_handler = error_handler
        if version is not None:
            self.add_argument('--version', action='version', version='%(prog)s '+version)
        if parser_mode not in {'yaml', 'jsonnet'}:
            raise ValueError('The only accepted values for parser_mode are {"yaml", "jsonnet"}.')
        if parser_mode == 'jsonnet':
            import_jsonnet('parser_mode=jsonnet')


    @property
    def error_handler(self):
        """The current error_handler."""
        return self._error_handler


    @error_handler.setter
    def error_handler(self, error_handler):
        """Sets a new value to the error_handler property.

        Args:
            error_handler (Callable or str or None): Handler for parsing errors (default=None). For same behavior as argparse use :func:`usage_and_exit_error_handler`.
        """
        if error_handler == 'usage_and_exit_error_handler':
            self._error_handler = usage_and_exit_error_handler
        elif callable(error_handler) or error_handler is None:
            self._error_handler = error_handler
        else:
            raise ValueError('error_handler can be either a Callable or the "usage_and_exit_error_handler" string or None.')


    def parse_known_args(self, args=None, namespace=None):
        """parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed."""
        raise NotImplementedError('parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed.')


    def _parse_known_args(self, args=None):
        """Parses known arguments for internal use only."""
        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)

        try:
            namespace, args = super()._parse_known_args(args, SimpleNamespace())
            if len(args) > 0:
                for action in self._actions:
                    if isinstance(action, ActionParser):
                        ns, args = action._parser._parse_known_args(args)
                        for key, val in vars(ns).items():
                            setattr(namespace, key, val)
                        if len(args) == 0:
                            break
            if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
                delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
            return namespace, args
        except (ArgumentError, ParserError):
            err = sys.exc_info()[1]
            self.error(str(err))


    def add_subparsers(self, **kwargs):
        """Raises a NotImplementedError."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommands method.')


    def add_subcommands(self, required=True, dest='subcommand', **kwargs):
        """Adds sub-command parsers to the ArgumentParser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        a required argument is accepted, dest by default is 'subcommand' and the
        values of the sub-command are stored using the sub-command's name as base key.
        """
        if 'required' in kwargs:
            required = kwargs.pop('required')
        subcommands = super().add_subparsers(dest=dest, **kwargs)
        subcommands.required = required
        _find_action(self, dest)._env_prefix = self.env_prefix
        return subcommands


    def parse_args(self, args=None, namespace=None, env:bool=None, defaults:bool=True, nested:bool=True, with_meta:bool=None):
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            types.SimpleNamespace: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            with _suppress_stderr():
                cfg, unk = self._parse_known_args(args=args)
                if unk:
                    self.error('Unrecognized arguments: %s' % ' '.join(unk))

            ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

            ActionParser._fix_conflicts(self, cfg)
            cfg_dict = namespace_to_dict(cfg)

            if nested:
                cfg_dict = _flat_namespace_to_dict(dict_to_namespace(cfg_dict))

            if env:
                cfg_dict = self._merge_config(cfg_dict, self.parse_env(defaults=defaults, nested=nested, _skip_check=True))

            elif defaults:
                cfg_dict = self._merge_config(cfg_dict, self.get_defaults(nested=nested))

            if not (with_meta or (with_meta is None and self._default_meta)):
                cfg_dict = strip_meta(cfg_dict)

            cfg_ns = dict_to_namespace(cfg_dict)
            self.check_config(cfg_ns)

            if with_meta or (with_meta is None and self._default_meta):
                if hasattr(cfg_ns, '__cwd__'):
                    if os.getcwd() not in cfg_ns.__cwd__:
                        cfg_ns.__cwd__.insert(0, os.getcwd())
                else:
                    cfg_ns.__cwd__ = [os.getcwd()]

            self._logger.info('Parsed arguments.')

            if not nested:
                return _dict_to_flat_namespace(namespace_to_dict(cfg_ns))

        except (TypeError, KeyError, ValueError) as ex:
            self.error(str(ex))

        return cfg_ns


    def parse_path(self, cfg_path:str, ext_vars:dict={}, env:bool=None, defaults:bool=True, nested:bool=True,
                   with_meta:bool=None, _skip_check:bool=False, _base=None) -> SimpleNamespace:
        """Parses a configuration file (yaml or jsonnet) given its path.

        Args:
            cfg_path (str or Path): Path to the configuration file to parse.
            ext_vars (dict): Optional external variables used for parsing jsonnet.
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            types.SimpleNamespace: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        fpath = Path(cfg_path, mode=config_read_mode)
        if not fpath.is_url:
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(fpath(absolute=False), os.pardir)))
        try:
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(cfg_str, cfg_path, ext_vars, env, defaults, nested, with_meta=with_meta,
                                           _skip_logging=True, _skip_check=_skip_check, _base=_base)
            if with_meta or (with_meta is None and self._default_meta):
                parsed_cfg.__path__ = fpath
        finally:
            if not fpath.is_url:
                os.chdir(cwd)

        self._logger.info('Parsed %s from path: %s', self.parser_mode, cfg_path)

        return parsed_cfg


    def parse_string(self, cfg_str:str, cfg_path:str='', ext_vars:dict={}, env:bool=None, defaults:bool=True, nested:bool=True,
                     with_meta:bool=None, _skip_logging:bool=False, _skip_check:bool=False, _base=None) -> SimpleNamespace:
        """Parses configuration (yaml or jsonnet) given as a string.

        Args:
            cfg_str (str): The configuration content.
            cfg_path (str): Optional path to original config path, just for error printing.
            ext_vars (dict): Optional external variables used for parsing jsonnet.
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            types.SimpleNamespace: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            cfg = self._load_cfg(cfg_str, cfg_path, ext_vars, _base)

            ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

            if nested:
                cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

            if env:
                cfg = self._merge_config(cfg, self.parse_env(defaults=defaults, nested=nested, _skip_check=True))

            elif defaults:
                cfg = self._merge_config(cfg, self.get_defaults(nested=nested))

            if not (with_meta or (with_meta is None and self._default_meta)):
                cfg = strip_meta(cfg)

            cfg_ns = dict_to_namespace(cfg)
            if not _skip_check:
                self.check_config(cfg_ns)

            if with_meta or (with_meta is None and self._default_meta):
                if hasattr(cfg_ns, '__cwd__'):
                    if os.getcwd() not in cfg_ns.__cwd__:
                        cfg_ns.__cwd__.insert(0, os.getcwd())
                else:
                    cfg_ns.__cwd__ = [os.getcwd()]

            if not _skip_logging:
                self._logger.info('Parsed %s string.', self.parser_mode)

        except (TypeError, KeyError) as ex:
            self.error(str(ex))

        return cfg_ns


    def _load_cfg(self, cfg_str:str, cfg_path:str='', ext_vars:dict=None, base=None) -> Dict[str, Any]:
        """Loads a configuration string (yaml or jsonnet) into a namespace checking all values against the parser.

        Args:
            cfg_str (str): The configuration content.
            cfg_path (str): Optional path to original config path, just for error printing.
            ext_vars (dict): Optional external variables used for parsing jsonnet.
            base (str or None): Base key to prepend.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        if self.parser_mode == 'jsonnet':
            ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
            cfg_str = _jsonnet.evaluate_snippet(cfg_path, cfg_str, ext_vars=ext_vars, ext_codes=ext_codes)  # type: ignore
        try:
            cfg = yaml.safe_load(cfg_str)
        except Exception as ex:
            raise type(ex)('Problems parsing config :: '+str(ex))
        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))
        if base is not None:
            cfg = {base+'.'+k: v for k, v in cfg.items()}
        for action in self._actions:
            if action.dest in cfg:
                value = self._check_value_key(action, cfg[action.dest], action.dest, cfg)
                if isinstance(action, ActionParser):
                    value = namespace_to_dict(_dict_to_flat_namespace(namespace_to_dict(value)))
                    if '__path__' in value:
                        value[action.dest+'.__path__'] = value.pop('__path__')
                    del cfg[action.dest]
                    cfg.update(value)
                else:
                    cfg[action.dest] = value
        return cfg


    def parse_object(self, cfg_obj:dict, cfg_base=None, env:bool=None, defaults:bool=True, nested:bool=True,
                     with_meta:bool=None, _skip_check:bool=False) -> SimpleNamespace:
        """Parses configuration given as an object.

        Args:
            cfg_obj (dict): The configuration object.
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            types.SimpleNamespace: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            cfg = vars(_dict_to_flat_namespace(cfg_obj))

            ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

            if nested:
                cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

            if cfg_base is not None:
                if isinstance(cfg_base, SimpleNamespace):
                    cfg_base = namespace_to_dict(cfg_base)
                cfg = self._merge_config(cfg, cfg_base)

            if env:
                cfg = self._merge_config(cfg, self.parse_env(defaults=defaults, nested=nested, _skip_check=True))

            elif defaults:
                cfg = self._merge_config(cfg, self.get_defaults(nested=nested))

            if not (with_meta or (with_meta is None and self._default_meta)):
                cfg = strip_meta(cfg)

            cfg_ns = dict_to_namespace(cfg)
            if not _skip_check:
                self.check_config(cfg_ns)

            if with_meta or (with_meta is None and self._default_meta):
                if hasattr(cfg_ns, '__cwd__'):
                    if os.getcwd() not in cfg_ns.__cwd__:
                        cfg_ns.__cwd__.insert(0, os.getcwd())
                else:
                    cfg_ns.__cwd__ = [os.getcwd()]

        except (TypeError, KeyError) as ex:
            self.error(str(ex))

        return cfg_ns


    def dump(self, cfg:Union[SimpleNamespace, dict], format:str='parser_mode', skip_none:bool=True, skip_check:bool=False) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object to dump.
            format (str): The output format: "yaml", "json", "json_indented" or "parser_mode".
            skip_none (bool): Whether to exclude checking values that are None.
            skip_check (bool): Whether to skip parser checking.

        Returns:
            str: The configuration in yaml or json format.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        cfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = namespace_to_dict(cfg)

        cfg = strip_meta(cfg)

        if not skip_check:
            self.check_config(cfg)

        def cleanup_actions(cfg, actions):
            for action in actions:
                if skip_none and action.dest in cfg and cfg[action.dest] is None:
                    del cfg[action.dest]
                elif isinstance(action, ActionPath):
                    if cfg[action.dest] is not None:
                        if isinstance(cfg[action.dest], list):
                            cfg[action.dest] = [p(absolute=False) for p in cfg[action.dest]]
                        else:
                            cfg[action.dest] = cfg[action.dest](absolute=False)
                elif isinstance(action, ActionConfigFile):
                    del cfg[action.dest]
                elif isinstance(action, ActionParser):
                    cleanup_actions(cfg, action._parser._actions)

        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))
        cleanup_actions(cfg, self._actions)
        cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

        if format == 'parser_mode':
            format = 'yaml' if self.parser_mode == 'yaml' else 'json_indented'
        if format == 'yaml':
            return yaml.dump(cfg, default_flow_style=False, allow_unicode=True)
        elif format == 'json_indented':
            return json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False)
        elif format == 'json':
            return json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        else:
            raise ValueError('Unknown output format '+str(format))


    def save(self, cfg:Union[SimpleNamespace, dict], path:str, format:str='parser_mode', skip_none:bool=True,
             skip_check:bool=False, overwrite:bool=False, multifile:bool=True, branch=None) -> None:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object to save.
            path (str): Path to the location where to save config.
            format (str): The output format: "yaml", "json", "json_indented" or "parser_mode".
            skip_none (bool): Whether to exclude checking values that are None.
            skip_check (bool): Whether to skip parser checking.
            overwrite (bool): Whether to overwrite existing files.
            multifile (bool): Whether to save multiple config files by using the __path__ metas.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        if not overwrite and os.path.isfile(path):
            raise ValueError('Refusing to overwrite existing file: '+path)
        path = Path(path, mode='fc')
        if format not in {'parser_mode', 'yaml', 'json_indented', 'json'}:
            raise ValueError('Unknown output format '+str(format))
        if format == 'parser_mode':
            format = 'yaml' if self.parser_mode == 'yaml' else 'json_indented'

        dump_kwargs = {'format': format, 'skip_none': skip_none, 'skip_check': skip_check}

        if not multifile:
            with open(path(), 'w') as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore

        else:
            cfg = deepcopy(cfg)
            if not isinstance(cfg, dict):
                cfg = namespace_to_dict(cfg)

            if not skip_check:
                self.check_config(strip_meta(cfg), branch=branch)

            dirname = os.path.dirname(path())
            save_kwargs = deepcopy(dump_kwargs)
            save_kwargs.update({'overwrite': overwrite, 'multifile': multifile})

            def save_paths(cfg, base=None):
                replace_keys = {}
                for key, val in cfg.items():
                    kbase = key if base is None else base+'.'+key
                    if isinstance(val, dict):
                        if '__path__' in val:
                            val_path = Path(os.path.join(dirname, os.path.basename(val['__path__']())), mode='fc')
                            if not overwrite and os.path.isfile(val_path()):
                                raise ValueError('Refusing to overwrite existing file: '+val_path)
                            action = _find_action(self, kbase)
                            if isinstance(action, ActionParser):
                                replace_keys[key] = val_path
                                action._parser.save(val, val_path(), branch=action.dest, **save_kwargs)
                            elif isinstance(action, (ActionJsonSchema, ActionJsonnet)):
                                replace_keys[key] = val_path
                                val_out = strip_meta(val)
                                if format == 'json_indented' or isinstance(action, ActionJsonnet):
                                    val_str = json.dumps(val_out, indent=2, sort_keys=True)
                                elif format == 'yaml':
                                    val_str = yaml.dump(val_out, default_flow_style=False, allow_unicode=True)
                                elif format == 'json':
                                    val_str = json.dumps(val_out, sort_keys=True)
                                with open(val_path(), 'w') as f:
                                    f.write(val_str)
                            else:
                                save_paths(val, kbase)
                        else:
                            save_paths(val, kbase)
                for key, val in replace_keys.items():
                    cfg[key] = os.path.basename(val())

            save_paths(cfg)
            dump_kwargs['skip_check'] = True
            with open(path(), 'w') as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore


    @property
    def default_env(self):
        """The current value of the default_env."""
        return self._default_env


    @default_env.setter
    def default_env(self, default_env):
        """Sets a new value to the default_env property.

        Args:
            default_env (bool): Whether default environment parsing is enabled or not.
        """
        self._default_env = default_env
        if self.formatter_class == DefaultHelpFormatter:
            setattr(self.formatter_class, '_default_env', default_env)


    @property
    def default_meta(self):
        """The current value of the default_meta."""
        return self._default_meta


    @default_meta.setter
    def default_meta(self, default_meta):
        """Sets a new value to the default_meta property.

        Args:
            default_meta (bool): Whether by default metadata is included in config objects.
        """
        self._default_meta = default_meta


    @property
    def env_prefix(self):
        """The current value of the env_prefix."""
        return self._env_prefix


    @env_prefix.setter
    def env_prefix(self, env_prefix):
        """Sets a new value to the env_prefix property.

        Args:
            env_prefix (str or None): Set prefix for environment variables, use None to derive it from prog.
        """
        if env_prefix is None:
            env_prefix = os.path.splitext(self.prog)[0]
        self._env_prefix = env_prefix
        if self.formatter_class == DefaultHelpFormatter:
            setattr(self.formatter_class, '_env_prefix', env_prefix)


    def parse_env(self, env:Dict[str, str]=None, defaults:bool=True, nested:bool=True, with_meta:bool=None,
                  _skip_logging:bool=False, _skip_check:bool=False) -> SimpleNamespace:
        """Parses environment variables.

        Args:
            env (dict[str, str]): The environment object to use, if None `os.environ` is used.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            types.SimpleNamespace: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            if env is None:
                env = dict(os.environ)
            cfg = {}  # type: ignore
            for action in self._actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, ActionConfigFile):
                    namespace = _dict_to_flat_namespace(cfg)
                    ActionConfigFile._apply_config(self, namespace, action.dest, env[env_var])
                    cfg = vars(namespace)
            for action in self._actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, ActionSubCommands):
                    env_val = env[env_var]
                    if env_val in action.choices:
                        cfg[action.dest] = subcommand = self._check_value_key(action, env_val, action.dest, cfg)
                        pcfg = action._name_parser_map[env_val].parse_env(env=env, defaults=defaults, nested=False, _skip_logging=True, _skip_check=True)  # type: ignore
                        for k, v in vars(pcfg).items():
                            cfg[subcommand+'.'+k] = v
            for action in [a for a in self._actions if a.default != SUPPRESS]:
                if isinstance(action, ActionParser):
                    subparser_cfg = {}
                    if defaults:
                        subparser_cfg = vars(action._parser.get_defaults(nested=False))
                    env_var = _get_env_var(self, action)
                    if env_var in env:
                        pcfg = self._check_value_key(action, env[env_var], action.dest, cfg)
                        subparser_cfg.update(vars(_dict_to_flat_namespace(namespace_to_dict(pcfg))))
                    pcfg = action._parser.parse_env(env=env, defaults=False, nested=False, with_meta=with_meta, _skip_logging=True, _skip_check=True)
                    subparser_cfg.update(namespace_to_dict(pcfg))
                    cfg.update(subparser_cfg)
                    continue
                env_var = _get_env_var(self, action)
                if env_var in env and not isinstance(action, ActionConfigFile):
                    env_val = env[env_var]
                    if _is_action_value_list(action):
                        if re.match('^ *\\[.+,.+] *$', env_val):
                            try:
                                env_val = yaml.safe_load(env_val)
                            except:
                                env_val = [env_val]  # type: ignore
                        else:
                            env_val = [env_val]  # type: ignore
                    cfg[action.dest] = self._check_value_key(action, env_val, action.dest, cfg)

            if nested:
                cfg = _flat_namespace_to_dict(SimpleNamespace(**cfg))

            if defaults:
                cfg = self._merge_config(cfg, self.get_defaults(nested=nested))

            if not (with_meta or (with_meta is None and self._default_meta)):
                cfg = strip_meta(cfg)

            cfg_ns = dict_to_namespace(cfg)
            if not _skip_check:
                self.check_config(cfg_ns)

            if with_meta or (with_meta is None and self._default_meta):
                if hasattr(cfg_ns, '__cwd__'):
                    if os.getcwd() not in cfg_ns.__cwd__:
                        cfg_ns.__cwd__.insert(0, os.getcwd())
                else:
                    cfg_ns.__cwd__ = [os.getcwd()]

            if not _skip_logging:
                self._logger.info('Parsed environment variables.')

        except TypeError as ex:
            self.error(str(ex))

        return cfg_ns


    def get_defaults(self, nested:bool=True) -> SimpleNamespace:
        """Returns a namespace with all default values.

        Args:
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all default values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            cfg = {}
            for action in self._actions:
                if action.default != SUPPRESS and action.dest != SUPPRESS:
                    if isinstance(action, ActionParser):
                        cfg.update(namespace_to_dict(action._parser.get_defaults(nested=False)))
                    else:
                        cfg[action.dest] = action.default

            cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))

            self._logger.info('Loaded default values from parser.')

            default_config_files = []  # type: List[str]
            for pattern in self._default_config_files:
                default_config_files += glob.glob(os.path.expanduser(pattern))
            if len(default_config_files) > 0:
                default_config = Path(default_config_files[0], mode=config_read_mode).get_content()
                cfg_file = self._load_cfg(default_config)
                cfg = self._merge_config(cfg_file, cfg)
                self._logger.info('Parsed configuration from default path: %s', default_config_files[0])

            if nested:
                cfg = _flat_namespace_to_dict(SimpleNamespace(**cfg))

        except TypeError as ex:
            self.error(str(ex))

        return dict_to_namespace(cfg)


    def error(self, message):
        """Logs error message if a logger is set, calls the error handler and raises a ParserError."""
        self._logger.error(message)
        if self._error_handler is not None:
            with redirect_stderr(self._stderr):
                self._error_handler(self, message)
        raise ParserError(message)


    def add_argument_group(self, *args, name:str=None, **kwargs):
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported. Additionally it accepts:

        Args:
            name (str): Name of the group. If set the group object will be included in the parser.groups dict.

        Returns:
            The group object.
        """
        group = _ArgumentGroup(self, *args, **kwargs)
        group.parser = self
        self._action_groups.append(group)
        if name is not None:
            self.groups[name] = group
        return group


    def check_config(self, cfg:Union[SimpleNamespace, dict], skip_none:bool=True, branch=None):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object to check.
            skip_none (bool): Whether to skip checking of values that are None.
            branch (str or None): Base key in case cfg corresponds only to a branch.

        Raises:
            TypeError: If any of the values are not valid.
            KeyError: If a key in cfg is not defined in the parser.
        """
        cfg = ccfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = namespace_to_dict(cfg)
        if isinstance(branch, str):
            cfg = _flat_namespace_to_dict(_dict_to_flat_namespace({branch: cfg}))

        def get_key_value(dct, key):
            keys = key.split('.')
            for key in keys:
                dct = dct[key]
            return dct

        def check_required(cfg):
            for reqkey in self.required_args:
                try:
                    val = get_key_value(cfg, reqkey)
                    if val is None:
                        raise TypeError('Key "'+reqkey+'" is required but its value is None.')
                except:
                    raise TypeError('Key "'+reqkey+'" is required but not included in config object.')

        def check_values(cfg, base=None):
            subcommand = None
            for key, val in cfg.items():
                if key in meta_keys:
                    continue
                kbase = key if base is None else base+'.'+key
                action = _find_action(self, kbase)
                if action is not None:
                    if val is None and skip_none:
                        continue
                    self._check_value_key(action, val, kbase, ccfg)
                    if isinstance(action, ActionSubCommands) and kbase != action.dest:
                        if subcommand is not None:
                            raise KeyError('Only values from a single sub-command are allowed ("'+subcommand+'", "'+kbase+'").')
                        subcommand = kbase
                elif isinstance(val, dict):
                    check_values(val, kbase)
                else:
                    raise KeyError('No action for key "'+kbase+'" to check its value.')

        try:
            check_required(cfg)
            check_values(cfg)
        except Exception as ex:
            trace = traceback.format_exc()
            self.error('Config checking failed :: '+str(ex)+' :: '+str(trace))


    def strip_unknown(self, cfg):
        """Removes all unknown keys from a configuration object.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object to strip.

        Returns:
            types.SimpleNamespace: The stripped configuration object.
        """
        cfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = namespace_to_dict(cfg)

        def strip_keys(cfg, base=None):
            del_keys = []
            for key, val in cfg.items():
                kbase = key if base is None else base+'.'+key
                action = _find_action(self, kbase)
                if action is not None:
                    pass
                elif isinstance(val, dict):
                    strip_keys(val, kbase)
                else:
                    del_keys.append(key)
            if base is None and any([k in del_keys for k in meta_keys]):
                del_keys = [v for v in del_keys if v not in meta_keys]
            for key in del_keys:
                del cfg[key]

        strip_keys(cfg)
        return dict_to_namespace(cfg)


    def get_config_files(self, cfg):
        """Returns a list of loaded config file paths.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object.

        Returns:
            list: Paths to loaded config files.
        """
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        cfg_files = []
        for action in self._actions:
            if isinstance(action, ActionConfigFile) and action.dest in cfg and cfg[action.dest] is not None:
                cfg_files = [p for p in cfg[action.dest] if p is not None]
        return cfg_files


    @staticmethod
    def merge_config(cfg_from:SimpleNamespace, cfg_to:SimpleNamespace) -> SimpleNamespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (types.SimpleNamespace): The configuration from which to merge.
            cfg_to (types.SimpleNamespace): The configuration into which to merge.

        Returns:
            types.SimpleNamespace: The merged configuration.
        """
        return dict_to_namespace(ArgumentParser._merge_config(cfg_from, cfg_to))


    @staticmethod
    def _merge_config(cfg_from:Union[SimpleNamespace, Dict[str, Any]], cfg_to:Union[SimpleNamespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (types.SimpleNamespace or dict): The configuration from which to merge.
            cfg_to (types.SimpleNamespace or dict): The configuration into which to merge.

        Returns:
            dict: The merged configuration.
        """
        def merge_values(cfg_from, cfg_to):
            for k, v in cfg_from.items():
                if v is None:
                    continue
                if k not in cfg_to or not isinstance(v, dict):
                    cfg_to[k] = v
                elif k in cfg_to and cfg_to[k] is None:
                    cfg_to[k] = cfg_from[k]
                else:
                    cfg_to[k] = merge_values(cfg_from[k], cfg_to[k])
            return cfg_to

        cfg_from = cfg_from if isinstance(cfg_from, dict) else namespace_to_dict(cfg_from)
        cfg_to = cfg_to if isinstance(cfg_to, dict) else namespace_to_dict(cfg_to)
        return merge_values(cfg_from, cfg_to.copy())


    @staticmethod
    def _check_value_key(action:Action, value:Any, key:str, cfg) -> Any:
        """Checks the value for a given action.

        Args:
            action (Action): The action used for parsing.
            value (Any): The value to parse.
            key (str): The configuration key.

        Raises:
            TypeError: If the value is not valid.
        """
        if action is None:
            raise ValueError('Parser key "'+str(key)+'": received action==None.')
        if action.choices is not None:
            if isinstance(action, ActionSubCommands):
                if key == action.dest:
                    if value not in action.choices:
                        raise KeyError('Unknown sub-command '+value+' (choices: '+', '.join(action.choices)+')')
                    return value
                parser = action._name_parser_map[key]
                parser.check_config(value)  # type: ignore
            else:
                vals = value if _is_action_value_list(action) else [value]
                if not all([v in action.choices for v in vals]):
                    args = {'value': value,
                            'choices': ', '.join(map(repr, action.choices))}
                    msg = 'invalid choice: %(value)r (choose from %(choices)s).'
                    raise TypeError('Parser key "'+str(key)+'": '+(msg % args))
        elif hasattr(action, '_check_type'):
            value = action._check_type(value, cfg=cfg)  # type: ignore
        elif action.type is not None:
            try:
                if action.nargs in {None, '?'} or action.nargs == 0:
                    value = action.type(value)
                else:
                    for k, v in enumerate(value):
                        value[k] = action.type(v)
            except (TypeError, ValueError) as ex:
                raise TypeError('Parser key "'+str(key)+'": '+str(ex))
        elif isinstance(action, argparse._StoreAction) and isinstance(value, dict):
            raise TypeError('StoreAction (key='+key+') does not allow dict value ('+str(value)+'), consider using ActionJsonSchema or ActionParser instead.')
        return value


def _get_key_value(cfg, key):
    """Gets the value for a given key in a config object (dict, SimpleNamespace or argparse.Namespace)."""
    def key_in_cfg(cfg, key):
        if isinstance(cfg, (SimpleNamespace, argparse.Namespace)) and hasattr(cfg, key):
            return True
        elif isinstance(cfg, dict) and key in cfg:
            return True
        return False

    c = cfg
    k = key
    while '.' in key and not key_in_cfg(c, k):
        kp, k = k.split('.', 1)
        c = c[kp] if isinstance(c, dict) else getattr(c, kp)

    return c[k] if isinstance(c, dict) else getattr(c, k)


def _flat_namespace_to_dict(cfg_ns:Union[SimpleNamespace, argparse.Namespace]) -> Dict[str, Any]:
    """Converts a flat namespace into a nested dictionary.

    Args:
        cfg_ns (types.SimpleNamespace): The configuration to process.

    Returns:
        dict: The nested configuration dictionary.
    """
    cfg_ns = deepcopy(cfg_ns)
    cfg_dict = {}
    for k, v in vars(cfg_ns).items():
        ksplit = k.split('.')
        if len(ksplit) == 1:
            if isinstance(v, list) and any([isinstance(x, SimpleNamespace) for x in v]):
                cfg_dict[k] = [namespace_to_dict(x) for x in v]
            elif isinstance(v, SimpleNamespace):
                cfg_dict[k] = vars(v)  # type: ignore
            elif not (v is None and k in cfg_dict):
                cfg_dict[k] = v
        else:
            kdict = cfg_dict
            for num, kk in enumerate(ksplit[:len(ksplit)-1]):
                if kk not in kdict or kdict[kk] is None:
                    kdict[kk] = {}  # type: ignore
                elif not isinstance(kdict[kk], dict):
                    raise ParserError('Conflicting namespace base: '+'.'.join(ksplit[:num+1]))
                kdict = kdict[kk]  # type: ignore
            if ksplit[-1] in kdict and kdict[ksplit[-1]] is not None:
                raise ParserError('Conflicting namespace base: '+k)
            if isinstance(v, list) and any([isinstance(x, SimpleNamespace) for x in v]):
                kdict[ksplit[-1]] = [namespace_to_dict(x) for x in v]
            elif not (v is None and ksplit[-1] in kdict):
                kdict[ksplit[-1]] = v
    return cfg_dict


def _dict_to_flat_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
    """Converts a nested dictionary into a flat namespace.

    Args:
        cfg_dict (dict): The configuration to process.

    Returns:
        types.SimpleNamespace: The configuration namespace.
    """
    cfg_dict = deepcopy(cfg_dict)
    cfg_ns = {}

    def flatten_dict(cfg, base=None):
        for key, val in cfg.items():
            kbase = key if base is None else base+'.'+key
            if isinstance(val, dict):
                flatten_dict(val, kbase)
            else:
                cfg_ns[kbase] = val

    flatten_dict(cfg_dict)

    return SimpleNamespace(**cfg_ns)


def dict_to_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
    """Converts a nested dictionary into a nested namespace.

    Args:
        cfg_dict (dict): The configuration to process.

    Returns:
        types.SimpleNamespace: The nested configuration namespace.
    """
    cfg_dict = deepcopy(cfg_dict)
    def expand_dict(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = expand_dict(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, dict):
                        cfg[k][nn] = expand_dict(vv)
        return SimpleNamespace(**cfg)
    return expand_dict(cfg_dict)


def namespace_to_dict(cfg_ns:SimpleNamespace) -> Dict[str, Any]:
    """Converts a nested namespace into a nested dictionary.

    Args:
        cfg_ns (types.SimpleNamespace): The configuration to process.

    Returns:
        dict: The nested configuration dictionary.
    """
    cfg_ns = deepcopy(cfg_ns)
    def expand_namespace(cfg):
        cfg = dict(vars(cfg))
        for k, v in cfg.items():
            if isinstance(v, SimpleNamespace):
                cfg[k] = expand_namespace(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, SimpleNamespace):
                        cfg[k][nn] = expand_namespace(vv)
        return cfg
    return expand_namespace(cfg_ns)


def strip_meta(cfg):
    """Removes all metadata keys from a configuration object.

    Args:
        cfg (types.SimpleNamespace or dict): The configuration object to strip.

    Returns:
        types.SimpleNamespace: The stripped configuration object.
    """
    cfg = deepcopy(cfg)
    if not isinstance(cfg, dict):
        cfg = namespace_to_dict(cfg)

    def strip_keys(cfg, base=None):
        del_keys = []
        for key, val in cfg.items():
            kbase = key if base is None else base+'.'+key
            if isinstance(val, dict):
                strip_keys(val, kbase)
            elif key in meta_keys:
                del_keys.append(key)
        for key in del_keys:
            del cfg[key]

    strip_keys(cfg)
    return cfg


class ActionConfigFile(Action):
    """Action to indicate that an argument is a configuration file or a configuration string."""
    def __init__(self, **kwargs):
        """Initializer for ActionConfigFile instance."""
        opt_name = kwargs['option_strings']
        opt_name = opt_name[0] if len(opt_name) == 1 else [x for x in opt_name if x[0:2] == '--'][0]
        if '.' in opt_name:
            raise ValueError('ActionConfigFile must be a top level option.')
        kwargs['type'] = str
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self._apply_config(parser, namespace, self.dest, values)

    @staticmethod
    def _apply_config(parser, namespace, dest, value):
        if not hasattr(namespace, dest) or not isinstance(getattr(namespace, dest), list):
            setattr(namespace, dest, [])
        try:
            cfg_path = Path(value, mode=config_read_mode)
        except TypeError as ex_path:
            if isinstance(yaml.safe_load(value), str):
                raise ex_path
            try:
                cfg_path = None
                cfg_file = parser.parse_string(value, env=False, defaults=False, _skip_check=True)
            except TypeError as ex_str:
                raise TypeError('Parser key "'+dest+'": '+str(ex_str))
        else:
            cfg_file = parser.parse_path(value, env=False, defaults=False, _skip_check=True)
        cfg_file = _dict_to_flat_namespace(namespace_to_dict(cfg_file))
        getattr(namespace, dest).append(cfg_path)
        for key, val in vars(cfg_file).items():
            if key == '__cwd__' and hasattr(namespace, '__cwd__'):
                setattr(namespace, key, getattr(namespace, key)+val)
            else:
                setattr(namespace, key, val)


class ActionYesNo(Action):
    """Paired options --{yes_prefix}opt, --{no_prefix}opt to set True or False respectively."""
    def __init__(self, **kwargs):
        """Initializer for ActionYesNo instance.

        Args:
            yes_prefix (str): Prefix for yes option (default='').
            no_prefix (str or None): Prefix for no option (default='no_').

        Raises:
            ValueError: If a parameter is invalid.
        """
        self._yes_prefix = ''
        self._no_prefix = 'no_'
        if 'yes_prefix' in kwargs or 'no_prefix' in kwargs or len(kwargs) == 0:
            _check_unknown_kwargs(kwargs, {'yes_prefix', 'no_prefix'})
            if 'yes_prefix' in kwargs:
                self._yes_prefix = kwargs['yes_prefix']
            if 'no_prefix' in kwargs:
                self._no_prefix = kwargs['no_prefix']
        elif 'option_strings' not in kwargs:
            raise ValueError('Expected yes_prefix and/or no_prefix keyword arguments.')
        else:
            self._yes_prefix = kwargs.pop('_yes_prefix') if '_yes_prefix' in kwargs else ''
            self._no_prefix = kwargs.pop('_no_prefix') if '_no_prefix' in kwargs else 'no_'
            if len(kwargs['option_strings']) == 0:
                raise ValueError(type(self).__name__+' not intended for positional arguments  ('+kwargs['dest']+').')
            opt_name = kwargs['option_strings'][0]
            if not opt_name.startswith('--'+self._yes_prefix):
                raise ValueError('Expected option string to start with "--'+self._yes_prefix+'".')
            if 'dest' not in kwargs:
                kwargs['dest'] = re.sub('^--', '', opt_name).replace('-', '_')
            if self._no_prefix is not None:
                kwargs['option_strings'] += [re.sub('^--'+self._yes_prefix, '--'+self._no_prefix, opt_name)]
            if self._no_prefix is None and 'nargs' in kwargs and kwargs['nargs'] != 1:
                raise ValueError('ActionYesNo with no_prefix=None only supports nargs=1.')
            if 'nargs' in kwargs and kwargs['nargs'] in {'?', 1}:
                kwargs['metavar'] = 'true|yes|false|no'
                if kwargs['nargs'] == 1:
                    kwargs['nargs'] = None
            else:
                kwargs['nargs'] = 0
                kwargs['metavar'] = None
            if 'default' not in kwargs:
                kwargs['default'] = False
            kwargs['type'] = ActionYesNo._boolean_type
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Sets the corresponding key to True or False depending on the option string used."""
        if len(args) == 0:
            kwargs['_yes_prefix'] = self._yes_prefix
            kwargs['_no_prefix'] = self._no_prefix
            return ActionYesNo(**kwargs)
        value = args[2][0] if isinstance(args[2], list) and len(args[2]) == 1 else args[2] if isinstance(args[2], bool) else True
        if self._no_prefix is not None and args[3].startswith('--'+self._no_prefix):
            setattr(args[1], self.dest, not value)
        else:
            setattr(args[1], self.dest, value)

    def _add_dest_prefix(self, prefix):
        self.dest = prefix+'.'+self.dest
        self.option_strings[0] = re.sub('^--'+self._yes_prefix, '--'+self._yes_prefix+prefix+'.', self.option_strings[0])
        if self._no_prefix is not None:
            self.option_strings[-1] = re.sub('^--'+self._no_prefix, '--'+self._no_prefix+prefix+'.', self.option_strings[-1])
        for n in range(1, len(self.option_strings)-1):
            self.option_strings[n] = re.sub('^--', '--'+prefix+'.', self.option_strings[n])

    def _check_type(self, value, cfg=None):
        if isinstance(value, list):
            value = [ActionYesNo._boolean_type(val) for val in value]
        else:
            value = ActionYesNo._boolean_type(value)
        if isinstance(value, list) and (self.nargs == 0 or self.nargs):
            return value[0]
        return value

    @staticmethod
    def _boolean_type(x):
        if isinstance(x, str) and x.lower() in {'true', 'yes', 'false', 'no'}:
            x = True if x.lower() in {'true', 'yes'} else False
        elif not isinstance(x, bool):
            raise TypeError('Value not boolean: '+str(x)+'.')
        return x


class ActionJsonSchema(Action):
    """Action to parse option as json validated by a jsonschema."""
    def __init__(self, **kwargs):
        """Initializer for ActionJsonSchema instance.

        Args:
            schema (str or object): Schema to validate values against.
            with_meta (bool): Whether to include metadata (def.=True).

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if 'schema' in kwargs:
            import_jsonschema('ActionJsonSchema')
            _check_unknown_kwargs(kwargs, {'schema', 'with_meta'})
            schema = kwargs['schema']
            if isinstance(schema, str):
                try:
                    schema = yaml.safe_load(schema)
                except Exception as ex:
                    raise type(ex)('Problems parsing schema :: '+str(ex))
            jsonvalidator.check_schema(schema)
            self._validator = self._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            self._with_meta = kwargs['with_meta'] if 'with_meta' in kwargs else True
        elif '_validator' not in kwargs:
            raise ValueError('Expected schema keyword argument.')
        else:
            self._validator = kwargs.pop('_validator')
            self._with_meta = kwargs.pop('_with_meta')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument validating against the corresponding jsonschema.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_validator'] = self._validator
            kwargs['_with_meta'] = self._with_meta
            if 'help' in kwargs and '%s' in kwargs['help']:
                kwargs['help'] = kwargs['help'] % json.dumps(self._validator.schema, indent=2, sort_keys=True)
            return ActionJsonSchema(**kwargs)
        val = self._check_type(args[2])
        if not self._with_meta:
            val = strip_meta(val)
        setattr(args[1], self.dest, val)

    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionJsonSchema with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        for num, val in enumerate(value):
            try:
                fpath = None
                if isinstance(val, str):
                    val = yaml.safe_load(val)
                if isinstance(val, str):
                    try:
                        fpath = Path(val, mode=config_read_mode)
                    except:
                        pass
                    else:
                        val = yaml.safe_load(fpath.get_content())
                if isinstance(val, SimpleNamespace):
                    val = namespace_to_dict(val)
                path_meta = val.pop('__path__') if isinstance(val, dict) and '__path__' in val else None
                self._validator.validate(val)
                if path_meta is not None:
                    val['__path__'] = path_meta
                if isinstance(val, dict) and fpath is not None:
                    val['__path__'] = fpath
                value[num] = val
            except (TypeError, yaml.parser.ParserError, jsonschema.exceptions.ValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex))
        return value if islist else value[0]

    @staticmethod
    def _extend_jsonvalidator_with_default(validator_class):
        """Extends a json schema validator so that it fills in default values."""
        validate_properties = validator_class.VALIDATORS['properties']

        def set_defaults(validator, properties, instance, schema):
            for property, subschema in properties.items():
                if 'default' in subschema:
                    instance.setdefault(property, subschema['default'])

            for error in validate_properties(validator, properties, instance, schema):
                yield error

        return jsonschema.validators.extend(validator_class, {'properties': set_defaults})


class ActionJsonnetExtVars(ActionJsonSchema):
    """Action to be used for jsonnet ext_vars."""
    def __init__(self, **kwargs):
        super().__init__(schema={'type': 'object'}, with_meta=False)


class ActionJsonnet(Action):
    """Action to parse a jsonnet, optionally validating against a jsonschema."""
    def __init__(self, **kwargs):
        """Initializer for ActionJsonnet instance.

        Args:
            ext_vars (str or None): Key where to find the external variables required to parse the jsonnet.
            schema (str or object or None): Schema to validate values against. Keyword argument required even if schema=None.

        Raises:
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if 'ext_vars' in kwargs or 'schema' in kwargs:
            import_jsonnet('ActionJsonnet')
            _check_unknown_kwargs(kwargs, {'schema', 'ext_vars'})
            if 'ext_vars' in kwargs and not isinstance(kwargs['ext_vars'], (str, type(None))):
                raise ValueError('ext_vars has to be either None or a string.')
            self._ext_vars = kwargs['ext_vars'] if 'ext_vars' in kwargs else None
            schema = kwargs['schema'] if 'schema' in kwargs else None
            if schema is not None:
                import_jsonschema('ActionJsonnet')
                if isinstance(schema, str):
                    try:
                        schema = yaml.safe_load(schema)
                    except Exception as ex:
                        raise type(ex)('Problems parsing schema :: '+str(ex))
                jsonvalidator.check_schema(schema)
                self._validator = ActionJsonSchema._extend_jsonvalidator_with_default(jsonvalidator)(schema)
            else:
                self._validator = None
        elif '_ext_vars' not in kwargs or '_validator' not in kwargs:
            raise ValueError('Expected ext_vars and/or schema keyword arguments.')
        else:
            self._ext_vars = kwargs.pop('_ext_vars')
            self._validator = kwargs.pop('_validator')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as jsonnet using ext_vars if defined.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_ext_vars'] = self._ext_vars
            kwargs['_validator'] = self._validator
            if 'help' in kwargs and '%s' in kwargs['help'] and self._validator is not None:
                kwargs['help'] = kwargs['help'] % json.dumps(self._validator.schema, indent=2, sort_keys=True)
            return ActionJsonnet(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2], cfg=args[1]))

    def _check_type(self, value, cfg):
        islist = _is_action_value_list(self)
        ext_vars = {}
        if self._ext_vars is not None:
            try:
                ext_vars = _get_key_value(cfg, self._ext_vars)
            except Exception as ex:
                raise ValueError('Unable to find key "'+self._ext_vars+'" in config object :: '+str(ex))
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionJsonnet with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        for num, val in enumerate(value):
            try:
                if isinstance(val, str):
                    val = self.parse(val, ext_vars=ext_vars, with_meta=True)
                elif self._validator is not None:
                    if isinstance(val, SimpleNamespace):
                        self._validator.validate(namespace_to_dict(val))
                    else:
                        self._validator.validate(val)
                value[num] = val
            except (TypeError, RuntimeError, yaml.parser.ParserError, jsonschema.exceptions.ValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': '+str(ex))
        return value if islist else value[0]

    @staticmethod
    def split_ext_vars(ext_vars):
        """Splits an ext_vars dict into the ext_codes and ext_vars required by jsonnet.

        Args:
            ext_vars (dict): External variables. Values can be strings or any other basic type.
        """
        if ext_vars is None:
            ext_vars = {}
        elif isinstance(ext_vars, SimpleNamespace):
            ext_vars = namespace_to_dict(ext_vars)
        ext_codes = {k: json.dumps(v) for k, v in ext_vars.items() if not isinstance(v, str)}
        ext_vars = {k: v for k, v in ext_vars.items() if isinstance(v, str)}
        return ext_vars, ext_codes

    def parse(self, jsonnet, ext_vars={}, with_meta=False):
        """Method that can be used to parse jsonnet independent from an ArgumentParser.

        Args:
            jsonnet (str): Either a path to a jsonnet file or the jsonnet content.
            ext_vars (dict): External variables. Values can be strings or any other basic type.

        Returns:
            SimpleNamespace: The parsed jsonnet object.

        Raises:
            TypeError: If the input is neither a path to an existent file nor a jsonnet.
        """
        ext_vars, ext_codes = self.split_ext_vars(ext_vars)
        fpath = None
        fname = 'snippet'
        snippet = jsonnet
        try:
            fpath = Path(jsonnet, mode=config_read_mode)
        except:
            pass
        else:
            fname = jsonnet(absolute=False) if isinstance(jsonnet, Path) else jsonnet
            snippet = fpath.get_content()
        try:
            values = yaml.safe_load(_jsonnet.evaluate_snippet(fname, snippet, ext_vars=ext_vars, ext_codes=ext_codes))
        except Exception as ex:
            raise ParserError('Problems evaluating jsonnet "'+fname+'" :: '+str(ex))
        if self._validator is not None:
            self._validator.validate(values)
        if with_meta and isinstance(values, dict) and fpath is not None:
            values['__path__'] = fpath
        return dict_to_namespace(values)


class ActionParser(Action):
    """Action to parse option with a given parser optionally loading from file if string value."""
    def __init__(self, **kwargs):
        """Initializer for ActionParser instance.

        Args:
            parser (ArgumentParser): A parser to parse the option with.

        Raises:
            ValueError: If the parser parameter is invalid.
        """
        if 'parser' in kwargs:
            ## Runs when first initializing class by external user ##
            _check_unknown_kwargs(kwargs, {'parser'})
            self._parser = kwargs['parser']
            if not isinstance(self._parser, ArgumentParser):
                raise ValueError('Expected parser keyword argument to be an ArgumentParser.')
        elif '_parser' not in kwargs:
            raise ValueError('Expected parser keyword argument.')
        else:
            ## Runs when initialied from the __call__ method below ##
            self._parser = kwargs.pop('_parser')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument with the corresponding parser and if valid, sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            ## Runs when within _ActionsContainer super().add_argument call ##
            kwargs['_parser'] = self._parser
            return ActionParser(**kwargs)
        ## Runs when parsing a value ##
        value = _dict_to_flat_namespace(namespace_to_dict(self._check_type(args[2])))
        for key, val in vars(value).items():
            setattr(args[1], key, val)
        if hasattr(value, '__path__'):
            setattr(args[1], self.dest+'.__path__', getattr(value, '__path__'))

    def _check_type(self, value, cfg=None):
        try:
            fpath = None
            if isinstance(value, str):
                value = yaml.safe_load(value)
            if isinstance(value, str):
                fpath = Path(value, mode=config_read_mode)
                value = self._parser.parse_path(fpath, _base=self.dest)
            else:
                value = dict_to_namespace(_flat_namespace_to_dict(dict_to_namespace({self.dest: value})))
                self._parser.check_config(value, skip_none=True)
            if fpath is not None:
                value.__path__ = fpath
        except TypeError as ex:
            raise TypeError(re.sub('^Parser key ([^:]+):', 'Parser key '+self.dest+'.\\1: ', str(ex)))
        return value

    @staticmethod
    def _fix_conflicts(parser, cfg):
        cfg_dict = namespace_to_dict(cfg)
        for action in parser._actions:
            if isinstance(action, ActionParser) and action.dest in cfg_dict and cfg_dict[action.dest] is None:
                children = [x for x in cfg_dict.keys() if x.startswith(action.dest+'.')]
                if len(children) > 0:
                    delattr(cfg, action.dest)


class ActionSubCommands(argparse._SubParsersAction):
    """Extension of argparse._SubParsersAction to modify sub-commands functionality."""

    _env_prefix = None


    def add_parser(self, **kwargs):
        """Raises a NotImplementedError."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommand method.')


    def add_subcommand(self, name, parser, **kwargs):
        """Adds a parser as a sub-command parser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        add_parser requires to be given a parser as argument.
        """
        parser.prog = '%s %s' % (self._prog_prefix, name)
        parser.env_prefix = self._env_prefix+'_'+name+'_'

        # create a pseudo-action to hold the choice help
        aliases = kwargs.pop('aliases', ())
        if 'help' in kwargs:
            help = kwargs.pop('help')
            choice_action = self._ChoicesPseudoAction(name, aliases, help)
            self._choices_actions.append(choice_action)

        # add the parser to the name-parser map
        self._name_parser_map[name] = parser
        for alias in aliases:
            self._name_parser_map[alias] = parser

        return parser


    def __call__(self, parser, namespace, values, option_string=None):
        """Adds sub-command dest and parses sub-command arguments."""
        subcommand = values[0]
        arg_strings = values[1:]

        # set the parser name
        setattr(namespace, self.dest, subcommand)

        # parse arguments
        if subcommand in self._name_parser_map:
            subparser = self._name_parser_map[subcommand]
            subnamespace, unk = subparser._parse_known_args(arg_strings)
            if unk:
                raise ParserError('Unrecognized arguments: %s' % ' '.join(unk))
            for key, value in vars(subnamespace).items():
                setattr(namespace, subcommand+'.'+key, value)


    @staticmethod
    def handle_subcommands(parser, cfg, env, defaults):
        """Adds sub-command dest if missing and parses defaults and environment variables."""
        if parser._subparsers is None:
            return

        cfg_dict = cfg.__dict__ if isinstance(cfg, SimpleNamespace) else cfg

        # Get subcommands action
        for action in parser._actions:
            if isinstance(action, ActionSubCommands):
                break

        # Get sub-command parser
        subcommand = None
        if action.dest in cfg_dict and cfg_dict[action.dest] is not None:
            subcommand = cfg_dict[action.dest]
        else:
            #for key in action._name_parser_map.keys():
            for key in action.choices.keys():
                if any([v.startswith(key+'.') for v in cfg_dict.keys()]):
                    subcommand = key
                    break
            cfg_dict[action.dest] = subcommand

        if subcommand in action._name_parser_map:
            subparser = action._name_parser_map[subcommand]
        else:
            raise ParserError('Unknown sub-commad '+subcommand+' (choices: '+', '.join(action.choices)+')')

        # merge environment variable values and default values
        subnamespace = None
        if env:
            subnamespace = subparser.parse_env(defaults=defaults, nested=False, _skip_check=True)
        elif defaults:
            subnamespace = subparser.get_defaults(nested=False)

        if subnamespace is not None:
            for key, value in vars(subnamespace).items():
                key = subcommand+'.'+key
                if key not in cfg_dict:
                    cfg_dict[key] = value


class ActionOperators(Action):
    """Action to restrict a value with comparison operators."""
    _operators = {operator.gt: '>', operator.ge: '>=', operator.lt: '<', operator.le: '<=', operator.eq: '==', operator.ne: '!='}

    def __init__(self, **kwargs):
        """Initializer for ActionOperators instance.

        Args:
            expr (tuple or list[tuple]): Pairs of operators (> >= < <= == !=) and reference values, e.g. [('>=', 1),...].
            join (str): How to combine multiple comparisons, must be 'or' or 'and' (default='and').
            type (type): The value type (default=int).

        Raises:
            ValueError: If any of the parameters (expr, join or type) are invalid.
        """
        if 'expr' in kwargs:
            _check_unknown_kwargs(kwargs, {'expr', 'join', 'type'})
            self._type = kwargs['type'] if 'type' in kwargs else int
            self._join = kwargs['join'] if 'join' in kwargs else 'and'
            if self._join not in {'or', 'and'}:
                raise ValueError("Expected join to be either 'or' or 'and'.")
            _operators = {v: k for k, v in self._operators.items()}
            expr = [kwargs['expr']] if isinstance(kwargs['expr'], tuple) else kwargs['expr']
            if not isinstance(expr, list) or not all([all([len(x) == 2, x[0] in _operators, x[1] == self._type(x[1])]) for x in expr]):
                raise ValueError('Expected expr to be a list of tuples each with a comparison operator (> >= < <= == !=)'
                                 ' and a reference value of type '+self._type.__name__+'.')
            self._expr = [(_operators[x[0]], x[1]) for x in expr]
        elif '_expr' not in kwargs:
            raise ValueError('Expected expr keyword argument.')
        else:
            self._expr = kwargs.pop('_expr')
            self._join = kwargs.pop('_join')
            self._type = kwargs.pop('_type')
            if 'type' in kwargs:
                del kwargs['type']
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument restricted by the operators and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise ValueError('Invalid nargs='+str(kwargs['nargs'])+' for ActionOperators.')
            kwargs['_expr'] = self._expr
            kwargs['_join'] = self._join
            kwargs['_type'] = self._type
            return ActionOperators(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionOperators with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        def test_op(op, val, ref):
            try:
                return op(val, ref)
            except TypeError:
                return False
        for num, val in enumerate(value):
            try:
                val = self._type(val)
            except:
                raise TypeError('Parser key "'+self.dest+'": invalid value, expected type to be '+self._type.__name__+' but got as value '+str(val)+'.')
            check = [test_op(op, val, ref) for op, ref in self._expr]
            if (self._join == 'and' and not all(check)) or (self._join == 'or' and not any(check)):
                expr = (' '+self._join+' ').join(['v'+self._operators[op]+str(ref) for op, ref in self._expr])
                raise TypeError('Parser key "'+self.dest+'": invalid value, for v='+str(val)+' it is false that '+expr+'.')
            value[num] = val
        return value if islist else value[0]


class ActionPath(Action):
    """Action to check and store a path."""
    def __init__(self, **kwargs):
        """Initializer for ActionPath instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument, e.g. ActionPath(mode='drw').
            skip_check (bool): Whether to skip path checks (def.=False).

        Raises:
            ValueError: If the mode parameter is invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode', 'skip_check'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
            self._skip_check = kwargs['skip_check'] if 'skip_check' in kwargs else False
        elif '_mode' not in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a Path and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid Path.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise ValueError('Invalid nargs='+str(kwargs['nargs'])+' for ActionPath.')
            kwargs['_mode'] = self._mode
            kwargs['_skip_check'] = self._skip_check
            return ActionPath(**kwargs)
        if hasattr(self, 'nargs') and self.nargs == '?' and args[2] is None:
            setattr(args[1], self.dest, args[2])
        else:
            setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None, islist=None):
        islist = _is_action_value_list(self) if islist is None else islist
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionPath with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        try:
            for num, val in enumerate(value):
                if isinstance(val, str):
                    val = Path(val, mode=self._mode, skip_check=self._skip_check)
                elif isinstance(val, Path):
                    val = Path(val(absolute=False), mode=self._mode, skip_check=self._skip_check, cwd=val.cwd)
                else:
                    raise TypeError('expected either a string or a Path object, received: value='+str(val)+' type='+str(type(val))+'.')
                value[num] = val
        except TypeError as ex:
            raise TypeError('Parser key "'+self.dest+'": '+str(ex))
        return value if islist else value[0]


class ActionPathList(Action):
    """Action to check and store a list of file paths read from a plain text file or stream."""
    def __init__(self, **kwargs):
        """Initializer for ActionPathList instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument (uppercase means not), e.g. ActionPathList(mode='fr').
            skip_check (bool): Whether to skip path checks (def.=False).
            rel (str): Whether relative paths are with respect to current working directory 'cwd' or the list's parent directory 'list' (default='cwd').

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode', 'skip_check', 'rel'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
            self._skip_check = kwargs['skip_check'] if 'skip_check' in kwargs else False
            self._rel = kwargs['rel'] if 'rel' in kwargs else 'cwd'
            if self._rel not in {'cwd', 'list'}:
                raise ValueError('rel must be either "cwd" or "list", got '+str(self._rel)+'.')
        elif '_mode' not in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
            self._rel = kwargs.pop('_rel')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a PathList and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid PathList.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] not in {'+', 1}:
                raise ValueError('ActionPathList only supports nargs of 1 or "+".')
            kwargs['_mode'] = self._mode
            kwargs['_skip_check'] = self._skip_check
            kwargs['_rel'] = self._rel
            return ActionPathList(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None):
        if value == []:
            return value
        islist = _is_action_value_list(self)
        if not islist and not isinstance(value, list):
            value = [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            path_list_files = value
            value = []
            for path_list_file in path_list_files:
                try:
                    with sys.stdin if path_list_file == '-' else open(path_list_file, 'r') as f:
                        path_list = [x.strip() for x in f.readlines()]
                except:
                    raise TypeError('Problems reading path list: '+path_list_file)
                cwd = os.getcwd()
                if self._rel == 'list' and path_list_file != '-':
                    os.chdir(os.path.abspath(os.path.join(path_list_file, os.pardir)))
                try:
                    for num, val in enumerate(path_list):
                        try:
                            path_list[num] = Path(val, mode=self._mode)
                        except TypeError as ex:
                            raise TypeError('Path number '+str(num+1)+' in list '+path_list_file+', '+str(ex))
                finally:
                    os.chdir(cwd)
                value += path_list
            return value
        else:
            return ActionPath._check_type(self, value, islist=True)


class Path(object):
    """Stores a (possibly relative) path and the corresponding absolute path.

    When a Path instance is created it is checked that: the path exists, whether
    it is a file or directory and whether has the required access permissions
    (f=file, d=directory, r=readable, w=writeable, x=executable, c=creatable,
    u=url or in uppercase meaning not, i.e., F=not-file, D=not-directory,
    R=not-readable, W=not-writeable and X=not-executable). The absolute path can
    be obtained without having to remember the working directory from when the
    object was created.
    """
    def __init__(self, path, mode:str='fr', cwd:str=None, skip_check:bool=False):
        """Initializer for Path instance.

        Args:
            path (str or Path): The path to check and store.
            mode (str): The required type and access permissions among [fdrwxcuFDRWX].
            cwd (str): Working directory for relative paths. If None, then os.getcwd() is used.
            skip_check (bool): Whether to skip path checks.

        Raises:
            ValueError: If the provided mode is invalid.
            TypeError: If the path does not exist or does not agree with the mode.
        """
        self._check_mode(mode)
        if cwd is None:
            cwd = os.getcwd()

        if isinstance(cwd, list):
            cwd = cwd[0]  # Temporal until multiple cwds is implemented.

        is_url = False
        if isinstance(path, Path):
            is_url = path.is_url
            cwd = path.cwd  # type: ignore
            abs_path = path.abs_path  # type: ignore
            path = path.path  # type: ignore
        elif isinstance(path, str):
            abs_path = path
            if re.match('^file:///?', abs_path):
                abs_path = re.sub('^file:///?', '/', abs_path)
            if 'u' in mode and url_validator(abs_path):  # type: ignore
                is_url = True
            elif 'f' in mode or 'd' in mode:
                abs_path = abs_path if os.path.isabs(abs_path) else os.path.join(cwd, abs_path)
        else:
            raise TypeError('Expected path to be a string or a Path object.')

        if not skip_check and is_url:
            if 'r' in mode:
                import_requests('Path with URL support')
                requests.head(abs_path).raise_for_status()  # type: ignore
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

        self.path = path
        self.abs_path = abs_path
        self.cwd = cwd
        self.mode = mode
        self.is_url = is_url  # type: bool

    def __str__(self):
        return self.abs_path

    def __repr__(self):
        return 'Path(path="'+self.path+'", abs_path="'+self.abs_path+'", cwd="'+self.cwd+'")'

    def __call__(self, absolute=True):
        """Returns the path as a string.

        Args:
            absolute (bool): If false returns the original path given, otherwise the corresponding absolute path.
        """
        return self.abs_path if absolute else self.path

    def get_content(self, mode='r'):
        """Returns the contents of the file or the response of a GET request to the URL."""
        if not self.is_url:
            with open(self.abs_path, mode) as input_file:
                return input_file.read()
        else:
            import_requests('Path with URL support')
            response = requests.get(self.abs_path)
            response.raise_for_status()
            return response.text

    @staticmethod
    def _check_mode(mode:str):
        if not isinstance(mode, str):
            raise ValueError('Expected mode to be a string.')
        if len(set(mode)-set('fdrwxcuFDRWX')) > 0:
            raise ValueError('Expected mode to only include [fdrwxcuFDRWX] flags.')
        if 'f' in mode and 'd' in mode:
            raise ValueError('Both modes "f" and "d" not possible.')
        if 'u' in mode:
            import_url_validator('Path with URL support')
            if 'd' in mode:
                raise ValueError('Both modes "d" and "u" not possible.')


def usage_and_exit_error_handler(self, message):
    """Error handler to get the same behavior as in argparse.

    Args:
        self (ArgumentParser): The ArgumentParser object.
        message (str): The message describing the error being handled.
    """
    self.print_usage(sys.stderr)
    args = {'prog': self.prog, 'message': message}
    sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
    sys.exit(2)


@contextmanager
def _suppress_stderr():
    """A context manager that redirects stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            yield None


def _find_action(parser, dest):
    """Finds an action in a parser given its dest.

    Args:
        parser (ArgumentParser): A parser where to search.
        dest (str): The dest string to search with.

    Returns:
        Action or None: The action if found, otherwise None.
    """
    for action in parser._actions:
        if action.dest == dest:
            return action
        elif isinstance(action, ActionParser) and dest.startswith(action.dest+'.'):
            return _find_action(action._parser, dest)
        elif isinstance(action, ActionSubCommands) and dest in action._name_parser_map:
            return action
    return None


def _set_inner_parser_prefix(parser, prefix, action):
    """Sets the value of env_prefix to an ActionParser and all sub ActionParsers it contains.

    Args:
        parser (ArgumentParser): The parser to which the action belongs.
        action (ActionParser): The action to set its env_prefix.
    """
    if not isinstance(action, ActionParser):
        raise ValueError('Expected action to be an ActionParser.')
    action._parser.env_prefix = parser.env_prefix
    action._parser.default_env = parser.default_env
    option_string_actions = {}
    for key, val in action._parser._option_string_actions.items():
        option_string_actions[re.sub('^--', '--'+prefix+'.', key)] = val
    action._parser._option_string_actions = option_string_actions
    for subaction in action._parser._actions:
        if isinstance(subaction, ActionYesNo):
            subaction._add_dest_prefix(prefix)
        else:
            subaction.dest = prefix+'.'+subaction.dest
            for n in range(len(subaction.option_strings)):
                subaction.option_strings[n] = re.sub('^--', '--'+prefix+'.', subaction.option_strings[n])
        if isinstance(subaction, ActionParser):
            _set_inner_parser_prefix(action._parser, prefix, subaction)


def _get_env_var(parser, action) -> str:
    """Returns the environment variable for a given parser and action."""
    env_var = (parser._env_prefix+'_' if parser._env_prefix else '') + action.dest
    env_var = env_var.replace('.', '__').upper()
    return env_var


def _is_action_value_list(action:Action):
    """Checks whether an action produces a list value.

    Args:
        action (Action): An argparse action to check.

    Returns:
        bool: True if produces list otherwise False.
    """
    if action.nargs in {'*', '+'} or isinstance(action.nargs, int):
        return True
    return False


def _check_unknown_kwargs(kwargs:Dict[str, Any], keys:Set[str]):
    """Checks whether a kwargs dict has unexpected keys.

    Args:
        kwargs (dict): The keyword arguments dict to check.
        keys (set): The expected keys.

    Raises:
        ValueError: If an unexpected keyword argument is found.
    """
    if len(set(kwargs.keys())-keys) > 0:
        raise ValueError('Unexpected keyword arguments: '+', '.join(set(kwargs.keys())-keys)+'.')
