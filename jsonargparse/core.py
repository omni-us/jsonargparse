import os
import re
import sys
import glob
import json
import yaml
import inspect
import argparse
import traceback
from enum import Enum
from copy import deepcopy
from contextlib import redirect_stderr
from typing import Any, List, Dict, Set, Union, Optional
from argparse import (Action, Namespace, OPTIONAL, REMAINDER, SUPPRESS, PARSER, ONE_OR_MORE, ZERO_OR_MORE,
                      ArgumentError, _UNRECOGNIZED_ARGS_ATTR)

from .formatters import DefaultHelpFormatter
from .signatures import SignatureArguments
from .optionals import _import_jsonnet, argcomplete_support, _import_argcomplete, set_url_support, get_config_read_mode
from .actions import (ActionConfigFile, ActionParser, _ActionSubCommands, ActionYesNo, ActionEnum, ActionPath,
                      ActionPathList, ActionOperators, _ActionPrintConfig, _find_action, _is_action_value_list)
from .jsonschema import ActionJsonSchema
from .jsonnet import ActionJsonnet, ActionJsonnetExtVars
from .util import (ParserError, _flat_namespace_to_dict, _dict_to_flat_namespace, dict_to_namespace, namespace_to_dict,
                   strip_meta, meta_keys, Path, LoggerProperty, null_logger, _get_env_var, _issubclass,
                   _suppress_stderr, usage_and_exit_error_handler)


class _ActionsContainer(argparse._ActionsContainer):
    """Extension of argparse._ActionsContainer to support additional functionalities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('action', 'parsers', _ActionSubCommands)


    def add_argument(self, *args, **kwargs):
        """Adds an argument to the parser or argument group.

        All the arguments from `argparse.ArgumentParser.add_argument
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
        are supported.
        """
        if 'type' in kwargs:
            if kwargs['type'] == bool:
                if 'nargs' in kwargs:
                    raise ValueError('Argument with type=bool does not support nargs.')
                kwargs['nargs'] = 1
                kwargs['action'] = ActionYesNo(no_prefix=None)
            elif _issubclass(kwargs['type'], Enum):
                kwargs['action'] = ActionEnum(enum=kwargs['type'])
            elif hasattr(kwargs['type'], '__origin__') and kwargs['type'].__origin__ in {Union, Dict, dict, List, list}:
                kwargs['action'] = ActionJsonSchema(annotation=kwargs.pop('type'), enable_path=False)
        action = super().add_argument(*args, **kwargs)
        for key in meta_keys:
            if key in action.dest:
                raise ValueError('Argument with destination name "'+key+'" not allowed.')
        parser = self.parser if hasattr(self, 'parser') else self
        if action.required:
            parser.required_args.add(action.dest)
            action._required = True
            action.required = False
        if isinstance(action, ActionParser):
            ActionParser._set_inner_parser_prefix(self, action.dest, action)
        return action


class _ArgumentGroup(_ActionsContainer, argparse._ArgumentGroup):
    """Extension of argparse._ArgumentGroup to support additional functionalities."""
    parser = None  # type: Union[ArgumentParser, None]


class ArgumentParser(SignatureArguments, _ActionsContainer, argparse.ArgumentParser, LoggerProperty):
    """Parser for command line, yaml/jsonnet files and environment variables."""

    groups = None  # type: Dict[str, argparse._ArgumentGroup]


    def __init__(self,
                 *args,
                 env_prefix: Optional[str] = None,
                 error_handler = usage_and_exit_error_handler,
                 formatter_class = DefaultHelpFormatter,
                 logger = None,
                 version: Optional[str] = None,
                 print_config: Optional[str] = '--print-config',
                 parser_mode: str = 'yaml',
                 parse_as_dict: bool = False,
                 default_config_files: Optional[List[str]] = None,
                 default_env: bool = False,
                 default_meta: bool = True,
                 **kwargs):
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            env_prefix (str): Prefix for environment variables.
            error_handler (Callable): Handler for parsing errors, set to None to simply raise exception.
            formatter_class (argparse.HelpFormatter): Class for printing help messages.
            logger: Configures the logger, see :class:`.LoggerProperty`.
            version (str or None): Program version string to add --version argument.
            print_config (str or None): Add this as argument to print config, set None to disable.
            parser_mode (str): Mode for parsing configuration files, either "yaml" or "jsonnet".
            parse_as_dict (bool): Whether to parse as dict instead of Namespace.
            default_config_files (list[str]): List of strings defining default config file locations. For example: :code:`['~/.config/myapp/*.yaml']`.
            default_env (bool): Set the default value on whether to parse environment variables.
            default_meta (bool): Set the default value on whether to include metadata in config objects.
        """
        kwargs['formatter_class'] = formatter_class
        super().__init__(*args, **kwargs)
        if self.groups is None:
            self.groups = {}
        if default_config_files is None:
            default_config_files = []
        self.required_args = set()  # type: Set[str]
        self._stderr = sys.stderr
        self._default_config_files = default_config_files
        self._parse_as_dict = parse_as_dict
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.parser_mode = parser_mode
        self.logger = logger
        self.error_handler = error_handler
        if print_config is not None:
            self.add_argument('--print-config', action=_ActionPrintConfig)
        if version is not None:
            self.add_argument('--version', action='version', version='%(prog)s '+version)
        if parser_mode not in {'yaml', 'jsonnet'}:
            raise ValueError('The only accepted values for parser_mode are {"yaml", "jsonnet"}.')
        if parser_mode == 'jsonnet':
            _import_jsonnet('parser_mode=jsonnet')


    ## Parsing methods ##

    def parse_known_args(self, args=None, namespace=None):
        """Raises NotImplementedError to dissuade its use, since typos in configs would go unnoticed."""
        caller = inspect.getmodule(inspect.stack()[1][0]).__package__
        if caller not in {'jsonargparse', 'argcomplete'}:
            raise NotImplementedError('parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed.')
        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)

        if namespace is None:
            namespace = Namespace()

        if caller == 'argcomplete':
            namespace = _dict_to_flat_namespace(self._merge_config(self.get_defaults(nested=False), namespace))

        try:
            namespace, args = self._parse_known_args(args, namespace)
            if len(args) > 0:
                for action in self._actions:
                    if isinstance(action, ActionParser):
                        ns, args = action._parser.parse_known_args(args)
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
            Namespace or Dict: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if argcomplete_support:
            argcomplete = _import_argcomplete('parse_args')
            argcomplete.autocomplete(self)

        if env is None and self._default_env:
            env = True

        try:
            with _suppress_stderr():
                cfg, unk = self.parse_known_args(args=args)
                if unk:
                    self.error('Unrecognized arguments: %s' % ' '.join(unk))

            _ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

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

            if hasattr(self, '_print_config') and self._print_config:  # type: ignore
                sys.stdout.write(self.dump(cfg_ns, skip_none=False, skip_check=True))
                self.exit()

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

        if self._parse_as_dict:
            return namespace_to_dict(cfg_ns)
        return cfg_ns


    def parse_object(self, cfg_obj:dict, cfg_base=None, env:bool=None, defaults:bool=True, nested:bool=True,
                     with_meta:bool=None, _skip_check:bool=False) -> Union[Namespace, Dict[str, Any]]:
        """Parses configuration given as an object.

        Args:
            cfg_obj (dict): The configuration object.
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            Namespace or dict: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            cfg = vars(_dict_to_flat_namespace(cfg_obj))
            self._apply_actions(cfg, self._actions)

            _ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

            if nested:
                cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

            if cfg_base is not None:
                if isinstance(cfg_base, Namespace):
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

        if self._parse_as_dict:
            return namespace_to_dict(cfg_ns)
        return cfg_ns


    def parse_env(self, env:Dict[str, str]=None, defaults:bool=True, nested:bool=True, with_meta:bool=None,
                  _skip_logging:bool=False, _skip_check:bool=False) -> Union[Namespace, Dict[str, Any]]:
        """Parses environment variables.

        Args:
            env (dict[str, str]): The environment object to use, if None `os.environ` is used.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            Namespace or dict: An object with all parsed values as attributes.

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
                if env_var in env and isinstance(action, _ActionSubCommands):
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
                cfg = _flat_namespace_to_dict(Namespace(**cfg))

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

        if self._parse_as_dict:
            return namespace_to_dict(cfg_ns)
        return cfg_ns


    def parse_path(self, cfg_path:str, ext_vars:dict={}, env:bool=None, defaults:bool=True, nested:bool=True,
                   with_meta:bool=None, _skip_check:bool=False, _base=None) -> Union[Namespace, Dict[str, Any]]:
        """Parses a configuration file (yaml or jsonnet) given its path.

        Args:
            cfg_path (str or Path): Path to the configuration file to parse.
            ext_vars (dict): Optional external variables used for parsing jsonnet.
            env (bool or None): Whether to merge with the parsed environment. None means use the ArgumentParser's default.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            Namespace or dict: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        fpath = Path(cfg_path, mode=get_config_read_mode())
        if not fpath.is_url:
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(fpath(absolute=False), os.pardir)))
        try:
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(cfg_str, cfg_path, ext_vars, env, defaults, nested, with_meta=with_meta,
                                           _skip_logging=True, _skip_check=_skip_check, _base=_base)
            if with_meta or (with_meta is None and self._default_meta):
                if self._parse_as_dict:
                    parsed_cfg['__path__'] = fpath
                else:
                    parsed_cfg.__path__ = fpath  # type: ignore
        finally:
            if not fpath.is_url:
                os.chdir(cwd)

        self._logger.info('Parsed %s from path: %s', self.parser_mode, cfg_path)

        return parsed_cfg


    def parse_string(self, cfg_str:str, cfg_path:str='', ext_vars:dict={}, env:bool=None, defaults:bool=True, nested:bool=True,
                     with_meta:bool=None, _skip_logging:bool=False, _skip_check:bool=False, _base=None) -> Union[Namespace, Dict[str, Any]]:
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
            Namespace or dict: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            cfg = self._load_cfg(cfg_str, cfg_path, ext_vars, _base)

            _ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

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

        if self._parse_as_dict:
            return namespace_to_dict(cfg_ns)
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
            _jsonnet = _import_jsonnet('_load_cfg')
            cfg_str = _jsonnet.evaluate_snippet(cfg_path, cfg_str, ext_vars=ext_vars, ext_codes=ext_codes)  # type: ignore
        try:
            cfg = yaml.safe_load(cfg_str)
        except Exception as ex:
            raise type(ex)('Problems parsing config :: '+str(ex))
        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))
        if base is not None:
            cfg = {base+'.'+k: v for k, v in cfg.items()}

        self._apply_actions(cfg, self._actions)

        return cfg


    ## Methods for adding to the parser ##

    def add_argument_group(self, *args, name:str=None, **kwargs):
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported. Additionally it accepts:

        Args:
            name (str): Name of the group. If set the group object will be included in the parser.groups dict.

        Returns:
            The group object.

        Raises:
            ValueError: If group with the same name already exists.
        """
        if name is not None and name in self.groups:
            raise ValueError('Group with name '+name+' already exists.')
        group = _ArgumentGroup(self, *args, **kwargs)
        group.parser = self
        self._action_groups.append(group)
        if name is not None:
            self.groups[name] = group
        return group


    def add_subparsers(self, **kwargs):
        """Raises a NotImplementedError since jsonargparse uses add_subcommands."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommands method.')


    def add_subcommands(self, required=True, dest='subcommand', **kwargs):
        """Adds sub-command parsers to the ArgumentParser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        a required argument is accepted, dest by default is 'subcommand' and the
        values of the sub-command are stored in a nested namespace using the
        sub-command's name as base key.
        """
        if 'required' in kwargs:
            required = kwargs.pop('required')
        subcommands = super().add_subparsers(dest=dest, **kwargs)
        subcommands.required = required
        _find_action(self, dest)._env_prefix = self.env_prefix
        return subcommands


    ## Methods for serializing config objects ##

    def dump(self, cfg:Union[Namespace, dict], format:str='parser_mode', skip_none:bool=True, skip_check:bool=False) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg (Namespace or dict): The configuration object to dump.
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
                if action.help == SUPPRESS or \
                   isinstance(action, ActionConfigFile) or \
                   (skip_none and action.dest in cfg and cfg[action.dest] is None):
                    del cfg[action.dest]
                elif isinstance(action, ActionEnum):
                    cfg[action.dest] = cfg[action.dest].name
                elif isinstance(action, ActionPath):
                    if cfg[action.dest] is not None:
                        if isinstance(cfg[action.dest], list):
                            cfg[action.dest] = [p(absolute=False) for p in cfg[action.dest]]
                        else:
                            cfg[action.dest] = cfg[action.dest](absolute=False)
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


    def save(self, cfg:Union[Namespace, dict], path:str, format:str='parser_mode', skip_none:bool=True,
             skip_check:bool=False, overwrite:bool=False, multifile:bool=True, branch=None) -> None:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg (Namespace or dict): The configuration object to save.
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


    ## Methods related to defaults ##

    def set_defaults(self, *args, **kwargs):
        """Sets default values from dictionary or keyword arguments.

        Args:
            *args (dict): Dictionary defining the default values to set.
            **kwargs: Sets default values based on keyword arguments.

        Raises:
            KeyError: If key not defined in the parser.
        """
        if len(args) > 0:
            for n in range(len(args)):
                self._defaults.update(args[n])

                for dest in args[n].keys():
                    action = _find_action(self, dest)
                    if action is None:
                        raise KeyError('No action for destination key "'+dest+'" to set its default.')
                    action.default = args[n][dest]
        if kwargs:
            self.set_defaults(kwargs)


    def get_default(self, dest):
        """Gets a single default value for the given destination key.

        Args:
            dest (str): Destination key from which to get the default.

        Raises:
            KeyError: If key not defined in the parser.
        """
        action = _find_action(self, dest)
        if action is None:
            raise KeyError('No action for destination key "'+dest+'" to get its default.')
        if action.default is not None:
            return action.default
        return self._defaults.get(dest, None)


    def get_defaults(self, nested:bool=True) -> Namespace:
        """Returns a namespace with all default values.

        Args:
            nested (bool): Whether the namespace should be nested.

        Returns:
            Namespace: An object with all default values as attributes.

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
                default_config = Path(default_config_files[0], mode=get_config_read_mode()).get_content()
                cfg_file = self._load_cfg(default_config)
                cfg = self._merge_config(cfg_file, cfg)
                self._logger.info('Parsed configuration from default path: %s', default_config_files[0])

            if nested:
                cfg = _flat_namespace_to_dict(Namespace(**cfg))

        except TypeError as ex:
            self.error(str(ex))

        return dict_to_namespace(cfg)


    ## Other methods ##

    def error(self, message):
        """Logs error message if a logger is set, calls the error handler and raises a ParserError."""
        self._logger.error(message)
        if self._error_handler is not None:
            with redirect_stderr(self._stderr):
                self._error_handler(self, message)
        raise ParserError(message)


    def check_config(self, cfg:Union[Namespace, dict], skip_none:bool=True, branch=None):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg (Namespace or dict): The configuration object to check.
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
                    if isinstance(action, _ActionSubCommands) and kbase != action.dest:
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
            cfg (Namespace or dict): The configuration object to strip.

        Returns:
            Namespace: The stripped configuration object.
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
            cfg (Namespace or dict): The configuration object.

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


    def _apply_actions(self, cfg, actions):
        """Runs _check_value_key on actions present in flat config dict."""
        for action in actions:
            if isinstance(action, ActionParser):
                self._apply_actions(cfg, action._parser._actions)
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


    @staticmethod
    def merge_config(cfg_from:Namespace, cfg_to:Namespace) -> Namespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (Namespace): The configuration from which to merge.
            cfg_to (Namespace): The configuration into which to merge.

        Returns:
            Namespace: The merged configuration.
        """
        return dict_to_namespace(ArgumentParser._merge_config(cfg_from, cfg_to))


    @staticmethod
    def _merge_config(cfg_from:Union[Namespace, Dict[str, Any]], cfg_to:Union[Namespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (Namespace or dict): The configuration from which to merge.
            cfg_to (Namespace or dict): The configuration into which to merge.

        Returns:
            dict: The merged configuration.
        """
        def merge_values(cfg_from, cfg_to):
            for k, v in cfg_from.items():
                if v is None or \
                   k not in cfg_to or \
                   not isinstance(v, dict) or \
                   (isinstance(v, dict) and not isinstance(cfg_to[k], dict)):
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
            if isinstance(action, _ActionSubCommands):
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


    ## Properties ##

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
        if issubclass(self.formatter_class, DefaultHelpFormatter):
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
        if issubclass(self.formatter_class, DefaultHelpFormatter):
            setattr(self.formatter_class, '_env_prefix', env_prefix)
