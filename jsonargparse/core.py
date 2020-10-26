import os
import re
import sys
import stat
import glob
import json
import yaml
import enum
import inspect
import logging
import operator
import argparse
import traceback
from copy import deepcopy
from contextlib import contextmanager, redirect_stderr
from typing import Any, Tuple, List, Dict, Set, Union
from argparse import (Action, Namespace, OPTIONAL, REMAINDER, SUPPRESS, PARSER, ONE_OR_MORE, ZERO_OR_MORE,
                      ArgumentError, _UNRECOGNIZED_ARGS_ATTR)

from .optionals import (url_support, docstring_parser_support, _import_docstring_parse, _import_jsonnet,
                        set_url_support, get_config_read_mode)
from .actions import (ActionConfigFile, ActionParser, ActionSubCommands, ActionYesNo, ActionEnum, ActionPath,
                      ActionPathList, ActionOperators, _ActionPrintConfig, _find_action, _is_action_value_list,
                      _set_inner_parser_prefix)
from .jsonschema import ActionJsonSchema
from .jsonnet import ActionJsonnet, ActionJsonnetExtVars
from .util import (ParserError, _flat_namespace_to_dict, _dict_to_flat_namespace, dict_to_namespace, namespace_to_dict,
                   strip_meta, meta_keys, Path, LoggerProperty, null_logger, _get_env_var, _check_unknown_kwargs,
                   _suppress_stderr, usage_and_exit_error_handler)


class DefaultHelpFormatter(argparse.HelpFormatter):
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

    def _get_help_string(self, action):
        help = ''
        if hasattr(action, '_required') and action._required:
            help = 'required'
        if '%(default)' not in action.help and action.default is not SUPPRESS:
            if action.option_strings or action.nargs in {OPTIONAL, ZERO_OR_MORE}:
                help += (', ' if help else '') + 'default: %(default)s'
        return action.help + (' ('+help+')' if help else '')

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

    def _get_default_metavar_for_optional(self, action):
        return action.dest.rsplit('.')[-1].upper()


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
        parser = self.parser if hasattr(self, 'parser') else self
        if action.required:
            parser.required_args.add(action.dest)
            action._required = True
            action.required = False
        if isinstance(action, ActionConfigFile) and parser.formatter_class == DefaultHelpFormatter:
            setattr(parser.formatter_class, '_conf_file', True)
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
                 parse_as_dict=False,
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
        self._parse_as_dict = parse_as_dict
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.parser_mode = parser_mode
        self.logger = logger
        self.error_handler = error_handler
        self.add_argument('--print-config', action=_ActionPrintConfig)
        if version is not None:
            self.add_argument('--version', action='version', version='%(prog)s '+version)
        if parser_mode not in {'yaml', 'jsonnet'}:
            raise ValueError('The only accepted values for parser_mode are {"yaml", "jsonnet"}.')
        if parser_mode == 'jsonnet':
            _import_jsonnet('parser_mode=jsonnet')


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
            namespace, args = super()._parse_known_args(args, Namespace())
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
            argparse.Namespace: An object with all parsed values as nested attributes.

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
            argparse.Namespace or dict: An object with all parsed values as nested attributes.

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
            argparse.Namespace or dict: An object with all parsed values as attributes.

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

        if self._parse_as_dict:
            return namespace_to_dict(cfg_ns)
        return cfg_ns


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
            argparse.Namespace or dict: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if env is None and self._default_env:
            env = True

        try:
            cfg = vars(_dict_to_flat_namespace(cfg_obj))
            self._apply_actions(cfg, self._actions)

            ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults)

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


    def dump(self, cfg:Union[Namespace, dict], format:str='parser_mode', skip_none:bool=True, skip_check:bool=False) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg (argparse.Namespace or dict): The configuration object to dump.
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
            cfg (argparse.Namespace or dict): The configuration object to save.
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
                  _skip_logging:bool=False, _skip_check:bool=False) -> Union[Namespace, Dict[str, Any]]:
        """Parses environment variables.

        Args:
            env (dict[str, str]): The environment object to use, if None `os.environ` is used.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.
            with_meta (bool): Whether to include metadata in config object.

        Returns:
            argparse.Namespace or dict: An object with all parsed values as attributes.

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
            argparse.Namespace: An object with all default values as attributes.

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


    def check_config(self, cfg:Union[Namespace, dict], skip_none:bool=True, branch=None):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg (argparse.Namespace or dict): The configuration object to check.
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
            cfg (argparse.Namespace or dict): The configuration object to strip.

        Returns:
            argparse.Namespace: The stripped configuration object.
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
            cfg (argparse.Namespace or dict): The configuration object.

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
    def merge_config(cfg_from:Namespace, cfg_to:Namespace) -> Namespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (argparse.Namespace): The configuration from which to merge.
            cfg_to (argparse.Namespace): The configuration into which to merge.

        Returns:
            argparse.Namespace: The merged configuration.
        """
        return dict_to_namespace(ArgumentParser._merge_config(cfg_from, cfg_to))


    @staticmethod
    def _merge_config(cfg_from:Union[Namespace, Dict[str, Any]], cfg_to:Union[Namespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (argparse.Namespace or dict): The configuration from which to merge.
            cfg_to (argparse.Namespace or dict): The configuration into which to merge.

        Returns:
            dict: The merged configuration.
        """
        def merge_values(cfg_from, cfg_to):
            for k, v in cfg_from.items():
                if v is None:
                    continue
                if k not in cfg_to or \
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
