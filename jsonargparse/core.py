"""Extensions of core argparse classes."""

import argparse
import glob
import inspect
import json
import logging
import os
import re
import sys
import yaml
from argparse import ArgumentError, Action, SUPPRESS
from contextlib import redirect_stderr
from copy import deepcopy
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence, Set, Tuple, Type, Union
from unittest.mock import patch

from .formatters import DefaultHelpFormatter, empty_help
from .jsonnet import ActionJsonnet
from .jsonschema import ActionJsonSchema
from .namespace import is_meta_key, Namespace, split_key, split_key_leaf, strip_meta
from .signatures import is_pure_dataclass, SignatureArguments
from .typehints import ActionTypeHint, LazyInitBaseClass
from .actions import (
    ActionParser,
    ActionConfigFile,
    _ActionSubCommands,
    _ActionPrintConfig,
    _ActionConfigLoad,
    _ActionLink,
    _is_branch_key,
    _find_action,
    _find_parent_action,
    _find_parent_action_and_subcommand,
    _is_action_value_list,
    filter_default_actions,
)
from .optionals import (
    argcomplete_support,
    dump_preserve_order_support,
    fsspec_support,
    FilesCompleterMethod,
    get_config_read_mode,
    import_jsonnet,
    import_argcomplete,
    import_fsspec,
    TypeCastCompleterMethod,
)
from .util import (
    yamlParserError,
    yamlScannerError,
    ParserError,
    usage_and_exit_error_handler,
    change_to_path_dir,
    Path,
    LoggerProperty,
    _check_valid_dump_format,
    _get_env_var,
    _suppress_stderr,
)


__all__ = ['ArgumentParser']


default_dump_yaml_kwargs = {
    'default_flow_style': False,
    'allow_unicode': True,
    'sort_keys': False if dump_preserve_order_support else True,
}

default_dump_json_kwargs = {
    'ensure_ascii': False,
    'sort_keys': False if dump_preserve_order_support else True,
}


class _ActionsContainer(SignatureArguments, argparse._ActionsContainer, LoggerProperty):
    """Extension of argparse._ActionsContainer to support additional functionalities."""

    _action_groups: Sequence['_ArgumentGroup']  # type: ignore


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('action', 'parsers', _ActionSubCommands)


    def add_argument(self, *args, enable_path:bool=False, **kwargs):
        """Adds an argument to the parser or argument group.

        All the arguments from `argparse.ArgumentParser.add_argument
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
        are supported. Additionally it accepts:

        Args:
            enable_path: Whether to try parsing path/subconfig when argument is a complex type.
        """
        parser = self.parser if hasattr(self, 'parser') else self  # type: ignore
        if 'action' in kwargs and isinstance(kwargs['action'], ActionParser):
            if kwargs['action']._parser == parser:
                raise ValueError('Parser cannot be added as a subparser of itself.')
            return ActionParser._move_parser_actions(parser, args, kwargs)
        if 'type' in kwargs:
            if is_pure_dataclass(kwargs['type']):
                theclass = kwargs.pop('type')
                nested_key = re.sub('^--', '', args[0])
                super().add_dataclass_arguments(theclass, nested_key, **kwargs)
                return _find_action(parser, nested_key)
            if ActionTypeHint.is_supported_typehint(kwargs['type']):
                if 'action' in kwargs:
                    raise ValueError('Type hint as type does not allow providing an action.')
                kwargs['action'] = ActionTypeHint(typehint=kwargs.pop('type'), enable_path=enable_path)
        action = super().add_argument(*args, **kwargs)
        if not hasattr(action, 'completer') and action.type is not None:
            completer_method = FilesCompleterMethod if isinstance(action.type, Path) else TypeCastCompleterMethod
            action_class = action.__class__
            action.__class__ = action_class.__class__(  # type: ignore
                action_class.__name__ + 'WithCompleter',
                (action_class, completer_method),
                {}
            )
        if isinstance(action, ActionConfigFile) and getattr(self, '_print_config', None) is not None:
            self.add_argument(self._print_config, action=_ActionPrintConfig)  # type: ignore
        if is_meta_key(action.dest):
            raise ValueError(f'Argument with destination name "{action.dest}" not allowed.')
        if action.help is None:
            action.help = empty_help
        if action.required:
            parser.required_args.add(action.dest)
            action._required = True  # type: ignore
            action.required = False
        return action


    def add_argument_group(self, *args, name:str=None, **kwargs):
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported. Additionally it accepts:

        Args:
            name: Name of the group. If set the group object will be included in the parser.groups dict.

        Returns:
            The group object.

        Raises:
            ValueError: If group with the same name already exists.
        """
        parser = self.parser if hasattr(self, 'parser') else self  # type: ignore
        if name is not None and name in parser.groups:
            raise ValueError(f'Group with name {name} already exists.')
        group = _ArgumentGroup(parser, *args, **kwargs)
        group.parser = parser
        parser._action_groups.append(group)
        if name is not None:
            parser.groups[name] = group
        return group


class _ArgumentGroup(_ActionsContainer, argparse._ArgumentGroup):
    """Extension of argparse._ArgumentGroup to support additional functionalities."""
    dest: Optional[str] = None
    parser: Optional['ArgumentParser'] = None


class ArgumentParser(_ActionsContainer, argparse.ArgumentParser):
    """Parser for command line, yaml/jsonnet files and environment variables."""

    formatter_class: Type[DefaultHelpFormatter]  # type: ignore
    groups: Optional[Dict[str, '_ArgumentGroup']] = None


    def __init__(
        self,
        *args,
        env_prefix: Optional[str] = None,
        error_handler: Optional[Callable[['ArgumentParser', str], None]] = usage_and_exit_error_handler,
        formatter_class: Type[DefaultHelpFormatter] = DefaultHelpFormatter,
        logger: Optional[Union[bool, Dict[str, str], logging.Logger]] = None,
        version: Optional[str] = None,
        print_config: Optional[str] = '--print_config',
        parser_mode: str = 'yaml',
        default_config_files: Optional[List[str]] = None,
        default_env: bool = False,
        default_meta: bool = True,
        **kwargs
    ):
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            env_prefix: Prefix for environment variables.
            error_handler: Handler for parsing errors, set to None to simply raise exception.
            formatter_class: Class for printing help messages.
            logger: Configures the logger, see :class:`.LoggerProperty`.
            version: Program version string to add --version argument.
            print_config: Add this as argument to print config, set None to disable.
            parser_mode: Mode for parsing configuration files, either "yaml" or "jsonnet".
            default_config_files: Default config file locations, e.g. :code:`['~/.config/myapp/*.yaml']`.
            default_env: Set the default value on whether to parse environment variables.
            default_meta: Set the default value on whether to include metadata in config objects.
        """
        class FormatterClass(formatter_class):  # type: ignore
            _parser = self

        super().__init__(*args, **kwargs)
        if self.groups is None:
            self.groups = {}
        self.required_args: Set[str] = set()
        self.save_path_content: Set[str] = set()
        self._stderr = sys.stderr
        self.default_config_files = default_config_files
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.formatter_class = FormatterClass
        self.parser_mode = parser_mode
        self.logger = logger
        self.error_handler = error_handler
        self._print_config = print_config
        if version is not None:
            self.add_argument('--version', action='version', version='%(prog)s '+version, help='Print version and exit.')
        if parser_mode not in {'yaml', 'jsonnet'}:
            raise ValueError('The only accepted values for parser_mode are {"yaml", "jsonnet"}.')
        if parser_mode == 'jsonnet':
            import_jsonnet('parser_mode=jsonnet')
        self.dump_yaml_kwargs = dict(default_dump_yaml_kwargs)
        self.dump_json_kwargs = dict(default_dump_json_kwargs)


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
            if not all(isinstance(a, str) for a in args):
                self.error(f'All arguments are expected to be strings: {args}')

        if namespace is None:
            namespace = Namespace()

        if caller == 'argcomplete':
            namespace.__class__ = Namespace
            namespace = self.merge_config(self.get_defaults(skip_check=True), namespace).as_flat()

        try:
            with patch('argparse.Namespace', Namespace), ActionTypeHint.subclass_arg_context(self):
                namespace, args = self._parse_known_args(args, namespace)
        except (ArgumentError, ParserError) as ex:
            self.error(str(ex), ex)

        return namespace, args


    def _parse_optional(self, arg_string):
        subclass_arg = ActionTypeHint.parse_subclass_arg(arg_string)
        if subclass_arg:
            return subclass_arg
        if arg_string == self._print_config:
            arg_string += '='
        return super()._parse_optional(arg_string)


    def _parse_common(
        self,
        cfg: Namespace,
        env: Optional[bool],
        defaults: bool,
        with_meta: Optional[bool],
        skip_check: bool,
        skip_required: bool = False,
        skip_subcommands: bool = False,
        fail_no_subcommand: bool = True,
        cfg_base: Optional[Namespace] = None,
        log_message: Optional[str] = None,
    ) -> Namespace:
        """Common parsing code used by other parse methods.

        Args:
            cfg: The configuration object.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.
            skip_check: Whether to skip check if configuration is valid.
            skip_required: Whether to skip check of required arguments.
            skip_subcommands: Whether to skip subcommand processing.
            fail_no_subcommand: Whether to fail if no subcommand given.
            cfg_base: A base configuration object.
            log_message: Message to log at INFO level after parsing.

        Returns:
            A config object with all parsed values.
        """
        if env is None and self._default_env:
            env = True

        if not skip_subcommands:
            _ActionSubCommands.handle_subcommands(self, cfg, env=env, defaults=defaults, fail_no_subcommand=fail_no_subcommand)

        if cfg_base is not None:
            cfg = self.merge_config(cfg, cfg_base)

        if env:
            cfg_env = self.parse_env(defaults=defaults, _skip_check=True, _skip_subcommands=True)
            cfg = self.merge_config(cfg, cfg_env)

        elif defaults:
            cfg = self.merge_config(cfg, self.get_defaults(skip_check=True))

        if not (with_meta or (with_meta is None and self._default_meta)):
            cfg = strip_meta(cfg)

        _ActionPrintConfig.print_config_if_requested(self, cfg)

        _ActionLink.apply_parsing_links(self, cfg)

        if not skip_check:
            self.check_config(cfg, skip_required=skip_required)

        if log_message is not None:
            self._logger.info(log_message)

        return cfg


    def parse_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Namespace = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        _skip_check: bool = False,
    ) -> Namespace:
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            args: List of arguments to parse or None to use sys.argv.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        if argcomplete_support:
            argcomplete = import_argcomplete('parse_args')
            argcomplete.autocomplete(self)

        try:
            with _suppress_stderr():
                cfg, unk = self.parse_known_args(args=args)
                if unk:
                    self.error(f'Unrecognized arguments: {" ".join(unk)}')

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_check=_skip_check,
                cfg_base=namespace,
                log_message='Parsed command line arguments.',
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        return parsed_cfg


    def parse_object(
        self,
        cfg_obj: Dict[str, Any],
        cfg_base: Optional[Namespace] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        _skip_check: bool = False,
        _skip_required: bool = False,
    ) -> Namespace:
        """Parses configuration given as an object.

        Args:
            cfg_obj: The configuration object.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            cfg = self._config_from_dict(cfg_obj)
            self._apply_actions(cfg)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_check=_skip_check,
                skip_required=_skip_required,
                cfg_base=cfg_base,
                log_message='Parsed object.',
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        return parsed_cfg


    def parse_env(
        self,
        env: Dict[str, str] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        _skip_check: bool = False,
        _skip_subcommands: bool = False,
    ) -> Namespace:
        """Parses environment variables.

        Args:
            env: The environment object to use, if None `os.environ` is used.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            if env is None:
                env = dict(os.environ)
            cfg = Namespace()
            actions = filter_default_actions(self._actions)
            for action in actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, ActionConfigFile):
                    ActionConfigFile.apply_config(self, cfg, action.dest, env[env_var])
            for action in actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, _ActionSubCommands):
                    env_val = env[env_var]
                    if env_val in action.choices:
                        cfg[action.dest] = subcommand = self._check_value_key(action, env_val, action.dest, cfg)
                        pcfg = action._name_parser_map[env_val].parse_env(env=env, defaults=defaults, _skip_check=True)  # type: ignore
                        for k, v in vars(pcfg).items():
                            cfg[subcommand+'.'+k] = v
            for action in actions:
                env_var = _get_env_var(self, action)
                if env_var in env and not isinstance(action, ActionConfigFile):
                    env_val = env[env_var]
                    if _is_action_value_list(action):
                        if re.match('^ *\\[.+,.+] *$', env_val):
                            try:
                                env_val = yaml.safe_load(env_val)
                            except (yamlParserError, yamlScannerError):
                                env_val = [env_val]  # type: ignore
                        else:
                            env_val = [env_val]  # type: ignore
                    cfg[action.dest] = self._check_value_key(action, env_val, action.dest, cfg)

            self._apply_actions(cfg)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=False,
                defaults=defaults,
                with_meta=with_meta,
                skip_check=_skip_check,
                skip_subcommands=_skip_subcommands,
                log_message='Parsed environment variables.',
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        return parsed_cfg


    def parse_path(
        self,
        cfg_path: str,
        ext_vars: Optional[dict] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        _skip_check: bool = False,
        _fail_no_subcommand: bool = True,
    ) -> Namespace:
        """Parses a configuration file (yaml or jsonnet) given its path.

        Args:
            cfg_path: Path to the configuration file to parse.
            ext_vars: Optional external variables used for parsing jsonnet.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        fpath = Path(cfg_path, mode=get_config_read_mode())
        with change_to_path_dir(fpath):
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(cfg_str,
                                           cfg_path,
                                           ext_vars,
                                           env,
                                           defaults,
                                           with_meta=with_meta,
                                           _skip_check=_skip_check,
                                           _fail_no_subcommand=_fail_no_subcommand)

        self._logger.info(f'Parsed {self.parser_mode} from path: {cfg_path}')

        return parsed_cfg


    def parse_string(
        self,
        cfg_str: str,
        cfg_path: str = '',
        ext_vars: Optional[dict] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        _skip_check: bool = False,
        _fail_no_subcommand: bool = True,
    ) -> Namespace:
        """Parses configuration (yaml or jsonnet) given as a string.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            nested: Whether the namespace should be nested.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            cfg = self._load_config_parser_mode(cfg_str, cfg_path, ext_vars)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_check=_skip_check,
                fail_no_subcommand=_fail_no_subcommand,
                log_message=(f'Parsed {self.parser_mode} string.'),
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        return parsed_cfg


    def _load_config_parser_mode(
        self,
        cfg_str: str,
        cfg_path: str = '',
        ext_vars: Optional[dict] = None,
    ) -> Namespace:
        """Loads a configuration string (yaml or jsonnet) into a namespace.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        if self.parser_mode == 'jsonnet':
            ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
            _jsonnet = import_jsonnet('_load_config_parser_mode')
            cfg_str = _jsonnet.evaluate_snippet(cfg_path, cfg_str, ext_vars=ext_vars, ext_codes=ext_codes)
        try:
            cfg_dict = yaml.safe_load(cfg_str)
        except (yamlParserError, yamlScannerError) as ex:
            raise TypeError(f'Problems parsing config :: {ex}') from ex

        cfg = self._config_from_dict(cfg_dict)
        self._apply_actions(cfg)

        return cfg


    def _config_from_dict(self, cfg_dict: Dict[str, Any], parent_key: str = '') -> Namespace:

        def branch_to_namespace(dic, parser, parent_key=''):
            ns = Namespace()
            for key, val in dic.items():
                full_key = parent_key + '.' + key if parent_key else key
                if isinstance(val, dict):
                    action = _find_action(parser, full_key, exclude=_ActionConfigLoad)
                    if action is None:
                        val = branch_to_namespace(val, parser, parent_key=full_key)
                    elif isinstance(action, _ActionSubCommands):
                        cfg = Namespace()
                        cfg[full_key] = Namespace()
                        _, subparser = _ActionSubCommands.get_subcommand(self, cfg, fail_no_subcommand=False)
                        val = branch_to_namespace(val, subparser)
                    elif isinstance(action, ActionTypeHint):
                        def class_branch_to_namespace(val):
                            klass, init_args, final_class = action.get_class_and_init_args(val)
                            val = Namespace(val)
                            if init_args is not None:
                                sub_add_kwargs = getattr(action, 'sub_add_kwargs', None)
                                subparser = ActionTypeHint.get_class_parser(klass, sub_add_kwargs=sub_add_kwargs)
                                init_args = branch_to_namespace(init_args, subparser)
                                if final_class:
                                    val = init_args
                                else:
                                    val['init_args'] = init_args
                            return val
                        if ActionTypeHint.is_class_typehint(action):
                            val = class_branch_to_namespace(val)
                        elif ActionTypeHint.is_mapping_class_typehint(action):
                            val = dict(val)
                            for subkey, subval in val.items():
                                val[subkey] = class_branch_to_namespace(subval)
                ns[key] = val
            return ns

        with _ActionSubCommands.not_single_subcommand():
            return branch_to_namespace(cfg_dict, self, parent_key=parent_key)


    def link_arguments(
        self,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Callable = None,
        apply_on: str = 'parse',
    ):
        """Makes an argument value be derived from the values of other arguments.

        Refer to :ref:`argument-linking` for a detailed explanation and examples-

        Args:
            source: Key(s) from which the target value is derived.
            target: Key to where the value is set.
            compute_fn: Function to compute target value from source.
            apply_on: At what point to set target value, 'parse' or 'instantiate'.

        Raises:
            ValueError: If an invalid parameter is given.
        """
        _ActionLink(self, source, target, compute_fn, apply_on)


    ## Methods for adding to the parser ##

    def add_subparsers(self, **kwargs) -> NoReturn:
        """Raises a NotImplementedError since jsonargparse uses add_subcommands."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommands method.')


    def add_subcommands(self, required: bool = True, dest: str = 'subcommand', **kwargs) -> _ActionSubCommands:
        """Adds sub-command parsers to the ArgumentParser.

        The aim is the same as `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        the difference being that dest by default is 'subcommand' and the parsed
        values of the sub-command are stored in a nested namespace using the
        sub-command's name as base key.

        Args:
            required: Whether the subcommand must be provided.
            dest: Destination key where the chosen subcommand name is stored.
            **kwargs: All options that `argparse.ArgumentParser.add_subparsers` accepts.
        """
        if 'description' not in kwargs:
            kwargs['description'] = 'For more details of each subcommand add it as argument followed by --help.'
        subcommands: _ActionSubCommands = super().add_subparsers(dest=dest, **kwargs)  # type: ignore
        if required:
            self.required_args.add(dest)
        subcommands._required = required  # type: ignore
        subcommands.required = False
        subcommands.parent_parser = self  # type: ignore
        subcommands._env_prefix = self.env_prefix
        self._subcommands_action = subcommands
        return subcommands


    ## Methods for serializing config objects ##

    def dump(
        self,
        cfg: Namespace,
        format: str = 'parser_mode',
        skip_none: bool = True,
        skip_check: bool = False,
        yaml_comments: bool = False,
    ) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg: The configuration object to dump.
            format: The output format: "yaml", "json", "json_indented" or "parser_mode".
            skip_none: Whether to exclude entries whose value is None.
            skip_check: Whether to skip parser checking.
            yaml_comments: Whether to add help content as comments.

        Returns:
            The configuration in yaml or json format.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        _check_valid_dump_format(format)

        cfg = deepcopy(cfg)
        cfg = strip_meta(cfg)
        _ActionLink.strip_link_target_keys(self, cfg)

        if not skip_check:
            self.check_config(cfg)

        def cleanup_actions(cfg, actions, prefix=''):
            for action in filter_default_actions(actions):
                action_dest = prefix + action.dest
                if (action.help == SUPPRESS and not isinstance(action, _ActionConfigLoad)) or \
                   isinstance(action, ActionConfigFile) or \
                   (skip_none and action_dest in cfg and cfg[action_dest] is None):
                    cfg.pop(action_dest, None)
                elif isinstance(action, _ActionSubCommands):
                    cfg.pop(action_dest, None)
                    for key, subparser in action.choices.items():
                        cleanup_actions(cfg, subparser._actions, prefix=prefix+key+'.')
                elif isinstance(action, ActionTypeHint):
                    value = cfg.get(action_dest)
                    if value is not None:
                        value = action.serialize(value)
                        cfg.update(value, action_dest)

        cleanup_actions(cfg, self._actions)

        if format == 'parser_mode':
            format = 'yaml' if self.parser_mode == 'yaml' else 'json_indented'
        if format == 'yaml':
            dump = yaml.safe_dump(cfg.as_dict(), **self.dump_yaml_kwargs)  # type: ignore
            if yaml_comments:
                formatter = self.formatter_class(self.prog)
                dump = formatter.add_yaml_comments(dump)
            return dump
        elif format == 'json_indented':
            return json.dumps(cfg.as_dict(), indent=2, **self.dump_json_kwargs)+'\n'  # type: ignore
        else:
            return json.dumps(cfg.as_dict(), separators=(',', ':'), **self.dump_json_kwargs)  # type: ignore


    def save(
        self,
        cfg: Namespace,
        path: str,
        format: str = 'parser_mode',
        skip_none: bool = True,
        skip_check: bool = False,
        overwrite: bool = False,
        multifile: bool = True,
        branch: str = None,
    ) -> None:
        """Writes to file(s) the yaml or json for the given configuration object.

        Args:
            cfg: The configuration object to save.
            path: Path to the location where to save config.
            format: The output format: "yaml", "json", "json_indented" or "parser_mode".
            skip_none: Whether to exclude entries whose value is None.
            skip_check: Whether to skip parser checking.
            overwrite: Whether to overwrite existing files.
            multifile: Whether to save multiple config files by using the __path__ metas.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        _check_valid_dump_format(format)

        def check_overwrite(path):
            if not overwrite and os.path.isfile(path()):
                raise ValueError('Refusing to overwrite existing file: '+path())

        dump_kwargs = {'format': format, 'skip_none': skip_none, 'skip_check': skip_check}

        if fsspec_support:
            try:
                path_sw = Path(path, mode='sw')
            except TypeError:
                pass
            else:
                if path_sw.is_fsspec:
                    if multifile:
                        raise NotImplementedError('multifile=True not supported for fsspec paths: '+path)
                    fsspec = import_fsspec('ArgumentParser.save')
                    with fsspec.open(path, 'w') as f:
                        f.write(self.dump(cfg, **dump_kwargs))  # type: ignore
                    return

        path_fc = Path(path, mode='fc')
        check_overwrite(path_fc)

        if not multifile:
            with open(path_fc(), 'w') as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore

        else:
            cfg = deepcopy(cfg)
            _ActionLink.strip_link_target_keys(self, cfg)

            if not skip_check:
                self.check_config(strip_meta(cfg), branch=branch)

            def save_paths(cfg):
                for key in cfg.get_sorted_keys():
                    val = cfg[key]
                    if isinstance(val, (Namespace, dict)) and '__path__' in val:
                        action = _find_action(self, key)
                        if isinstance(action, (ActionJsonSchema, ActionJsonnet, ActionTypeHint, _ActionConfigLoad)):
                            val_path = Path(os.path.basename(val['__path__']()), mode='fc')
                            check_overwrite(val_path)
                            val_out = strip_meta(val)
                            if isinstance(val, Namespace):
                                val_out = val_out.as_dict()
                            if '__orig__' in val:
                                val_str = val['__orig__']
                            elif str(val_path).lower().endswith('.json'):
                                val_str = json.dumps(val_out, indent=2, **self.dump_json_kwargs)+'\n'
                            else:
                                val_str = yaml.safe_dump(val_out, **self.dump_yaml_kwargs)
                            with open(val_path(), 'w') as f:
                                f.write(val_str)
                            cfg[key] = os.path.basename(val_path())
                    elif isinstance(val, Path) and key in self.save_path_content and 'r' in val.mode:
                        val_path = Path(os.path.basename(val()), mode='fc')
                        check_overwrite(val_path)
                        with open(val_path(), 'w') as f:
                            f.write(val.get_content())
                        cfg[key] = type(val)(str(val_path))

            with change_to_path_dir(path_fc):
                save_paths(cfg)
            dump_kwargs['skip_check'] = True
            with open(path_fc(), 'w') as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore


    ## Methods related to defaults ##

    def set_defaults(self, *args, **kwargs) -> None:
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
                        raise KeyError(f'No action for destination key "{dest}" to set its default.')
                    action.default = args[n][dest]
                    if isinstance(action.default, LazyInitBaseClass):
                        action.default = action.default.lazy_get_init_data()
        if kwargs:
            self.set_defaults(kwargs)


    def _get_default_config_files(self) -> List[Path]:
        default_config_files: List[Path] = []
        for pattern in self.default_config_files:
            default_config_files += sorted(glob.glob(os.path.expanduser(pattern)))
        if len(default_config_files) > 0:
            try:
                return [Path(x, mode=get_config_read_mode()) for x in default_config_files]
            except TypeError:
                pass
        return []


    def get_default(self, dest: str) -> Any:
        """Gets a single default value for the given destination key.

        Args:
            dest: Destination key from which to get the default.

        Raises:
            KeyError: If key or its default not defined in the parser.
        """
        action, _ = _find_parent_action_and_subcommand(self, dest)
        if action is None or dest != action.dest or action.dest == SUPPRESS:
            raise KeyError(f'No action for destination key "{dest}" to get its default.')

        def check_suppressed_default():
            if action.default == SUPPRESS:
                raise KeyError(f'Action for destination key "{dest}" does not specify a default.')

        if not self._get_default_config_files():
            check_suppressed_default()
            return action.default

        defaults = self.get_defaults()
        if action.dest not in defaults:
            check_suppressed_default()
        return defaults.get(action.dest)


    def get_defaults(self, skip_check: bool = False) -> Namespace:
        """Returns a namespace with all default values.

        Args:
            nested: Whether the namespace should be nested.
            skip_check: Whether to skip check if configuration is valid.

        Returns:
            An object with all default values as attributes.
        """
        cfg = Namespace()
        for action in filter_default_actions(self._actions):
            if action.default != SUPPRESS and action.dest != SUPPRESS:
                cfg[action.dest] = action.default

        self._logger.info('Loaded default values from parser.')

        default_config_files = self._get_default_config_files()
        for default_config_file in default_config_files:
            with change_to_path_dir(default_config_file):
                cfg_file = self._load_config_parser_mode(default_config_file.get_content())
                try:
                    self.print_config_skip = True
                    cfg_file = self._parse_common(
                        cfg=cfg_file,
                        env=None,
                        defaults=False,
                        with_meta=None,
                        skip_check=skip_check,
                        skip_required=True,
                    )
                    delattr(self, 'print_config_skip')
                except (TypeError, KeyError, ParserError) as ex:
                    raise ParserError(f'Problem in default config file "{default_config_file}" :: {ex.args[0]}') from ex
            cfg = self.merge_config(cfg_file, cfg)
            meta = cfg.get('__default_config__')
            if isinstance(meta, list):
                meta.append(default_config_file)
            elif isinstance(meta, Path):
                cfg['__default_config__'] = [meta, default_config_file]
            else:
                cfg['__default_config__'] = default_config_file
            self._logger.info(f'Parsed configuration from default path: {default_config_file}')

        return cfg


    ## Other methods ##

    def error(self, message: str, ex: Exception = None) -> NoReturn:
        """Logs error message if a logger is set, calls the error handler and raises a ParserError."""
        self._logger.error(message)
        if self._error_handler is not None:
            with redirect_stderr(self._stderr):
                self._error_handler(self, message)
        if ex is None:
            raise ParserError(message)
        else:
            raise ParserError(message) from ex


    def check_config(
        self,
        cfg: Namespace,
        skip_none: bool = True,
        skip_required: bool = False,
        branch: str = None,
    ) -> None:
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg: The configuration object to check.
            skip_none: Whether to skip checking of values that are None.
            skip_required: Whether to skip checking required arguments.
            branch: Base key in case cfg corresponds only to a branch.

        Raises:
            TypeError: If any of the values are not valid.
            KeyError: If a key in cfg is not defined in the parser.
        """
        cfg = ccfg = cfg.clone()
        if isinstance(branch, str):
            branch_cfg = cfg
            cfg = Namespace()
            cfg[branch] = branch_cfg

        def check_required(cfg, parser, prefix=''):
            for reqkey in parser.required_args:
                try:
                    val = cfg[reqkey]
                    if val is None:
                        raise TypeError
                except (KeyError, TypeError) as ex:
                    raise TypeError(f'Key "{prefix}{reqkey}" is required but not included in config object or its value is None.') from ex
            subcommand, subparser = _ActionSubCommands.get_subcommand(parser, cfg, fail_no_subcommand=False)
            if subcommand is not None and subparser is not None:
                check_required(cfg.get(subcommand), subparser, subcommand+'.')

        def check_values(cfg):
            for key in cfg.get_sorted_keys():
                val = cfg[key]
                action = _find_action(self, key)
                if action is None:
                    if _is_branch_key(self, key) or key.endswith('.class_path') or '.init_args' in key:
                        continue
                    action = _find_parent_action(self, key)
                    if action and not ActionTypeHint.is_class_typehint(action, only_subclasses=True):
                        continue
                if action is not None:
                    if val is None and skip_none:
                        continue
                    try:
                        self._check_value_key(action, val, key, ccfg)
                    except TypeError as ex:
                        if not (val == {} and ActionTypeHint.is_class_typehint(action) and key not in self.required_args):
                            raise ex
                else:
                    raise KeyError(f'No action for destination key "{key}" to check its value.')

        try:
            if not skip_required:
                check_required(cfg, self)
            check_values(cfg)
        except (TypeError, KeyError) as ex:
            prefix = 'Configuration check failed :: '
            message = ex.args[0]
            if prefix not in message:
                message = prefix+message
            raise type(ex)(message) from ex


    def instantiate_classes(
        self,
        cfg: Namespace,
        instantiate_groups: bool = True,
    ) -> Namespace:
        """Recursively instantiates all subclasses defined by 'class_path' and 'init_args' and class groups.

        Args:
            cfg: The configuration object to use.
            instantiate_groups: Whether class groups should be instantiated.

        Returns:
            A configuration object with all subclasses and class groups instantiated.
        """
        components: List[Union[ActionTypeHint, _ActionConfigLoad, _ArgumentGroup]] = []
        for action in filter_default_actions(self._actions):
            if isinstance(action, ActionTypeHint) or \
               (isinstance(action, _ActionConfigLoad) and is_pure_dataclass(action.basetype)):
                components.append(action)

        if instantiate_groups:
            skip = set(c.dest for c in components)
            groups = [g for g in self._action_groups if hasattr(g, 'instantiate_class') and g.dest not in skip]
            components.extend(groups)

        components.sort(key=lambda x: -len(split_key(x.dest)))  # type: ignore
        order = _ActionLink.instantiation_order(self)
        components = _ActionLink.reorder(order, components)

        cfg = strip_meta(cfg)
        for component in components:
            if isinstance(component, (ActionTypeHint, _ActionConfigLoad)):
                try:
                    value, parent, key = cfg.get_value_and_parent(component.dest)
                except KeyError:
                    pass
                else:
                    if value is not None:
                        parent[key] = component.instantiate_classes(value)
                        _ActionLink.apply_instantiation_links(self, cfg, component.dest)
            else:
                component.instantiate_class(component, cfg)
                _ActionLink.apply_instantiation_links(self, cfg, component.dest)

        subcommand, subparser = _ActionSubCommands.get_subcommand(self, cfg, fail_no_subcommand=False)
        if subcommand is not None and subparser is not None:
            cfg[subcommand] = subparser.instantiate_classes(cfg[subcommand], instantiate_groups=instantiate_groups)

        return cfg


    def strip_unknown(self, cfg: Namespace) -> Namespace:
        """Removes all unknown keys from a configuration object.

        Args:
            cfg: The configuration object to strip.

        Returns:
            The stripped configuration object.
        """
        cfg = deepcopy(cfg)

        del_keys = []
        for key in cfg.keys():
            if _find_action(self, key) is None and not is_meta_key(key):
                del_keys.append(key)

        for key in del_keys:
            del cfg[key]

        return cfg


    def get_config_files(self, cfg: Namespace) -> List[str]:
        """Returns a list of loaded config file paths.

        Args:
            cfg: The configuration object.

        Returns:
            Paths to loaded config files.
        """
        cfg_files = []
        if '__default_config__' in cfg:
            cfg_files.append(cfg['__default_config__'])
        for action in filter_default_actions(self._actions):
            if isinstance(action, ActionConfigFile) and action.dest in cfg and cfg[action.dest] is not None:
                cfg_files.extend(p for p in cfg[action.dest] if p is not None)
        return cfg_files


    def format_help(self) -> str:
        if len(self._default_config_files) > 0:
            note = 'no existing default config file found.'
            try:
                defaults = self.get_defaults()
                if '__default_config__' in defaults:
                    config_files = defaults['__default_config__']
                    if isinstance(config_files, list):
                        config_files = [str(x) for x in config_files]
                    note = f'default values below will be ones overridden by the contents of: {config_files}'
                    self.formatter_class.defaults = defaults
            except ParserError as ex:
                note = f'tried getting defaults considering default_config_files but failed due to: {ex}'
            group = self._default_config_files_group
            group.description = f'{self._default_config_files}, Note: {note}'
        help_str = super().format_help()
        self.formatter_class.defaults = None
        return help_str


    def _apply_actions(self, cfg: Namespace, parent_key: str = '') -> None:
        """Runs _check_value_key on actions present in config."""
        if parent_key:
            cfg_branch = cfg
            cfg = Namespace()
            cfg[parent_key] = cfg_branch
        keys = cfg.get_sorted_keys()
        seen_keys: Set[str] = set()
        num = 0
        while num < len(keys):
            key = keys[num]
            num += 1
            if key in seen_keys:
                continue
            action, subcommand = _find_parent_action_and_subcommand(self, key)
            if action is not None and not isinstance(action, _ActionSubCommands):
                if subcommand is not None:
                    action_dest = subcommand+'.'+action.dest
                else:
                    action_dest = action.dest
                if action_dest == key:
                    value = self._check_value_key(action, cfg[key], key, cfg)
                else:
                    value = cfg[action_dest]
                    value = self._check_value_key(action, value, action_dest, cfg)
                    seen_keys.update(action_dest+'.'+k for k in value.keys())
                cfg[action_dest] = value
                if isinstance(value, Namespace):
                    new_keys = value.get_sorted_keys()
                    keys += [action_dest+'.'+k for k in new_keys if action_dest+'.'+k not in keys]


    @staticmethod
    def merge_config(cfg_from: Namespace, cfg_to: Namespace) -> Namespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from: The configuration from which to merge.
            cfg_to: The configuration into which to merge.

        Returns:
            A new object with the merged configuration.
        """
        del_keys = []
        for key_class_path in [k for k in cfg_from.keys() if k.endswith('.class_path')]:
            key_init_args = key_class_path[:-len('class_path')] + 'init_args'
            if key_class_path in cfg_to and key_init_args in cfg_to and cfg_from[key_class_path] != cfg_to[key_class_path]:
                del_keys.append(key_init_args)
        cfg = cfg_to.clone()
        for key in reversed(del_keys):
            del cfg[key]
        cfg.update(cfg_from)
        return cfg


    def _check_value_key(self, action: Action, value: Any, key: str, cfg: Namespace) -> Any:
        """Checks the value for a given action.

        Args:
            action: The action used for parsing.
            value: The value to parse.
            key: The configuration key.

        Raises:
            TypeError: If the value is not valid.
        """
        if action.choices is not None and isinstance(action, _ActionSubCommands):
            leaf_key = split_key_leaf(key)[-1]
            if leaf_key == action.dest:
                return value
            subparser = action._name_parser_map[leaf_key]
            subparser.check_config(value)  # type: ignore
        elif isinstance(action, _ActionConfigLoad):
            if isinstance(value, str):
                fpath = None
                if '.' in key:
                    parent = cfg.get(split_key_leaf(key)[0])
                    if isinstance(parent, Namespace):
                        fpath = parent.get('__path__')
                with change_to_path_dir(fpath):
                    value = action.check_type(value, self)
        elif hasattr(action, '_check_type'):
            value = action._check_type(value, cfg=cfg)  # type: ignore
        elif action.type is not None:
            try:
                if action.nargs in {None, '?'} or action.nargs == 0:
                    value = action.type(value)
                elif value is not None:
                    for k, v in enumerate(value):
                        value[k] = action.type(v)
            except (TypeError, ValueError) as ex:
                raise TypeError(f'Parser key "{key}": {ex}') from ex
        return value


    ## Properties ##

    @property
    def error_handler(self):
        """Property for the error_handler function that is called when there are parsing errors.

        :getter: Returns the current error_handler function.
        :setter: Sets a new error_handler function (Callable[self, message:str] or None).

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._error_handler


    @error_handler.setter
    def error_handler(self, error_handler):
        if callable(error_handler) or error_handler is None:
            self._error_handler = error_handler
        else:
            raise ValueError('error_handler can be either a Callable or None.')


    @property
    def default_config_files(self):
        """Default config file locations.

        :getter: Returns the current default config file locations.
        :setter: Sets new default config file locations, e.g. :code:`['~/.config/myapp/*.yaml']`.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._default_config_files


    @default_config_files.setter
    def default_config_files(self, default_config_files:Optional[List[str]]):
        if default_config_files is None:
            self._default_config_files = []
        elif isinstance(default_config_files, list) and all(isinstance(x, str) for x in default_config_files):
            self._default_config_files = default_config_files
        else:
            raise ValueError('default_config_files has to be None or List[str].')

        if len(self._default_config_files) > 0:
            if not hasattr(self, '_default_config_files_group'):
                group_title = 'default config file locations'
                group = _ArgumentGroup(self, title=group_title)
                self._action_groups = [group] + self._action_groups  # type: ignore
                self._default_config_files_group = group
        elif hasattr(self, '_default_config_files_group'):
            self._action_groups = [g for g in self._action_groups if g != self._default_config_files_group]
            delattr(self, '_default_config_files_group')


    @property
    def default_env(self) -> bool:
        """Whether by default environment variables parsing is enabled.

        :getter: Returns the current default environment variables parsing setting.
        :setter: Sets the default environment variables parsing setting.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._default_env


    @default_env.setter
    def default_env(self, default_env:bool):
        if isinstance(default_env, bool):
            self._default_env = default_env
        else:
            raise ValueError('default_env has to be a boolean.')


    @property
    def default_meta(self) -> bool:
        """Whether by default metadata is included in config objects.

        :getter: Returns the current default metadata setting.
        :setter: Sets the default metadata setting.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._default_meta


    @default_meta.setter
    def default_meta(self, default_meta:bool):
        if isinstance(default_meta, bool):
            self._default_meta = default_meta
        else:
            raise ValueError('default_meta has to be a boolean.')


    @property
    def env_prefix(self) -> Optional[str]:
        """The environment variables prefix property.

        :getter: Returns the current environment variables prefix.
        :setter: Sets the environment variables prefix.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._env_prefix


    @env_prefix.setter
    def env_prefix(self, env_prefix: Optional[str]):
        if env_prefix is None:
            self._env_prefix = os.path.splitext(self.prog)[0]
        elif isinstance(env_prefix, str):
            self._env_prefix = env_prefix
        else:
            raise ValueError('env_prefix has to be a string or None.')


from .deprecated import parse_as_dict_patch, instantiate_subclasses_patch
instantiate_subclasses_patch()
if 'JSONARGPARSE_SKIP_DEPRECATION_PATCH' not in os.environ:
    parse_as_dict_patch()
