import os
import re
import sys
import glob
import json
import yaml
import logging
import inspect
import argparse
from copy import deepcopy
from contextlib import redirect_stderr
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from argparse import ArgumentError, Action, Namespace, SUPPRESS

from .formatters import DefaultHelpFormatter
from .jsonnet import ActionJsonnet
from .jsonschema import ActionJsonSchema
from .signatures import is_pure_dataclass, SignatureArguments
from .typehints import ActionTypeHint
from .actions import (
    ActionParser,
    ActionConfigFile,
    ActionPath,
    _ActionSubCommands,
    _ActionPrintConfig,
    _ActionConfigLoad,
    _ActionLink,
    _find_action,
    _find_parent_action,
    _is_action_value_list,
    filter_default_actions,
)
from .optionals import (
    argcomplete_support,
    dump_preserve_order_support,
    FilesCompleterMethod,
    get_config_read_mode,
    import_jsonnet,
    import_argcomplete,
    TypeCastCompleterMethod,
)
from .util import (
    namespace_to_dict,
    dict_to_namespace,
    _flat_namespace_to_dict,
    _dict_to_flat_namespace,
    yamlParserError,
    yamlScannerError,
    ParserError,
    meta_keys,
    strip_meta,
    usage_and_exit_error_handler,
    change_to_path_dir,
    Path,
    LoggerProperty,
    get_key_value_from_flat_dict,
    update_key_value_in_flat_dict,
    _get_key_value,
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


class _ActionsContainer(SignatureArguments, argparse._ActionsContainer):
    """Extension of argparse._ActionsContainer to support additional functionalities."""

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
                    raise ValueError('Type hint as type does not allow providing an action')
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
        for key in meta_keys:
            if key in action.dest:
                raise ValueError('Argument with destination name "'+key+'" not allowed.')
        if action.help is None and \
           issubclass(parser.formatter_class, DefaultHelpFormatter):
            action.help = ' '
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
            raise ValueError('Group with name '+name+' already exists.')
        group = _ArgumentGroup(parser, *args, **kwargs)
        group.parser = parser
        parser._action_groups.append(group)
        if name is not None:
            parser.groups[name] = group
        return group


class _ArgumentGroup(_ActionsContainer, argparse._ArgumentGroup):
    """Extension of argparse._ArgumentGroup to support additional functionalities."""
    parser = None  # type: Union[ArgumentParser, None]


class ArgumentParser(_ActionsContainer, argparse.ArgumentParser, LoggerProperty):
    """Parser for command line, yaml/jsonnet files and environment variables."""

    groups = None  # type: Dict[str, argparse._ArgumentGroup]


    def __init__(
        self,
        *args,
        env_prefix: str = None,
        error_handler: Optional[Callable[[Type, str], None]] = usage_and_exit_error_handler,
        formatter_class: Type[argparse.HelpFormatter] = DefaultHelpFormatter,
        logger: Union[bool, Dict[str, str], logging.Logger] = None,
        version: str = None,
        print_config: Optional[str] = '--print_config',
        parser_mode: str = 'yaml',
        parse_as_dict: bool = False,
        default_config_files: List[str] = None,
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
            parse_as_dict: Whether to parse as dict instead of Namespace.
            default_config_files: Default config file locations, e.g. :code:`['~/.config/myapp/*.yaml']`.
            default_env: Set the default value on whether to parse environment variables.
            default_meta: Set the default value on whether to include metadata in config objects.
        """
        class FormatterClass(formatter_class):  # type: ignore
            _parser = self

        kwargs['formatter_class'] = FormatterClass
        super().__init__(*args, **kwargs)
        if self.groups is None:
            self.groups = {}
        self.required_args = set()  # type: Set[str]
        self.save_path_content = set()  # type: Set[str]
        self._stderr = sys.stderr
        self._parse_as_dict = parse_as_dict
        self.default_config_files = default_config_files
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.parser_mode = parser_mode
        self.logger = logger
        self.error_handler = error_handler
        if print_config is not None:
            self.add_argument(print_config, action=_ActionPrintConfig)
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

        if namespace is None:
            namespace = Namespace()

        if caller == 'argcomplete':
            namespace = _dict_to_flat_namespace(self._merge_config(self.get_defaults(nested=False, skip_check=True), namespace))

        try:
            namespace, args = self._parse_known_args(args, namespace)
        except (ArgumentError, ParserError) as ex:
            self.error(str(ex), ex)

        return namespace, args


    def _parse_common(
        self,
        cfg: Dict[str, Any],
        env: Optional[bool],
        defaults: bool,
        nested: bool,
        with_meta: Optional[bool],
        skip_check: bool,
        skip_required: bool = False,
        skip_subcommands: bool = False,
        fail_no_subcommand: bool = True,
        cfg_base: Union[Namespace, Dict[str, Any]] = None,
        log_message: str = None,
    ) -> Union[Namespace, Dict[str, Any]]:
        """Common parsing code used by other parse methods.

        Args:
            cfg: The configuration object.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            nested: Whether the namespace should be nested.
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

        if nested:
            cfg = _flat_namespace_to_dict(_dict_to_flat_namespace(cfg))

        if cfg_base is not None:
            if isinstance(cfg_base, Namespace):
                cfg_base = namespace_to_dict(cfg_base)
            cfg = self._merge_config(cfg, cfg_base)

        if env:
            cfg_env = self.parse_env(defaults=defaults, nested=nested, _skip_check=True, _skip_subcommands=True)
            cfg = self._merge_config(cfg, cfg_env)

        elif defaults:
            cfg = self._merge_config(cfg, self.get_defaults(nested=nested, skip_check=True))

        if not (with_meta or (with_meta is None and self._default_meta)):
            cfg = strip_meta(cfg)

        cfg_ns = dict_to_namespace(cfg)

        _ActionPrintConfig.print_config_if_requested(self, cfg_ns)

        _ActionLink.propagate_arguments(self, cfg_ns)

        if not skip_check:
            self.check_config(cfg_ns, skip_required=skip_required)

        if not nested:
            cfg_ns = _dict_to_flat_namespace(namespace_to_dict(cfg_ns))

        if log_message is not None:
            self._logger.info(log_message)

        return namespace_to_dict(cfg_ns) if self._parse_as_dict else cfg_ns


    def parse_args(  # type: ignore[override]
        self,
        args: List[str] = None,
        namespace: Namespace = None,
        env: bool = None,
        defaults: bool = True,
        nested: bool = True,
        with_meta: bool = None,
        _skip_check: bool = False,
    ) -> Union[Namespace, Dict[str, Any]]:
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            args: List of arguments to parse or None to use sys.argv.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            nested: Whether the namespace should be nested.
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
                    self.error('Unrecognized arguments: %s' % ' '.join(unk))

            parsed_cfg = self._parse_common(
                cfg=namespace_to_dict(cfg),
                env=env,
                defaults=defaults,
                nested=nested,
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
        cfg_base = None,
        env: bool = None,
        defaults: bool = True,
        nested: bool = True,
        with_meta: bool = None,
        _skip_check: bool = False,
        _skip_required: bool = False,
    ) -> Union[Namespace, Dict[str, Any]]:
        """Parses configuration given as an object.

        Args:
            cfg_obj: The configuration object.
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
            cfg = vars(_dict_to_flat_namespace(cfg_obj))
            self._apply_actions(cfg)
            cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                nested=nested,
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
        nested: bool = True,
        with_meta: bool = None,
        _skip_logging: bool = False,
        _skip_check: bool = False,
        _skip_subcommands: bool = False,
    ) -> Union[Namespace, Dict[str, Any]]:
        """Parses environment variables.

        Args:
            env: The environment object to use, if None `os.environ` is used.
            defaults: Whether to merge with the parser's defaults.
            nested: Whether the namespace should be nested.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            if env is None:
                env = dict(os.environ)
            cfg = {}  # type: ignore
            actions = filter_default_actions(self._actions)
            for action in actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, ActionConfigFile):
                    namespace = _dict_to_flat_namespace(cfg)
                    ActionConfigFile._apply_config(self, namespace, action.dest, env[env_var])
                    cfg = vars(namespace)
            for action in actions:
                env_var = _get_env_var(self, action)
                if env_var in env and isinstance(action, _ActionSubCommands):
                    env_val = env[env_var]
                    if env_val in action.choices:
                        cfg[action.dest] = subcommand = self._check_value_key(action, env_val, action.dest, cfg)
                        pcfg = action._name_parser_map[env_val].parse_env(env=env, defaults=defaults, nested=False, _skip_logging=True, _skip_check=True)  # type: ignore
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

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=False,
                defaults=defaults,
                nested=nested,
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
        ext_vars: dict = None,
        env: bool = None,
        defaults: bool = True,
        nested: bool = True,
        with_meta: bool = None,
        _skip_check: bool = False,
        _fail_no_subcommand: bool = True,
        _base = None,
    ) -> Union[Namespace, Dict[str, Any]]:
        """Parses a configuration file (yaml or jsonnet) given its path.

        Args:
            cfg_path: Path to the configuration file to parse.
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
        fpath = Path(cfg_path, mode=get_config_read_mode())
        with change_to_path_dir(fpath):
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(cfg_str,
                                           cfg_path,
                                           ext_vars,
                                           env,
                                           defaults,
                                           nested,
                                           with_meta=with_meta,
                                           _skip_logging=True,
                                           _skip_check=_skip_check,
                                           _fail_no_subcommand=_fail_no_subcommand)

        self._logger.info('Parsed %s from path: %s', self.parser_mode, cfg_path)

        return parsed_cfg


    def parse_string(
        self,
        cfg_str: str,
        cfg_path: str = '',
        ext_vars: dict = None,
        env: bool = None,
        defaults: bool = True,
        nested: bool = True,
        with_meta: bool = None,
        _skip_logging: bool = False,
        _skip_check: bool = False,
        _fail_no_subcommand: bool = True,
    ) -> Union[Namespace, Dict[str, Any]]:
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
            cfg = self._load_cfg(cfg_str, cfg_path, ext_vars)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                nested=nested,
                with_meta=with_meta,
                skip_check=_skip_check,
                fail_no_subcommand=_fail_no_subcommand,
                log_message=('Parsed %s string.' % self.parser_mode),
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        return parsed_cfg


    def _load_cfg(
        self,
        cfg_str: str,
        cfg_path: str = '',
        ext_vars: dict = None,
    ) -> Dict[str, Any]:
        """Loads a configuration string (yaml or jsonnet) into a namespace checking all values against the parser.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.
            base: Base key to prepend.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        if self.parser_mode == 'jsonnet':
            ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
            _jsonnet = import_jsonnet('_load_cfg')
            cfg_str = _jsonnet.evaluate_snippet(cfg_path, cfg_str, ext_vars=ext_vars, ext_codes=ext_codes)
        try:
            cfg = yaml.safe_load(cfg_str)
        except (yamlParserError, yamlScannerError) as ex:
            raise TypeError('Problems parsing config :: '+str(ex)) from ex
        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))

        self._apply_actions(cfg)

        return cfg


    def link_arguments(
        self,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Callable = None,
    ):
        """Makes an argument value be derived from the values other arguments.

        Source keys can be individual arguments or nested groups. The target key
        has to be an single argument. The keys can be inside init_args of a
        subclass. The compute function should accept as many positional
        arguments as there are sources and return a value of type compatible
        with the target.

        Args:
            source: Key(s) from which the target value is derived.
            target: Key to where the value is set.
            compute_fn: Function to compute target value from source.

        Raises:
            ValueError: If an invalid parameter is given.
        """
        _ActionLink(self, source, target, compute_fn)


    ## Methods for adding to the parser ##

    def add_subparsers(self, **kwargs):
        """Raises a NotImplementedError since jsonargparse uses add_subcommands."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommands method.')


    def add_subcommands(self, required:bool=True, dest:str='subcommand', **kwargs) -> Action:
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
        subcommands = super().add_subparsers(dest=dest, **kwargs)
        if required:
            self.required_args.add(dest)
        subcommands._required = required  # type: ignore
        subcommands.required = False
        _find_action(self, dest)._env_prefix = self.env_prefix
        return subcommands


    ## Methods for serializing config objects ##

    def dump(
        self,
        cfg: Union[Namespace, Dict[str, Any]],
        format: str = 'parser_mode',
        skip_none: bool = True,
        skip_check: bool = False,
    ) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg: The configuration object to dump.
            format: The output format: "yaml", "json", "json_indented" or "parser_mode".
            skip_none: Whether to exclude entries whose value is None.
            skip_check: Whether to skip parser checking.

        Returns:
            The configuration in yaml or json format.

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
            for action in filter_default_actions(actions):
                if (action.help == SUPPRESS and not isinstance(action, _ActionConfigLoad)) or \
                   isinstance(action, ActionConfigFile) or \
                   (skip_none and action.dest in cfg and cfg[action.dest] is None):
                    cfg.pop(action.dest, None)
                elif isinstance(action, ActionPath):
                    if cfg[action.dest] is not None:
                        if isinstance(cfg[action.dest], list):
                            cfg[action.dest] = [str(p) for p in cfg[action.dest]]
                        else:
                            cfg[action.dest] = str(cfg[action.dest])
                elif isinstance(action, ActionTypeHint):
                    value = get_key_value_from_flat_dict(cfg, action.dest)
                    if value is not None and value != {}:
                        value = ActionTypeHint.serialize(value, action._typehint)
                        update_key_value_in_flat_dict(cfg, action.dest, value)

        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))
        cleanup_actions(cfg, self._actions)
        cfg = _flat_namespace_to_dict(_dict_to_flat_namespace(cfg))

        if format == 'parser_mode':
            format = 'yaml' if self.parser_mode == 'yaml' else 'json_indented'
        if format == 'yaml':
            return yaml.safe_dump(cfg, **self.dump_yaml_kwargs)  # type: ignore
        elif format == 'json_indented':
            return json.dumps(cfg, indent=2, **self.dump_json_kwargs)+'\n'  # type: ignore
        elif format == 'json':
            return json.dumps(cfg, separators=(',', ':'), **self.dump_json_kwargs)  # type: ignore
        else:
            raise ValueError('Unknown output format "'+str(format)+'".')


    def save(
        self,
        cfg: Union[Namespace, Dict[str, Any]],
        path: str,
        format: str = 'parser_mode',
        skip_none: bool = True,
        skip_check: bool = False,
        overwrite: bool = False,
        multifile: bool = True,
        branch: str = None,
    ):
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
        def check_overwrite(path):
            if not overwrite and os.path.isfile(path()):
                raise ValueError('Refusing to overwrite existing file: '+path())

        path_fc = Path(path, mode='fc')
        check_overwrite(path_fc)

        dump_kwargs = {'format': format, 'skip_none': skip_none, 'skip_check': skip_check}

        if not multifile:
            with open(path_fc(), 'w') as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore

        else:
            cfg = deepcopy(cfg)
            if not isinstance(cfg, dict):
                cfg = namespace_to_dict(cfg)

            if not skip_check:
                self.check_config(strip_meta(cfg), branch=branch)

            def save_paths(cfg, base=None):
                replace_keys = {}
                for key, val in cfg.items():
                    full_key = ('' if base is None else base+'.')+key
                    if isinstance(val, dict):
                        kbase = str(key) if base is None else base+'.'+str(key)
                        if '__path__' in val:
                            val_path = Path(os.path.basename(val['__path__']()), mode='fc')
                            check_overwrite(val_path)
                            action = _find_action(self, kbase)
                            if isinstance(action, (ActionJsonSchema, ActionJsonnet, _ActionConfigLoad)):
                                val_out = strip_meta(val)
                                if '__orig__' in val:
                                    val_str = val['__orig__']
                                elif str(val_path).lower().endswith('.json'):
                                    val_str = json.dumps(val_out, indent=2, **self.dump_json_kwargs)+'\n'
                                else:
                                    val_str = yaml.safe_dump(val_out, **self.dump_yaml_kwargs)
                                with open(val_path(), 'w') as f:
                                    f.write(val_str)
                                replace_keys[key] = os.path.basename(val_path())
                        else:
                            save_paths(val, kbase)
                    elif isinstance(val, Path) and full_key in self.save_path_content and 'r' in val.mode:
                        val_path = Path(os.path.basename(val()), mode='fc')
                        check_overwrite(val_path)
                        with open(val_path(), 'w') as f:
                            f.write(val.get_content())
                        replace_keys[key] = type(val)(str(val_path))
                for key, val in replace_keys.items():
                    cfg[key] = val

            with change_to_path_dir(path_fc):
                save_paths(cfg)
            dump_kwargs['skip_check'] = True
            with open(path_fc(), 'w') as f:
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
                    action = _find_action(self, dest, within_subcommands=True)
                    if action is None:
                        raise KeyError('No action for destination key "'+dest+'" to set its default.')
                    action.default = args[n][dest]
        if kwargs:
            self.set_defaults(kwargs)


    def _get_default_config_file(self):
        default_config_files = []  # type: List[str]
        for pattern in self.default_config_files:
            default_config_files += glob.glob(os.path.expanduser(pattern))
        if len(default_config_files) > 0:
            try:
                return Path(default_config_files[0], mode=get_config_read_mode())
            except TypeError:
                pass


    def get_default(self, dest:str):
        """Gets a single default value for the given destination key.

        Args:
            dest: Destination key from which to get the default.

        Raises:
            KeyError: If key not defined in the parser.
        """
        action = _find_action(self, dest)
        if action is None or action.default == SUPPRESS or action.dest == SUPPRESS:
            raise KeyError('No action for destination key "'+dest+'" to get its default.')
        default_config_file = self._get_default_config_file()
        if default_config_file is None:
            return action.default
        return getattr(self.get_defaults(), action.dest)


    def get_defaults(self, nested:bool=True, skip_check:bool=False) -> Namespace:
        """Returns a namespace with all default values.

        Args:
            nested: Whether the namespace should be nested.
            skip_check: Whether to skip check if configuration is valid.

        Returns:
            An object with all default values as attributes.
        """
        cfg = {}
        for action in filter_default_actions(self._actions):
            if action.default != SUPPRESS and action.dest != SUPPRESS:
                cfg[action.dest] = action.default

        cfg = namespace_to_dict(_dict_to_flat_namespace(cfg))

        self._logger.info('Loaded default values from parser.')

        default_config_file = self._get_default_config_file()
        if default_config_file is not None:
            with change_to_path_dir(default_config_file):
                cfg_file = self._load_cfg(default_config_file.get_content())
                try:
                    self.print_config_skip = True
                    cfg_file = self.parse_object(  # type: ignore
                        _flat_namespace_to_dict(dict_to_namespace(cfg_file)),
                        defaults=False,
                        nested=False,
                        _skip_check=skip_check,
                        _skip_required=True,
                    )
                    delattr(self, 'print_config_skip')
                except (TypeError, KeyError, ParserError) as ex:
                    raise ParserError('Problem in default config file "'+str(default_config_file)+'" :: '+ex.args[0]) from ex
            cfg = self._merge_config(cfg_file, cfg)
            cfg['__default_config__'] = default_config_file
            self._logger.info('Parsed configuration from default path: %s', str(default_config_file))

        if nested:
            cfg = _flat_namespace_to_dict(dict_to_namespace(cfg))

        return dict_to_namespace(cfg)


    ## Other methods ##

    def error(self, message:str, ex:Exception=None):
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
        cfg: Union[Namespace, Dict[str, Any]],
        skip_none: bool = True,
        skip_required: bool = False,
        branch: str = None,
    ):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg: The configuration object to check.
            skip_none: Whether to skip checking of values that are None.
            branch: Base key in case cfg corresponds only to a branch.

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

        def check_required(cfg, parser):
            for reqkey in parser.required_args:
                try:
                    val = get_key_value(cfg, reqkey)
                    if val is None:
                        raise TypeError()
                except (KeyError, TypeError) as ex:
                    raise TypeError('Key "'+reqkey+'" is required but not included in config object or its value is None.') from ex

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
                    elif isinstance(action, _ActionConfigLoad) and isinstance(val, dict):
                        check_values(val, kbase)
                elif isinstance(val, dict):
                    check_values(val, kbase)
                else:
                    raise KeyError('No action for key "'+kbase+'" to check its value.')

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


    def instantiate_subclasses(self, cfg:Union[Namespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Recursively instantiates all subclasses defined by 'class_path' and 'init_args'.

        Args:
            cfg: The configuration object to use.

        Returns:
            A configuration object with all subclasses instantiated.
        """
        cfg = strip_meta(cfg)
        actions = filter_default_actions(self._actions)
        actions.sort(key=lambda x: -len(x.dest.split('.')))
        for action in actions:
            if isinstance(action, ActionTypeHint) or \
               (isinstance(action, _ActionConfigLoad) and is_pure_dataclass(action.basetype)):
                value, parent, key = _get_key_value(cfg, action.dest, parent=True)
                if value is not None:
                    parent[key] = action._instantiate_classes(value)
                #try:
                #    value, parent, key = _get_key_value(cfg, action.dest, parent=True)
                #except KeyError:
                #    pass
                #else:
                #    if value is not None:
                #        parent[key] = action._instantiate_classes(value)
        return cfg


    def strip_unknown(self, cfg:Union[Namespace, Dict[str, Any]]) -> Namespace:
        """Removes all unknown keys from a configuration object.

        Args:
            cfg: The configuration object to strip.

        Returns:
            The stripped configuration object.
        """
        cfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = namespace_to_dict(cfg)

        def strip_keys(cfg, base=None):
            del_keys = []
            for key, val in cfg.items():
                kbase = key if base is None else base+'.'+key
                if _find_action(self, kbase) is None:
                    if isinstance(val, dict):
                        strip_keys(val, kbase)
                    elif key not in meta_keys:
                        del_keys.append(key)
            for key in del_keys:
                del cfg[key]

        strip_keys(cfg)
        return dict_to_namespace(cfg)


    def get_config_files(self, cfg:Union[Namespace, Dict[str, Any]]) -> List[str]:
        """Returns a list of loaded config file paths.

        Args:
            cfg: The configuration object.

        Returns:
            Paths to loaded config files.
        """
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        cfg_files = []
        if '__default_config__' in cfg:
            cfg_files.append(cfg['__default_config__'])
        for action in filter_default_actions(self._actions):
            if isinstance(action, ActionConfigFile) and action.dest in cfg and cfg[action.dest] is not None:
                cfg_files.extend(p for p in cfg[action.dest] if p is not None)
        return cfg_files


    def format_help(self):
        if len(self._default_config_files) > 0:
            defaults = namespace_to_dict(self.get_defaults())
            note = 'no existing default config file found.'
            if '__default_config__' in defaults:
                note = 'default values below will be ones overridden by the contents of: '+str(defaults['__default_config__'])
                self.formatter_class.defaults = defaults
            group = self._default_config_files_group
            group.description = str(self._default_config_files) + ', Note: '+note
        help_str = super().format_help()
        if hasattr(self.formatter_class, 'defaults'):
            delattr(self.formatter_class, 'defaults')
        return help_str


    def _apply_actions(self, cfg):
        """Runs _check_value_key on actions present in flat config dict."""
        keys = [k for k in cfg.keys() if k.rsplit('.', 1)[-1] not in meta_keys]
        keys.sort(key=lambda x: -len(x.split('.')))
        seen_keys = set()
        for key in keys:
            if key in seen_keys:
                continue
            action = _find_parent_action(self, key)
            if action is not None:
                if action.dest == key:
                    value = self._check_value_key(action, cfg[action.dest], action.dest, cfg)
                    cfg[action.dest] = value
                else:
                    value = get_key_value_from_flat_dict(cfg, action.dest)
                    value = self._check_value_key(action, value, action.dest, cfg)
                    update_key_value_in_flat_dict(cfg, action.dest, value)
                    seen_keys.update(action.dest+'.'+k for k in value.keys())


    @staticmethod
    def merge_config(cfg_from:Namespace, cfg_to:Namespace) -> Namespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from: The configuration from which to merge.
            cfg_to: The configuration into which to merge.

        Returns:
            The merged configuration.
        """
        return dict_to_namespace(ArgumentParser._merge_config(cfg_from, cfg_to))


    @staticmethod
    def _merge_config(cfg_from:Union[Namespace, Dict[str, Any]], cfg_to:Union[Namespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from: The configuration from which to merge.
            cfg_to: The configuration into which to merge.

        Returns:
            The merged configuration.
        """
        def merge_values(cfg_from, cfg_to):
            for k, v in cfg_from.items():
                if v is None or \
                   k not in cfg_to or \
                   not isinstance(v, dict) or \
                   (isinstance(v, dict) and not isinstance(cfg_to[k], dict)):
                    cfg_to[k] = v
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
            action: The action used for parsing.
            value: The value to parse.
            key: The configuration key.

        Raises:
            TypeError: If the value is not valid.
        """
        if action.choices is not None and isinstance(action, _ActionSubCommands):
            if key == action.dest:
                return value
            parser = action._name_parser_map[key]
            parser.check_config(value)  # type: ignore
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
                raise TypeError('Parser key "'+str(key)+'": '+str(ex)) from ex
        if isinstance(value, Namespace):
            value = namespace_to_dict(value)
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
    def default_env(self):
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
    def default_meta(self):
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
    def env_prefix(self):
        """The environment variables prefix property.

        :getter: Returns the current environment variables prefix.
        :setter: Sets the environment variables prefix.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._env_prefix


    @env_prefix.setter
    def env_prefix(self, env_prefix:Optional[str]):
        if env_prefix is None:
            self._env_prefix = os.path.splitext(self.prog)[0]
        elif isinstance(env_prefix, str):
            self._env_prefix = env_prefix
        else:
            raise ValueError('env_prefix has to be a string or None.')
