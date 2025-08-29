"""Extensions of core argparse classes."""

import argparse
import glob
import inspect
import logging
import os
import sys
from contextlib import suppress
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from ._actions import (
    ActionConfigFile,
    ActionParser,
    _ActionConfigLoad,
    _ActionPrintConfig,
    _ActionSubCommands,
    _find_action,
    _find_action_and_subcommand,
    _find_parent_action_and_subcommand,
    _is_action_value_list,
    _is_branch_key,
    filter_default_actions,
    parent_parsers,
    previous_config,
)
from ._common import (
    InstantiatorCallable,
    InstantiatorsDictType,
    LoggerProperty,
    class_instantiators,
    debug_mode_active,
    get_optionals_as_positionals_actions,
    is_dataclass_like,
    lenient_check,
    parser_context,
    supports_optionals_as_positionals,
    validate_default,
)
from ._completions import (
    argcomplete_namespace,
    handle_completions,
)
from ._deprecated import ParserDeprecations, deprecated_skip_check
from ._formatters import DefaultHelpFormatter, empty_help, get_env_var
from ._jsonnet import ActionJsonnet
from ._jsonschema import ActionJsonSchema
from ._link_arguments import ActionLink, ArgumentLinking
from ._loaders_dumpers import (
    check_valid_dump_format,
    dump_using_format,
    get_loader_exceptions,
    load_value,
    loaders,
    set_omegaconf_loader,
)
from ._namespace import (
    Namespace,
    NSKeyError,
    is_meta_key,
    patch_namespace,
    recreate_branches,
    split_key,
    split_key_leaf,
    split_key_root,
    strip_meta,
)
from ._optionals import (
    _get_config_read_mode,
    fsspec_support,
    import_fsspec,
    import_jsonnet,
    omegaconf_apply,
    pyyaml_available,
)
from ._parameter_resolvers import UnknownDefault
from ._signatures import SignatureArguments
from ._typehints import ActionTypeHint, is_subclass_spec
from ._util import (
    ClassType,
    Path,
    argument_error,
    change_to_path_dir,
    get_argument_group_class,
    get_private_kwargs,
    identity,
    return_parser_if_captured,
)

__all__ = ["ActionsContainer", "ArgumentParser"]


_parse_known_has_intermixed = "intermixed" in inspect.signature(argparse.ArgumentParser._parse_known_args).parameters


class ActionsContainer(SignatureArguments, argparse._ActionsContainer):
    """Extension of argparse._ActionsContainer to support additional functionalities."""

    _action_groups: Sequence["ArgumentGroup"]  # type: ignore[assignment]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register("type", None, identity)
        self.register("action", "parsers", _ActionSubCommands)
        self.register("action", "config", ActionConfigFile)

    def add_argument(self, *args, enable_path: bool = False, **kwargs):
        """Adds an argument to the parser or argument group.

        All the arguments from `argparse.ArgumentParser.add_argument
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
        are supported. Additionally it accepts:

        Args:
            enable_path: Whether to try parsing path/subconfig when argument is a complex type.
        """
        parser = self.parser if hasattr(self, "parser") else self
        if kwargs.get("action") is not None:
            if ActionParser._is_valid_action_parser(parser, kwargs["action"]):
                return ActionParser._move_parser_actions(parser, args, kwargs)
            ActionConfigFile._ensure_single_config_argument(self, kwargs["action"])
        if "type" in kwargs:
            if is_dataclass_like(kwargs["type"]):
                nested_key = args[0].lstrip("-")
                self.add_class_arguments(kwargs.pop("type"), nested_key, **kwargs)
                return _find_action(parser, nested_key)
            if ActionTypeHint.is_supported_typehint(kwargs["type"]):
                args = ActionTypeHint.prepare_add_argument(
                    args=args,
                    kwargs=kwargs,
                    enable_path=enable_path,
                    container=super(),
                    logger=self._logger,
                )
        if "choices" in kwargs and not isinstance(kwargs["choices"], (list, tuple)):
            kwargs["choices"] = tuple(kwargs["choices"])
        action = super().add_argument(*args, **kwargs)
        action.logger = self._logger  # type: ignore[attr-defined]
        ActionConfigFile._add_print_config_argument(self, action)
        ActionJsonnet._check_ext_vars_action(parser, action)
        if is_meta_key(action.dest):
            raise ValueError(f'Argument with destination name "{action.dest}" not allowed.')
        if action.option_strings == [] and "default" in kwargs and kwargs["default"] is not argparse.SUPPRESS:
            raise ValueError("Positional arguments not allowed to have a default value.")
        validate_default(self, action)
        if action.help is None:
            action.help = empty_help
        if action.required:
            parser.required_args.add(action.dest)  # type: ignore[union-attr]
            action._required = True  # type: ignore[attr-defined]
            action.required = False
        return action

    def add_argument_group(self, *args, name: Optional[str] = None, **kwargs) -> "ArgumentGroup":
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported. Additionally it accepts:

        Args:
            name: Name of the group. If set, the group object will be included in the parser.groups dict.

        Returns:
            The group object.

        Raises:
            ValueError: If group with the same name already exists.
        """
        parser = self.parser if hasattr(self, "parser") else self
        if name is not None and name in parser.groups:  # type: ignore[union-attr]
            raise ValueError(f"Group with name {name} already exists.")
        group_class = getattr(parser, "_group_class", ArgumentGroup)
        group = group_class(parser, *args, logger=parser._logger, **kwargs)
        group.parser = parser
        parser._action_groups.append(group)  # type: ignore[union-attr]
        if name is not None:
            parser.groups[name] = group  # type: ignore[union-attr]
        return group

    def set_defaults(self, *args: Dict[str, Any], **kwargs: Any) -> None:
        """Sets default values from dictionary or keyword arguments.

        Args:
            *args: Dictionary defining the default values to set.
            **kwargs: Sets default values based on keyword arguments.

        Raises:
            KeyError: If key not defined in the parser.
        """
        for arg in args:
            for dest, default in arg.items():
                dest = dest.replace("-", "_")
                action = _find_action(self, dest)
                if action is None:
                    raise NSKeyError(f'No action for key "{dest}" to set its default.')
                elif isinstance(action, ActionConfigFile):
                    ActionConfigFile.set_default_error()
                elif isinstance(action, _ActionConfigLoad):
                    default = {f"{dest}.{k}": v for k, v in default.items()}
                    self.set_defaults(default)
                    continue
                if isinstance(action, ActionTypeHint):
                    default = action.normalize_default(default)
                self._defaults[dest] = action.default = default
        if kwargs:
            self.set_defaults(kwargs)


class ArgumentGroup(ActionsContainer, argparse._ArgumentGroup):
    """Extension of argparse._ArgumentGroup to support additional functionalities."""

    dest: Optional[str] = None
    parser: Optional[Union["ArgumentParser", ActionsContainer]] = None


class ArgumentParser(ParserDeprecations, ActionsContainer, ArgumentLinking, LoggerProperty, argparse.ArgumentParser):
    """Parser for command line, configuration files and environment variables."""

    formatter_class: Type[DefaultHelpFormatter]
    groups: Optional[Dict[str, ArgumentGroup]] = None
    _group_class: Type[ArgumentGroup]
    _subcommands_action: Optional[_ActionSubCommands] = None
    _instantiators: Optional[InstantiatorsDictType] = None

    def __init__(
        self,
        *args,
        env_prefix: Union[bool, str] = True,
        formatter_class: Type[DefaultHelpFormatter] = DefaultHelpFormatter,
        exit_on_error: bool = True,
        logger: Union[bool, str, dict, logging.Logger] = False,
        version: Optional[str] = None,
        print_config: Optional[str] = "--print_config",
        parser_mode: str = "yaml" if pyyaml_available else "json",
        dump_header: Optional[List[str]] = None,
        default_config_files: Optional[List[Union[str, os.PathLike]]] = None,
        default_env: bool = False,
        default_meta: bool = True,
        **kwargs,
    ) -> None:
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            env_prefix: Prefix for environment variables. ``True`` to derive from ``prog``.
            formatter_class: Class for printing help messages.
            logger: Logger to use or configuration for logger.
            version: Program version which will be printed by the --version argument.
            print_config: Name for print config argument, ``%s`` is replaced by config dest, set None to disable.
            parser_mode: Mode for parsing values: ``yaml``, ``json``, ``jsonnet`` or added via :func:`.set_loader`.
            dump_header: Header to include as comment when dumping a config object.
            default_config_files: Default config file locations, e.g. ``['~/.config/myapp/*.yaml']``.
            default_env: Set the default value on whether to parse environment variables.
            default_meta: Set the default value on whether to include metadata in config objects.
        """
        super().__init__(*args, formatter_class=formatter_class, logger=logger, **kwargs)
        self._group_class = get_argument_group_class(self)
        if self.groups is None:
            self.groups = {}
        self.exit_on_error = exit_on_error
        self.required_args: Set[str] = set()
        self.save_path_content: Set[str] = set()
        self.default_config_files = default_config_files
        self.default_meta = default_meta
        self.default_env = default_env
        self.env_prefix = env_prefix
        self.parser_mode = parser_mode
        self.dump_header = dump_header
        self._print_config = print_config
        if version is not None:
            self.add_argument(
                "--version", action="version", version="%(prog)s " + version, help="Print version and exit."
            )

    ## Parsing methods ##

    def parse_known_args(self, args=None, namespace=None):
        """Raises NotImplementedError to dissuade its use, since typos in configs would go unnoticed."""
        caller_mod = inspect.getmodule(inspect.stack()[1][0])
        caller = None if caller_mod is None else caller_mod.__package__
        if caller not in {"jsonargparse", "argcomplete"}:
            raise NotImplementedError(
                "parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed."
            )

        namespace = argcomplete_namespace(caller, self, namespace)

        try:
            with (
                patch_namespace(),
                parser_context(parent_parser=self, lenient_check=True),
                ActionTypeHint.subclass_arg_context(self),
            ):
                kwargs = {}
                if _parse_known_has_intermixed:
                    kwargs["intermixed"] = False
                namespace, args = self._parse_known_args(args, namespace, **kwargs)
        except argparse.ArgumentError as ex:
            self.error(str(ex), ex)

        return namespace, args

    def _positional_optionals(self, cfg, unk):
        if len(unk) == 0 or not supports_optionals_as_positionals(self):
            return cfg, unk

        for action in get_optionals_as_positionals_actions(self, include_positionals=True):
            if action.option_strings == []:
                if cfg.get(action.dest) is None:
                    self._logger.debug(f"Positional argument {action.dest} missing, aborting _positional_optionals")
                    break
                continue

            cfg[action.dest] = self._check_value_key(action, unk.pop(0), action.dest, cfg)
            if len(unk) == 0:
                break

        return cfg, unk

    def _parse_optional(self, arg_string):
        subclass_arg = ActionTypeHint.parse_argv_item(arg_string)
        if subclass_arg:
            return subclass_arg
        if arg_string == self._print_config:
            arg_string += "="
        return super()._parse_optional(arg_string)

    def _parse_common(
        self,
        cfg: Namespace,
        env: Optional[bool],
        defaults: bool,
        with_meta: Optional[bool],
        skip_validation: bool,
        skip_required: bool = False,
        skip_subcommands: bool = False,
        fail_no_subcommand: bool = True,
    ) -> Namespace:
        """Common parsing code used by other parse methods.

        Args:
            cfg: The configuration object.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.
            skip_validation: Whether to skip validation of configuration.
            skip_required: Whether to skip check of required arguments.
            skip_subcommands: Whether to skip subcommand processing.
            fail_no_subcommand: Whether to fail if no subcommand given.

        Returns:
            A config object with all parsed values.
        """
        if env is None and self._default_env:
            env = True

        if not skip_subcommands:
            _ActionSubCommands.handle_subcommands(
                self, cfg, env=env, defaults=defaults, fail_no_subcommand=fail_no_subcommand
            )

        if defaults:
            with parser_context(lenient_check=True):
                ActionTypeHint.add_sub_defaults(self, cfg)

        with parser_context(parent_parser=self):
            if not lenient_check.get() and self.parser_mode == "omegaconf+":
                cfg = omegaconf_apply(self, cfg)

            _ActionPrintConfig.print_config_if_requested(self, cfg)

            try:
                ActionLink.apply_parsing_links(self, cfg)
            except Exception as ex:
                self.error(str(ex), ex)

            if not skip_validation:
                self.validate(cfg, skip_required=skip_required)

        if not (with_meta or (with_meta is None and self._default_meta)):
            cfg = strip_meta(cfg)

        return cfg

    def _parse_defaults_and_environ(
        self,
        defaults: bool = True,
        env: Optional[bool] = None,
        environ: Optional[Union[Dict[str, str], os._Environ]] = None,
    ):
        cfg = Namespace()
        if defaults:
            cfg = self.get_defaults(skip_validation=True)

        if env or (env is None and self._default_env):
            if environ is None:
                environ = os.environ
            with parser_context(load_value_mode=self.parser_mode):
                cfg_env = self._load_env_vars(env=environ, defaults=defaults)
            cfg = self.merge_config(cfg_env, cfg)

        return cfg

    def parse_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[Namespace] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
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
            ArgumentError: If the parsing fails error and exit_on_error=True.
        """
        skip_validation = get_private_kwargs(kwargs, _skip_validation=False)
        return_parser_if_captured(self)
        handle_completions(self)

        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)
            if not all(isinstance(a, str) for a in args):
                self.error(f"All arguments are expected to be strings: {args}")
        self.args = args

        try:
            cfg = self._parse_defaults_and_environ(defaults, env)
            if namespace:
                cfg = self.merge_config(namespace, cfg)

            with _ActionSubCommands.parse_kwargs_context({"env": env, "defaults": defaults}):
                cfg, unk = self.parse_known_args(args=args, namespace=cfg)
                cfg, unk = self._positional_optionals(cfg, unk)
            if unk:
                self.error(f"Unrecognized arguments: {' '.join(unk)}")

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_validation=skip_validation,
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        self._logger.debug("Parsed command line arguments: %s", args)
        return parsed_cfg

    def parse_object(
        self,
        cfg_obj: Union[Namespace, Dict[str, Any]],
        cfg_base: Optional[Namespace] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
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
            ArgumentError: If the parsing fails error and exit_on_error=True.
        """
        skip_validation, skip_required = get_private_kwargs(kwargs, _skip_validation=False, _skip_required=False)

        try:
            cfg = self._parse_defaults_and_environ(defaults, env)
            if cfg_base:
                cfg = self.merge_config(cfg_base, cfg)

            cfg = self._apply_actions(cfg)
            cfg_apply = self._apply_actions(cfg_obj, prev_cfg=cfg)
            cfg = self.merge_config(cfg_apply, cfg)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_validation=skip_validation,
                skip_required=skip_required,
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        self._logger.debug("Parsed object: %s", cfg_obj)
        return parsed_cfg

    def _load_env_vars(self, env: Union[Dict[str, str], os._Environ], defaults: bool) -> Namespace:
        cfg = Namespace()
        actions = filter_default_actions(self._actions)
        for action in actions:
            env_var = get_env_var(self, action)
            if env_var in env and isinstance(action, ActionConfigFile):
                ActionConfigFile.apply_config(self, cfg, action.dest, env[env_var])
        for action in actions:
            env_var = get_env_var(self, action)
            if env_var in env and isinstance(action, _ActionSubCommands):
                env_val = env[env_var]
                if env_val in action.choices:
                    cfg[action.dest] = subcommand = self._check_value_key(action, env_val, action.dest, cfg)
                    pcfg = action._name_parser_map[env_val].parse_env(env=env, defaults=defaults, _skip_validation=True)
                    for k, v in vars(pcfg).items():
                        cfg[subcommand + "." + k] = v
        for action in actions:
            env_var = get_env_var(self, action)
            if env_var in env and not isinstance(action, (ActionConfigFile, _ActionSubCommands)):
                env_val = env[env_var]
                if _is_action_value_list(action):
                    try:
                        list_env_val = load_value(env_val)
                        env_val = list_env_val if isinstance(list_env_val, list) else [env_val]
                    except get_loader_exceptions():
                        env_val = [env_val]
                cfg[action.dest] = self._check_value_key(action, env_val, action.dest, cfg)
        self._apply_actions(cfg)
        return cfg

    def parse_env(
        self,
        env: Optional[Dict[str, str]] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
    ) -> Namespace:
        """Parses environment variables.

        Args:
            env: The environment object to use, if None `os.environ` is used.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ArgumentError: If the parsing fails error and exit_on_error=True.
        """
        skip_validation, skip_subcommands = get_private_kwargs(kwargs, _skip_validation=False, _skip_subcommands=False)

        try:
            cfg = self._parse_defaults_and_environ(defaults, env=True, environ=env)

            kwargs = {
                "env": True,
                "defaults": defaults,
                "with_meta": with_meta,
                "skip_validation": skip_validation,
                "skip_subcommands": skip_subcommands,
            }
            if skip_validation:
                kwargs["fail_no_subcommand"] = False

            parsed_cfg = self._parse_common(cfg=cfg, **kwargs)

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        self._logger.debug("Parsed environment variables")
        return parsed_cfg

    def parse_path(
        self,
        cfg_path: Union[str, os.PathLike],
        ext_vars: Optional[dict] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
    ) -> Namespace:
        """Parses a configuration file given its path.

        Args:
            cfg_path: Path to the configuration file to parse.
            ext_vars: Optional external variables used for parsing jsonnet.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ArgumentError: If the parsing fails error and exit_on_error=True.
        """
        fpath = Path(cfg_path, mode=_get_config_read_mode())
        with change_to_path_dir(fpath):
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(
                cfg_str,
                os.path.basename(cfg_path),
                ext_vars,
                env,
                defaults,
                with_meta,
                **kwargs,
            )

        self._logger.debug("Parsed configuration from path: %s", cfg_path)
        return parsed_cfg

    def parse_string(
        self,
        cfg_str: str,
        cfg_path: Union[str, os.PathLike] = "",
        ext_vars: Optional[dict] = None,
        env: Optional[bool] = None,
        defaults: bool = True,
        with_meta: Optional[bool] = None,
        **kwargs,
    ) -> Namespace:
        """Parses configuration given as a string.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ArgumentError: If the parsing fails error and exit_on_error=True.
        """
        skip_validation, fail_no_subcommand = get_private_kwargs(
            kwargs, _skip_validation=False, _fail_no_subcommand=True
        )

        try:
            with parser_context(load_value_mode=self.parser_mode):
                cfg = self._load_config_parser_mode(cfg_str, cfg_path, ext_vars, previous_config.get())

            if defaults or env:
                cfg_base = self._parse_defaults_and_environ(defaults, env)
                cfg = self.merge_config(cfg, cfg_base)

            parsed_cfg = self._parse_common(
                cfg=cfg,
                env=env,
                defaults=defaults,
                with_meta=with_meta,
                skip_validation=skip_validation,
                fail_no_subcommand=fail_no_subcommand,
            )

        except (TypeError, KeyError) as ex:
            self.error(str(ex), ex)

        self._logger.debug("Parsed %s string: %s", self.parser_mode, cfg_str)
        return parsed_cfg

    def _load_config_parser_mode(
        self,
        cfg_str: str,
        cfg_path: Union[str, os.PathLike] = "",
        ext_vars: Optional[dict] = None,
        prev_cfg: Optional[Namespace] = None,
        key: Optional[str] = None,
    ) -> Namespace:
        """Loads a configuration string into a namespace.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        try:
            cfg_dict = load_value(cfg_str, path=cfg_path, ext_vars=ext_vars)
        except get_loader_exceptions() as ex:
            raise TypeError(f"Problems parsing config: {ex}") from ex
        if key and isinstance(cfg_dict, dict):
            cfg_dict = cfg_dict.get(key, {})
        if not isinstance(cfg_dict, dict):
            raise TypeError(f"Unexpected config: {cfg_str}")
        return self._apply_actions(cfg_dict, prev_cfg=prev_cfg)

    ## Methods for adding to the parser ##

    def add_subparsers(self, **kwargs) -> NoReturn:
        """Raises a NotImplementedError since jsonargparse uses add_subcommands."""
        raise NotImplementedError("In jsonargparse sub-commands are added using the add_subcommands method.")

    def add_subcommands(self, required: bool = True, dest: str = "subcommand", **kwargs) -> _ActionSubCommands:
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
        if "description" not in kwargs:
            kwargs["description"] = "For more details of each subcommand, add it as an argument followed by --help."
        default_config_files = self.default_config_files
        self.default_config_files = []
        subcommands: _ActionSubCommands = super().add_subparsers(dest=dest, **kwargs)  # type: ignore[assignment]
        self.default_config_files = default_config_files
        if required:
            self.required_args.add(dest)
        subcommands._required = required  # type: ignore[attr-defined]
        subcommands.required = False
        subcommands.parent_parser = self
        subcommands.env_prefix = get_env_var(self)
        self._subcommands_action = subcommands
        return subcommands

    ## Methods for serializing config objects ##

    def dump(
        self,
        cfg: Namespace,
        format: str = "parser_mode",
        skip_none: bool = True,
        skip_default: bool = False,
        skip_validation: bool = False,
        yaml_comments: bool = False,
        skip_link_targets: bool = True,
        **kwargs,
    ) -> str:
        """Generates a yaml or json string for the given configuration object.

        Args:
            cfg: The configuration object to dump.
            format: The output format: ``'yaml'``, ``'json'``, ``'json_indented'``, ``'parser_mode'`` or ones added via
                :func:`.set_dumper`.
            skip_none: Whether to exclude entries whose value is None.
            skip_default: Whether to exclude entries whose value is the same as the default.
            skip_validation: Whether to skip parser checking.
            yaml_comments: Whether to add help content as comments. ``yaml_comments=True`` implies ``format='yaml'``.
            skip_link_targets: Whether to exclude link targets.

        Returns:
            The configuration in yaml or json format.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        skip_validation = deprecated_skip_check(ArgumentParser.dump, kwargs, skip_validation)
        check_valid_dump_format(format)

        cfg = strip_meta(cfg)

        with parser_context(load_value_mode=self.parser_mode):
            if not skip_validation:
                self.validate(cfg)

            if skip_link_targets:
                ActionLink.strip_link_target_keys(self, cfg)

            dump_kwargs = {"skip_validation": skip_validation, "skip_none": skip_none}
            self._dump_cleanup_actions(cfg, self._actions, dump_kwargs)

            cfg_dict = cfg.as_dict()

            if skip_default:
                defaults = self.get_defaults(skip_validation=True)
                ActionLink.strip_link_target_keys(self, defaults)
                self._dump_cleanup_actions(defaults, self._actions, {"skip_validation": True, "skip_none": skip_none})
                self._dump_delete_default_entries(cfg_dict, defaults.as_dict())

        with parser_context(parent_parser=self):
            return dump_using_format(self, cfg_dict, "yaml_comments" if yaml_comments else format)

    def _dump_cleanup_actions(self, cfg, actions, dump_kwargs, prefix=""):
        skip_none = dump_kwargs["skip_none"]
        for action in filter_default_actions(actions):
            action_dest = prefix + action.dest
            if (
                (action.help == argparse.SUPPRESS and not isinstance(action, _ActionConfigLoad))
                or isinstance(action, ActionConfigFile)
                or (skip_none and action_dest in cfg and cfg[action_dest] is None)
            ):
                cfg.pop(action_dest, None)
            elif isinstance(action, _ActionSubCommands):
                cfg.pop(action_dest, None)
                for key, subparser in action.choices.items():
                    self._dump_cleanup_actions(cfg, subparser._actions, dump_kwargs, prefix=prefix + key + ".")
            elif isinstance(action, ActionLink):
                action = action.target[1]
            if isinstance(action, ActionTypeHint):
                value = cfg.get(action_dest)
                if value is not None:
                    with parser_context(parent_parser=self, lenient_check=True):
                        if dump_kwargs.get("skip_validation"):
                            with suppress(ValueError):
                                value = action.serialize(value, dump_kwargs=dump_kwargs)
                        else:
                            value = action.serialize(value, dump_kwargs=dump_kwargs)
                    cfg.update(value, action_dest)

    def _dump_delete_default_entries(self, subcfg, subdefaults):
        for key in list(subcfg.keys()):
            if key in subdefaults:
                val = subcfg[key]
                default = subdefaults[key]
                class_object_val = None
                if is_subclass_spec(val):
                    if val["class_path"] != default.get("class_path"):
                        with parser_context(parent_parser=self):
                            parser = ActionTypeHint.get_class_parser(val["class_path"])
                        default = {"init_args": parser.get_defaults().as_dict()}
                    class_object_val = val
                    val = val.get("init_args")
                    default = default.get("init_args")
                if val == default:
                    del subcfg[key]
                elif isinstance(val, dict) and isinstance(default, dict):
                    self._dump_delete_default_entries(val, default)
                    if class_object_val and class_object_val.get("init_args") == {}:
                        del class_object_val["init_args"]

    def save(
        self,
        cfg: Namespace,
        path: Union[str, os.PathLike],
        format: str = "parser_mode",
        skip_none: bool = True,
        skip_validation: bool = False,
        overwrite: bool = False,
        multifile: bool = True,
        branch: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Writes to file(s) the yaml or json for the given configuration object.

        Args:
            cfg: The configuration object to save.
            path: Path to the location where to save config.
            format: The output format: ``'yaml'``, ``'json'``, ``'json_indented'``, ``'parser_mode'`` or ones added via
                :func:`.set_dumper`.
            skip_none: Whether to exclude entries whose value is None.
            skip_validation: Whether to skip parser checking.
            overwrite: Whether to overwrite existing files.
            multifile: Whether to save multiple config files by using the __path__ metas.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        skip_validation = deprecated_skip_check(ArgumentParser.save, kwargs, skip_validation)
        check_valid_dump_format(format)

        def check_overwrite(path):
            if not overwrite and os.path.isfile(path.absolute):
                raise ValueError(f"Refusing to overwrite existing file: {path.absolute}")

        dump_kwargs = {"format": format, "skip_none": skip_none, "skip_validation": skip_validation}

        if fsspec_support:
            try:
                path_sw = Path(path, mode="sw")
            except TypeError:
                pass
            else:
                if path_sw.is_fsspec:
                    if multifile:
                        raise NotImplementedError(f"multifile=True not supported for fsspec paths: {path}")
                    fsspec = import_fsspec("ArgumentParser.save")
                    with fsspec.open(path, "w") as f:
                        f.write(self.dump(cfg, **dump_kwargs))  # type: ignore[arg-type]
                    return

        path_fc = Path(path, mode="fc")
        check_overwrite(path_fc)

        if not multifile:
            with open(path_fc.absolute, "w") as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore[arg-type]

        else:
            cfg = cfg.clone()

            if not skip_validation:
                with parser_context(load_value_mode=self.parser_mode):
                    self.validate(strip_meta(cfg), branch=branch)

            ActionLink.strip_link_target_keys(self, cfg)

            def save_paths(cfg):
                for key in cfg.get_sorted_keys():
                    val = cfg[key]
                    if isinstance(val, (Namespace, dict)) and "__path__" in val:
                        action = _find_action(self, key)
                        if isinstance(action, (ActionJsonSchema, ActionJsonnet, ActionTypeHint, _ActionConfigLoad)):
                            val_path = Path(os.path.basename(val["__path__"].absolute), mode="fc")
                            check_overwrite(val_path)
                            val_out = strip_meta(val)
                            if isinstance(val, Namespace):
                                val_out = val_out.as_dict()
                            if "__orig__" in val:
                                val_str = val["__orig__"]
                            else:
                                is_json = str(val_path).lower().endswith(".json")
                                val_str = dump_using_format(self, val_out, "json_indented" if is_json else format)
                            with open(val_path.absolute, "w") as f:
                                f.write(val_str)
                            cfg[key] = os.path.basename(val_path.absolute)
                    elif isinstance(val, Path) and key in self.save_path_content and "r" in val.mode:
                        val_path = Path(os.path.basename(val.absolute), mode="fc")
                        check_overwrite(val_path)
                        with open(val_path.absolute, "w") as f:
                            f.write(val.get_content())
                        cfg[key] = type(val)(str(val_path))

            with change_to_path_dir(path_fc), parser_context(parent_parser=self):
                save_paths(cfg)
            dump_kwargs["skip_validation"] = True
            with open(path_fc.absolute, "w") as f:
                f.write(self.dump(cfg, **dump_kwargs))  # type: ignore[arg-type]

    ## Methods related to defaults ##

    def _get_default_config_files(self) -> List[Tuple[Optional[str], Path]]:
        if getattr(self, "_inner_parser", False):
            return []

        default_config_files = []

        for key, parser in parent_parsers.get():
            for pattern in parser.default_config_files:
                files = sorted(glob.glob(os.path.expanduser(pattern)))
                default_config_files += [(key, v) for v in files]

        for pattern in self.default_config_files:
            files = sorted(glob.glob(os.path.expanduser(pattern)))
            default_config_files += [(None, x) for x in files]

        if len(default_config_files) > 0:
            with suppress(TypeError):
                return [(k, Path(v, mode=_get_config_read_mode())) for k, v in default_config_files]
        return []

    def get_default(self, dest: str) -> Any:
        """Gets a single default value for the given destination key.

        Args:
            dest: Destination key from which to get the default.

        Raises:
            KeyError: If key or its default not defined in the parser.
        """
        action, _ = _find_parent_action_and_subcommand(self, dest)
        if action is None or dest != action.dest:
            raise NSKeyError(f'No action for key "{dest}" to get its default.')

        def check_suppressed_default():
            if action.default == argparse.SUPPRESS:
                raise NSKeyError(f'Action for key "{dest}" does not specify a default.')

        if not self._get_default_config_files():
            check_suppressed_default()
            return action.default

        defaults = self.get_defaults()
        if action.dest not in defaults:
            check_suppressed_default()
        return defaults.get(action.dest)

    def get_defaults(self, skip_validation: bool = False, **kwargs) -> Namespace:
        """Returns a namespace with all default values.

        Args:
            skip_validation: Whether to skip validation of defaults.

        Returns:
            An object with all default values as attributes.
        """
        skip_validation = deprecated_skip_check(ArgumentParser.get_defaults, kwargs, skip_validation)
        cfg = Namespace()
        for action in filter_default_actions(self._actions):
            if (
                action.default != argparse.SUPPRESS
                and action.dest != argparse.SUPPRESS
                and not isinstance(action.default, UnknownDefault)
            ):
                cfg[action.dest] = recreate_branches(action.default)

        self._logger.debug("Loaded parser defaults: %s", cfg)

        default_config_files = self._get_default_config_files()
        for key, default_config_file in default_config_files:
            default_config_file_content = default_config_file.get_content()
            if not default_config_file_content.strip():
                continue
            with change_to_path_dir(default_config_file), parser_context(parent_parser=self):
                cfg_file = self._load_config_parser_mode(default_config_file_content, key=key, prev_cfg=cfg)
                cfg = self.merge_config(cfg_file, cfg)
                try:
                    with _ActionPrintConfig.skip_print_config():
                        cfg = self._parse_common(
                            cfg=cfg,
                            env=False,
                            defaults=False,
                            with_meta=None,
                            skip_validation=skip_validation,
                            skip_required=True,
                        )
                except (TypeError, KeyError, argparse.ArgumentError) as ex:
                    raise argument_error(
                        f'Problem in default config file "{default_config_file}": {ex.args[0]}'
                    ) from ex
            meta = cfg.get("__default_config__")
            if isinstance(meta, list):
                meta.append(default_config_file)
            elif isinstance(meta, Path):
                cfg["__default_config__"] = [meta, default_config_file]
            else:
                cfg["__default_config__"] = default_config_file
            self._logger.debug("Parsed default configuration from path: %s", default_config_file)

        ActionTypeHint.add_sub_defaults(self, cfg)

        return cfg

    ## Other methods ##

    def error(self, message: str, ex: Optional[Exception] = None) -> NoReturn:
        """Logs error message if a logger is set and exits or raises an ArgumentError."""
        self._logger.error(message)
        if callable(self._error_handler):
            self._error_handler(self, message)
        if not self.exit_on_error:
            raise argument_error(message) from ex
        elif debug_mode_active():
            self._logger.debug("Debug enabled, thus raising exception instead of exit.")
            raise argument_error(message) from ex
        self.print_usage(sys.stderr)
        sys.stderr.write(f"error: {message}\n")
        self.exit(2)

    def validate(
        self,
        cfg: Namespace,
        skip_none: bool = True,
        skip_required: bool = False,
        branch: Optional[str] = None,
        **kwargs,
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
        prefix = get_private_kwargs(kwargs, _prefix="")
        cfg = ccfg = cfg.clone()
        if isinstance(branch, str):
            branch_cfg = cfg
            cfg = Namespace()
            cfg[branch] = branch_cfg

        def check_required(cfg, parser, prefix):
            for reqkey in parser.required_args:
                try:
                    val = cfg[reqkey]
                    if val is None:
                        raise TypeError
                except (KeyError, TypeError) as ex:
                    raise TypeError(
                        f'Key "{prefix}{reqkey}" is required but not included in config object or its value is None.'
                    ) from ex
            subcommand, subparser = _ActionSubCommands.get_subcommand(parser, cfg, fail_no_subcommand=False)
            if subcommand is not None and subparser is not None:
                check_required(cfg.get(subcommand), subparser, prefix + subcommand + ".")

        def check_values(cfg):
            sorted_keys = {k: _find_action(self, k) for k in cfg.get_sorted_keys()}
            for key, action in sorted_keys.items():
                parent_action = None
                if action is None:
                    if _is_branch_key(self, key):
                        continue
                    parent_action, subcommand = _find_parent_action_and_subcommand(self, key, exclude=_ActionConfigLoad)
                    if parent_action:
                        parent_key = subcommand + "." + parent_action.dest if subcommand else parent_action.dest
                        if key.startswith(parent_key + ".") and sorted_keys.get(parent_key) is parent_action:
                            # only check action once with entire value
                            continue
                val = cfg[key]
                if action is not None:
                    if (val is None and skip_none) or lenient_check.get():
                        continue
                    try:
                        self._check_value_key(action, val, key, ccfg)
                    except TypeError as ex:
                        if not (
                            val == {} and ActionTypeHint.is_subclass_typehint(action) and key not in self.required_args
                        ):
                            raise ex
                else:
                    if isinstance(parent_action, _ActionSubCommands) and "." in key:
                        subcommand, subkey = split_key_root(key)
                        raise NSKeyError(f"Subcommand '{subcommand}' does not accept nested key '{subkey}'")
                    group_key = next((g for g in self.groups if key.startswith(g + ".")), None)
                    if group_key:
                        subkey = key[len(group_key) + 1 :]
                        raise NSKeyError(f"Group '{group_key}' does not accept nested key '{subkey}'")
                    raise NSKeyError(f"Key '{key}' is not expected")

        try:
            with parser_context(load_value_mode=self.parser_mode):
                check_values(cfg)
            if not skip_required and not lenient_check.get():
                check_required(cfg, self, prefix)
        except (TypeError, KeyError) as ex:
            prefix = "Validation failed: "
            message = ex.args[0]
            if prefix not in message:
                message = prefix + message
            raise type(ex)(message) from ex

    def add_instantiator(
        self,
        instantiator: InstantiatorCallable,
        class_type: Type[ClassType],
        subclasses: bool = True,
        prepend: bool = False,
    ) -> None:
        """Adds a custom instantiator for a class type. Used by ``instantiate_classes``.

        Instantiator functions are expected to have as signature ``(class_type:
        Type[ClassType], *args, **kwargs) -> ClassType``.

        For reference, the default instantiator is ``return class_type(*args,
        **kwargs)``.

        In some use cases, the instantiator function might need access to values
        applied by instantiation links. For this, the instantiator function can
        have an additional keyword parameter ``applied_instantiation_links:
        dict``. This parameter will be populated with a dictionary having as
        keys the targets of the instantiation links and corresponding values
        that were applied. Support for ``applied_instantiation_links`` parameter
        is EXPERIMENTAL and subject to change or removal in future versions.

        Args:
            instantiator: Function that instantiates a class.
            class_type: The class type to instantiate.
            subclasses: Whether to instantiate subclasses of ``class_type``.
            prepend: Whether to prepend the instantiator to the existing instantiators.
        """
        if self._instantiators is None:
            self._instantiators = {}
        key = (class_type, subclasses)
        instantiators = {k: v for k, v in self._instantiators.items() if k != key}
        if prepend:
            self._instantiators = {key: instantiator, **instantiators}
        else:
            instantiators[key] = instantiator
            self._instantiators = instantiators

    def _get_instantiators(self):
        instantiators = self._instantiators or {}
        if hasattr(self, "parent_parser"):
            parent_instantiators = self.parent_parser._get_instantiators()
            instantiators = instantiators.copy()
            instantiators.update({k: v for k, v in parent_instantiators.items() if k not in instantiators})
        context_instantiators = class_instantiators.get()
        if context_instantiators:
            instantiators = instantiators.copy()
            instantiators.update({k: v for k, v in context_instantiators.items() if k not in instantiators})
        return instantiators

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
        components: List[Union[ActionTypeHint, _ActionConfigLoad, ArgumentGroup]] = []
        for action in filter_default_actions(self._actions):
            if isinstance(action, ActionTypeHint) or (
                isinstance(action, _ActionConfigLoad) and is_dataclass_like(action.basetype)
            ):
                components.append(action)

        if instantiate_groups:
            skip = {c.dest for c in components}
            groups = [g for g in self._action_groups if hasattr(g, "instantiate_class") and g.dest not in skip]
            components.extend(groups)

        components.sort(key=lambda x: -len(split_key(x.dest)))  # type: ignore[arg-type]
        order = ActionLink.instantiation_order(self)
        components = ActionLink.reorder(order, components)

        cfg = strip_meta(cfg)
        for component in components:
            ActionLink.apply_instantiation_links(self, cfg, target=component.dest)
            if isinstance(component, ActionTypeHint):
                try:
                    value, parent, key = cfg.get_value_and_parent(component.dest)
                except (KeyError, AttributeError):
                    pass
                else:
                    if value is not None:
                        with parser_context(
                            parent_parser=self,
                            nested_links=ActionLink.get_nested_links(self, component),
                            class_instantiators=self._get_instantiators(),
                            applied_instantiation_links=cfg.get("__applied_instantiation_links__"),
                        ):
                            parent[key] = component.instantiate_classes(value)
            else:
                with parser_context(
                    load_value_mode=self.parser_mode,
                    class_instantiators=self._get_instantiators(),
                    applied_instantiation_links=cfg.get("__applied_instantiation_links__"),
                ):
                    component.instantiate_class(component, cfg)

        ActionLink.apply_instantiation_links(self, cfg, order=order)

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
        cfg = cfg.clone()

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
        if "__default_config__" in cfg:
            cfg_files.append(cfg["__default_config__"])
        for action in filter_default_actions(self._actions):
            if isinstance(action, ActionConfigFile) and action.dest in cfg and cfg[action.dest] is not None:
                cfg_files.extend(p for p in cfg[action.dest] if p is not None)
        return cfg_files

    def format_help(self) -> str:
        defaults = None
        if len(self._default_config_files) > 0:
            note = "no existing default config file found."
            try:
                defaults = self.get_defaults()
                if "__default_config__" in defaults:
                    config_files = defaults["__default_config__"]
                    if isinstance(config_files, list):
                        config_files = [str(x) for x in config_files]
                    note = f"default values below are the ones overridden by the contents of: {config_files}"
            except argparse.ArgumentError as ex:
                note = f"tried getting defaults considering default_config_files but failed due to: {ex}"
            group = self._default_config_files_group
            group.description = f"{self._default_config_files}, Note: {note}"
        with parser_context(parent_parser=self, defaults_cache=defaults):
            help_str = super().format_help()
        return help_str

    def print_usage(self, *args, **kwargs) -> None:
        with parser_context(parent_parser=self):
            return super().print_usage(*args, **kwargs)

    def _apply_actions(
        self,
        cfg: Union[Namespace, Dict[str, Any]],
        parent_key: str = "",
        prev_cfg: Optional[Namespace] = None,
        skip_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Namespace:
        """Runs _check_value_key on actions present in config."""
        if isinstance(cfg, dict):
            cfg = Namespace(cfg)
        if parent_key:
            cfg_branch = cfg
            cfg = Namespace()
            cfg[parent_key] = cfg_branch
            keys = [parent_key + "." + k for k in cfg_branch.__dict__]
        else:
            keys = list(cfg.__dict__)

        if prev_cfg:
            prev_cfg = prev_cfg.clone()
        else:
            prev_cfg = Namespace()

        config_keys: Set[str] = set()
        num = 0
        while num < len(keys):
            key = keys[num]
            exclude = _ActionConfigLoad if key in config_keys else None
            action, subcommand = _find_action_and_subcommand(self, key, exclude=exclude)

            if isinstance(action, ActionJsonnet):
                ext_vars_key = action._ext_vars
                if ext_vars_key and ext_vars_key not in keys[:num]:
                    keys = keys[:num] + [ext_vars_key] + [k for k in keys[num:] if k != ext_vars_key]
                    continue

            num += 1

            if action is None or isinstance(action, _ActionSubCommands):
                value = cfg[key]
                if isinstance(value, dict):
                    value = Namespace(value)
                if isinstance(value, Namespace):
                    new_keys = value.__dict__.keys()
                    keys += [key + "." + k for k in new_keys if key + "." + k not in keys]
                cfg[key] = value
                continue

            action_dest = action.dest if subcommand is None else subcommand + "." + action.dest
            append = False
            if action_dest not in cfg and key.endswith("+"):
                append = True
                cfg[action_dest] = cfg.pop(key)
            value = cfg[action_dest]
            if skip_fn and skip_fn(value):
                continue
            with parser_context(parent_parser=self, lenient_check=True):
                value = self._check_value_key(action, value, action_dest, prev_cfg, append=append)
            if isinstance(action, _ActionConfigLoad):
                config_keys.add(action_dest)
                keys.append(action_dest)
            elif isinstance(action, ActionConfigFile):
                if isinstance(value, str):
                    cfg.pop(action_dest)
                    preserve = Namespace({k: cfg[k] for k in keys[num:]})
                    ActionConfigFile.apply_config(self, cfg, action_dest, value)
                    cfg.update(preserve)
                    continue
            elif getattr(action, "jsonnet_ext_vars", False):
                prev_cfg[action_dest] = value
            cfg[action_dest] = value
        return cfg[parent_key] if parent_key else cfg

    def merge_config(self, cfg_from: Namespace, cfg_to: Namespace) -> Namespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from: The configuration from which to merge.
            cfg_to: The configuration into which to merge.

        Returns:
            A new object with the merged configuration.
        """
        cfg_from = cfg_from.clone()
        cfg_to = cfg_to.clone()
        with parser_context(parent_parser=self):
            ActionTypeHint.discard_init_args_on_class_path_change(self, cfg_to, cfg_from)
        cfg_to.update(cfg_from)
        return cfg_to

    def _check_value_key(
        self, action: argparse.Action, value: Any, key: str, cfg: Optional[Namespace], append: bool = False
    ) -> Any:
        """Checks the value for a given action.

        Args:
            action: The action used for parsing.
            value: The value to parse.
            key: The configuration key.

        Raises:
            TypeError: If the value is not valid.
        """
        if value is None and lenient_check.get():
            return value
        is_subcommand = isinstance(action, _ActionSubCommands)
        if is_subcommand and action.choices:
            leaf_key = split_key_leaf(key)[-1]
            if leaf_key == action.dest:
                return value
            subparser = action._name_parser_map[leaf_key]  # type: ignore[attr-defined]
            subparser.validate(value, _prefix=key + ".")
        elif isinstance(action, _ActionConfigLoad):
            if isinstance(value, str):
                value = action.check_type(value, self)
        elif hasattr(action, "_check_type"):
            with parser_context(parent_parser=self):
                value = action._check_type_(value, cfg=cfg, append=append, mode=self.parser_mode)  # type: ignore[attr-defined]
        elif action.type is not None:
            try:
                if action.nargs in {None, "?"} or action.nargs == 0:
                    value = action.type(value)  # type: ignore[operator]
                elif value is not None:
                    for k, v in enumerate(value):
                        value[k] = action.type(v)  # type: ignore[operator]
            except (TypeError, ValueError) as ex:
                raise TypeError(f'Parser key "{key}": {ex}') from ex
        if not is_subcommand and action.choices:
            vals = value if _is_action_value_list(action) else [value]
            assert isinstance(vals, list)
            for val in vals:
                if val not in action.choices:
                    raise TypeError(f'Parser key "{key}": {val!r} not among choices {action.choices}')
        return value

    ## Properties ##

    @property
    def default_config_files(self) -> List[str]:
        """Default config file locations.

        :getter: Returns the current default config file locations.
        :setter: Sets new default config file locations, e.g. ``['~/.config/myapp/*.yaml']``.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._default_config_files

    @default_config_files.setter
    def default_config_files(self, default_config_files: Optional[Sequence[Union[str, os.PathLike]]]):
        if default_config_files is None:
            self._default_config_files = []
        elif isinstance(default_config_files, list) and all(
            isinstance(x, (str, os.PathLike)) for x in default_config_files
        ):
            self._default_config_files = [os.fspath(d) for d in default_config_files]
        else:
            raise ValueError("default_config_files expects None or List[str | os.PathLike].")

        if len(self._default_config_files) > 0:
            if not hasattr(self, "_default_config_files_group"):
                group_title = "default config file locations"
                group_class = getattr(self, "_group_class", ArgumentGroup)
                group = group_class(self, title=group_title)
                self._action_groups = [group] + self._action_groups  # type: ignore[operator]
                self._default_config_files_group = group
        elif hasattr(self, "_default_config_files_group"):
            self._action_groups = [g for g in self._action_groups if g != self._default_config_files_group]
            delattr(self, "_default_config_files_group")

    @property
    def default_env(self) -> bool:
        """Whether by default environment variables parsing is enabled.

        If the JSONARGPARSE_DEFAULT_ENV environment variable is set to true or
        false, that value will take precedence.

        :getter: Returns the current default environment variables parsing setting.
        :setter: Sets the default environment variables parsing setting.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._default_env

    @default_env.setter
    def default_env(self, default_env: bool):
        os_default_env = os.getenv("JSONARGPARSE_DEFAULT_ENV", "").lower()
        if os_default_env in {"true", "false"}:
            self._default_env = os_default_env == "true"
        elif isinstance(default_env, bool):
            self._default_env = default_env
        else:
            raise ValueError("default_env expects a boolean.")
        if self._subcommands_action:
            for subparser in self._subcommands_action._name_parser_map.values():
                subparser.default_env = self._default_env

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
    def default_meta(self, default_meta: bool):
        if isinstance(default_meta, bool):
            self._default_meta = default_meta
        else:
            raise ValueError("default_meta expects a boolean.")

    @property
    def env_prefix(self) -> Union[bool, str]:
        """The environment variables prefix property.

        :getter: Returns the current environment variables prefix.
        :setter: Sets the environment variables prefix.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._env_prefix

    @env_prefix.setter
    def env_prefix(self, env_prefix: Union[bool, str]):
        if env_prefix is None:
            from ._deprecated import (
                deprecation_warning,
                env_prefix_property_none_message,
            )

            deprecation_warning(ArgumentParser, env_prefix_property_none_message, stacklevel=3)
            env_prefix = False
        elif env_prefix is True:
            env_prefix = os.path.splitext(self.prog)[0]
        elif not isinstance(env_prefix, (bool, str)):
            raise ValueError("env_prefix expects a string or a boolean.")
        self._env_prefix = env_prefix

    @property
    def parser_mode(self) -> str:
        """Mode for parsing configuration files: ``'yaml'``, ``'jsonnet'`` or ones added via :func:`.set_loader`.

        :getter: Returns the current parser mode.
        :setter: Sets the parser mode.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._parser_mode

    @parser_mode.setter
    def parser_mode(self, parser_mode: str):
        if parser_mode in {"omegaconf", "omegaconf+"}:
            set_omegaconf_loader(parser_mode)
        if parser_mode not in loaders:
            raise ValueError(f"The only accepted values for parser_mode are {set(loaders)}.")
        if parser_mode == "jsonnet":
            import_jsonnet("parser_mode=jsonnet")
        self._parser_mode = parser_mode
        if self._subcommands_action:
            for subparser in self._subcommands_action._name_parser_map.values():
                subparser.parser_mode = parser_mode

    @property
    def dump_header(self) -> Optional[List[str]]:
        """Header to include as comment when dumping a config object.

        :getter: Returns the current dump header.
        :setter: Sets the dump header.

        Raises:
            ValueError: If an invalid value is given.
        """
        return self._dump_header

    @dump_header.setter
    def dump_header(self, dump_header: Optional[List[str]]):
        if not (
            dump_header is None or (isinstance(dump_header, list) and all(isinstance(x, str) for x in dump_header))
        ):
            raise ValueError("Expected dump_header to be None or a list of strings.")
        self._dump_header = dump_header


from ._deprecated import parse_as_dict_patch  # noqa: E402

if "SPHINX_BUILD" not in os.environ:
    parse_as_dict_patch()
