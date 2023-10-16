"""Collection of useful actions to define arguments."""

import re
import sys
import warnings
from argparse import SUPPRESS, _HelpAction, _SubParsersAction
from argparse import Action as ArgparseAction
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ._common import get_class_instantiator, is_subclass, parser_context
from ._loaders_dumpers import get_loader_exceptions, load_value
from ._namespace import Namespace, NSKeyError, split_key, split_key_root
from ._optionals import FilesCompleterMethod, get_config_read_mode
from ._type_checking import ArgumentParser
from ._util import (
    LoggerProperty,
    NoneType,
    Path,
    argument_error,
    change_to_path_dir,
    default_config_option_help,
    get_typehint_origin,
    import_object,
    indent_text,
    iter_to_set_str,
    parse_value_or_config,
)

__all__ = [
    "ActionConfigFile",
    "ActionYesNo",
    "ActionParser",
]


class Action(LoggerProperty, ArgparseAction):
    """Base for jsonargparse Action classes."""


def _is_branch_key(parser, key: str) -> bool:
    root_key = split_key_root(key)[0]
    for action in filter_default_actions(parser._actions):
        if isinstance(action, _ActionSubCommands) and root_key in action._name_parser_map:
            subparser = action._name_parser_map[root_key]
            return _is_branch_key(subparser, split_key_root(key)[1])
        elif action.dest.startswith(key + "."):
            return True
    return False


def _find_action_and_subcommand(
    parser: "ArgumentParser",
    dest: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Tuple[Optional[ArgparseAction], Optional[str]]:
    """Finds an action in a parser given its destination key.

    Args:
        parser: A parser where to search.
        dest: The destination key to search with.

    Returns:
        The action if found, otherwise None.
    """
    actions = filter_default_actions(parser._actions)
    if exclude is not None:
        actions = [a for a in actions if not isinstance(a, exclude)]
    fallback_action = None
    for action in actions:
        if action.dest == dest:
            if isinstance(action, _ActionConfigLoad):
                fallback_action = action
            else:
                return action, None
        elif isinstance(action, _ActionSubCommands):
            if dest in action._name_parser_map:
                return action, None
            elif split_key_root(dest)[0] in action._name_parser_map:
                subcommand, subdest = split_key_root(dest)
                subparser = action._name_parser_map[subcommand]
                subaction, subsubcommand = _find_action_and_subcommand(subparser, subdest, exclude=exclude)
                if subsubcommand is not None:
                    subcommand += "." + subsubcommand
                return subaction, subcommand
    return fallback_action, None


def _find_action(
    parser: "ArgumentParser",
    dest: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Optional[ArgparseAction]:
    return _find_action_and_subcommand(parser, dest, exclude=exclude)[0]


def _find_parent_action_and_subcommand(
    parser: "ArgumentParser",
    key: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Tuple[Optional[ArgparseAction], Optional[str]]:
    action, subcommand = _find_action_and_subcommand(parser, key, exclude=exclude)
    if action is None and "." in key:
        parts = split_key(key)
        for n in reversed(range(len(parts) - 1)):
            action, subcommand = _find_action_and_subcommand(parser, ".".join(parts[: n + 1]), exclude=exclude)
            if action is not None:
                break
    return action, subcommand


def _find_parent_action(
    parser: "ArgumentParser",
    key: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Optional[ArgparseAction]:
    return _find_parent_action_and_subcommand(parser, key, exclude=exclude)[0]


def _is_action_value_list(action: ArgparseAction) -> bool:
    """Checks whether an action produces a list value.

    Args:
        action: An argparse action to check.

    Returns:
        bool: True if produces list otherwise False.
    """
    if action.nargs in {"*", "+"} or (isinstance(action.nargs, int) and action.nargs != 0):
        return True
    return False


def remove_actions(parser, types):
    def remove(actions):
        rm_actions = [a for a in actions if isinstance(a, types)]
        for action in rm_actions:
            actions.remove(action)

    remove(parser._actions)
    for action_group in parser._action_groups:
        remove(action_group._group_actions)


def filter_default_actions(actions):
    default = (_HelpAction, _ActionHelpClassPath, _ActionPrintConfig)
    if isinstance(actions, list):
        return [a for a in actions if not isinstance(a, default)]
    return {k: a for k, a in actions.items() if not isinstance(a, default)}


class ActionConfigFile(Action, FilesCompleterMethod):
    """Action to indicate that an argument is a configuration file or a configuration string."""

    def __init__(self, **kwargs):
        """Initializer for ActionConfigFile instance."""
        if "default" in kwargs:
            self.set_default_error()
        opt_name = kwargs["option_strings"]
        opt_name = opt_name[0] if len(opt_name) == 1 else [x for x in opt_name if x[0:2] == "--"][0]
        if "." in opt_name:
            raise ValueError("ActionConfigFile must be a top level option.")
        if "help" not in kwargs:
            kwargs["help"] = "Path to a configuration file."
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_config(parser, cfg, self.dest, values)

    @staticmethod
    def set_default_error():
        raise ValueError("ActionConfigFile does not accept a default, use default_config_files.")

    @staticmethod
    def _ensure_single_config_argument(container, action):
        if is_subclass(action, ActionConfigFile) and any(isinstance(a, ActionConfigFile) for a in container._actions):
            raise ValueError("A parser is only allowed to have a single ActionConfigFile argument.")

    @staticmethod
    def _add_print_config_argument(container, action):
        if isinstance(action, ActionConfigFile) and getattr(container, "_print_config", None) is not None:
            container.add_argument(container._print_config, action=_ActionPrintConfig)

    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
        from ._link_arguments import skip_apply_links

        with _ActionSubCommands.not_single_subcommand(), previous_config_context(cfg), skip_apply_links():
            kwargs = {"env": False, "defaults": False, "_skip_check": True, "_fail_no_subcommand": False}
            try:
                cfg_path: Optional[Path] = Path(value, mode=get_config_read_mode())
            except TypeError as ex_path:
                try:
                    if isinstance(load_value(value), str):
                        raise ex_path
                    cfg_path = None
                    cfg_file = parser.parse_string(value, **kwargs)
                except (TypeError,) + get_loader_exceptions() as ex_str:
                    raise TypeError(f'Parser key "{dest}": {ex_str}') from ex_str
            else:
                cfg_file = parser.parse_path(value, **kwargs)
            cfg_merged = parser.merge_config(cfg_file, cfg)
            cfg.__dict__.update(cfg_merged.__dict__)
            if cfg.get(dest) is None:
                cfg[dest] = []
            cfg[dest].append(cfg_path)


previous_config: ContextVar = ContextVar("previous_config", default=None)


@contextmanager
def previous_config_context(cfg):
    token = previous_config.set(cfg)
    try:
        yield
    finally:
        previous_config.reset(token)


print_config_skip: ContextVar = ContextVar("print_config_skip", default=False)


class _ActionPrintConfig(Action):
    def __init__(
        self,
        option_strings,
        dest=SUPPRESS,
        default=SUPPRESS,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=1,
            metavar="\b[=flags]",
            help=(
                "Print the configuration after applying all other arguments and exit. The optional "
                "flags customizes the output and are one or more keywords separated by comma. The "
                "supported flags are: comments, skip_default, skip_null."
            ),
        )

    def __call__(self, parser, namespace, value, option_string=None):
        kwargs = {"subparser": parser, "key": None, "skip_none": False, "skip_check": False}
        valid_flags = {"": None, "comments": "yaml_comments", "skip_default": "skip_default", "skip_null": "skip_none"}
        if value is not None:
            flags = value[0].split(",")
            invalid_flags = [f for f in flags if f not in valid_flags]
            if len(invalid_flags) > 0:
                raise argument_error(f'Invalid option "{invalid_flags[0]}" for {option_string}')
            for flag in [f for f in flags if f != ""]:
                kwargs[valid_flags[flag]] = True
        while hasattr(parser, "parent_parser"):
            kwargs["key"] = parser.subcommand if kwargs["key"] is None else parser.subcommand + "." + kwargs["key"]
            parser = parser.parent_parser
        parser.print_config = kwargs

    @staticmethod
    @contextmanager
    def skip_print_config():
        t = print_config_skip.set(True)
        try:
            yield
        finally:
            print_config_skip.reset(t)

    @staticmethod
    def print_config_if_requested(parser, cfg):
        if hasattr(parser, "print_config") and not print_config_skip.get():
            key = parser.print_config.pop("key")
            subparser = parser.print_config.pop("subparser")
            if key is not None:
                cfg = cfg[key]
            with parser_context(lenient_check=True):
                sys.stdout.write(subparser.dump(cfg, **parser.print_config))
            delattr(parser, "print_config")
            parser.exit()

    @staticmethod
    def is_print_config_requested(parser):
        while parser:
            if hasattr(parser, "print_config"):
                return True
            parser = getattr(parser, "parent_parser", None)
        return False


class _ActionConfigLoad(Action):
    def __init__(self, basetype: Optional[Type] = None, **kwargs):
        if len(kwargs) == 0:
            self._basetype = basetype
        else:
            self.basetype = kwargs.pop("_basetype", None)
            kwargs["metavar"] = "CONFIG"
            kwargs["help"] = default_config_option_help
            kwargs["default"] = SUPPRESS
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs["_basetype"] = self._basetype
            return _ActionConfigLoad(**kwargs)
        parser, namespace, value = args[:3]
        loaded_value = self._load_config(value, parser)
        if isinstance(namespace.get(self.dest), Namespace):
            loaded_value = parser.merge_config(
                Namespace({self.dest: loaded_value}), Namespace({self.dest: namespace[self.dest]})
            )[self.dest]
        namespace[self.dest] = loaded_value
        return None

    def _load_config(self, value, parser):
        try:
            cfg, cfg_path = parse_value_or_config(value)
            if not isinstance(cfg, dict):
                raise TypeError(f'Parser key "{self.dest}": Unable to load config "{value}"')
            with change_to_path_dir(cfg_path):
                cfg = parser._apply_actions(cfg, parent_key=self.dest)
            return cfg
        except (TypeError,) + get_loader_exceptions() as ex:
            str_ex = indent_text(f"- {ex}")
            raise TypeError(f'Parser key "{self.dest}":\nUnable to load config {value!r}\n{str_ex}') from ex

    def check_type(self, value, parser):
        return self._load_config(value, parser)

    def instantiate_classes(self, value):
        instantiator_fn = get_class_instantiator()
        return instantiator_fn(self.basetype, **value)


class _ActionHelpClassPath(Action):
    sub_add_kwargs: Dict[str, Any] = {}

    def __init__(self, baseclass=None, **kwargs):
        if baseclass is not None:
            if get_typehint_origin(baseclass) == Union:
                baseclasses = [c for c in baseclass.__args__ if c is not NoneType]
                if len(baseclasses) == 1:
                    baseclass = baseclasses[0]
            self._baseclass = baseclass
        else:
            self._baseclass = kwargs.pop("_baseclass")
            self.update_init_kwargs(kwargs)
            super().__init__(**kwargs)

    def update_init_kwargs(self, kwargs):
        if get_typehint_origin(self._baseclass) == Union:
            from ._typehints import ActionTypeHint

            self._basename = iter_to_set_str(
                c.__name__ for c in self._baseclass.__args__ if ActionTypeHint.is_subclass_typehint(c)
            )
        else:
            self._basename = self._baseclass.__name__
        kwargs.update(
            {
                "metavar": "CLASS_PATH_OR_NAME",
                "default": SUPPRESS,
                "help": f"Show the help for the given subclass of {self._basename} and exit.",
            }
        )

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs["_baseclass"] = self._baseclass
            return type(self)(**kwargs)
        dest = re.sub("\\.help$", "", self.dest)
        return self.print_help(args, self._baseclass, dest)

    def print_help(self, call_args, baseclass, dest):
        from ._typehints import resolve_class_path_by_name

        parser, _, value, option_string = call_args
        try:
            val_class = import_object(resolve_class_path_by_name(baseclass, value))
        except Exception as ex:
            raise TypeError(f"{option_string}: {ex}") from ex
        if get_typehint_origin(self._baseclass) == Union:
            baseclasses = self._baseclass.__args__
        else:
            baseclasses = [baseclass]
        if not any(is_subclass(val_class, b) for b in baseclasses):
            raise TypeError(f'{option_string}: Class "{value}" is not a subclass of {self._basename}')
        dest += ".init_args"
        subparser = type(parser)()
        subparser.add_class_arguments(val_class, dest, **self.sub_add_kwargs)
        remove_actions(subparser, (_HelpAction, _ActionPrintConfig, _ActionConfigLoad))
        args = self.get_args_after_opt(parser.args)
        if args:
            subparser.parse_args(args)
            raise argument_error(f"Expected a nested --*.help option, got: {args}.")
        else:
            subparser.print_help()
            parser.exit()

    def get_args_after_opt(self, args):
        opt_str = self.option_strings[0]
        for num, arg in enumerate(args):
            parts = arg.split("=", 1)
            if parts[0] == opt_str:
                if len(parts) == 1:
                    num += 1
                break
        return args[num + 1 :]


class ActionYesNo(Action):
    """Paired options --[yes_prefix]opt, --[no_prefix]opt to set True or False respectively."""

    def __init__(self, yes_prefix: str = "", no_prefix: str = "no_", **kwargs):
        """Initializer for ActionYesNo instance.

        Args:
            yes_prefix: Prefix for yes option.
            no_prefix: Prefix for no option.

        Raises:
            ValueError: If a parameter is invalid.
        """
        if len(kwargs) == 0:
            self._yes_prefix = yes_prefix
            self._no_prefix = no_prefix
        else:
            self._yes_prefix = kwargs.pop("_yes_prefix") if "_yes_prefix" in kwargs else ""
            self._no_prefix = kwargs.pop("_no_prefix") if "_no_prefix" in kwargs else "no_"
            if len(kwargs["option_strings"]) == 0:
                raise ValueError(f'{type(self).__name__} not intended for positional arguments  ({kwargs["dest"]}).')
            opt_name = kwargs["option_strings"][0]
            if not opt_name.startswith("--" + self._yes_prefix):
                raise ValueError(f'Expected option string to start with "--{self._yes_prefix}".')
            if self._no_prefix is not None:
                kwargs["option_strings"] += [re.sub("^--" + self._yes_prefix, "--" + self._no_prefix, opt_name)]
            if self._no_prefix is None and "nargs" in kwargs and kwargs["nargs"] != 1:
                raise ValueError("ActionYesNo with no_prefix=None only supports nargs=1.")
            if "nargs" in kwargs and kwargs["nargs"] in {"?", 1}:
                kwargs["metavar"] = "{true,yes,false,no}"
                if kwargs["nargs"] == 1:
                    kwargs["nargs"] = None
            else:
                kwargs["nargs"] = 0
                kwargs["metavar"] = None
            if "default" not in kwargs:
                kwargs["default"] = False
            kwargs["type"] = ActionYesNo._boolean_type
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Sets the corresponding key to True or False depending on the option string used."""
        if len(args) == 0:
            kwargs["_yes_prefix"] = self._yes_prefix
            kwargs["_no_prefix"] = self._no_prefix
            return ActionYesNo(**kwargs)
        value = args[2] if isinstance(args[2], bool) else True
        if self._no_prefix is not None and args[3].startswith("--" + self._no_prefix):
            setattr(args[1], self.dest, not value)
        else:
            setattr(args[1], self.dest, value)
        return None

    def _add_dest_prefix(self, prefix):
        self.dest = prefix + "." + self.dest
        self.option_strings[0] = re.sub(
            "^--" + self._yes_prefix, "--" + self._yes_prefix + prefix + ".", self.option_strings[0]
        )
        if self._no_prefix is not None:
            self.option_strings[-1] = re.sub(
                "^--" + self._no_prefix, "--" + self._no_prefix + prefix + ".", self.option_strings[-1]
            )

    def _check_type(self, value, cfg=None):
        return ActionYesNo._boolean_type(value)

    @staticmethod
    def _boolean_type(x):
        if isinstance(x, str) and x.lower() in {"true", "yes", "false", "no"}:
            x = True if x.lower() in {"true", "yes"} else False
        elif not isinstance(x, bool):
            raise TypeError(f"Value not boolean: {x}.")
        return x

    def completer(self, **kwargs):
        """Used by argcomplete to support tab completion of arguments."""
        return ["true", "false", "yes", "no"]


class ActionParser:
    """Action to parse option with a given parser optionally loading from file if string value."""

    def __init__(
        self,
        parser: Optional["ArgumentParser"] = None,
    ):
        """Initializer for ActionParser instance.

        Args:
            parser (Optional[ArgumentParser]): A parser to parse the option with.

        Raises:
            ValueError: If the parser parameter is invalid.
        """
        self._parser = parser
        if not isinstance(self._parser, import_object("jsonargparse.ArgumentParser")):
            raise ValueError("Expected parser keyword argument to be an ArgumentParser.")

    @staticmethod
    def _is_valid_action_parser(parser, action) -> bool:
        if not isinstance(action, ActionParser):
            return False
        if action._parser == parser:
            raise ValueError("Parser cannot be added as a subparser of itself.")
        return True

    @staticmethod
    def _move_parser_actions(parser, args, kwargs):
        subparser = kwargs.pop("action")._parser
        title = kwargs.pop("title", kwargs.pop("help", None))
        description = kwargs.pop("description", subparser.description)
        if len(kwargs) > 0:
            raise ValueError(f"ActionParser does not accept the following parameters: {set(kwargs.keys())}")
        if not (len(args) == 1 and args[0][:2] == "--"):
            raise ValueError(f"ActionParser only accepts a single optional key but got {args}")
        prefix = args[0][2:]

        def add_prefix(key):
            return re.sub("^--", "--" + prefix + ".", key)

        required_args = {prefix + "." + x for x in subparser.required_args}

        option_string_actions = {}
        for key, action in filter_default_actions(subparser._option_string_actions).items():
            option_string_actions[add_prefix(key)] = action

        isect = set(option_string_actions.keys()).intersection(set(parser._option_string_actions.keys()))
        if len(isect) > 0:
            raise ValueError(f"ActionParser conflicting keys: {isect}")

        actions = []
        dest = prefix.replace("-", "_")
        for action in filter_default_actions(subparser._actions):
            if isinstance(action, ActionYesNo):
                action._add_dest_prefix(prefix)
            else:
                action.dest = dest + "." + action.dest
                action.option_strings = [add_prefix(key) for key in action.option_strings]
            actions.append(action)

        base_action_group = subparser._action_groups[1]
        base_action_group.title = title
        if description is not None:
            base_action_group.description = description
        base_action_group.parser = parser
        base_action_group._actions = filter_default_actions(base_action_group._actions)
        base_action_group._group_actions = filter_default_actions(base_action_group._group_actions)
        extra_action_groups = subparser._action_groups[2:]

        parser.add_argument(args[0], action=_ActionConfigLoad)
        parser.required_args.update(required_args)
        parser._option_string_actions.update(option_string_actions)
        parser._actions.extend(actions)
        parser._action_groups.extend([base_action_group] + extra_action_groups)

        subparser._option_string_actions = {}
        subparser._actions = []
        subparser._action_groups = []

        return base_action_group


single_subcommand: ContextVar = ContextVar("single_subcommand", default=True)
parent_parsers: ContextVar = ContextVar("parent_parsers", default=[])
parse_kwargs: ContextVar = ContextVar("parse_kwargs", default={})


@contextmanager
def parent_parsers_context(key, parser):
    prev = parent_parsers.get()
    curr = prev + [(key, parser)]
    t = parent_parsers.set(curr)
    try:
        yield
    finally:
        parent_parsers.reset(t)


class _ActionSubCommands(_SubParsersAction):
    """Extension of argparse._SubParsersAction to modify subcommands functionality."""

    parent_parser: "ArgumentParser"
    env_prefix: str

    def add_parser(self, name, **kwargs):
        """Raises a NotImplementedError."""
        raise NotImplementedError("In jsonargparse subcommands are added using the add_subcommand method.")

    def add_subcommand(self, name, parser, **kwargs):
        """Adds a parser as a sub-command parser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        add_parser requires to be given a parser as argument.
        """
        if parser._subparsers is not None:
            raise ValueError("Multiple levels of subcommands must be added in level order.")

        parser.prog = f"{self._prog_prefix} [options] {name}"
        parser.env_prefix = f"{self.env_prefix}{name}_"
        parser.default_env = self.parent_parser.default_env
        parser.parent_parser = self.parent_parser
        parser.parser_mode = self.parent_parser.parser_mode
        parser._error_handler = self.parent_parser._error_handler
        parser.exit_on_error = self.parent_parser.exit_on_error
        parser.logger = self.parent_parser.logger
        parser.subcommand = name

        # create a pseudo-action to hold the choice help
        aliases = kwargs.pop("aliases", ())
        help_arg = None
        if "help" in kwargs:
            help_arg = kwargs.pop("help")
        choice_action = self._ChoicesPseudoAction(name, aliases, help_arg)
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
        namespace[self.dest] = subcommand

        # parse arguments
        if subcommand in self._name_parser_map:
            subparser = self._name_parser_map[subcommand]
            subnamespace = namespace.get(subcommand).clone() if subcommand in namespace else None
            kwargs = dict(_skip_check=True, **parse_kwargs.get())
            namespace[subcommand] = subparser.parse_args(arg_strings, namespace=subnamespace, **kwargs)

    @staticmethod
    @contextmanager
    def parse_kwargs_context(kwargs):
        parse_kwargs.set(kwargs)
        yield

    @staticmethod
    @contextmanager
    def not_single_subcommand():
        t = single_subcommand.set(False)
        try:
            yield
        finally:
            single_subcommand.reset(t)

    @staticmethod
    def get_subcommands(
        parser: "ArgumentParser",
        cfg: Namespace,
        prefix: str = "",
        fail_no_subcommand: bool = True,
    ) -> Tuple[Optional[List[str]], Optional[List["ArgumentParser"]]]:
        """Returns subcommand names and corresponding subparsers."""
        if parser._subcommands_action is None:
            return None, None
        action = parser._subcommands_action

        require_single = single_subcommand.get()

        # Get subcommand settings keys
        subcommand_keys = [k for k in action.choices.keys() if isinstance(cfg.get(prefix + k), Namespace)]

        # Get subcommand
        subcommand = None
        dest = prefix + action.dest
        if dest in cfg and cfg.get(dest) is not None:
            subcommand = cfg[dest]
        elif len(subcommand_keys) > 0 and (fail_no_subcommand or require_single):
            cfg[dest] = subcommand = subcommand_keys[0]
            if len(subcommand_keys) > 1:
                warnings.warn(
                    f'Multiple subcommand settings provided ({", ".join(subcommand_keys)}) without an '
                    f'explicit "{dest}" key. Subcommand "{subcommand}" will be used.'
                )

        # Remove extra subcommand settings
        if subcommand and len(subcommand_keys) > 1:
            for key in [k for k in subcommand_keys if k != subcommand]:
                del cfg[prefix + key]

        if subcommand:
            subcommand_keys = [subcommand]

        if fail_no_subcommand:
            if subcommand is None and not (fail_no_subcommand and action._required):  # type: ignore
                return None, None
            if action._required and subcommand not in action._name_parser_map:  # type: ignore
                # If subcommand is required and no subcommand is provided,
                # present the user with a friendly error message to remind them of
                # the available subcommands and to select one.
                available_subcommands = list(action._name_parser_map.keys())
                if len(available_subcommands) <= 5:
                    candidate_subcommands_str = "{" + ",".join(available_subcommands) + "}"
                else:
                    candidate_subcommands_str = "{" + ",".join(available_subcommands[:5]) + ", ...}"
                raise NSKeyError(
                    f'expected "{dest}" to be one of {candidate_subcommands_str}, but it was not provided.'
                )

        return subcommand_keys, [action._name_parser_map.get(s) for s in subcommand_keys]  # type: ignore

    @staticmethod
    def get_subcommand(
        parser: "ArgumentParser",
        cfg: Namespace,
        prefix: str = "",
        fail_no_subcommand: bool = True,
    ) -> Tuple[Optional[str], Optional["ArgumentParser"]]:
        """Returns a single subcommand name and corresponding subparser."""
        subcommands, subparsers = _ActionSubCommands.get_subcommands(
            parser,
            cfg,
            prefix=prefix,
            fail_no_subcommand=fail_no_subcommand,
        )
        return subcommands[0] if subcommands else None, subparsers[0] if subparsers else None

    @staticmethod
    def handle_subcommands(
        parser: "ArgumentParser",
        cfg: Namespace,
        env: Optional[bool],
        defaults: bool,
        prefix: str = "",
        fail_no_subcommand: bool = True,
    ) -> None:
        """Takes care of parsing subcommand values."""

        subcommands, subparsers = _ActionSubCommands.get_subcommands(
            parser, cfg, prefix=prefix, fail_no_subcommand=fail_no_subcommand
        )
        if not subcommands or not subparsers:
            return

        for subcommand, subparser in zip(subcommands, subparsers):
            # Merge environment variable values and default values
            subnamespace = None
            key = prefix + subcommand
            with parent_parsers_context(key, parser):
                if env:
                    subnamespace = subparser.parse_env(defaults=defaults, _skip_check=True)
                elif defaults:
                    subnamespace = subparser.get_defaults(skip_check=True)

            # Update all subcommand settings
            if subnamespace is not None:
                cfg[key] = subparser.merge_config(cfg.get(key, Namespace()), subnamespace)

            # Handle inner subcommands
            if subparser._subparsers is not None:
                _ActionSubCommands.handle_subcommands(
                    subparser, cfg, env, defaults, key + ".", fail_no_subcommand=fail_no_subcommand
                )
