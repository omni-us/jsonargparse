"""Simple creation of command line interfaces."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._actions import ActionConfigFile, _ActionPrintConfig, remove_actions
from ._core import ArgumentParser
from ._deprecated import deprecation_warning_cli_return_parser
from ._optionals import get_doc_short_description
from ._util import default_config_option_help

__all__ = ["CLI"]


def CLI(
    components: Optional[Union[Callable, Type, List[Union[Callable, Type]]]] = None,
    args: Optional[List[str]] = None,
    config_help: str = default_config_option_help,
    set_defaults: Optional[Dict[str, Any]] = None,
    as_positional: bool = True,
    fail_untyped: bool = True,
    parser_class: Type[ArgumentParser] = ArgumentParser,
    **kwargs,
):
    """Simple creation of command line interfaces.

    Creates an argument parser from one or more functions/classes, parses
    arguments and runs one of the functions or class methods depending on what
    was parsed. If the 'components' parameter is not given, then the components
    will be all the locals in the context and defined in the same module as from
    where CLI is called.

    Args:
        components: One or more functions/classes to include in the command line interface.
        args: List of arguments to parse or None to use sys.argv.
        config_help: Help string for config file option in help.
        set_defaults: Dictionary of values to override components defaults.
        as_positional: Whether to add required parameters as positional arguments.
        fail_untyped: Whether to raise exception if a required parameter does not have a type.
        parser_class: The ArgumentParser class to use.
        **kwargs: Used to instantiate :class:`.ArgumentParser`.

    Returns:
        The value returned by the executed function or class method.
    """
    return_parser = kwargs.pop("return_parser", False)
    caller = inspect.stack()[1][0]

    if components is None:
        module = inspect.getmodule(caller).__name__  # type: ignore
        components = [
            v
            for v in caller.f_locals.values()
            if ((inspect.isclass(v) or callable(v)) and getattr(inspect.getmodule(v), "__name__", None) == module)
        ]
        if len(components) == 0:
            raise ValueError(
                "Either components argument must be given or there must be at least one "
                "function or class among the locals in the context where CLI is called."
            )

    elif not isinstance(components, list):
        components = [components]

    if len(components) == 0:
        raise ValueError("components argument not allowed to be an empty list")

    unexpected = [c for c in components if not (inspect.isclass(c) or callable(c))]
    if unexpected:
        raise ValueError(f"Unexpected components, not class or function: {unexpected}")

    parser = parser_class(default_meta=False, **kwargs)
    parser.add_argument("--config", action=ActionConfigFile, help=config_help)

    if len(components) == 1:
        component = components[0]
        _add_component_to_parser(component, parser, as_positional, fail_untyped, config_help)
        if set_defaults is not None:
            parser.set_defaults(set_defaults)
        if return_parser:
            deprecation_warning_cli_return_parser()
            return parser
        cfg = parser.parse_args(args)
        cfg_init = parser.instantiate_classes(cfg)
        return _run_component(component, cfg_init)

    subcommands = parser.add_subcommands(required=True)
    comp_dict = {c.__name__: c for c in components}
    for name, component in comp_dict.items():
        description = get_help_str(component, parser.logger)
        subparser = parser_class(description=description)
        subparser.add_argument("--config", action=ActionConfigFile, help=config_help)
        subcommands.add_subcommand(name, subparser, help=get_help_str(component, parser.logger))
        added_args = _add_component_to_parser(component, subparser, as_positional, fail_untyped, config_help)
        if not added_args:
            remove_actions(subparser, (ActionConfigFile, _ActionPrintConfig))

    if set_defaults is not None:
        parser.set_defaults(set_defaults)
    if return_parser:
        deprecation_warning_cli_return_parser()
        return parser
    cfg = parser.parse_args(args)
    cfg_init = parser.instantiate_classes(cfg)
    subcommand = cfg_init.pop("subcommand")
    component = comp_dict[subcommand]
    return _run_component(component, cfg_init.get(subcommand))


def get_help_str(component, logger):
    help_str = get_doc_short_description(component, logger=logger)
    if not help_str:
        help_str = str(component)
    return help_str


def _add_component_to_parser(component, parser, as_positional, fail_untyped, config_help):
    kwargs = dict(as_positional=as_positional, fail_untyped=fail_untyped, sub_configs=True)
    if inspect.isclass(component):
        subcommand_keys = [k for k, v in inspect.getmembers(component) if callable(v) and k[0] != "_"]
        if not subcommand_keys:
            added_args = parser.add_class_arguments(component, as_group=False, **kwargs)
            if not parser.description:
                parser.description = get_help_str(component, parser.logger)
            return added_args
        added_args = parser.add_class_arguments(component, **kwargs)
        subcommands = parser.add_subcommands(required=True)
        for key in subcommand_keys:
            description = get_help_str(getattr(component, key), parser.logger)
            subparser = type(parser)(description=description)
            subparser.add_argument("--config", action=ActionConfigFile, help=config_help)
            added_subargs = subparser.add_method_arguments(component, key, as_group=False, **kwargs)
            added_args += [f"{key}.{a}" for a in added_subargs]
            if not added_subargs:
                remove_actions(subparser, (ActionConfigFile, _ActionPrintConfig))
            subcommands.add_subcommand(key, subparser, help=get_help_str(getattr(component, key), parser.logger))
    else:
        added_args = parser.add_function_arguments(component, as_group=False, **kwargs)
        if not parser.description:
            parser.description = get_help_str(component, parser.logger)
    return added_args


def _run_component(component, cfg):
    cfg.pop("config", None)
    if not inspect.isclass(component):
        return component(**cfg)
    subcommand = cfg.pop("subcommand")
    if not subcommand:
        return component(**cfg)
    subcommand_cfg = cfg.pop(subcommand, {})
    subcommand_cfg.pop("config", None)
    component_obj = component(**cfg)
    return getattr(component_obj, subcommand)(**subcommand_cfg)
