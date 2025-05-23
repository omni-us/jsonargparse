"""Simple creation of command line interfaces."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._actions import ActionConfigFile, _ActionPrintConfig, remove_actions
from ._core import ArgumentParser
from ._deprecated import deprecation_warning_cli_return_parser
from ._namespace import Namespace, dict_to_namespace
from ._optionals import get_doc_short_description
from ._util import capture_parser, default_config_option_help

__all__ = [
    "CLI",
    "auto_cli",
    "auto_parser",
]


ComponentType = Union[Callable, Type]
DictComponentsType = Dict[str, Union[ComponentType, "DictComponentsType"]]
ComponentsType = Optional[Union[ComponentType, List[ComponentType], DictComponentsType]]


def CLI(*args, **kwargs):
    """Alias of :func:`auto_cli`."""
    return auto_cli(*args, _stacklevel=3, **kwargs)


def auto_cli(
    components: ComponentsType = None,
    args: Optional[List[str]] = None,
    config_help: str = default_config_option_help,
    set_defaults: Optional[Dict[str, Any]] = None,
    as_positional: bool = True,
    fail_untyped: bool = True,
    parser_class: Type[ArgumentParser] = ArgumentParser,
    **kwargs,
):
    """Simple creation of command line interfaces.

    Previously CLI, renamed to follow the standard of functions in lowercase.

    Creates an argument parser from one or more functions/classes, parses
    arguments and runs one of the functions or class methods depending on what
    was parsed. If the 'components' parameter is not given, then the components
    will be all the locals in the context and defined in the same module as from
    where auto_cli is called.

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
    stacklevel = kwargs.pop("_stacklevel", 2)

    if components is None:
        caller = inspect.stack()[stacklevel - 1][0]
        module = inspect.getmodule(caller)
        components = [
            v for v in vars(module).values() if ((inspect.isclass(v) or callable(v)) and inspect.getmodule(v) is module)
        ]
        if len(components) == 0:
            raise ValueError(
                "Either components parameter must be given or there must be at least one "
                "function or class among the locals in the context where CLI is called."
            )

    if isinstance(components, list) and len(components) == 1:
        components = components[0]

    elif not components:
        raise ValueError("components parameter expected to be non-empty")

    if isinstance(components, list):
        unexpected = [c for c in components if not (inspect.isclass(c) or callable(c))]
    elif isinstance(components, dict):
        ns = dict_to_namespace(components)
        unexpected = [c for k, c in ns.items() if not k.endswith("._help") and not (inspect.isclass(c) or callable(c))]
    else:
        unexpected = [c for c in [components] if not (inspect.isclass(c) or callable(c))]
    if unexpected:
        raise ValueError(f"Unexpected components, not class or function: {unexpected}")

    parser = parser_class(default_meta=False, **kwargs)
    parser.add_argument("--config", action=ActionConfigFile, help=config_help)

    if not isinstance(components, (list, dict)):
        _add_component_to_parser(components, parser, as_positional, fail_untyped, config_help)
        if set_defaults is not None:
            parser.set_defaults(set_defaults)
        if return_parser:
            deprecation_warning_cli_return_parser(stacklevel)
            return parser
        cfg = parser.parse_args(args)
        init = parser.instantiate_classes(cfg)
        return _run_component(components, init)

    elif isinstance(components, list):
        components = {c.__name__: c for c in components}

    _add_subcommands(components, parser, config_help, as_positional, fail_untyped)

    if set_defaults is not None:
        parser.set_defaults(set_defaults)
    if return_parser:
        deprecation_warning_cli_return_parser(stacklevel)
        return parser
    cfg = parser.parse_args(args)
    init = parser.instantiate_classes(cfg)
    components_ns = dict_to_namespace(components)
    subcommand = init.get("subcommand")
    while isinstance(init.get(subcommand), Namespace) and isinstance(init[subcommand].get("subcommand"), str):
        subsubcommand = subcommand + "." + init[subcommand].get("subcommand")
        if subsubcommand in components_ns:
            subcommand = subsubcommand
        else:
            break
    component = components_ns[subcommand]
    return _run_component(component, init.get(subcommand))


def auto_parser(*args, **kwargs) -> ArgumentParser:
    """Same as auto_cli, but returns the parser, so doesn't parse arguments or run.

    This is a shorthand for ``capture_parser(lambda: auto_cli(*args, **kwargs))``.
    """
    return capture_parser(lambda: auto_cli(*args, **kwargs))


def get_help_str(component, logger):
    if isinstance(component, dict):
        return component.get("_help")
    help_str = get_doc_short_description(component, logger=logger)
    if not help_str:
        help_str = str(component)
    return help_str


def _add_subcommands(
    components,
    parser: ArgumentParser,
    config_help: str,
    as_positional: bool,
    fail_untyped: bool,
) -> None:
    subcommands = parser.add_subcommands(required=True)
    for name, component in components.items():
        if name == "_help":
            continue
        description = get_help_str(component, parser.logger)
        subparser = type(parser)(description=description)
        subparser.add_argument("--config", action=ActionConfigFile, help=config_help)
        subcommands.add_subcommand(name, subparser, help=description)
        if isinstance(component, dict):
            _add_subcommands(component, subparser, config_help, as_positional, fail_untyped)
        else:
            added_args = _add_component_to_parser(component, subparser, as_positional, fail_untyped, config_help)
            if not added_args:
                remove_actions(subparser, (ActionConfigFile, _ActionPrintConfig))


def has_parameter(component, name) -> bool:
    return name in inspect.signature(component).parameters.keys()


def _add_component_to_parser(
    component,
    parser: ArgumentParser,
    as_positional: bool,
    fail_untyped: bool,
    config_help: str,
):
    kwargs: dict = dict(as_positional=as_positional, fail_untyped=fail_untyped, sub_configs=True)
    if inspect.isclass(component):
        class_methods = [
            k for k, v in inspect.getmembers(component) if (callable(v) or isinstance(v, property)) and k[0] != "_"
        ]
        if not class_methods:
            added_args = parser.add_class_arguments(component, as_group=False, **kwargs)
            if not parser.description:
                parser.description = get_help_str(component, parser.logger)
            return added_args
        added_args = parser.add_class_arguments(component, **kwargs)
        subcommands = parser.add_subcommands(required=True)
        for method in class_methods:
            method_object = getattr(component, method)
            description = get_help_str(method_object, parser.logger)
            subparser = type(parser)(description=description)
            if not isinstance(method_object, property):
                if not has_parameter(method_object, "config"):
                    subparser.add_argument("--config", action=ActionConfigFile, help=config_help)
                added_subargs = subparser.add_method_arguments(component, method, as_group=False, **kwargs)
                added_args += [f"{method}.{a}" for a in added_subargs]
                if not added_subargs:
                    remove_actions(subparser, (ActionConfigFile, _ActionPrintConfig))
            subcommands.add_subcommand(method, subparser, help=get_help_str(method_object, parser.logger))
    else:
        added_args = parser.add_function_arguments(component, as_group=False, **kwargs)
        if not parser.description:
            parser.description = get_help_str(component, parser.logger)
    return added_args


def _run_component(component, cfg):
    cfg.pop("config", None)
    subcommand = cfg.pop("subcommand")
    if inspect.isclass(component) and subcommand:
        subcommand_cfg = cfg.pop(subcommand, {})
        subcommand_cfg.pop("config", None)
        component_obj = component(**cfg)
        if isinstance(getattr(component, subcommand), property):
            return getattr(component_obj, subcommand)
        component = getattr(component_obj, subcommand)
        cfg = subcommand_cfg
    if inspect.iscoroutinefunction(component):
        return __import__("asyncio").run(component(**cfg))
    return component(**cfg)
