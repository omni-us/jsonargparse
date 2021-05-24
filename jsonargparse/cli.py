"""Simple creation of command line interfaces."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union
from .actions import ActionConfigFile
from .core import ArgumentParser
from .optionals import docstring_parser_support, import_docstring_parse


__all__ = ['CLI']


def CLI(
    components: Union[Callable, Type, List[Union[Callable, Type]]] = None,
    args: List[str] = None,
    config_help: str = 'Path to a configuration file in json or yaml format.',
    set_defaults: Optional[Dict[str, Any]] = None,
    as_positional: bool = True,
    return_parser: bool = False,
    **kwargs
):
    """Function for simple creation of command line interfaces.

    Creates an argument parser from one or more functions/classes, parses
    arguments and runs one of the functions or class methods depending on what
    was parsed. If the components argument is not given, then the components
    will be all the locals in the context and defined in the same module as from
    where CLI is called.

    Args:
        components: One or more functions/classes to include in the command line interface.
        args: List of arguments to parse or None to use sys.argv.
        config_help: Help string for config file option in help.
        set_defaults: Dictionary of values to override components defaults.
        as_positional: Whether to add required parameters as positional arguments.
        return_parser: Whether to return the parser instead of parsing and running.
        **kwargs: Used to instantiate :class:`.ArgumentParser`.

    Returns:
        The value returned by the executed function or class method.
    """
    caller = inspect.stack()[1][0]
    if 'description' not in kwargs:
        kwargs['description'] = caller.f_globals['__doc__']

    if components is None:
        module = inspect.getmodule(caller).__name__  # type: ignore
        components = [
            v for v in caller.f_locals.values()
            if (inspect.isfunction(v) or inspect.isclass(v)) and inspect.getmodule(v).__name__ == module  # type: ignore
        ]
        if len(components) == 0:
            raise ValueError('Either components argument must be given or there must be at least one '
                             'function or class among the locals in the context where CLI is called.')

    elif not isinstance(components, list):
        components = [components]

    parser = ArgumentParser(parse_as_dict=True, default_meta=False, **kwargs)
    parser.add_argument('--config', action=ActionConfigFile, help=config_help)

    if len(components) == 1:
        component = components[0]
        _add_component_to_parser(component, parser, as_positional, config_help)
        if set_defaults is not None:
            parser.set_defaults(set_defaults)
        if return_parser:
            return parser
        cfg = parser.instantiate_subclasses(parser.parse_args(args))
        return _run_component(component, cfg)

    subcommands = parser.add_subcommands(required=True)
    comp_dict = {c.__name__: c for c in components}
    for name, component in comp_dict.items():
        subparser = ArgumentParser()
        subparser.add_argument('--config', action=ActionConfigFile, help=config_help)
        subcommands.add_subcommand(name, subparser, help=_get_help_str(component))  # type: ignore
        _add_component_to_parser(component, subparser, as_positional, config_help)

    if set_defaults is not None:
        parser.set_defaults(set_defaults)
    if return_parser:
        return parser
    cfg = parser.instantiate_subclasses(parser.parse_args(args))
    subcommand = cfg.pop('subcommand')
    component = comp_dict[subcommand]
    return _run_component(component, cfg.get(subcommand))


def _get_help_str(component):
    help_str = str(component)
    if docstring_parser_support:
        docstring_parse = import_docstring_parse('_get_help_str')
        description = None
        if inspect.isclass(component):
            description = docstring_parse(component.__init__.__doc__).short_description
        if description is None:
            description = docstring_parse(component.__doc__).short_description
        if description is not None:
            help_str = description
    return help_str


def _add_component_to_parser(component, parser, as_positional, config_help):
    kwargs = {'as_positional': as_positional, 'sub_configs': True}
    if inspect.isfunction(component):
        parser.add_function_arguments(component, **kwargs)
    else:
        parser.add_class_arguments(component, **kwargs)
        subcommands = parser.add_subcommands(required=True)
        for key in [k for k, v in inspect.getmembers(component) if callable(v) and k[0] != '_']:
            subparser = ArgumentParser()
            subparser.add_argument('--config', action=ActionConfigFile, help=config_help)
            subparser.add_method_arguments(component, key, **kwargs)
            subcommands.add_subcommand(key, subparser, help=_get_help_str(getattr(component, key)))


def _run_component(component, cfg):
    cfg.pop('config', None)
    if inspect.isfunction(component):
        return component(**cfg)
    subcommand = cfg.pop('subcommand')
    subcommand_cfg = cfg.pop(subcommand, {})
    subcommand_cfg.pop('config', None)
    component_obj = component(**cfg)
    return getattr(component_obj, subcommand)(**subcommand_cfg)
