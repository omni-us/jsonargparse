"""Simple creation of command line interfaces."""

import inspect
from typing import Union, List, Callable
from .core import ArgumentParser
from .actions import ActionConfigFile


__all__ = ['CLI']


def CLI(
    functions: Union[Callable, List[Callable]] = None,
    args: List[str] = None,
    config_help: str = 'Path to a configuration file in json or yaml format.',
    as_positional: bool = True,
    **kwargs
):
    """Function for simple creation of command line interfaces.

    Creates an argument parser from one or more functions, parses arguments and
    runs one of the functions depending on what was parsed. If the functions
    argument is not given, then the functions will be all the locals in the
    context and defined in the same module as from where CLI is called.

    Args:
        functions: One or more functions to include in the command line interface.
        args: List of arguments to parse or None to use sys.argv.
        config_help: Help string for config file option in help.
        as_positional: Whether to add required parameters as positional arguments.
        **kwargs: Used to instantiate :class:`.ArgumentParser`.

    Returns:
        The value returned by the executed function.
    """
    caller = inspect.stack()[1][0]
    if 'description' not in kwargs:
        kwargs['description'] = caller.f_globals['__doc__']

    if functions is None:
        module = inspect.getmodule(caller).__name__  # type: ignore
        functions = [
            v for v in caller.f_locals.values()
            if inspect.isfunction(v) and inspect.getmodule(v).__name__ == module  # type: ignore
        ]
        if len(functions) == 0:
            raise ValueError('Either functions argument must be given or there must be at least one '
                             'function among the locals in the context where CLI is called.')

    elif not isinstance(functions, list):
        functions = [functions]

    parser = ArgumentParser(parse_as_dict=True, default_meta=False, **kwargs)
    parser.add_argument('--config', action=ActionConfigFile, help=config_help)

    if len(functions) == 1:
        parser.add_function_arguments(functions[0], as_positional=as_positional)
        cfg = parser.parse_args(args)
        del cfg['config']  # pylint: disable=unsupported-delete-operation
        return functions[0](**cfg)  # type: ignore  # pylint: disable=not-a-mapping

    else:
        subcommands = parser.add_subcommands(required=True)
        func_dict = {c.__name__: c for c in functions}
        for name, function in func_dict.items():
            subparser = ArgumentParser()
            subparser.add_function_arguments(function, as_positional=as_positional)
            subcommands.add_subcommand(name, subparser)  # type: ignore

        cfg = parser.parse_args(args)
        function = func_dict[cfg['subcommand']]  # type: ignore  # pylint: disable=unsubscriptable-object
        return function(**cfg[cfg['subcommand']])  # type: ignore  # pylint: disable=unsubscriptable-object
