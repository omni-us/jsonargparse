"""Formatter classes."""

from enum import Enum
from argparse import HelpFormatter, OPTIONAL, SUPPRESS, ZERO_OR_MORE

from .util import _get_env_var
from .actions import ActionParser, ActionEnum
from .jsonschema import type_to_str, ActionJsonSchema


__all__ = ['DefaultHelpFormatter']


class DefaultHelpFormatter(HelpFormatter):
    """Help message formatter that includes types, default values and env var names.

    This class is an extension of `argparse.HelpFormatter
    <https://docs.python.org/3/library/argparse.html#argparse.HelpFormatter>`_.
    Default values are always included. Furthermore, if the parser is configured
    with :code:`default_env=True` command line options are preceded by 'ARG:' and
    the respective environment variable name is included preceded by 'ENV:'.
    """

    _env_prefix = None
    _default_env = True

    def _get_help_string(self, action):
        help_str = ''
        if hasattr(action, '_required') and action._required:
            help_str = 'required'
        if '%(type)' not in action.help and self._get_type_str(action) is not None:
            help_str += (', ' if help_str else '') + 'type: %(type)s'
        if '%(default)' not in action.help and action.default is not SUPPRESS and \
           (action.option_strings or action.nargs in {OPTIONAL, ZERO_OR_MORE}):
            help_str += (', ' if help_str else '') + 'default: %(default)s'
        return action.help + (' ('+help_str+')' if help_str else '')


    def _format_action_invocation(self, action):
        if action.option_strings == [] or action.default == SUPPRESS or not self._default_env:
            return super()._format_action_invocation(action)
        extr = ''
        if self._default_env:
            extr += '\n  ENV:   ' + _get_env_var(self, action)
        if isinstance(action, ActionParser):
            extr += '\n                        For more details run command with --'+action.dest+'.help.'
        return 'ARG:   ' + super()._format_action_invocation(action) + extr


    def _get_default_metavar_for_optional(self, action):
        return action.dest.rsplit('.')[-1].upper()


    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        for name in list(params):
            if params[name] is SUPPRESS:
                del params[name]
        for name in list(params):
            if hasattr(params[name], '__name__'):
                params[name] = params[name].__name__
        if params.get('choices') is not None:
            choices_str = ', '.join([str(c) for c in params['choices']])
            params['choices'] = choices_str
        type_str = self._get_type_str(action)
        if type_str is not None:
            params['type'] = type_str
        if 'default' in params:
            if params['default'] is None:
                params['default'] = 'null'
            elif isinstance(params['default'], Enum) and hasattr(params['default'], 'name'):
                params['default'] = action.default.name
        return self._get_help_string(action) % params


    def _get_type_str(self, action):
        type_str = None
        if action.type is not None:
            type_str = type_to_str(action.type)
        elif isinstance(action, ActionEnum):
            type_str = type_to_str(action._enum)
        elif isinstance(action, ActionJsonSchema):
            type_str = type_to_str(action._annotation)
        return type_str
