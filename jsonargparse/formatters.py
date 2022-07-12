"""Formatter classes."""

import re
from argparse import Action, _HelpAction, HelpFormatter, OPTIONAL, SUPPRESS, ZERO_OR_MORE
from contextlib import contextmanager
from contextvars import ContextVar
from io import StringIO
from string import Template
from typing import Optional, Union

from .actions import (
    ActionConfigFile,
    ActionYesNo,
    _ActionConfigLoad,
    _ActionSubCommands,
    _find_action,
    filter_default_actions,
)
from .link_arguments import ActionLink
from .namespace import Namespace
from .optionals import import_ruyaml
from .type_checking import ArgumentParser, ruyamlCommentedMap
from .typehints import ActionTypeHint, type_to_str


__all__ = ['DefaultHelpFormatter']


empty_help: str = '_EMPTY_HELP_'


formatter_parser: ContextVar = ContextVar('formatter_parser')
formatter_defaults: ContextVar = ContextVar('formatter_defaults')


@contextmanager
def formatter_context(parser: 'ArgumentParser', defaults: Optional[Namespace] = None):
    prev_parser = formatter_parser.set(parser)
    prev_defaults = formatter_defaults.set(defaults)
    try:
        yield
    finally:
        formatter_parser.reset(prev_parser)
        formatter_defaults.reset(prev_defaults)


class PercentTemplate(Template):
    delimiter = '%'
    pattern = r'''
    \%\((?:
    (?P<escaped>\%\%)|
    (?P<named>[_a-z][_a-z0-9]*)\)s|
    (?P<braced>[_a-z][_a-z0-9]*)\)s|
    (?P<invalid>)
    )
    '''  # type: ignore


class DefaultHelpFormatter(HelpFormatter):
    """Help message formatter that includes types, default values and env var names.

    This class is an extension of `argparse.HelpFormatter
    <https://docs.python.org/3/library/argparse.html#argparse.HelpFormatter>`_.
    Default values are always included. Furthermore, if the parser is configured
    with :code:`default_env=True` command line options are preceded by 'ARG:' and
    the respective environment variable name is included preceded by 'ENV:'.
    """

    def _get_help_string(self, action):
        action_help = ' ' if action.help == empty_help else action.help
        if isinstance(action, ActionConfigFile):
            return action_help
        if isinstance(action, _HelpAction):
            help_str = action_help[0].upper() + action_help[1:]
            if help_str[-1] != '.':
                help_str += '.'
            return help_str
        help_str = ''
        is_required = hasattr(action, '_required') and action._required
        if is_required:
            help_str = 'required'
        if '%(type)' not in action_help and self._get_type_str(action) is not None:
            help_str += (', ' if help_str else '') + 'type: %(type)s'
        if '%(default)' not in action_help and \
           action.default != SUPPRESS and \
           (action.default is not None or not is_required) and \
           (action.option_strings or action.nargs in {OPTIONAL, ZERO_OR_MORE}):
            help_str += (', ' if help_str else '') + 'default: %(default)s'
        if isinstance(action, ActionTypeHint):
            help_str += action.extra_help()
        return action_help + (' ('+help_str+')' if help_str else '')


    def _format_usage(self, *args, **kwargs):
        usage = super()._format_usage(*args, **kwargs)
        parser = formatter_parser.get()
        for key in parser.required_args:
            try:
                default = parser.get_default(key)
            except KeyError:
                default = None
            if default is None and f'[--{key} ' in usage:
                usage = re.sub(f'\\[(--{key} [^\\]]+)]', r'\1', usage, count=1)
        return usage


    def _format_action_invocation(self, action):
        parser = formatter_parser.get()
        if action.option_strings == [] or action.default == SUPPRESS or not parser.default_env:
            return super()._format_action_invocation(action)
        extr = ''
        if parser.default_env:
            extr += '\n  ENV:   ' + get_env_var(self, action)
        return 'ARG:   ' + super()._format_action_invocation(action) + extr


    def _get_default_metavar_for_optional(self, action):
        return action.dest.rsplit('.')[-1].upper()


    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        if params.get('choices') is not None:
            choices_str = ', '.join([str(c) for c in params['choices']])
            params['choices'] = choices_str
        type_str = self._get_type_str(action)
        if type_str is not None:
            params['type'] = type_str
        orig_default = action.default
        if params.get('default') == SUPPRESS:
            del params['default']
        elif 'default' in params:
            defaults = formatter_defaults.get()
            if defaults is not None:
                params['default'] = action.default = defaults.get(action.dest)
            if params['default'] is None:
                params['default'] = 'null'
            elif isinstance(params['default'], Namespace):
                params['default'] = params['default'].as_dict()
        help_str = PercentTemplate(self._get_help_string(action)).safe_substitute(params)
        action.default = orig_default
        return help_str


    def _get_type_str(self, action):
        type_str = None
        if isinstance(action, ActionYesNo):
            type_str = 'bool'
        elif action.type is not None:
            type_str = type_to_str(action.type)
        elif isinstance(action, ActionTypeHint):
            type_str = type_to_str(action._typehint)
        return type_str


    def add_usage(self, usage, actions, groups, prefix=None):
        actions = [a for a in actions if not isinstance(a, ActionLink)]
        super().add_usage(usage, actions, groups, prefix=prefix)


    def add_yaml_comments(self, cfg: str) -> str:
        """Adds help text as yaml comments."""
        ruyaml = import_ruyaml('add_yaml_comments')
        yaml = ruyaml.YAML()
        cfg = yaml.load(cfg)

        def get_subparsers(parser, prefix=''):
            subparsers = {}
            if parser._subparsers is not None:
                for key, subparser in parser._subparsers._group_actions[0].choices.items():
                    full_key = (prefix+'.' if prefix else '')+key
                    subparsers[full_key] = subparser
                    subparsers.update(get_subparsers(subparser, prefix=full_key))
            return subparsers

        parser = formatter_parser.get()
        parsers = get_subparsers(parser)
        parsers[None] = parser

        group_titles = {}
        for parser_key, parser in parsers.items():
            prefix = '' if parser_key is None else parser_key + '.'
            for group in parser._action_groups:
                actions = filter_default_actions(group._group_actions)
                actions = [a for a in actions if not isinstance(a, (_ActionConfigLoad, ActionConfigFile, _ActionSubCommands))]
                keys = {re.sub(r'\.?[^.]+$', '', a.dest) for a in actions}
                for key in keys:
                    full_key = prefix + key if key != '' else parser_key
                    group_titles[full_key] = group.title

        def set_comments(cfg, prefix='', depth=0):
            for key in cfg.keys():
                full_key = (prefix+'.' if prefix else '')+key
                action = _find_action(parser, full_key)
                text = None
                if full_key in group_titles and isinstance(cfg[key], dict):
                    text = group_titles[full_key]
                elif action is not None and action.help not in {None, SUPPRESS}:
                    text = self._expand_help(action)
                if isinstance(cfg[key], dict):
                    if text:
                        self.set_yaml_group_comment(text, cfg, key, depth)
                    set_comments(cfg[key], full_key, depth+1)
                elif text:
                    self.set_yaml_argument_comment(text, cfg, key, depth)

        if parser.description is not None:
            self.set_yaml_start_comment(parser.description, cfg)
        set_comments(cfg)
        out = StringIO()
        yaml.dump(cfg, out)
        return out.getvalue()


    def set_yaml_start_comment(
        self,
        text: str,
        cfg: 'ruyamlCommentedMap',
    ):
        """Sets the start comment to a ruyaml object.

        Args:
            text: The content to use for the comment.
            cfg: The ruyaml object.
        """
        cfg.yaml_set_start_comment(text)


    def set_yaml_group_comment(
        self,
        text: str,
        cfg: 'ruyamlCommentedMap',
        key: str,
        depth: int,
    ):
        """Sets the comment for a group to a ruyaml object.

        Args:
            text: The content to use for the comment.
            cfg: The parent ruyaml object.
            key: The key of the group.
            depth: The nested level of the group.
        """
        cfg.yaml_set_comment_before_after_key(key, before='\n'+text, indent=2*depth)


    def set_yaml_argument_comment(
        self,
        text: str,
        cfg: 'ruyamlCommentedMap',
        key: str,
        depth: int,
    ):
        """Sets the comment for an argument to a ruyaml object.

        Args:
            text: The content to use for the comment.
            cfg: The parent ruyaml object.
            key: The key of the argument.
            depth: The nested level of the argument.
        """
        cfg.yaml_set_comment_before_after_key(key, before='\n'+text, indent=2*depth)


def get_env_var(parser_or_formatter: Union['ArgumentParser', DefaultHelpFormatter], action: Action) -> str:
    """Returns the environment variable for a given parser or formatter and action."""
    if isinstance(parser_or_formatter, DefaultHelpFormatter):
        parser = formatter_parser.get()
    else:
        parser = parser_or_formatter
    env_var = (parser.env_prefix+'_' if parser.env_prefix else '') + action.dest
    env_var = env_var.replace('.', '__').upper()
    return env_var
