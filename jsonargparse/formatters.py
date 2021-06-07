"""Formatter classes."""

import re
from argparse import _HelpAction, HelpFormatter, OPTIONAL, SUPPRESS, ZERO_OR_MORE
from enum import Enum
from io import StringIO
from string import Template

from .actions import (
    ActionConfigFile,
    ActionYesNo,
    _ActionConfigLoad,
    _ActionLink,
    _ActionSubCommands,
    _find_action,
    filter_default_actions,
)
from .optionals import import_ruyaml
from .typehints import ActionTypeHint, type_to_str
from .util import _get_env_var, _get_key_value


__all__ = ['DefaultHelpFormatter']


class PercentTemplate(Template):
    delimiter = '%'
    pattern = r'''
    \%\((?:
    (?P<escaped>\%\%)|
    (?P<named>[_a-z][_a-z0-9]*)\)s|
    (?P<braced>[_a-z][_a-z0-9]*)\)s|
    (?P<invalid>)
    )
    '''


class DefaultHelpFormatter(HelpFormatter):
    """Help message formatter that includes types, default values and env var names.

    This class is an extension of `argparse.HelpFormatter
    <https://docs.python.org/3/library/argparse.html#argparse.HelpFormatter>`_.
    Default values are always included. Furthermore, if the parser is configured
    with :code:`default_env=True` command line options are preceded by 'ARG:' and
    the respective environment variable name is included preceded by 'ENV:'.
    """

    def _get_help_string(self, action):
        if isinstance(action, ActionConfigFile):
            return action.help
        if isinstance(action, _HelpAction):
            help_str = action.help[0].upper() + action.help[1:]
            if help_str[-1] != '.':
                help_str += '.'
            return help_str
        help_str = ''
        is_required = hasattr(action, '_required') and action._required
        if is_required:
            help_str = 'required'
        if '%(type)' not in action.help and self._get_type_str(action) is not None:
            help_str += (', ' if help_str else '') + 'type: %(type)s'
        if '%(default)' not in action.help and \
           action.default is not SUPPRESS and \
           (action.default is not None or not is_required) and \
           (action.option_strings or action.nargs in {OPTIONAL, ZERO_OR_MORE}):
            help_str += (', ' if help_str else '') + 'default: %(default)s'
        return action.help + (' ('+help_str+')' if help_str else '')


    def _format_action_invocation(self, action):
        if action.option_strings == [] or action.default == SUPPRESS or not self._parser.default_env:
            return super()._format_action_invocation(action)
        extr = ''
        if self._parser.default_env:
            extr += '\n  ENV:   ' + _get_env_var(self, action)
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
            if hasattr(self, 'defaults'):
                params['default'] = _get_key_value(self.defaults, action.dest)
            if params['default'] is None:
                params['default'] = 'null'
            elif isinstance(params['default'], Enum) and hasattr(params['default'], 'name'):
                params['default'] = action.default.name
        return PercentTemplate(self._get_help_string(action)).safe_substitute(params)


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
        actions = [a for a in actions if not isinstance(a, _ActionLink)]
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

        parsers = get_subparsers(self._parser)  # type: ignore
        parsers[None] = self._parser  # type: ignore

        group_titles = {}
        for prefix, parser in parsers.items():
            for group in parser._action_groups:
                actions = filter_default_actions(group._group_actions)
                actions = [a for a in actions if not isinstance(a, (_ActionConfigLoad, ActionConfigFile, _ActionSubCommands))]
                keys = set(re.sub(r'\.?[^.]+$', '', a.dest) for a in actions)
                if len(keys) == 1:
                    key = keys.pop()
                    full_key = (prefix+('.' if key else '') if prefix else '')+key
                    group_titles[full_key] = group.title

        def set_comments(cfg, prefix='', depth=0):
            for key in cfg.keys():
                full_key = (prefix+'.' if prefix else '')+key
                action = _find_action(self._parser, full_key, within_subcommands=True)
                if isinstance(action, tuple):
                    action = action[0]
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

        if self._parser.description is not None:  # type: ignore
            self.set_yaml_start_comment(self._parser.description, cfg)  # type: ignore
        set_comments(cfg)
        out = StringIO()
        yaml.dump(cfg, out)
        return out.getvalue()


    def set_yaml_start_comment(
        self,
        text: str,
        cfg: 'ruyaml.comments.CommentedMap',  # type: ignore
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
        cfg: 'ruyaml.comments.CommentedMap',  # type: ignore
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
        cfg: 'ruyaml.comments.CommentedMap',  # type: ignore
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
