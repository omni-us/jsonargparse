"""Collection of useful actions to define arguments."""

import inspect
import os
import re
import sys
import warnings
from argparse import Action, SUPPRESS, _HelpAction, _SubParsersAction
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from .loaders_dumpers import get_loader_exceptions, load_value
from .namespace import is_empty_namespace, Namespace, split_key, split_key_leaf, split_key_root
from .optionals import FilesCompleterMethod, get_config_read_mode
from .type_checking import ArgumentParser, _ArgumentGroup
from .typing import get_import_path, path_type
from .util import (
    default_config_option_help,
    DirectedGraph,
    ParserError,
    import_object,
    change_to_path_dir,
    NoneType,
    indent_text,
    Path,
    _parse_value_or_config,
    _issubclass,
)


__all__ = [
    'ActionConfigFile',
    'ActionYesNo',
    'ActionParser',
    'ActionPathList',
]


def _is_branch_key(parser, key: str) -> bool:
    root_key = split_key_root(key)[0]
    for action in filter_default_actions(parser._actions):
        if isinstance(action, _ActionSubCommands) and root_key in action._name_parser_map:
            subparser = action._name_parser_map[root_key]
            return _is_branch_key(subparser, split_key_root(key)[1])
        elif action.dest.startswith(key+'.'):
            return True
    return False


def _find_action_and_subcommand(
    parser: 'ArgumentParser',
    dest: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Tuple[Optional[Action], Optional[str]]:
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
                    subcommand += '.' + subsubcommand
                return subaction, subcommand
    return fallback_action, None


def _find_action(
    parser: 'ArgumentParser',
    dest: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Optional[Action]:
    return _find_action_and_subcommand(parser, dest, exclude=exclude)[0]


def _find_parent_action_and_subcommand(
    parser: 'ArgumentParser',
    key: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Tuple[Optional[Action], Optional[str]]:
    action, subcommand = _find_action_and_subcommand(parser, key, exclude=exclude)
    if action is None and '.' in key:
        parts = split_key(key)
        for n in reversed(range(len(parts)-1)):
            action, subcommand = _find_action_and_subcommand(parser, '.'.join(parts[:n+1]), exclude=exclude)
            if action is not None:
                break
    return action, subcommand


def _find_parent_action(
    parser: 'ArgumentParser',
    key: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Optional[Action]:
    return _find_parent_action_and_subcommand(parser, key, exclude=exclude)[0]


def _find_parent_actions(
    parser: 'ArgumentParser',
    key: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Optional[List[Action]]:
    found: List[Action]
    action = _find_parent_action(parser, key, exclude=exclude)
    if action is not None:
        found = [action]
    else:
        actions = filter_default_actions(parser._actions)
        if exclude is not None:
            actions = [a for a in actions if not isinstance(a, exclude)]
        parts = split_key(key)
        for n in reversed(range(len(parts))):
            prefix = '.'.join(parts[:n+1])+'.'
            found = [a for a in actions if a.dest.startswith(prefix)]
            if found != []:
                break
    return None if found == [] else found


def _find_subclass_action_or_class_group(
    parser: 'ArgumentParser',
    key: str,
    exclude: Optional[Union[Type[Action], Tuple[Type[Action], ...]]] = None,
) -> Optional[Union[Action, '_ArgumentGroup']]:
    from .typehints import ActionTypeHint
    action = _find_parent_action(parser, key, exclude=exclude)
    if ActionTypeHint.is_class_typehint(action):
        return action
    key_set = {key, split_key_leaf(key)[0]}
    for group in parser._action_groups:
        if getattr(group, 'dest', None) in key_set and hasattr(group, 'instantiate_class'):
            return group
    return None


def _is_action_value_list(action: Action) -> bool:
    """Checks whether an action produces a list value.

    Args:
        action: An argparse action to check.

    Returns:
        bool: True if produces list otherwise False.
    """
    if action.nargs in {'*', '+'} or (isinstance(action.nargs, int) and action.nargs != 0):
        return True
    return False


def _remove_actions(parser, types):

    def remove(actions):
        rm_actions = [a for a in actions if isinstance(a, types)]
        for action in rm_actions:
            actions.remove(action)

    remove(parser._actions)
    remove(parser._action_groups[1]._group_actions)


def filter_default_actions(actions):
    default = (_HelpAction, _ActionHelpClass, _ActionPrintConfig)
    if isinstance(actions, list):
        return [a for a in actions if not isinstance(a, default)]
    return {k: a for k, a in actions.items() if not isinstance(a, default)}


class ActionConfigFile(Action, FilesCompleterMethod):
    """Action to indicate that an argument is a configuration file or a configuration string."""

    def __init__(self, **kwargs):
        """Initializer for ActionConfigFile instance."""
        if 'default' in kwargs:
            raise ValueError('default not allowed for ActionConfigFile, use default_config_files.')
        opt_name = kwargs['option_strings']
        opt_name = opt_name[0] if len(opt_name) == 1 else [x for x in opt_name if x[0:2] == '--'][0]
        if '.' in opt_name:
            raise ValueError('ActionConfigFile must be a top level option.')
        if 'help' not in kwargs:
            kwargs['help'] = 'Path to a configuration file.'
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_config(parser, cfg, self.dest, values)

    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
        with _ActionSubCommands.not_single_subcommand():
            if dest not in cfg:
                cfg[dest] = []
            kwargs = {'env': False, 'defaults': False, '_skip_check': True, '_fail_no_subcommand': False}
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
            cfg[dest].append(cfg_path)
            cfg.update(cfg_file)


print_config_skip: ContextVar = ContextVar('print_config_skip', default=False)


class _ActionPrintConfig(Action):
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS):
        super().__init__(option_strings=option_strings,
                         dest=dest,
                         default=default,
                         nargs=1,
                         metavar='[={comments,skip_null}+]',
                         help='Print configuration and exit.')

    def __call__(self, parser, namespace, value, option_string=None):
        kwargs = {'subparser': parser, 'key': None, 'skip_none': False, 'skip_check': True}
        valid_flags = {'': None, 'comments': 'yaml_comments', 'skip_null': 'skip_none'}
        if value is not None:
            flags = value[0].split(',')
            invalid_flags = [f for f in flags if f not in valid_flags]
            if len(invalid_flags) > 0:
                raise ParserError(f'Invalid option "{invalid_flags[0]}" for {option_string}')
            for flag in [f for f in flags if f != '']:
                kwargs[valid_flags[flag]] = True
        while hasattr(parser, 'parent_parser'):
            kwargs['key'] = parser.subcommand if kwargs['key'] is None else parser.subcommand+'.'+kwargs['key']
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
        if hasattr(parser, 'print_config') and not print_config_skip.get():
            key = parser.print_config.pop('key')
            subparser = parser.print_config.pop('subparser')
            if key is not None:
                cfg = cfg[key]
            sys.stdout.write(subparser.dump(cfg, **parser.print_config))
            delattr(parser, 'print_config')
            parser.exit()


class _ActionConfigLoad(Action):

    def __init__(
        self,
        basetype: Type = None,
        **kwargs
    ):
        if len(kwargs) == 0:
            self._basetype = basetype
        else:
            self.basetype = kwargs.pop('_basetype', None)
            kwargs['metavar'] = 'CONFIG'
            kwargs['help'] = default_config_option_help
            kwargs['default'] = SUPPRESS
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['_basetype'] = self._basetype
            return _ActionConfigLoad(**kwargs)
        parser, namespace, value = args[:3]
        cfg_file = Namespace()
        cfg_file[self.dest] = self._load_config(value, parser)
        namespace.update(cfg_file)

    def _load_config(self, value, parser):
        try:
            cfg, cfg_path = _parse_value_or_config(value)
            if not isinstance(cfg, dict):
                raise TypeError(f'Parser key "{self.dest}": Unable to load config "{value}"')
            with change_to_path_dir(cfg_path):
                cfg = parser._apply_actions(cfg, parent_key=self.dest)
            return cfg
        except (TypeError,) + get_loader_exceptions() as ex:
            str_ex = indent_text(str(ex))
            raise TypeError(f'Parser key "{self.dest}": Unable to load config "{value}"\n- {str_ex}') from ex

    def check_type(self, value, parser):
        return self._load_config(value, parser)

    def instantiate_classes(self, value):
        return self.basetype(**value)


class _ActionHelpClass(Action):

    def __init__(self, baseclass=None, **kwargs):
        if baseclass is not None:
            if getattr(baseclass, '__origin__', None) == Union:
                baseclasses = [c for c in baseclass.__args__ if c is not NoneType]
                if len(baseclasses) == 1:
                    baseclass = baseclasses[0]
            self._baseclass = baseclass
        else:
            self._baseclass = kwargs.pop('_baseclass')
            self.update_init_kwargs(kwargs)
            super().__init__(**kwargs)

    def update_init_kwargs(self, kwargs):
        kwargs.update({
            'nargs': 0,
            'default': SUPPRESS,
            'help': f'Show the help for the class {self._baseclass.__name__} and exit.',
        })

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['_baseclass'] = self._baseclass
            return type(self)(**kwargs)
        dest = re.sub('\\.help$', '',  self.dest)
        self.print_help(args, self._baseclass, dest)

    def print_help(self, call_args, val_class, dest):
        tmp = import_object('jsonargparse.ArgumentParser')()
        tmp.add_class_arguments(val_class, dest, **self.sub_add_kwargs)
        _remove_actions(tmp, (_HelpAction, _ActionHelpClass, _ActionPrintConfig))
        tmp.print_help()
        call_args[0].exit()


class _ActionHelpClassPath(_ActionHelpClass):

    def update_init_kwargs(self, kwargs):
        if getattr(self._baseclass, '__origin__', None) == Union:
            self._basename = '{'+', '.join(get_import_path(c) for c in self._baseclass.__args__)+'}'
        else:
            self._basename = get_import_path(self._baseclass)
        kwargs.update({
            'metavar': 'CLASS',
            'default': SUPPRESS,
            'help': f'Show the help for the given subclass of {self._basename} and exit.',
        })

    def print_help(self, call_args, baseclass, dest):
        try:
            val_class = import_object(call_args[2])
        except Exception as ex:
            raise TypeError(f'{call_args[3]}: {ex}')
        if getattr(self._baseclass, '__origin__', None) == Union:
            baseclasses = self._baseclass.__args__
        else:
            baseclasses = [baseclass]
        if not any(_issubclass(val_class, b) for b in baseclasses):
            raise TypeError(f'{call_args[3]}: Class "{call_args[2]}" is not a subclass of {self._basename}')
        super().print_help(call_args, val_class, dest+'.init_args')


class _ActionLink(Action):

    def __init__(
        self,
        parser,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Optional[Callable] = None,
        apply_on: str = 'parse',
    ):
        self.parser = parser

        # Set and check apply_on
        self.apply_on = apply_on
        if apply_on not in {'parse', 'instantiate'}:
            raise ValueError("apply_on must be 'parse' or 'instantiate'.")

        # Set and check compute function
        self.compute_fn = compute_fn
        if compute_fn is None and not isinstance(source, str):
            raise ValueError('Multiple source keys requires a compute function.')

        # Set and check source actions or group
        exclude = (_ActionLink, _ActionConfigLoad, _ActionSubCommands, ActionConfigFile)
        source = (source,) if isinstance(source, str) else source
        if apply_on == 'instantiate':
            if len(source) != 1:
                raise ValueError('Links applied on instantiation only supported for a single source.')
            self.source = [(source[0], _find_subclass_action_or_class_group(parser, source[0], exclude=exclude))]
            if self.source[0][1] is None:
                raise ValueError('Links applied on instantiation require source to be a subclass action or a class group.')
            if '.' in self.source[0][1].dest:
                raise ValueError('Links applied on instantiation only supported for first level objects.')
            if source[0] == self.source[0][1].dest and compute_fn is None:
                raise ValueError('Links applied on instantiation with object as source requires a compute function.')
        else:
            self.source = [(s, _find_parent_actions(parser, s, exclude=exclude)) for s in source]

        # Set and check target action
        self.target = (target, _find_parent_action(parser, target, exclude=exclude))
        for key, action in self.source + [self.target]:
            if action is None:
                raise ValueError(f'No action for key "{key}".')
        assert self.target[1] is not None

        from .typehints import ActionTypeHint
        is_target_subclass = ActionTypeHint.is_subclass_typehint(self.target[1])
        valid_target_subclass = is_target_subclass and target.startswith(self.target[1].dest+'.init_args.')
        valid_target_leaf = self.target[1].dest == target and not is_target_subclass
        if not (valid_target_leaf or valid_target_subclass):
            raise ValueError(f'Target key "{target}" must be for an individual argument.')

        # Replace target action with link action
        if not is_target_subclass:
            for key in self.target[1].option_strings:
                parser._option_string_actions[key] = self
            parser._actions[parser._actions.index(self.target[1])] = self
            for group in parser._action_groups:
                if self.target[1] in group._group_actions:
                    group._group_actions.remove(self.target[1])

        # Remove target from required
        if target in parser.required_args:
            parser.required_args.remove(target)
        if is_target_subclass:
            sub_add_kwargs = getattr(self.target[1], 'sub_add_kwargs')
            if 'linked_targets' not in sub_add_kwargs:
                sub_add_kwargs['linked_targets'] = set()
            subtarget = target.split('.init_args.', 1)[1]
            sub_add_kwargs['linked_targets'].add(subtarget)

        # Add link action to group to show in help
        if not hasattr(parser, '_links_group'):
            parser._links_group = parser.add_argument_group('Linked arguments')
        parser._links_group._group_actions.append(self)

        # Check instantiation link does not create cycle
        if apply_on == 'instantiate':
            try:
                self.instantiation_order(parser)
            except ValueError as ex:
                raise ValueError(f'Invalid link {source[0]} --> {target}: {ex}') from ex

        # Initialize link action
        link_str = target+' <-- '
        if compute_fn is None:
            link_str += source[0]
        else:
            link_str += getattr(compute_fn, '__name__', str(compute_fn))+'('+', '.join(source)+')'

        help_str: Optional[str]
        if is_target_subclass:
            type_attr = None
            help_str = f'Use --{self.target[1].dest}.help CLASS_PATH for details.'
        else:
            type_attr = getattr(self.target[1], '_typehint', self.target[1].type)
            help_str = self.target[1].help

        super().__init__(
            [link_str],
            dest=target,
            default=SUPPRESS,
            metavar=f'[applied on {self.apply_on}]',
            type=type_attr,
            help=help_str,
        )

    def __call__(self, *args, **kwargs):
        source = ', '.join(s[0] for s in self.source)
        raise TypeError(f'Linked "{self.target[0]}" must be given via "{source}".')

    def _check_type(self, value, cfg=None):
        return self.parser._check_value_key(self.target[1], value, self.target[0], cfg)

    @staticmethod
    def apply_parsing_links(parser: 'ArgumentParser', cfg: Namespace) -> None:
        subcommand, subparser = _ActionSubCommands.get_subcommand(parser, cfg, fail_no_subcommand=False)
        if subcommand:
            _ActionLink.apply_parsing_links(subparser, cfg[subcommand])
        if not hasattr(parser, '_links_group'):
            return
        for action in parser._links_group._group_actions:
            if action.apply_on != 'parse':
                continue
            try:
                args = []
                for key, _ in action.source:
                    args.append(cfg[key])
            except KeyError:
                continue
            from .typehints import ActionTypeHint
            if action.compute_fn is None:
                value = args[0]
                # Automatic namespace to dict based on link target type hint
                target_key, target_action = action.target
                if isinstance(value, Namespace) and isinstance(target_action, ActionTypeHint):
                    same_key = target_key == target_action.dest
                    if (same_key and target_action.is_mapping_typehint(target_action._typehint)) or \
                       target_action.is_init_arg_mapping_typehint(target_key, cfg):
                        value = value.as_dict()
            else:
                # Automatic namespace to dict based on compute_fn param type hint
                params = list(inspect.signature(action.compute_fn).parameters.values())
                for n, param in enumerate(params):
                    if n < len(args) and isinstance(args[n], Namespace) and ActionTypeHint.is_mapping_typehint(param.annotation):
                        args[n] = args[n].as_dict()
                # Compute value
                value = action.compute_fn(*args)
            _ActionLink.set_target_value(action, value, cfg)

    @staticmethod
    def apply_instantiation_links(parser, cfg, source):
        if not hasattr(parser, '_links_group'):
            return
        for action in parser._links_group._group_actions:
            if action.apply_on != 'instantiate' or source != action.source[0][1].dest:
                continue
            source_object = cfg[source]
            if action.source[0][0] == action.source[0][1].dest:
                value = action.compute_fn(source_object)
            else:
                attr = split_key_leaf(action.source[0][0])[1]
                value = getattr(source_object, attr)
                if action.compute_fn is not None:
                    value = action.compute_fn(value)
            _ActionLink.set_target_value(action, value, cfg)

    @staticmethod
    def set_target_value(action: '_ActionLink', value: Any, cfg: Namespace) -> None:
        key = action.target[0]
        cfg[key] = value

    @staticmethod
    def instantiation_order(parser):
        if hasattr(parser, '_links_group'):
            actions = [a for a in parser._links_group._group_actions if a.apply_on == 'instantiate']
            if len(actions) > 0:
                graph = DirectedGraph()
                for action in actions:
                    source = action.source[0][1].dest
                    target = re.sub(r'\.init_args$', '', split_key_leaf(action.target[0])[0])
                    graph.add_edge(source, target)
                return graph.get_topological_order()
        return []

    @staticmethod
    def reorder(order, components):
        ordered = []
        for key in order:
            after = []
            for component in components:
                if key == component.dest or component.dest.startswith(key+'.'):
                    ordered.append(component)
                else:
                    after.append(component)
            components = after
        return ordered + components

    @staticmethod
    def strip_link_target_keys(parser, cfg):
        def del_taget_key(target_key):
            cfg.pop(target_key, None)
            parent_key, _ = split_key_leaf(target_key)
            if '.' in target_key and is_empty_namespace(cfg.get(parent_key)):
                del cfg[parent_key]

        for action in [a for a in parser._actions if isinstance(a, _ActionLink)]:
            del_taget_key(action.target[0])
        from .typehints import ActionTypeHint
        for action in [a for a in parser._actions if isinstance(a, ActionTypeHint) and hasattr(a, 'sub_add_kwargs')]:
            for key in action.sub_add_kwargs.get('linked_targets', []):
                del_taget_key(action.dest+'.init_args.'+key)

        with _ActionSubCommands.not_single_subcommand():
            subcommands, subparsers = _ActionSubCommands.get_subcommands(parser, cfg)
        if subcommands is not None:
            for num, subcommand in enumerate(subcommands):
                if subcommand in cfg:
                    _ActionLink.strip_link_target_keys(subparsers[num], cfg[subcommand])


class ActionYesNo(Action):
    """Paired options --[yes_prefix]opt, --[no_prefix]opt to set True or False respectively."""

    def __init__(
        self,
        yes_prefix: str = '',
        no_prefix: str = 'no_',
        **kwargs
    ):
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
            self._yes_prefix = kwargs.pop('_yes_prefix') if '_yes_prefix' in kwargs else ''
            self._no_prefix = kwargs.pop('_no_prefix') if '_no_prefix' in kwargs else 'no_'
            if len(kwargs['option_strings']) == 0:
                raise ValueError(f'{type(self).__name__} not intended for positional arguments  ({kwargs["dest"]}).')
            opt_name = kwargs['option_strings'][0]
            if not opt_name.startswith('--'+self._yes_prefix):
                raise ValueError(f'Expected option string to start with "--{self._yes_prefix}".')
            if self._no_prefix is not None:
                kwargs['option_strings'] += [re.sub('^--'+self._yes_prefix, '--'+self._no_prefix, opt_name)]
            if self._no_prefix is None and 'nargs' in kwargs and kwargs['nargs'] != 1:
                raise ValueError('ActionYesNo with no_prefix=None only supports nargs=1.')
            if 'nargs' in kwargs and kwargs['nargs'] in {'?', 1}:
                kwargs['metavar'] = '{true,yes,false,no}'
                if kwargs['nargs'] == 1:
                    kwargs['nargs'] = None
            else:
                kwargs['nargs'] = 0
                kwargs['metavar'] = None
            if 'default' not in kwargs:
                kwargs['default'] = False
            kwargs['type'] = ActionYesNo._boolean_type
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Sets the corresponding key to True or False depending on the option string used."""
        if len(args) == 0:
            kwargs['_yes_prefix'] = self._yes_prefix
            kwargs['_no_prefix'] = self._no_prefix
            return ActionYesNo(**kwargs)
        value = args[2] if isinstance(args[2], bool) else True
        if self._no_prefix is not None and args[3].startswith('--'+self._no_prefix):
            setattr(args[1], self.dest, not value)
        else:
            setattr(args[1], self.dest, value)

    def _add_dest_prefix(self, prefix):
        self.dest = prefix+'.'+self.dest
        self.option_strings[0] = re.sub('^--'+self._yes_prefix, '--'+self._yes_prefix+prefix+'.', self.option_strings[0])
        if self._no_prefix is not None:
            self.option_strings[-1] = re.sub('^--'+self._no_prefix, '--'+self._no_prefix+prefix+'.', self.option_strings[-1])

    def _check_type(self, value, cfg=None):
        return ActionYesNo._boolean_type(value)

    @staticmethod
    def _boolean_type(x):
        if isinstance(x, str) and x.lower() in {'true', 'yes', 'false', 'no'}:
            x = True if x.lower() in {'true', 'yes'} else False
        elif not isinstance(x, bool):
            raise TypeError(f'Value not boolean: {x}.')
        return x

    def completer(self, **kwargs):
        """Used by argcomplete to support tab completion of arguments."""
        return ['true', 'false', 'yes', 'no']


class ActionParser:
    """Action to parse option with a given parser optionally loading from file if string value."""

    def __init__(
        self,
        parser: 'ArgumentParser' = None,
    ):
        """Initializer for ActionParser instance.

        Args:
            parser: A parser to parse the option with.

        Raises:
            ValueError: If the parser parameter is invalid.
        """
        self._parser = parser
        if not isinstance(self._parser, import_object('jsonargparse.ArgumentParser')):
            raise ValueError('Expected parser keyword argument to be an ArgumentParser.')


    @staticmethod
    def _move_parser_actions(parser, args, kwargs):
        subparser = kwargs.pop('action')._parser
        title = kwargs.pop('title', kwargs.pop('help', None))
        description = kwargs.pop('description', subparser.description)
        if len(kwargs) > 0:
            raise ValueError(f'ActionParser does not accept the following parameters: {set(kwargs.keys())}')
        if not (len(args) == 1 and args[0][:2] == '--'):
            raise ValueError(f'ActionParser only accepts a single optional key but got {args}')
        prefix = args[0][2:]

        def add_prefix(key):
            return re.sub('^--', '--'+prefix+'.', key)

        required_args = set(prefix+'.'+x for x in subparser.required_args)

        option_string_actions = {}
        for key, action in filter_default_actions(subparser._option_string_actions).items():
            option_string_actions[add_prefix(key)] = action

        isect = set(option_string_actions.keys()).intersection(set(parser._option_string_actions.keys()))
        if len(isect) > 0:
            raise ValueError(f'ActionParser conflicting keys: {isect}')

        actions = []
        dest = prefix.replace('-', '_')
        for action in filter_default_actions(subparser._actions):
            if isinstance(action, ActionYesNo):
                action._add_dest_prefix(prefix)
            else:
                action.dest = dest+'.'+action.dest
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
        parser._action_groups.extend([base_action_group]+extra_action_groups)

        subparser._option_string_actions = {}
        subparser._actions = []
        subparser._action_groups = []

        return base_action_group


single_subcommand: ContextVar = ContextVar('single_subcommand', default=True)

parent_parsers: ContextVar = ContextVar('parent_parsers', default=[])


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

    _env_prefix: Optional[str] = None


    def add_parser(self, name, **kwargs):
        """Raises a NotImplementedError."""
        raise NotImplementedError('In jsonargparse subcommands are added using the add_subcommand method.')


    def add_subcommand(self, name, parser, **kwargs):
        """Adds a parser as a sub-command parser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        add_parser requires to be given a parser as argument.
        """
        if parser._subparsers is not None:
            raise ValueError('Multiple levels of subcommands must be added in level order.')

        parser.prog = f'{self._prog_prefix} [options] {name}'
        parser.env_prefix = f'{self._env_prefix}_{name}_'
        parser.parent_parser = self.parent_parser
        parser.parser_mode = self.parent_parser.parser_mode
        parser.subcommand = name

        # create a pseudo-action to hold the choice help
        aliases = kwargs.pop('aliases', ())
        if 'help' in kwargs:
            help_arg = kwargs.pop('help')
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
            subnamespace, unk = subparser.parse_known_args(arg_strings)
            if unk:
                raise ParserError(f'Unrecognized arguments: {" ".join(unk)}')
            namespace.update(subnamespace, subcommand)


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
        parser: 'ArgumentParser',
        cfg: Namespace,
        prefix: str = '',
        fail_no_subcommand: bool = True,
    ) -> Tuple[Optional[List[str]], Optional[List['ArgumentParser']]]:
        """Returns subcommand names and corresponding subparsers."""
        if parser._subparsers is None:
            return None, None
        action = parser._subcommands_action

        require_single = single_subcommand.get()

        # Get subcommand settings keys
        subcommand_keys = [k for k in action.choices.keys() if isinstance(cfg.get(prefix+k), Namespace)]

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
                del cfg[prefix+key]

        if subcommand:
            subcommand_keys = [subcommand]

        if fail_no_subcommand:
            if subcommand is None and not (fail_no_subcommand and action._required):
                return None, None
            if action._required and subcommand not in action._name_parser_map:
                raise KeyError(f'"{dest}" is required but not given or its value is None.')

        return subcommand_keys, [action._name_parser_map.get(s) for s in subcommand_keys]


    @staticmethod
    def get_subcommand(
        parser: 'ArgumentParser',
        cfg: Namespace,
        prefix: str = '',
        fail_no_subcommand: bool = True,
    ) -> Tuple[Optional[str], Optional['ArgumentParser']]:
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
        parser: 'ArgumentParser',
        cfg: Namespace,
        env: Optional[bool],
        defaults: bool,
        prefix: str = '',
        fail_no_subcommand: bool = True,
    ) -> None:
        """Takes care of parsing subcommand values."""

        subcommands, subparsers = _ActionSubCommands.get_subcommands(parser, cfg, prefix=prefix, fail_no_subcommand=fail_no_subcommand)
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
                _ActionSubCommands.handle_subcommands(subparser, cfg, env, defaults, key+'.', fail_no_subcommand=fail_no_subcommand)


class ActionPathList(Action, FilesCompleterMethod):
    """Action to check and store a list of file paths read from a plain text file or stream."""

    def __init__(
        self,
        mode: str = None,
        rel: str = 'cwd',
        **kwargs
    ):
        """Initializer for ActionPathList instance.

        Args:
            mode: The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument (uppercase means not), e.g. ActionPathList(mode='fr').
            rel: Whether relative paths are with respect to current working directory 'cwd' or the list's parent directory 'list'.

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if mode is not None:
            self._type = path_type(mode)
            self._rel = rel
            if self._rel not in {'cwd', 'list'}:
                raise ValueError(f'rel must be either "cwd" or "list", got {self._rel}.')
        elif '_type' not in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._type = kwargs.pop('_type')
            self._rel = kwargs.pop('_rel')
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a PathList and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid PathList.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] not in {'+', 1}:
                raise ValueError('ActionPathList only supports nargs of 1 or "+".')
            kwargs['_type'] = self._type
            kwargs['_rel'] = self._rel
            return ActionPathList(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None):
        if value == []:
            return value
        islist = _is_action_value_list(self)
        if not islist and not isinstance(value, list):
            value = [value]
        if isinstance(value, list) and all(not isinstance(v, self._type) for v in value):
            path_list_files = value
            value = []
            for path_list_file in path_list_files:
                try:
                    with sys.stdin if path_list_file == '-' else open(path_list_file, 'r') as f:
                        path_list = [x.strip() for x in f.readlines()]
                except FileNotFoundError as ex:
                    raise TypeError(f'Problems reading path list: {path_list_file} :: {ex}') from ex
                cwd = os.getcwd()
                if self._rel == 'list' and path_list_file != '-':
                    os.chdir(os.path.abspath(os.path.join(path_list_file, os.pardir)))
                try:
                    for num, val in enumerate(path_list):
                        try:
                            path_list[num] = self._type(val)
                        except TypeError as ex:
                            raise TypeError(f'Path number {num+1} in list {path_list_file}, {ex}') from ex
                finally:
                    os.chdir(cwd)
                value += path_list
        return value
