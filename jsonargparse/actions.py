"""Collection of useful actions to define arguments."""

import os
import re
import sys
import yaml
import argparse
from typing import Callable, Tuple, Type, Union
from argparse import Namespace, Action, SUPPRESS, _HelpAction, _SubParsersAction

from .optionals import get_config_read_mode, FilesCompleterMethod
from .util import (
    yamlParserError,
    yamlScannerError,
    ParserError,
    namespace_to_dict,
    dict_to_namespace,
    import_object,
    change_to_path_dir,
    Path,
    _get_key_value,
    _load_config,
    _dict_to_flat_namespace,
    _issubclass
)


__all__ = [
    'ActionConfigFile',
    'ActionYesNo',
    'ActionParser',
    'ActionPath',
    'ActionPathList',
]


def _find_action(parser, dest:str, within_subcommands:bool=False, exclude=None):
    """Finds an action in a parser given its dest.

    Args:
        parser (ArgumentParser): A parser where to search.
        dest: The dest string to search with.

    Returns:
        Action or None: The action if found, otherwise None.
    """
    actions = filter_default_actions(parser._actions)
    if exclude is not None:
        actions = [a for a in actions if not isinstance(a, exclude)]
    for action in actions:
        if action.dest == dest:
            return action
        elif isinstance(action, _ActionSubCommands):
            if dest in action._name_parser_map:
                return action
            elif within_subcommands and dest.split('.', 1)[0] in action._name_parser_map:
                subcommand, subdest = dest.split('.', 1)
                subparser = action._name_parser_map[subcommand]
                return _find_action(subparser, subdest, True, exclude=exclude)
    return None


def _find_parent_action(parser, key:str, exclude=None):
    action = _find_action(parser, key, exclude=exclude)
    if action is None and '.' in key:
        parts = key.split('.')
        for n in reversed(range(len(parts)-1)):
            action = _find_action(parser, '.'.join(parts[:n+1]), exclude=exclude)
            if action is not None:
                break
    return action


def _find_parent_actions(parser, key:str, exclude=None):
    action = _find_parent_action(parser, key, exclude=exclude)
    if action is None:
        actions = filter_default_actions(parser._actions)
        if exclude is not None:
            actions = [a for a in actions if not isinstance(a, exclude)]
        parts = key.split('.')
        for n in reversed(range(len(parts))):
            prefix = '.'.join(parts[:n+1])+'.'
            action = [a for a in actions if a.dest.startswith(prefix)]
            if action != []:
                break
    return None if action == [] else action


def _is_action_value_list(action:Action):
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
    default = (_HelpAction, _ActionHelpClassPath, _ActionPrintConfig)
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

    def __call__(self, parser, namespace, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self._apply_config(parser, namespace, self.dest, values)

    @staticmethod
    def _apply_config(parser, namespace, dest, value):
        if not hasattr(namespace, dest) or not isinstance(getattr(namespace, dest), list):
            setattr(namespace, dest, [])
        kwargs = {'env': False, 'defaults': False, '_skip_check': True, '_fail_no_subcommand': False}
        try:
            cfg_path = Path(value, mode=get_config_read_mode())
        except TypeError as ex_path:
            try:
                if isinstance(yaml.safe_load(value), str):
                    raise ex_path
                cfg_path = None
                cfg_file = parser.parse_string(value, **kwargs)
            except (TypeError, yamlParserError, yamlScannerError) as ex_str:
                raise TypeError('Parser key "'+dest+'": '+str(ex_str)) from ex_str
        else:
            cfg_file = parser.parse_path(value, **kwargs)
        cfg_file = _dict_to_flat_namespace(namespace_to_dict(cfg_file))
        getattr(namespace, dest).append(cfg_path)
        for key, val in vars(cfg_file).items():
            setattr(namespace, key, val)


class _ActionPrintConfig(Action):
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS):
        super().__init__(option_strings=option_strings,
                         dest=dest,
                         default=default,
                         nargs='?',
                         metavar='skip_null',
                         help='Print configuration and exit.')

    def __call__(self, parser, namespace, value, option_string=None):
        kwargs = {'skip_none': False}
        if value is not None and 'skip_null' in value:
            kwargs['skip_none'] = True
        parser.print_config = kwargs

    @staticmethod
    def print_config_if_requested(parser, cfg):
        if hasattr(parser, 'print_config') and not hasattr(parser, 'print_config_skip'):
            sys.stdout.write(parser.dump(cfg, skip_check=True, **parser.print_config))
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
            kwargs['help'] = SUPPRESS
            kwargs['default'] = SUPPRESS
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['_basetype'] = self._basetype
            return _ActionConfigLoad(**kwargs)
        parser, namespace, value = args[:3]
        cfg_file = self._load_config(value)
        cfg_file = {self.dest+'.'+k: v for k, v in vars(cfg_file).items()}
        with change_to_path_dir(cfg_file.get(self.dest+'.__path__')):
            parser._apply_actions(cfg_file)
        for key, val in cfg_file.items():
            setattr(namespace, key, val)

    def _load_config(self, value):
        try:
            return _load_config(value)
        except (TypeError, yamlParserError, yamlScannerError) as ex:
            raise TypeError('Parser key "'+self.dest+'": '+str(ex)) from ex

    def _check_type(self, value, cfg=None):
        return self._load_config(value)

    def _instantiate_classes(self, value):
        return self.basetype(**value)


class _ActionHelpClassPath(Action):

    def __init__(self, baseclass=None, **kwargs):
        if baseclass is not None:
            self._baseclass = baseclass
        else:
            self._baseclass = kwargs.pop('_baseclass')
            kwargs['help'] = 'Show the help for the given class path and exit.'
            kwargs['metavar'] = 'CLASS'
            kwargs['default'] = SUPPRESS
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['_baseclass'] = self._baseclass
            return _ActionHelpClassPath(**kwargs)
        val_class = import_object(args[2])
        if not _issubclass(val_class, self._baseclass):
            raise TypeError('Class "'+args[2]+'" is not a subclass of '+self._baseclass.__name__)
        dest = re.sub('\\.help$', '',  self.dest) + '.init_args'
        tmp = import_object('jsonargparse.ArgumentParser')()
        tmp.add_class_arguments(val_class, dest)
        _remove_actions(tmp, (_HelpAction, _ActionPrintConfig))
        tmp.print_help()
        args[0].exit()


class _ActionLink(Action):

    def __init__(
        self,
        parser,
        source: Union[str, Tuple[str, ...]],
        target: str,
        compute_fn: Callable = None,
    ):
        self.parser = parser

        # Set and check compute function
        self.compute_fn = compute_fn
        if compute_fn is None and not isinstance(source, str):
            raise ValueError('Multiple source keys requires a compute function.')

        # Set and check source and target actions
        exclude = (_ActionLink, _ActionConfigLoad, _ActionSubCommands, ActionConfigFile)
        source = (source,) if isinstance(source, str) else source
        self.source = [(s, _find_parent_actions(parser, s, exclude=exclude)) for s in source]
        self.target = (target, _find_parent_action(parser, target, exclude=exclude))
        for key, action in self.source + [self.target]:
            if action is None:
                raise ValueError('No action for key "'+key+'".')

        from .typehints import ActionTypeHint
        is_target_subclass = ActionTypeHint.is_subclass_typehint(self.target[1])
        valid_target_subclass = is_target_subclass and target.startswith(self.target[1].dest+'.init_args.')
        valid_target_leaf = self.target[1].dest == target and not is_target_subclass
        if not (valid_target_leaf or valid_target_subclass):
            raise ValueError('Target key "'+target+'" must be for a individual argument.')

        # Replace target action with link action
        if not is_target_subclass:
            for key in self.target[1].option_strings:
                parser._option_string_actions[key] = self
            parser._actions[parser._actions.index(self.target[1])] = self
            for group in parser._action_groups:
                if self.target[1] in group._group_actions:
                    group._group_actions.remove(self.target[1])

        # Add link action to group to show in help
        if not hasattr(parser, '_links_group'):
            parser._links_group = parser.add_argument_group('Linked arguments')
        parser._links_group._group_actions.append(self)

        # Initialize link action
        link_str = target+' <= '
        if compute_fn is None:
            link_str += source[0]
        else:
            link_str += getattr(compute_fn, '__name__', str(compute_fn))+'('+', '.join(source)+')'

        if is_target_subclass:
            type_attr = None
            help_str = 'Use --'+self.target[1].dest+'.help CLASS_PATH for details.'
        else:
            type_attr = getattr(self.target[1], '_typehint', self.target[1].type)
            help_str = self.target[1].help

        super().__init__(
            [link_str],
            dest=target,
            default=SUPPRESS,
            metavar='',
            type=type_attr,
            help=help_str,
        )

    def __call__(self, *args, **kwargs):
        source = ', '.join(s[0] for s in self.source)
        raise TypeError('Linked "'+self.target[0]+'" must be given via "'+source+'".')

    def _check_type(self, value, cfg=None):
        return self.parser._check_value_key(self.target[1], value, self.target[0], cfg)

    @staticmethod
    def propagate_arguments(parser, cfg_ns):
        if hasattr(parser, '_links_group'):
            for action in parser._links_group._group_actions:
                try:
                    args = []
                    for key, _ in action.source:
                        arg = _get_key_value(cfg_ns, key)
                        if isinstance(arg, Namespace):
                            arg = namespace_to_dict(arg)
                        args.append(arg)
                except AttributeError:
                    continue
                if action.compute_fn is None:
                    value = args[0]
                else:
                    value = action.compute_fn(*args)
                key = action.target[0]
                parent_ns = cfg_ns
                while True:
                    if '.' not in key:
                        setattr(parent_ns, key, value)
                        break
                    parent_key, key = key.rsplit('.', 1)
                    try:
                        parent_ns = _get_key_value(cfg_ns, parent_key)
                    except AttributeError:
                        value = dict_to_namespace({key: value})
                        key = parent_key


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
                raise ValueError(type(self).__name__+' not intended for positional arguments  ('+kwargs['dest']+').')
            opt_name = kwargs['option_strings'][0]
            if not opt_name.startswith('--'+self._yes_prefix):
                raise ValueError('Expected option string to start with "--'+self._yes_prefix+'".')
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
            raise TypeError('Value not boolean: '+str(x)+'.')
        return x

    def completer(self, **kwargs):
        """Used by argcomplete to support tab completion of arguments."""
        return ['true', 'false', 'yes', 'no']


class ActionParser:
    """Action to parse option with a given parser optionally loading from file if string value."""

    def __init__(
        self,
        parser: argparse.ArgumentParser = None,
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
        title = kwargs.pop('title', None)
        description = kwargs.pop('description', subparser.description)
        if len(kwargs) > 0:
            raise ValueError('ActionParser does not accept '+str(set(kwargs.keys())))
        if not (len(args) == 1 and args[0][:2] == '--'):
            raise ValueError('ActionParser only accepts a single optional key but got '+str(args))
        prefix = args[0][2:]

        def add_prefix(key):
            return re.sub('^--', '--'+prefix+'.', key)

        required_args = set(prefix+'.'+x for x in subparser.required_args)

        option_string_actions = {}
        for key, action in filter_default_actions(subparser._option_string_actions).items():
            option_string_actions[add_prefix(key)] = action

        isect = set(option_string_actions.keys()).intersection(set(parser._option_string_actions.keys()))
        if len(isect) > 0:
            raise ValueError('ActionParser conflicting keys: '+str(isect))

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


class _ActionSubCommands(_SubParsersAction):
    """Extension of argparse._SubParsersAction to modify sub-commands functionality."""

    _env_prefix = None


    def add_parser(self, name, **kwargs):
        """Raises a NotImplementedError."""
        raise NotImplementedError('In jsonargparse sub-commands are added using the add_subcommand method.')


    def add_subcommand(self, name, parser, **kwargs):
        """Adds a parser as a sub-command parser.

        In contrast to `argparse.ArgumentParser.add_subparsers
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_
        add_parser requires to be given a parser as argument.
        """
        if parser._subparsers is not None:
            raise ValueError('Multiple levels of subcommands must be added in level order.')

        parser.prog = '%s [options] %s' % (self._prog_prefix, name)
        parser.env_prefix = self._env_prefix+'_'+name+'_'
        _remove_actions(parser, _ActionPrintConfig)

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
        setattr(namespace, self.dest, subcommand)

        # parse arguments
        if subcommand in self._name_parser_map:
            subparser = self._name_parser_map[subcommand]
            subnamespace, unk = subparser.parse_known_args(arg_strings)
            if unk:
                raise ParserError('Unrecognized arguments: %s' % ' '.join(unk))
            for key, value in vars(subnamespace).items():
                setattr(namespace, subcommand+'.'+key, value)


    @staticmethod
    def handle_subcommands(parser, cfg, env, defaults, prefix='', fail_no_subcommand=True):
        """Adds sub-command dest if missing and parses defaults and environment variables."""
        if parser._subparsers is None:
            return

        cfg_dict = cfg.__dict__ if isinstance(cfg, Namespace) else cfg
        cfg_keys = set(vars(_dict_to_flat_namespace(cfg)).keys())
        cfg_keys = cfg_keys.union(set(cfg_dict.keys()))

        # Get subcommands action
        for action in parser._actions:
            if isinstance(action, _ActionSubCommands):
                break

        # Get sub-command parser
        subcommand = None
        dest = prefix + action.dest
        if dest in cfg_dict and cfg_dict[dest] is not None:
            subcommand = cfg_dict[dest]
        else:
            for key in action.choices.keys():
                if any([v.startswith(key+'.') for v in cfg_dict.keys()]):
                    subcommand = key
                    break
            cfg_dict[dest] = subcommand

        if subcommand is None and not fail_no_subcommand:
            return
        if action._required and subcommand not in action._name_parser_map:
            raise KeyError('Sub-command "'+dest+'" is required but not given or its value is None.')

        subparser = action._name_parser_map[subcommand]

        # merge environment variable values and default values
        subnamespace = None
        if env:
            subnamespace = subparser.parse_env(defaults=defaults, nested=False, _skip_check=True)
        elif defaults:
            subnamespace = subparser.get_defaults(nested=False, skip_check=True)

        if subnamespace is not None:
            for key, value in vars(subnamespace).items():
                key = prefix + subcommand+'.'+key
                if key not in cfg_keys:
                    cfg_dict[key] = value

        if subparser._subparsers is not None:
            prefix = prefix + subcommand + '.'
            _ActionSubCommands.handle_subcommands(subparser, cfg, env, defaults, prefix)


class ActionPath(Action, FilesCompleterMethod):
    """Action to check and store a path."""

    def __init__(
        self,
        mode: str = None,
        skip_check: bool = False,
        **kwargs
    ):
        """Initializer for ActionPath instance.

        Args:
            mode: The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument, e.g. ActionPath(mode='drw').
            skip_check: Whether to skip path checks.

        Raises:
            ValueError: If the mode parameter is invalid.
        """
        if mode is not None:
            Path._check_mode(mode)
            self._mode = mode
            self._skip_check = skip_check
        elif '_mode' not in kwargs:
            raise ValueError('ActionPath expects mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a Path and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid Path.
        """
        if len(args) == 0:
            kwargs['_mode'] = self._mode
            kwargs['_skip_check'] = self._skip_check
            return ActionPath(**kwargs)
        if hasattr(self, 'nargs') and self.nargs == '?' and args[2] is None:
            setattr(args[1], self.dest, args[2])
        else:
            setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None, islist=None):
        islist = _is_action_value_list(self) if islist is None else islist
        if not islist:
            value = [value]
        try:
            for num, val in enumerate(value):
                if isinstance(val, Path):
                    val = Path(str(val), mode=self._mode, skip_check=self._skip_check, cwd=val.cwd)
                else:
                    val = Path(val, mode=self._mode, skip_check=self._skip_check)
                value[num] = val
        except TypeError as ex:
            raise TypeError('Parser key "'+self.dest+'": '+str(ex)) from ex
        return value if islist else value[0]


class ActionPathList(Action, FilesCompleterMethod):
    """Action to check and store a list of file paths read from a plain text file or stream."""

    def __init__(
        self,
        mode: str = None,
        skip_check: bool = False,
        rel: str = 'cwd',
        **kwargs
    ):
        """Initializer for ActionPathList instance.

        Args:
            mode: The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument (uppercase means not), e.g. ActionPathList(mode='fr').
            skip_check: Whether to skip path checks.
            rel: Whether relative paths are with respect to current working directory 'cwd' or the list's parent directory 'list'.

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if mode is not None:
            Path._check_mode(mode)
            self._mode = mode
            self._skip_check = skip_check
            self._rel = rel
            if self._rel not in {'cwd', 'list'}:
                raise ValueError('rel must be either "cwd" or "list", got '+str(self._rel)+'.')
        elif '_mode' not in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
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
            kwargs['_mode'] = self._mode
            kwargs['_skip_check'] = self._skip_check
            kwargs['_rel'] = self._rel
            return ActionPathList(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None):
        if value == []:
            return value
        islist = _is_action_value_list(self)
        if not islist and not isinstance(value, list):
            value = [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            path_list_files = value
            value = []
            for path_list_file in path_list_files:
                try:
                    with sys.stdin if path_list_file == '-' else open(path_list_file, 'r') as f:
                        path_list = [x.strip() for x in f.readlines()]
                except FileNotFoundError as ex:
                    raise TypeError('Problems reading path list: '+path_list_file+' :: '+str(ex)) from ex
                cwd = os.getcwd()
                if self._rel == 'list' and path_list_file != '-':
                    os.chdir(os.path.abspath(os.path.join(path_list_file, os.pardir)))
                try:
                    for num, val in enumerate(path_list):
                        try:
                            path_list[num] = Path(val, mode=self._mode)
                        except TypeError as ex:
                            raise TypeError('Path number '+str(num+1)+' in list '+path_list_file+', '+str(ex)) from ex
                finally:
                    os.chdir(cwd)
                value += path_list
            return value
        else:
            return ActionPath._check_type(self, value, islist=True)
