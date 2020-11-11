"""Collection of useful actions to define arguments."""

import os
import re
import sys
import yaml
import operator
import argparse
from enum import Enum
from argparse import ArgumentParser, Namespace, Action, SUPPRESS, _StoreAction

from .optionals import get_config_read_mode
from .typing import restricted_number_type
from .util import (ParserError, _flat_namespace_to_dict, _dict_to_flat_namespace, namespace_to_dict, 
                   dict_to_namespace, Path, _check_unknown_kwargs, _issubclass)


def _find_action(parser, dest):
    """Finds an action in a parser given its dest.

    Args:
        parser (ArgumentParser): A parser where to search.
        dest (str): The dest string to search with.

    Returns:
        Action or None: The action if found, otherwise None.
    """
    for action in parser._actions:
        if action.dest == dest:
            return action
        elif isinstance(action, ActionParser) and dest.startswith(action.dest+'.'):
            return _find_action(action._parser, dest)
        elif isinstance(action, _ActionSubCommands) and dest in action._name_parser_map:
            return action
    return None


def _is_action_value_list(action:Action):
    """Checks whether an action produces a list value.

    Args:
        action (Action): An argparse action to check.

    Returns:
        bool: True if produces list otherwise False.
    """
    if action.nargs in {'*', '+'} or isinstance(action.nargs, int):
        return True
    return False


class ActionConfigFile(Action):
    """Action to indicate that an argument is a configuration file or a configuration string."""

    def __init__(self, **kwargs):
        """Initializer for ActionConfigFile instance."""
        opt_name = kwargs['option_strings']
        opt_name = opt_name[0] if len(opt_name) == 1 else [x for x in opt_name if x[0:2] == '--'][0]
        if '.' in opt_name:
            raise ValueError('ActionConfigFile must be a top level option.')
        kwargs['type'] = str
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
        try:
            cfg_path = Path(value, mode=get_config_read_mode())
        except TypeError as ex_path:
            if isinstance(yaml.safe_load(value), str):
                raise ex_path
            try:
                cfg_path = None
                cfg_file = parser.parse_string(value, env=False, defaults=False, _skip_check=True)
            except TypeError as ex_str:
                raise TypeError('Parser key "'+dest+'": '+str(ex_str))
        else:
            cfg_file = parser.parse_path(value, env=False, defaults=False, _skip_check=True)
        cfg_file = _dict_to_flat_namespace(namespace_to_dict(cfg_file))
        getattr(namespace, dest).append(cfg_path)
        for key, val in vars(cfg_file).items():
            if key == '__cwd__' and hasattr(namespace, '__cwd__'):
                setattr(namespace, key, getattr(namespace, key)+val)
            else:
                setattr(namespace, key, val)


class _ActionPrintConfig(Action):
    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help='print configuration and exit'):
        super().__init__(option_strings=option_strings,
                         dest=dest,
                         default=default,
                         nargs=0,
                         help=help)

    def __call__(self, parser, *args, **kwargs):
        parser._print_config = True


class ActionYesNo(Action):
    """Paired options --{yes_prefix}opt, --{no_prefix}opt to set True or False respectively."""

    def __init__(self, **kwargs):
        """Initializer for ActionYesNo instance.

        Args:
            yes_prefix (str): Prefix for yes option (default='').
            no_prefix (str or None): Prefix for no option (default='no_').

        Raises:
            ValueError: If a parameter is invalid.
        """
        self._yes_prefix = ''
        self._no_prefix = 'no_'
        if 'yes_prefix' in kwargs or 'no_prefix' in kwargs or len(kwargs) == 0:
            _check_unknown_kwargs(kwargs, {'yes_prefix', 'no_prefix'})
            if 'yes_prefix' in kwargs:
                self._yes_prefix = kwargs['yes_prefix']
            if 'no_prefix' in kwargs:
                self._no_prefix = kwargs['no_prefix']
        elif 'option_strings' not in kwargs:
            raise ValueError('Expected yes_prefix and/or no_prefix keyword arguments.')
        else:
            self._yes_prefix = kwargs.pop('_yes_prefix') if '_yes_prefix' in kwargs else ''
            self._no_prefix = kwargs.pop('_no_prefix') if '_no_prefix' in kwargs else 'no_'
            if len(kwargs['option_strings']) == 0:
                raise ValueError(type(self).__name__+' not intended for positional arguments  ('+kwargs['dest']+').')
            opt_name = kwargs['option_strings'][0]
            if not opt_name.startswith('--'+self._yes_prefix):
                raise ValueError('Expected option string to start with "--'+self._yes_prefix+'".')
            if 'dest' not in kwargs:
                kwargs['dest'] = re.sub('^--', '', opt_name).replace('-', '_')
            if self._no_prefix is not None:
                kwargs['option_strings'] += [re.sub('^--'+self._yes_prefix, '--'+self._no_prefix, opt_name)]
            if self._no_prefix is None and 'nargs' in kwargs and kwargs['nargs'] != 1:
                raise ValueError('ActionYesNo with no_prefix=None only supports nargs=1.')
            if 'nargs' in kwargs and kwargs['nargs'] in {'?', 1}:
                kwargs['metavar'] = 'true|yes|false|no'
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
        value = args[2][0] if isinstance(args[2], list) and len(args[2]) == 1 else args[2] if isinstance(args[2], bool) else True
        if self._no_prefix is not None and args[3].startswith('--'+self._no_prefix):
            setattr(args[1], self.dest, not value)
        else:
            setattr(args[1], self.dest, value)

    def _add_dest_prefix(self, prefix):
        self.dest = prefix+'.'+self.dest
        self.option_strings[0] = re.sub('^--'+self._yes_prefix, '--'+self._yes_prefix+prefix+'.', self.option_strings[0])
        if self._no_prefix is not None:
            self.option_strings[-1] = re.sub('^--'+self._no_prefix, '--'+self._no_prefix+prefix+'.', self.option_strings[-1])
        for n in range(1, len(self.option_strings)-1):
            self.option_strings[n] = re.sub('^--', '--'+prefix+'.', self.option_strings[n])

    def _check_type(self, value, cfg=None):
        if isinstance(value, list):
            value = [ActionYesNo._boolean_type(val) for val in value]
        else:
            value = ActionYesNo._boolean_type(value)
        if isinstance(value, list) and (self.nargs == 0 or self.nargs):
            return value[0]
        return value

    @staticmethod
    def _boolean_type(x):
        if isinstance(x, str) and x.lower() in {'true', 'yes', 'false', 'no'}:
            x = True if x.lower() in {'true', 'yes'} else False
        elif not isinstance(x, bool):
            raise TypeError('Value not boolean: '+str(x)+'.')
        return x

    def completer(self, **kwargs):
        """Used by argcomplete to support tab completion of arguments."""
        return ['true', 'false']


class ActionEnum(Action):
    """An action based on an Enum that maps to-from strings and enum values."""

    def __init__(self, **kwargs):
        """Initializer for ActionEnum instance.

        Args:
            enum (Enum): Enum instance.

        Raises:
            ValueError: If a parameter is invalid.
        """
        if 'enum' in kwargs:
            _check_unknown_kwargs(kwargs, {'enum'})
            if not _issubclass(kwargs['enum'], Enum):
                raise ValueError('Expected enum to be an instance of Enum.')
            self._enum = kwargs['enum']
        elif '_enum' not in kwargs:
            raise ValueError('Expected enum keyword argument.')
        else:
            self._enum = kwargs.pop('_enum')
            kwargs['type'] = str
            kwargs['metavar'] = '{'+','.join(self._enum.__members__.keys())+'}'
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument mapping a string to its Enum value.

        Raises:
            TypeError: If value not present in the Enum.
        """
        if len(args) == 0:
            kwargs['_enum'] = self._enum
            return ActionEnum(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, cfg=None):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('For ActionEnum with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        for num, val in enumerate(value):
            try:
                if isinstance(val, str):
                    value[num] = self._enum[val]
                else:
                    self._enum(val)
            except KeyError:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('Parser key "'+self.dest+'"'+elem+': value '+str(val)+' not in '+self._enum.__name__+'.')
        return value if islist else value[0]

    def completer(self, **kwargs):
        """Used by argcomplete to support tab completion of arguments."""
        return list(self._enum.__members__.keys())


class ActionOperators:
    """DEPRECATED: Action to restrict a value with comparison operators.

    The new alternative is explained in :ref:`restricted-numbers`.
    """

    def __init__(self, **kwargs):
        if 'expr' in kwargs:
            _check_unknown_kwargs(kwargs, {'expr', 'join', 'type'})
            self._type = restricted_number_type(None, kwargs.get('type', int), kwargs['expr'], kwargs.get('join', 'and'))
        else:
            raise ValueError('Expected expr keyword argument.')

    def __call__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('ActionOperators does not allow type given to add_argument.')
        kwargs['type'] = self._type
        return _StoreAction(**kwargs)


class ActionParser(Action):
    """Action to parse option with a given parser optionally loading from file if string value."""

    def __init__(self, **kwargs):
        """Initializer for ActionParser instance.

        Args:
            parser (ArgumentParser): A parser to parse the option with.

        Raises:
            ValueError: If the parser parameter is invalid.
        """
        if 'parser' in kwargs:
            ## Runs when first initializing class by external user ##
            _check_unknown_kwargs(kwargs, {'parser'})
            self._parser = kwargs['parser']
            if not isinstance(self._parser, ArgumentParser):
                raise ValueError('Expected parser keyword argument to be an ArgumentParser.')
        elif '_parser' not in kwargs:
            raise ValueError('Expected parser keyword argument.')
        else:
            ## Runs when initialied from the __call__ method below ##
            self._parser = kwargs.pop('_parser')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument with the corresponding parser and if valid, sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            ## Runs when within _ActionsContainer super().add_argument call ##
            kwargs['_parser'] = self._parser
            return ActionParser(**kwargs)
        ## Runs when parsing a value ##
        value = _dict_to_flat_namespace(namespace_to_dict(self._check_type(args[2])))
        for key, val in vars(value).items():
            setattr(args[1], key, val)
        if hasattr(value, '__path__'):
            setattr(args[1], self.dest+'.__path__', getattr(value, '__path__'))

    def _check_type(self, value, cfg=None):
        try:
            fpath = None
            if isinstance(value, str):
                value = yaml.safe_load(value)
            if isinstance(value, str):
                fpath = Path(value, mode=get_config_read_mode())
                value = self._parser.parse_path(fpath, _base=self.dest)
            else:
                value = dict_to_namespace(_flat_namespace_to_dict(dict_to_namespace({self.dest: value})))
                self._parser.check_config(value, skip_none=True)
            if fpath is not None:
                value.__path__ = fpath
        except TypeError as ex:
            raise TypeError(re.sub('^Parser key ([^:]+):', 'Parser key '+self.dest+'.\\1: ', str(ex)))
        return value

    @staticmethod
    def _set_inner_parser_prefix(parser, prefix, action):
        """Sets the value of env_prefix to an ActionParser and all sub ActionParsers it contains.

        Args:
            parser (ArgumentParser): The parser to which the action belongs.
            action (ActionParser): The action to set its env_prefix.
        """
        assert isinstance(action, ActionParser)
        action._parser.env_prefix = parser.env_prefix
        action._parser.default_env = parser.default_env
        option_string_actions = {}
        for key, val in action._parser._option_string_actions.items():
            option_string_actions[re.sub('^--', '--'+prefix+'.', key)] = val
        action._parser._option_string_actions = option_string_actions
        for subaction in action._parser._actions:
            if isinstance(subaction, ActionYesNo):
                subaction._add_dest_prefix(prefix)
            else:
                subaction.dest = prefix+'.'+subaction.dest
                for n in range(len(subaction.option_strings)):
                    subaction.option_strings[n] = re.sub('^--', '--'+prefix+'.', subaction.option_strings[n])
            if isinstance(subaction, ActionParser):
                ActionParser._set_inner_parser_prefix(action._parser, prefix, subaction)

    @staticmethod
    def _fix_conflicts(parser, cfg):
        cfg_dict = namespace_to_dict(cfg)
        for action in parser._actions:
            if isinstance(action, ActionParser) and action.dest in cfg_dict and cfg_dict[action.dest] is None:
                children = [x for x in cfg_dict.keys() if x.startswith(action.dest+'.')]
                if len(children) > 0:
                    delattr(cfg, action.dest)


class _ActionSubCommands(argparse._SubParsersAction):
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
        parser.prog = '%s %s' % (self._prog_prefix, name)
        parser.env_prefix = self._env_prefix+'_'+name+'_'

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
    def handle_subcommands(parser, cfg, env, defaults):
        """Adds sub-command dest if missing and parses defaults and environment variables."""
        if parser._subparsers is None:
            return

        cfg_dict = cfg.__dict__ if isinstance(cfg, Namespace) else cfg

        # Get subcommands action
        for action in parser._actions:
            if isinstance(action, _ActionSubCommands):
                break

        # Get sub-command parser
        subcommand = None
        if action.dest in cfg_dict and cfg_dict[action.dest] is not None:
            subcommand = cfg_dict[action.dest]
        else:
            for key in action.choices.keys():
                if any([v.startswith(key+'.') for v in cfg_dict.keys()]):
                    subcommand = key
                    break
            cfg_dict[action.dest] = subcommand

        assert subcommand in action._name_parser_map
        subparser = action._name_parser_map[subcommand]

        # merge environment variable values and default values
        subnamespace = None
        if env:
            subnamespace = subparser.parse_env(defaults=defaults, nested=False, _skip_check=True)
        elif defaults:
            subnamespace = subparser.get_defaults(nested=False)

        if subnamespace is not None:
            for key, value in vars(subnamespace).items():
                key = subcommand+'.'+key
                if key not in cfg_dict:
                    cfg_dict[key] = value


class ActionPath(Action):
    """Action to check and store a path."""

    def __init__(self, **kwargs):
        """Initializer for ActionPath instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument, e.g. ActionPath(mode='drw').
            skip_check (bool): Whether to skip path checks (def.=False).

        Raises:
            ValueError: If the mode parameter is invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode', 'skip_check'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
            self._skip_check = kwargs.get('skip_check', False)
        elif '_mode' not in kwargs:
            raise ValueError('ActionPath expects mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a Path and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid Path.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise ValueError('Invalid nargs='+str(kwargs['nargs'])+' for ActionPath.')
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
        elif not isinstance(value, list):
            raise TypeError('For ActionPath with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        try:
            for num, val in enumerate(value):
                if isinstance(val, str):
                    val = Path(val, mode=self._mode, skip_check=self._skip_check)
                elif isinstance(val, Path):
                    val = Path(val(absolute=False), mode=self._mode, skip_check=self._skip_check, cwd=val.cwd)
                else:
                    raise TypeError('expected either a string or a Path object, received: value='+str(val)+' type='+str(type(val))+'.')
                value[num] = val
        except TypeError as ex:
            raise TypeError('Parser key "'+self.dest+'": '+str(ex))
        return value if islist else value[0]


class ActionPathList(Action):
    """Action to check and store a list of file paths read from a plain text file or stream."""

    def __init__(self, **kwargs):
        """Initializer for ActionPathList instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxcuFDRWX] as a keyword argument (uppercase means not), e.g. ActionPathList(mode='fr').
            skip_check (bool): Whether to skip path checks (def.=False).
            rel (str): Whether relative paths are with respect to current working directory 'cwd' or the list's parent directory 'list' (default='cwd').

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode', 'skip_check', 'rel'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
            self._skip_check = kwargs.get('skip_check', False)
            self._rel = kwargs.get('rel', 'cwd')
            if self._rel not in {'cwd', 'list'}:
                raise ValueError('rel must be either "cwd" or "list", got '+str(self._rel)+'.')
        elif '_mode' not in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._skip_check = kwargs.pop('_skip_check')
            self._rel = kwargs.pop('_rel')
            kwargs['type'] = str
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
                except Exception as ex:
                    raise TypeError('Problems reading path list: '+path_list_file+' :: '+str(ex))
                cwd = os.getcwd()
                if self._rel == 'list' and path_list_file != '-':
                    os.chdir(os.path.abspath(os.path.join(path_list_file, os.pardir)))
                try:
                    for num, val in enumerate(path_list):
                        try:
                            path_list[num] = Path(val, mode=self._mode)
                        except TypeError as ex:
                            raise TypeError('Path number '+str(num+1)+' in list '+path_list_file+', '+str(ex))
                finally:
                    os.chdir(cwd)
                value += path_list
            return value
        else:
            return ActionPath._check_type(self, value, islist=True)
