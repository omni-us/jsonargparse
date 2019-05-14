
import os
import re
import yaml
import operator
import argparse
from argparse import Action, ArgumentError, ArgumentTypeError, OPTIONAL, REMAINDER, SUPPRESS, PARSER, ONE_OR_MORE, ZERO_OR_MORE
from types import SimpleNamespace
from typing import Any, Dict, Set, Union


__version__ = '1.11.0'


class ArgumentParser(argparse.ArgumentParser):
    """Extension to python's argparse which simplifies parsing of configuration
    options from command line arguments, yaml configuration files, environment
    variables and hard-coded defaults.
    """

    groups = {} # type: Dict[str, argparse._ArgumentGroup]

    def parse_args(self, args=None, namespace=None, env:bool=True, nested:bool=True):
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            env (bool): Whether to merge with the parsed environment.
            nested (bool): Whether the namespace should be nested.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        if namespace is not None:
            namespace = self.parse_env(nested=False) if env else None

        cfg = super().parse_args(args=args, namespace=namespace)

        ActionParser._fix_conflicts(self, cfg)

        if not nested:
            return cfg

        return self._dict_to_namespace(self._flat_namespace_to_dict(cfg))


    def parse_yaml_path(self, yaml_path:str, env:bool=True, defaults:bool=True, nested:bool=True) -> SimpleNamespace:
        """Parses a yaml file given its path.

        Args:
            yaml_path (str): Path to the yaml file to parse.
            env (bool): Whether to merge with the parsed environment.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.join(yaml_path, os.pardir)))
        try:
            with open(os.path.basename(yaml_path), 'r') as f:
                parsed_yaml = self.parse_yaml_string(f.read(), env, defaults, nested)
        finally:
            os.chdir(cwd)
        return parsed_yaml


    def parse_yaml_string(self, yaml_str:str, env:bool=True, defaults:bool=True, nested:bool=True) -> SimpleNamespace:
        """Parses yaml given as a string.

        Args:
            yaml_str (str): The yaml content.
            env (bool): Whether to merge with the parsed environment.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            SimpleNamespace: An object with all parsed values as attributes.
        """
        cfg = yaml.safe_load(yaml_str)

        cfg = self._namespace_to_dict(self._dict_to_flat_namespace(cfg))
        for action in self._actions:
            if action.dest in cfg:
                cfg[action.dest] = self._check_value_key(action, cfg[action.dest], action.dest)

        if nested:
            cfg = self._flat_namespace_to_dict(self._dict_to_namespace(cfg))

        if env:
            cfg = self.merge_config(cfg, self.parse_env(defaults=defaults, nested=nested))

        if defaults:
            cfg = self.merge_config(cfg, self.get_defaults(nested=nested))

        return self._dict_to_namespace(cfg)


    def dump_yaml(self, cfg:Union[SimpleNamespace, dict]) -> str:
        """Generates a yaml string for a configuration object.

        Args:
            cfg (SimpleNamespace | dict): The configuration object to dump.

        Returns:
            str: The configuration in yaml format.
        """
        if not isinstance(cfg, dict):
            cfg = self._namespace_to_dict(cfg)

        self.check_config(cfg, skip_none=True)

        cfg = self._namespace_to_dict(self._dict_to_flat_namespace(cfg))
        for action in self._actions:
            if isinstance(action, ActionPath):
                if cfg[action.dest] is not None:
                    cfg[action.dest] = cfg[action.dest](absolute=False)
            elif isinstance(action, ActionConfigFile):
                del cfg[action.dest]
        cfg = self._flat_namespace_to_dict(self._dict_to_namespace(cfg))

        return yaml.dump(cfg, default_flow_style=False, allow_unicode=True)


    def parse_env(self, env:Dict[str, str]=None, defaults:bool=True, nested:bool=True) -> SimpleNamespace:
        """Parses environment variables.

        Args:
            env (Dict[str, str]): The environment object to use, if None `os.environ` is used.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            SimpleNamespace: An object with all parsed values as attributes.
        """        
        if env is None:
            env = dict(os.environ)
        cfg = {} # type: Dict[str, Any]
        for action in self._actions:
            if action.default == '==SUPPRESS==':
                continue
            env_var = (self.prog+'_' if self.prog else '') + action.dest
            env_var = env_var.replace('.', '__').upper()
            if env_var in env:
                cfg[action.dest] = self._check_value_key(action, env[env_var], env_var)

        if nested:
            cfg = self._flat_namespace_to_dict(SimpleNamespace(**cfg))

        if defaults:
            cfg = self.merge_config(cfg, self.get_defaults(nested=nested)) # type: ignore

        return self._dict_to_namespace(cfg)


    def get_defaults(self, nested:bool=True) -> SimpleNamespace:
        """Returns a namespace with all default values.

        Args:
            nested (bool): Whether the namespace should be nested.

        Returns:
            SimpleNamespace: An object with all default values as attributes.
        """
        cfg = {}
        for action in self._actions:
            if len(action.option_strings) > 0 and action.default != '==SUPPRESS==':
                cfg[action.dest] = action.default

        if nested:
            cfg = self._flat_namespace_to_dict(SimpleNamespace(**cfg))

        return self._dict_to_namespace(cfg)


    def add_argument_group(self, *args, name:str=None, **kwargs):
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported. Additionally it accepts:

        Args:
            name (str): Name of the group. If set the group object will be included in the parser.groups dict.

        Returns:
            The group object.
        """
        group = argparse._ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        if name is not None:
            self.groups[name] = group
        return group


    def check_config(self, cfg:Union[SimpleNamespace, dict], skip_none:bool=False):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg (SimpleNamespace | dict): The configuration object to check.
            skip_none (bool): Whether to skip checking of values that are None.
        """
        if not isinstance(cfg, dict):
            cfg = self._namespace_to_dict(cfg)

        def find_action(dest):
            for action in self._actions:
                if action.dest == dest:
                    return action
            return None

        def check_values(cfg, base=None):
            for key, val in cfg.items():
                if skip_none and val is None:
                    continue
                kbase = key if base is None else base+'.'+key
                if isinstance(val, dict):
                    check_values(val, kbase)
                else:
                    action = find_action(kbase)
                    if action is not None:
                        self._check_value_key(action, val, kbase)
                    elif not skip_none:
                        raise Exception('no action for key '+key+' to check its value')

        check_values(cfg)


    @staticmethod
    def merge_config(cfg_from:Union[SimpleNamespace, Dict[str, Any]], cfg_to:Union[SimpleNamespace, Dict[str, Any]]) -> Union[SimpleNamespace, Dict[str, Any]]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (SimpleNamespace | dict): The configuration from which to merge.
            cfg_to (SimpleNamespace | dict): The configuration into which to merge.

        Returns:
            SimpleNamespace | dict: The merged configuration with same type as cfg_from.
        """
        def merge_values(cfg_from, cfg_to):
            for k, v in cfg_from.items():
                if v is None:
                    continue
                if k not in cfg_to or not isinstance(v, dict):
                    cfg_to[k] = v
                elif k in cfg_to and cfg_to[k] is None:
                    cfg_to[k] = cfg_from[k]
                else:
                    cfg_to[k] = merge_values(cfg_from[k], cfg_to[k])
            return cfg_to

        out_dict = isinstance(cfg_from, dict)
        cfg_from = cfg_from if isinstance(cfg_from, dict) else ArgumentParser._namespace_to_dict(cfg_from)
        cfg_to = cfg_to if isinstance(cfg_to, dict) else ArgumentParser._namespace_to_dict(cfg_to)
        cfg = merge_values(cfg_from, cfg_to.copy())
        return cfg if out_dict else ArgumentParser._dict_to_namespace(cfg)


    @staticmethod
    def _check_value_key(action:Action, value:Any, key:str) -> Any:
        """Checks the value for a given action.

        Args:
            action (Action): The action used for parsing.
            value (Any): The value to parse.
            key (str): The configuration key.
        """
        if action is None:
            raise Exception('parser key "'+str(key)+'": received action==None')
        if action.choices is not None:
            if value not in action.choices:
                args = {'value': value,
                        'choices': ', '.join(map(repr, action.choices))}
                msg = 'invalid choice: %(value)r (choose from %(choices)s)'
                raise ArgumentTypeError('parser key "'+str(key)+'": '+(msg % args))
        elif hasattr(action, '_check_type'):
            value = action._check_type(value) # type: ignore
        elif action.type is not None:
            try:
                value = action.type(value)
            except TypeError as ex:
                raise ArgumentTypeError('parser key "'+str(key)+'": '+str(ex))
        return value


    @staticmethod
    def _flat_namespace_to_dict(cfg_ns:SimpleNamespace) -> Dict[str, Any]:
        """Converts a flat namespace into a nested dictionary.

        Args:
            cfg_ns (SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        cfg_dict = {}
        for k, v in vars(cfg_ns).items():
            ksplit = k.split('.')
            if len(ksplit) == 1:
                cfg_dict[k] = v
            else:
                kdict = cfg_dict
                for num, kk in enumerate(ksplit[:len(ksplit)-1]):
                    if kk not in kdict:
                        kdict[kk] = {}
                    elif not isinstance(kdict[kk], dict):
                        raise Exception('Conflicting namespace base: '+'.'.join(ksplit[:num+1]))
                    kdict = kdict[kk]
                if ksplit[-1] in kdict:
                    raise Exception('Conflicting namespace base: '+k)
                kdict[ksplit[-1]] = v
        return cfg_dict


    @staticmethod
    def _dict_to_flat_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
        """Converts a nested dictionary into a flat namespace.

        Args:
            cfg_dict (dict): The configuration to process.

        Returns:
            SimpleNamespace: The configuration namespace.
        """
        cfg_ns = {}

        def flatten_dict(cfg, base=None):
            for key, val in cfg.items():
                kbase = key if base is None else base+'.'+key
                if isinstance(val, dict):
                    flatten_dict(val, kbase)
                else:
                    cfg_ns[kbase] = val

        flatten_dict(cfg_dict)

        return SimpleNamespace(**cfg_ns)


    @staticmethod
    def _dict_to_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
        """Converts a nested dictionary into a nested namespace.

        Args:
            cfg_args (dict): The configuration to process.

        Returns:
            SimpleNamespace: The nested configuration namespace.
        """
        def expand_dict(cfg):
            for k, v in cfg.items():
                if isinstance(v, dict):
                    cfg[k] = expand_dict(v)
            return SimpleNamespace(**cfg)
        return expand_dict(cfg_dict)


    @staticmethod
    def _namespace_to_dict(cfg_ns:SimpleNamespace) -> Dict[str, Any]:
        """Converts a nested namespace into a nested dictionary.

        Args:
            cfg_args (SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        def expand_namespace(cfg):
            cfg = dict(vars(cfg))
            for k, v in cfg.items():
                if isinstance(v, SimpleNamespace):
                    cfg[k] = expand_namespace(v)
            return cfg
        return expand_namespace(cfg_ns)


class ActionConfigFile(Action):
    """Action to indicate that an argument is a configuration file."""
    def __init__(self, **kwargs):
        opt_name = kwargs['option_strings']
        opt_name = opt_name[0] if len(opt_name) == 1 else [x for x in opt_name if x[0:2] == '--'][0]
        if '.' in opt_name:
            raise Exception('Config file must be a top level option.')
        kwargs['type'] = str
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not isinstance(getattr(namespace, self.dest), list):
            setattr(namespace, self.dest, [])
        try:
            getattr(namespace, self.dest).append(Path(values, mode='r'))
        except ArgumentTypeError as ex:
            raise ArgumentTypeError('parser key "'+self.dest+'": '+str(ex))
        cfg_file = parser.parse_yaml_path(values, env=False, defaults=False, nested=False)
        for key, val in vars(cfg_file).items():
            setattr(namespace, key, val)


class ActionYesNo(Action):
    """Paired action --opt, --no_opt to set True or False respectively."""
    def __init__(self, **kwargs):
        opt_name = kwargs['option_strings'][0]
        if 'dest' not in kwargs:
            kwargs['dest'] = re.sub('^--', '', opt_name).replace('-', '_')
        kwargs['option_strings'] += [re.sub('^--', '--no_', opt_name)]
        kwargs['nargs'] = 0
        def boolean(x):
            if not isinstance(x, bool):
                raise ValueError('value not boolean: '+str(x))
            return x
        kwargs['type'] = boolean
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no_'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


class ActionParser(Action):
    """Action to parse option with a given yamlargparse parser optionally loading from yaml file if string value.

    Args:
        parser (ArgumentParser): A yamlargparse parser to parse the option with.
    """
    def __init__(self, **kwargs):
        if 'parser' in kwargs:
            _check_unknown_kwargs(kwargs, {'parser'})
            self._parser = kwargs['parser']
            if not isinstance(self._parser, ArgumentParser):
                raise Exception('Expected parser keyword argument to be a yamlargparse parser.')
        elif not '_parser' in kwargs:
            raise Exception('Expected parser keyword argument.')
        else:
            self._parser = kwargs.pop('_parser')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['_parser'] = self._parser
            return ActionParser(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        try:
            if isinstance(value, str):
                yaml_path = Path(value, mode='r')
                value = self._parser.parse_yaml_path(yaml_path())
            else:
                self._parser.check_config(value, skip_none=True)
        except ArgumentTypeError as ex:
            raise ArgumentTypeError(re.sub('^parser key ([^:]+):', 'parser key '+self.dest+'.\\1: ', str(ex)))
        return value

    @staticmethod
    def _fix_conflicts(parser, cfg):
        cfg_dict = parser._namespace_to_dict(cfg)
        for action in parser._actions:
            if isinstance(action, ActionParser) and action.dest in cfg_dict and cfg_dict[action.dest] is None:
                children = [x for x in cfg_dict.keys() if x.startswith(action.dest+'.')]
                if len(children) > 0:
                    delattr(cfg, action.dest)


class ActionOperators(Action):
    """Action to restrict a number range with comparison operators.

    Args:
        expr (tuple or list of tuples): Pairs of operators (> >= < <= == !=) and reference values, e.g. [('>=', 1),...].
        join (str): How to combine multiple comparisons, must be 'or' or 'and' (default='and').
        type (type): The value type (default=int).
    """
    _operators = {operator.gt: '>', operator.ge: '>=', operator.lt: '<', operator.le: '<=', operator.eq: '==', operator.ne: '!='}

    def __init__(self, **kwargs):
        if 'expr' in kwargs:
            _check_unknown_kwargs(kwargs, {'expr', 'join', 'type'})
            self._type = kwargs['type'] if 'type' in kwargs else int
            self._join = kwargs['join'] if 'join' in kwargs else 'and'
            if self._join not in {'or', 'and'}:
                raise Exception("Expected join to be either 'or' or 'and'.")
            _operators = {v: k for k, v in self._operators.items()}
            expr = [kwargs['expr']] if isinstance(kwargs['expr'], tuple) else kwargs['expr']
            if not isinstance(expr, list) or not all([all([len(x)==2, x[0] in _operators, x[1] == self._type(x[1])]) for x in expr]):
                raise Exception('Expected expr to be a list of tuples each with a comparison operator (> >= < <= == !=) and a reference value of type '+self._type.__name__+'.')
            self._expr = [(_operators[x[0]], x[1]) for x in expr]
        elif not '_expr' in kwargs:
            raise Exception('Expected expr keyword argument.')
        else:
            self._expr = kwargs.pop('_expr')
            self._join = kwargs.pop('_join')
            self._type = kwargs.pop('_type')
            if 'type' in kwargs:
                del kwargs['type']
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise Exception('invalid nargs='+str(kwargs['nargs'])+' for ActionOperators.')
            kwargs['_expr'] = self._expr
            kwargs['_join'] = self._join
            kwargs['_type'] = self._type
            return ActionOperators(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise Exception('for ActionOperators with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        def test_op(op, val, ref):
            try:
                return op(val, ref)
            except TypeError:
                return False
        for num, val in enumerate(value):
            try:
                val = self._type(val)
            except:
                raise ArgumentTypeError('parser key "'+self.dest+'": invalid value, expected type to be '+self._type.__name__+' but got as value '+str(val)+'.')
            check = [test_op(op, val, ref) for op, ref in self._expr]
            if (self._join == 'and' and not all(check)) or (self._join == 'or' and not any(check)):
                expr = (' '+self._join+' ').join(['v'+self._operators[op]+str(ref) for op, ref in self._expr])
                raise ArgumentTypeError('parser key "'+self.dest+'": invalid value, for v='+str(val)+' it is false that '+expr+'.')
            value[num] = val
        return value if islist else value[0]


class ActionPath(Action):
    """Action to check and store a file path.

    Args:
        mode (str): The required type and access permissions among [drwx] as a keyword argument, e.g. ActionPath(mode='drw').
    """
    def __init__(self, **kwargs):
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
        elif not '_mode' in kwargs:
            raise Exception('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise Exception('invalid nargs='+str(kwargs['nargs'])+' for ActionPath.')
            kwargs['_mode'] = self._mode
            return ActionPath(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise Exception('for ActionPath with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        try:
            for num, val in enumerate(value):
                if isinstance(val, str):
                    val = Path(val, mode=self._mode)
                elif isinstance(val, Path):
                    val = Path(val(absolute=False), mode=self._mode, cwd=val.cwd)
                else:
                    raise ArgumentTypeError('expected either a string or a Path object, received: value='+str(val)+' type='+str(type(val))+'.')
                value[num] = val
        except ArgumentTypeError as ex:
            raise ArgumentTypeError('parser key "'+self.dest+'": '+str(ex))
        return value if islist else value[0]


class Path(object):
    """Stores a (possibly relative) path and the corresponding absolute path.

    When a Path instance is created it is checked that: the path exists, whether
    it is a file or directory and whether has the required access permissions
    (d=directory, r=readable, w=writeable, x=executable). The absolute path can
    be obtained without having to remember the working directory from when the
    object was created.

    Args:
        path (str): The path to check and store.
        mode (str): The required type and access permissions among [drwx].
        cwd (str): Working directory for relative paths. If None, then os.getcwd() is used.

    Args called:
        absolute (bool): If false returns the original path given, otherwise the corresponding absolute path.
    """
    def __init__(self, path, mode:str='r', cwd:str=None):
        self._check_mode(mode)
        if cwd is None:
            cwd = os.getcwd()

        if isinstance(path, Path):
            abs_path = path(absolute=True)
            path = path()
        elif not isinstance(path, str):
            raise Exception('Expected path to be a string or a Path.')
        else:
            abs_path = path if os.path.isabs(path) else os.path.join(cwd, path)

        ptype = 'directory' if 'd' in mode else 'file'
        if not os.access(abs_path, os.F_OK):
            raise ArgumentTypeError(ptype+' does not exist: '+abs_path)
        if 'd' in mode and not os.path.isdir(abs_path):
            raise ArgumentTypeError('path is not a directory: '+abs_path)
        if 'd' not in mode and not os.path.isfile(abs_path):
            raise ArgumentTypeError('path is not a file: '+abs_path)
        if 'r' in mode and not os.access(abs_path, os.R_OK):
            raise ArgumentTypeError(ptype+' is not readable: '+abs_path)
        if 'w' in mode and not os.access(abs_path, os.W_OK):
            raise ArgumentTypeError(ptype+' is not writeable: '+abs_path)
        if 'x' in mode and not os.access(abs_path, os.X_OK):
            raise ArgumentTypeError(ptype+' is not executable: '+abs_path)

        self.path = path
        self.abs_path = abs_path
        self.cwd = cwd

    def __str__(self):
        return self.abs_path

    def __call__(self, absolute=True):
        return self.abs_path if absolute else self.path

    @staticmethod
    def _check_mode(mode:str):
        if not isinstance(mode, str):
            raise Exception('Expected mode to be a string.')
        if len(set(mode)-set('drwx')) > 0:
            raise Exception('Expected mode to only include [drwx] flags.')


def _is_action_value_list(action:Action):
    if action.nargs in {'*', '+'} or isinstance(action.nargs, int):
      return True
    return False


def _check_unknown_kwargs(kwargs:Dict[str, Any], keys:Set[str]):
    """Raises exception if a kwargs has unexpected keys.

    Args:
        kwargs (dict): The keyword arguments to check.
        keys (set): The expected keys.
    """
    if len(set(kwargs.keys())-keys) > 0:
        raise Exception('Unknown keyword arguments: '+', '.join(set(kwargs.keys())-keys))
