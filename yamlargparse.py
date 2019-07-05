
import os
import re
import sys
import glob
import yaml
import logging
import operator
import argparse
from argparse import Action, OPTIONAL, REMAINDER, SUPPRESS, PARSER, ONE_OR_MORE, ZERO_OR_MORE
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, List, Dict, Set, Union
from contextlib import contextmanager, redirect_stderr

try:
    import jsonschema
    from jsonschema import Draft4Validator as jsonvalidator
except Exception as ex:
    jsonschema = jsonvalidator = ex


__version__ = '1.22.0'


class ArgumentParser(argparse.ArgumentParser):
    """Parser for command line, yaml files and environment variables."""

    groups = {} # type: Dict[str, argparse._ArgumentGroup]


    def __init__(self, *args, default_config_files:List[str]=[], logger=None, error_handler=None, formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs):
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            default_config_files (list[str]): List of strings defining default config file locations. For example: :code:`['~/.config/myapp/*.yaml']`.
            logger (logging.Logger): Object for logging events.
            error_handler (Callable): Handler for parsing errors (default=None). For same behavior as argparse use :func:`usage_and_exit_error_handler`.
        """
        self._default_config_files = default_config_files
        if logger is None:
            self._logger = logging.getLogger('null')
            self._logger.addHandler(logging.NullHandler())
        else:
            self._logger = logger
        self.error_handler = error_handler
        self._stderr = sys.stderr
        kwargs['formatter_class'] = formatter_class
        super().__init__(*args, **kwargs)


    def parse_args(self, args=None, namespace=None, env:bool=True, nested:bool=True):
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            env (bool): Whether to merge with the parsed environment.
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            if namespace is not None:
                namespace = self.parse_env(nested=False) if env else None

            with _suppress_stderr():
                cfg = super().parse_args(args=args, namespace=namespace)

            ActionParser._fix_conflicts(self, cfg)

            if not nested:
                return cfg

            self._logger.info('parsed arguments')

        except TypeError as ex:
            self.error(str(ex))

        return self.dict_to_namespace(self._flatnamespace_to_dict(cfg))


    def parse_yaml_path(self, yaml_path:str, env:bool=True, defaults:bool=True, nested:bool=True) -> SimpleNamespace:
        """Parses a yaml file given its path.

        Args:
            yaml_path (str): Path to the yaml file to parse.
            env (bool): Whether to merge with the parsed environment.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all parsed values as nested attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.join(yaml_path, os.pardir)))
        try:
            with open(os.path.basename(yaml_path), 'r') as f:
                parsed_yaml = self.parse_yaml_string(f.read(), env, defaults, nested, log=False)
        finally:
            os.chdir(cwd)

        self._logger.info('parsed yaml from path: '+yaml_path)

        return parsed_yaml


    def parse_yaml_string(self, yaml_str:str, env:bool=True, defaults:bool=True, nested:bool=True, log=True) -> SimpleNamespace:
        """Parses yaml given as a string.

        Args:
            yaml_str (str): The yaml content.
            env (bool): Whether to merge with the parsed environment.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            cfg = self._load_yaml(yaml_str)

            if nested:
                cfg = self._flatnamespace_to_dict(self.dict_to_namespace(cfg))

            if env:
                cfg = self._merge_config(cfg, self.parse_env(defaults=defaults, nested=nested))

            if defaults:
                cfg = self._merge_config(cfg, self.get_defaults(nested=nested))

            if log:
                self._logger.info('parsed yaml string')

        except TypeError as ex:
            self.error(str(ex))

        return self.dict_to_namespace(cfg)


    def _load_yaml(self, yaml_str:str) -> Dict[str, Any]:
        """Loads a yaml string into a namespace checking all values against the parser.

        Args:
            yaml_str (str): The yaml content.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        cfg = yaml.safe_load(yaml_str)
        cfg = self.namespace_to_dict(self._dict_to_flat_namespace(cfg))
        for action in self._actions:
            if action.dest in cfg:
                cfg[action.dest] = self._check_value_key(action, cfg[action.dest], action.dest)
        return cfg


    def dump_yaml(self, cfg:Union[SimpleNamespace, dict], skip_none:bool=True, skip_check:bool=False) -> str:
        """Generates a yaml string for a configuration object.

        Args:
            cfg (types.SimpleNamespace or dict): The configuration object to dump.
            skip_none (bool): Whether to exclude settings whose value is None.
            skip_check (bool): Whether to skip parser checking.

        Returns:
            str: The configuration in yaml format.

        Raises:
            TypeError: If any of the values of cfg is invalid according to the parser.
        """
        cfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = self.namespace_to_dict(cfg)

        if not skip_check:
            self.check_config(cfg, skip_none=True)

        cfg = self.namespace_to_dict(self._dict_to_flat_namespace(cfg))
        for action in self._actions:
            if skip_none and action.dest in cfg and cfg[action.dest] is None:
                del cfg[action.dest]
            elif isinstance(action, ActionPath):
                if cfg[action.dest] is not None:
                    if isinstance(cfg[action.dest], list):
                        cfg[action.dest] = [p(absolute=False) for p in cfg[action.dest]]
                    else:
                        cfg[action.dest] = cfg[action.dest](absolute=False)
            elif isinstance(action, ActionConfigFile):
                del cfg[action.dest]
        cfg = self._flatnamespace_to_dict(self.dict_to_namespace(cfg))

        return yaml.dump(cfg, default_flow_style=False, allow_unicode=True)


    def parse_env(self, env:Dict[str, str]=None, defaults:bool=True, nested:bool=True) -> SimpleNamespace:
        """Parses environment variables.

        Args:
            env (dict[str, str]): The environment object to use, if None `os.environ` is used.
            defaults (bool): Whether to merge with the parser's defaults.
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all parsed values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            if env is None:
                env = dict(os.environ)
            cfg = {}
            for action in self._actions:
                if action.default == '==SUPPRESS==':
                    continue
                env_var = (self.prog+'_' if self.prog else '') + action.dest
                env_var = env_var.replace('.', '__').upper()
                if env_var in env:
                    cfg[action.dest] = self._check_value_key(action, env[env_var], env_var)

            if nested:
                cfg = self._flatnamespace_to_dict(SimpleNamespace(**cfg))

            if defaults:
                cfg = self._merge_config(cfg, self.get_defaults(nested=nested))

            self._logger.info('parsed environment variables')

        except TypeError as ex:
            self.error(str(ex))

        return self.dict_to_namespace(cfg)


    def get_defaults(self, nested:bool=True) -> SimpleNamespace:
        """Returns a namespace with all default values.

        Args:
            nested (bool): Whether the namespace should be nested.

        Returns:
            types.SimpleNamespace: An object with all default values as attributes.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        try:
            cfg = {}
            for action in self._actions:
                if len(action.option_strings) > 0 and action.default != '==SUPPRESS==':
                    cfg[action.dest] = action.default

            self._logger.info('loaded default values from parser')

            default_config_files = [] # type: List[str]
            for pattern in self._default_config_files:
                default_config_files += glob.glob(os.path.expanduser(pattern))
            if len(default_config_files) > 0:
                with open(default_config_files[0], 'r') as f:
                    cfg_file = self._load_yaml(f.read())
                cfg = self._merge_config(cfg_file, cfg)
                self._logger.info('parsed configuration from default path: '+default_config_files[0])

            if nested:
                cfg = self._flatnamespace_to_dict(SimpleNamespace(**cfg))

        except TypeError as ex:
            self.error(str(ex))

        return self.dict_to_namespace(cfg)


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
            cfg (types.SimpleNamespace or dict): The configuration object to check.
            skip_none (bool): Whether to skip checking of values that are None.

        Raises:
            TypeError: If any of the values are not valid.
            KeyError: If a key in cfg is not defined in the parser.
        """
        cfg = deepcopy(cfg)
        if not isinstance(cfg, dict):
            cfg = self.namespace_to_dict(cfg)

        def find_action(parser, dest):
            for action in parser._actions:
                if action.dest == dest:
                    return action, dest
                elif isinstance(action, ActionParser) and dest.startswith(action.dest+'.'):
                    _action, _dest = find_action(action._parser, dest[len(action.dest)+1:])
                    if _action is not None:
                        return _action, _dest
            return None, dest

        def check_values(cfg, base=None):
            for key, val in cfg.items():
                if skip_none and val is None:
                    continue
                kbase = key if base is None else base+'.'+key
                if isinstance(val, dict):
                    check_values(val, kbase)
                else:
                    action, kbase = find_action(self, kbase)
                    if action is not None:
                        self._check_value_key(action, val, kbase)
                    elif not skip_none:
                        raise KeyError('no action for key '+key+' to check its value')

        check_values(cfg)


    @staticmethod
    def merge_config(cfg_from:SimpleNamespace, cfg_to:SimpleNamespace) -> SimpleNamespace:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (types.SimpleNamespace): The configuration from which to merge.
            cfg_to (types.SimpleNamespace): The configuration into which to merge.

        Returns:
            types.SimpleNamespace: The merged configuration.
        """
        return ArgumentParser.dict_to_namespace(ArgumentParser._merge_config(cfg_from, cfg_to))


    @staticmethod
    def _merge_config(cfg_from:Union[SimpleNamespace, Dict[str, Any]], cfg_to:Union[SimpleNamespace, Dict[str, Any]]) -> Dict[str, Any]:
        """Merges the first configuration into the second configuration.

        Args:
            cfg_from (types.SimpleNamespace or dict): The configuration from which to merge.
            cfg_to (types.SimpleNamespace or dict): The configuration into which to merge.

        Returns:
            dict: The merged configuration.
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

        cfg_from = cfg_from if isinstance(cfg_from, dict) else ArgumentParser.namespace_to_dict(cfg_from)
        cfg_to = cfg_to if isinstance(cfg_to, dict) else ArgumentParser.namespace_to_dict(cfg_to)
        return merge_values(cfg_from, cfg_to.copy())


    @staticmethod
    def _check_value_key(action:Action, value:Any, key:str) -> Any:
        """Checks the value for a given action.

        Args:
            action (Action): The action used for parsing.
            value (Any): The value to parse.
            key (str): The configuration key.

        Raises:
            TypeError: If the value is not valid.
        """
        if action is None:
            raise ValueError('parser key "'+str(key)+'": received action==None')
        if action.choices is not None:
            if value not in action.choices:
                args = {'value': value,
                        'choices': ', '.join(map(repr, action.choices))}
                msg = 'invalid choice: %(value)r (choose from %(choices)s)'
                raise TypeError('parser key "'+str(key)+'": '+(msg % args))
        elif hasattr(action, '_check_type'):
            value = action._check_type(value)  # type: ignore
        elif action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as ex:
                raise TypeError('parser key "'+str(key)+'": '+str(ex))
        return value


    @staticmethod
    def _flatnamespace_to_dict(cfg_ns:SimpleNamespace) -> Dict[str, Any]:
        """Converts a flat namespace into a nested dictionary.

        Args:
            cfg_ns (types.SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        cfg_dict = {}
        for k, v in vars(cfg_ns).items():
            ksplit = k.split('.')
            if len(ksplit) == 1:
                if isinstance(v, list) and any([isinstance(x, SimpleNamespace) for x in v]):
                    cfg_dict[k] = [ArgumentParser.namespace_to_dict(x) for x in v]
                else:
                    cfg_dict[k] = v
            else:
                kdict = cfg_dict
                for num, kk in enumerate(ksplit[:len(ksplit)-1]):
                    if kk not in kdict:
                        kdict[kk] = {}  # type: ignore
                    elif not isinstance(kdict[kk], dict):
                        raise ParserError('Conflicting namespace base: '+'.'.join(ksplit[:num+1]))
                    kdict = kdict[kk]  # type: ignore
                if ksplit[-1] in kdict:
                    raise ParserError('Conflicting namespace base: '+k)
                if isinstance(v, list) and any([isinstance(x, SimpleNamespace) for x in v]):
                    cfg_dict[k] = [ArgumentParser.namespace_to_dict(x) for x in v]
                else:
                    kdict[ksplit[-1]] = v
        return cfg_dict


    @staticmethod
    def _dict_to_flat_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
        """Converts a nested dictionary into a flat namespace.

        Args:
            cfg_dict (dict): The configuration to process.

        Returns:
            types.SimpleNamespace: The configuration namespace.
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
    def dict_to_namespace(cfg_dict:Dict[str, Any]) -> SimpleNamespace:
        """Converts a nested dictionary into a nested namespace.

        Args:
            cfg_args (dict): The configuration to process.

        Returns:
            types.SimpleNamespace: The nested configuration namespace.
        """
        def expand_dict(cfg):
            for k, v in cfg.items():
                if isinstance(v, dict):
                    cfg[k] = expand_dict(v)
                elif isinstance(v, list):
                    for nn, vv in enumerate(v):
                        if isinstance(vv, dict):
                            cfg[k][nn] = expand_dict(vv)
            return SimpleNamespace(**cfg)
        return expand_dict(cfg_dict)


    @staticmethod
    def namespace_to_dict(cfg_ns:SimpleNamespace) -> Dict[str, Any]:
        """Converts a nested namespace into a nested dictionary.

        Args:
            cfg_args (types.SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        def expand_namespace(cfg):
            cfg = dict(vars(cfg))
            for k, v in cfg.items():
                if isinstance(v, SimpleNamespace):
                    cfg[k] = expand_namespace(v)
                elif isinstance(v, list):
                    for nn, vv in enumerate(v):
                        if isinstance(vv, SimpleNamespace):
                            cfg[k][nn] = expand_namespace(vv)
            return cfg
        return expand_namespace(cfg_ns)


    def error(self, message):
        """Logs error message if a logger is set, calls the error handler and raises a ParserError."""
        self._logger.error(message)
        if self.error_handler is not None:
            with redirect_stderr(self._stderr):
                self.error_handler(self, message)
        raise ParserError(message)


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
        if not isinstance(getattr(namespace, self.dest), list):
            setattr(namespace, self.dest, [])
        try:
            cfg_path = Path(values, mode='fr')
        except TypeError as ex:
            try:
                cfg_path = None
                cfg_file = parser.parse_yaml_string(values, env=False, defaults=False, nested=False)
            except:
                raise TypeError('parser key "'+self.dest+'": '+str(ex))
        else:
            cfg_file = parser.parse_yaml_path(values, env=False, defaults=False, nested=False)
        parser.check_config(cfg_file)
        getattr(namespace, self.dest).append(cfg_path)
        for key, val in vars(cfg_file).items():
            setattr(namespace, key, val)


class ActionYesNo(Action):
    """Paired options --{yes_prefix}opt, --{no_prefix}opt to set True or False respectively."""
    def __init__(self, **kwargs):
        """Initializer for ActionYesNo instance.

        Args:
            yes_prefix (str): Prefix for yes option (default='').
            no_prefix (str): Prefix for no option (default='no_').

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
        elif not 'option_strings' in kwargs:
            raise ValueError('Expected yes_prefix and/or no_prefix keyword arguments.')
        else:
            self._yes_prefix = kwargs.pop('_yes_prefix') if '_yes_prefix' in kwargs else ''
            self._no_prefix = kwargs.pop('_no_prefix') if '_no_prefix' in kwargs else 'no_'
            opt_name = kwargs['option_strings'][0]
            if 'dest' not in kwargs:
                kwargs['dest'] = re.sub('^--', '', opt_name).replace('-', '_')
            kwargs['option_strings'] += [re.sub('^--'+self._yes_prefix, '--'+self._no_prefix, opt_name)]
            kwargs['nargs'] = 0
            def boolean(x):
                if not isinstance(x, bool):
                    raise TypeError('value not boolean: '+str(x))
                return x
            kwargs['type'] = boolean
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Sets the corresponding key to True or False depending on the option string used."""
        if len(args) == 0:
            kwargs['_yes_prefix'] = self._yes_prefix
            kwargs['_no_prefix'] = self._no_prefix
            return ActionYesNo(**kwargs)
        if args[3].startswith('--'+self._no_prefix):
            setattr(args[1], self.dest, False)
        else:
            setattr(args[1], self.dest, True)


class ActionJsonSchema(Action):
    """Action to parse option as json validated by a jsonschema."""
    def __init__(self, **kwargs):
        """Initializer for ActionJsonSchema instance.

        Args:
            schema (str or object): Schema to validate values against.

        Raises:
            ImportError: If jsonschema package is not available.
            ValueError: If a parameter is invalid.
            jsonschema.exceptions.SchemaError: If the schema is invalid.
        """
        if 'schema' in kwargs:
            if isinstance(jsonvalidator, Exception):
                raise ImportError('jsonschema is required by ActionJsonSchema :: '+str(jsonvalidator))
            _check_unknown_kwargs(kwargs, {'schema'})
            self._schema = kwargs['schema']
            if isinstance(self._schema, str):
                self._schema = yaml.safe_load(self._schema)
            jsonvalidator.check_schema(self._schema)
        elif not '_schema' in kwargs:
            raise ValueError('Expected schema keyword argument.')
        else:
            self._schema = kwargs.pop('_schema')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument with the corresponding jsonschema.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_schema'] = self._schema
            return ActionJsonSchema(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        islist = _is_action_value_list(self)
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('for ActionJsonSchema with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        for num, val in enumerate(value):
            try:
                if isinstance(val, str):
                    val = yaml.safe_load(val)
                    value[num] = val
                jsonschema.validate(val, schema=self._schema)
            except (TypeError, yaml.parser.ParserError, jsonschema.exceptions.ValidationError) as ex:
                elem = '' if not islist else ' element '+str(num+1)
                raise TypeError('parser key "'+self.dest+'"'+elem+': '+str(ex))
        return value if islist else value[0]


class ActionParser(Action):
    """Action to parse option with a given yamlargparse parser optionally loading from yaml file if string value."""
    def __init__(self, **kwargs):
        """Initializer for ActionParser instance.

        Args:
            parser (ArgumentParser): A yamlargparse parser to parse the option with.

        Raises:
            ValueError: If the parser parameter is invalid.
        """
        if 'parser' in kwargs:
            _check_unknown_kwargs(kwargs, {'parser'})
            self._parser = kwargs['parser']
            if not isinstance(self._parser, ArgumentParser):
                raise ValueError('Expected parser keyword argument to be a yamlargparse parser.')
        elif not '_parser' in kwargs:
            raise ValueError('Expected parser keyword argument.')
        else:
            self._parser = kwargs.pop('_parser')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument with the corresponding parser and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            kwargs['_parser'] = self._parser
            return ActionParser(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        try:
            if isinstance(value, str):
                yaml_path = Path(value, mode='fr')
                value = self._parser.parse_yaml_path(yaml_path())
            else:
                self._parser.check_config(value, skip_none=True)
        except TypeError as ex:
            raise TypeError(re.sub('^parser key ([^:]+):', 'parser key '+self.dest+'.\\1: ', str(ex)))
        return value

    @staticmethod
    def _fix_conflicts(parser, cfg):
        cfg_dict = parser.namespace_to_dict(cfg)
        for action in parser._actions:
            if isinstance(action, ActionParser) and action.dest in cfg_dict and cfg_dict[action.dest] is None:
                children = [x for x in cfg_dict.keys() if x.startswith(action.dest+'.')]
                if len(children) > 0:
                    delattr(cfg, action.dest)


class ActionOperators(Action):
    """Action to restrict a value with comparison operators."""
    _operators = {operator.gt: '>', operator.ge: '>=', operator.lt: '<', operator.le: '<=', operator.eq: '==', operator.ne: '!='}

    def __init__(self, **kwargs):
        """Initializer for ActionOperators instance.

        Args:
            expr (tuple or list[tuple]): Pairs of operators (> >= < <= == !=) and reference values, e.g. [('>=', 1),...].
            join (str): How to combine multiple comparisons, must be 'or' or 'and' (default='and').
            type (type): The value type (default=int).

        Raises:
            ValueError: If any of the parameters (expr, join or type) are invalid.
        """
        if 'expr' in kwargs:
            _check_unknown_kwargs(kwargs, {'expr', 'join', 'type'})
            self._type = kwargs['type'] if 'type' in kwargs else int
            self._join = kwargs['join'] if 'join' in kwargs else 'and'
            if self._join not in {'or', 'and'}:
                raise ValueError("Expected join to be either 'or' or 'and'.")
            _operators = {v: k for k, v in self._operators.items()}
            expr = [kwargs['expr']] if isinstance(kwargs['expr'], tuple) else kwargs['expr']
            if not isinstance(expr, list) or not all([all([len(x)==2, x[0] in _operators, x[1] == self._type(x[1])]) for x in expr]):
                raise ValueError('Expected expr to be a list of tuples each with a comparison operator (> >= < <= == !=) and a reference value of type '+self._type.__name__+'.')
            self._expr = [(_operators[x[0]], x[1]) for x in expr]
        elif not '_expr' in kwargs:
            raise ValueError('Expected expr keyword argument.')
        else:
            self._expr = kwargs.pop('_expr')
            self._join = kwargs.pop('_join')
            self._type = kwargs.pop('_type')
            if 'type' in kwargs:
                del kwargs['type']
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument restricted by the operators and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not valid.
        """
        if len(args) == 0:
            if 'nargs' in kwargs and kwargs['nargs'] == 0:
                raise ValueError('Invalid nargs='+str(kwargs['nargs'])+' for ActionOperators.')
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
            raise TypeError('for ActionOperators with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        def test_op(op, val, ref):
            try:
                return op(val, ref)
            except TypeError:
                return False
        for num, val in enumerate(value):
            try:
                val = self._type(val)
            except:
                raise TypeError('parser key "'+self.dest+'": invalid value, expected type to be '+self._type.__name__+' but got as value '+str(val)+'.')
            check = [test_op(op, val, ref) for op, ref in self._expr]
            if (self._join == 'and' and not all(check)) or (self._join == 'or' and not any(check)):
                expr = (' '+self._join+' ').join(['v'+self._operators[op]+str(ref) for op, ref in self._expr])
                raise TypeError('parser key "'+self.dest+'": invalid value, for v='+str(val)+' it is false that '+expr+'.')
            value[num] = val
        return value if islist else value[0]


class ActionPath(Action):
    """Action to check and store a path."""
    def __init__(self, **kwargs):
        """Initializer for ActionPath instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxFDRWX] as a keyword argument, e.g. ActionPath(mode='drw').

        Raises:
            ValueError: If the mode parameter is invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
        elif not '_mode' in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
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
            return ActionPath(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value, islist=None):
        islist = _is_action_value_list(self) if islist is None else islist
        if not islist:
            value = [value]
        elif not isinstance(value, list):
            raise TypeError('for ActionPath with nargs='+str(self.nargs)+' expected value to be list, received: value='+str(value)+'.')
        try:
            for num, val in enumerate(value):
                if isinstance(val, str):
                    val = Path(val, mode=self._mode)
                elif isinstance(val, Path):
                    val = Path(val(absolute=False), mode=self._mode, cwd=val.cwd)
                else:
                    raise TypeError('expected either a string or a Path object, received: value='+str(val)+' type='+str(type(val))+'.')
                value[num] = val
        except TypeError as ex:
            raise TypeError('parser key "'+self.dest+'": '+str(ex))
        return value if islist else value[0]


class ActionPathList(Action):
    """Action to check and store a list of file paths read from a plain text file or stream."""
    def __init__(self, **kwargs):
        """Initializer for ActionPathList instance.

        Args:
            mode (str): The required type and access permissions among [fdrwxFDRWX] as a keyword argument, e.g. ActionPathList(mode='fr').
            rel (str): Whether relative paths are with respect to current working directory 'cwd' or the list's parent directory 'list' (default='cwd').

        Raises:
            ValueError: If any of the parameters (mode or rel) are invalid.
        """
        if 'mode' in kwargs:
            _check_unknown_kwargs(kwargs, {'mode', 'rel'})
            Path._check_mode(kwargs['mode'])
            self._mode = kwargs['mode']
            self._rel = kwargs['rel'] if 'rel' in kwargs else 'cwd'
            if self._rel not in {'cwd', 'list'}:
                raise ValueError('rel must be either "cwd" or "list", got '+str(self._rel)+'.')
        elif not '_mode' in kwargs:
            raise ValueError('Expected mode keyword argument.')
        else:
            self._mode = kwargs.pop('_mode')
            self._rel = kwargs.pop('_rel')
            kwargs['type'] = str
            super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Parses an argument as a PathList and if valid sets the parsed value to the corresponding key.

        Raises:
            TypeError: If the argument is not a valid PathList.
        """
        if len(args) == 0:
            if 'nargs' in kwargs:
                raise ValueError('nargs not allowed for ActionPathList.')
            kwargs['_mode'] = self._mode
            kwargs['_rel'] = self._rel
            return ActionPathList(**kwargs)
        setattr(args[1], self.dest, self._check_type(args[2]))

    def _check_type(self, value):
        if isinstance(value, str):
            path_list = value
            try:
                with sys.stdin if path_list == '-' else open(path_list, 'r') as f:
                    value = [x.strip() for x in f.readlines()]
            except:
                raise TypeError('problems reading path list: '+path_list)
            cwd = os.getcwd()
            if self._rel == 'list' and path_list != '-':
                os.chdir(os.path.abspath(os.path.join(path_list, os.pardir)))
            try:
                for num, val in enumerate(value):
                    try:
                        value[num] = Path(val, mode=self._mode)
                    except TypeError as ex:
                        raise TypeError('path number '+str(num+1)+' in list '+path_list+', '+str(ex))
            finally:
                os.chdir(cwd)
            return value
        else:
            return ActionPath._check_type(self, value, islist=True)


class Path(object):
    """Stores a (possibly relative) path and the corresponding absolute path.

    When a Path instance is created it is checked that: the path exists, whether
    it is a file or directory and whether has the required access permissions
    (f=file, d=directory, r=readable, w=writeable, x=executable, or the same in
    uppercase meaning not, e.g. W=not_writeable). The absolute path can be
    obtained without having to remember the working directory from when the
    object was created.
    """
    def __init__(self, path, mode:str='fr', cwd:str=None, skip_check:bool=False):
        """Initializer for Path instance.

        Args:
            path (str): The path to check and store.
            mode (str): The required type and access permissions among [fdrwxFDRWX].
            cwd (str): Working directory for relative paths. If None, then os.getcwd() is used.
            skip_check (bool): Whether to skip path checks.

        Raises:
            ValueError: If the provided mode is invalid.
            TypeError: If the path does not exist or does not agree with the mode.
        """
        self._check_mode(mode)
        if cwd is None:
            cwd = os.getcwd()

        if isinstance(path, Path):
            abs_path = path(absolute=True)
            path = path()
        elif not isinstance(path, str):
            raise TypeError('Expected path to be a string or a Path object.')
        else:
            abs_path = path if os.path.isabs(path) else os.path.join(cwd, path)

        if not skip_check:
            ptype = 'directory' if 'd' in mode else 'file'
            if not os.access(abs_path, os.F_OK):
                raise TypeError(ptype+' does not exist: '+abs_path)
            if 'd' in mode and not os.path.isdir(abs_path):
                raise TypeError('path is not a directory: '+abs_path)
            if 'f' in mode and not os.path.isfile(abs_path):
                raise TypeError('path is not a file: '+abs_path)
            if 'r' in mode and not os.access(abs_path, os.R_OK):
                raise TypeError(ptype+' is not readable: '+abs_path)
            if 'w' in mode and not os.access(abs_path, os.W_OK):
                raise TypeError(ptype+' is not writeable: '+abs_path)
            if 'x' in mode and not os.access(abs_path, os.X_OK):
                raise TypeError(ptype+' is not executable: '+abs_path)
            if 'D' in mode and os.path.isdir(abs_path):
                raise TypeError('path is a directory: '+abs_path)
            if 'F' in mode and os.path.isfile(abs_path):
                raise TypeError('path is a file: '+abs_path)
            if 'R' in mode and os.access(abs_path, os.R_OK):
                raise TypeError(ptype+' is readable: '+abs_path)
            if 'W' in mode and os.access(abs_path, os.W_OK):
                raise TypeError(ptype+' is writeable: '+abs_path)
            if 'X' in mode and os.access(abs_path, os.X_OK):
                raise TypeError(ptype+' is executable: '+abs_path)

        self.path = path
        self.abs_path = abs_path
        self.cwd = cwd

    def __str__(self):
        return self.abs_path

    def __call__(self, absolute=True):
        """Returns the path as a string.

        Args:
            absolute (bool): If false returns the original path given, otherwise the corresponding absolute path.
        """
        return self.abs_path if absolute else self.path

    @staticmethod
    def _check_mode(mode:str):
        if not isinstance(mode, str):
            raise ValueError('Expected mode to be a string.')
        if len(set(mode)-set('fdrwxFDRWX')) > 0:
            raise ValueError('Expected mode to only include [fdrwxFDRWX] flags.')


class ParserError(Exception):
    """Error raised when parsing a value fails."""
    pass


def usage_and_exit_error_handler(self, message):
    """Error handler to get the same behavior as in argparse.

    Args:
        self (ArgumentParser): The ArgumentParser object.
        message (str): The message describing the error being handled.
    """
    self.print_usage(sys.stderr)
    args = {'prog': self.prog, 'message': message}
    sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
    sys.exit(2)


@contextmanager
def _suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            yield None


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
        raise ValueError('Unknown keyword arguments: '+', '.join(set(kwargs.keys())-keys))
