
import os
import re
import argparse
from argparse import *
from types import SimpleNamespace
import yaml


__version__ = '0.0.0'


def raise_(ex):
    """Raise that works within lambda functions.
    
    Args:
        ex (Exception): The exception object to raise.    
    """
    raise ex


class ArgumentParser(argparse.ArgumentParser): #pylint: disable=function-redefined
    """Extension to python's argparse which simplifies parsing of configuration
    options from command line arguments, yaml configuration files, environment
    variables and hard-coded defaults.
    """

    groups = dict()

    def parse_args(self, *args, merge_env=True, **kwargs):
        """Parses command line argument strings.

        All the arguments from `argparse.ArgumentParser.parse_args
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args>`_
        are supported. Additionally it accepts:

        Args:
            merge_env (bool): Whether environment variables should be parsed and merged.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        cfg = super().parse_args(*args, **kwargs)
        cfg = self.args_to_dict(cfg)
        if merge_env:
            cfg_env = self.parse_env(merge_defaults=False)
            cfg = self.merge_config(self.namespace_to_dict(cfg_env), cfg)
        return self.dict_to_namespace(cfg)


    def parse_yaml(self, file_path, merge_defaults=True):
        """Parses a yaml file given its path.

        Args:
            file_path (str): Path to the yaml file to parse.
            merge_defaults (bool): Whether to merge with the parser's defaults.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        with open(file_path, 'r') as f:
            return self.parse_yaml_from_string(f.read(), merge_defaults)


    def parse_yaml_from_string(self, yaml_str, merge_defaults=True):
        """Parses yaml given as a string.

        Args:
            yaml_str (str): The yaml content.
            merge_defaults (bool): Whether to merge with the parser's defaults.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        cfg = yaml.safe_load(yaml_str)
        self.check_config(cfg)
        if merge_defaults:
            cfg = self.merge_config(cfg, self.parse_args([], merge_env=False))
        return self.dict_to_namespace(cfg)


    def dump_yaml(self, cfg):
        """Generates a yaml string for a configuration object.

        Args:
            cfg (SimpleNamespace | dict): The configuration object to dump.

        Returns:
            str: The configuration in yaml format.
        """
        if not isinstance(cfg, dict):
            cfg = self.namespace_to_dict(cfg)
        self.check_config(cfg, skip_none=True)
        return yaml.dump(cfg, default_flow_style=False)


    def parse_env(self, env=None, merge_defaults=True):
        """Parses environment variables.

        Args:
            env (object): The environment object to use, if None `os.environ` is used.
            merge_defaults (bool): Whether to merge with the parser's defaults.

        Returns:
            SimpleNamespace: An object with all parsed values as nested attributes.
        """
        if env is None:
            env = os.environ
        cfg = {}
        for action in self.__dict__['_actions']:
            env_var = (self.prog+'_' if self.prog else '') + action.dest
            env_var = env_var.replace('.', '__').upper()
            if env_var in env:
                cfg[action.dest] = self.parse_value(env[env_var], action, env_var)

        cfg = self.args_to_dict(SimpleNamespace(**cfg))
        if merge_defaults:
            cfg = self.merge_config(cfg, self.parse_args([], merge_env=False))

        return self.dict_to_namespace(cfg)


    def add_argument(self, *args, **kwargs):
        """Define how a single configuration argument should be parsed.

        All the arguments from `argparse.ArgumentParser.add_argument
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
        are supported.
        """
        ArgumentParser._add_argument(super(), self, args, kwargs)


    def add_argument_group(self, *args, name=None, **kwargs):
        """Adds a group to the parser.

        All the arguments from `argparse.ArgumentParser.add_argument_group
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group>`_
        are supported.  Additionally it accepts:

        Args:
            name (str): Name of the group. If set the group object will be included in the parser.groups dict.

        Returns:
            The group object.
        """
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        if name is not None:
            self.groups[name] = group
        return group


    def check_config(self, cfg, skip_none=False):
        """Checks that the content of a given configuration object conforms with the parser.

        Args:
            cfg (SimpleNamespace | dict): The configuration object to check.
            skip_none (bool): Whether to skip checking of values that are None.

        Raises:
            Exception: For any part of the configuration object that does not conform.
        """
        if not isinstance(cfg, dict):
            cfg = self.namespace_to_dict(cfg)

        def find_action(dest):
            for action in self.__dict__['_actions']:
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
                    self.parse_value(val, find_action(kbase), kbase)

        check_values(cfg)


    @staticmethod
    def parse_value(value, action, key):
        """Parses a value for a given action.

        Args:
            value (object): The value to parse.
            action (Action): The action used for parsing.
            key (str): The configuration key.

        Raises:
            Exception: If parsing of value fails.
        """
        if action is None:
            raise Exception('Unexpected configuration key: '+key)
        if action.type is None:
            if action.choices is not None:
                if value not in set(action.choices):
                    raise Exception('Unexpected configuration value for key: '+key)
            elif not isinstance(value, str):
                raise Exception('Unexpected configuration value for key: '+key)
        else:
            try:
                value = action.type(value)
            except:
                raise Exception('Unexpected configuration value for key: '+key)
        return value


    @staticmethod
    def merge_config(cfg_from, cfg_to):
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
                else:
                    cfg_to[k] = merge_values(cfg_from[k], cfg_to[k])
            return cfg_to

        out_dict = isinstance(cfg_from, dict)
        cfg_from = cfg_from if isinstance(cfg_from, dict) else ArgumentParser.namespace_to_dict(cfg_from)
        cfg_to = cfg_to if isinstance(cfg_to, dict) else ArgumentParser.namespace_to_dict(cfg_to)
        cfg = merge_values(cfg_from, cfg_to.copy())
        return cfg if out_dict else ArgumentParser.dict_to_namespace(cfg)


    @staticmethod
    def _add_argument(_super, self, args, kwargs):
        """Auxiliary function to allow add_argument to support the ActionYesNo actions."""
        if 'action' in kwargs and kwargs['action'] == ActionYesNo:
            action_class = kwargs.pop('action')
            _super._add_action(action_class(*args, **kwargs))
        else:
            if 'metavar' not in kwargs and '.' in args[0]:
                kwargs['metavar'] = args[0].split('.')[-1].upper()
            _super.add_argument(*args, **kwargs)


    @staticmethod
    def args_to_dict(cfg_args):
        """Converts a flat parsed namespace into a nested dictionary.

        Args:
            cfg_args (SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        cfg_dict = dict()
        for k, v in vars(cfg_args).items():
            ksplit = k.split('.')
            if len(ksplit) == 1:
                cfg_dict[k] = v
            else:
                kdict = cfg_dict
                for num, kk in enumerate(ksplit[:len(ksplit)-1]):
                    if kk not in kdict:
                        kdict[kk] = dict()
                    elif not isinstance(kdict[kk], dict):
                        raise Exception('Conflicting namespace base: '+'.'.join(ksplit[:num]))
                    kdict = kdict[kk]
                if ksplit[-1] in kdict:
                    raise Exception('Conflicting namespace base: '+k)
                kdict[ksplit[-1]] = v
        return cfg_dict


    @staticmethod
    def dict_to_namespace(cfg_dict):
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
    def namespace_to_dict(cfg_ns):
        """Converts a nested namespace into a nested dictionary.

        Args:
            cfg_args (SimpleNamespace): The configuration to process.

        Returns:
            dict: The nested configuration dictionary.
        """
        def expand_namespace(cfg):
            cfg = vars(cfg)
            for k, v in cfg.items():
                if isinstance(v, SimpleNamespace):
                    cfg[k] = expand_namespace(v)
            return cfg
        return expand_namespace(cfg_ns)


class _ArgumentGroup(argparse._ArgumentGroup):
    def add_argument(self, *args, **kwargs):
        ArgumentParser._add_argument(super(), self, args, kwargs)


class ActionYesNo(Action): 
    """Paired --opt, --no_opt action."""
    # Based on https://stackoverflow.com/questions/9234258/in-python-argparse-is-it-possible-to-have-paired-no-something-something-arg

    def __init__(self, opt_name, dest=None, default=True, required=False, help=None):
        opt_name = re.sub('^--', '', opt_name)
        if dest is None:
            dest = opt_name.replace('-', '_')
        bool_type = lambda x: x if isinstance(x, bool) else raise_(ValueError)
        super().__init__(['--' + opt_name, '--no_' + opt_name], dest, nargs=0, const=None, default=default, required=required, help=help, type=bool_type)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no_'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)
