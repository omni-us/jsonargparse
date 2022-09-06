"""Classes and functions related to namespace objects."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, overload, Tuple, Union


__all__ = [
    'Namespace',
    'namespace_to_dict',
    'dict_to_namespace',
    'strip_meta',
]


meta_keys = {'__default_config__', '__path__', '__orig__'}


def split_key(key: str) -> List[str]:
    return key.split('.')


def split_key_root(key: str) -> List[str]:
    return key.split('.', 1)


def split_key_leaf(key: str) -> List[str]:
    return key.rsplit('.', 1)


def is_meta_key(key: Union[str, int]) -> bool:
    if not isinstance(key, str):
        return False
    leaf_key = split_key_leaf(key)[-1]
    return leaf_key in meta_keys


@overload
def strip_meta(cfg: 'Namespace') -> 'Namespace': ...
@overload
def strip_meta(cfg: Dict[str, Any]) -> Dict[str, Any]: ...


def strip_meta(cfg):
    """Removes all metadata keys from a configuration object.

    Args:
        cfg: The configuration object to strip.

    Returns:
        A deepcopy of the configuration object excluding all metadata keys.
    """
    cfg = deepcopy(cfg)

    del_keys = []
    for key in cfg.keys():
        if is_meta_key(key):
            del_keys.append(key)
        elif isinstance(cfg[key], dict):
            dic = cfg[key]
            for dic_key in [k for k in dic.keys() if is_meta_key(k)]:
                del dic[dic_key]

    for key in del_keys:
        del cfg[key]

    return cfg


@contextmanager
def patch_namespace():
    namespace_class = argparse.Namespace
    argparse.Namespace = Namespace
    try:
        yield
    finally:
        argparse.Namespace = namespace_class


def namespace_to_dict(namespace: 'Namespace') -> Dict[str, Any]:
    """Returns a deepcopy of a nested namespace converted into a nested dictionary."""
    return namespace.clone().as_dict()


def dict_to_namespace(cfg_dict: Union[Dict[str, Any], 'Namespace']) -> 'Namespace':
    """Converts a nested dictionary into a nested namespace."""
    cfg_dict = deepcopy(cfg_dict)
    def expand_dict(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict) and all(isinstance(k, str) for k in v.keys()):
                cfg[k] = expand_dict(v)
            elif isinstance(v, list):
                for nn, vv in enumerate(v):
                    if isinstance(vv, dict) and all(isinstance(k, str) for k in vv.keys()):
                        cfg[k][nn] = expand_dict(vv)
        return Namespace(**cfg)
    return expand_dict(cfg_dict)


class Namespace(argparse.Namespace):
    """Extension of argparse's Namespace to support nesting and subscript access."""

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            super().__init__(**kwargs)
        else:
            if len(kwargs) != 0 or len(args) != 1 or not isinstance(args[0], (argparse.Namespace, dict)):
                raise ValueError('Expected a single positional parameter of type Namespace or dict.')
            for key, val in (args[0].items() if type(args[0]) is dict else vars(args[0]).items()):
                self[key] = val

    def _parse_key(self, key: str) -> Tuple[str, Optional['Namespace'], str]:
        """Parses a key for the nested namespace.

        Args:
            key: The key that is being parsed.

        Returns:
            Tuple with three elements corresponding to:
            - The leaf key.
            - The parent namespace object.
            - The parent namespace key.

        Raises:
            KeyError: When given invalid key.
        """
        if ' ' in key:
            raise KeyError(f'Spaces not allowed in keys: "{key}".')
        key_split = split_key(key)
        if any(k == '' for k in key_split):
            raise KeyError(f'Empty nested key: "{key}".')
        leaf_key = key_split[-1]
        parent_ns: Namespace = self
        parent_key = ''
        if len(key_split) > 1:
            parent_key = '.'.join(key_split[:-1])
            for subkey in key_split[:-1]:
                if hasattr(parent_ns, subkey) or (isinstance(parent_ns, dict) and subkey in parent_ns):
                    parent_ns = parent_ns[subkey]
                    if parent_ns is not None and not isinstance(parent_ns, (Namespace, dict)):
                        return leaf_key, None, parent_key
                else:
                    return leaf_key, None, parent_key
        return leaf_key, parent_ns, parent_key

    def _parse_required_key(self, key: str) -> Tuple[str, 'Namespace', str]:
        """Same as _parse_key but raises KeyError if key not found."""
        leaf_key, parent_ns, parent_key = self._parse_key(key)
        if parent_ns is None or not hasattr(parent_ns, leaf_key):
            raise KeyError(f'Key "{key}" not found in namespace.')
        return leaf_key, parent_ns, parent_key

    def _create_nested_namespace(self, key: str) -> 'Namespace':
        """Creates a nested namespace object.

        Args:
            key: The key where the nested namespace is created.

        Returns:
            The created nested namespace.
        """
        parent_ns = self
        for key in split_key(key):
            if not isinstance(getattr(parent_ns, key, None), Namespace):
                setattr(parent_ns, key, Namespace())
            parent_ns = getattr(parent_ns, key)

        return parent_ns

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute to a possibly nested namespace."""
        if '.' in name:
            self.__setitem__(name, value)
        else:
            super().__setattr__(name, value)

    def __setitem__(self, key: str, item: Any) -> None:
        """Sets an item to a possibly nested namespace."""
        leaf_key, parent_ns, parent_key = self._parse_key(key)
        if parent_ns is None:
            parent_ns = self._create_nested_namespace(parent_key)
        if isinstance(parent_ns, dict):
            parent_ns[leaf_key] = item
        else:
            setattr(parent_ns, leaf_key, item)

    def __getitem__(self, key: str) -> Any:
        """Gets an item from a possibly nested namespace."""
        leaf_key, parent_ns, _ = self._parse_required_key(key)
        return getattr(parent_ns, leaf_key)

    def __delitem__(self, key: str) -> None:
        """Deletes an item from a possibly nested namespace."""
        leaf_key, parent_ns, _ = self._parse_key(key)
        del parent_ns.__dict__[leaf_key]

    def __contains__(self, key: str) -> bool:
        """Checks if an item is set possibly in a nested namespace."""
        if not isinstance(key, str):
            return False
        try:
            leaf_key, parent_ns, _ = self._parse_required_key(key)
        except KeyError:
            return False
        return leaf_key in parent_ns.__dict__

    def __bool__(self) -> bool:
        """Returns False if namespace is empty, otherwise True."""
        return bool(self.__dict__)

    def as_dict(self) -> Dict[str, Any]:
        """Converts the nested namespaces into nested dictionaries."""
        dic = {}
        for key, val in vars(self).items():
            if isinstance(val, Namespace):
                val = val.as_dict()
            elif isinstance(val, dict) and val != {} and all(isinstance(v, Namespace) for v in val.values()):
                val = {k: v.as_dict() for k, v in val.items()}
            elif isinstance(val, list) and val != [] and all(isinstance(v, Namespace) for v in val):
                val = [v.as_dict() for v in val]
            dic[key] = val
        return dic

    def as_flat(self) -> argparse.Namespace:
        """Converts the nested namespaces into a single argparse flat namespace."""
        flat = argparse.Namespace()
        for key, val in self.items():
            setattr(flat, key, val)
        return flat

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Returns a generator of all nested (key, value) items."""
        for key, val in vars(self).items():
            if isinstance(val, Namespace):
                for subkey, subval in val.items():
                    yield key+'.'+subkey, subval
            else:
                yield key, val

    def keys(self) -> Iterator[str]:
        """Returns a generator of all nested keys."""
        for key, _ in self.items():
            yield key

    def values(self) -> Iterator[Any]:
        """Returns a generator of all nested values."""
        for _, val in self.items():
            yield val

    def get_sorted_keys(self, branches: bool = True, key_filter: Callable = is_meta_key) -> List[str]:
        """Returns a list of keys sorted by descending depth.

        Args:
            branches: Whether to include branch keys instead of only leaves.
            key_filter: Function that selects keys to exclude.
        """
        keys = [k for k in self.keys() if not key_filter(k)]
        if branches:
            for key in [k for k in keys if '.' in k]:
                key_split = split_key(key)
                for num in range(len(key_split)-1):
                    parent_key = '.'.join(key_split[:num+1])
                    if parent_key not in keys:
                        keys.append(parent_key)
        keys.sort(key=lambda x: -len(split_key(x)))
        return keys

    def clone(self) -> 'Namespace':
        """Creates an new identical nested namespace."""
        return deepcopy(self)

    def update(self, value: Union['Namespace', Any], key: Optional[str] = None, only_unset: bool = False) -> 'Namespace':
        """Sets or replaces all items from the given nested namespace.

        Args:
            value: A namespace to update multiple values or other type to set in a single key.
            key: Branch key where to set the value. Required if value is not namespace.
            only_unset: Whether to only set the value if not set in namespace.
        """
        if not isinstance(value, Namespace):
            if not key:
                raise KeyError('Key is required if value not a Namespace.')
            if not only_unset or key not in self:
                self[key] = value
        else:
            prefix = key+'.' if key else ''
            for key, val in value.items():
                if not only_unset or prefix+key not in self:
                    self[prefix+key] = val
        return self

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def get_value_and_parent(self, key: str) -> Tuple[Any, 'Namespace', str]:
        leaf_key, parent_ns, _ = self._parse_required_key(key)
        return parent_ns[leaf_key], parent_ns, leaf_key

    def pop(self, key: str, default: Any = None) -> Any:
        leaf_key, parent_ns, _ = self._parse_key(key)
        if not parent_ns:
            return default
        return parent_ns.__dict__.pop(leaf_key, default)


# Temporal to provide backward compatibility in pytorch-lightning
import yaml
yaml.SafeDumper.add_representer(Namespace, lambda d, x: d.represent_mapping('tag:yaml.org,2002:map', x.as_dict()))
