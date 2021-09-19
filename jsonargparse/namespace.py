"""Classes and functions related to namespace objects."""

from argparse import Namespace as ArgparseNamespace
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


__all__ = [
    'Namespace',
    'namespace_to_dict',
    'strip_meta',
]


meta_keys = {'__default_config__', '__path__', '__orig__'}


def is_meta_key(key: Union[str, int]) -> bool:
    if not isinstance(key, str):
        return False
    leaf_key = key.rsplit('.', 1)[-1]
    return leaf_key in meta_keys


def strip_meta(cfg: Union['Namespace', Dict[str, Any]]) -> Union['Namespace', Dict[str, Any]]:
    """Removes all metadata keys from a configuration object.

    Args:
        cfg: The configuration object to strip.

    Returns:
        A copy of the configuration object without any metadata keys.
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


def namespace_to_dict(namespace: 'Namespace') -> Dict[str, Any]:
    """Converts a nested namespace into a nested dictionary.

    Args:
        namespace: Object to process.

    Returns:
        The nested dictionary.
    """
    return namespace.clone().as_dict()


class Namespace(ArgparseNamespace):
    """Extension of argparse's Namespace to support nesting and subscript access."""

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            super().__init__(**kwargs)
        else:
            if len(kwargs) != 0 or len(args) != 1 or type(args[0]) not in (ArgparseNamespace, dict):
                raise ValueError('Expected a single positional parameter of type argparse.Namespace or dict.')
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
        """
        key_split = key.split('.')
        if ' ' in key:
            raise KeyError('Spaces not allowed in keys: "'+key+'".')
        if any(k == '' for k in key_split):
            raise KeyError('Empty nested key: "'+key+'".')
        leaf_key = key_split[-1]
        parent_ns = self  # type: Optional[Namespace]
        parent_key = ''
        if len(key_split) > 1:
            parent_key = '.'.join(key_split[:-1])
            for num, subkey in enumerate(key_split[:-1]):
                if hasattr(parent_ns, subkey):
                    parent_ns = getattr(parent_ns, subkey)
                    if parent_ns is not None and not isinstance(parent_ns, Namespace):
                        parent_ns = None
                        break
                else:
                    parent_ns = None
                    break
        return leaf_key, parent_ns, parent_key

    def _create_nested_namespace(self, key: str) -> 'Namespace':
        """Creates a nested namespace object.

        Args:
            key: The key where the nested namespace is created.

        Returns:
            The created nested namespace.
        """
        parent_ns = self
        for key in key.split('.'):
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
        setattr(parent_ns, leaf_key, item)

    def __getitem__(self, key: str) -> Any:
        """Gets an item from a possibly nested namespace."""
        leaf_key, parent_ns, _ = self._parse_key(key)
        if parent_ns is None or not hasattr(parent_ns, leaf_key):
            raise KeyError('Key "'+key+'" not found in namespace.')
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
            leaf_key, parent_ns, _ = self._parse_key(key)
        except KeyError:
            return False
        return parent_ns and leaf_key in parent_ns.__dict__

    def as_dict(self) -> Dict[str, Any]:
        """Converts the nested namespaces into nested dictionaries."""
        dic = {}
        for key, val in vars(self).items():
            if isinstance(val, Namespace):
                val = val.as_dict()
            dic[key] = val
        return dic

    def as_flat(self) -> ArgparseNamespace:
        """Converts the nested namespaces into a single argparse flat namespace."""
        flat = ArgparseNamespace()
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
                key_split = key.split('.')
                for num in range(len(key_split)-1):
                    parent_key = '.'.join(key_split[:num+1])
                    if parent_key not in keys:
                        keys.append(parent_key)
        keys.sort(key=lambda x: -len(x.split('.')))
        return keys

    def clone(self) -> 'Namespace':
        """Creates an new identical nested namespace."""
        return deepcopy(self)

    def update(self, value: Union['Namespace', Any], key: Optional[str] = None, only_unset: bool = False) -> 'Namespace':
        """Sets or replaces all items from the given nested namespace."""
        if not isinstance(value, Namespace):
            if not key:
                raise KeyError('Key is required if value not a Namespace.')
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
        except KeyError:
            return default

    def get_value_and_parent(self, key: str) -> Tuple[Any, 'Namespace', str]:
        leaf_key, parent_ns, _ = self._parse_key(key)
        return parent_ns[leaf_key], parent_ns, leaf_key

    def pop(self, key: str, default: Any = None) -> Any:
        leaf_key, parent_ns, _ = self._parse_key(key)
        if not parent_ns:
            return default
        value = parent_ns.__dict__.pop(leaf_key, default)
        return value


import argparse
argparse.Namespace = Namespace  # TODO: Change to temporal patch during parse
