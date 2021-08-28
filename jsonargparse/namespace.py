"""Classes and functions related to namespace objects."""

from argparse import Namespace as ArgparseNamespace
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple


__all__ = ['Namespace']


class Namespace(ArgparseNamespace):
    """Extension of argparse's Namespace to support nesting and subscript access."""

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            super().__init__(**kwargs)
        else:
            if len(kwargs) != 0 or len(args) != 1 or type(args[0]) is not ArgparseNamespace:
                raise ValueError('Expected a single positional parameter of type argparse.Namespace.')
            for key, val in vars(args[0]).items():
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
                    if not isinstance(parent_ns, Namespace):
                        raise KeyError('Accessing key "'+key+'" expected parent "'+'.'.join(key_split[:num+1])+'" to be a Namespace but is '+type(parent_ns).__name__+'.')
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
            if not hasattr(parent_ns, key):
                setattr(parent_ns, key, Namespace())
            parent_ns = getattr(parent_ns, key)
        return parent_ns

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
        try:
            leaf_key, parent_ns, _ = self._parse_key(key)
        except KeyError:
            return False
        return leaf_key in parent_ns.__dict__

    def as_dict(self) -> Dict[str, Any]:
        """Converts the nested namespaces into nested dictionaries."""
        dic = {}
        for key, val in vars(self).items():
            if isinstance(val, Namespace):
                val = val.as_dict()
            dic[key] = val
        return dic

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

    def clone(self) -> 'Namespace':
        """Creates an new identical nested namespace."""
        return deepcopy(self)

    def update(self, namespace: 'Namespace') -> None:
        """Sets or replaces all items from the given nested namespace."""
        for key, val in namespace.items():
            self[key] = val
