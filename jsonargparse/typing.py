"""Collection of types and type generators."""

import operator
import os
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Type, Union
from .optionals import final
from .util import get_import_path, import_object, Path


__all__ = [
    'final',
    'is_final_class',
    'register_type',
    'registered_types',
    'restricted_number_type',
    'restricted_string_type',
    'path_type',
    'PositiveInt',
    'NonNegativeInt',
    'PositiveFloat',
    'NonNegativeFloat',
    'ClosedUnitInterval',
    'OpenUnitInterval',
    'NotEmptyStr',
    'Email',
    'Path_fr',
    'Path_fc',
    'Path_dw',
    'Path_dc',
    'Path_drw',
]


_operators1 = {
    operator.gt: '>',
    operator.ge: '>=',
    operator.lt: '<',
    operator.le: '<=',
    operator.eq: '==',
    operator.ne: '!=',
}
_operators2 = {v: k for k, v in _operators1.items()}

registered_types: Dict[Union[str, Type, Tuple], Any] = {}


def create_type(
    name: str,
    base_type: Type,
    check_value: Callable,
    register_key: Tuple = None,
    docstring: str = None,
    extra_attrs: dict = None,
) -> Type:

    if register_key in registered_types:
        registered_type = registered_types[register_key]
        if registered_type.__name__ != name:
            raise ValueError(f'Same type already registered with a different name: {registered_type.__name__}.')
        return registered_type

    class TypeCore:

        _check_value = check_value

        def __new__(cls, v):
            cls._check_value(cls, v)
            return super().__new__(cls, v)

    if extra_attrs is not None:
        for key, value in extra_attrs.items():
            setattr(TypeCore, key, value)

    created_type = type(name, (TypeCore, base_type), {'__doc__': docstring})
    add_type(created_type, register_key)

    return created_type


def restricted_number_type(
    name: Optional[str],
    base_type: Type,
    restrictions: Union[Tuple, List[Tuple]],
    join: str = 'and',
    docstring: Optional[str] = None,
) -> Type:
    """Creates or returns an already registered restricted number type class.

    Args:
        name: Name for the type or None for an automatic name.
        base_type: One of {int, float}.
        restrictions: Tuples of pairs (comparison, reference), e.g. ('>', 0).
        join: How to combine multiple comparisons, one of {'or', 'and'}.
        docstring: Docstring for the type class.

    Returns:
        The created or retrieved type class.
    """
    if base_type not in {int, float}:
        raise ValueError('Expected base_type to be one of {int, float}.')
    if join not in {'or', 'and'}:
        raise ValueError("Expected join to be one of {'or', 'and'}.")

    restrictions = [restrictions] if isinstance(restrictions, tuple) else restrictions
    if not isinstance(restrictions, list) or \
       not all(isinstance(x, tuple) and len(x) == 2 for x in restrictions) or \
       not all(x[0] in _operators2 and x[1] == base_type(x[1]) for x in restrictions):
        raise ValueError('Expected restrictions to be a list of tuples each with a comparison operator '
                         f'(> >= < <= == !=) and a reference value of type {base_type.__name__}.')

    register_key = (tuple(sorted(restrictions)), base_type, join)

    restrictions = [(_operators2[x[0]], x[1]) for x in restrictions]
    expression = (' '+join+' ').join(['v'+_operators1[op]+str(ref) for op, ref in restrictions])

    if name is None:
        name = base_type.__name__
        for num, (comparison, ref) in enumerate(restrictions):
            name += '_'+join+'_' if num > 0 else '_'
            name += comparison.__name__ + str(ref).replace('.', '')

    extra_attrs = {
        '_restrictions': restrictions,
        '_expression': expression,
        '_join': join,
        '_type': base_type,
    }

    def check_value(cls, v):
        if cls._type == int and isinstance(v, float) and not float.is_integer(v):
            raise ValueError(f'invalid value, {v} not an integer')
        vv = cls._type(v)
        check = [comparison(vv, ref) for comparison, ref in cls._restrictions]
        if (cls._join == 'and' and not all(check)) or \
           (cls._join == 'or' and not any(check)):
            raise ValueError(f'invalid value, "{v}" does not conform to restriction {cls._expression}')

    return create_type(
        name=name,
        base_type=base_type,
        check_value=check_value,
        register_key=register_key,
        docstring=docstring,
        extra_attrs=extra_attrs
    )


def restricted_string_type(
    name: str,
    regex: Union[str, Pattern],
    docstring: Optional[str] = None,
) -> Type:
    """Creates or returns an already registered restricted string type class.

    Args:
        name: Name for the type or None for an automatic name.
        regex: Regular expression that the string must match.
        docstring: Docstring for the type class.

    Returns:
        The created or retrieved type class.
    """
    if isinstance(regex, str):
        regex = re.compile(regex)
    expression = 'matching '+regex.pattern

    extra_attrs = {
        '_regex': regex,
        '_expression': expression,
        '_type': str,
    }

    def check_value(cls, v):
        if not cls._regex.match(v):
            raise ValueError(f'invalid value, "{v}" does not match regular expression {cls._regex.pattern}')

    return create_type(
        name=name,
        base_type=str,
        check_value=check_value,
        register_key=(expression, str),
        docstring=docstring,
        extra_attrs=extra_attrs
    )


def _is_path_type(value, type_class):
    return isinstance(value, Path)


def path_type(
    mode: str,
    docstring: Optional[str] = None,
    skip_check: bool = False,
) -> Type:
    """Creates or returns an already registered path type class.

    Args:
        mode: The required type and access permissions among [fdrwxcuFDRWX].
        docstring: Docstring for the type class.
        skip_check: Whether to skip path checks.

    Returns:
        The created or retrieved type class.
    """
    Path._check_mode(mode)
    name = 'Path_'+mode
    key_name = 'path '+''.join(sorted(mode))
    if skip_check:
        name += '_skip_check'
        key_name += ' skip_check'

    register_key = (key_name, str)
    if register_key in registered_types:
        return registered_types[register_key]

    class PathType(Path):

        _expression = name
        _mode = mode
        _skip_check = skip_check
        _type = str

        def __init__(self, v):
            super().__init__(v, mode=self._mode, skip_check=self._skip_check)

    restricted_type = type(name, (PathType, str), {'__doc__': docstring})
    add_type(restricted_type, register_key, type_check=_is_path_type)

    return restricted_type


class RegisteredType:
    def __init__(
        self,
        type_class: Any,
        serializer: Callable,
        deserializer: Optional[Callable],
        deserializer_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
        type_check: Callable,
    ):
        self.type_class = type_class
        self.serializer = serializer
        self.base_deserializer = type_class if deserializer is None else deserializer
        self.deserializer_exceptions = deserializer_exceptions
        self.type_check = type_check

    def __eq__(self, other):
        return all(getattr(self, k) == getattr(other, k) for k in ['type_class', 'serializer', 'base_deserializer'])

    def is_value_of_type(self, value):
        return self.type_check(value, self.type_class)

    def deserializer(self, value):
        try:
            return self.base_deserializer(value)
        except self.deserializer_exceptions as ex:
            type_class_name = getattr(self.type_class, '__name__', str(self.type_class))
            raise ValueError(f'Not of type {type_class_name}. {ex}') from ex


def register_type(
    type_class: Any,
    serializer: Callable = str,
    deserializer: Optional[Callable] = None,
    deserializer_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (ValueError, TypeError, AttributeError),
    type_check: Callable = lambda v, t: v.__class__ == t,
    fail_already_registered: bool = True,
    uniqueness_key: Optional[Tuple] = None,
):
    """Registers a new type for use in jsonargparse parsers.

    Args:
        type_class: The type object to be registered.
        serializer: Function that converts an instance of the class to a basic type.
        deserializer: Function that converts a basic type to an instance of the class. Default None instantiates type_class.
        deserializer_exceptions: Exceptions that deserializer raises when it fails.
        type_check: Function to check if a value is of type_class. Gets as arguments the value and type_class.
        fail_already_registered: Whether to fail if type has already been registered.
        uniqueness_key: Key to determine uniqueness of type.
    """
    type_wrapper = RegisteredType(type_class, serializer, deserializer, deserializer_exceptions, type_check)
    fail_already_registered = globals().get('_fail_already_registered', fail_already_registered)
    if not uniqueness_key and fail_already_registered and get_registered_type(type_class):
        if type_wrapper == registered_types[type_class]:
            return
        raise ValueError(f'Type "{type_class}" already registered with different serializer and/or deserializer.')
    registered_types[type_class] = type_wrapper
    if uniqueness_key is not None:
        registered_types[uniqueness_key] = type_class


def register_type_on_first_use(import_path: str, *args, **kwargs):
    registered_types[import_path] = lambda: register_type(
        import_object(import_path),
        *args,
        **kwargs,
    )


def get_registered_type(type_class):
    if type_class in registered_types:
        return registered_types[type_class]
    try:
        import_path = get_import_path(type_class)
        if import_path in registered_types:
            registered_types.pop(import_path)()
            return registered_types[type_class]
    except (AttributeError, ValueError):
        pass


def add_type(type_class: Type, uniqueness_key: Optional[Tuple], type_check: Callable = None):
    assert uniqueness_key not in registered_types
    if type_class.__name__ in globals():
        raise ValueError(f'Type name "{type_class.__name__}" clashes with name already defined in jsonargparse.typing.')
    globals()[type_class.__name__] = type_class
    kwargs = {'uniqueness_key': uniqueness_key}
    if type_check is not None:
        kwargs['type_check'] = type_check  # type: ignore
    register_type(type_class, type_class._type, **kwargs)  # type: ignore


def is_final_class(cls):
    """Checks whether a class is final, i.e. decorated with ``final``."""
    return getattr(cls, '__final__', False)


_fail_already_registered = False

PositiveInt        = restricted_number_type('PositiveInt',        int, ('>', 0),
                                            docstring='int restricted to be >0')
NonNegativeInt     = restricted_number_type('NonNegativeInt',     int, ('>=', 0),
                                            docstring='int restricted to be ≥0')
PositiveFloat      = restricted_number_type('PositiveFloat',      float, ('>', 0),
                                            docstring='float restricted to be >0')
NonNegativeFloat   = restricted_number_type('NonNegativeFloat',   float, ('>=', 0),
                                            docstring='float restricted to be ≥0')
ClosedUnitInterval = restricted_number_type('ClosedUnitInterval', float, [('>=', 0), ('<=', 1)],
                                            docstring='float restricted to be ≥0 and ≤1')
OpenUnitInterval   = restricted_number_type('OpenUnitInterval',   float, [('>', 0), ('<', 1)],
                                            docstring='float restricted to be >0 and <1')

NotEmptyStr = restricted_string_type('NotEmptyStr', r'^.*[^ ].*$',
                                     docstring=r'str restricted to not-empty pattern ^.*[^ ].*$')
Email       = restricted_string_type('Email', r'^[^@ ]+@[^@ ]+\.[^@ ]+$',
                                     docstring=r'str restricted to the email pattern ^[^@ ]+@[^@ ]+\.[^@ ]+$')

Path_fr = path_type('fr', docstring='str pointing to a file that exists and is readable')
Path_fc = path_type('fc', docstring='str pointing to a file that can be created if it does not exist')
Path_dw = path_type('dw', docstring='str pointing to a directory that exists and is writeable')
Path_dc = path_type('dc', docstring='str pointing to a directory that can be created if it does not exist')
Path_drw = path_type('drw', docstring='str pointing to a directory that exists and is readable and writeable')

register_type(os.PathLike, str, str)
register_type(complex)
register_type_on_first_use('uuid.UUID')


def timedelta_deserializer(value):
    def raise_error():
        raise ValueError(f'Expected a string with form "h:m:s" or "d days, h:m:s" but got "{value}"')
    if not isinstance(value, str):
        raise_error()
    pattern = r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)'
    if 'day' in value:
        pattern = r'(?P<days>[-\d]+) day[s]*, ' + pattern
    match = re.match(pattern, value)
    if not match:
        raise_error()
    kwargs = {key: float(val) for key, val in match.groupdict().items()}
    from datetime import timedelta
    return timedelta(**kwargs)


register_type_on_first_use('datetime.timedelta', deserializer=timedelta_deserializer)

del _fail_already_registered
