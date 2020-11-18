"""Collection of types and type generators."""

import re
import operator
from typing import Dict, List, Tuple, Any, Union, Optional, Type, Pattern
from .util import Path


__all__ = [
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
    'Email',
    'Path_fr',
    'Path_dw',
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
_schema_operator_map = {
    operator.gt: 'exclusiveMinimum',
    operator.ge: 'minimum',
    operator.lt: 'exclusiveMaximum',
    operator.le: 'maximum',
}

registered_types = {}  # type: Dict[Tuple, Any]


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
                         '(> >= < <= == !=) and a reference value of type '+base_type.__name__+'.')

    register_key = (tuple(sorted(restrictions)), base_type, join)
    if register_key in registered_types:
        registered_type = registered_types[register_key]
        if name is not None and registered_type.__name__ != name:
            raise ValueError('Same type already registered with a different name: '+registered_type.__name__+'.')
        return registered_type

    restrictions = [(_operators2[x[0]], x[1]) for x in restrictions]
    expression = (' '+join+' ').join(['v'+_operators1[op]+str(ref) for op, ref in restrictions])

    class RestrictedNumber:

        _restrictions = restrictions
        _expression = expression
        _type = base_type
        _join = join

        def __new__(cls, v):
            def within_restriction(cls, v):
                check = [comparison(v, ref) for comparison, ref in cls._restrictions]
                if (cls._join == 'and' and not all(check)) or \
                   (cls._join == 'or' and not any(check)):
                    return False
                return True

            v = cls._type(v)
            if not within_restriction(cls, v):
                raise ValueError('invalid value, '+str(v)+' does not conform to restriction '+cls._expression)
            return super().__new__(cls, v)

    if name is None:
        name = base_type.__name__
        for num, (comparison, ref) in enumerate(restrictions):
            name += '_'+join+'_' if num > 0 else '_'
            name += comparison.__name__ + str(ref).replace('.', '')


    restricted_type = type(name, (RestrictedNumber, base_type), {})
    if docstring is not None:
        restricted_type.__doc__ = docstring
    registered_types[register_key] = restricted_type

    return restricted_type


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

    register_key = (expression, str)
    if register_key in registered_types:
        registered_type = registered_types[register_key]
        if registered_type.__name__ != name:
            raise ValueError('Same type already registered with a different name: '+registered_type.__name__+'.')
        return registered_type

    class RestrictedString:

        _regex = regex
        _expression = expression
        _type = str

        def __new__(cls, v):
            v = str(v)
            if not cls._regex.match(v):
                raise ValueError('invalid value, '+v+' does not match regular expression '+cls._expression)
            return super().__new__(cls, v)

    restricted_type = type(name, (RestrictedString, str), {})
    if docstring is not None:
        restricted_type.__doc__ = docstring
    registered_types[register_key] = restricted_type

    return restricted_type


def path_type(
    mode: str,
    docstring: Optional[str] = None,
) -> Type:
    """Creates or returns an already registered path type class.

    Args:
        mode: The required type and access permissions among [fdrwxcuFDRWX].
        docstring: Docstring for the type class.

    Returns:
        The created or retrieved type class.
    """
    expression = 'path mode='+mode
    name = 'Path_'+mode

    register_key = ('path '+''.join(sorted(mode)), str)
    if register_key in registered_types:
        return registered_types[register_key]

    class PathType(Path):

        _expression = expression
        _mode = mode
        _type = str

        def __init__(self, v):
            super().__init__(v, mode=self._mode)

        def __repr__(self):
            return self.path

    restricted_type = type(name, (PathType, str), {})
    if docstring is not None:
        restricted_type.__doc__ = docstring
    registered_types[register_key] = restricted_type

    return restricted_type


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

Email = restricted_string_type('Email', r'^[^@ ]+@[^@ ]+\.[^@ ]+$',
                               docstring=r'str restricted to the email pattern ^[^@ ]+@[^@ ]+\.[^@ ]+$')

Path_fr = path_type('fr',
                    docstring='str pointing to a file that exists and is readable')
Path_dw = path_type('dw',
                    docstring='str pointing to a directory that exists and is writeable')


def annotation_to_schema(annotation) -> Optional[Dict[str, str]]:
    """Generates a json schema from a type annotation if possible.

    Args:
        annotation: The type annotation to process.

    Returns:
        The json schema or None if an unsupported type.
    """
    schema = None
    if issubclass(annotation, (int, float)) and \
       hasattr(annotation, '_join') and annotation._join == 'and' and \
       hasattr(annotation, '_restrictions') and \
       all(x[0] in _schema_operator_map for x in annotation._restrictions):
        schema = {'type': 'integer' if issubclass(annotation, int) else 'number'}
        for comparison, ref in annotation._restrictions:
            schema[_schema_operator_map[comparison]] = ref
    elif issubclass(annotation, str) and hasattr(annotation, '_regex'):
        schema = {'type': 'string', 'pattern': annotation._regex.pattern}
    return schema
