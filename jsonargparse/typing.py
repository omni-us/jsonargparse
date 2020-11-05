"""Collection of types and type generators."""

import operator
from typing import Dict, Tuple, Any


_operators1 = {operator.gt: '>',
               operator.ge: '>=',
               operator.lt: '<',
               operator.le: '<=',
               operator.eq: '==',
               operator.ne: '!='}
_operators2 = {v: k for k, v in _operators1.items()}


registered_types = {}  # type: Dict[Tuple, Any]


def restricted_number_type(name, base_type, restrictions, join='and'):
    """Function that creates or returns an already registered restricted number type class.

    Args:
        name (str): Name for the type.
        base_type (type): One of {int, float}.
        restrictions (tuple or list[tuple]): Tuples of pairs (comparison, reference), e.g. ('>', 0).
        join (str): How to combine multiple comparisons, one of {'or', 'and'}
    """

    if base_type not in {int, float}:
        raise ValueError('Expected base_type to be one of {int, float}.')
    if join not in {'or', 'and'}:
        raise ValueError("Expected join to be one of {'or', 'and'}.")

    restrictions = [restrictions] if isinstance(restrictions, tuple) else restrictions
    if not isinstance(restrictions, list) or \
       not all([all([len(x) == 2, x[0] in _operators2, x[1] == base_type(x[1])]) for x in restrictions]):
        raise ValueError('Expected restrictions to be a list of tuples each with a comparison operator '
                         '(> >= < <= == !=) and a reference value of type '+base_type.__name__+'.')

    register_key = (tuple(sorted(restrictions)), base_type, join)
    if register_key in registered_types:
        registered_type = registered_types[register_key]
        if registered_type.__name__ != name:
            raise ValueError('Same type already registered with a different name: '+registered_type.__name__+'.')
        return registered_type

    restrictions = [(_operators2[x[0]], x[1]) for x in restrictions]
    expression = (' '+join+' ').join(['v'+_operators1[op]+str(ref) for op, ref in restrictions])

    class RestrictedNumberType(base_type):

        _restrictions = restrictions
        _expression = expression
        _type = base_type
        _join = join

        def __new__(self, v):
            v = self._type(v)
            if not self._within_restriction(self, v):
                raise ValueError('invalid value, '+str(v)+' does not conform to restriction '+self._expression)
            return super().__new__(self, v)

        def _within_restriction(self, v):
            def test_comparison(operation, val, ref):
                try:
                    return operation(val, ref)
                except TypeError:
                    return False
            check = [test_comparison(op, v, ref) for op, ref in self._restrictions]
            if (self._join == 'and' and not all(check)) or \
               (self._join == 'or' and not any(check)):
                return False
            return True

    RestrictedNumberType.__name__ = name
    registered_types[register_key] = RestrictedNumberType
    return RestrictedNumberType


PositiveInt        = restricted_number_type('PositiveInt',        int, ('>', 0))
NonNegativeInt     = restricted_number_type('NonNegativeInt',     int, ('>=', 0))
PositiveFloat      = restricted_number_type('PositiveFloat',      float, ('>', 0))
NonNegativeFloat   = restricted_number_type('NonNegativeFloat',   float, ('>=', 0))
ClosedUnitInterval = restricted_number_type('ClosedUnitInterval', float, [('>=', 0), ('<=', 1)])
OpenUnitInterval   = restricted_number_type('OpenUnitInterval',   float, [('>', 0), ('<', 1)])
