"""Collection of types and type generators."""

import inspect
import operator
import os
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Type, Union

from ._common import is_final_class
from ._optionals import final, pydantic_support
from ._util import Path, get_import_path, get_private_kwargs, import_object

__all__ = [
    "final",
    "is_final_class",
    "register_type",
    "extend_base_type",
    "restricted_number_type",
    "restricted_string_type",
    "path_type",
    "PositiveInt",
    "NonNegativeInt",
    "PositiveFloat",
    "NonNegativeFloat",
    "ClosedUnitInterval",
    "OpenUnitInterval",
    "NotEmptyStr",
    "Email",
    "Path_fr",
    "Path_fc",
    "Path_dw",
    "Path_dc",
    "Path_drw",
    "SecretStr",
]


_operators1 = {
    operator.gt: ">",
    operator.ge: ">=",
    operator.lt: "<",
    operator.le: "<=",
    operator.eq: "==",
    operator.ne: "!=",
}
_operators2 = {v: k for k, v in _operators1.items()}

registered_types: Dict[tuple, type] = {}
registered_type_handlers: Dict[type, "RegisteredType"] = {}
registration_pending: Dict[str, Callable] = {}


def extend_base_type(
    name: str,
    base_type: type,
    validation_fn: Callable,
    docstring: Optional[str] = None,
    extra_attrs: Optional[dict] = None,
    register_key: Optional[Tuple] = None,
) -> type:
    """Creates and registers an extension of base type.

    Args:
        name: How the new type will be called.
        base_type: The type from which the created type is extended.
        validation_fn: Function that validates the value on instantiation/casting. Gets two arguments: type_class and
            value.
        docstring: The __doc__ attribute value for the created type.
        extra_attrs: Attributes set to the type class that the validation_fn can access.
        register_key: Used to determine the uniqueness of registered types.

    Raises:
        ValueError: If the type has already been registered with a different name.
    """
    if register_key in registered_types:
        registered_type = registered_types[register_key]
        if registered_type.__name__ != name:
            raise ValueError(f"Same type already registered with a different name: {registered_type.__name__}.")
        return registered_type

    class TypeCore:
        _validation_fn = validation_fn
        _type = base_type

        def __new__(cls, v):
            cls._validation_fn(cls, v)
            return super().__new__(cls, cls._type(v))

    if extra_attrs is not None:
        for key, value in extra_attrs.items():
            setattr(TypeCore, key, value)

    created_type = type(name, (TypeCore, base_type), {"__doc__": docstring})
    add_type(created_type, register_key)

    return created_type


def restricted_number_type(
    name: Optional[str],
    base_type: type,
    restrictions: Union[Tuple, List[Tuple]],
    join: str = "and",
    docstring: Optional[str] = None,
) -> type:
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
        raise ValueError("Expected base_type to be one of {int, float}.")
    if join not in {"or", "and"}:
        raise ValueError("Expected join to be one of {'or', 'and'}.")

    restrictions = [restrictions] if isinstance(restrictions, tuple) else restrictions
    if (
        not isinstance(restrictions, list)
        or not all(isinstance(x, tuple) and len(x) == 2 for x in restrictions)
        or not all(x[0] in _operators2 and x[1] == base_type(x[1]) for x in restrictions)
    ):
        raise ValueError(
            "Expected restrictions to be a list of tuples each with a comparison operator "
            f"(> >= < <= == !=) and a reference value of type {base_type.__name__}."
        )

    register_key = (tuple(sorted(restrictions)), base_type, join)

    restrictions = [(_operators2[x[0]], x[1]) for x in restrictions]
    expression = (" " + join + " ").join(["v" + _operators1[op] + str(ref) for op, ref in restrictions])

    if name is None:
        name = base_type.__name__
        for num, (comparison, ref) in enumerate(restrictions):
            name += "_" + join + "_" if num > 0 else "_"
            name += comparison.__name__ + str(ref).replace(".", "")

    extra_attrs = {
        "_restrictions": restrictions,
        "_expression": expression,
        "_join": join,
        "_type": base_type,
    }

    def validation_fn(cls, v):
        if isinstance(v, bool):
            raise ValueError(f"{v} not a number")
        if cls._type == int and isinstance(v, float) and not float.is_integer(v):
            raise ValueError(f"{v} not an integer")
        vv = cls._type(v)
        check = [comparison(vv, ref) for comparison, ref in cls._restrictions]
        if (cls._join == "and" and not all(check)) or (cls._join == "or" and not any(check)):
            raise ValueError(f"{v} does not conform to restriction {cls._expression}")

    return extend_base_type(
        name=name,
        base_type=base_type,
        validation_fn=validation_fn,
        register_key=register_key,
        docstring=docstring,
        extra_attrs=extra_attrs,
    )


def restricted_string_type(
    name: str,
    regex: Union[str, Pattern],
    docstring: Optional[str] = None,
) -> type:
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
    expression = "matching " + regex.pattern

    extra_attrs = {
        "_regex": regex,
        "_expression": expression,
        "_type": str,
    }

    def validation_fn(cls, v):
        if not cls._regex.match(v):
            raise ValueError(f"{v} does not match regular expression {cls._regex.pattern}")

    return extend_base_type(
        name=name,
        base_type=str,
        validation_fn=validation_fn,
        register_key=(expression, str),
        docstring=docstring,
        extra_attrs=extra_attrs,
    )


def _is_path_type(value, type_class):
    return isinstance(value, Path)


def path_type(mode: str, docstring: Optional[str] = None, **kwargs) -> type:
    """Creates or returns an already registered path type class.

    Args:
        mode: The required type and access permissions among [fdrwxcuFDRWX].
        docstring: Docstring for the type class.

    Returns:
        The created or retrieved type class.
    """
    Path._check_mode(mode)
    name = "Path_" + mode
    key_name = "path " + "".join(sorted(mode))

    skip_check = get_private_kwargs(kwargs, skip_check=False)
    if skip_check:
        from ._deprecated import path_skip_check_deprecation

        path_skip_check_deprecation(stacklevel=4)
        name += "_skip_check"
        key_name += " skip_check"

    register_key = (key_name, str)
    if register_key in registered_types:
        return registered_types[register_key]

    class PathType(Path):
        _expression = name
        _mode = mode
        _skip_check = skip_check
        _type = str

        def __init__(self, v, **k):
            super().__init__(v, mode=self._mode, skip_check=self._skip_check, **k)

    restricted_type = type(name, (PathType,), {"__doc__": docstring})
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
        return all(getattr(self, k) == getattr(other, k) for k in ["type_class", "serializer", "base_deserializer"])

    def is_value_of_type(self, value):
        return self.type_check(value, self.type_class)

    def deserializer(self, value):
        try:
            return self.base_deserializer(value)
        except self.deserializer_exceptions as ex:
            type_class_name = getattr(self.type_class, "__name__", str(self.type_class))
            raise ValueError(f"Not of type {type_class_name}: {ex}") from ex


def register_type(
    type_class: Any,
    serializer: Callable = str,
    deserializer: Optional[Callable] = None,
    deserializer_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        ValueError,
        TypeError,
        AttributeError,
    ),
    type_check: Callable = lambda v, t: v.__class__ == t,
    fail_already_registered: bool = True,
    uniqueness_key: Optional[Tuple] = None,
) -> None:
    """Registers a new type for use in jsonargparse parsers.

    Args:
        type_class: The type object to be registered.
        serializer: Function that converts an instance of the class to a basic type.
        deserializer: Function that converts a basic type to an instance of the class. Default instantiates type_class.
        deserializer_exceptions: Exceptions that deserializer raises when it fails.
        type_check: Function to check if a value is of type_class. Gets as arguments the value and type_class.
        fail_already_registered: Whether to fail if type has already been registered.
        uniqueness_key: Key to determine uniqueness of type.
    """
    type_handler = RegisteredType(type_class, serializer, deserializer, deserializer_exceptions, type_check)
    fail_already_registered = globals().get("_fail_already_registered", fail_already_registered)
    if not uniqueness_key and fail_already_registered and get_registered_type(type_class):
        if type_handler == registered_type_handlers[type_class]:
            return
        raise ValueError(f'Type "{type_class}" already registered with different serializer and/or deserializer.')
    registered_type_handlers[type_class] = type_handler
    if uniqueness_key is not None:
        registered_types[uniqueness_key] = type_class


def register_type_on_first_use(import_path: str, *args, **kwargs):
    registration_pending[import_path] = lambda: register_type(
        import_object(import_path),
        *args,
        **kwargs,
    )


def get_registered_type(type_class) -> Optional[RegisteredType]:
    if type_class not in registered_type_handlers:
        from contextlib import suppress

        with suppress(AttributeError, ValueError):
            import_path = get_import_path(type_class)
            if import_path in registration_pending:
                registration_pending.pop(import_path)()
    return registered_type_handlers.get(type_class)


def add_type(type_class: Type, uniqueness_key: Optional[Tuple], type_check: Optional[Callable] = None):
    assert uniqueness_key not in registered_types
    if type_class.__name__ in globals():
        raise ValueError(f'Type name "{type_class.__name__}" clashes with name already defined in jsonargparse.typing.')
    globals()[type_class.__name__] = type_class
    kwargs = {"uniqueness_key": uniqueness_key}
    if type_check is not None:
        kwargs["type_check"] = type_check  # type: ignore
    register_type(type_class, type_class._type, **kwargs)  # type: ignore


_fail_already_registered = False

PositiveInt = restricted_number_type("PositiveInt", int, (">", 0), docstring="int restricted to be >0")
NonNegativeInt = restricted_number_type("NonNegativeInt", int, (">=", 0), docstring="int restricted to be ≥0")
PositiveFloat = restricted_number_type("PositiveFloat", float, (">", 0), docstring="float restricted to be >0")
NonNegativeFloat = restricted_number_type("NonNegativeFloat", float, (">=", 0), docstring="float restricted to be ≥0")
ClosedUnitInterval = restricted_number_type(
    "ClosedUnitInterval", float, [(">=", 0), ("<=", 1)], docstring="float restricted to be ≥0 and ≤1"
)
OpenUnitInterval = restricted_number_type(
    "OpenUnitInterval", float, [(">", 0), ("<", 1)], docstring="float restricted to be >0 and <1"
)

NotEmptyStr = restricted_string_type(
    "NotEmptyStr", r"^.*[^ ].*$", docstring=r"str restricted to not-empty pattern ^.*[^ ].*$"
)
Email = restricted_string_type(
    "Email", r"^[^@ ]+@[^@ ]+\.[^@ ]+$", docstring=r"str restricted to the email pattern ^[^@ ]+@[^@ ]+\.[^@ ]+$"
)

Path_fr = path_type("fr", docstring="path to a file that exists and is readable")
Path_fc = path_type("fc", docstring="path to a file that can be created if it does not exist")
Path_dw = path_type("dw", docstring="path to a directory that exists and is writeable")
Path_dc = path_type("dc", docstring="path to a directory that can be created if it does not exist")
Path_drw = path_type("drw", docstring="path to a directory that exists and is readable and writeable")

register_type(os.PathLike, str, str)
register_type(complex)
register_type_on_first_use("decimal.Decimal", float)
register_type_on_first_use("uuid.UUID")

for _path in [pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath]:
    register_type(_path, str, _path, type_check=isinstance)


def timedelta_deserializer(value):
    def raise_error():
        raise ValueError(f'Expected a string with form "h:m:s" or "d days, h:m:s" but got "{value}"')

    if not isinstance(value, str):
        raise_error()
    pattern = r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)"
    if "day" in value:
        pattern = r"(?P<days>[-\d]+) day[s]*, " + pattern
    match = re.match(pattern, value)
    if not match:
        raise_error()
    kwargs = {key: float(val) for key, val in match.groupdict().items()}
    from datetime import timedelta

    return timedelta(**kwargs)


register_type_on_first_use("datetime.timedelta", deserializer=timedelta_deserializer)


def bytes_serializer(value: Union[bytes, bytearray]) -> str:
    from base64 import b64encode

    return b64encode(value).decode()


def bytes_deserializer(value: str) -> bytes:
    from base64 import b64decode

    return b64decode(value)


def bytearray_deserializer(value: str) -> bytearray:
    from base64 import b64decode

    return bytearray(b64decode(value))


register_type_on_first_use("builtins.bytes", serializer=bytes_serializer, deserializer=bytes_deserializer)
register_type_on_first_use("builtins.bytearray", serializer=bytes_serializer, deserializer=bytearray_deserializer)


def range_serializer(value):
    if value.step == 1:
        if value.start == 0:
            return f"range({value.stop})"
        return f"range({value.start}, {value.stop})"
    return f"range({value.start}, {value.stop}, {value.step})"


re_range_stop = re.compile(r"^(-?\d+)$")
re_range_start_stop = re.compile(r"^(-?\d+),(-?\d+)$")
re_range_start_stop_step = re.compile(r"^(-?\d+),(-?\d+),(-?\d+)$")


def range_deserializer(value):
    value = value.strip()
    if value.startswith("range(") and value.endswith(")"):
        value = value[6:-1].replace(" ", "")
        match = re_range_stop.match(value)
        if match:
            return range(int(match[1]))
        match = re_range_start_stop.match(value)
        if match:
            return range(int(match[1]), int(match[2]))
        match = re_range_start_stop_step.match(value)
        if match:
            return range(int(match[1]), int(match[2]), int(match[3]))
    raise ValueError("Expected 'range(<stop>)' or 'range(<start>, <stop>)' or 'range(<start>, <stop>, <step>)'")


register_type(range, serializer=range_serializer, deserializer=range_deserializer)


class SecretStr:
    """Holds a secret string that serializes to **********."""

    def __init__(self, value: str):
        self._value = value

    def __str__(self) -> str:
        return "**********"

    def __len__(self) -> int:
        return len(self._value)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def get_secret_value(self) -> str:
        """Returns the actual secret value."""
        return self._value


register_type(SecretStr)
register_type_on_first_use("pydantic.SecretStr")


def pydantic_deserializer(type_class):
    from pydantic import create_model  # pylint: disable=no-name-in-module

    pydantic_model = create_model("pydantic_model", pydantic_field=(type_class, ...))

    def deserialize(value):
        return pydantic_model(pydantic_field=value).pydantic_field

    return deserialize


def pydantic_serializer(type_class):
    if type_class.__name__ == "Url":
        return str
    serializer = str
    for base in [int, float, bool, list, dict, (set, list)]:
        if not isinstance(base, tuple):
            base = (base, base)
        if issubclass(type_class, base[0]):
            serializer = base[1]
            break
    return serializer


pydantic_type_modules = {
    "pydantic_core._pydantic_core",
    "pydantic.types",
    "pydantic.networks",
    "pydantic_extra_types",
}


def is_pydantic_type(type_class):
    return (
        pydantic_support
        and inspect.isclass(type_class)
        and any(getattr(t, "__module__", "") in pydantic_type_modules for t in inspect.getmro(type_class))
    )


def register_pydantic_type(type_class):
    from ._optionals import is_annotated

    if is_annotated(type_class):
        type_class = type_class.__origin__
    if not is_pydantic_type(type_class):
        return
    if not get_registered_type(type_class):
        from pydantic import ValidationError

        register_type(
            type_class=type_class,
            serializer=pydantic_serializer(type_class),
            deserializer=pydantic_deserializer(type_class),
            deserializer_exceptions=(ValidationError, TypeError),
        )


del _fail_already_registered
