from __future__ import annotations

import inspect
import pickle
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Union

import pytest

from jsonargparse import ArgumentError
from jsonargparse.typing import (
    ClosedUnitInterval,
    Email,
    NonNegativeFloat,
    NonNegativeInt,
    NotEmptyStr,
    OpenUnitInterval,
    PositiveFloat,
    PositiveInt,
    extend_base_type,
    get_registered_type,
    register_type,
    register_type_on_first_use,
    registered_types,
    registration_pending,
    restricted_number_type,
    restricted_string_type,
)
from jsonargparse_tests.conftest import get_parser_help


def test_public_api():
    import jsonargparse.typing

    names = {
        n
        for n, v in vars(jsonargparse.typing).items()
        if n[0] != "_"
        and getattr(v, "__module__", "").split(".")[0] not in {"jsonargparse", "typing"}
        and (inspect.isclass(v) or inspect.isfunction(v))
    }
    assert set() == names - set(jsonargparse.typing.__all__)


# restricted number tests


def test_positive_int():
    assert 1 == PositiveInt(1)
    assert 2 == PositiveInt("2")
    pytest.raises(ValueError, lambda: PositiveInt(0))
    pytest.raises(ValueError, lambda: PositiveInt("-3"))
    pytest.raises(ValueError, lambda: PositiveInt("4.0"))
    pytest.raises(ValueError, lambda: PositiveInt(5.6))


def test_non_negative_int():
    assert 0 == NonNegativeInt(0)
    assert 1 == NonNegativeInt("1")
    pytest.raises(ValueError, lambda: NonNegativeInt(-1))
    pytest.raises(ValueError, lambda: NonNegativeInt("-2"))
    pytest.raises(ValueError, lambda: NonNegativeInt("3.0"))
    pytest.raises(ValueError, lambda: NonNegativeInt(4.5))


def test_positive_float():
    assert 0.1 == PositiveFloat(0.1)
    assert 0.2 == PositiveFloat("0.2")
    assert 3.0 == PositiveFloat(3)
    pytest.raises(ValueError, lambda: PositiveFloat(0))
    pytest.raises(ValueError, lambda: PositiveFloat("-0.4"))


def test_non_negative_float():
    assert 0.0 == NonNegativeFloat(0.0)
    assert 0.1 == NonNegativeFloat("0.1")
    assert 2.0 == NonNegativeFloat(2)
    pytest.raises(ValueError, lambda: NonNegativeFloat(-0.1))
    pytest.raises(ValueError, lambda: NonNegativeFloat("-2"))


def test_closed_unit_interval():
    assert 0.0 == ClosedUnitInterval(0.0)
    assert 1.0 == ClosedUnitInterval("1")
    assert 0.5 == ClosedUnitInterval(0.5)
    pytest.raises(ValueError, lambda: ClosedUnitInterval(-0.1))
    pytest.raises(ValueError, lambda: ClosedUnitInterval("1.1"))


def test_open_unit_interval():
    assert 0.1 == OpenUnitInterval(0.1)
    assert 0.9 == OpenUnitInterval("0.9")
    assert 0.5 == OpenUnitInterval(0.5)
    pytest.raises(ValueError, lambda: OpenUnitInterval(0))
    pytest.raises(ValueError, lambda: OpenUnitInterval("1.0"))


def test_restricted_number_invalid_type():
    pytest.raises(ValueError, lambda: restricted_number_type("Invalid", str, ("<", 0)))
    pytest.raises(ValueError, lambda: restricted_number_type("Invalid", int, ("<", 0), join="xor"))
    pytest.raises(ValueError, lambda: restricted_number_type("Invalid", int, ["<", 0]))


def test_restricted_number_already_registered():
    NewClosedUnitInterval = restricted_number_type("ClosedUnitInterval", float, [("<=", 1), (">=", 0)])
    assert ClosedUnitInterval is NewClosedUnitInterval
    with pytest.raises(ValueError):
        restricted_number_type("NewName", float, [("<=", 1), (">=", 0)])


def test_restricted_number_not_equal_operator():
    NotTwoOrThree = restricted_number_type("NotTwoOrThree", float, [("!=", 2), ("!=", 3)])
    assert 1.0 == NotTwoOrThree(1)
    pytest.raises(ValueError, lambda: NotTwoOrThree(2))
    pytest.raises(ValueError, lambda: NotTwoOrThree("3"))


def test_restricted_number_operator_join():
    PositiveOrMinusOne = restricted_number_type("PositiveOrMinusOne", float, [(">", 0), ("==", -1)], join="or")
    assert 1.0 == PositiveOrMinusOne(1)
    assert -1.0 == PositiveOrMinusOne("-1.0")
    pytest.raises(ValueError, lambda: PositiveOrMinusOne(-0.5))
    pytest.raises(ValueError, lambda: PositiveOrMinusOne("-2"))


def test_non_negative_float_add_argument(parser):
    parser.add_argument("--le0", type=NonNegativeFloat)
    assert 0.0 == parser.parse_args(["--le0", "0"]).le0
    assert 5.6 == parser.parse_args(["--le0", "5.6"]).le0
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--le0", "-2.1"]))


def test_restricted_number_add_argument_optional(parser):
    limit_val = random.randint(100, 10000)
    larger_than = restricted_number_type(f"larger_than_{limit_val}", int, (">", limit_val))
    parser.add_argument("--val", type=larger_than, default=limit_val + 1, help="Help")

    assert limit_val + 1 == parser.parse_args([f"--val={limit_val+1}"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args([f"--val={limit_val-1}"]))

    help_str = get_parser_help(parser)
    assert f"Help (type: larger_than_{limit_val}, default: {limit_val+1})" in help_str


def test_restricted_number_add_argument_optional_nargs_plus(parser):
    TenToTwenty = restricted_number_type("TenToTwenty", int, [(">=", 10), ("<=", 20)])
    parser.add_argument("--f10t20", type=TenToTwenty, nargs="+")
    assert [11, 14, 16] == parser.parse_args(["--f10t20", "11", "14", "16"]).f10t20
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "9"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "21"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "10.5"]))


def test_restricted_number_positional_nargs_questionmark(parser):
    parser.add_argument("p1")
    parser.add_argument("p2", nargs="?", type=OpenUnitInterval)
    assert None is parser.parse_args(["a"]).p2
    assert 0.5 == parser.parse_args(["a", "0.5"]).p2
    pytest.raises(ArgumentError, lambda: parser.parse_args(["a", "b"]))


def test_restricted_number_optional_union(parser):
    parser.add_argument("--num", type=Optional[Union[PositiveInt, OpenUnitInterval]])
    assert 0.1 == parser.parse_args(["--num", "0.1"]).num
    assert 0.9 == parser.parse_args(["--num", "0.9"]).num
    assert 1 == parser.parse_args(["--num", "1"]).num
    assert 12 == parser.parse_args(["--num", "12"]).num
    assert None is parser.parse_args(["--num=null"]).num
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--num", "0.0"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--num", "4.5"]))


def test_restricted_number_dump(parser):
    parser.add_argument("--float", type=PositiveFloat)
    assert "float: 1.1\n" == parser.dump(parser.parse_args(["--float", "1.1"]))


def test_type_function_parse(parser):
    def gt0_or_off(x):
        return x if x == "off" else PositiveInt(x)

    parser.add_argument("--gt0_or_off", type=gt0_or_off)
    parser.add_argument("--multi_gt0_or_off", type=gt0_or_off, nargs="+")

    assert 1 == parser.parse_args(["--gt0_or_off", "1"]).gt0_or_off
    assert "off" == parser.parse_args(["--gt0_or_off", "off"]).gt0_or_off
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--gt0_or_off", "0"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--gt0_or_off", "on"]))

    assert [1, "off"] == parser.parse_args(["--multi_gt0_or_off", "1", "off"]).multi_gt0_or_off
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--multi_gt0_or_off", "1", "0"]))
    pytest.raises(ArgumentError, lambda: parser.parse_object({"multi_gt0_or_off": [1, 0]}))


def test_extend_base_type(parser):
    def is_even(t, v):
        if int(v) % 2 != 0:
            raise ValueError(f"{v} is not even")

    EvenInt = extend_base_type("EvenInt", int, is_even)
    parser.add_argument("--even_int", type=EvenInt)
    assert 2 == parser.parse_args(["--even_int=2"]).even_int
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--even_int=3"]))


# restricted string tests


def test_email():
    assert "name@eg.org" == Email("name@eg.org")
    pytest.raises(ValueError, lambda: Email(""))
    pytest.raises(ValueError, lambda: Email("name @ eg.org"))
    pytest.raises(ValueError, lambda: Email("name_at_eg.org"))


def test_optional_email_parse(parser):
    parser.add_argument("--email", type=Optional[Email])
    assert "a@b.c" == parser.parse_args(["--email", "a@b.c"]).email
    assert None is parser.parse_args(["--email=null"]).email
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--email", "abc"]))


def test_non_empty_string():
    assert " value " == NotEmptyStr(" value ")
    pytest.raises(ValueError, lambda: NotEmptyStr(""))
    pytest.raises(ValueError, lambda: NotEmptyStr(" "))


def test_restricted_string_already_registered():
    NewEmail = restricted_string_type("Email", r"^[^@ ]+@[^@ ]+\.[^@ ]+$")
    assert Email is NewEmail
    pytest.raises(
        ValueError,
        lambda: restricted_string_type("NewName", r"^[^@ ]+@[^@ ]+\.[^@ ]+$"),
    )


def test_restricted_string_parse(parser):
    FourDigits = restricted_string_type("FourDigits", "^[0-9]{4}$")
    parser.add_argument("--op", type=FourDigits)
    assert "1234" == parser.parse_args(["--op", "1234"]).op
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "123"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "12345"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "abcd"]))


def test_restricted_string_dump(parser):
    ThreeChars = restricted_string_type("ThreeChars", "^[A-Z]{3}$")
    parser.add_argument("--op", type=ThreeChars)
    assert "op: ABC\n" == parser.dump(parser.parse_args(["--op", "ABC"]))


# other types


@pytest.mark.parametrize(
    ["delta_in", "delta_out"],
    [
        ("1:2:3", "1:02:03"),
        ("0:05:30", "0:05:30"),
        ("3 days, 2:0:0", "3 days, 2:00:00"),
        ("345:0:0", "14 days, 9:00:00"),
    ],
)
def test_timedelta_deserializer(delta_in, delta_out):
    delta = get_registered_type(timedelta).deserializer(delta_in)
    assert isinstance(delta, timedelta)
    assert str(delta) == delta_out


@pytest.mark.parametrize("delta_in", ["not delta", 1234])
def test_timedelta_deserializer_failure(delta_in):
    with pytest.raises(ValueError):
        get_registered_type(timedelta).deserializer(delta_in)


bytes_parametrize = (
    ["serialized", "deserialized"],
    [
        ("", b""),
        ("iVBORw0KGgo=", b"\x89PNG\r\n\x1a\n"),
        ("AAAAB3NzaC1yc2E=", b"\x00\x00\x00\x07ssh-rsa"),
    ],
)


@pytest.mark.parametrize(*bytes_parametrize)
def test_bytes(serialized, deserialized):
    assert deserialized == get_registered_type(bytes).deserializer(serialized)
    assert serialized == get_registered_type(bytes).serializer(deserialized)


@pytest.mark.parametrize(*bytes_parametrize)
def test_bytearray(serialized, deserialized):
    deserialized = bytearray(deserialized)
    assert deserialized == get_registered_type(bytearray).deserializer(serialized)
    assert serialized == get_registered_type(bytearray).serializer(deserialized)


@pytest.mark.parametrize(
    ["serialized", "expected", "reserialized"],
    [
        ("range(5)", [0, 1, 2, 3, 4], "="),
        ("range(-2)", [], "="),
        ("range(2, 6)", [2, 3, 4, 5], "="),
        ("range(-6, -4)", [-6, -5], "="),
        ("range(1, 7, 2)", [1, 3, 5], "="),
        ("range(-1, -7, -2)", [-1, -3, -5], "="),
        ("range(0, 4)", [0, 1, 2, 3], "range(4)"),
        ("range(1, 3, 1)", [1, 2], "range(1, 3)"),
    ],
)
def test_range(serialized, expected, reserialized):
    deserialized = get_registered_type(range).deserializer(serialized)
    assert expected == list(deserialized)
    if reserialized == "=":
        reserialized = serialized
    assert reserialized == get_registered_type(range).serializer(deserialized)


def test_range_deserialize_failure():
    with pytest.raises(ValueError) as ex:
        get_registered_type(range).deserializer("not a range")
    ex.match("Expected 'range")


# other tests


@pytest.mark.parametrize("type_class", registered_types.values())
def test_pickle_module_type(type_class):
    assert type_class == pickle.loads(pickle.dumps(type_class))


def test_module_name_clash():
    pytest.raises(ValueError, lambda: restricted_string_type("List", "^clash$"))


def test_register_non_bool_cast_type(parser):
    class Elems:
        def __init__(self, *elems):
            self.elems = list(elems)

        def __bool__(self):
            raise TypeError("bool not supported")

    pytest.raises(TypeError, lambda: not Elems(1, 2))
    register_type(Elems, lambda x: x.elems, lambda x: Elems(*x))

    parser.add_argument("--elems", type=Elems)
    cfg = parser.parse_args(["--elems=[1, 2, 3]"])
    assert isinstance(cfg.elems, Elems)
    assert [1, 2, 3] == cfg.elems.elems
    assert '{"elems":[1,2,3]}' == parser.dump(cfg, format="json")


def test_register_type_datetime(parser):
    def serializer(v):
        return v.isoformat()

    def deserializer(v):
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")

    register_type(datetime, serializer, deserializer)

    parser.add_argument("--datetime", type=datetime)
    cfg = parser.parse_args(["--datetime=2008-09-03T20:56:35"])
    assert cfg.datetime == datetime(2008, 9, 3, 20, 56, 35)
    assert parser.dump(cfg) == "datetime: '2008-09-03T20:56:35'\n"

    register_type(datetime, serializer, deserializer)  # identical re-registering is okay
    pytest.raises(ValueError, lambda: register_type(datetime))  # different registration not okay


class RegisterOnFirstUse:
    pass


def test_register_type_on_first_use():
    register_type_on_first_use(f"{__name__}.RegisterOnFirstUse")
    assert f"{__name__}.RegisterOnFirstUse" in registration_pending
    registered = get_registered_type(RegisterOnFirstUse)
    assert registered.type_class is RegisterOnFirstUse
    assert f"{__name__}.RegisterOnFirstUse" not in registration_pending


def test_decimal(parser):
    parser.add_argument("--decimal", type=Decimal)
    cfg = parser.parse_args(["--decimal=0.1"])
    assert isinstance(cfg.decimal, Decimal)
    assert cfg.decimal == Decimal("0.1")
    assert parser.dump(cfg) == "decimal: 0.1\n"


def test_uuid(parser):
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    parser.add_argument("--uuid", type=uuid.UUID)
    parser.add_argument("--uuids", type=List[uuid.UUID])
    cfg = parser.parse_args([f"--uuid={id1}", f'--uuids=["{id1}", "{id2}"]'])
    assert cfg.uuid == id1
    assert cfg.uuids == [id1, id2]
    assert f"uuid: {id1}\nuuids:\n- {id1}\n- {id2}\n" == parser.dump(cfg)
