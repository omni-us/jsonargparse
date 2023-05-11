import os
import pathlib
import pickle
from datetime import timedelta
from typing import Iterator

import pytest

from jsonargparse import ArgumentError
from jsonargparse.typing import (
    ClosedUnitInterval,
    Email,
    NonNegativeFloat,
    NonNegativeInt,
    OpenUnitInterval,
    Path_fc,
    Path_fr,
    PositiveFloat,
    PositiveInt,
    get_registered_type,
    path_type,
    register_type,
    registered_types,
    restricted_number_type,
    restricted_string_type,
)

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
    pytest.raises(
        ValueError,
        lambda: restricted_number_type("NewName", float, [("<=", 1), (">=", 0)]),
    )


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


def test_restricted_number_add_argument(parser):
    TenToTwenty = restricted_number_type("TenToTwenty", int, [(">=", 10), ("<=", 20)])
    parser.add_argument("--f10t20", type=TenToTwenty, nargs="+")
    assert [11, 14, 16] == parser.parse_args(["--f10t20", "11", "14", "16"]).f10t20
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "9"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "21"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--f10t20", "10.5"]))


def test_add_argument_type_function(parser):
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


# restricted string tests


def test_email():
    assert "name@eg.org" == Email("name@eg.org")
    pytest.raises(ValueError, lambda: Email(""))
    pytest.raises(ValueError, lambda: Email("name @ eg.org"))
    pytest.raises(ValueError, lambda: Email("name_at_eg.org"))


def test_restricted_string_already_registered():
    NewEmail = restricted_string_type("Email", r"^[^@ ]+@[^@ ]+\.[^@ ]+$")
    assert Email is NewEmail
    pytest.raises(
        ValueError,
        lambda: restricted_string_type("NewName", r"^[^@ ]+@[^@ ]+\.[^@ ]+$"),
    )


def test_restricted_string_add_argument(parser):
    FourDigits = restricted_string_type("FourDigits", "^[0-9]{4}$")
    parser.add_argument("--op", type=FourDigits)
    assert "1234" == parser.parse_args(["--op", "1234"]).op
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "123"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "12345"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--op", "abcd"]))


# path tests


@pytest.fixture
def file_fr(tmp_cwd) -> Iterator[str]:
    file_fr = "file_r"
    pathlib.Path(file_fr).touch()
    yield file_fr


def test_path_fr(file_fr):
    path = Path_fr(file_fr)
    assert path == file_fr
    assert path() == os.path.realpath(file_fr)
    pytest.raises(TypeError, lambda: Path_fr("does_not_exist"))


def test_path_fc_with_kwargs(tmpdir):
    path = Path_fc("some-file.txt", cwd=str(tmpdir))
    assert path() == os.path.join(tmpdir, "some-file.txt")


def test_path_fr_already_registered():
    assert Path_fr is path_type("fr")


def test_os_pathlike(parser, file_fr):
    parser.add_argument("--path", type=os.PathLike)
    assert file_fr == parser.parse_args([f"--path={file_fr}"]).path


def test_pathlib_path(parser, file_fr):
    parser.add_argument("--path", type=pathlib.Path)
    cfg = parser.parse_args([f"--path={file_fr}"])
    assert isinstance(cfg.path, pathlib.Path)
    assert str(cfg.path) == file_fr
    assert parser.dump(cfg) == "path: file_r\n"


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
