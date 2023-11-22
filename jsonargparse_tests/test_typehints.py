from __future__ import annotations

import random
import sys
import time
import uuid
from calendar import Calendar
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from unittest import mock
from warnings import catch_warnings

import pytest
import yaml

from jsonargparse import ActionConfigFile, ArgumentError, Namespace, lazy_instance
from jsonargparse._typehints import (
    ActionTypeHint,
    Literal,
    get_all_subclass_paths,
    is_optional,
)
from jsonargparse._util import get_import_path
from jsonargparse.typing import (
    NotEmptyStr,
    Path_fc,
    Path_fr,
    PositiveFloat,
    PositiveInt,
)
from jsonargparse_tests.conftest import (
    capture_logs,
    get_parse_args_stdout,
    get_parser_help,
)


def test_add_argument_failure_given_type_and_action(parser):
    with pytest.raises(ValueError) as ctx:
        parser.add_argument("--op1", type=Optional[bool], action=True)
    ctx.match("Providing both type and action not allowed")


# basic types tests


def test_str_no_strip(parser):
    parser.add_argument("--op", type=Optional[str])
    parser.add_argument("--cfg", action=ActionConfigFile)
    assert "  " == parser.parse_args(["--op", "  "]).op
    assert "" == parser.parse_args(["--op", ""]).op
    assert " abc " == parser.parse_args(["--op= abc "]).op
    assert "xyz: " == parser.parse_args(["--op=xyz: "]).op
    assert " " == parser.parse_args(['--cfg={"op":" "}']).op
    assert None is parser.parse_args(["--op=null"]).op


@pytest.mark.parametrize("value", ["2022-04-12", "2022-04-32"])
def test_str_not_timestamp(parser, value):
    parser.add_argument("foo", type=str)
    assert value == parser.parse_args([value]).foo


@pytest.mark.parametrize("value", ["1", "02", "3.40", "5.7e-8"])
def test_str_number_value(parser, value):
    parser.add_argument("--val", type=str)
    assert value == parser.parse_args([f"--val={value}"]).val


def test_str_yaml_constructor_error(parser):
    parser.add_argument("--val", type=str)
    assert "{{something}}" == parser.parse_args(["--val={{something}}"]).val


def test_bool_parse(parser):
    parser.add_argument("--val", type=bool)
    assert None is parser.get_defaults().val
    assert True is parser.parse_args(["--val", "true"]).val
    assert True is parser.parse_args(["--val", "TRUE"]).val
    assert False is parser.parse_args(["--val", "false"]).val
    assert False is parser.parse_args(["--val", "FALSE"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--val", "1"]))


@pytest.mark.parametrize("num_type", [int, float, PositiveInt, PositiveFloat])
def test_bool_not_a_number(parser, num_type):
    parser.add_argument("--num", type=num_type)
    for value in [True, False]:
        with pytest.raises(ArgumentError):
            parser.parse_object({"num": value})


def test_float_scientific_notation(parser):
    parser.add_argument("--num", type=float)
    assert 1e-3 == parser.parse_args(["--num=1e-3"]).num


def test_complex_number(parser):
    parser.add_argument("--complex", type=complex)
    cfg = parser.parse_args(["--complex=(2+3j)"])
    assert cfg.complex == 2 + 3j
    assert parser.dump(cfg) == "complex: (2+3j)\n"


@pytest.mark.skipif(not Literal, reason="Literal introduced in python 3.8 or backported in typing_extensions")
def test_literal(parser):
    parser.add_argument("--str", type=Literal["a", "b", None])
    parser.add_argument("--int", type=Literal[3, 4])
    parser.add_argument("--true", type=Literal[True])
    parser.add_argument("--false", type=Literal[False])
    assert "a" == parser.parse_args(["--str=a"]).str
    assert "b" == parser.parse_args(["--str=b"]).str
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--str=x"]))
    assert None is parser.parse_args(["--str=null"]).str
    assert 4 == parser.parse_args(["--int=4"]).int
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--int=5"]))
    assert True is parser.parse_args(["--true=true"]).true
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--true=false"]))
    assert False is parser.parse_args(["--false=false"]).false
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--false=true"]))
    help_str = get_parser_help(parser)
    for value in ["--str {a,b,null}", "--int {3,4}", "--true True", "--false False"]:
        assert value in help_str


@pytest.mark.skipif(not Literal, reason="Literal introduced in python 3.8 or backported in typing_extensions")
def test_union_of_literals(parser):
    literal_type = __import__("typing").Literal if sys.version_info[:2] == (3, 9) else Literal
    parser.add_argument("--literal", type=Union[literal_type[1, 2], literal_type["a", "b"]])  # noqa: F821
    assert "a" == parser.parse_args(["--literal=a"]).literal
    assert 2 == parser.parse_args(["--literal=2"]).literal


def test_type_any(parser):
    parser.add_argument("--any", type=Any)
    assert "abc" == parser.parse_args(["--any=abc"]).any
    assert 123 == parser.parse_args(["--any=123"]).any
    assert 5.6 == parser.parse_args(["--any=5.6"]).any
    assert [7, 8] == parser.parse_args(["--any=[7, 8]"]).any
    assert {"a": 0, "b": 1} == parser.parse_args(['--any={"a":0, "b":1}']).any
    assert True is parser.parse_args(["--any=True"]).any
    assert False is parser.parse_args(["--any=False"]).any
    assert None is parser.parse_args(["--any=null"]).any
    assert " " == parser.parse_args(["--any= "]).any
    assert " xyz " == parser.parse_args(["--any= xyz "]).any
    assert "[[[" == parser.parse_args(["--any=[[["]).any


def test_type_any_dump(parser):
    parser.add_argument("--any", type=Any, default=EnumABC.B)
    cfg = parser.parse_args([])
    assert "any: B\n" == parser.dump(cfg)


def test_type_typehint_without_arg(parser):
    type_class = Type if sys.version_info < (3, 9) else type
    parser.add_argument("--type", type=type_class)
    cfg = parser.parse_args(["--type=uuid.UUID"])
    assert cfg.type is uuid.UUID
    assert parser.dump(cfg) == "type: uuid.UUID\n"


def test_type_typehint_with_arg(parser):
    type_class = Type if sys.version_info < (3, 9) else type
    parser.add_argument("--cal", type=type_class[Calendar])
    cfg = parser.parse_args(["--cal=calendar.Calendar"])
    assert cfg.cal is Calendar
    assert parser.dump(cfg) == "cal: calendar.Calendar\n"
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cal=uuid.UUID"]))


def test_type_typehint_help_known_subclasses(parser):
    parser.add_argument("--cal", type=Type[Calendar])
    help_str = get_parser_help(parser)
    assert "known subclasses: calendar.Calendar," in help_str


# enum tests


class EnumABC(Enum):
    A = 1
    B = 2
    C = 3


def test_enum_parse(parser):
    parser.add_argument("--enum", type=EnumABC)
    for val in ["A", "B", "C"]:
        assert EnumABC[val] == parser.parse_args([f"--enum={val}"]).enum
    for val in ["X", "b", 2]:
        pytest.raises(ArgumentError, lambda: parser.parse_args([f"--enum={val}"]))


def test_enum_dump(parser):
    parser.add_argument("--enum", type=EnumABC)
    cfg = parser.parse_args(["--enum=C"])
    assert "enum: C\n" == parser.dump(cfg)
    with pytest.raises(TypeError):
        parser.dump(Namespace(enum="x"))


def test_enum_help(parser):
    parser.add_argument("--enum", type=EnumABC, default=EnumABC.B, help="Help")
    assert EnumABC.B == parser.get_defaults().enum
    help_str = get_parser_help(parser)
    assert "--enum {A,B,C}" in help_str
    assert "Help (type: EnumABC, default: B)" in help_str


def test_enum_optional(parser):
    parser.add_argument("--enum", type=Optional[EnumABC])
    assert EnumABC.B == parser.parse_args(["--enum=B"]).enum
    assert None is parser.parse_args(["--enum=null"]).enum
    help_str = get_parser_help(parser)
    assert "--enum {A,B,C,null}" in help_str


class EnumStr(str, Enum):
    A = "A"
    B = "B"


def test_enum_str_optional(parser):
    parser.add_argument("--enum", type=Optional[EnumStr])
    assert "B" == parser.parse_args(["--enum=B"]).enum
    assert None is parser.parse_args(["--enum=null"]).enum


# set tests


def test_set(parser):
    parser.add_argument("--set", type=Set[int])
    assert {1, 2} == parser.parse_args(["--set=[1, 2]"]).set
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(['--set=["a", "b"]'])
    ctx.match("Expected a <class 'int'>")


# tuple tests


def test_tuple_without_arg(parser):
    parser.add_argument("--tuple", type=tuple)
    cfg = parser.parse_args(['--tuple=[1, "a", True]'])
    assert (1, "a", True) == cfg.tuple
    help_str = get_parser_help(parser)
    assert "--tuple [ITEM,...]  (type: tuple, default: null)" in help_str


def test_tuples_nested(parser):
    parser.add_argument("--tuple", type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], Tuple[int, float]]])
    cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
    assert (("foo", "bar"), ((1, 2.02), (3, 3.09))) == cfg.tuple


def test_tuple_ellipsis(parser):
    parser.add_argument("--tuple", type=Tuple[float, ...])
    assert (1.2,) == parser.parse_args(["--tuple=[1.2]"]).tuple
    assert (1.2, 3.4) == parser.parse_args(["--tuple=[1.2, 3.4]"]).tuple
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--tuple=[]"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--tuple=[2, "a"]']))


def test_tuples_nested_ellipsis(parser):
    parser.add_argument("--tuple", type=Tuple[Tuple[str, str], Tuple[Tuple[int, float], ...]])
    cfg = parser.parse_args(['--tuple=[["foo", "bar"], [[1, 2.02], [3, 3.09]]]'])
    assert (("foo", "bar"), ((1, 2.02), (3, 3.09))) == cfg.tuple


def test_tuple_union(parser, tmp_cwd):
    parser.add_argument("--tuple", type=Tuple[Union[int, EnumABC], Path_fc, NotEmptyStr])
    cfg = parser.parse_args(['--tuple=[2, "a", "b"]'])
    assert (2, "a", "b") == cfg.tuple
    assert isinstance(cfg.tuple[1], Path_fc)
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--tuple=[]"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--tuple=[2, "a", "b", 5]']))
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--tuple=[2, "a"]']))
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--tuple={"a":1, "b":"2"}']))
    help_str = get_parser_help(parser)
    assert "--tuple [ITEM,...]  (type: Tuple[Union[int, EnumABC], Path_fc, NotEmptyStr], default: null)" in help_str


# list tests


@pytest.mark.parametrize("list_type", [Iterable, List, Sequence], ids=str)
def test_list_variants(parser, list_type):
    parser.add_argument("--list", type=list_type[int])
    cfg = parser.parse_args(["--list=[1, 2]"])
    assert [1, 2] == cfg.list


def test_list_dump(parser):
    parser.add_argument("--list", type=Union[PositiveInt, List[PositiveInt]])
    dump = yaml.safe_load(parser.dump(Namespace(list=[1, 2])))
    assert [1, 2] == dump["list"]
    with pytest.raises(TypeError):
        parser.dump(Namespace(list=[1, -2]))


def test_list_enum(parser):
    parser.add_argument("--list", type=List[EnumABC])
    assert [EnumABC.B, EnumABC.A] == parser.parse_args(['--list=["B", "A"]']).list


def test_list_tuple(parser):
    parser.add_argument("--list", type=List[Tuple[int, float]])
    cfg = parser.parse_args(["--list=[[1, 2.02], [3, 3.09]]"])
    assert [(1, 2.02), (3, 3.09)] == cfg.list


def test_list_union(parser):
    parser.add_argument("--list1", type=List[Union[float, str, type(None)]])
    parser.add_argument("--list2", type=List[Union[int, EnumABC]])
    assert [1.2, "B"] == parser.parse_args(['--list1=[1.2, "B"]']).list1
    assert [3, EnumABC.B] == parser.parse_args(['--list2=[3, "B"]']).list2
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--list1={"a":1, "b":"2"}']))


def test_list_str_positional(parser):
    parser.add_argument("list", type=List[str])
    cfg = parser.parse_args(['["a", "b"]'])
    assert cfg.list == ["a", "b"]


def test_sequence_default_tuple(parser):
    parser.add_argument("--seq", type=Sequence[str], default=("one", "two"))
    cfg = parser.parse_args([])
    assert cfg == parser.get_defaults()


# list append tests


def test_list_append(parser):
    parser.add_argument("--val", type=Union[int, float, List[int]])
    assert 0 == parser.parse_args(["--val=0"]).val
    assert [0] == parser.parse_args(["--val+=0"]).val
    assert [1, 2, 3] == parser.parse_args(["--val=1", "--val+=2", "--val+=3"]).val
    assert [1, 2, 3] == parser.parse_args(["--val=[1,2]", "--val+=3"]).val
    assert [1] == parser.parse_args(["--val=0.1", "--val+=1"]).val
    assert 3 == parser.parse_args(["--val=[1,2]", "--val=3"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--val=a", "--val+=1"]))


def test_list_append_default_empty(parser):
    parser.add_argument("--list", type=List[str], default=[])
    assert [] == parser.get_defaults().list
    assert ["a"] == parser.parse_args(["--list=[a]"]).list
    assert [] == parser.get_defaults().list
    assert ["b", "c"] == parser.parse_args(["--list+=[b, c]"]).list
    assert [] == parser.get_defaults().list


def test_list_append_config(parser):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--val", type=List[int], default=[1, 2])
    assert [3, 4] == parser.parse_args(["--cfg", "val: [3, 4]"]).val
    assert [1, 2, 3] == parser.parse_args(["--cfg", "val+: 3"]).val
    assert [1, 2, 3, 4] == parser.parse_args(["--cfg", "val+: [3, 4]"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg", "val+: a"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--val=2", "--cfg", "val+: 3"]))


def test_list_append_default_config_files(parser, tmp_cwd, subtests):
    config_path = tmp_cwd / "config.yaml"
    parser.default_config_files = [config_path]
    parser.add_argument("--nums", type=List[int], default=[0])

    with subtests.test("replace"):
        config_path.write_text("nums: [1]\n")
        cfg = parser.parse_args(["--nums+=2"])
        assert cfg.nums == [1, 2]
        cfg = parser.parse_args(["--nums+=[2, 3]"])
        assert cfg.nums == [1, 2, 3]

    with subtests.test("append"):
        config_path.write_text("nums+: [1]\n")
        cfg = parser.get_defaults()
        assert cfg.nums == [0, 1]
        cfg = parser.parse_args(["--nums+=2"])
        assert cfg.nums == [0, 1, 2]
        cfg = parser.parse_args(["--nums+=[2, 3]"])
        assert cfg.nums == [0, 1, 2, 3]
        assert str(cfg.__default_config__) == str(config_path)

    with subtests.test("append in second default config"):
        config_path2 = tmp_cwd / "config2.yaml"
        config_path2.write_text("nums+: [2]\n")
        parser.default_config_files += [str(config_path2)]
        cfg = parser.get_defaults()
        assert cfg.nums == [0, 1, 2]
        assert [str(c) for c in cfg.__default_config__] == parser.default_config_files


def test_list_append_subcommand_global_default_config_files(parser, subparser, tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    parser.default_config_files = [config_path]
    subcommands = parser.add_subcommands()
    subparser.add_argument("--nums", type=List[int], default=[0])
    subcommands.add_subcommand("sub", subparser)
    config_path.write_text("sub:\n  nums: [1]\n")

    cfg = parser.parse_args(["sub", "--nums+=2"])
    assert cfg.sub.nums == [1, 2]
    assert str(cfg.__default_config__) == str(config_path)
    cfg = parser.parse_args(["sub", "--nums+=2"], defaults=False)
    assert cfg.sub.nums == [2]


def test_list_append_subcommand_subparser_default_config_files(parser, subparser, tmp_cwd):
    config_path = tmp_cwd / "config.yaml"
    subcommands = parser.add_subcommands()
    subparser.default_config_files = [config_path]
    subparser.add_argument("--nums", type=List[int], default=[0])
    subcommands.add_subcommand("sub", subparser)
    config_path.write_text("nums: [1]\n")

    cfg = parser.parse_args(["sub", "--nums+=2"])
    assert cfg.sub.nums == [1, 2]
    assert str(cfg.sub.__default_config__) == str(config_path)
    cfg = parser.parse_args(["sub", "--nums+=2"], defaults=False)
    assert cfg.sub.nums == [2]


# dict tests


def test_dict_without_arg(parser):
    parser.add_argument("--dict", type=dict)
    assert {} == parser.parse_args(["--dict={}"])["dict"]
    assert {"a": 1, "b": "2"} == parser.parse_args(['--dict={"a":1, "b":"2"}'])["dict"]
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--dict=1"]))


def test_dict_int_keys(parser):
    parser.add_argument("--d", type=Dict[int, str])
    parser.add_argument("--cfg", action=ActionConfigFile)
    cfg = {"d": {1: "val1", 2: "val2"}}
    assert cfg["d"] == parser.parse_args(["--cfg", str(cfg)]).d
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--cfg={"d": {"a": "b"}}']))


def test_dict_union(parser, tmp_cwd):
    parser.add_argument("--dict1", type=Dict[int, Optional[Union[float, EnumABC]]])
    parser.add_argument("--dict2", type=Dict[str, Union[bool, Path_fc]])
    cfg = parser.parse_args(['--dict1={"2":4.5, "6":"C"}', '--dict2={"a":true, "b":"f"}'])
    assert {2: 4.5, 6: EnumABC.C} == cfg.dict1
    assert {"a": True, "b": "f"} == cfg.dict2
    assert isinstance(cfg.dict2["b"], Path_fc)
    assert {5: None} == parser.parse_args(['--dict1={"5":null}']).dict1
    pytest.raises(ArgumentError, lambda: parser.parse_args(['--dict1=["a", "b"]']))
    cfg = yaml.safe_load(parser.dump(cfg))
    assert {"dict1": {"2": 4.5, "6": "C"}, "dict2": {"a": True, "b": "f"}} == cfg


def test_dict_union_int_keys(parser):
    parser.add_argument("--dict", type=Union[int, Dict[int, int]], default=1)
    assert 1 == parser.get_defaults().dict
    assert {2: 7, 4: 9} == parser.parse_args(['--dict={"2": 7, "4": 9}']).dict


def test_dict_command_line_set_items(parser):
    parser.add_argument("--dict", type=Dict[str, int])
    cfg = parser.parse_args(["--dict.one=1", "--dict.two=2"])
    assert cfg.dict == {"one": 1, "two": 2}


def test_mapping_nested_without_args(parser):
    parser.add_argument("--map", type=Mapping[str, Union[int, Mapping]])
    assert {"a": 1} == parser.parse_args(['--map={"a": 1}']).map
    assert {"b": {"c": 2}} == parser.parse_args(['--map={"b": {"c": 2}}']).map


# union tests


@pytest.mark.parametrize(
    ["subtypes", "arg", "expected"],
    [
        ((bool, str), "=true", True),
        ((str, bool), "=true", "true"),
        ((int, str), "=1", 1),
        ((str, int), "=2", "2"),
        ((float, int), "=3", 3.0),
        ((int, float), "=4", 4),
        ((int, List[int]), "=5", 5),
        ((List[int], int), "=6", 6),
        ((int, List[int]), "+=7", [7]),
        ((List[int], int), "+=8", [8]),
    ],
    ids=str,
)
def test_union_subtypes_order(parser, subtypes, arg, expected):
    parser.add_argument("--val", type=Union[subtypes])
    val = parser.parse_args([f"--val{arg}"]).val
    assert isinstance(val, type(expected))
    assert val == expected


def test_union_unsupported_subtype(parser, logger):
    parser.logger = logger
    with capture_logs(logger) as logs:
        parser.add_argument("--union", type=Union[int, str, "unsupported"])  # noqa: F821
    assert "Discarding unsupported subtypes" in logs.getvalue()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="new union syntax introduced in python 3.10")
def test_union_new_syntax_simple_types(parser):
    parser.add_argument("--val", type=eval("int | None"))
    assert 123 == parser.parse_args(["--val=123"]).val
    assert None is parser.parse_args(["--val=null"]).val
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--val=abc"]))


@pytest.mark.skipif(sys.version_info < (3, 10), reason="new union syntax introduced in python 3.10")
def test_union_new_syntax_subclass_type(parser):
    parser.add_argument("--op", type=eval("Calendar | bool"))
    help_str = get_parse_args_stdout(parser, ["--op.help=calendar.TextCalendar"])
    assert "--op.init_args.firstweekday" in help_str


# callable tests


def test_callable_function_path(parser):
    parser.add_argument("--callable", type=Callable, default=time.time)

    cfg = parser.get_defaults()
    assert cfg.callable is time.time
    assert parser.dump(cfg) == "callable: time.time\n"

    cfg = parser.parse_args(["--callable=random.randint"])
    assert cfg.callable is random.randint
    assert parser.dump(cfg) == "callable: random.randint\n"

    help_str = get_parser_help(parser)
    assert "(type: Callable, default: time.time)" in help_str

    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["--callable=jsonargparse.not_exist"])
    ctx.match("Callable expects a function or a callable class")


def test_callable_list_of_function_paths(parser):
    parser.add_argument("--callables", type=List[Callable])

    cfg = parser.parse_args(["--callables=[random.randint, time.time]"])
    assert [random.randint, time.time] == cfg.callables

    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["--callables=[jsonargparse.not_exist]"])
    ctx.match("Callable expects a function or a callable class")


class CallableClassPath:
    def __init__(self, p1: int = 1):
        self.p1 = p1

    def __call__(self):
        return self.p1


def test_callable_class_path_simple(parser):
    parser.add_argument("--callable", type=Callable)

    value = {"class_path": f"{__name__}.CallableClassPath", "init_args": {"p1": 2}}
    cfg = parser.parse_args([f"--callable={value}"])
    assert value == cfg.callable.as_dict()
    assert value == yaml.safe_load(parser.dump(cfg))["callable"]
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.callable, CallableClassPath)
    assert 2 == init.callable()

    pytest.raises(ArgumentError, lambda: parser.parse_args(["--callable={}"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--callable=jsonargparse.SUPPRESS"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--callable=calendar.Calendar"]))
    value = {"class_path": f"{__name__}.CallableClassPath", "key": "val"}
    pytest.raises(ArgumentError, lambda: parser.parse_args([f"--callable={value}"]))


class CallableParent(CallableClassPath):
    pass


def test_callable_class_path_parent(parser):
    parser.add_argument("--callable", type=Callable)
    value = {"class_path": f"{__name__}.CallableParent", "init_args": {"p1": 1}}
    cfg = parser.parse_args([f"--callable={__name__}.CallableParent"])
    assert value == cfg.callable.as_dict()


class CallableGiveName:
    def __init__(self, name: str):
        self.name = name

    def __call__(self):
        return self.name


@pytest.mark.parametrize("callable_type", [Callable, Optional[Callable], Union[int, Callable]])
def test_callable_class_path_short_init_args(parser, callable_type):
    parser.add_argument("--call", type=callable_type)
    cfg = parser.parse_args([f"--call={__name__}.CallableGiveName", "--call.name=Bob"])
    assert cfg.call.class_path == f"{__name__}.CallableGiveName"
    assert cfg.call.init_args == Namespace(name="Bob")
    init = parser.instantiate_classes(cfg)
    assert init.call() == "Bob"


def int_to_str(p: int) -> str:
    return str(p)


def str_to_int(p: str) -> int:
    return int(p)


def test_callable_args_function_path(parser):
    parser.add_argument("--callable", type=Callable[[int], str])
    cfg = parser.parse_args([f"--callable={__name__}.int_to_str"])
    assert int_to_str is cfg.callable
    cfg = parser.parse_args([f"--callable={__name__}.str_to_int"])
    assert str_to_int is cfg.callable  # Currently callable args are ignored


class Optimizer:
    def __init__(self, params: List[float], lr: float = 1e-3, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Adam(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_callable_args_return_type_class(parser, subtests):
    parser.add_argument("--optimizer", type=Callable[[List[float]], Optimizer], default=SGD)

    with subtests.test("default"):
        cfg = parser.get_defaults()
        init = parser.instantiate_classes(cfg)
        optimizer = init.optimizer([0.1, 2, 3])
        assert isinstance(optimizer, SGD)
        assert [0.1, 2, 3] == optimizer.params
        assert 1e-3 == optimizer.lr
        assert 0.0 == optimizer.momentum

    with subtests.test("parse dict"):
        value = {
            "class_path": "Adam",
            "init_args": {
                "lr": 0.01,
            },
        }
        cfg = parser.parse_args([f"--optimizer={value}"])
        assert f"{__name__}.Adam" == cfg.optimizer.class_path
        assert Namespace(lr=0.01, momentum=0.0) == cfg.optimizer.init_args
        init = parser.instantiate_classes(cfg)
        optimizer = init.optimizer([4.5, 6.7])
        assert isinstance(optimizer, Adam)
        assert [4.5, 6.7] == optimizer.params
        assert 0.01 == optimizer.lr
        assert 0.0 == optimizer.momentum
        dump = parser.dump(cfg)
        assert yaml.safe_load(dump) == cfg.as_dict()

    with subtests.test("short notation"):
        assert cfg == parser.parse_args(["--optimizer=Adam", "--optimizer.lr=0.01"])

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        for name in ["Optimizer", "SGD", "Adam"]:
            assert f"{__name__}.{name}" in help_str


def test_callable_multiple_args_return_type_class(parser, subtests):
    parser.add_argument("--optimizer", type=Callable[[List[float], float], Optimizer], default=SGD)

    with subtests.test("default"):
        cfg = parser.get_defaults()
        init = parser.instantiate_classes(cfg)
        optimizer = init.optimizer([0.1, 2, 3], 1e-3)
        assert isinstance(optimizer, SGD)
        assert [0.1, 2, 3] == optimizer.params
        assert 1e-3 == optimizer.lr
        assert 0.0 == optimizer.momentum

    with subtests.test("parse dict"):
        value = {
            "class_path": "Adam",
            "init_args": {"momentum": 0.9},
        }
        cfg = parser.parse_args([f"--optimizer={value}"])
        assert f"{__name__}.Adam" == cfg.optimizer.class_path
        assert Namespace(momentum=0.9) == cfg.optimizer.init_args
        init = parser.instantiate_classes(cfg)
        optimizer = init.optimizer([4.5, 6.7], 0.01)
        assert isinstance(optimizer, Adam)
        assert [4.5, 6.7] == optimizer.params
        assert 0.01 == optimizer.lr
        assert 0.9 == optimizer.momentum
        dump = parser.dump(cfg)
        assert yaml.safe_load(dump) == cfg.as_dict()

    with subtests.test("short notation"):
        assert cfg == parser.parse_args(["--optimizer=Adam", "--optimizer.momentum=0.9"])

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        for name in ["Optimizer", "SGD", "Adam"]:
            assert f"{__name__}.{name}" in help_str


class StepLR:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch


class ReduceLROnPlateau:
    def __init__(self, optimizer: Optimizer, monitor: str):
        self.optimizer = optimizer
        self.monitor = monitor


def test_callable_args_return_type_union_of_classes(parser, subtests):
    parser.add_argument(
        "--scheduler",
        type=Callable[[Optimizer], Union[StepLR, ReduceLROnPlateau]],
        default=StepLR,
    )
    optimizer = Optimizer([])

    with subtests.test("default"):
        cfg = parser.get_defaults()
        init = parser.instantiate_classes(cfg)
        scheduler = init.scheduler(optimizer)
        assert isinstance(scheduler, StepLR)
        assert scheduler.optimizer is optimizer
        assert -1 == scheduler.last_epoch

    with subtests.test("parse"):
        value = {
            "class_path": "ReduceLROnPlateau",
            "init_args": {
                "monitor": "loss",
            },
        }
        cfg = parser.parse_args([f"--scheduler={value}"])
        assert f"{__name__}.ReduceLROnPlateau" == cfg.scheduler.class_path
        assert Namespace(monitor="loss") == cfg.scheduler.init_args
        init = parser.instantiate_classes(cfg)
        scheduler = init.scheduler(optimizer)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.optimizer is optimizer
        assert "loss" == scheduler.monitor

    with subtests.test("help"):
        help_str = get_parser_help(parser)
        for name in ["StepLR", "ReduceLROnPlateau"]:
            assert f"{__name__}.{name}" in help_str


class CallableSubconfig:
    def __init__(self, o: Callable[[int], Optimizer]):
        self.o = o


def test_callable_args_return_type_class_subconfig(parser, tmp_cwd):
    config = {
        "class_path": "Adam",
        "init_args": {"momentum": 0.8},
    }
    Path("optimizer.yaml").write_text(yaml.safe_dump(config))

    parser.add_class_arguments(CallableSubconfig, "m", sub_configs=True)
    cfg = parser.parse_args(["--m.o=optimizer.yaml"])
    assert cfg.m.o.class_path == f"{__name__}.Adam"
    init = parser.instantiate_classes(cfg)
    optimizer = init.m.o(1)
    assert isinstance(optimizer, Adam)
    assert optimizer.momentum == 0.8


# lazy_instance tests


def test_lazy_instance_init_postponed():
    class SubCalendar(Calendar):
        init_called = False
        getfirst = Calendar.getfirstweekday

        def __init__(self, *args, **kwargs):
            self.init_called = True
            super().__init__(*args, **kwargs)

    lazy_calendar = lazy_instance(SubCalendar, firstweekday=3)
    assert isinstance(lazy_calendar, SubCalendar)
    assert lazy_calendar.init_called is False
    assert lazy_calendar.getfirstweekday() == 3
    assert lazy_calendar.init_called is True


class IntParam:
    def __init__(self, param: int = 1):
        pass


def test_lazy_instance_invalid_init_value():
    with pytest.raises(ValueError) as ctx:
        lazy_instance(IntParam, param="not an int")
    ctx.match("Expected a <class 'int'>")


# other tests


@pytest.mark.parametrize(
    "typehint",
    [
        lambda: None,
        "unsupported",
        Optional["unsupported"],  # noqa: F821
        Tuple[int, "unsupported"],  # noqa: F821
        Union["unsupported1", "unsupported2"],  # noqa: F821
    ],
    ids=str,
)
def test_action_typehint_unsupported_type(typehint):
    with pytest.raises(ValueError) as ctx:
        ActionTypeHint(typehint=typehint)
    ctx.match("Unsupported type hint")


def test_action_typehint_none_type_error():
    with pytest.raises(ValueError) as ctx:
        ActionTypeHint(typehint=None)
    ctx.match("Expected typehint keyword argument")


@pytest.mark.parametrize(
    ["typehint", "ref_type", "expected"],
    [
        (Optional[bool], bool, True),
        (Union[type(None), bool], bool, True),
        (Dict[bool, type(None)], bool, False),  # type: ignore
        (Optional[Path_fr], Path_fr, True),
        (Union[type(None), Path_fr], Path_fr, True),
        (Dict[Path_fr, type(None)], Path_fr, False),  # type: ignore
        (Optional[EnumABC], Enum, True),
        (Union[type(None), EnumABC], Enum, True),
        (Dict[EnumABC, type(None)], Enum, False),  # type: ignore
    ],
    ids=str,
)
def test_is_optional(typehint, ref_type, expected):
    assert expected == is_optional(typehint, ref_type)


class SkipDefault(Calendar):
    def __init__(self, *args, param: str = "0", **kwargs):
        super().__init__(*args, **kwargs)


def test_dump_skip_default(parser):
    parser.add_argument("--g1.op1", default=1)
    parser.add_argument("--g1.op2", default="abc")
    parser.add_argument("--g2.op1", type=Callable, default=uuid.uuid4)
    parser.add_argument("--g2.op2", type=Calendar, default=lazy_instance(Calendar, firstweekday=2))

    cfg = parser.get_defaults()
    dump = parser.dump(cfg, skip_default=True)
    assert dump == "{}\n"

    cfg.g2.op2.class_path = f"{__name__}.SkipDefault"
    dump = parser.dump(cfg, skip_default=True)
    assert dump == f"g2:\n  op2:\n    class_path: {__name__}.SkipDefault\n    init_args:\n      firstweekday: 2\n"

    cfg.g2.op2.init_args.firstweekday = 0
    dump = parser.dump(cfg, skip_default=True)
    assert dump == f"g2:\n  op2:\n    class_path: {__name__}.SkipDefault\n"

    parser.link_arguments("g1.op1", "g2.op2.init_args.firstweekday")
    parser.link_arguments("g1.op2", "g2.op2.init_args.param")
    del cfg["g2.op2.init_args"]
    dump = parser.dump(cfg, skip_default=True)
    assert dump == f"g2:\n  op2:\n    class_path: {__name__}.SkipDefault\n"


class ImportClass:
    pass


def test_get_all_subclass_paths_import_error():
    def mocked_get_import_path(cls):
        if cls is ImportClass:
            raise ImportError("Failed to import ImportClass")
        return get_import_path(cls)

    with mock.patch("jsonargparse._typehints.get_import_path", mocked_get_import_path):
        with catch_warnings(record=True) as w:
            subclass_paths = get_all_subclass_paths(ImportClass)
    assert "Failed to import ImportClass" in str(w[0].message)
    assert subclass_paths == []
