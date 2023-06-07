from calendar import Calendar, TextCalendar
from typing import Any, List, Mapping, Optional, Union

import pytest
import yaml

from jsonargparse import (
    ActionConfigFile,
    ArgumentError,
    ArgumentParser,
    Namespace,
    lazy_instance,
)
from jsonargparse_tests.conftest import get_parser_help

# tests for links applied on parse


def test_on_parse_compute_fn_single_arguments(parser, subtests):
    def a_prod(a):
        return a["v1"] * a["v2"]

    parser.add_argument("--a.v1", default=2)
    parser.add_argument("--a.v2", type=int, default=3)
    parser.add_argument("--b.v2", type=int, default=4)
    parser.link_arguments("a", "b.v2", a_prod)

    with subtests.test("parse_args"):
        cfg = parser.parse_args(["--a.v2=-5"])
        assert cfg.b.v2 == cfg.a.v1 * cfg.a.v2

    with subtests.test("dump removal of target"):
        dump = yaml.safe_load(parser.dump(cfg))
        assert dump == {"a": {"v1": 2, "v2": -5}}

    with subtests.test("invalid compute_fn result type"):
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args(["--a.v1=x"])
        ctx.match('Parser key "b.v2"')


def test_on_parse_compute_fn_subclass_spec(parser, subtests):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--cal1", type=Calendar, default=lazy_instance(TextCalendar))
    parser.add_argument("--cal2", type=Calendar, default=lazy_instance(Calendar))
    parser.link_arguments(
        "cal1",
        "cal2.init_args.firstweekday",
        compute_fn=lambda c: c.init_args.firstweekday + 1,
    )

    with subtests.test("from config and default init_args"):
        cfg = parser.parse_args(['--cfg={"cal1": "Calendar"}'])
        assert cfg.cal1.init_args.firstweekday == 0
        assert cfg.cal2.init_args.firstweekday == 1

    with subtests.test("from given init_args"):
        cfg = parser.parse_args(["--cal1.init_args.firstweekday=2"])
        assert cfg.cal1.init_args.firstweekday == 2
        assert cfg.cal2.init_args.firstweekday == 3

    with subtests.test("invalid init parameter"):
        parser.set_defaults(cal1=None)
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args(["--cal1.firstweekday=-"])
        ctx.match('Parser key "cal1"')


class ClassA:
    def __init__(self, v1: int = 2, v2: int = 3):
        pass


class ClassB:
    def __init__(self, v1: int = -1, v2: int = 4, v3: int = 2):
        pass


def parser_classes_links_on_parse():
    def add(*args):
        return sum(args)

    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(ClassA, "a")
    parser.add_class_arguments(ClassB, "b")
    parser.link_arguments("a.v2", "b.v1")
    parser.link_arguments(("a.v1", "a.v2"), "b.v2", add)
    return parser


def test_on_parse_add_class_arguments(subtests):
    parser = parser_classes_links_on_parse()

    with subtests.test("without defaults"):
        with pytest.raises(ArgumentError) as ctx:
            parser.parse_args([], defaults=False)
        ctx.match('Key "a.v2" not found')

    with subtests.test("no arguments"):
        cfg = parser.parse_args([])
        assert cfg.b.v1 == cfg.a.v2
        assert cfg.b.v2 == cfg.a.v1 + cfg.a.v2

    with subtests.test("dump removal of targets"):
        cfg = parser.parse_args(["--a.v1=11", "--a.v2=7"])
        assert 7 == cfg.b.v1
        assert 11 + 7 == cfg.b.v2
        dump = yaml.safe_load(parser.dump(cfg))
        assert dump == {"a": {"v1": 11, "v2": 7}, "b": {"v3": 2}}

    with subtests.test("argument error"):
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--b.v1=5"]))


class ClassS:
    def __init__(
        self,
        v1: Union[int, str] = 1,
        v2: Union[int, str] = 2,
    ):
        pass


def test_on_parse_add_subclass_arguments(parser, subtests):
    def add(v1, v2):
        return v1 + v2

    parser.add_subclass_arguments(ClassS, "s")
    parser.add_subclass_arguments(Calendar, "c")
    parser.link_arguments(("s.init_args.v1", "s.init_args.v2"), "c.init_args.firstweekday", add)

    s_value = {
        "class_path": f"{__name__}.ClassS",
        "init_args": {"v2": 3},
    }

    with subtests.test("compute_fn result"):
        cfg = parser.parse_args([f"--s={s_value}", "--c=calendar.Calendar"])
        assert cfg.c.init_args.firstweekday == 4
        assert cfg.c.init_args.firstweekday == cfg.s.init_args.v1 + cfg.s.init_args.v2

    with subtests.test("dump removal of target"):
        cfg = parser.parse_args([f"--s={s_value}", "--c=calendar.Calendar"])
        dump = yaml.safe_load(parser.dump(cfg))
        assert dump["c"] == {"class_path": "calendar.Calendar"}

    with subtests.test("compute_fn invalid result type"):
        s_value["init_args"] = {"v1": "a", "v2": "b"}
        with pytest.raises(ArgumentError):
            parser.parse_args([f"--s={s_value}", "--c=calendar.Calendar"])


class Logger:
    def __init__(self, save_dir: Optional[str] = None):
        pass


class Trainer:
    def __init__(
        self,
        save_dir: Optional[str] = None,
        logger: Union[bool, Logger] = False,
    ):
        pass


def test_on_parse_subclass_target_in_union(parser):
    parser.add_class_arguments(Trainer, "trainer")
    parser.link_arguments("trainer.save_dir", "trainer.logger.init_args.save_dir")
    cfg = parser.parse_args([])
    assert cfg.trainer == Namespace(logger=False, save_dir=None)
    cfg = parser.parse_args(["--trainer.save_dir=logs", "--trainer.logger=Logger"])
    assert cfg.trainer.save_dir == "logs"
    assert cfg.trainer.logger.init_args == Namespace(save_dir="logs")


class ClassF:
    def __init__(
        self,
        v: Union[int, str] = 1,
        c: Optional[Calendar] = None,
    ):
        self.c = c


def test_on_parse_add_subclass_arguments_with_instantiate_false(parser, subtests):
    parser.add_subclass_arguments(ClassF, "f")
    parser.add_subclass_arguments(Calendar, "c", instantiate=False)
    parser.link_arguments("c", "f.init_args.c")

    f_value = {"class_path": f"{__name__}.ClassF"}
    c_value = {
        "class_path": "calendar.Calendar",
        "init_args": {
            "firstweekday": 3,
        },
    }

    with subtests.test("parse_args"):
        cfg = parser.parse_args([f"--f={f_value}", f"--c={c_value}"])
        assert cfg.c.as_dict() == {
            "class_path": "calendar.Calendar",
            "init_args": {"firstweekday": 3},
        }
        assert cfg.c == cfg.f.init_args.c

    with subtests.test("class instantiation"):
        init = parser.instantiate_classes(cfg)
        assert isinstance(init.c, Namespace)
        assert isinstance(init.f, ClassF)
        assert isinstance(init.f.c, Calendar)
        assert init.f.c.firstweekday == 3

    with subtests.test("dump removal of target"):
        dump = yaml.safe_load(parser.dump(cfg))
        assert "c" not in dump["f"]["init_args"]


class ClassD:
    def __init__(
        self,
        a1: dict,
        a2: Optional[dict] = None,
        a3: Any = None,
    ):
        pass


def test_on_parse_add_subclass_arguments_compute_fn_return_dict(parser):
    def return_dict(value: dict):
        return value

    parser.add_subclass_arguments(ClassD, "d")
    parser.add_subclass_arguments(Calendar, "c")
    parser.link_arguments("c", "d.init_args.a1", compute_fn=return_dict)
    parser.link_arguments("c", "d.init_args.a2")
    parser.link_arguments("c", "d.init_args.a3")

    d_value = {"class_path": f"{__name__}.ClassD"}
    c_value = {
        "class_path": "calendar.Calendar",
        "init_args": {
            "firstweekday": 3,
        },
    }

    cfg = parser.parse_args([f"--d={d_value}", f"--c={c_value}"])
    assert cfg.d.init_args.a1 == c_value
    assert cfg.d.init_args.a2 == c_value

    init = parser.instantiate_classes(cfg)
    assert isinstance(init.d, ClassD)
    assert isinstance(init.c, Calendar)


class Foo:
    def __init__(self, a: int):
        self.a = a


def test_on_parse_within_subcommand(parser, subparser):
    subparser.add_class_arguments(Foo, nested_key="foo")
    subparser.add_argument("--b", type=int)
    subparser.link_arguments("b", "foo.a")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("cmd", subparser)

    cfg = parser.parse_args(["cmd", "--b=2"])
    assert cfg["cmd"]["foo"].as_dict() == {"a": 2}

    init = parser.instantiate_classes(cfg)
    assert isinstance(init["cmd"]["foo"], Foo)


# tests for links applied on instantiate


class ClassX:
    def __init__(self, x1: int, x2: float = 2.3):
        self.x1 = x1
        self.x2 = x2


class ClassY:
    def __init__(self, y1: float = 4.5, y2: int = 6, y3: str = "7"):
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3


class ClassZ:
    def __init__(self, z1: int = 7, z2: str = "8"):
        self.z1 = z1
        self.z2 = z2


def get_parser_classes_links_on_instantiate():
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(ClassX, "x")
    parser.add_class_arguments(ClassY, "y")
    parser.add_class_arguments(ClassZ, "z")
    parser.add_argument("--d", type=int, default=-1)
    parser.link_arguments("y.y2", "x.x1", apply_on="instantiate")
    parser.link_arguments("z.z1", "y.y1", apply_on="instantiate")
    return parser


def get_parser_subclasses_link_on_instantiate():
    def get_y2(obj_y):
        return obj_y.y2

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(ClassX, "x")
    parser.add_subclass_arguments(ClassY, "y")
    parser.link_arguments("y", "x.init_args.x1", get_y2, apply_on="instantiate")
    return parser


def test_on_instantiate_link_instance_attribute():
    parser = get_parser_classes_links_on_instantiate()
    parser.link_arguments("z.z2", "y.y3", compute_fn=lambda v: f'"{v}"', apply_on="instantiate")
    cfg = parser.parse_args([])
    assert "x1" not in cfg.x
    assert "y3" not in cfg.y
    init = parser.instantiate_classes(cfg)
    assert init.x.x1 == 6
    assert init.y.y3 == '"8"'


def test_on_instantiate_link_all_group_arguments():
    parser = get_parser_classes_links_on_instantiate()
    parser.link_arguments("y.y1", "x.x2", apply_on="instantiate")
    cfg = parser.parse_args([])
    assert "x" not in cfg
    init = parser.instantiate_classes(cfg)
    assert init["x"].x1 == 6
    assert init["x"].x2 == 7


def test_on_instantiate_link_from_subclass_with_compute_fn():
    parser = get_parser_subclasses_link_on_instantiate()
    cfg = parser.parse_args(
        [
            f"--x={__name__}.ClassX",
            f"--y={__name__}.ClassY",
        ]
    )
    init = parser.instantiate_classes(cfg)
    assert init.x.x1 == 6


class ClassN:
    def __init__(self, calendar: Calendar):
        self.calendar = calendar


def test_on_parse_and_instantiate_link_entire_instance(parser):
    parser.add_argument("--firstweekday", type=int)
    parser.add_class_arguments(ClassN, "n", instantiate=False)
    parser.add_class_arguments(Calendar, "c")
    parser.link_arguments("firstweekday", "c.firstweekday", apply_on="parse")
    parser.link_arguments("c", "n.calendar", apply_on="instantiate")

    cfg = parser.parse_args(["--firstweekday=2"])
    assert cfg == Namespace(c=Namespace(firstweekday=2), firstweekday=2)
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.n, Namespace)
    assert isinstance(init.c, Calendar)
    assert init.c is init.n.calendar


class ClassM:
    def __init__(self, calendars: List[Calendar]):
        self.calendars = calendars


def test_on_instantiate_link_multi_source(parser):
    def as_list(*items):
        return [*items]

    parser.add_class_arguments(ClassM, "m")
    parser.add_class_arguments(Calendar, "c.one")
    parser.add_class_arguments(TextCalendar, "c.two")
    parser.link_arguments(("c.one", "c.two"), "m.calendars", apply_on="instantiate", compute_fn=as_list)

    cfg = parser.parse_args([])
    assert cfg.as_dict() == {"c": {"one": {"firstweekday": 0}, "two": {"firstweekday": 0}}}
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.c.one, Calendar)
    assert isinstance(init.c.two, TextCalendar)
    assert init.m.calendars == [init.c.one, init.c.two]


class ClassP:
    def __init__(self, firstweekday: int = 1):
        self.calendar = Calendar(firstweekday=firstweekday)


class ClassQ:
    def __init__(self, calendar: Calendar, q2: int = 2):
        self.calendar = calendar


def test_on_instantiate_link_object_in_attribute(parser):
    parser.add_class_arguments(ClassP, "p")
    parser.add_class_arguments(ClassQ, "q")
    parser.link_arguments("p.calendar", "q.calendar", apply_on="instantiate")

    cfg = parser.parse_args(["--p.firstweekday=2", "--q.q2=3"])
    assert cfg.p == Namespace(firstweekday=2)
    assert cfg.q == Namespace(q2=3)
    init = parser.instantiate_classes(cfg)
    assert init.p.calendar is init.q.calendar
    assert init.q.calendar.firstweekday == 2


def test_on_parse_link_entire_subclass(parser):
    parser.add_class_arguments(ClassN, "n")
    parser.add_class_arguments(ClassQ, "q")
    parser.link_arguments("n.calendar", "q.calendar", apply_on="parse")

    cal = {"class_path": "Calendar", "init_args": {"firstweekday": 4}}
    cfg = parser.parse_args([f"--n.calendar={cal}", "--q.q2=7"])
    assert cfg.n.calendar == cfg.q.calendar
    assert cfg.q.q2 == 7


class ClassV:
    def __init__(self, v1: int = 1):
        self.v1 = v1


class ClassW:
    def __init__(self, w1: int = 2):
        self.w1 = w1


def test_on_parse_subclass_link_ignored_missing_param(parser, caplog):
    parser.logger = {"level": "DEBUG"}
    parser.logger.handlers = [caplog.handler]

    parser.add_subclass_arguments(ClassV, "v", default=lazy_instance(ClassV))
    parser.add_subclass_arguments(ClassW, "w", default=lazy_instance(ClassW))
    parser.link_arguments("v.init_args.v2", "w.init_args.w1", apply_on="parse")
    parser.link_arguments("v.init_args.v1", "w.init_args.w2", apply_on="parse")

    parser.parse_args([f"--v={__name__}.ClassV", f"--w={__name__}.ClassW"])
    assert "v.init_args.v2 --> w.init_args.w1 ignored since source" in caplog.text
    assert "v.init_args.v1 --> w.init_args.w2 ignored since target" in caplog.text


def test_on_instantiate_subclass_link_ignored_missing_param(parser, caplog):
    parser.logger = {"level": "DEBUG"}
    parser.logger.handlers = [caplog.handler]

    parser.add_subclass_arguments(ClassV, "v", default=lazy_instance(ClassV))
    parser.add_subclass_arguments(ClassW, "w", default=lazy_instance(ClassW))
    parser.link_arguments("v.init_args.v2", "w.init_args.w1", apply_on="instantiate")
    parser.link_arguments("v.init_args.v1", "w.init_args.w2", apply_on="instantiate")

    cfg = parser.parse_args([f"--v={__name__}.ClassV", f"--w={__name__}.ClassW"])
    parser.instantiate_classes(cfg)
    assert "v.init_args.v2 --> w.init_args.w1 ignored since source" in caplog.text
    assert "v.init_args.v1 --> w.init_args.w2 ignored since target" in caplog.text


# link creation failures


def test_link_failure_invalid_apply_on(parser):
    parser.add_argument("--a")
    parser.add_argument("--b")
    with pytest.raises(ValueError):
        parser.link_arguments("a", "b", apply_on="bad")


def test_on_parse_link_failure_previous_target_as_source(parser):
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--c")
    parser.link_arguments("a", "b")
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("b", "c")
    ctx.match('Source "b" not allowed')


def test_on_parse_link_failure_previous_source_as_target(parser):
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--c")
    parser.link_arguments("b", "c")
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("a", "b")
    ctx.match('Target "b" not allowed')


def test_on_parse_link_failure_already_linked():
    parser = parser_classes_links_on_parse()
    with pytest.raises(ValueError):
        parser.link_arguments("a.v2", "b.v1")


def test_on_parse_link_failure_non_existing_source():
    parser = parser_classes_links_on_parse()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("x", "b.v2")
    ctx.match('No action for key "x"')


def test_on_parse_link_failure_non_existing_target():
    parser = parser_classes_links_on_parse()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("a.v1", "x")
    ctx.match('No action for key "x"')


def test_on_parse_link_failure_multi_source_missing_compute_fn():
    parser = parser_classes_links_on_parse()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments(("a.v1", "a.v2"), "b.v3")
    ctx.match("Multiple source keys requires a compute function")


def test_on_parse_link_failure_invalid_subclass_target(parser):
    parser.add_subclass_arguments(ClassS, "s")
    parser.add_subclass_arguments(Calendar, "c")
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("s.init_args.v1", "c.init_args")
    ctx.match('Target key expected to start with "c.init_args."')


def test_on_parse_link_failure_invalid_source_attribute(parser):
    parser.add_class_arguments(ClassP, "p")
    parser.add_class_arguments(ClassQ, "q")
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("p.calendar", "q.calendar", apply_on="parse")
    ctx.match('key "p.calendar"')


def test_on_instantiate_link_failure_cycle_self():
    parser = get_parser_classes_links_on_instantiate()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("y.y2", "y.y3", apply_on="instantiate")
    ctx.match("cycle")


def test_on_instantiate_link_failure_cycle_pair():
    parser = get_parser_classes_links_on_instantiate()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("x.x2", "y.y3", apply_on="instantiate")
    ctx.match("cycle")


def test_on_instantiate_link_failure_cycle_triple():
    parser = get_parser_classes_links_on_instantiate()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("x.x2", "z.z2", apply_on="instantiate")
    ctx.match("cycle")


def test_on_instantiate_link_failure_source_not_class():
    parser = get_parser_classes_links_on_instantiate()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("d", "c.c2", apply_on="instantiate")
    ctx.match("require source")


def test_on_instantiate_link_failure_missing_init_args():
    parser = get_parser_subclasses_link_on_instantiate()
    with pytest.raises(ValueError) as ctx:
        parser.link_arguments("x.y.z", "y.y3", apply_on="instantiate")
    ctx.match('Target key expected to start with "y.init_args."')


# help tests


def test_help_link_on_parse_single_source(parser):
    parser.add_class_arguments(ClassA, "a")
    parser.add_class_arguments(ClassB, "b")
    parser.link_arguments("a.v2", "b.v1")
    help_str = get_parser_help(parser)
    assert "a.v2 --> b.v1 [applied on parse]" in help_str


def test_help_link_on_parse_multi_source(parser):
    parser.add_class_arguments(ClassA, "a")
    parser.add_class_arguments(ClassB, "b")
    parser.link_arguments(("a.v1", "a.v2"), "b.v2", sum)
    help_str = get_parser_help(parser)
    assert "sum(a.v1, a.v2) --> b.v2 [applied on parse]" in help_str


def test_help_link_on_parse_lambda_compute_fn():
    parser = get_parser_classes_links_on_instantiate()
    parser.link_arguments("z.z2", "y.y3", lambda v: v)
    help_str = get_parser_help(parser)
    assert "<lambda>(z.z2) --> y.y3 [applied on parse]" in help_str


def test_help_link_on_instantiate_single_source():
    parser = get_parser_classes_links_on_instantiate()
    help_str = get_parser_help(parser)
    assert "y.y2 --> x.x1 [applied on instantiate]" in help_str


def test_help_link_on_instantiate_multi_source():
    parser = get_parser_subclasses_link_on_instantiate()
    help_str = get_parser_help(parser)
    assert "get_y2(y) --> x.init_args.x1 [applied on instantiate]" in help_str


# other tests


class DeepD:
    pass


class DeepA:
    def __init__(self, d: DeepD) -> None:
        self.d = d


class DeepBSuper:
    pass


class DeepBSub(DeepBSuper):
    def __init__(self, a: DeepA) -> None:
        self.a = a


class DeepC:
    def fn(self) -> DeepD:
        return DeepD()


def test_on_instantiate_linking_deep_targets(parser, tmp_path):
    config = {
        "b": {
            "class_path": f"{__name__}.DeepBSub",
            "init_args": {
                "a": {
                    "class_path": f"{__name__}.DeepA",
                },
            },
        },
        "c": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_subclass_arguments(DeepBSuper, nested_key="b", required=True)
    parser.add_class_arguments(DeepC, nested_key="c")
    parser.link_arguments("c", "b.init_args.a.init_args.d", compute_fn=DeepC.fn, apply_on="instantiate")

    config = parser.parse_args([f"--config={config_path}"])
    config_init = parser.instantiate_classes(config)
    assert isinstance(config_init["b"].a.d, DeepD)


class DeepBSub2(DeepBSuper):
    def __init__(self, a_map: Mapping[str, DeepA]) -> None:
        self.a_map = a_map


def test_on_instantiate_linking_deep_targets_mapping(parser, tmp_path):
    config = {
        "b": {
            "class_path": f"{__name__}.DeepBSub2",
            "init_args": {
                "a_map": {
                    "name": {
                        "class_path": f"{__name__}.DeepA",
                    },
                },
            },
        },
        "c": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_subclass_arguments(DeepBSuper, nested_key="b", required=True)
    parser.add_class_arguments(DeepC, nested_key="c")
    parser.link_arguments(
        "c",
        "b.init_args.a_map.name.init_args.d",
        compute_fn=DeepC.fn,
        apply_on="instantiate",
    )

    config = parser.parse_args([f"--config={config_path}"])
    config_init = parser.instantiate_classes(config)
    assert isinstance(config_init["b"].a_map["name"].d, DeepD)

    config_init = parser.instantiate_classes(config)
    assert isinstance(config_init["b"].a_map["name"].d, DeepD)


class DeepTarget:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


class Node:
    def __init__(self, sub_class: DeepTarget) -> None:
        self.sub_class = sub_class


class Source:
    def __init__(self) -> None:
        self.a = 1


def test_on_instantiate_linking_deep_targets_multiple(parser):
    parser.add_subclass_arguments(Node, "Node")
    parser.add_class_arguments(Source, "Source")
    parser.link_arguments("Source.a", "Node.init_args.sub_class.init_args.a", apply_on="instantiate")
    parser.link_arguments("Source.a", "Node.init_args.sub_class.init_args.b", apply_on="instantiate")
    cfg = parser.parse_args(["--Node=Node", "--Node.init_args.sub_class=DeepTarget"])
    init = parser.instantiate_classes(cfg)
    assert 1 == init.Node.sub_class.a
    assert 1 == init.Node.sub_class.b
