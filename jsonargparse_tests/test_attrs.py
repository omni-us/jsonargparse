from __future__ import annotations

from typing import List

import pytest

from jsonargparse import Namespace
from jsonargparse._optionals import attrs_support
from jsonargparse_tests.conftest import get_parser_help

if attrs_support:
    import attrs

    @attrs.define
    class AttrsData:
        p1: float
        p2: str = "-"

    @attrs.define
    class AttrsSubData(AttrsData):
        p3: int = 3

    @attrs.define
    class AttrsFieldFactory:
        p1: List[str] = attrs.field(factory=lambda: ["one", "two"])

    @attrs.define
    class AttrsFieldInitFalse:
        p1: dict = attrs.field(init=False)

        def __attrs_post_init__(self):
            self.p1 = {}

    @attrs.define
    class AttrsSubField:
        p1: str = "-"
        p2: int = 0

    @attrs.define
    class AttrsWithNestedDefaultDataclass:
        p1: float
        subfield: AttrsSubField = attrs.field(factory=AttrsSubField)

    @attrs.define
    class AttrsWithNestedDataclassNoDefault:
        p1: float
        subfield: AttrsSubField


@pytest.mark.skipif(not attrs_support, reason="attrs package is required")
class TestAttrs:
    def test_define(self, parser):
        parser.add_argument("--data", type=AttrsData)
        defaults = parser.get_defaults()
        assert Namespace(p1=None, p2="-") == defaults.data
        cfg = parser.parse_args(["--data.p1=0.2", "--data.p2=x"])
        assert Namespace(p1=0.2, p2="x") == cfg.data

    def test_subclass(self, parser):
        parser.add_argument("--data", type=AttrsSubData)
        defaults = parser.get_defaults()
        assert Namespace(p1=None, p2="-", p3=3) == defaults.data

    def test_field_factory(self, parser):
        parser.add_argument("--data", type=AttrsFieldFactory)
        cfg1 = parser.parse_args([])
        cfg2 = parser.parse_args([])
        assert cfg1.data.p1 == ["one", "two"]
        assert cfg1.data.p1 == cfg2.data.p1
        assert cfg1.data.p1 is not cfg2.data.p1

    def test_field_init_false(self, parser):
        parser.add_argument("--data", type=AttrsFieldInitFalse)
        cfg = parser.parse_args([])
        help_str = get_parser_help(parser)
        assert "--data.p1" not in help_str
        assert cfg == Namespace()
        init = parser.instantiate_classes(cfg)
        assert init.data.p1 == {}

    def test_nested_with_default(self, parser):
        parser.add_argument("--data", type=AttrsWithNestedDefaultDataclass)
        cfg = parser.parse_args(["--data.p1=1.23"])
        assert cfg.data == Namespace(p1=1.23, subfield=Namespace(p1="-", p2=0))

    def test_nested_without_default(self, parser):
        parser.add_argument("--data", type=AttrsWithNestedDataclassNoDefault)
        cfg = parser.parse_args(["--data.p1=1.23"])
        assert cfg.data == Namespace(p1=1.23, subfield=Namespace(p1="-", p2=0))
