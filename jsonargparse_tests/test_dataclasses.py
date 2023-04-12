#!/usr/bin/env python3

import dataclasses
import sys
import unittest
from io import StringIO
from typing import Dict, List

import yaml

from jsonargparse import ArgumentError, ArgumentParser, Namespace, compose_dataclasses
from jsonargparse.optionals import (
    attrs_support,
    docstring_parser_support,
    import_attrs,
    import_pydantic,
    pydantic_support,
    set_docstring_parse_options,
)
from jsonargparse.typing import PositiveFloat, PositiveInt
from jsonargparse_tests.base import mock_module


@dataclasses.dataclass(frozen=True)
class MyDataClassA:
    """MyDataClassA description

    Args:
        a1: a1 help
        a2: a2 help
    """
    a1: PositiveInt = PositiveInt(1)  # type: ignore
    a2: str = '2'

@dataclasses.dataclass
class MyDataClassB:
    """MyDataClassB description

    Args:
        b1: b1 help
        b2: b2 help
    """
    b1: PositiveFloat = PositiveFloat(3.0)  # type: ignore
    b2: MyDataClassA = MyDataClassA()


class DataclassesTests(unittest.TestCase):

    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_add_dataclass_arguments(self):
        parser = ArgumentParser(exit_on_error=False)
        parser.add_dataclass_arguments(MyDataClassA, 'a', default=MyDataClassA(), title='CustomA title')
        parser.add_dataclass_arguments(MyDataClassB, 'b', default=MyDataClassB())

        cfg = parser.get_defaults()
        self.assertEqual(dataclasses.asdict(MyDataClassA()), cfg['a'].as_dict())
        self.assertEqual(dataclasses.asdict(MyDataClassB()), cfg['b'].as_dict())
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dataclasses.asdict(MyDataClassA()), dump['a'])
        self.assertEqual(dataclasses.asdict(MyDataClassB()), dump['b'])

        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['a'], MyDataClassA)
        self.assertIsInstance(cfg['b'], MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, MyDataClassA)

        self.assertEqual(5, parser.parse_args(['--b.b2.a1=5']).b.b2.a1)
        self.assertRaises(ArgumentError, lambda: parser.parse_args(['--b.b2.a1=x']))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('CustomA title:', help_str.getvalue())
        self.assertIn('MyDataClassB description:', help_str.getvalue())
        self.assertIn('b2 help:', help_str.getvalue())

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(1, 'c')

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(MyDataClassB, 'c', default=MyDataClassB(b2=MyDataClassB()))

        class MyClass(int, MyDataClassA):
            """MyClass description"""

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(MyClass, 'c')


    def test_dataclass_typehint(self):

        class MyClass:
            def __init__(
                self,
                a1: MyDataClassA = MyDataClassA(),
                a2: MyDataClassB = MyDataClassB(),
            ):
                self.a1 = a1
                self.a2 = a2

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass, 'g')

        cfg = parser.get_defaults()
        self.assertEqual(dataclasses.asdict(MyDataClassA()), cfg['g']['a1'].as_dict())
        self.assertEqual(dataclasses.asdict(MyDataClassB()), cfg['g']['a2'].as_dict())
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dataclasses.asdict(MyDataClassA()), dump['g']['a1'])
        self.assertEqual(dataclasses.asdict(MyDataClassB()), dump['g']['a2'])

        cfg_init = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg_init.g.a1, MyDataClassA)
        self.assertIsInstance(cfg_init.g.a2, MyDataClassB)
        self.assertIsInstance(cfg_init.g.a2.b2, MyDataClassA)


    def test_dataclass_typehint_in_subclass(self):

        class MyClass1:
            def __init__(self, a1: MyDataClassB = MyDataClassB()):
                """MyClass1 description"""
                self.a1 = a1

        class MyClass2:
            def __init__(self, c1: MyClass1):
                """MyClass2 description"""
                self.c1 = c1

        parser = ArgumentParser(exit_on_error=False)
        parser.add_class_arguments(MyClass2)

        with mock_module(MyClass1, MyClass2) as module:
            class_path = f'"class_path": "{module}.MyClass1"'
            init_args = '"init_args": {"a1": {"b2": {"a1": 7}}}'
            cfg = parser.parse_args(['--c1={'+class_path+', '+init_args+'}'])
            self.assertEqual(cfg.c1.class_path, f'{module}.MyClass1')
            self.assertEqual(cfg.c1.init_args.a1.b2.a1, 7)
            self.assertIsInstance(cfg.c1.init_args.a1.b2.a1, PositiveInt)
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['c1'], MyClass1)
            self.assertIsInstance(cfg['c1'].a1, MyDataClassB)
            self.assertIsInstance(cfg['c1'].a1.b2, MyDataClassA)
            self.assertIsInstance(cfg['c1'].a1.b1, PositiveFloat)


    def test_dataclass_add_argument_type(self):
        parser = ArgumentParser()
        parser.add_argument('--b', type=MyDataClassB, default=MyDataClassB(b1=7.0))

        cfg = parser.get_defaults()
        self.assertEqual({'b1': 7.0, 'b2': {'a1': 1, 'a2': '2'}}, cfg['b'].as_dict())

        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['b'], MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, MyDataClassA)


    def test_dataclass_add_argument_type_some_required(self):

        @dataclasses.dataclass
        class MyDataClass:
            a1: str
            a2: float = 1.2

        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument('--b', type=MyDataClass)

        self.assertEqual(Namespace(a1='v', a2=1.2), parser.parse_args(['--b.a1=v']).b)
        self.assertRaises(ArgumentError, lambda: parser.parse_args([]))


    def test_dataclass_field_init_false(self):

        @dataclasses.dataclass
        class DataInitFalse:
            p1: str = '-'
            p2: str = dataclasses.field(init=False)

        parser = ArgumentParser(exit_on_error=False)
        added = parser.add_dataclass_arguments(DataInitFalse, 'd')
        self.assertEqual(added, ['d.p1'])
        self.assertEqual(parser.get_defaults(), Namespace(d=Namespace(p1='-')))


    def test_dataclass_field_default_factory(self):

        @dataclasses.dataclass
        class MyDataClass:
            a1: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
            a2: Dict[str, float] = dataclasses.field(default_factory=lambda: {'a': 1.2, 'b': 3.4})

        parser = ArgumentParser()
        parser.add_class_arguments(MyDataClass)

        cfg = parser.get_defaults().as_dict()
        self.assertEqual([1, 2, 3], cfg['a1'])
        self.assertEqual({'a': 1.2, 'b': 3.4}, cfg['a2'])


    def test_dataclass_fail_untyped(self):

        class MyClass:
            def __init__(self, c1) -> None:
                self.c1 = c1

        @dataclasses.dataclass
        class MyDataclass:
            a1: MyClass
            a2: str = "a2"
            a3: str = "a3"

        parser = ArgumentParser(exit_on_error=False)
        parser.add_argument('--cfg', type=MyDataclass, fail_untyped=False)

        with mock_module(MyDataclass, MyClass) as module:
            class_path = f'"class_path": "{module}.MyClass"'
            init_args = '"init_args": {"c1": 1}'
            cfg = parser.parse_args(['--cfg.a1={'+class_path+', '+init_args+'}'])
            cfg = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg['cfg'], MyDataclass)
            self.assertIsInstance(cfg['cfg'].a1, MyClass)
            self.assertIsInstance(cfg['cfg'].a2, str)
            self.assertIsInstance(cfg['cfg'].a3, str)


    def test_compose_dataclasses(self):

        @dataclasses.dataclass
        class MyDataClassA:
            a: int = 1

            def __post_init__(self):
                self.a += 1

        @dataclasses.dataclass
        class MyDataClassB:
            b: str = '1'

        MyDataClassAB = compose_dataclasses(MyDataClassA, MyDataClassB)
        self.assertEqual(2, len(dataclasses.fields(MyDataClassAB)))
        self.assertEqual({'a': 3, 'b': '2'}, dataclasses.asdict(MyDataClassAB(a=2, b='2')))  # pylint: disable=unexpected-keyword-arg


    def test_instantiate_classes_dataclasses_lightning_issue_9207(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/9207

        @dataclasses.dataclass
        class MyDataClass:
            name: str = 'name'

        class MyClass:
            def __init__(self, data: MyDataClass):
                self.data = data

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass, 'class')
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['class'], MyClass)
        self.assertIsInstance(cfg['class'].data, MyDataClass)


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_attribute_docstrings(self):

        @dataclasses.dataclass
        class WithAttrDocs:
            attr_str: str = 'a'
            "attr_str description"
            attr_int: int = 1
            "attr_int description"

        set_docstring_parse_options(attribute_docstrings=True)

        parser = ArgumentParser()
        parser.add_class_arguments(WithAttrDocs)
        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('attr_str description (type: str, default: a)', help_str.getvalue())
        self.assertIn('attr_int description (type: int, default: 1)', help_str.getvalue())

        set_docstring_parse_options(attribute_docstrings=False)


@unittest.skipIf(sys.version_info == (3, 6), 'pydantic not supported in python 3.10')
@unittest.skipIf(not pydantic_support, 'pydantic package is required')
class PydanticTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pydantic = import_pydantic('PydanticTests')


    def test_dataclass(self):
        @self.pydantic.dataclasses.dataclass
        class Data:
            p1: float = 0.1
            p2: str = '-'

        parser = ArgumentParser()
        parser.add_argument('--data', type=Data)
        defaults = parser.get_defaults()
        self.assertEqual(Namespace(p1=0.1, p2='-'), defaults.data)
        cfg = parser.parse_args(['--data.p1=0.2', '--data.p2=x'])
        self.assertEqual(Namespace(p1=0.2, p2='x'), cfg.data)


    def test_basemodel(self):
        class Model(self.pydantic.BaseModel):
            p1: str
            p2: int = 3

        parser = ArgumentParser()
        parser.add_argument('--model', type=Model, default=Model(p1='a'))
        cfg = parser.parse_args(['--model.p2=5'])
        self.assertEqual(Namespace(p1='a', p2=5), cfg.model)


    def test_field_default_factory(self):
        class Model(self.pydantic.BaseModel):
            p1: List[int] = self.pydantic.Field(default_factory=lambda: [1, 2])

        parser = ArgumentParser()
        parser.add_argument('--model', type=Model)
        cfg1 = parser.parse_args([])
        cfg2 = parser.parse_args([])
        self.assertEqual(cfg1.model.p1, [1, 2])
        self.assertEqual(cfg1.model.p1, cfg2.model.p1)
        self.assertIsNot(cfg1.model.p1, cfg2.model.p1)


    def test_field_description(self):
        class Model(self.pydantic.BaseModel):
            """
            Args:
                p1: p1 help
            """
            p1: str
            p2: int = self.pydantic.Field(2, description="p2 help")

        parser = ArgumentParser()
        parser.add_argument('--model', type=Model)
        help_str = StringIO()
        parser.print_help(help_str)
        if docstring_parser_support:
            self.assertIn('p1 help (required, type: str)', help_str.getvalue())
        self.assertIn('p2 help (type: int, default: 2)', help_str.getvalue())


@unittest.skipIf(not attrs_support, 'attrs package is required')
class AttrsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.attrs = import_attrs('AttrsTests')

    def test_define(self):
        @self.attrs.define
        class Data:
            p1: float = 0.1
            p2: str = '-'

        parser = ArgumentParser()
        parser.add_argument('--data', type=Data)
        defaults = parser.get_defaults()
        self.assertEqual(Namespace(p1=0.1, p2='-'), defaults.data)
        cfg = parser.parse_args(['--data.p1=0.2', '--data.p2=x'])
        self.assertEqual(Namespace(p1=0.2, p2='x'), cfg.data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
