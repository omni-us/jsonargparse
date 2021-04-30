#!/usr/bin/env python3

import yaml
from io import StringIO
from typing import Dict, List
from jsonargparse_tests.base import *


@unittest.skipIf(not dataclasses_support, 'dataclasses package is required')
class SignaturesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataclasses = import_dataclasses('SignaturesTests')

        @cls.dataclasses.dataclass
        class MyDataClassA:
            """MyDataClassA description

            Args:
                a1: a1 help
                a2: a2 help
            """
            a1: PositiveInt = PositiveInt(1)
            a2: str = '2'

        @cls.dataclasses.dataclass
        class MyDataClassB:
            """MyDataClassB description

            Args:
                b1: b1 help
                b2: b2 help
            """
            b1: PositiveFloat = PositiveFloat(3.0)
            b2: MyDataClassA = MyDataClassA()

        cls.MyDataClassA = MyDataClassA
        cls.MyDataClassB = MyDataClassB


    @unittest.skipIf(not docstring_parser_support, 'docstring-parser package is required')
    def test_add_dataclass_arguments(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_dataclass_arguments(self.MyDataClassA, 'a', default=self.MyDataClassA(), title='CustomA title')
        parser.add_dataclass_arguments(self.MyDataClassB, 'b', default=self.MyDataClassB())

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassA()), cfg['a'])
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassB()), cfg['b'])
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassA()), dump['a'])
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassB()), dump['b'])

        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['a'], self.MyDataClassA)
        self.assertIsInstance(cfg['b'], self.MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, self.MyDataClassA)

        self.assertEqual(5, parser.parse_args(['--b.b2.a1=5']).b.b2.a1)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--b.b2.a1=x']))

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('CustomA title:', help_str.getvalue())
        self.assertIn('MyDataClassB description:', help_str.getvalue())
        self.assertIn('b2 help:', help_str.getvalue())

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(1, 'c')

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(self.MyDataClassB, 'c', default=self.MyDataClassB(b2=self.MyDataClassB()))

        class MyClass(int, self.MyDataClassA):
            """MyClass description"""

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(MyClass, 'c')


    def test_dataclass_typehint(self):

        class MyClass:
            def __init__(
                self,
                a1: self.MyDataClassA = self.MyDataClassA(),  # type: ignore
                a2: self.MyDataClassB = self.MyDataClassB(),  # type: ignore
            ):
                """MyClass description"""

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass, 'g')

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassA()), cfg['g']['a1'])
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassB()), cfg['g']['a2'])
        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassA()), dump['g']['a1'])
        self.assertEqual(self.dataclasses.asdict(self.MyDataClassB()), dump['g']['a2'])

        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['g']['a1'], self.MyDataClassA)
        self.assertIsInstance(cfg['g']['a2'], self.MyDataClassB)
        self.assertIsInstance(cfg['g']['a2'].b2, self.MyDataClassA)


    def test_dataclass_typehint_in_subclass(self):

        class MyClass1:
            def __init__(self, a1: self.MyDataClassB = self.MyDataClassB()):  # type: ignore
                """MyClass1 description"""
                self.a1 = a1

        class MyClass2:
            def __init__(self, c1: MyClass1):
                """MyClass2 description"""
                self.c1 = c1

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(MyClass2)

        from jsonargparse_tests import signatures_tests_gt_py35
        setattr(signatures_tests_gt_py35, 'MyClass1', MyClass1)

        class_path = '"class_path": "jsonargparse_tests.signatures_tests_gt_py35.MyClass1"'
        init_args = '"init_args": {"a1": {"b2": {"a1": 7}}}'
        cfg = parser.parse_args(['--c1={'+class_path+', '+init_args+'}'])
        self.assertEqual(cfg.c1.class_path, 'jsonargparse_tests.signatures_tests_gt_py35.MyClass1')
        self.assertEqual(cfg.c1.init_args.a1.b2.a1, 7)
        self.assertIsInstance(cfg.c1.init_args.a1.b2.a1, PositiveInt)
        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['c1'], MyClass1)
        self.assertIsInstance(cfg['c1'].a1, self.MyDataClassB)
        self.assertIsInstance(cfg['c1'].a1.b2, self.MyDataClassA)
        self.assertIsInstance(cfg['c1'].a1.b1, PositiveFloat)


    def test_dataclass_add_argument_type(self):
        parser = ArgumentParser()
        parser.add_argument('--b', type=self.MyDataClassB, default=self.MyDataClassB(b1=7.0))

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual({'b1': 7.0, 'b2': {'a1': 1, 'a2': '2'}}, cfg['b'])

        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg['b'], self.MyDataClassB)
        self.assertIsInstance(cfg['b'].b2, self.MyDataClassA)


    def test_dataclass_add_argument_type_some_required(self):

        @self.dataclasses.dataclass
        class MyDataClass:
            a1: str
            a2: float = 1.2

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--b', type=MyDataClass)

        self.assertEqual(Namespace(a1='v', a2=1.2), parser.parse_args(['--b.a1=v']).b)
        self.assertRaises(ParserError, lambda: parser.parse_args([]))


    def test_dataclass_field_default_factory(self):

        @self.dataclasses.dataclass
        class MyDataClass:
            a1: List[int] = self.dataclasses.field(default_factory=lambda: [1, 2, 3])
            a2: Dict[str, float] = self.dataclasses.field(default_factory=lambda: {'a': 1.2, 'b': 3.4})

        parser = ArgumentParser()
        parser.add_class_arguments(MyDataClass)

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual([1, 2, 3], cfg['a1'])
        self.assertEqual({'a': 1.2, 'b': 3.4}, cfg['a2'])


    def test_compose_dataclasses(self):
        dataclasses = import_dataclasses('test_compose_dataclasses')

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
        self.assertEqual({'a': 3, 'b': '2'}, dataclasses.asdict(MyDataClassAB(a=2, b='2')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
