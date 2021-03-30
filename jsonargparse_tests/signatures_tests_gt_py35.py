#!/usr/bin/env python3

from io import StringIO
from typing import Dict, List
from jsonargparse_tests.base import *


@unittest.skipIf(not jsonschema_support or not dataclasses_support, 'jsonschema and dataclasses packages are required')
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
            a1: int = 1
            a2: str = '2'

        @cls.dataclasses.dataclass
        class MyDataClassB:
            """MyDataClassB description

            Args:
                b1: b1 help
                b2: b2 help
            """
            b1: float = 3.0
            b2: MyDataClassA = MyDataClassA()

        cls.MyDataClassA = MyDataClassA
        cls.MyDataClassB = MyDataClassB


    def test_add_dataclass_arguments(self):
        parser = ArgumentParser()
        parser.add_dataclass_arguments(self.MyDataClassA, 'a', default=self.MyDataClassA(), title='CustomA title')
        parser.add_dataclass_arguments(self.MyDataClassB, 'b', default=self.MyDataClassB())

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual({'a1': 1, 'a2': '2'}, cfg['a'])
        self.assertEqual({'b1': 3.0, 'b2': {'a1': 1, 'a2': '2'}}, cfg['b'])

        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg.a, self.MyDataClassA)
        self.assertIsInstance(cfg.b, self.MyDataClassB)
        self.assertIsInstance(cfg.b.b2, self.MyDataClassA)

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('CustomA title:', help_str.getvalue())
        self.assertIn('MyDataClassB description:', help_str.getvalue())
        self.assertIn('b2 help:', help_str.getvalue())

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(1, 'c')

        with self.assertRaises(ValueError):
            parser.add_dataclass_arguments(self.MyDataClassB, 'c', default=self.MyDataClassB(b2=self.MyDataClassB()))


    def test_dataclass_type(self):
        parser = ArgumentParser()
        parser.add_argument('--b', type=self.MyDataClassB, default=self.MyDataClassB(b1=-3.0))

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual({'b1': -3.0, 'b2': {'a1': 1, 'a2': '2'}}, cfg['b'])

        cfg = parser.instantiate_subclasses(cfg)
        self.assertIsInstance(cfg.b, self.MyDataClassB)
        self.assertIsInstance(cfg.b.b2, self.MyDataClassA)


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
