#!/usr/bin/env python3

from typing import Dict, List
from jsonargparse_tests.base import *


@unittest.skipIf(not jsonschema_support or not dataclasses_support, 'jsonschema and dataclasses packages are required')
class SignaturesTests(unittest.TestCase):

    def test_dataclass_field_default_factory(self):
        dataclasses = import_dataclasses('test_dataclass_field_default_factory')

        @dataclasses.dataclass
        class MyClass:
            a1: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
            a2: Dict[str, float] = dataclasses.field(default_factory=lambda: {'a': 1.2, 'b': 3.4})

        parser = ArgumentParser()
        parser.add_class_arguments(MyClass)

        cfg = namespace_to_dict(parser.get_defaults())
        self.assertEqual([1, 2, 3], cfg['a1'])
        self.assertEqual({'a': 1.2, 'b': 3.4}, cfg['a2'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
