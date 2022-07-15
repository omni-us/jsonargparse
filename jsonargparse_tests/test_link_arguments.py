#!/usr/bin/env python3

import json
import os
import unittest
import yaml
from calendar import Calendar, TextCalendar
from io import StringIO
from typing import Any, List, Mapping, Optional, Union
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    lazy_instance,
    Namespace,
    ParserError,
)
from jsonargparse.optionals import docstring_parser_support
from jsonargparse_tests.base import mock_module, TempDirTestCase


class LinkArgumentsTests(unittest.TestCase):

    def test_link_arguments_on_parse_compute_fn_single_arguments(self):
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--a.v1', default=2)
        parser.add_argument('--a.v2', type=int, default=3)
        parser.add_argument('--b.v2', type=int, default=4)
        def a_prod(a):
            return a['v1'] * a['v2']
        parser.link_arguments('a', 'b.v2', a_prod)

        cfg = parser.parse_args(['--a.v2=-5'])
        self.assertEqual(cfg.b.v2, cfg.a.v1*cfg.a.v2)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--a.v1=x']))

        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dump, {'a': {'v1': 2, 'v2': -5}})


    def test_link_arguments_on_parse_add_class_arguments(self):
        class ClassA:
            def __init__(self, v1: int = 2, v2: int = 3):
                pass

        class ClassB:
            def __init__(self, v1: int = -1, v2: int = 4, v3: int = 2):
                """ClassB title
                Args:
                    v1: b v1 help
                """

        parser = ArgumentParser(error_handler=None)
        parser.add_class_arguments(ClassA, 'a')
        parser.add_class_arguments(ClassB, 'b')
        parser.link_arguments('a.v2', 'b.v1')
        def add(*args):
            return sum(args)
        parser.link_arguments(('a.v1', 'a.v2'), 'b.v2', add)

        cfg = parser.parse_args([])
        self.assertEqual(cfg.b.v1, cfg.a.v2)
        self.assertEqual(cfg.b.v2, cfg.a.v1+cfg.a.v2)
        cfg = parser.parse_args(['--a.v1=11', '--a.v2=7'])
        self.assertEqual(7, cfg.b.v1)
        self.assertEqual(11+7, cfg.b.v2)
        self.assertEqual(Namespace(), parser.parse_args([], defaults=False))

        dump = yaml.safe_load(parser.dump(cfg))
        self.assertEqual(dump, {'a': {'v1': 11, 'v2': 7}, 'b': {'v3': 2}})

        with self.subTest('print_help'):
            help_str = StringIO()
            parser.print_help(help_str)
            self.assertIn('Linked arguments', help_str.getvalue())
            self.assertIn('a.v2 --> b.v1', help_str.getvalue())
            self.assertIn('add(a.v1, a.v2) --> b.v2', help_str.getvalue())
            if docstring_parser_support:
                self.assertIn('b v1 help', help_str.getvalue())

        with self.subTest('failure cases'):
            self.assertRaises(ParserError, lambda: parser.parse_args(['--b.v1=5']))
            self.assertRaises(ValueError, lambda: parser.link_arguments('a.v2', 'b.v1'))
            self.assertRaises(ValueError, lambda: parser.link_arguments('x', 'b.v2'))
            self.assertRaises(ValueError, lambda: parser.link_arguments('a.v1', 'x'))
            self.assertRaises(ValueError, lambda: parser.link_arguments(('a.v1', 'a.v2'), 'b.v3'))
            self.assertRaises(ValueError, lambda: parser.link_arguments('a.v1', 'b.v2', apply_on='bad'))


    def test_link_arguments_on_parse_add_subclass_arguments(self):
        class ClassA:
            def __init__(
                self,
                v1: Union[int, str] = 1,
                v2: Union[int, str] = 2,
            ):
                pass

        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(ClassA, 'a')
        parser.add_subclass_arguments(Calendar, 'c')

        def add(v1, v2):
            return v1 + v2
        parser.link_arguments(('a.init_args.v1', 'a.init_args.v2'), 'c.init_args.firstweekday', add)

        with mock_module(ClassA) as module:
            a_value = {
                'class_path': f'{module}.ClassA',
                'init_args': {'v2': 3},
            }

            cfg = parser.parse_args(['--a='+json.dumps(a_value), '--c=calendar.Calendar'])
            self.assertEqual(cfg.c.init_args.firstweekday, 4)
            self.assertEqual(cfg.c.init_args.firstweekday, cfg.a.init_args.v1+cfg.a.init_args.v2)

            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.a, ClassA)
            self.assertIsInstance(cfg_init.c, Calendar)

            dump = yaml.safe_load(parser.dump(cfg))
            a_value['init_args']['v1'] = 1
            self.assertEqual(dump, {'a': a_value, 'c': {'class_path': 'calendar.Calendar'}})

            self.assertRaises(ValueError, lambda: parser.link_arguments('a.init_args.v1', 'c.init_args'))

            a_value['init_args'] = {'v1': 'a', 'v2': 'b'}
            with self.assertRaises(ParserError):
                parser.parse_args(['--a='+json.dumps(a_value), '--c=calendar.Calendar'])


    def test_link_arguments_on_parse_add_subclass_arguments_with_instantiate_false(self):
        class ClassA:
            def __init__(
                self,
                v: Union[int, str] = 1,
                c: Optional[Calendar] = None,
            ):
                self.c = c

        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(ClassA, 'a')
        parser.add_subclass_arguments(Calendar, 'c', instantiate=False)
        parser.link_arguments('c', 'a.init_args.c')

        with mock_module(ClassA) as module:
            a_value = {'class_path': f'{module}.ClassA'}
            c_value = {
                'class_path': 'calendar.Calendar',
                'init_args': {
                    'firstweekday': 3,
                },
            }

            cfg = parser.parse_args(['--a='+json.dumps(a_value), '--c='+json.dumps(c_value)])
            self.assertEqual(cfg.c.as_dict(), {'class_path': 'calendar.Calendar', 'init_args': {'firstweekday': 3}})
            self.assertEqual(cfg.c, cfg.a.init_args.c)

            cfg_init = parser.instantiate_classes(cfg)
            self.assertIsInstance(cfg_init.c, Namespace)
            self.assertIsInstance(cfg_init.a, ClassA)
            self.assertIsInstance(cfg_init.a.c, Calendar)
            self.assertEqual(cfg_init.a.c.firstweekday, 3)

            dump = yaml.safe_load(parser.dump(cfg))
            self.assertNotIn('c', dump['a']['init_args'])


    def test_link_arguments_on_parse_add_subclass_arguments_as_dict(self):
        class ClassA:
            def __init__(
                self,
                a1: dict,
                a2: Optional[dict] = None,
                a3: Any = None,
            ):
                pass

        def return_dict(value: dict):
            return value

        parser = ArgumentParser(error_handler=None)
        parser.add_subclass_arguments(ClassA, 'a')
        parser.add_subclass_arguments(Calendar, 'c')
        parser.link_arguments('c', 'a.init_args.a1', compute_fn=return_dict)
        parser.link_arguments('c', 'a.init_args.a2')
        parser.link_arguments('c', 'a.init_args.a3')

        with mock_module(ClassA) as module:
            a_value = {'class_path': f'{module}.ClassA'}
            c_value = {
                'class_path': 'calendar.Calendar',
                'init_args': {
                    'firstweekday': 3,
                },
            }

            cfg = parser.parse_args(['--a='+json.dumps(a_value), '--c='+json.dumps(c_value)])
            self.assertEqual(cfg.a.init_args.a1, c_value)
            self.assertEqual(cfg.a.init_args.a2, c_value)


    def test_link_arguments_on_parse_within_subcommand(self):
        class Foo:
            def __init__(self, a: int):
                self.a = a

        parser = ArgumentParser()
        subparser = ArgumentParser()

        subcommands = parser.add_subcommands()
        subparser.add_class_arguments(Foo, nested_key='foo')
        subparser.add_argument('--b', type=int)
        subparser.link_arguments('b', 'foo.a')
        subcommands.add_subcommand('cmd', subparser)

        cfg = parser.parse_args(['cmd', '--b=2'])
        self.assertEqual(cfg['cmd']['foo'].as_dict(), {'a': 2})

        cfg = parser.instantiate_classes(cfg)
        self.assertIsInstance(cfg['cmd']['foo'], Foo)


    def test_link_arguments_on_instantiate(self):
        class ClassA:
            def __init__(self, a1: int, a2: float = 2.3):
                self.a1 = a1
                self.a2 = a2

        class ClassB:
            def __init__(self, b1: float = 4.5, b2: int = 6, b3: str = '7'):
                self.b1 = b1
                self.b2 = b2
                self.b3 = b3

        class ClassC:
            def __init__(self, c1: int = 7, c2: str = '8'):
                self.c1 = c1
                self.c2 = c2

        def make_parser_1():
            parser = ArgumentParser(error_handler=None)
            parser.add_class_arguments(ClassA, 'a')
            parser.add_class_arguments(ClassB, 'b')
            parser.add_class_arguments(ClassC, 'c')
            parser.add_argument('--d', type=int, default=-1)
            parser.link_arguments('b.b2', 'a.a1', apply_on='instantiate')
            parser.link_arguments('c.c1', 'b.b1', apply_on='instantiate')
            return parser

        def get_b2(obj_b):
            return obj_b.b2

        def make_parser_2():
            parser = ArgumentParser(error_handler=None)
            parser.add_subclass_arguments(ClassA, 'a')
            parser.add_subclass_arguments(ClassB, 'b')
            parser.link_arguments('b', 'a.init_args.a1', get_b2, apply_on='instantiate')
            return parser

        # Link object attribute
        parser = make_parser_1()
        parser.link_arguments('c.c2', 'b.b3', compute_fn=lambda v: f'"{v}"', apply_on='instantiate')
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertEqual(cfg.a.a1, 6)
        self.assertEqual(cfg.b.b3, '"8"')

        # Link all group arguments
        parser = make_parser_1()
        parser.link_arguments('b.b1', 'a.a2', apply_on='instantiate')
        cfg = parser.parse_args([])
        cfg = parser.instantiate_classes(cfg)
        self.assertEqual(cfg['a'].a1, 6)
        self.assertEqual(cfg['a'].a2, 7)

        # Links with cycle
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.a2', 'b.b3', apply_on='instantiate')
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.a2', 'c.c2', apply_on='instantiate')

        # Not subclass action or a class group
        parser = make_parser_1()
        with self.assertRaises(ValueError):
            parser.link_arguments('d', 'c.c2', apply_on='instantiate')

        # Link subclass and compute function
        with mock_module(ClassA, ClassB) as module:
            parser = make_parser_2()
            cfg = parser.parse_args([
                f'--a={module}.ClassA',
                f'--b={module}.ClassB',
            ])
            cfg = parser.instantiate_classes(cfg)
            self.assertEqual(cfg['a'].a1, 6)

        # Unsupported multi-source
        parser = make_parser_2()
        with self.assertRaises(ValueError):
            parser.link_arguments(('b.b2', 'c.c1'), 'b.b3', compute_fn=lambda x, y: x, apply_on='instantiate')

        # Source must be subclass action or class group
        parser = make_parser_2()
        with self.assertRaises(ValueError):
            parser.link_arguments('a.b.c', 'b.b3', apply_on='instantiate')


    def test_link_arguments_on_instantiate_no_compute_no_target_instantiate(self):
        class ClassA:
            def __init__(self, calendar: Calendar):
                self.calendar = calendar

        parser = ArgumentParser()
        parser.add_argument('--firstweekday', type=int)
        parser.add_class_arguments(ClassA, 'a', instantiate=False)
        parser.add_class_arguments(Calendar, 'c')
        parser.link_arguments('firstweekday', 'c.firstweekday')
        parser.link_arguments('c', 'a.calendar', apply_on='instantiate')

        cfg = parser.parse_args(['--firstweekday=2'])
        self.assertEqual(cfg, Namespace(c=Namespace(firstweekday=2), firstweekday=2))
        init = parser.instantiate_classes(cfg)
        self.assertIsInstance(init.a, Namespace)
        self.assertIsInstance(init.c, Calendar)
        self.assertIs(init.c, init.a.calendar)


    def test_link_arguments_on_instantiate_multi_source(self):
        class ClassA:
            def __init__(self, calendars: List[Calendar]):
                self.calendars = calendars

        def as_list(*items):
            return [*items]

        parser = ArgumentParser()
        parser.add_class_arguments(ClassA, 'a')
        parser.add_class_arguments(Calendar, 'c.one')
        parser.add_class_arguments(TextCalendar, 'c.two')
        parser.link_arguments(('c.one', 'c.two'), 'a.calendars', apply_on='instantiate', compute_fn=as_list)

        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('as_list(c.one, c.two) --> a.calendars [applied on instantiate]', help_str.getvalue())

        cfg = parser.parse_args([])
        self.assertEqual(cfg.as_dict(), {'c': {'one': {'firstweekday': 0}, 'two': {'firstweekday': 0}}})
        init = parser.instantiate_classes(cfg)
        self.assertIsInstance(init.c.one, Calendar)
        self.assertIsInstance(init.c.two, TextCalendar)
        self.assertEqual(init.a.calendars, [init.c.one, init.c.two])


    def test_link_arguments_on_instantiate_object_in_attribute(self):
        class ClassA:
            def __init__(self, firstweekday: int = 1):
                self.calendar = Calendar(firstweekday=firstweekday)

        class ClassB:
            def __init__(self, calendar: Calendar, p2: int = 2):
                self.calendar = calendar

        parser = ArgumentParser()
        parser.add_class_arguments(ClassA, 'a')
        parser.add_class_arguments(ClassB, 'b')
        with self.assertRaises(ValueError):
            parser.link_arguments('a.calendar', 'b.calendar', apply_on='parse')
        parser.link_arguments('a.calendar', 'b.calendar', apply_on='instantiate')

        cfg = parser.parse_args(['--a.firstweekday=2', '--b.p2=3'])
        self.assertEqual(cfg.a, Namespace(firstweekday=2))
        self.assertEqual(cfg.b, Namespace(p2=3))
        init = parser.instantiate_classes(cfg)
        self.assertIs(init.a.calendar, init.b.calendar)
        self.assertEqual(init.b.calendar.firstweekday, 2)


    def test_link_arguments_on_parse_entire_subclass(self):
        class ClassC:
            def __init__(self, calendar: Calendar):
                self.calendar = calendar

        class ClassB:
            def __init__(self, calendar: Calendar, p2: int = 2):
                self.calendar = calendar

        parser = ArgumentParser()
        parser.add_class_arguments(ClassC, 'c')
        parser.add_class_arguments(ClassB, 'b')
        parser.link_arguments('c.calendar', 'b.calendar', apply_on='parse')

        cal = {'class_path': 'Calendar', 'init_args': {'firstweekday': 4}}
        cfg = parser.parse_args([f'--c.calendar={cal}', '--b.p2=7'])
        self.assertEqual(cfg.c.calendar, cfg.b.calendar)
        self.assertEqual(cfg.b.p2, 7)


    def test_link_arguments_subclass_missing_param_issue_129(self):
        class ClassA:
            def __init__(self, a1: int = 1):
                self.a1 = a1

        class ClassB:
            def __init__(self, b1: int = 2):
                self.b1 = b1

        parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
        with mock_module(ClassA, ClassB) as module, self.assertLogs(logger=parser.logger, level='DEBUG') as log:
            parser.add_subclass_arguments(ClassA, 'a', default=lazy_instance(ClassA))
            parser.add_subclass_arguments(ClassB, 'b', default=lazy_instance(ClassB))
            parser.link_arguments('a.init_args.a2', 'b.init_args.b1', apply_on='parse')
            parser.link_arguments('a.init_args.a1', 'b.init_args.b2', apply_on='parse')

            parser.parse_args([f'--a={module}.ClassA', f'--b={module}.ClassB'])
            self.assertTrue(any('a.init_args.a2 --> b.init_args.b1 ignored since source' in x for x in log.output))
            self.assertTrue(any('a.init_args.a1 --> b.init_args.b2 ignored since target' in x for x in log.output))

        parser = ArgumentParser(error_handler=None, logger={'level': 'DEBUG'})
        with mock_module(ClassA, ClassB) as module, self.assertLogs(logger=parser.logger, level='DEBUG') as log:
            parser.add_subclass_arguments(ClassA, 'a', default=lazy_instance(ClassA))
            parser.add_subclass_arguments(ClassB, 'b', default=lazy_instance(ClassB))
            parser.link_arguments('a.init_args.a2', 'b.init_args.b1', apply_on='instantiate')
            parser.link_arguments('a.init_args.a1', 'b.init_args.b2', apply_on='instantiate')

            cfg = parser.parse_args([f'--a={module}.ClassA', f'--b={module}.ClassB'])
            parser.instantiate_classes(cfg)
            self.assertTrue(any('a.init_args.a2 --> b.init_args.b1 ignored since source' in x for x in log.output))
            self.assertTrue(any('a.init_args.a1 --> b.init_args.b2 ignored since target' in x for x in log.output))


class LinkArgumentsTempDirTests(TempDirTestCase):

    def test_linking_deep_targets(self):
        class D:
            pass

        class A:
            def __init__(self, d: D) -> None:
                self.d = d

        class BSuper:
            pass

        class BSub(BSuper):
            def __init__(self, a: A) -> None:
                self.a = a

        class C:
            def fn(self) -> D:
                return D()

        with mock_module(D, A, BSuper, BSub, C) as module:
            config = {
                "b": {
                    "class_path": f"{module}.BSub",
                    "init_args": {
                        "a": {
                            "class_path": f"{module}.A",
                        },
                    },
                },
                "c": {},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')  # TODO: Change to pathlib.Path
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f)

            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_subclass_arguments(BSuper, nested_key="b", required=True)
            parser.add_class_arguments(C, nested_key="c")
            parser.link_arguments("c", "b.init_args.a.init_args.d", compute_fn=C.fn, apply_on="instantiate")

            config = parser.parse_args(["--config", config_path])
            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a.d, D)


    def test_linking_deep_targets_mapping(self):
        class D:
            pass

        class A:
            def __init__(self, d: D) -> None:
                self.d = d

        class BSuper:
            pass

        class BSub(BSuper):
            def __init__(self, a_map: Mapping[str, A]) -> None:
                self.a_map = a_map

        class C:
            def fn(self) -> D:
                return D()

        with mock_module(D, A, BSuper, BSub, C) as module:
            config = {
                "b": {
                    "class_path": f"{module}.BSub",
                    "init_args": {
                        "a_map": {
                            "name": {
                                "class_path": f"{module}.A",
                            },
                        },
                    },
                },
                "c": {},
            }
            config_path = os.path.join(self.tmpdir, 'config.yaml')  # TODO: Change to pathlib.Path
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f)

            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile)
            parser.add_subclass_arguments(BSuper, nested_key="b", required=True)
            parser.add_class_arguments(C, nested_key="c")
            parser.link_arguments("c", "b.init_args.a_map.name.init_args.d", compute_fn=C.fn, apply_on="instantiate")

            config = parser.parse_args(["--config", config_path])
            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a_map["name"].d, D)

            config_init = parser.instantiate_classes(config)
            self.assertIsInstance(config_init["b"].a_map["name"].d, D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
