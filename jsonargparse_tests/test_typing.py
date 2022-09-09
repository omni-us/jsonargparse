#!/usr/bin/env python3

import os
import pathlib
import pickle
import unittest
from datetime import timedelta
from jsonargparse import ArgumentParser, ParserError
from jsonargparse.typing import (
    ClosedUnitInterval,
    Email,
    NonNegativeFloat,
    NonNegativeInt,
    OpenUnitInterval,
    Path_fr,
    path_type,
    PositiveFloat,
    PositiveInt,
    RegisteredType,
    registered_types,
    restricted_number_type,
    restricted_string_type,
)
from jsonargparse.util import object_path_serializer
from jsonargparse_tests.base import mock_module, TempDirTestCase


class RestrictedNumberTests(unittest.TestCase):

    def test_PositiveInt(self):
        self.assertEqual(1, PositiveInt(1))
        self.assertEqual(2, PositiveInt('2'))
        self.assertRaises(ValueError, lambda: PositiveInt(0))
        self.assertRaises(ValueError, lambda: PositiveInt('-3'))
        self.assertRaises(ValueError, lambda: PositiveInt('4.0'))
        self.assertRaises(ValueError, lambda: PositiveInt(5.6))


    def test_NonNegativeInt(self):
        self.assertEqual(0, NonNegativeInt(0))
        self.assertEqual(1, NonNegativeInt('1'))
        self.assertRaises(ValueError, lambda: NonNegativeInt(-1))
        self.assertRaises(ValueError, lambda: NonNegativeInt('-2'))
        self.assertRaises(ValueError, lambda: NonNegativeInt('3.0'))
        self.assertRaises(ValueError, lambda: NonNegativeInt(4.5))


    def test_PositiveFloat(self):
        self.assertEqual(0.1, PositiveFloat(0.1))
        self.assertEqual(0.2, PositiveFloat('0.2'))
        self.assertEqual(3.0, PositiveFloat(3))
        self.assertRaises(ValueError, lambda: PositiveFloat(0))
        self.assertRaises(ValueError, lambda: PositiveFloat('-0.4'))


    def test_NonNegativeFloat(self):
        self.assertEqual(0.0, NonNegativeFloat(0.0))
        self.assertEqual(0.1, NonNegativeFloat('0.1'))
        self.assertEqual(2.0, NonNegativeFloat(2))
        self.assertRaises(ValueError, lambda: NonNegativeFloat(-0.1))
        self.assertRaises(ValueError, lambda: NonNegativeFloat('-2'))


    def test_ClosedUnitInterval(self):
        self.assertEqual(0.0, ClosedUnitInterval(0.0))
        self.assertEqual(1.0, ClosedUnitInterval('1'))
        self.assertEqual(0.5, ClosedUnitInterval(0.5))
        self.assertRaises(ValueError, lambda: ClosedUnitInterval(-0.1))
        self.assertRaises(ValueError, lambda: ClosedUnitInterval('1.1'))


    def test_OpenUnitInterval(self):
        self.assertEqual(0.1, OpenUnitInterval(0.1))
        self.assertEqual(0.9, OpenUnitInterval('0.9'))
        self.assertEqual(0.5, OpenUnitInterval(0.5))
        self.assertRaises(ValueError, lambda: OpenUnitInterval(0))
        self.assertRaises(ValueError, lambda: OpenUnitInterval('1.0'))


    def test_invalid_restricted_number_type(self):
        self.assertRaises(ValueError, lambda: restricted_number_type('Invalid', str, ('<', 0)))
        self.assertRaises(ValueError, lambda: restricted_number_type('Invalid', int, ('<', 0), join='xor'))
        self.assertRaises(ValueError, lambda: restricted_number_type('Invalid', int, ['<', 0]))


    def test_already_registered(self):
        NewClosedUnitInterval = restricted_number_type('ClosedUnitInterval', float, [('<=', 1), ('>=', 0)])
        self.assertEqual(ClosedUnitInterval, NewClosedUnitInterval)
        self.assertRaises(ValueError, lambda: restricted_number_type('NewName', float, [('<=', 1), ('>=', 0)]))


    def test_other_operators(self):
        NotTwoOrThree = restricted_number_type('NotTwoOrThree', float, [('!=', 2), ('!=', 3)])
        self.assertEqual(1.0, NotTwoOrThree(1))
        self.assertRaises(ValueError, lambda: NotTwoOrThree(2))
        self.assertRaises(ValueError, lambda: NotTwoOrThree('3'))
        PositiveOrMinusOne = restricted_number_type('PositiveOrMinusOne', float, [('>', 0), ('==', -1)], join='or')
        self.assertEqual(1.0, PositiveOrMinusOne(1))
        self.assertEqual(-1.0, PositiveOrMinusOne('-1.0'))
        self.assertRaises(ValueError, lambda: PositiveOrMinusOne(-0.5))
        self.assertRaises(ValueError, lambda: PositiveOrMinusOne('-2'))


    def test_add_argument_type(self):
        TenToTwenty = restricted_number_type('TenToTwenty', int, [('>=', 10), ('<=', 20)])

        def gt0_or_off(x):
            return x if x == 'off' else PositiveInt(x)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--le0', type=NonNegativeFloat)
        parser.add_argument('--f10t20', type=TenToTwenty, nargs='+')
        parser.add_argument('--gt0_or_off', type=gt0_or_off)
        parser.add_argument('--multi_gt0_or_off', type=gt0_or_off, nargs='+')

        self.assertEqual(0.0, parser.parse_args(['--le0', '0']).le0)
        self.assertEqual(5.6, parser.parse_args(['--le0', '5.6']).le0)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--le0', '-2.1']))

        self.assertEqual([11, 14, 16], parser.parse_args(['--f10t20', '11', '14', '16']).f10t20)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--f10t20', '9']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--f10t20', '21']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--f10t20', '10.5']))

        self.assertEqual(1, parser.parse_args(['--gt0_or_off', '1']).gt0_or_off)
        self.assertEqual('off', parser.parse_args(['--gt0_or_off', 'off']).gt0_or_off)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt0_or_off', '0']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--gt0_or_off', 'on']))

        self.assertEqual([1, 'off'], parser.parse_args(['--multi_gt0_or_off', '1', 'off']).multi_gt0_or_off)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--multi_gt0_or_off', '1', '0']))
        self.assertRaises(ParserError, lambda: parser.parse_object({'multi_gt0_or_off': [1, 0]}))


class RestrictedStringTests(unittest.TestCase):

    def test_Email(self):
        self.assertEqual('name@eg.org', Email('name@eg.org'))
        self.assertRaises(ValueError, lambda: Email(''))
        self.assertRaises(ValueError, lambda: Email('name @ eg.org'))
        self.assertRaises(ValueError, lambda: Email('name_at_eg.org'))


    def test_already_registered(self):
        NewEmail = restricted_string_type('Email', r'^[^@ ]+@[^@ ]+\.[^@ ]+$')
        self.assertEqual(Email, NewEmail)
        self.assertRaises(ValueError, lambda: restricted_string_type('NewName', r'^[^@ ]+@[^@ ]+\.[^@ ]+$'))


    def test_add_argument_type(self):
        FourDigits = restricted_string_type('FourDigits', '^[0-9]{4}$')
        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--op', type=FourDigits)
        self.assertEqual('1234', parser.parse_args(['--op', '1234']).op)
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op', '123']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op', '12345']))
        self.assertRaises(ParserError, lambda: parser.parse_args(['--op', 'abcd']))


class PathTypeTests(TempDirTestCase):

    def setUp(self):
        super().setUp()
        self.file_fr = 'file_r'
        pathlib.Path(self.file_fr).touch()


    def test_Path_fr(self):
        path = Path_fr(self.file_fr)
        self.assertEqual(path, self.file_fr)
        self.assertEqual(path(), os.path.realpath(self.file_fr))
        self.assertRaises(TypeError, lambda: Path_fr('does_not_exist'))


    def test_already_registered(self):
        NewPath_fr = path_type('fr')
        self.assertEqual(Path_fr, NewPath_fr)


    def test_path_like(self):
        parser = ArgumentParser()
        parser.add_argument('--path', type=os.PathLike)
        self.assertEqual(self.file_fr, parser.parse_args([f'--path={self.file_fr}']).path)


class OtherTests(unittest.TestCase):

    def test_pickle_module_types(self):
        for otype in registered_types.values():
            if isinstance(otype, RegisteredType) or hasattr(__import__('jsonargparse.typing'), otype.__name__):
                if isinstance(otype, RegisteredType):
                    otype = otype.type_class
                with self.subTest(str(otype)):
                    utype = pickle.loads(pickle.dumps(otype))
                    self.assertEqual(otype, utype)


    def test_name_clash(self):
        self.assertRaises(ValueError, lambda: restricted_string_type('List', '^clash$'))


    def test_serialize_class_method_path(self):
        class MyClass:
            @staticmethod
            def my_method1():
                pass
            def my_method2(self):
                pass

        with mock_module(MyClass) as module:
            self.assertEqual(object_path_serializer(MyClass.my_method1), f'{module}.MyClass.my_method1')
            self.assertEqual(object_path_serializer(MyClass.my_method2), f'{module}.MyClass.my_method2')


    def test_object_path_serializer_reimport_differs(self):
        class MyClass:
            pass

        with mock_module(MyClass) as module:
            class FakeMyClass:
                pass

            FakeMyClass.__module__ = module
            FakeMyClass.__qualname__ = MyClass.__qualname__
            self.assertRaises(ValueError, lambda: object_path_serializer(FakeMyClass))


    def test_timedelta(self):
        timedelta_type = registered_types[timedelta]
        for delta_in, delta_out in [
            ('1:2:3', '1:02:03'),
            ('0:05:30', '0:05:30'),
            ('3 days, 2:0:0', '3 days, 2:00:00'),
            ('345:0:0', '14 days, 9:00:00'),
        ]:
            with self.subTest(delta_in):
                delta = timedelta_type.deserializer(delta_in)
                self.assertIsInstance(delta, timedelta)
                self.assertEqual(str(delta), delta_out)
        for delta_in in ['not delta', 1234]:
            with self.subTest(delta_in):
                self.assertRaises(ValueError, lambda: timedelta_type.deserializer(delta_in))


if __name__ == '__main__':
    unittest.main(verbosity=2)
