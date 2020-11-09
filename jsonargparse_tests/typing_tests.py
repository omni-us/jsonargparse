#!/usr/bin/env python3

import pickle
import unittest
import jsonargparse.typing
from jsonargparse import ArgumentParser, ParserError
from jsonargparse.typing import (restricted_number_type, PositiveInt, NonNegativeInt, PositiveFloat,
                                 NonNegativeFloat, ClosedUnitInterval, OpenUnitInterval)


class RestrictedNumberTests(unittest.TestCase):

    def test_PositiveInt(self):
        self.assertEqual(1, PositiveInt(1))
        self.assertEqual(2, PositiveInt('2'))
        self.assertRaises(ValueError, lambda: PositiveInt(0))
        self.assertRaises(ValueError, lambda: PositiveInt('-3'))
        self.assertRaises(ValueError, lambda: PositiveInt('4.0'))


    def test_NonNegativeInt(self):
        self.assertEqual(0, NonNegativeInt(0))
        self.assertEqual(1, NonNegativeInt('1'))
        self.assertRaises(ValueError, lambda: NonNegativeInt(-1))
        self.assertRaises(ValueError, lambda: NonNegativeInt('-2'))
        self.assertRaises(ValueError, lambda: NonNegativeInt('3.0'))


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


    def test_invalid_type(self):
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


    def test_pickle_module_types(self):
        for otype in jsonargparse.typing.registered_types.values():
            if hasattr(jsonargparse.typing, otype.__name__):
                with self.subTest(otype.__name__):
                    utype = pickle.loads(pickle.dumps(otype))
                    self.assertEqual(otype, utype)


    def test_add_argument_type(self):
        TenToTwenty = restricted_number_type('TenToTwenty', int, [('>=', 10), ('<=', 20)])

        def gt0_or_off(x):
            return x if x == 'off' else PositiveInt(x)

        parser = ArgumentParser(error_handler=None)
        parser.add_argument('--le0', type=NonNegativeFloat)
        parser.add_argument('--f10t20', type=TenToTwenty, nargs='+')
        parser.add_argument('--gt0_or_off', type=gt0_or_off)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
