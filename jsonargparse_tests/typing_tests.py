#!/usr/bin/env python3

import unittest
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
