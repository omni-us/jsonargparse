#!/usr/bin/env python3

import logging
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from jsonargparse import class_from_function, Namespace
from jsonargparse.ast_analysis import get_parameters
from jsonargparse.optionals import docstring_parser_support


class ClassA:
    def __init__(self, ka1: float = 1.2, ka2: bool = False, **kwargs):
        """
        Args:
            ka1: help for ka1
            ka2: help for ka2
        """

    def method_a(self, pma1: int, pma2: float, kma1: str = 'x'):
        """
        Args:
            pma1: help for pma1
            pma2: help for pma2
            kma1: help for kma1
        """

class ClassB(ClassA):
    def __init__(self, pkb1: str, kb1: int = 3, kb2: str = '4', **kwargs):
        """
        Args:
            pkb1: help for pkb1
            kb1: help for kb1
            kb2: help for kb2
        """
        super().__init__(ka2=True, **kwargs)

    @classmethod
    def make(cls, pkcm1: str, kcm1: bool = False, **kws):
        """
        Args:
            pkcm1: help for pkcm1
            kcm1: help for kcm1
        """
        return ClassB(pkcm1, **kws)

class ClassC(ClassB):
    def __init__(self, kc1: str = '-', **kargs):
        """
        Args:
            kc1: help for kc1
        """
        super().__init__(**kargs)

class Param:
    p1: int = 2

class ClassD(Param, ClassA):
    def __init__(self, kd1: bool = False, **kwargs):
        """
        Args:
            kd1: help for kd1
        """
        super().__init__(**kwargs)

    def method_d(self, pmd1: int, *args, kmd1: int = 2, **kws):
        """
        Args:
            pmd1: help for pmd1
            kmd1: help for kmd1
        """
        return super().method_a(*args, **kws)

    @staticmethod
    def staticmethod_d(ksmd1: str = 'z', **kw):
        """
        Args:
            ksmd1: help for ksmd1
        """
        return function_return_class_c(ksmd1, k2=2, **kw)

class ClassE:
    """
    Args:
        ke1: help for ke1
    """
    def __init__(self, ke1: int = 1, **kwargs):
        self._kwd = dict(k2=3, **kwargs)

    def start(self):
        return function_no_args_no_kwargs(**self._kwd)

class ClassF:
    def __init__(self, **kw):
        self._ini = dict(k2=4)
        self._ini.update(**kw)

    def _run(self):
        self.staticmethod_f(**self._ini)

    @staticmethod
    def staticmethod_f(ksmf1: str = 'w', **kw):
        """
        Args:
            ksmf1: help for ksmf1
        """
        return function_no_args_no_kwargs(**kw)

class ClassG:
    def __init__(self, func: str, **kws):
        """
        Args:
            func: help for func
        """
        self.func = func
        self.kws = kws

    def _run(self):
        if self.func == '1':
            self.method1(**self.kws)
        elif self.func == '2':
            self.method2(**self.kws)

    def method1(self, kmg1: int, kmg2: str, kmg3: bool):
        """
        Args:
            kmg1: help for kmg1
            kmg3: help for kmg3
        """

    def method2(self, kmg1: int, kmg2: float, kmg3: bool, kmg4: int):
        """
        Args:
            kmg1: help for kmg1
            kmg3: help for kmg3
        """

class ClassU1:
    def __init__(self, k1: int = 1, **ka):
        data = Namespace()
        data.ka = ka

class ClassU2:
    def __init__(self, k1: int = 1, **ka):
        self.method_u2(ka=ka)

    def method_u2(self, ka: dict):
        pass

class ClassU3(ClassU1, ClassU2):
    def __init__(self, **ka):
        super(ClassU2, self).__init__(**ka)  # pylint: disable=bad-super-call

class ClassU4:
    def __init__(self, k1: int = 1, **ka):
        self._ka = ka

class ClassU5:
    def __init__(self, **kws):
        self.kws = kws

    def _run(self):
        self.method1(kws=self.kws)

    def method1(self, kws: dict):
        pass

def function_no_args_no_kwargs(pk1: str, k2: int = 1):
    """
    Args:
        pk1: help for pk1
        k2: help for k2
    """

def function_return_class_c(pk1: str, k2: int = 1, **ka):
    """
    Args:
        pk1: help for pk1
        k2: help for k2
    """
    return ClassC(pk1, kb1=k2, **ka)

def function_make_class_b(*args, k1: str = '-', **kwargs):
    """
    Args:
        k1: help for k1
    """
    return ClassB.make(*args, **kwargs)

def function_with_bug(**kws):
    return does_not_exist(**kws)  # pylint: disable=undefined-variable


@contextmanager
def source_unavailable():
    with patch('inspect.getsource', side_effect=OSError('could not get source code')):
        yield


def assert_params(self, params, expected):
    self.assertEqual(expected, [p.name for p in params])
    docs = [f'help for {p.name}' for p in params] if docstring_parser_support else [None] * len(params)
    self.assertEqual(docs, [p.doc for p in params])


logger = logging.getLogger('ast_analysis_tests')
logger.level = logging.DEBUG


class GetClassParametersTests(unittest.TestCase):

    def test_get_parameters_class_no_inheritance_unused_kwargs(self):
        params = get_parameters(ClassA)
        assert_params(self, params, ['ka1', 'ka2'])
        with source_unavailable():
            self.assertEqual(params, get_parameters(ClassA))

    def test_get_parameters_class_with_inheritance_hard_coded_kwargs(self):
        assert_params(self, get_parameters(ClassB), ['pkb1', 'kb1', 'kb2', 'ka1'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassB), ['pkb1', 'kb1', 'kb2', 'ka1', 'ka2'])

    def test_get_parameters_class_with_inheritance_unused_args(self):
        assert_params(self, get_parameters(ClassC), ['kc1', 'pkb1', 'kb1', 'kb2', 'ka1'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassC), ['kc1', 'pkb1', 'kb1', 'kb2', 'ka1', 'ka2'])

    def test_get_parameters_class_with_inheritance_parent_without_init(self):
        params = get_parameters(ClassD)
        assert_params(self, params, ['kd1', 'ka1', 'ka2'])
        with source_unavailable():
            self.assertEqual(params, get_parameters(ClassD))

    def test_get_parameters_class_with_kwargs_in_dict_attribute(self):
        assert_params(self, get_parameters(ClassE), ['ke1', 'pk1', 'k2'])
        assert_params(self, get_parameters(ClassF), ['ksmf1', 'pk1', 'k2'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassE), ['ke1'])
            assert_params(self, get_parameters(ClassF), [])

    def test_get_parameters_class_kwargs_in_attr_method_conditioned_on_arg(self):
        assert_params(self, get_parameters(ClassG), ['func', 'kmg1', 'kmg3'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassG), ['func'])

    def test_class_from_function(self):
        class_a = class_from_function(function_return_class_c)
        params = get_parameters(class_a)
        assert_params(self, params, ['pk1', 'k2', 'pkb1', 'kb2', 'ka1'])
        with source_unavailable():
            params = get_parameters(class_a)
            assert_params(self, params, ['pk1', 'k2'])


class GetMethodParametersTests(unittest.TestCase):

    def test_get_parameters_method_no_args_no_kwargs(self):
        params = get_parameters(ClassA, 'method_a')
        assert_params(self, params, ['pma1', 'pma2', 'kma1'])
        with source_unavailable():
            self.assertEqual(params, get_parameters(ClassA, 'method_a'))

    def test_get_parameters_method_call_super_method(self):
        assert_params(self, get_parameters(ClassD, 'method_d'), ['pmd1', 'kmd1', 'pma1', 'pma2', 'kma1'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassD, 'method_d'), ['pmd1', 'kmd1'])

    def test_get_parameters_staticmethod_call_function_return_class_c(self):
        params = get_parameters(ClassD.staticmethod_d)
        self.assertEqual(params, get_parameters(ClassD, 'staticmethod_d'))
        assert_params(self, params, ['ksmd1', 'pkb1', 'kb2', 'ka1'])
        with source_unavailable():
            params = get_parameters(ClassD.staticmethod_d)
            self.assertEqual(params, get_parameters(ClassD, 'staticmethod_d'))
            assert_params(self, params, ['ksmd1'])

    def test_get_parameters_classmethod_make_class(self):
        assert_params(self, get_parameters(ClassB.make), ['pkcm1', 'kcm1', 'kb1', 'kb2', 'ka1'])
        with source_unavailable():
            assert_params(self, get_parameters(ClassB.make), ['pkcm1', 'kcm1'])


class GetFunctionParametersTests(unittest.TestCase):

    def test_get_parameters_function_no_args_no_kwargs(self):
        params = get_parameters(function_no_args_no_kwargs)
        self.assertEqual(['pk1', 'k2'], [p.name for p in params])
        with source_unavailable():
            self.assertEqual(params, get_parameters(function_no_args_no_kwargs))

    def test_get_parameters_function_return_class_c(self):
        assert_params(self, get_parameters(function_return_class_c), ['pk1', 'k2', 'pkb1', 'kb2', 'ka1'])
        with source_unavailable():
            assert_params(self, get_parameters(function_return_class_c), ['pk1', 'k2'])

    def test_get_parameters_function_call_classmethod(self):
        assert_params(self, get_parameters(function_make_class_b), ['k1', 'pkcm1', 'kcm1', 'kb1', 'kb2', 'ka1'])
        with source_unavailable():
            assert_params(self, get_parameters(function_make_class_b), ['k1'])


class OtherTests(unittest.TestCase):

    def test_unsupported_type_of_assign(self):
        with self.assertLogs(logger, level='DEBUG') as log:
            get_parameters(ClassU1, logger=logger)
            self.assertIn('Unsupported type of assign', log.output[0])

    def test_unsupported_kwarg_as_keyword(self):
        with self.assertLogs(logger, level='DEBUG') as log:
            get_parameters(ClassU2, logger=logger)
            self.assertIn('kwargs given as keyword parameter not supported', log.output[0])

    def test_unsupported_super_with_arbitrary_params(self):
        with self.assertLogs(logger, level='DEBUG') as log:
            get_parameters(ClassU3, logger=logger)
            self.assertIn('super with arbitrary parameters not supported', log.output[0])

    def test_unsupported_self_attr_not_found_in_members(self):
        with self.assertLogs(logger, level='DEBUG') as log:
            get_parameters(ClassU4, logger=logger)
            self.assertIn('Did not find use of self._ka in members of', log.output[0])

    def test_unsupported_kwarg_attr_as_keyword(self):
        with self.assertLogs(logger, level='DEBUG') as log:
            get_parameters(ClassU5, logger=logger)
            self.assertIn('kwargs attribute given as keyword parameter not supported', log.output[0])

    def test_get_parameters_failures(self):
        self.assertRaises(ValueError, lambda: get_parameters('invalid'))
        self.assertRaises(ValueError, lambda: get_parameters(Param, 'p1'))
        self.assertRaises(AttributeError, lambda: get_parameters(Param, 'p2'))
        self.assertRaises(Exception, lambda: get_parameters(function_with_bug))


if __name__ == '__main__':
    unittest.main(verbosity=2)
