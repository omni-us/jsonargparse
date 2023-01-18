#!/usr/bin/env python3

import inspect
import sys
import unittest
from calendar import Calendar, TextCalendar
from contextlib import contextmanager
from email.headerregistry import DateHeader
from importlib.util import find_spec
from io import StringIO
from ipaddress import ip_network
from random import Random, SystemRandom, uniform
from tarfile import TarFile
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid5

import yaml

from jsonargparse import ArgumentParser
from jsonargparse._stubs_resolver import get_mro_method_parent, get_stubs_resolver
from jsonargparse.parameter_resolvers import get_signature_parameters as get_params
from jsonargparse_tests.base import get_debug_level_logger

logger = get_debug_level_logger(__name__)
torch_available = find_spec('torch')


@contextmanager
def mock_typeshed_client_unavailable():
    with patch('jsonargparse.parameter_resolvers.add_stub_types'):
        yield


def get_param_types(params):
    return [(p.name, p.annotation) for p in params]


def get_param_names(params):
    return [p.name for p in params]


@unittest.skipIf(not find_spec('typeshed_client'), 'typeshed-client package is required')
class StubsResolverTests(unittest.TestCase):

    def test_get_mro_method_parent(self):
        class WithoutParent:
            ...

        self.assertIs(None, get_mro_method_parent(WithoutParent, '__init__'))
        self.assertIs(None, get_mro_method_parent(SystemRandom, 'unknown'))
        self.assertIs(Random, get_mro_method_parent(Random, '__init__'))
        self.assertIs(Random, get_mro_method_parent(SystemRandom, '__init__'))
        self.assertIs(Random, get_mro_method_parent(SystemRandom, 'uniform'))
        self.assertIs(SystemRandom, get_mro_method_parent(SystemRandom, 'getrandbits'))

    @unittest.skipIf(not find_spec('requests'), 'requests package is required')
    def test_stubs_resolver_get_imported_info(self):
        resolver = get_stubs_resolver()
        imported_info = resolver.get_imported_info('requests.api.get')
        self.assertEqual(imported_info.source_module, ('requests', 'api'))
        imported_info = resolver.get_imported_info('requests.get')
        self.assertEqual(imported_info.source_module, ('requests', 'api'))

    def test_get_params_class_without_inheritance(self):
        params = get_params(Calendar)
        self.assertEqual([('firstweekday', int)], get_param_types(params))
        with mock_typeshed_client_unavailable():
            params = get_params(Calendar)
        self.assertEqual([('firstweekday', inspect._empty)], get_param_types(params))

    def test_get_params_class_with_inheritance(self):
        params = get_params(TextCalendar)
        self.assertEqual([('firstweekday', int)], get_param_types(params))
        with mock_typeshed_client_unavailable():
            params = get_params(TextCalendar)
        self.assertEqual([('firstweekday', inspect._empty)], get_param_types(params))

    def test_get_params_method(self):
        params = get_params(Random, 'randint')
        self.assertEqual([('a', int), ('b', int)], get_param_types(params))
        with mock_typeshed_client_unavailable():
            params = get_params(Random, 'randint')
        self.assertEqual([('a', inspect._empty), ('b', inspect._empty)], get_param_types(params))

    def test_get_params_object_instance_method(self):
        params = get_params(uniform)
        self.assertEqual([('a', float), ('b', float)], get_param_types(params))
        with mock_typeshed_client_unavailable():
            params = get_params(uniform)
        self.assertEqual([('a', inspect._empty), ('b', inspect._empty)], get_param_types(params))

    def test_get_params_conditional_python_version(self):
        params = get_params(Random, 'seed')
        self.assertEqual(['a', 'version'], get_param_names(params))
        if sys.version_info >= (3, 10):
            self.assertEqual('int | float | str | bytes | bytearray | None', str(params[0].annotation))
        else:
            expected = Any if sys.version_info < (3, 9) else inspect._empty
            self.assertEqual(expected, params[0].annotation)
        self.assertEqual(int, params[1].annotation)
        with mock_typeshed_client_unavailable():
            params = get_params(Random, 'seed')
        self.assertEqual([('a', inspect._empty), ('version', inspect._empty)], get_param_types(params))

    @patch('jsonargparse._stubs_resolver.exec')
    def test_get_params_exec_failure(self, mock_exec):
        mock_exec.side_effect = NameError('failed')
        params = get_params(Random, 'seed')
        self.assertEqual([('a', inspect._empty), ('version', inspect._empty)], get_param_types(params))

    def test_get_params_classmethod(self):
        params = get_params(TarFile, 'open')
        expected = ['name', 'mode', 'fileobj', 'bufsize', 'format', 'tarinfo', 'dereference', 'ignore_zeros', 'encoding', 'errors', 'pax_headers', 'debug', 'errorlevel']
        self.assertEqual(expected, get_param_names(params)[:len(expected)])
        if sys.version_info >= (3, 10):
            self.assertTrue(all(p.annotation != inspect._empty for p in params))
        with mock_typeshed_client_unavailable():
            params = get_params(TarFile, 'open')
        self.assertEqual(expected, get_param_names(params)[:len(expected)])
        self.assertTrue(all(p.annotation == inspect._empty for p in params))

    def test_get_params_staticmethod(self):
        params = get_params(DateHeader, 'value_parser')
        self.assertEqual([('value', str)], get_param_types(params))
        with mock_typeshed_client_unavailable():
            params = get_params(DateHeader, 'value_parser')
        self.assertEqual([('value', inspect._empty)], get_param_types(params))

    def test_get_params_function(self):
        params = get_params(ip_network)
        self.assertEqual(['address', 'strict'], get_param_names(params))
        if sys.version_info >= (3, 10):
            self.assertIn('int | str | bytes | ipaddress.IPv4Address | ', str(params[0].annotation))
        self.assertEqual(bool, params[1].annotation)
        with mock_typeshed_client_unavailable():
            params = get_params(ip_network)
        self.assertEqual([('address', inspect._empty), ('strict', inspect._empty)], get_param_types(params))

    def test_get_param_relative_import_from_init(self):
        params = get_params(yaml.safe_load)
        self.assertEqual(['stream'], get_param_names(params))
        if sys.version_info >= (3, 10):
            self.assertNotEqual(params[0].annotation, inspect._empty)
        else:
            self.assertEqual(params[0].annotation, inspect._empty)
        with mock_typeshed_client_unavailable():
            params = get_params(yaml.safe_load)
        self.assertEqual(['stream'], get_param_names(params))
        self.assertEqual(params[0].annotation, inspect._empty)

    def test_get_params_non_unique_alias(self):
        params = get_params(uuid5)
        self.assertEqual([('namespace', UUID), ('name', str)], get_param_types(params))

        def alias_is_unique(aliases, name, source, value):
            if name == 'UUID':
                aliases[name] = ('module', 'problem')
            return name != 'UUID'

        with patch('jsonargparse._stubs_resolver.alias_is_unique', alias_is_unique):
            with self.assertLogs(logger, level='DEBUG') as log:
                params = get_params(uuid5, logger=logger)
                self.assertEqual([('namespace', inspect._empty), ('name', str)], get_param_types(params))
                self.assertIn("non-unique alias 'UUID': problem (module)", log.output[0])

    @unittest.skipIf(not find_spec('requests'), 'requests package is required')
    def test_get_params_complex_function_requests_get(self):
        from requests import get
        with mock_typeshed_client_unavailable():
            params = get_params(get)
        expected = ['url', 'params']
        self.assertEqual(expected, get_param_names(params))
        self.assertTrue(all(p.annotation == inspect._empty for p in params))

        params = get_params(get)
        expected += ['data', 'headers', 'cookies', 'files', 'auth', 'timeout', 'allow_redirects', 'proxies', 'hooks', 'stream', 'verify', 'cert', 'json']
        self.assertEqual(expected, get_param_names(params))
        if sys.version_info >= (3, 10):
            self.assertTrue(all(p.annotation != inspect._empty for p in params))

        parser = ArgumentParser(exit_on_error=False)
        parser.add_function_arguments(get, fail_untyped=False)
        self.assertEqual(['url', 'params'], list(parser.get_defaults().keys()))
        help_str = StringIO()
        parser.print_help(help_str)
        self.assertIn('default: Unknown<stubs-resolver>', help_str.getvalue())

    @unittest.skipIf(not torch_available, 'torch package is required')
    def test_get_params_torch_optimizer(self):
        import torch.optim  # pylint: disable=import-error

        def skip_stub_inconsistencies(cls, params):
            # https://github.com/pytorch/pytorch/pull/90216
            skip = {('Optimizer', 'defaults'), ('SGD', 'maximize'), ('SGD', 'differentiable')}
            return [p for p in params if (cls.__name__, p.name) not in skip]

        for class_name in [
            'Optimizer',
            'Adadelta',
            'Adagrad',
            'Adam',
            'Adamax',
            'AdamW',
            'ASGD',
            'LBFGS',
            'NAdam',
            'RAdam',
            'RMSprop',
            'Rprop',
            'SGD',
            'SparseAdam',
        ]:
            with self.subTest(class_name):
                cls = getattr(torch.optim, class_name)
                params = get_params(cls)
                self.assertTrue(all(p.annotation != inspect._empty for p in skip_stub_inconsistencies(cls, params)))
                with mock_typeshed_client_unavailable():
                    params = get_params(cls)
                self.assertTrue(any(p.annotation == inspect._empty for p in params))

    @unittest.skipIf(not torch_available, 'torch package is required')
    def test_get_params_torch_lr_scheduler(self):
        import torch.optim.lr_scheduler  # pylint: disable=import-error

        for class_name in [
            '_LRScheduler',
            'LambdaLR',
            'MultiplicativeLR',
            'StepLR',
            'MultiStepLR',
            'ConstantLR',
            'LinearLR',
            'ExponentialLR',
            'ChainedScheduler',
            'SequentialLR',
            'CosineAnnealingLR',
            'ReduceLROnPlateau',
            'CyclicLR',
            'CosineAnnealingWarmRestarts',
            'OneCycleLR',
            'PolynomialLR',
        ]:
            with self.subTest(class_name):
                cls = getattr(torch.optim.lr_scheduler, class_name)
                params = get_params(cls)
                self.assertTrue(all(p.annotation != inspect._empty for p in params))
                with mock_typeshed_client_unavailable():
                    params = get_params(cls)
                self.assertTrue(any(p.annotation == inspect._empty for p in params))


if __name__ == '__main__':
    unittest.main(verbosity=2)
