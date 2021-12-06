"""Code related to loading and dumping."""

import json
import re
import yaml
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Tuple, Type

from .optionals import dump_preserve_order_support
from .type_checking import ArgumentParser


__all__ = [
    'set_loader',
    'set_dumper',
]


load_value_mode: ContextVar = ContextVar('load_value_mode')


@contextmanager
def load_value_context(mode):
    t = load_value_mode.set(mode)
    try:
        yield
    finally:
        load_value_mode.reset(t)


class DefaultLoader(yaml.SafeLoader):
    pass


DefaultLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'),
)


def yaml_load(stream):
    return yaml.load(stream, Loader=DefaultLoader)


loaders: Dict[str, Callable] = {
    'yaml': yaml_load,
    'jsonnet': yaml_load,
}

pyyaml_exceptions = (yaml.parser.ParserError, yaml.scanner.ScannerError)

loader_exceptions: Dict[str, Tuple[Type[Exception], ...]] = {
    'yaml': pyyaml_exceptions,
    'jsonnet': pyyaml_exceptions,
}


def get_loader_exceptions():
    return loader_exceptions[load_value_mode.get()]


def load_value(value: str):
    return loaders[load_value_mode.get()](value)


dump_yaml_kwargs = {
    'default_flow_style': False,
    'allow_unicode': True,
    'sort_keys': False if dump_preserve_order_support else True,
}

dump_json_kwargs = {
    'ensure_ascii': False,
    'sort_keys': False if dump_preserve_order_support else True,
}


def yaml_dump(data):
    return yaml.safe_dump(data, **dump_yaml_kwargs)


def yaml_comments_dump(data, parser):
    dump = dumpers['yaml'](data)
    formatter = parser.formatter_class(parser.prog)
    return formatter.add_yaml_comments(dump)


def json_dump(data):
    return json.dumps(data, separators=(',', ':'), **dump_json_kwargs)


def json_indented_dump(data):
    return json.dumps(data, indent=2, **dump_json_kwargs)+'\n'


dumpers: Dict[str, Callable] = {
    'yaml': yaml_dump,
    'yaml_comments': yaml_comments_dump,
    'json': json_dump,
    'json_indented': json_indented_dump,
}


def check_valid_dump_format(dump_format: str):
    if dump_format not in {'parser_mode'}.union(set(dumpers.keys())):
        raise ValueError(f'Unknown output format "{dump_format}".')


def dump_using_format(parser: 'ArgumentParser', data: dict, dump_format: str) -> str:
    if dump_format == 'parser_mode':
        dump_format = parser.parser_mode if parser.parser_mode in dumpers else 'yaml'
    args = (data, parser) if dump_format == 'yaml_comments' else (data,)
    return dumpers[dump_format](*args)


def set_loader(mode: str, loader_fn: Callable[[str], Any], exceptions: Tuple[Type[Exception], ...] = pyyaml_exceptions):
    """Sets the value loader function to be used when parsing with a certain mode.

    The ``loader_fn`` function must accept as input a single str type parameter
    and return any of the basic types {str, bool, int, float, list, dict, None}.
    If this function is not based on PyYAML for things to work correctly the
    exceptions types that can be raised when parsing a value fails should be
    provided.

    Args:
        mode: The parser mode for which to set its loader function. Example: "yaml".
        loader_fn: The loader function to set. Example: ``yaml.safe_load``.
        exceptions: Exceptions that the loader can raise when load fails. Example: (yaml.parser.ParserError, yaml.scanner.ScannerError).
    """
    loaders[mode] = loader_fn
    loader_exceptions[mode] = exceptions


def set_dumper(format_name: str, dumper_fn: Callable[[Any], str]):
    """Sets the dumping function for a given format name.

    Args:
        format_name: Name to use for dumping with this function. Example: "yaml_custom".
        dumper_fn: The dumper function to set. Example: ``yaml.safe_dump``.
    """
    dumpers[format_name] = dumper_fn
