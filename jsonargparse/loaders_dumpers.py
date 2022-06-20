"""Code related to loading and dumping."""

import inspect
import json
import re
import yaml
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Tuple, Type

from .optionals import dump_preserve_order_support, import_jsonnet
from .type_checking import ArgumentParser


__all__ = [
    'set_loader',
    'set_dumper',
]


load_value_mode: ContextVar = ContextVar('load_value_mode')
regex_curly_comma = re.compile(' *[{},] *')


@contextmanager
def load_value_context(mode):
    t = load_value_mode.set(mode)
    try:
        yield
    finally:
        load_value_mode.reset(t)


class DefaultLoader(getattr(yaml, 'CSafeLoader', yaml.SafeLoader)):  # type: ignore
    pass


# https://stackoverflow.com/a/37958106/2732151
def remove_implicit_resolver(cls, tag_to_remove):
    if 'yaml_implicit_resolvers' not in cls.__dict__:
        cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

    for first_letter, mappings in cls.yaml_implicit_resolvers.items():
        cls.yaml_implicit_resolvers[first_letter] = [
            (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
        ]


remove_implicit_resolver(DefaultLoader, 'tag:yaml.org,2002:timestamp')
remove_implicit_resolver(DefaultLoader, 'tag:yaml.org,2002:float')


DefaultLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile('''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list('-+0123456789.'),
)


def yaml_load(stream):
    if stream.strip() == '-':
        value = stream
    else:
        value = yaml.load(stream, Loader=DefaultLoader)
    if isinstance(value, dict) and all(v is None for v in value.values()):
        keys = {k for k in regex_curly_comma.split(stream) if k}
        if len(keys) > 0 and keys == set(value.keys()):
            value = stream
    return value


def jsonnet_load(stream, path='', ext_vars=None):
    from .jsonnet import ActionJsonnet
    ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
    _jsonnet = import_jsonnet('jsonnet_load')
    try:
        val = _jsonnet.evaluate_snippet(path, stream, ext_vars=ext_vars, ext_codes=ext_codes)
    except RuntimeError as ex:
        try:
            return yaml_load(stream)
        except pyyaml_exceptions:
            raise ValueError(str(ex)) from ex
    return yaml_load(val)


loaders: Dict[str, Callable] = {
    'yaml': yaml_load,
    'jsonnet': jsonnet_load,
}

pyyaml_exceptions = (yaml.parser.ParserError, yaml.scanner.ScannerError)
jsonnet_exceptions = pyyaml_exceptions + (ValueError,)

loader_exceptions: Dict[str, Tuple[Type[Exception], ...]] = {
    'yaml': pyyaml_exceptions,
    'jsonnet': jsonnet_exceptions,
}


def get_loader_exceptions():
    return loader_exceptions[load_value_mode.get()]


def load_value(value: str, **kwargs):
    loader = loaders[load_value_mode.get()]
    if kwargs:
        params = set(list(inspect.signature(loader).parameters)[1:])
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return loader(value, **kwargs)


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
    'jsonnet': json_indented_dump,
}

comment_prefix: Dict[str,str] = {
    'yaml': '# ',
    'yaml_comments': '# ',
    'jsonnet': '// ',
}


def check_valid_dump_format(dump_format: str):
    if dump_format not in {'parser_mode'}.union(set(dumpers.keys())):
        raise ValueError(f'Unknown output format "{dump_format}".')


def dump_using_format(parser: 'ArgumentParser', data: dict, dump_format: str) -> str:
    if dump_format == 'parser_mode':
        dump_format = parser.parser_mode if parser.parser_mode in dumpers else 'yaml'
    args = (data, parser) if dump_format == 'yaml_comments' else (data,)
    dump = dumpers[dump_format](*args)
    if parser.dump_header and comment_prefix.get(dump_format):
        prefix = comment_prefix[dump_format]
        header = '\n'.join(prefix + line for line in parser.dump_header)
        dump = f'{header}\n{dump}'
    return dump


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
