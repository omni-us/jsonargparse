"""Code related to optional dependencies."""

import inspect
import locale
import os
import typing
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Optional


__all__ = [
    'get_config_read_mode',
    'set_config_read_mode',
    'set_docstring_parse_options',
]


typing_extensions_support = find_spec('typing_extensions') is not None
jsonschema_support = find_spec('jsonschema') is not None
jsonnet_support = find_spec('_jsonnet') is not None
url_support = False if any(find_spec(x) is None for x in ['validators', 'requests']) else True
docstring_parser_support = find_spec('docstring_parser') is not None
argcomplete_support = find_spec('argcomplete') is not None
fsspec_support = find_spec('fsspec') is not None
ruyaml_support = find_spec('ruyaml') is not None
omegaconf_support = find_spec('omegaconf') is not None
reconplogger_support = find_spec('reconplogger') is not None

_config_read_mode = 'fr'
_docstring_parse_options = {
    'style': None,
    'attribute_docstrings': False,
}


def typing_extensions_import(name):
    if typing_extensions_support:
        return getattr(__import__('typing_extensions'), name)
    else:
        return getattr(typing, name, False)


def is_compatible_final(final) -> bool:
    @final
    class FinalClass:
        pass
    return getattr(FinalClass, '__final__', False)


final = typing_extensions_import('final')
if not final or not is_compatible_final(final) or 'SPHINX_BUILD' in os.environ:
    def final(cls):  # pylint: disable=function-redefined
        """Decorator to make a class ``final``, i.e., it shouldn't be subclassed.

        It is the same as ``typing.final`` or an equivalent implementation
        depending on the python version and whether typing-extensions is
        installed.
        """
        setattr(cls, '__final__', True)
        return cls


class UndefinedException(Exception):
    pass


def get_jsonschema_exceptions():
    from jsonschema.exceptions import ValidationError
    return (ValidationError,)


@contextmanager
def missing_package_raise(package, importer):
    try:
        yield None
    except ImportError as ex:
        raise ImportError(f'{package} package is required by {importer} :: {ex}') from ex


def import_jsonschema(importer):
    with missing_package_raise('jsonschema', importer):
        import jsonschema
        from jsonschema import Draft7Validator as jsonvalidator
    return jsonschema, jsonvalidator


def import_jsonnet(importer):
    with missing_package_raise('jsonnet', importer):
        import _jsonnet
    return _jsonnet


def import_url_validator(importer):
    with missing_package_raise('validators', importer):
        from validators.url import url as url_validator
    return url_validator


def import_requests(importer):
    with missing_package_raise('requests', importer):
        import requests
    return requests


def import_docstring_parser(importer):
    with missing_package_raise('docstring-parser', importer):
        import docstring_parser
    return docstring_parser


def import_argcomplete(importer):
    with missing_package_raise('argcomplete', importer):
        import argcomplete
    return argcomplete


def import_fsspec(importer):
    with missing_package_raise('fsspec', importer):
        import fsspec
    return fsspec


def import_ruyaml(importer):
    with missing_package_raise('ruyaml', importer):
        import ruyaml
    return ruyaml


def import_reconplogger(importer):
    with missing_package_raise('reconplogger', importer):
        import reconplogger
    return reconplogger


def set_config_read_mode(
    urls_enabled: bool = False,
    fsspec_enabled: bool = False,
):
    """Enables/disables optional config read modes.

    Args:
        urls_enabled: Whether to read config files from URLs using requests package.
        fsspec_enabled: Whether to read config files from fsspec supported file systems.
    """
    imports = {
        'u': [import_url_validator, import_requests],
        's': [import_fsspec],
    }

    def update_mode(flag, enabled):
        global _config_read_mode
        if enabled:
            for import_func in imports[flag]:
                import_func('set_config_read_mode')
            if flag not in _config_read_mode:
                _config_read_mode = _config_read_mode.replace('f', 'f'+flag)
        else:
            _config_read_mode = _config_read_mode.replace(flag, '')

    update_mode('u', urls_enabled)
    update_mode('s', fsspec_enabled)


def get_config_read_mode() -> str:
    """Returns the current config reading mode."""
    return _config_read_mode


def set_docstring_parse_options(style = None, attribute_docstrings: Optional[bool] = None):
    """Sets options for docstring parsing.

    Args:
        style (docstring_parser.DocstringStyle): The docstring style to expect.
        attribute_docstrings: Whether to parse attribute docstrings (slower).
    """
    global _docstring_parse_options
    dp = import_docstring_parser('set_docstring_parse_options')
    if style is not None:
        if not isinstance(style, dp.DocstringStyle):
            raise ValueError(f'Expected style to be of type {dp.DocstringStyle}.')
        _docstring_parse_options['style'] = style
    if attribute_docstrings is not None:
        if not isinstance(attribute_docstrings, bool):
            raise ValueError('Expected attribute_docstrings to be boolean.')
        _docstring_parse_options['attribute_docstrings'] = attribute_docstrings


def get_docstring_parse_options():
    if _docstring_parse_options['style'] is None:
        dp = import_docstring_parser('get_docstring_parse_options')
        _docstring_parse_options['style'] = dp.DocstringStyle.AUTO
    return _docstring_parse_options


def parse_docstring(component, params=False, logger=None):
    dp = import_docstring_parser('parse_docstring')
    options = get_docstring_parse_options()
    try:
        if params and options['attribute_docstrings']:
            return dp.parse_from_object(component, style=options['style'])
        else:
            return dp.parse(component.__doc__, style=options['style'])
    except (ValueError, dp.ParseError) as ex:
        if logger:
            logger.debug(f'Failed parsing docstring for {component}: {ex}')


def parse_docs(component, parent, logger):
    docs = []
    if docstring_parser_support:
        doc_sources = [component]
        if inspect.isclass(parent) and component.__name__ == '__init__':
            doc_sources = [parent] + doc_sources
        for src in doc_sources:
            doc = parse_docstring(src, params=True, logger=logger)
            if doc:
                docs.append(doc)
    return docs


def get_doc_short_description(function_or_class, method_name=None, logger=None):
    if docstring_parser_support:
        component = function_or_class
        if inspect.isclass(function_or_class):
            if not method_name:
                docstring = parse_docstring(function_or_class, params=False, logger=logger)
                if docstring and docstring.short_description:
                    return docstring.short_description
            component = getattr(function_or_class, method_name or '__init__')
        docstring = parse_docstring(component, params=False, logger=logger)
        if docstring:
            return docstring.short_description


def get_files_completer():
    from argcomplete.completers import FilesCompleter
    return FilesCompleter()


class FilesCompleterMethod:
    """Completer method for Action classes that should complete files."""
    def completer(self, prefix, **kwargs):
        files_completer = get_files_completer()
        return sorted(files_completer(prefix, **kwargs))


def argcomplete_autocomplete(parser):
    if argcomplete_support:
        argcomplete = import_argcomplete('parse_args')
        from .loaders_dumpers import load_value_context
        with load_value_context(parser.parser_mode):
            argcomplete.autocomplete(parser)


def argcomplete_namespace(caller, parser, namespace):
    if caller == 'argcomplete':
        namespace.__class__ = __import__('jsonargparse').Namespace
        namespace = parser.merge_config(parser.get_defaults(skip_check=True), namespace).as_flat()
    return namespace


def argcomplete_warn_redraw_prompt(prefix, message):
    argcomplete = import_argcomplete('argcomplete_warn_redraw_prompt')
    if prefix != '':
        argcomplete.warn(message)
        try:
            shell_pid = int(os.popen('ps -p %d -oppid=' % os.getppid()).read().strip())
            os.kill(shell_pid, 28)
        except ValueError:
            pass
    _ = '_' if locale.getlocale()[1] != 'UTF-8' else '\xa0'
    return [_+message.replace(' ', _), '']


def get_omegaconf_loader():
    """Returns a yaml loader function based on OmegaConf which supports variable interpolation."""
    import io
    from .loaders_dumpers import yaml_load
    with missing_package_raise('omegaconf', 'get_omegaconf_loader'):
        from omegaconf import OmegaConf

    def omegaconf_load(value):
        value_pyyaml = yaml_load(value)
        if isinstance(value_pyyaml, (str, int, float, bool)) or value_pyyaml is None:
            return value_pyyaml
        value_omegaconf = OmegaConf.to_object(OmegaConf.load(io.StringIO(value)))
        str_ref = {k: None for k in [value]}
        return value_pyyaml if value_omegaconf == str_ref else value_omegaconf

    return omegaconf_load
