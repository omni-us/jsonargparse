"""Code related to optional dependencies."""

import locale
import os
import platform
from contextlib import contextmanager
from importlib.util import find_spec


__all__ = [
    'get_config_read_mode',
    'set_config_read_mode',
]


_jsonschema = jsonvalidator = find_spec('jsonschema')
_jsonnet = find_spec('_jsonnet')
_url_validator = find_spec('validators')
_requests = find_spec('requests')
_docstring_parser = find_spec('docstring_parser')
_argcomplete = find_spec('argcomplete')
_dataclasses = find_spec('dataclasses')
_fsspec = find_spec('fsspec')
_ruyaml = find_spec('ruyaml')
_omegaconf = find_spec('omegaconf')

jsonschema_support = False if _jsonschema is None else True
jsonnet_support = False if _jsonnet is None else True
url_support = False if any(x is None for x in [_url_validator, _requests]) else True
docstring_parser_support = False if _docstring_parser is None else True
argcomplete_support = False if _argcomplete is None else True
dataclasses_support = False if _dataclasses is None else True
fsspec_support = False if _fsspec is None else True
ruyaml_support = False if _ruyaml is None else True
omegaconf_support = False if _omegaconf is None else True

_config_read_mode = 'fr'


class UndefinedException(Exception):
    pass


if jsonschema_support:
    from jsonschema.exceptions import ValidationError as jsonschemaValidationError
else:
    jsonschemaValidationError = UndefinedException


dump_preserve_order_support = True
if platform.python_implementation() != 'CPython':
    dump_preserve_order_support = False


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


def import_docstring_parse(importer):
    with missing_package_raise('docstring-parser', importer):
        from docstring_parser import parse as docstring_parse
    return docstring_parse


def import_argcomplete(importer):
    with missing_package_raise('argcomplete', importer):
        import argcomplete
    return argcomplete


def import_dataclasses(importer):
    with missing_package_raise('dataclasses', importer):
        import dataclasses
    return dataclasses


def import_fsspec(importer):
    with missing_package_raise('fsspec', importer):
        import fsspec
    return fsspec


def import_ruyaml(importer):
    with missing_package_raise('ruyaml', importer):
        import ruyaml
    return ruyaml


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


files_completer = None
if argcomplete_support:
    from argcomplete.completers import FilesCompleter
    files_completer = FilesCompleter()


class FilesCompleterMethod:
    """Completer method for Action classes that should complete files."""
    def completer(self, prefix, **kwargs):
        return sorted(files_completer(prefix, **kwargs))


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
