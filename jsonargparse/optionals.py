"""Code related to optional dependencies."""

import locale
import os
import platform
import sys
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

jsonschema_support = False if _jsonschema is None else True
jsonnet_support = False if _jsonnet is None else True
url_support = False if any(x is None for x in [_url_validator, _requests]) else True
docstring_parser_support = False if _docstring_parser is None else True
argcomplete_support = False if _argcomplete is None else True
dataclasses_support = False if _dataclasses is None else True
fsspec_support = False if _fsspec is None else True
ruyaml_support = False if _ruyaml is None else True

_config_read_mode = 'fr'


if sys.version_info.minor > 5:
    ModuleNotFound = ModuleNotFoundError
else:
    class ModuleNotFound(Exception):  # type: ignore
        pass


if jsonschema_support:
    from jsonschema.exceptions import ValidationError as jsonschemaValidationError
else:
    jsonschemaValidationError = None


dump_preserve_order_support = True
if sys.version_info.minor < 6 or platform.python_implementation() != 'CPython':
    dump_preserve_order_support = False


def import_jsonschema(importer):
    try:
        import jsonschema
        from jsonschema import Draft7Validator as jsonvalidator
        return jsonschema, jsonvalidator
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('jsonschema package is required by '+importer+' :: '+str(ex)) from ex


def import_jsonnet(importer):
    try:
        import _jsonnet
        return _jsonnet
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('jsonnet package is required by '+importer+' :: '+str(ex)) from ex


def import_url_validator(importer):
    try:
        from validators.url import url as url_validator
        return url_validator
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('validators package is required by '+importer+' :: '+str(ex)) from ex


def import_requests(importer):
    try:
        import requests
        return requests
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('requests package is required by '+importer+' :: '+str(ex)) from ex


def import_docstring_parse(importer):
    try:
        from docstring_parser import parse as docstring_parse
        return docstring_parse
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('docstring-parser package is required by '+importer+' :: '+str(ex)) from ex


def import_argcomplete(importer):
    try:
        import argcomplete
        return argcomplete
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('argcomplete package is required by '+importer+' :: '+str(ex)) from ex


def import_dataclasses(importer):
    try:
        import dataclasses
        return dataclasses
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('dataclasses package is required by '+importer+' :: '+str(ex)) from ex


def import_fsspec(importer):
    try:
        import fsspec
        return fsspec
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('fsspec package is required by '+importer+' :: '+str(ex)) from ex


def import_ruyaml(importer):
    try:
        import ruyaml
        return ruyaml
    except (ImportError, ModuleNotFound) as ex:
        raise ImportError('ruyaml package is required by '+importer+' :: '+str(ex)) from ex


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


class TypeCastCompleterMethod:
    """Completer method for Action classes with a casting type."""
    def completer(self, prefix, **kwargs):
        if chr(int(os.environ['COMP_TYPE'])) == '?':
            try:
                self.type(prefix)  # pylint: disable=no-member
                msg = 'value already valid, '
            except ValueError:
                msg = 'value not yet valid, '
            msg += 'expected type '+self.type.__name__  # pylint: disable=no-member
            return argcomplete_warn_redraw_prompt(prefix, msg)


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
