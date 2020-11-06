"""Functions related to optional dependencies."""

from importlib.util import find_spec


_jsonschema = jsonvalidator = find_spec('jsonschema')
_jsonnet = find_spec('_jsonnet')
_url_validator = find_spec('validators')
_requests = find_spec('requests')
_docstring_parser = find_spec('docstring_parser')
_argcomplete = find_spec('argcomplete')
_dataclasses = find_spec('dataclasses')

jsonschema_support = False if _jsonschema is None else True
jsonnet_support = False if any(x is None for x in [_jsonnet, _jsonschema]) else True
url_support = False if any(x is None for x in [_url_validator, _requests]) else True
docstring_parser_support = False if _docstring_parser is None else True
argcomplete_support = False if _argcomplete is None else True
dataclasses_support = False if _dataclasses is None else True

_config_read_mode = 'fr'


def _import_jsonschema(importer):
    try:
        import jsonschema
        from jsonschema import Draft7Validator as jsonvalidator
        return jsonschema, jsonvalidator
    except Exception as ex:
        raise ImportError('jsonschema package is required by '+importer+' :: '+str(ex))


def _import_jsonnet(importer):
    try:
        import _jsonnet
        return _jsonnet
    except Exception as ex:
        raise ImportError('jsonnet package is required by '+importer+' :: '+str(ex))


def _import_url_validator(importer):
    try:
        from validators.url import url as url_validator
        return url_validator
    except Exception as ex:
        raise ImportError('validators package is required by '+importer+' :: '+str(ex))


def _import_requests(importer):
    try:
        import requests
        return requests
    except Exception as ex:
        raise ImportError('requests package is required by '+importer+' :: '+str(ex))


def _import_docstring_parse(importer):
    try:
        from docstring_parser import parse as docstring_parse
        return docstring_parse
    except Exception as ex:
        raise ImportError('docstring-parser package is required by '+importer+' :: '+str(ex))


def _import_argcomplete(importer):
    try:
        import argcomplete
        return argcomplete
    except Exception as ex:
        raise ImportError('argcomplete package is required by '+importer+' :: '+str(ex))


def _import_dataclasses(importer):
    try:
        import dataclasses
        return dataclasses
    except Exception as ex:
        raise ImportError('dataclasses package is required by '+importer+' :: '+str(ex))


def set_url_support(enabled):
    """Enables/disables URL support for config read mode."""
    if enabled and not url_support:
        pkg = ['validators', 'requests']
        missing = {pkg[n] for n, x in enumerate([_url_validator, _requests]) if x is None}
        raise ImportError('Missing packages for URL support: '+str(missing))
    global _config_read_mode
    _config_read_mode = 'fur' if enabled else 'fr'


def get_config_read_mode():
    """Returns the current config reading mode."""
    return _config_read_mode
