"""Code related to optional dependencies."""

import inspect
import os
from contextlib import contextmanager
from importlib.metadata import version
from importlib.util import find_spec
from typing import Optional, Union

__all__ = [
    "_get_config_read_mode",
]


pyyaml_available = bool(find_spec("yaml"))
toml_load_available = bool(find_spec("toml") or find_spec("tomllib"))
toml_dump_available = bool(find_spec("toml"))
typing_extensions_support = find_spec("typing_extensions") is not None
typeshed_client_support = find_spec("typeshed_client") is not None
jsonschema_support = find_spec("jsonschema") is not None
jsonnet_support = find_spec("_jsonnet") is not None
url_support = find_spec("requests") is not None
docstring_parser_support = find_spec("docstring_parser") is not None
fsspec_support = find_spec("fsspec") is not None
ruyaml_support = find_spec("ruyaml") is not None
omegaconf_support = find_spec("omegaconf") is not None
reconplogger_support = find_spec("reconplogger") is not None
attrs_support = find_spec("attrs") is not None

_config_read_mode = "fr"
_docstring_parse_options = {
    "style": None,
    "attribute_docstrings": False,
}


def typing_extensions_import(name):
    if typing_extensions_support:
        return getattr(__import__("typing_extensions"), name, False)
    else:
        return getattr(__import__("typing"), name, False)


def capture_typing_extension_shadows(typehint, name: str, *collections) -> None:
    """
    Ensure different origins for types in typing_extensions are captured.
    """
    if (typehint is False or getattr(typehint, "__module__", None) == "typing_extensions") and hasattr(
        __import__("typing"), name
    ):
        for collection in collections:
            collection.add(getattr(__import__("typing"), name))


def final(cls):
    """Decorator to make a class ``final``, i.e., it shouldn't be subclassed.

    It is the same as ``typing.final`` or an equivalent implementation
    depending on the python version and whether typing-extensions is
    installed.
    """
    setattr(cls, "__final__", True)
    return cls


def is_compatible_final(final) -> bool:
    @final
    class FinalClass:
        pass

    return getattr(FinalClass, "__final__", False)  # __final__ available in stdlib from python 3.11


fallback_final = final
stdlib_final = typing_extensions_import("final")
if stdlib_final and is_compatible_final(stdlib_final) and "SPHINX_BUILD" not in os.environ:
    final = stdlib_final


def import_typeshed_client():
    if typeshed_client_support:
        import typeshed_client

        return typeshed_client
    else:
        return __import__("argparse").Namespace(ImportedInfo=object, ModulePath=object, Resolver=object)


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
        raise ImportError(f"{package} package is required by {importer} :: {ex}") from ex


def import_toml_loads(importer):
    if find_spec("tomllib"):
        import tomllib

        return tomllib.loads, tomllib.TOMLDecodeError
    else:
        with missing_package_raise("toml", importer):
            import toml

        return toml.loads, toml.TomlDecodeError


def import_toml_dumps(importer):
    with missing_package_raise("toml", importer):
        import toml
    return toml.dumps


def import_jsonschema(importer):
    with missing_package_raise("jsonschema", importer):
        import jsonschema
    return jsonschema, jsonschema.Draft7Validator


def import_jsonnet(importer):
    with missing_package_raise("jsonnet", importer):
        import _jsonnet
    return _jsonnet


def import_requests(importer):
    with missing_package_raise("requests", importer):
        import requests
    return requests


def import_docstring_parser(importer):
    with missing_package_raise("docstring-parser", importer):
        import docstring_parser
    return docstring_parser


def import_fsspec(importer):
    with missing_package_raise("fsspec", importer):
        import fsspec
    return fsspec


def import_ruyaml(importer):
    with missing_package_raise("ruyaml", importer):
        import ruyaml
    return ruyaml


def import_reconplogger(importer):
    with missing_package_raise("reconplogger", importer):
        import reconplogger
    return reconplogger


def _set_config_read_mode(
    urls_enabled: bool = False,
    fsspec_enabled: bool = False,
):
    """Enables/disables optional config read modes.

    Args:
        urls_enabled: Whether to read config files from URLs using requests package.
        fsspec_enabled: Whether to read config files from fsspec supported file systems.
    """
    imports = {
        "u": import_requests,
        "s": import_fsspec,
    }

    def update_mode(flag, enabled):
        global _config_read_mode
        if enabled:
            imports[flag]("_set_config_read_mode")
            if flag not in _config_read_mode:
                _config_read_mode = _config_read_mode.replace("f", "f" + flag)
        else:
            _config_read_mode = _config_read_mode.replace(flag, "")

    update_mode("u", urls_enabled)
    update_mode("s", fsspec_enabled)


def _get_config_read_mode() -> str:
    """Returns the current config reading mode."""
    return _config_read_mode


def _set_docstring_parse_options(style=None, attribute_docstrings: Optional[bool] = None):
    """Sets options for docstring parsing.

    Args:
        style (docstring_parser.DocstringStyle): The docstring style to expect.
        attribute_docstrings: Whether to parse attribute docstrings (slower).
    """
    global _docstring_parse_options
    dp = import_docstring_parser("_set_docstring_parse_options")
    if style is not None:
        if not isinstance(style, dp.DocstringStyle):
            raise ValueError(f"Expected style to be of type {dp.DocstringStyle}.")
        _docstring_parse_options["style"] = style
    if attribute_docstrings is not None:
        if not isinstance(attribute_docstrings, bool):
            raise ValueError("Expected attribute_docstrings to be boolean.")
        _docstring_parse_options["attribute_docstrings"] = attribute_docstrings


def get_docstring_parse_options():
    if _docstring_parse_options["style"] is None:
        dp = import_docstring_parser("get_docstring_parse_options")
        _docstring_parse_options["style"] = dp.DocstringStyle.AUTO
    return _docstring_parse_options


def parse_docstring(component, params=False, logger=None):
    dp = import_docstring_parser("parse_docstring")
    options = get_docstring_parse_options()
    try:
        if params and options["attribute_docstrings"]:
            return dp.parse_from_object(component, style=options["style"])
        else:
            return dp.parse(component.__doc__, style=options["style"])
    except (ValueError, dp.ParseError) as ex:
        if logger:
            logger.debug(f"Failed parsing docstring for {component}: {ex}")
    return None


def parse_docs(component, parent, logger):
    docs = {}
    if docstring_parser_support:
        doc_sources = [component]
        if inspect.isclass(parent) and component.__name__ == "__init__":
            doc_sources += [parent]
        for src in doc_sources:
            doc = parse_docstring(src, params=True, logger=logger)
            if doc:
                for param in doc.params:
                    docs[param.arg_name] = param.description
    return docs


def get_doc_short_description(function_or_class, method_name=None, logger=None):
    if docstring_parser_support:
        component = function_or_class
        if inspect.isclass(function_or_class):
            if not method_name:
                docstring = parse_docstring(function_or_class, params=False, logger=logger)
                if docstring and docstring.short_description:
                    return docstring.short_description
            component = getattr(function_or_class, method_name or "__init__")
        docstring = parse_docstring(component, params=False, logger=logger)
        if docstring:
            return docstring.short_description
    return None


def get_omegaconf_loader():
    """Returns a yaml loader function based on OmegaConf which supports variable interpolation."""
    import io

    from ._loaders_dumpers import yaml_load

    with missing_package_raise("omegaconf", "get_omegaconf_loader"):
        from omegaconf import OmegaConf

    def omegaconf_load(value):
        value_pyyaml = yaml_load(value)
        if isinstance(value_pyyaml, (str, int, float, bool)) or value_pyyaml is None:
            return value_pyyaml
        value_omegaconf = OmegaConf.to_object(OmegaConf.load(io.StringIO(value)))
        str_ref = {k: None for k in [value]}
        return value_pyyaml if value_omegaconf == str_ref else value_omegaconf

    return omegaconf_load


annotated_alias = typing_extensions_import("_AnnotatedAlias")


def is_annotated(typehint: type) -> bool:
    return annotated_alias and isinstance(typehint, annotated_alias)


def get_annotated_base_type(typehint: type) -> type:
    return typehint.__origin__  # type: ignore[attr-defined]


type_alias_type = typing_extensions_import("TypeAliasType")


def is_alias_type(typehint: type) -> bool:
    return type_alias_type and isinstance(typehint, type_alias_type)


def get_alias_target(typehint: type) -> bool:
    return typehint.__value__  # type: ignore[attr-defined]


def get_pydantic_support() -> int:
    support = "0"
    if find_spec("pydantic"):
        support = version("pydantic")
    return int(support.split(".", 1)[0])


pydantic_support = get_pydantic_support()


def get_pydantic_supports_field_init() -> bool:
    if find_spec("pydantic"):
        support = version("pydantic")
        major, minor = tuple(int(x) for x in support.split(".")[:2])
        return major > 2 or (major == 2 and minor >= 6)
    return False


pydantic_supports_field_init = get_pydantic_supports_field_init()


def is_pydantic_model(class_type) -> int:
    classes = inspect.getmro(class_type) if pydantic_support and inspect.isclass(class_type) else []
    for cls in classes:
        if getattr(cls, "__module__", "").startswith("pydantic") and getattr(cls, "__name__", "") == "BaseModel":
            import pydantic

            if issubclass(cls, pydantic.BaseModel):
                return pydantic_support
            elif pydantic_support > 1 and issubclass(cls, pydantic.v1.BaseModel):
                return 1
    return 0


def get_module(value):
    return getattr(type(value), "__module__", "").split(".", 1)[0]


def is_annotated_validator(typehint: type) -> bool:
    from ._util import get_typehint_origin

    return (
        pydantic_support > 1
        and is_annotated(typehint)
        and any(get_module(m) in {"pydantic", "annotated_types"} for m in typehint.__metadata__)  # type: ignore[attr-defined]
        and get_typehint_origin(typehint.__origin__) != Union  # type: ignore[attr-defined]
    )


def validate_annotated(value, typehint: type):
    from pydantic import TypeAdapter

    return TypeAdapter(typehint).validate_python(value)
