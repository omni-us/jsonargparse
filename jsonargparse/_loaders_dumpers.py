"""Code related to loading and dumping."""

import inspect
import re
from contextlib import suppress
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

from ._common import load_value_mode, parent_parser
from ._optionals import import_jsonnet, import_toml_dumps, import_toml_loads, omegaconf_support, pyyaml_available
from ._type_checking import ArgumentParser

__all__ = [
    "get_loader",
    "set_loader",
    "set_dumper",
]


not_loaded = object()
yaml_default_loader = None


def load_basic(value):
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    if value.replace(".", "", 1).replace("e", "", 1).replace("-", "", 2).isdigit() and ("e" in value or "." in value):
        return float(value)
    return not_loaded


def get_yaml_default_loader():
    global yaml_default_loader
    if yaml_default_loader:
        return yaml_default_loader

    import yaml

    class DefaultLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
        pass

    # https://stackoverflow.com/a/37958106/2732151
    def remove_implicit_resolver(cls, tag_to_remove):
        if "yaml_implicit_resolvers" not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            cls.yaml_implicit_resolvers[first_letter] = [
                (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
            ]

    remove_implicit_resolver(DefaultLoader, "tag:yaml.org,2002:timestamp")
    remove_implicit_resolver(DefaultLoader, "tag:yaml.org,2002:float")

    DefaultLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    yaml_default_loader = DefaultLoader
    return yaml_default_loader


def yaml_load(stream):
    import yaml

    value = yaml.load(stream, Loader=get_yaml_default_loader())
    if isinstance(value, dict) and value and all(v is None for v in value.values()):
        if len(value) == 1 and stream.strip() == next(iter(value.keys())) + ":":
            value = stream
        else:
            keys = set(stream.strip(" {}").replace(" ", "").split(","))
            if len(keys) > 0 and keys == set(value.keys()):
                value = stream
    return value


def json_load(value):
    import json

    return json.loads(value)


def toml_load(value):
    toml_loads, _ = import_toml_loads("toml_load")
    return toml_loads(value)


def jsonnet_load(stream, path="", ext_vars=None):
    from ._jsonnet import ActionJsonnet

    ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
    _jsonnet = import_jsonnet("jsonnet_load")
    try:
        val = _jsonnet.evaluate_snippet(path, stream, ext_vars=ext_vars, ext_codes=ext_codes)
    except RuntimeError:
        try:
            return json_or_yaml_load(stream)
        except json_or_yaml_loader_exceptions as ex:
            raise ValueError(str(ex)) from ex
    return json_or_yaml_load(val)


loaders: Dict[str, Callable] = {
    "yaml": yaml_load,
    "json": json_load,
    "toml": toml_load,
}
loader_exceptions: Dict[str, Tuple[Type[Exception], ...]] = {}
loader_json_superset: Dict[str, bool] = {
    "yaml": True,
    "json": True,
    "toml": False,
}
loader_params: Dict[str, Set[str]] = {}


def get_load_value_mode() -> str:
    mode = load_value_mode.get()
    if mode is None:
        parser = parent_parser.get()
        assert parser is not None
        mode = parser.parser_mode
    return mode


def get_loader_exceptions(mode: Optional[str] = None) -> Tuple[Type[Exception], ...]:
    if mode is None:
        mode = get_load_value_mode()
    if mode not in loader_exceptions:
        if mode == "yaml":
            loader_exceptions[mode] = (__import__("yaml").YAMLError,)
        elif mode == "json":
            loader_exceptions[mode] = (__import__("json").JSONDecodeError,)
        elif mode == "toml":
            loader_exceptions[mode] = (import_toml_loads("get_loader_exceptions")[1],)
        elif mode == "jsonnet":
            return get_loader_exceptions("yaml" if pyyaml_available else "json") + (ValueError,)
    return loader_exceptions[mode]


def json_or_yaml_load(value):
    if pyyaml_available:
        if isinstance(value, str) and value.strip() == "":
            return value
        return yaml_load(value)
    return json_load(value)


json_or_yaml_loader_exceptions = get_loader_exceptions("yaml" if pyyaml_available else "json")


def load_list_or_dict(value: str):
    strip = value.strip()
    if (strip.startswith("[") and strip.endswith("]")) or (strip.startswith("{") and strip.endswith("}")):
        import json

        with suppress(json.JSONDecodeError):
            return json.loads(strip)
    return not_loaded


def load_value(value: str, simple_types: bool = False, **kwargs):
    if value.strip() == "-":
        return value

    loaded_value = load_basic(value)

    mode = get_load_value_mode()
    if loaded_value is not_loaded and not loader_json_superset[mode]:
        loaded_value = load_list_or_dict(value)

    if loaded_value is not_loaded:
        loader = loaders[mode]
        load_kwargs = {}
        if kwargs and mode in loader_params:
            params = loader_params[mode]
            load_kwargs = {k: v for k, v in kwargs.items() if k in params}
        loaded_value = loader(value, **load_kwargs)

    if not simple_types and isinstance(loaded_value, (int, float, bool, str)):
        loaded_value = value

    return loaded_value


dump_yaml_kwargs = {
    "default_flow_style": False,
    "allow_unicode": True,
    "sort_keys": False,
}

dump_json_kwargs = {
    "ensure_ascii": False,
    "sort_keys": False,
}


def yaml_dump(data):
    import yaml

    return yaml.safe_dump(data, **dump_yaml_kwargs)


def yaml_comments_dump(data, parser):
    dump = dumpers["yaml"](data)
    formatter = parser.formatter_class(parser.prog)
    return formatter.add_yaml_comments(dump)


def json_compact_dump(data):
    import json

    return json.dumps(data, separators=(",", ":"), **dump_json_kwargs)


def json_indented_dump(data):
    import json

    return json.dumps(data, indent=2, **dump_json_kwargs) + "\n"


def toml_dump(data):
    toml_dumps = import_toml_dumps("toml_dump")
    return toml_dumps(data)


dumpers: Dict[str, Callable] = {
    "yaml": yaml_dump,
    "yaml_comments": yaml_comments_dump,
    "json": json_compact_dump,
    "json_compact": json_compact_dump,
    "json_indented": json_indented_dump,
    "toml": toml_dump,
    "jsonnet": json_indented_dump,
}

comment_prefix: Dict[str, str] = {
    "yaml": "# ",
    "yaml_comments": "# ",
    "jsonnet": "// ",
    "toml": "# ",
}


def check_valid_dump_format(dump_format: str):
    if dump_format not in {"parser_mode"}.union(set(dumpers)):
        raise ValueError(f'Unknown output format "{dump_format}".')


def dump_using_format(parser: "ArgumentParser", data: dict, dump_format: str) -> str:
    if dump_format == "parser_mode":
        dump_format = parser.parser_mode if parser.parser_mode in dumpers else "yaml"
    args = (data, parser) if dump_format == "yaml_comments" else (data,)
    dump = dumpers[dump_format](*args)
    if parser.dump_header and comment_prefix.get(dump_format):
        prefix = comment_prefix[dump_format]
        header = "\n".join(prefix + line for line in parser.dump_header)
        dump = f"{header}\n{dump}"
    return dump


def set_loader(
    mode: str,
    loader_fn: Callable[[str], Any],
    exceptions: Tuple[Type[Exception], ...] = tuple(),
    json_superset: bool = True,
):
    """Sets the value loader function to be used when parsing with a certain mode.

    The ``loader_fn`` function must accept as input a single str type parameter
    and return any of the basic types {str, bool, int, float, list, dict, None}.
    If this function is not based on PyYAML for things to work correctly the
    exceptions types that can be raised when parsing a value fails should be
    provided.

    Args:
        mode: The parser mode for which to set its loader function. Example: "yaml".
        loader_fn: The loader function to set. Example: ``yaml.safe_load``.
        exceptions: Exceptions that the loader can raise when load fails.
            Example: (yaml.YAMLError,).
        json_superset: Whether the loader can load JSON data.
    """
    loaders[mode] = loader_fn
    loader_exceptions[mode] = exceptions
    loader_json_superset[mode] = json_superset
    params = set(list(inspect.signature(loader_fn).parameters)[1:])
    if params:
        loader_params[mode] = params


def get_loader(mode: str):
    """Returns the current loader function for a given mode."""
    return loaders[mode]


def set_dumper(format_name: str, dumper_fn: Callable[[Any], str]):
    """Sets the dumping function for a given format name.

    Args:
        format_name: Name to use for dumping with this function. Example: "yaml_custom".
        dumper_fn: The dumper function to set. Example: ``yaml.safe_dump``.
    """
    dumpers[format_name] = dumper_fn


def set_omegaconf_loader():
    if omegaconf_support and "omegaconf" not in loaders:
        from ._optionals import get_omegaconf_loader

        set_loader("omegaconf", get_omegaconf_loader())


set_loader("jsonnet", jsonnet_load, get_loader_exceptions("jsonnet"))
