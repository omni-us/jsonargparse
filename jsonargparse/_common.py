import dataclasses
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Union

from ._namespace import Namespace
from ._type_checking import ArgumentParser

parent_parser: ContextVar["ArgumentParser"] = ContextVar("parent_parser")
parser_capture: ContextVar[bool] = ContextVar("parser_capture", default=False)
defaults_cache: ContextVar[Optional[Namespace]] = ContextVar("defaults_cache", default=None)
lenient_check: ContextVar[Union[bool, str]] = ContextVar("lenient_check", default=False)
load_value_mode: ContextVar[Optional[str]] = ContextVar("load_value_mode", default=None)


parser_context_vars = dict(
    parent_parser=parent_parser,
    parser_capture=parser_capture,
    defaults_cache=defaults_cache,
    lenient_check=lenient_check,
    load_value_mode=load_value_mode,
)


@contextmanager
def parser_context(**kwargs):
    context_var_tokens = []
    for name, value in kwargs.items():
        context_var = parser_context_vars[name]
        token = context_var.set(value)
        context_var_tokens.append((context_var, token))
    try:
        yield
    finally:
        for context_var, token in context_var_tokens:
            context_var.reset(token)


def is_subclass(cls, class_or_tuple) -> bool:
    """Extension of issubclass that supports non-class arguments."""
    try:
        return inspect.isclass(cls) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def is_final_class(cls) -> bool:
    """Checks whether a class is final, i.e. decorated with ``typing.final``."""
    return getattr(cls, "__final__", False)


def is_dataclass_like(cls) -> bool:
    if not inspect.isclass(cls):
        return False
    if is_final_class(cls):
        return True
    classes = [c for c in inspect.getmro(cls) if c != object]
    all_dataclasses = all(dataclasses.is_dataclass(c) for c in classes)
    from ._optionals import attrs_support, pydantic_support

    if not all_dataclasses and pydantic_support:
        import pydantic.utils

        classes = [c for c in classes if c != pydantic.utils.Representation]
        all_dataclasses = all(is_subclass(c, pydantic.BaseModel) for c in classes)
    if not all_dataclasses and attrs_support:
        import attrs

        if attrs.has(cls):
            return True
    return all_dataclasses
