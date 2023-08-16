import dataclasses
import inspect
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional, Tuple, Type, TypeVar, Union

from ._namespace import Namespace
from ._type_checking import ArgumentParser

ClassType = TypeVar("ClassType")

if sys.version_info < (3, 8):
    from typing import Callable

    InstantiatorCallable = Callable[..., ClassType]
else:
    from typing import Protocol

    class InstantiatorCallable(Protocol):
        def __call__(self, class_type: Type[ClassType], *args, **kwargs) -> ClassType:
            pass  # pragma: no cover


InstantiatorsDictType = Dict[Tuple[type, bool], InstantiatorCallable]


parent_parser: ContextVar["ArgumentParser"] = ContextVar("parent_parser")
parser_capture: ContextVar[bool] = ContextVar("parser_capture", default=False)
defaults_cache: ContextVar[Optional[Namespace]] = ContextVar("defaults_cache", default=None)
lenient_check: ContextVar[Union[bool, str]] = ContextVar("lenient_check", default=False)
load_value_mode: ContextVar[Optional[str]] = ContextVar("load_value_mode", default=None)
class_instantiators: ContextVar[Optional[InstantiatorsDictType]] = ContextVar("class_instantiators")


parser_context_vars = dict(
    parent_parser=parent_parser,
    parser_capture=parser_capture,
    defaults_cache=defaults_cache,
    lenient_check=lenient_check,
    load_value_mode=load_value_mode,
    class_instantiators=class_instantiators,
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


def default_class_instantiator(class_type: Type[ClassType], *args, **kwargs) -> ClassType:
    return class_type(*args, **kwargs)


class ClassInstantiator:
    def __init__(self, instantiators: InstantiatorsDictType) -> None:
        self.instantiators = instantiators

    def __call__(self, class_type: Type[ClassType], *args, **kwargs) -> ClassType:
        for (cls, subclasses), instantiator in self.instantiators.items():
            if class_type is cls or (subclasses and is_subclass(class_type, cls)):
                return instantiator(class_type, *args, **kwargs)
        return default_class_instantiator(class_type, *args, **kwargs)


def get_class_instantiator() -> InstantiatorCallable:
    instantiators = class_instantiators.get()
    if not instantiators:
        return default_class_instantiator
    return ClassInstantiator(instantiators)
