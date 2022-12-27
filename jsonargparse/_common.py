from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Union

from .namespace import Namespace
from .type_checking import ArgumentParser

parent_parser: ContextVar['ArgumentParser'] = ContextVar('parent_parser')
parser_capture: ContextVar[bool] = ContextVar('parser_capture', default=False)
defaults_cache: ContextVar[Optional[Namespace]] = ContextVar('defaults_cache', default=None)
lenient_check: ContextVar[Union[bool, str]] = ContextVar('lenient_check', default=False)
load_value_mode: ContextVar[Optional[str]] = ContextVar('load_value_mode', default=None)


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
