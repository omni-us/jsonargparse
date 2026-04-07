from argparse import Action, _SubParsersAction
from contextlib import contextmanager
from typing import Iterator, Optional, Union

from ._type_checking import ArgumentParser


def _iter_required_action_keys(parser: ArgumentParser) -> Iterator[str]:
    """Yields required destinations backed by real argparse actions."""
    for action in parser._actions:
        if action.required:
            yield action.dest


def _iter_extra_required_keys(parser: ArgumentParser) -> Iterator[str]:
    """Yields required keys tracked outside concrete argparse actions."""
    yield from parser._extra_required_keys


def iter_required_keys(parser: ArgumentParser) -> Iterator[str]:
    """Yields required keys with action-backed ones first."""
    yielded = set()
    for key in _iter_required_action_keys(parser):
        yielded.add(key)
        yield key
    for key in sorted(_iter_extra_required_keys(parser)):
        if key not in yielded:
            yield key


def set_required(parser: ArgumentParser, key_or_action: Union[str, Action], value: bool = True) -> None:
    """Sets required state for either an action-backed or virtual key."""
    action = key_or_action if isinstance(key_or_action, Action) else None
    key: str = action.dest if action is not None else key_or_action  # type: ignore[assignment]
    if action is None:
        action = _find_exact_action(parser, key)
    if action is not None and getattr(action, "dest", None) == key:
        action.required = value
    elif value:
        parser._extra_required_keys.add(key)
    else:
        parser._extra_required_keys.discard(key)


def clear_required(parser: ArgumentParser, key_or_action: Union[str, Action]) -> None:
    """Clears required state for either an action-backed or virtual key."""
    set_required(parser, key_or_action, value=False)


def _find_exact_action(parser: ArgumentParser, key: str) -> Optional[Action]:
    for action in parser._actions:
        if getattr(action, "dest", None) == key:
            return action
    return None


@contextmanager
def suppress_required_actions(parser: ArgumentParser):
    """Temporarily disables required enforcement on real argparse actions."""
    suppressed = []
    visited = set()

    def visit(subparser):
        if id(subparser) in visited:
            return
        visited.add(id(subparser))
        for action in subparser._actions:
            if action.required:
                suppressed.append(action)
                action.required = False
            if isinstance(action, _SubParsersAction):
                for choice_parser in action.choices.values():
                    visit(choice_parser)

    visit(parser)
    try:
        yield
    finally:
        for action in reversed(suppressed):
            action.required = True
