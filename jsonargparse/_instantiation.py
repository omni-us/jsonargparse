import inspect

from ._common import (
    ClassType,
    InstantiatorCallable,
    InstantiatorsDictType,
    applied_instantiation_links,
    class_instantiators,
    is_subclass,
)

__all__ = ["add_instantiator"]

_global_class_instantiators: InstantiatorsDictType = {}


def add_instantiator(
    instantiator: InstantiatorCallable,
    class_type: type[ClassType],
    subclasses: bool = True,
    prepend: bool = False,
) -> None:
    """Adds a custom instantiator for a class type. Used by ``ArgumentParser.instantiate``.

    Instantiator functions are expected to have as signature ``(class_type:
    Type[ClassType], *args, **kwargs) -> ClassType``.

    For reference, the default instantiator is ``return class_type(*args,
    **kwargs)``.

    In some use cases, the instantiator function might need access to values
    applied by instantiation links. For this, the instantiator function can
    have an additional keyword parameter ``applied_instantiation_links:
    dict``. This parameter will be populated with a dictionary having as
    keys the targets of the instantiation links and corresponding values
    that were applied.

    Args:
        instantiator: Function that instantiates a class.
        class_type: The class type to instantiate.
        subclasses: Whether to instantiate subclasses of ``class_type``.
        prepend: Whether to prepend the instantiator to the existing instantiators.
    """
    _register_instantiator(
        _global_class_instantiators, instantiator, class_type, subclasses=subclasses, prepend=prepend
    )


def _register_instantiator(
    registry: InstantiatorsDictType,
    instantiator: InstantiatorCallable,
    class_type: type[ClassType],
    subclasses: bool = True,
    prepend: bool = False,
) -> None:
    """Registers an instantiator in the given registry dict (in-place)."""
    key = (class_type, subclasses)
    items = {k: v for k, v in registry.items() if k != key}
    if prepend:
        registry.clear()
        registry.update({key: instantiator, **items})
    else:
        items[key] = instantiator
        registry.clear()
        registry.update(items)


def _get_global_class_instantiators() -> InstantiatorsDictType:
    """Returns the global instantiators registry."""
    return _global_class_instantiators


def default_class_instantiator(class_type: type[ClassType], *args, **kwargs) -> ClassType:
    return class_type(*args, **kwargs)


class ClassInstantiator:
    def __init__(self, instantiators: InstantiatorsDictType) -> None:
        self.instantiators = instantiators

    def __call__(self, class_type: type[ClassType], *args, **kwargs) -> ClassType:
        for (cls, subclasses), instantiator in self.instantiators.items():
            if class_type is cls or (subclasses and is_subclass(class_type, cls)):
                param_names = set(inspect.signature(instantiator).parameters)
                if "applied_instantiation_links" in param_names:
                    applied_links = applied_instantiation_links.get() or set()
                    kwargs["applied_instantiation_links"] = {
                        action.target[0]: action.applied_value for action in applied_links
                    }
                return instantiator(class_type, *args, **kwargs)
        return default_class_instantiator(class_type, *args, **kwargs)


def get_class_instantiator() -> InstantiatorCallable:
    instantiators = class_instantiators.get()
    if not instantiators:
        return default_class_instantiator
    return ClassInstantiator(instantiators)


def get_class_instantiators(parser) -> InstantiatorsDictType:
    """Gathers all instantiators applicable to the given parser."""
    instantiators = parser._get_parser_instantiators()
    context_instantiators = class_instantiators.get()
    if context_instantiators:
        instantiators = instantiators.copy()
        instantiators.update({k: v for k, v in context_instantiators.items() if k not in instantiators})
    global_instantiators = _get_global_class_instantiators()
    if global_instantiators:
        instantiators = instantiators.copy()
        instantiators.update({k: v for k, v in global_instantiators.items() if k not in instantiators})
    return instantiators
