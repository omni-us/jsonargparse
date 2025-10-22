from __future__ import annotations

import logging
import os
from calendar import Calendar
from importlib import import_module
from random import Random
from unittest.mock import patch

import pytest

from jsonargparse import (
    ArgumentParser,
    Namespace,
    capture_parser,
    class_from_function,
)
from jsonargparse._common import LoggerProperty, null_logger
from jsonargparse._optionals import docstring_parser_support, reconplogger_support
from jsonargparse._util import (
    CaptureParserException,
    get_import_path,
    import_object,
    object_path_serializer,
    register_unresolvable_import_paths,
    unique,
)
from jsonargparse_tests.conftest import capture_logs, get_parser_help

# logger property tests


class WithLogger(LoggerProperty):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


log_message = "testing log message"


def test_logger_true():
    test = WithLogger(logger=True)
    if reconplogger_support:
        assert test.logger.name == "plain_logger"
    else:
        assert test.logger.handlers[0].level == logging.WARNING
        assert test.logger.name == "WithLogger"
    with capture_logs(test.logger) as logs:
        test.logger.error(log_message)
    assert "ERROR" in logs.getvalue()
    assert log_message in logs.getvalue()


def test_logger_false():
    test = WithLogger(logger=False)
    assert test.logger is null_logger
    with capture_logs(test.logger) as logs:
        test.logger.error(log_message)
    assert "" == logs.getvalue()


def test_no_init_logger():
    class WithLoggerNoInit(LoggerProperty):
        pass

    test = WithLoggerNoInit()
    assert test.logger is null_logger


def test_logger_str():
    logger = logging.getLogger("test_logger_str")
    test = WithLogger(logger="test_logger_str")
    assert test.logger is logger


def test_logger_object():
    logger = logging.getLogger("test_logger_object")
    test = WithLogger(logger=logger)
    assert test.logger is logger
    assert test.logger.name == "test_logger_object"


def test_logger_name():
    test = WithLogger(logger={"name": "test_logger_name"})
    assert test.logger.name == "test_logger_name"


def test_logger_failure_cases():
    pytest.raises(ValueError, lambda: WithLogger(logger={"level": "invalid"}))
    pytest.raises(ValueError, lambda: WithLogger(logger=WithLogger))
    pytest.raises(ValueError, lambda: WithLogger(logger={"invalid": "value"}))


levels = {0: "DEBUG", 1: "INFO", 2: "WARNING", 3: "ERROR", 4: "CRITICAL"}


@pytest.mark.parametrize(["num", "level"], levels.items())
@pytest.mark.skipif(reconplogger_support, reason="level not overridden when using reconplogger")
def test_logger_levels(num, level):
    test = WithLogger(logger={"level": level})
    with capture_logs(test.logger) as logs:
        getattr(test.logger, level.lower())(log_message)
    assert level in logs.getvalue()
    assert log_message in logs.getvalue()
    if level != "DEBUG":
        with capture_logs(test.logger) as logs:
            getattr(test.logger, levels[num - 1].lower())(log_message)
        assert "" == logs.getvalue()


@patch.dict(os.environ, {"JSONARGPARSE_DEBUG": "true"})
def test_logger_jsonargparse_debug():
    parser = ArgumentParser(logger=False)
    with capture_logs(parser.logger) as logs:
        parser.logger.debug(log_message)
    assert "DEBUG" in logs.getvalue()
    assert log_message in logs.getvalue()


# import paths tests


def test_import_object_invalid():
    with pytest.raises(ValueError) as ctx:
        import_object(True)
    ctx.match("Expected a dot import path string")
    with pytest.raises(ValueError) as ctx:
        import_object("jsonargparse-tests.os")
    ctx.match("Unexpected import path format")


def test_get_import_path():
    assert get_import_path(ArgumentParser) == "jsonargparse.ArgumentParser"
    assert get_import_path(ArgumentParser.merge_config) == "jsonargparse.ArgumentParser.merge_config"
    from email.mime.base import MIMEBase

    assert get_import_path(MIMEBase) == "email.mime.base.MIMEBase"
    from dataclasses import MISSING

    assert get_import_path(MISSING) == "dataclasses.MISSING"


class _StaticMethods:
    @staticmethod
    def static_method():
        pass


static_method = _StaticMethods.static_method


def test_get_import_path_static_method_shorthand():
    assert get_import_path(static_method) == f"{__name__}.static_method"


class ParentClassmethod:
    __module__ = "jsonargparse_tests"

    @classmethod
    def class_method(cls):
        pass


class ChildClassmethod(ParentClassmethod):
    pass


def test_get_import_path_classpath_inheritance():
    assert get_import_path(ParentClassmethod.class_method) == "jsonargparse_tests.ParentClassmethod.class_method"
    assert get_import_path(ChildClassmethod.class_method) == f"{__name__}.ChildClassmethod.class_method"


def unresolvable_import():
    pass


@patch.dict("jsonargparse._util.unresolvable_import_paths")
def test_register_unresolvable_import_paths():
    unresolvable_import.__module__ = None
    pytest.raises(ValueError, lambda: get_import_path(unresolvable_import))
    register_unresolvable_import_paths(import_module(__name__))
    assert get_import_path(unresolvable_import) == f"{__name__}.unresolvable_import"


class Class:
    @staticmethod
    def method1():
        pass

    def method2(self):
        pass


def test_object_path_serializer_class_method():
    assert object_path_serializer(Class.method1) == f"{__name__}.Class.method1"
    assert object_path_serializer(Class.method2) == f"{__name__}.Class.method2"


def test_object_path_serializer_reimport_differs():
    class FakeClass:
        pass

    FakeClass.__module__ = Class.__module__
    FakeClass.__qualname__ = Class.__qualname__
    pytest.raises(ValueError, lambda: object_path_serializer(FakeClass))


# class_from_function tests


def get_random() -> Random:
    return Random()


class Foo:
    @classmethod
    def get_foo(cls) -> "Foo":
        return cls()


def closure_get_foo():
    def get_foo() -> Foo:
        return Foo()

    return get_foo


@pytest.mark.parametrize(
    ["function", "class_type"],
    [
        (get_random, Random),
        (Foo.get_foo, Foo),
        (closure_get_foo(), Foo),
    ],
)
def test_class_from_function(function, class_type):
    cls = class_from_function(function)
    assert issubclass(cls, class_type)
    assert isinstance(cls(), class_type)
    module_path, name = get_import_path(cls).rsplit(".", 1)
    assert module_path == __name__
    assert cls is globals()[name]
    assert cls is class_from_function(function)


def test_class_from_function_name_clash():
    with pytest.raises(ValueError) as ctx:
        class_from_function(get_random, name="get_random")
    ctx.match("already defines 'get_random', please use a different name")


def get_unknown() -> "Unknown":  # type: ignore  # noqa: F821
    return None


def test_invalid_class_from_function():
    with pytest.raises(ValueError) as ctx:
        class_from_function(get_unknown)
    ctx.match("Unable to dereference '?Unknown'?, the return type of")


def get_random_untyped():
    return Random()


def test_class_from_function_given_return_type():
    cls = class_from_function(get_random_untyped, Random)
    assert issubclass(cls, Random)
    assert isinstance(cls(), Random)


def get_calendar(a1: str, a2: int = 2) -> Calendar:
    """Returns instance of Calendar"""
    cal = Calendar()
    cal.a1 = a1  # type: ignore[attr-defined]
    cal.a2 = a2  # type: ignore[attr-defined]
    return cal


def test_add_class_from_function_arguments(parser):
    get_calendar_class = class_from_function(get_calendar)
    parser.add_class_arguments(get_calendar_class, "a")

    if docstring_parser_support:
        help_str = get_parser_help(parser)
        assert "Returns instance of Calendar" in help_str

    cfg = parser.parse_args(["--a.a1=v", "--a.a2=3"])
    assert cfg.a == Namespace(a1="v", a2=3)
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.a, Calendar)
    assert init.a.a1 == "v"
    assert init.a.a2 == 3


def without_return_type():
    pass


def test_class_from_function_missing_return():
    with pytest.raises(ValueError) as ctx:
        class_from_function(without_return_type)
    ctx.match("does not have a return type annotation")


# other tests


def test_unique():
    data = [1.0, 2, {}, "x", ([], {}), 2, [], {}, [], ([], {}), 2]
    assert unique(data) == [1.0, 2, {}, "x", ([], {}), []]


def test_capture_parser():
    def parse_args(args=[]):
        parser = ArgumentParser()
        parser.add_argument("--int", type=int, default=1)
        return parser.parse_args(args)

    parser = capture_parser(parse_args, ["--int=2"])
    assert isinstance(parser, ArgumentParser)
    assert parser.get_defaults() == Namespace(int=1)

    with pytest.raises(CaptureParserException) as ctx:
        capture_parser(lambda: None)
    ctx.match("No parse_args call to capture the parser")
