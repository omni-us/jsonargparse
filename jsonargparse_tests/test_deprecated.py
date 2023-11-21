from __future__ import annotations

import os
import pathlib
from calendar import Calendar
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from enum import Enum
from importlib import import_module
from io import StringIO
from warnings import catch_warnings

import pytest

from jsonargparse import (
    CLI,
    ActionConfigFile,
    ActionJsonnet,
    ArgumentError,
    ArgumentParser,
    Path,
    get_config_read_mode,
    set_url_support,
)
from jsonargparse._deprecated import (
    ActionEnum,
    ActionJsonnetExtVars,
    ActionOperators,
    ActionPath,
    ActionPathList,
    ParserError,
    deprecation_warning,
    shown_deprecation_warnings,
    usage_and_exit_error_handler,
)
from jsonargparse._optionals import jsonnet_support, url_support
from jsonargparse._util import LoggerProperty, argument_error
from jsonargparse_tests.conftest import (
    get_parser_help,
    is_posix,
    skip_if_docstring_parser_unavailable,
    skip_if_requests_unavailable,
)
from jsonargparse_tests.test_jsonnet import example_2_jsonnet


@pytest.fixture(autouse=True)
def clear_shown_deprecation_warnings():
    yield
    shown_deprecation_warnings.clear()


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull):
            yield


source = pathlib.Path(__file__).read_text().splitlines()


def assert_deprecation_warn(warns, message, code):
    assert message in str(warns[-1].message)
    if code is None:
        return
    assert pathlib.Path(warns[-1].filename).name == pathlib.Path(__file__).name
    assert code in source[warns[-1].lineno - 1]


def test_deprecation_warning():
    with catch_warnings(record=True) as w:
        message = "Deprecation warning"
        deprecation_warning(None, message)
        assert 2 == len(w)
        assert "only one JsonargparseDeprecationWarning per type is shown" in str(w[0].message)
        assert message == str(w[1].message).strip()


class MyEnum(Enum):
    A = 1
    B = 2
    C = 3


def func(a1: MyEnum = MyEnum["A"]):
    return a1


def test_ActionEnum():
    parser = ArgumentParser(exit_on_error=False)
    with catch_warnings(record=True) as w:
        action = ActionEnum(enum=MyEnum)
    assert_deprecation_warn(
        w,
        message="ActionEnum was deprecated",
        code="ActionEnum(enum=MyEnum)",
    )
    parser.add_argument("--enum", action=action, default=MyEnum.C, help="Description")

    for val in ["A", "B", "C"]:
        assert MyEnum[val] == parser.parse_args(["--enum=" + val]).enum
    for val in ["X", "b", 2]:
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--enum=" + str(val)]))

    cfg = parser.parse_args(["--enum=C"], with_meta=False)
    assert "enum: C\n" == parser.dump(cfg)

    help_str = get_parser_help(parser)
    assert "Description (type: MyEnum, default: C)" in help_str

    parser = ArgumentParser()
    parser.add_function_arguments(func)
    assert MyEnum["A"] == parser.get_defaults().a1
    assert MyEnum["B"] == parser.parse_args(["--a1=B"]).a1

    pytest.raises(ValueError, ActionEnum)
    pytest.raises(ValueError, lambda: ActionEnum(enum=object))
    pytest.raises(ValueError, lambda: parser.add_argument("--bad1", type=MyEnum, action=True))
    pytest.raises(ValueError, lambda: parser.add_argument("--bad2", type=float, action=action))


def test_ActionOperators():
    parser = ArgumentParser(prog="app", exit_on_error=False)
    with catch_warnings(record=True) as w:
        parser.add_argument("--le0", action=ActionOperators(expr=("<", 0)))
    assert_deprecation_warn(
        w,
        message="ActionOperators was deprecated",
        code='ActionOperators(expr=("<", 0))',
    )
    parser.add_argument(
        "--gt1.a.le4",
        action=ActionOperators(expr=[(">", 1.0), ("<=", 4.0)], join="and", type=float),
    )
    parser.add_argument(
        "--lt5.o.ge10.o.eq7",
        action=ActionOperators(expr=[("<", 5), (">=", 10), ("==", 7)], join="or", type=int),
    )
    parser.add_argument("--ge0", nargs=3, action=ActionOperators(expr=(">=", 0)))

    assert 1.5 == parser.parse_args(["--gt1.a.le4", "1.5"]).gt1.a.le4
    assert 4.0 == parser.parse_args(["--gt1.a.le4", "4.0"]).gt1.a.le4
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--gt1.a.le4", "1.0"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--gt1.a.le4", "5.5"]))

    assert 1.5 == parser.parse_string("gt1:\n  a:\n    le4: 1.5").gt1.a.le4
    assert 4.0 == parser.parse_string("gt1:\n  a:\n    le4: 4.0").gt1.a.le4
    pytest.raises(ArgumentError, lambda: parser.parse_string("gt1:\n  a:\n    le4: 1.0"))
    pytest.raises(ArgumentError, lambda: parser.parse_string("gt1:\n  a:\n    le4: 5.5"))

    assert 1.5 == parser.parse_env({"APP_GT1__A__LE4": "1.5"}).gt1.a.le4
    assert 4.0 == parser.parse_env({"APP_GT1__A__LE4": "4.0"}).gt1.a.le4
    pytest.raises(ArgumentError, lambda: parser.parse_env({"APP_GT1__A__LE4": "1.0"}))
    pytest.raises(ArgumentError, lambda: parser.parse_env({"APP_GT1__A__LE4": "5.5"}))

    assert 2 == parser.parse_args(["--lt5.o.ge10.o.eq7", "2"]).lt5.o.ge10.o.eq7
    assert 7 == parser.parse_args(["--lt5.o.ge10.o.eq7", "7"]).lt5.o.ge10.o.eq7
    assert 10 == parser.parse_args(["--lt5.o.ge10.o.eq7", "10"]).lt5.o.ge10.o.eq7
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--lt5.o.ge10.o.eq7", "5"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--lt5.o.ge10.o.eq7", "8"]))

    assert [0, 1, 2] == parser.parse_args(["--ge0", "0", "1", "2"]).ge0

    pytest.raises(ValueError, lambda: parser.add_argument("--op1", action=ActionOperators))
    action = ActionOperators(expr=("<", 0))
    pytest.raises(ValueError, lambda: parser.add_argument("--op2", type=float, action=action))
    pytest.raises(ValueError, lambda: parser.add_argument("--op3", nargs=0, action=action))
    pytest.raises(ValueError, ActionOperators)
    pytest.raises(ValueError, lambda: ActionOperators(expr="<"))
    pytest.raises(ValueError, lambda: ActionOperators(expr=[("<", 5), (">=", 10)], join="xor"))


@skip_if_requests_unavailable
def test_url_support_true():
    assert "fr" == get_config_read_mode()
    with catch_warnings(record=True) as w:
        set_url_support(True)
    assert_deprecation_warn(
        w,
        message="set_url_support was deprecated",
        code="set_url_support(True)",
    )
    assert "fur" == get_config_read_mode()
    set_url_support(False)
    assert "fr" == get_config_read_mode()


@pytest.mark.skipif(url_support, reason="requests package should not be installed")
def test_url_support_false():
    assert "fr" == get_config_read_mode()
    with catch_warnings(record=True) as w:
        with pytest.raises(ImportError):
            set_url_support(True)
        assert "set_url_support was deprecated" in str(w[-1].message)
    assert "fr" == get_config_read_mode()
    set_url_support(False)
    assert "fr" == get_config_read_mode()


def test_instantiate_subclasses():
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--cal", type=Calendar)
    cfg = parser.parse_object({"cal": {"class_path": "calendar.Calendar"}})
    with catch_warnings(record=True) as w:
        cfg_init = parser.instantiate_subclasses(cfg)
    assert_deprecation_warn(
        w,
        message="instantiate_subclasses was deprecated",
        code="parser.instantiate_subclasses(cfg)",
    )
    assert isinstance(cfg_init["cal"], Calendar)


def function(a1: float):
    return a1


def test_single_function_cli():
    with catch_warnings(record=True) as w:
        parser = CLI(function, return_parser=True, set_defaults={"a1": 3.4})
    assert_deprecation_warn(
        w,
        message="return_parser parameter was deprecated",
        code="CLI(function, return_parser=True,",
    )
    assert isinstance(parser, ArgumentParser)


def cmd1(a1: int):
    return a1


def cmd2(a2: str = "X"):
    return a2


def test_multiple_functions_cli():
    with catch_warnings(record=True) as w:
        parser = CLI([cmd1, cmd2], return_parser=True, set_defaults={"cmd2.a2": "Z"})
    assert_deprecation_warn(
        w,
        message="return_parser parameter was deprecated",
        code="CLI([cmd1, cmd2], return_parser=True,",
    )
    assert isinstance(parser, ArgumentParser)


def test_logger_property_none():
    with catch_warnings(record=True) as w:
        LoggerProperty(logger=None)
    assert_deprecation_warn(
        w,
        message=" Setting the logger property to None was deprecated",
        code="LoggerProperty(logger=None)",
    )


def test_env_prefix_none():
    with catch_warnings(record=True) as w:
        ArgumentParser(env_prefix=None)
    assert_deprecation_warn(
        w,
        message="env_prefix",
        code="ArgumentParser(env_prefix=None)",
    )


def test_error_handler_parameter():
    with catch_warnings(record=True) as w:
        parser = ArgumentParser(error_handler=usage_and_exit_error_handler)
    code = "ArgumentParser(error_handler=usage_"
    if not is_posix:
        code = None  # for some reason the stack trace differs in windows
    assert_deprecation_warn(
        w,
        message="error_handler was deprecated in v4.20.0",
        code=code,
    )
    assert parser.error_handler == usage_and_exit_error_handler
    with suppress_stderr(), pytest.raises(SystemExit), catch_warnings(record=True):
        parser.parse_args(["--invalid"])


def test_error_handler_property():
    def custom_error_handler(self, message):
        print("custom_error_handler")
        self.exit(2)

    parser = ArgumentParser()
    with catch_warnings(record=True) as w:
        parser.error_handler = custom_error_handler
    assert_deprecation_warn(
        w,
        message="error_handler was deprecated in v4.20.0",
        code="parser.error_handler = custom_error_handler",
    )
    assert parser.error_handler == custom_error_handler

    out = StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit):
        parser.parse_args(["--invalid"])
    assert out.getvalue() == "custom_error_handler\n"

    with pytest.raises(ValueError):
        parser.error_handler = "invalid"


def test_ParserError():
    assert isinstance(argument_error(""), ParserError)


def test_parse_as_dict(tmp_cwd):
    with open("config.json", "w") as f:
        f.write("{}")
    with catch_warnings(record=True) as w:
        parser = ArgumentParser(parse_as_dict=True, default_meta=False)
    assert_deprecation_warn(
        w,
        message="``parse_as_dict`` parameter was deprecated",
        code="ArgumentParser(parse_as_dict=True,",
    )
    assert {} == parser.parse_args([])
    assert {} == parser.parse_env([])
    assert {} == parser.parse_string("{}")
    assert {} == parser.parse_object({})
    assert {} == parser.parse_path("config.json")
    assert {} == parser.instantiate_classes({})
    assert "{}\n" == parser.dump({})
    parser.save({}, "config.yaml")
    with open("config.yaml") as f:
        assert "{}\n", f.read()


def test_ActionPath(tmp_cwd):
    os.mkdir(os.path.join(tmp_cwd, "example"))
    rel_yaml_file = os.path.join("..", "example", "example.yaml")
    abs_yaml_file = os.path.realpath(os.path.join(tmp_cwd, "example", rel_yaml_file))
    with open(abs_yaml_file, "w") as output_file:
        output_file.write("file: " + rel_yaml_file + "\ndir: " + str(tmp_cwd) + "\n")

    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--cfg", action=ActionConfigFile)
    with catch_warnings(record=True) as w:
        parser.add_argument("--file", action=ActionPath(mode="fr"))
    assert_deprecation_warn(
        w,
        message="ActionPath was deprecated",
        code='ActionPath(mode="fr")',
    )
    parser.add_argument("--dir", action=ActionPath(mode="drw"))
    parser.add_argument("--files", nargs="+", action=ActionPath(mode="fr"))

    cfg = parser.parse_args(["--cfg", abs_yaml_file])
    assert str(tmp_cwd) == os.path.realpath(cfg.dir(absolute=True))
    assert abs_yaml_file == os.path.realpath(cfg.cfg[0](absolute=False))
    assert abs_yaml_file == os.path.realpath(cfg.cfg[0](absolute=True))
    assert rel_yaml_file == cfg.file(absolute=False)
    assert abs_yaml_file == os.path.realpath(cfg.file(absolute=True))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg", abs_yaml_file + "~"]))

    cfg = parser.parse_args(["--cfg", "file: " + abs_yaml_file + "\ndir: " + str(tmp_cwd) + "\n"])
    assert str(tmp_cwd) == os.path.realpath(cfg.dir(absolute=True))
    assert cfg.cfg[0] is None
    assert abs_yaml_file == os.path.realpath(cfg.file(absolute=True))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--cfg", '{"k":"v"}']))

    cfg = parser.parse_args(["--file", abs_yaml_file, "--dir", str(tmp_cwd)])
    assert str(tmp_cwd) == os.path.realpath(cfg.dir(absolute=True))
    assert abs_yaml_file == os.path.realpath(cfg.file(absolute=True))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--dir", abs_yaml_file]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--file", str(tmp_cwd)]))

    cfg = parser.parse_args(["--files", abs_yaml_file, abs_yaml_file])
    assert isinstance(cfg.files, list)
    assert 2 == len(cfg.files)
    assert abs_yaml_file == os.path.realpath(cfg.files[-1](absolute=True))

    pytest.raises(TypeError, lambda: parser.add_argument("--op1", action=ActionPath))
    pytest.raises(
        ValueError,
        lambda: parser.add_argument("--op3", action=ActionPath(mode="+")),
    )
    pytest.raises(
        ValueError,
        lambda: parser.add_argument("--op4", type=str, action=ActionPath(mode="fr")),
    )


def test_ActionPath_skip_check(tmp_cwd):
    parser = ArgumentParser(exit_on_error=False)
    with catch_warnings(record=True) as w:
        parser.add_argument("--file", action=ActionPath(mode="fr", skip_check=True))
    assert_deprecation_warn(
        w,
        message="skip_check parameter of Path was deprecated",
        code='ActionPath(mode="fr", skip_check=True)',
    )
    cfg = parser.parse_args(["--file=not-exist"])
    assert isinstance(cfg.file, Path)
    assert str(cfg.file) == "not-exist"
    assert parser.dump(cfg) == "file: not-exist\n"
    assert repr(cfg.file).startswith("Path_fr_skip_check")


def test_ActionPath_dump(tmp_cwd):
    parser = ArgumentParser()
    with catch_warnings(record=True):
        parser.add_argument("--path", action=ActionPath(mode="fc"))
    cfg = parser.parse_string("path: path")
    assert parser.dump(cfg) == "path: path\n"

    parser = ArgumentParser()
    parser.add_argument("--paths", nargs="+", action=ActionPath(mode="fc"))
    cfg = parser.parse_args(["--paths", "path1", "path2"])
    assert parser.dump(cfg) == "paths:\n- path1\n- path2\n"


def test_ActionPath_nargs_questionmark(tmp_cwd):
    parser = ArgumentParser()
    parser.add_argument("val", type=int)
    with catch_warnings(record=True):
        parser.add_argument("path", nargs="?", action=ActionPath(mode="fc"))
    assert None is parser.parse_args(["1"]).path
    assert None is not parser.parse_args(["2", "file"]).path


def test_Path_attr_set(tmp_cwd):
    path = Path("file", "fc")
    with catch_warnings(record=True) as w:
        path.rel_path = "file"
        path.abs_path = os.path.join(tmp_cwd, "file")
        path.skip_check = False
        path.cwd = str(tmp_cwd)
        assert "Path objects are not meant to be mutable" in str(w[-1].message)
    with catch_warnings(record=True) as w:
        assert path.rel_path == "file"
        assert path.abs_path == os.path.join(tmp_cwd, "file")
        assert path.skip_check is False
        assert "Path objects are not meant to be mutable" in str(w[-1].message)


def test_ActionPathList(tmp_cwd):
    tmpdir = os.path.join(tmp_cwd, "subdir")
    os.mkdir(tmpdir)
    pathlib.Path(os.path.join(tmpdir, "file1")).touch()
    pathlib.Path(os.path.join(tmpdir, "file2")).touch()
    pathlib.Path(os.path.join(tmpdir, "file3")).touch()
    pathlib.Path(os.path.join(tmpdir, "file4")).touch()
    pathlib.Path(os.path.join(tmpdir, "file5")).touch()
    list_file = os.path.join(tmpdir, "files.lst")
    list_file2 = os.path.join(tmpdir, "files2.lst")
    list_file3 = os.path.join(tmpdir, "files3.lst")
    list_file4 = os.path.join(tmpdir, "files4.lst")
    with open(list_file, "w") as output_file:
        output_file.write("file1\nfile2\nfile3\nfile4\n")
    with open(list_file2, "w") as output_file:
        output_file.write("file5\n")
    pathlib.Path(list_file3).touch()
    with open(list_file4, "w") as output_file:
        output_file.write("file1\nfile2\nfile6\n")

    parser = ArgumentParser(prog="app", exit_on_error=False)
    with catch_warnings(record=True) as w:
        parser.add_argument("--list", nargs="+", action=ActionPathList(mode="fr", rel="list"))
    assert_deprecation_warn(
        w,
        message="ActionPathList was deprecated",
        code="ActionPathList(mode=",
    )
    parser.add_argument("--list_cwd", action=ActionPathList(mode="fr", rel="cwd"))

    cfg = parser.parse_args(["--list", list_file])
    assert 4 == len(cfg.list)
    assert ["file1", "file2", "file3", "file4"] == [str(x) for x in cfg.list]

    cfg = parser.parse_args(["--list", list_file, list_file2])
    assert 5 == len(cfg.list)
    assert ["file1", "file2", "file3", "file4", "file5"] == [str(x) for x in cfg.list]

    assert 0 == len(parser.parse_args(["--list", list_file3]).list)

    cwd = os.getcwd()
    os.chdir(tmpdir)
    cfg = parser.parse_args(["--list_cwd", list_file])
    assert 4 == len(cfg.list_cwd)
    assert ["file1", "file2", "file3", "file4"] == [str(x) for x in cfg.list_cwd]
    os.chdir(cwd)

    pytest.raises(ArgumentError, lambda: parser.parse_args(["--list"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--list", list_file4]))
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--list", "no-such-file"]))

    pytest.raises(ValueError, lambda: parser.add_argument("--op1", action=ActionPathList))
    pytest.raises(
        ValueError,
        lambda: parser.add_argument("--op2", action=ActionPathList(mode="fr"), nargs="*"),
    )
    pytest.raises(
        ValueError,
        lambda: parser.add_argument("--op3", action=ActionPathList(mode="fr", rel=".")),
    )


@skip_if_docstring_parser_unavailable
def test_import_import_docstring_parse():
    from jsonargparse._optionals import import_docstring_parser

    with catch_warnings(record=True) as w:
        from jsonargparse.optionals import import_docstring_parse

    assert_deprecation_warn(
        w,
        message="Only use the public API",
        code="from jsonargparse.optionals import import_docstring_parse",
    )
    assert import_docstring_parse is import_docstring_parser


def test_import_from_deprecated():
    import jsonargparse.deprecated as deprecated

    with catch_warnings(record=True) as w:
        func = deprecated.set_url_support

    assert_deprecation_warn(
        w,
        message="Only use the public API",
        code="func = deprecated.set_url_support",
    )
    assert func is set_url_support


@pytest.mark.parametrize(
    ["module", "attr"],
    [
        ("actions", "ActionYesNo"),
        ("cli", "CLI"),
        ("core", "ArgumentParser"),
        ("formatters", "DefaultHelpFormatter"),
        ("jsonnet", "ActionJsonnet"),
        ("jsonschema", "ActionJsonSchema"),
        ("link_arguments", "ArgumentLinking"),
        ("loaders_dumpers", "set_loader"),
        ("namespace", "Namespace"),
        ("signatures", "compose_dataclasses"),
        ("typehints", "lazy_instance"),
        ("util", "Path"),
        ("parameter_resolvers", "ParamData"),
    ],
)
def test_import_from_module(module, attr):
    module = import_module(f"jsonargparse.{module}")
    with catch_warnings(record=True) as w:
        getattr(module, attr)
    assert_deprecation_warn(
        w,
        message="Only use the public API",
        code="getattr(module, attr)",
    )


@pytest.mark.skipif(not jsonnet_support, reason="jsonnet package is required")
def test_action_jsonnet_ext_vars(parser):
    with catch_warnings(record=True) as w:
        parser.add_argument("--ext_vars", action=ActionJsonnetExtVars())
    assert_deprecation_warn(
        w,
        message="ActionJsonnetExtVars was deprecated",
        code="action=ActionJsonnetExtVars()",
    )
    parser.add_argument("--jsonnet", action=ActionJsonnet(ext_vars="ext_vars"))

    cfg = parser.parse_args(["--ext_vars", '{"param": 123}', "--jsonnet", example_2_jsonnet])
    assert 123 == cfg.jsonnet["param"]
    assert 9 == len(cfg.jsonnet["records"])
    assert "#8" == cfg.jsonnet["records"][-2]["ref"]
    assert 15.5 == cfg.jsonnet["records"][-2]["val"]
