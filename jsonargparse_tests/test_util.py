from __future__ import annotations

import logging
import os
import pathlib
import stat
import zipfile
from calendar import Calendar
from importlib import import_module
from random import Random
from unittest.mock import patch

import pytest

from jsonargparse import (
    ArgumentParser,
    LoggerProperty,
    Namespace,
    Path,
    capture_parser,
    class_from_function,
    null_logger,
)
from jsonargparse._optionals import (
    docstring_parser_support,
    fsspec_support,
    reconplogger_support,
    url_support,
)
from jsonargparse._util import (
    CaptureParserException,
    current_path_dir,
    get_import_path,
    import_object,
    object_path_serializer,
    parse_url,
    register_unresolvable_import_paths,
    unique,
)
from jsonargparse_tests.conftest import (
    capture_logs,
    get_parser_help,
    is_posix,
    responses_activate,
    responses_available,
    skip_if_fsspec_unavailable,
    skip_if_requests_unavailable,
    skip_if_responses_unavailable,
)

if responses_available:
    import responses
if fsspec_support:
    import fsspec


# path tests


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    cwd = os.getcwd()
    tmp_path = tmp_path_factory.mktemp("paths_fixture")
    os.chdir(tmp_path)

    try:
        paths = Namespace()
        paths.tmp_path = tmp_path
        paths.file_rw = file_rw = pathlib.Path("file_rw")
        paths.file_r = file_r = pathlib.Path("file_r")
        paths.file_ = file_ = pathlib.Path("file_")
        paths.dir_rwx = dir_rwx = pathlib.Path("dir_rwx")
        paths.dir_rx = dir_rx = pathlib.Path("dir_rx")
        paths.dir_x = dir_x = pathlib.Path("dir_x")
        paths.dir_file_rx = dir_file_rx = dir_x / "file_rx"

        file_r.write_text("file contents")
        file_rw.touch()
        file_.touch()
        dir_rwx.mkdir()
        dir_rx.mkdir()
        dir_x.mkdir()
        dir_file_rx.touch()

        file_rw.chmod(stat.S_IREAD | stat.S_IWRITE)
        file_r.chmod(stat.S_IREAD)
        file_.chmod(0)
        dir_file_rx.chmod(stat.S_IREAD | stat.S_IEXEC)
        dir_rwx.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        dir_rx.chmod(stat.S_IREAD | stat.S_IEXEC)
        dir_x.chmod(stat.S_IEXEC)

        yield paths
    finally:
        dir_x.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        os.chdir(cwd)


def test_path_init(paths):
    path1 = Path(paths.file_rw, "frw")
    path2 = Path(path1)
    assert path1.cwd == path2.cwd
    assert path1.absolute == path2.absolute
    assert path1.relative == path2.relative
    assert path1.is_url == path2.is_url


def test_path_init_failures(paths):
    pytest.raises(TypeError, lambda: Path(True))
    pytest.raises(ValueError, lambda: Path(paths.file_rw, "-"))
    pytest.raises(ValueError, lambda: Path(paths.file_rw, "frr"))


def test_path_cwd(paths):
    path = Path("file_rx", mode="fr", cwd=(paths.tmp_path / paths.dir_x))
    assert path.cwd == Path("file_rx", mode="fr", cwd=path.cwd).cwd


def test_path_empty_mode(paths):
    path = Path("does_not_exist", "")
    assert path() == str(paths.tmp_path / "does_not_exist")


def test_path_pathlike(paths):
    path = Path(paths.file_rw)
    assert isinstance(path, os.PathLike)
    assert os.fspath(path) == str(paths.tmp_path / paths.file_rw)
    assert os.path.dirname(path) == str(paths.tmp_path)


def test_path_equality_operator(paths):
    path1 = Path(paths.file_rw)
    path2 = Path(paths.tmp_path / paths.file_rw)
    assert path1 == path2
    assert Path("123", "fc") != 123


def test_path_file_access_mode(paths):
    Path(paths.file_rw, "frw")
    Path(paths.file_r, "fr")
    Path(paths.file_, "f")
    Path(paths.dir_file_rx, "fr")
    if is_posix:
        pytest.raises(TypeError, lambda: Path(paths.file_rw, "fx"))
        pytest.raises(TypeError, lambda: Path(paths.file_, "fr"))
    pytest.raises(TypeError, lambda: Path(paths.file_r, "fw"))
    pytest.raises(TypeError, lambda: Path(paths.dir_file_rx, "fw"))
    pytest.raises(TypeError, lambda: Path(paths.dir_rx, "fr"))
    pytest.raises(TypeError, lambda: Path("file_ne", "fr"))


def test_path_dir_access_mode(paths):
    Path(paths.dir_rwx, "drwx")
    Path(paths.dir_rx, "drx")
    Path(paths.dir_x, "dx")
    if is_posix:
        pytest.raises(TypeError, lambda: Path(paths.dir_rx, "dw"))
        pytest.raises(TypeError, lambda: Path(paths.dir_x, "dr"))
    pytest.raises(TypeError, lambda: Path(paths.file_r, "dr"))


def test_path_get_content(paths):
    assert "file contents" == Path(paths.file_r, "fr").get_content()
    assert "file contents" == Path(f"file://{paths.tmp_path}/{paths.file_r}", "fr").get_content()
    assert "file contents" == Path(f"file://{paths.tmp_path}/{paths.file_r}", "ur").get_content()


def test_path_create_mode(paths):
    Path(paths.file_rw, "fcrw")
    Path(paths.tmp_path / "file_c", "fc")
    Path(paths.tmp_path / "not_existing_dir" / "file_c", "fcc")
    Path(paths.dir_rwx, "dcrwx")
    Path(paths.tmp_path / "dir_c", "dc")
    if is_posix:
        pytest.raises(TypeError, lambda: Path(paths.dir_rx / "file_c", "fc"))
        pytest.raises(TypeError, lambda: Path(paths.dir_rx / "dir_c", "dc"))
        pytest.raises(TypeError, lambda: Path(paths.dir_rx / "not_existing_dir" / "file_c", "fcc"))
    pytest.raises(TypeError, lambda: Path(paths.file_rw, "dc"))
    pytest.raises(TypeError, lambda: Path(paths.dir_rwx, "fc"))
    pytest.raises(TypeError, lambda: Path(paths.dir_rwx / "ne" / "file_c", "fc"))


def test_path_complement_modes(paths):
    pytest.raises(TypeError, lambda: Path(paths.file_rw, "fW"))
    pytest.raises(TypeError, lambda: Path(paths.file_rw, "fR"))
    pytest.raises(TypeError, lambda: Path(paths.dir_rwx, "dX"))
    pytest.raises(TypeError, lambda: Path(paths.file_rw, "F"))
    pytest.raises(TypeError, lambda: Path(paths.dir_rwx, "D"))


def test_path_invalid_modes(paths):
    pytest.raises(ValueError, lambda: Path(paths.file_rw, True))
    pytest.raises(ValueError, lambda: Path(paths.file_rw, "≠"))
    pytest.raises(ValueError, lambda: Path(paths.file_rw, "fd"))
    if url_support:
        pytest.raises(ValueError, lambda: Path(paths.file_rw, "du"))


def test_path_class_hidden_methods(paths):
    path = Path(paths.file_rw, "frw")
    assert path(False) == str(paths.file_rw)
    assert path(True) == str(paths.tmp_path / paths.file_rw)
    assert path() == str(paths.tmp_path / paths.file_rw)
    assert str(path) == str(paths.file_rw)
    assert path.__repr__().startswith("Path_frw(")


def test_path_tilde_home(paths):
    home_env = "USERPROFILE" if os.name == "nt" else "HOME"
    with patch.dict(os.environ, {home_env: str(paths.tmp_path)}):
        home = Path("~", "dr")
        path = Path(os.path.join("~", paths.file_rw), "frw")
        assert str(home) == "~"
        assert str(path) == os.path.join("~", paths.file_rw)
        assert home() == str(paths.tmp_path)
        assert path() == os.path.join(paths.tmp_path, paths.file_rw)


# url tests


@pytest.mark.parametrize(
    ["url", "scheme", "path"],
    [
        ("https://eg.com:8080/eg", "https://", "eg.com:8080/eg"),
        ("dask::s3://bucket/key", "dask::s3://", "bucket/key"),
        ("filecache::s3://bucket/key", "filecache::s3://", "bucket/key"),
        (
            "zip://*.csv::simplecache::gcs://bucket/file.zip",
            "zip://*.csv::simplecache::gcs://",
            "bucket/file.zip",
        ),
        (
            "simplecache::zip://*.csv::gcs://bucket/file.zip",
            "simplecache::zip://*.csv::gcs://",
            "bucket/file.zip",
        ),
        (
            "zip://existing.txt::file://file1.zip",
            "zip://existing.txt::file://",
            "file1.zip",
        ),
        ("file.txt", None, None),
        ("../../file.txt", None, None),
        ("/tmp/file.txt", None, None),
    ],
)
def test_parse_url(url, scheme, path):
    url_data = parse_url(url)
    if scheme is None:
        assert url_data is None
    else:
        assert url_data.scheme == scheme
        assert url_data.url_path == path


@skip_if_responses_unavailable
@responses_activate
def test_path_url_200():
    existing = "http://example.com/existing-url"
    existing_body = "url contents"
    responses.add(responses.GET, existing, status=200, body=existing_body)
    responses.add(responses.HEAD, existing, status=200)
    path = Path(existing, mode="ur")
    assert existing_body == path.get_content()


@skip_if_responses_unavailable
@responses_activate
def test_path_url_404():
    nonexisting = "http://example.com/non-existing-url"
    responses.add(responses.HEAD, nonexisting, status=404)
    with pytest.raises(TypeError) as ctx:
        Path(nonexisting, mode="ur")
    ctx.match("404")


# fsspec tests


def create_zip(zip_path, file_path):
    ziph = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    ziph.write(file_path)
    ziph.close()


@skip_if_fsspec_unavailable
def test_path_fsspec_zipfile(tmp_cwd):
    existing = pathlib.Path("existing.txt")
    existing_body = "existing content"
    existing.write_text(existing_body)
    nonexisting = "non-existing.txt"
    zip1_path = "file1.zip"
    zip2_path = pathlib.Path("file2.zip")
    create_zip(zip1_path, existing)
    create_zip(zip2_path, existing)
    zip2_path.chmod(0)

    path = Path(f"zip://{existing}::file://{zip1_path}", mode="sr")
    assert existing_body == path.get_content()

    with pytest.raises(TypeError) as ctx:
        Path(f"zip://{nonexisting}::file://{zip1_path}", mode="sr")
    ctx.match("does not exist")

    if is_posix:
        with pytest.raises(TypeError) as ctx:
            Path(f"zip://{existing}::file://{zip2_path}", mode="sr")
        ctx.match("exists but no permission to access")


@skip_if_fsspec_unavailable
def test_path_fsspec_memory():
    file_content = "content in memory"
    memfile = "memfile.txt"
    path = Path(f"memory://{memfile}", mode="sw")
    with fsspec.open(path, "w") as f:
        f.write(file_content)
    assert file_content == path.get_content()


def test_path_fsspec_invalid_mode():
    with pytest.raises(ValueError) as ctx:
        Path("memory://file.txt", mode="ds")
    ctx.match('Both modes "d" and "s" not possible')


@skip_if_fsspec_unavailable
def test_path_fsspec_invalid_scheme():
    with pytest.raises(TypeError) as ctx:
        Path("unsupported://file.txt", mode="sr")
    ctx.match("not readable")


# path open tests


def test_path_open_local(tmp_cwd):
    pathlib.Path("file.txt").write_text("content")
    path = Path("file.txt", mode="fr")
    with path.open() as f:
        assert "content" == f.read()


@skip_if_responses_unavailable
@responses_activate
def test_path_open_url():
    url = "http://example.com/file.txt"
    responses.add(responses.GET, url, status=200, body="content")
    responses.add(responses.HEAD, url, status=200)
    path = Path(url, mode="ur")
    with path.open() as f:
        assert "content" == f.read()


@skip_if_fsspec_unavailable
def test_path_open_fsspec():
    path = Path("memory://nested/file.txt", mode="sc")
    with fsspec.open(path, "w") as f:
        f.write("content")
    path = Path("memory://nested/file.txt", mode="sr")
    with path.open() as f:
        assert "content" == f.read()


# path relative path context tests


@skip_if_requests_unavailable
def test_path_relative_path_context_url():
    path1 = Path("http://example.com/nested/path/file1.txt", mode="u")
    with path1.relative_path_context() as dir:
        assert "http://example.com/nested/path" == dir
        path2 = Path("../file2.txt", mode="u")
        assert path2() == "http://example.com/nested/file2.txt"


@skip_if_fsspec_unavailable
def test_relative_path_context_fsspec(tmp_cwd, subtests):
    local_path = tmp_cwd / "file0.txt"
    local_path.write_text("zero")

    mem_path = Path("memory://one/two/file1.txt", mode="sc")
    with fsspec.open(mem_path, "w") as f:
        f.write("one")

    with Path("memory://one/two/three/file1.txt", mode="sc").relative_path_context() as dir1:
        with subtests.test("get current path dir"):
            assert "memory://one/two/three" == dir1
            assert "memory://one/two/three" == current_path_dir.get()

        with subtests.test("absolute local path"):
            path0 = Path(local_path, mode="fr")
            assert "zero" == path0.get_content()

        with subtests.test("relative fsspec path"):
            path1 = Path("../file1.txt", mode="fsr")
            assert "one" == path1.get_content()
            assert str(path1) == "../file1.txt"
            assert path1() == "memory://one/two/file1.txt"
            assert path1._url_data is not None

        with subtests.test("nested fsspec dir"):
            with path1.relative_path_context() as dir2:
                assert "memory://one/two" == dir2
                path2 = Path("four/five/six/../file2.txt", mode="fsc")
                assert path2() == "memory://one/two/four/five/file2.txt"

        with subtests.test("non-fsspec path"):
            path3 = Path("file3.txt", mode="fc")
            assert path3() == str(tmp_cwd / "file3.txt")

    with subtests.test("current path dir unset"):
        assert current_path_dir.get() is None


# logger property tests


class WithLogger(LoggerProperty):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


log_message = "testing log message"


def test_logger_true():
    test = WithLogger(logger=True)
    assert test.logger.handlers[0].level == logging.WARNING
    assert test.logger.name == ("plain_logger" if reconplogger_support else "WithLogger")
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
    pytest.raises(ValueError, lambda: import_object(True))
    pytest.raises(ValueError, lambda: import_object("jsonargparse-tests.os"))


def test_get_import_path():
    assert get_import_path(ArgumentParser) == "jsonargparse.ArgumentParser"
    assert get_import_path(ArgumentParser.merge_config) == "jsonargparse.ArgumentParser.merge_config"
    from email.mime.base import MIMEBase

    assert get_import_path(MIMEBase) == "email.mime.base.MIMEBase"
    from dataclasses import MISSING

    assert get_import_path(MISSING) == "dataclasses.MISSING"


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
    cal.a1 = a1  # type: ignore
    cal.a2 = a2  # type: ignore
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
