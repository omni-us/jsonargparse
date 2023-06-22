from __future__ import annotations

import json
import os
from calendar import Calendar
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from jsonargparse import ActionConfigFile, ArgumentError, Namespace
from jsonargparse.typing import Path_drw, Path_fc, Path_fr, path_type
from jsonargparse_tests.conftest import get_parser_help

# stdlib path types tests


def test_pathlib_path(parser, file_r):
    parser.add_argument("--path", type=Path)
    cfg = parser.parse_args([f"--path={file_r}"])
    assert isinstance(cfg.path, Path)
    assert str(cfg.path) == file_r
    assert parser.dump(cfg) == "path: file_r\n"


def test_os_pathlike(parser, file_r):
    parser.add_argument("--path", type=os.PathLike)
    assert file_r == parser.parse_args([f"--path={file_r}"]).path


# jsonargparse path types tests


def test_path_fr(file_r):
    path = Path_fr(file_r)
    assert path == file_r
    assert path() == os.path.realpath(file_r)
    pytest.raises(TypeError, lambda: Path_fr("does_not_exist"))


def test_path_fc_with_kwargs(tmpdir):
    path = Path_fc("some-file.txt", cwd=tmpdir)
    assert path() == os.path.join(tmpdir, "some-file.txt")


def test_path_fr_already_registered():
    assert Path_fr is path_type("fr")


def test_paths_config_relative_absolute(parser, tmp_cwd):
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--file", type=Path_fr)
    parser.add_argument("--dir", type=Path_drw)

    (tmp_cwd / "example").mkdir()
    rel_yaml_file = Path("..", "example", "example.yaml")
    abs_yaml_file = (tmp_cwd / "example" / rel_yaml_file).resolve()
    abs_yaml_file.write_text(f"file: {rel_yaml_file}\ndir: {tmp_cwd}\n")

    cfg = parser.parse_args([f"--cfg={abs_yaml_file}"])
    assert os.path.realpath(tmp_cwd) == os.path.realpath(cfg.dir)
    assert str(rel_yaml_file) == str(cfg.file)
    assert str(abs_yaml_file) == os.path.realpath(cfg.file)

    cfg = parser.parse_args([f"--file={abs_yaml_file}", f"--dir={tmp_cwd}"])
    assert str(abs_yaml_file) == os.path.realpath(cfg.file)
    assert str(tmp_cwd) == os.path.realpath(cfg.dir)

    pytest.raises(ArgumentError, lambda: parser.parse_args([f"--dir={abs_yaml_file}"]))
    pytest.raises(ArgumentError, lambda: parser.parse_args([f"--file={tmp_cwd}"]))


def test_path_fc_nargs_plus(parser, tmp_cwd):
    parser.add_argument("--files", nargs="+", type=Path_fc)
    (tmp_cwd / "subdir").mkdir()
    cfg = parser.parse_args(["--files", "file1", "subdir/file2"])
    assert isinstance(cfg.files, list)
    assert 2 == len(cfg.files)
    assert str(tmp_cwd / "subdir" / "file2") == os.path.realpath(cfg.files[1])


def test_list_path_fc(parser, tmp_cwd):
    parser.add_argument("--paths", type=List[Path_fc])
    cfg = parser.parse_args(['--paths=["file1", "file2"]'])
    assert ["file1", "file2"] == cfg.paths
    assert isinstance(cfg.paths[0], Path_fc)
    assert isinstance(cfg.paths[1], Path_fc)


def test_optional_path_fr(parser, file_r):
    parser.add_argument("--path", type=Optional[Path_fr])
    assert None is parser.parse_args(["--path=null"]).path
    cfg = parser.parse_args([f"--path={file_r}"])
    assert file_r == cfg.path
    assert isinstance(cfg.path, Path_fr)
    pytest.raises(ArgumentError, lambda: parser.parse_args(["--path=not_exist"]))


def test_register_path_dcc_default_path(parser, tmp_cwd):
    path_dcc = path_type("dcc")
    parser.add_argument("--path", type=path_dcc, default=path_dcc("test"))
    cfg = parser.parse_args([])
    assert "path: test\n" == parser.dump(cfg)
    help_str = get_parser_help(parser)
    assert "(type: Path_dcc, default: test)" in help_str


def test_path_dump(parser, tmp_cwd):
    parser.add_argument("--path", type=Path_fc)
    cfg = parser.parse_string("path: path")
    assert parser.dump(cfg) == "path: path\n"


def test_paths_dump(parser, tmp_cwd):
    parser.add_argument("--paths", nargs="+", type=Path_fc)
    cfg = parser.parse_args(["--paths", "path1", "path2"])
    assert parser.dump(cfg) == "paths:\n- path1\n- path2\n"


# enable_path tests


def test_enable_path_dict(parser, tmp_cwd):
    data = {"a": 1, "b": 2, "c": [3, 4]}
    Path("data.yaml").write_text(json.dumps(data))

    parser.add_argument("--data", type=Dict[str, Any], enable_path=True)
    cfg = parser.parse_args(["--data=data.yaml"])
    assert "data.yaml" == str(cfg["data"].pop("__path__"))
    assert data == cfg["data"]
    with pytest.raises(ArgumentError) as ctx:
        parser.parse_args(["--data=does-not-exist.yaml"])
    ctx.match("does-not-exist.yaml either not accessible or invalid")


def test_enable_path_subclass(parser, tmp_cwd):
    cal = {"class_path": "calendar.Calendar"}
    Path("cal.yaml").write_text(json.dumps(cal))

    parser.add_argument("--cal", type=Calendar, enable_path=True)
    cfg = parser.parse_args(["--cal=cal.yaml"])
    init = parser.instantiate_classes(cfg)
    assert isinstance(init["cal"], Calendar)


def test_enable_path_list_path_fr(parser, tmp_cwd, subtests):
    tmpdir = tmp_cwd / "subdir"
    tmpdir.mkdir()
    (tmpdir / "file1").touch()
    (tmpdir / "file2").touch()
    (tmpdir / "file3").touch()
    (tmpdir / "file4").touch()
    (tmpdir / "file5").touch()
    list_file1 = tmpdir / "files1.lst"
    list_file2 = tmpdir / "files2.lst"
    list_file3 = tmpdir / "files3.lst"
    list_file4 = tmpdir / "files4.lst"
    list_file1.write_text("file1\nfile2\nfile3\nfile4\n")
    list_file2.write_text("file5\n")
    list_file3.touch()
    list_file4.write_text("file1\nfile2\nfile6\n")

    parser.add_argument(
        "--list",
        type=List[Path_fr],
        enable_path=True,
    )
    parser.add_argument(
        "--lists",
        nargs="+",
        type=List[Path_fr],
        enable_path=True,
    )

    with subtests.test("paths list from file"):
        cfg = parser.parse_args([f"--list={list_file1}"])
        assert all(isinstance(x, Path_fr) for x in cfg.list)
        assert ["file1", "file2", "file3", "file4"] == [str(x) for x in cfg.list]

    with subtests.test("paths list from stdin"):
        with patch("sys.stdin", StringIO("file1\nfile2\n")):
            with Path_drw("subdir").relative_path_context():
                cfg = parser.parse_args(["--list", "-"])
        assert all(isinstance(x, Path_fr) for x in cfg.list)
        assert ["file1", "file2"] == [str(x) for x in cfg.list]

    with subtests.test("paths list from stdin path not exist"):
        with patch("sys.stdin", StringIO("file1\nfile2\n")):
            with pytest.raises(ArgumentError) as ctx:
                parser.parse_args(["--list", "-"])
            ctx.match("File does not exist")

    with subtests.test("paths list nargs='+' single"):
        cfg = parser.parse_args(["--lists", str(list_file1)])
        assert 1 == len(cfg.lists)
        assert ["file1", "file2", "file3", "file4"] == [str(x) for x in cfg.lists[0]]
        assert all(isinstance(x, Path_fr) for x in cfg.lists[0])

    with subtests.test("paths list nargs='+' multiple"):
        cfg = parser.parse_args(["--lists", str(list_file1), str(list_file2)])
        assert 2 == len(cfg.lists)
        assert ["file1", "file2", "file3", "file4"] == [str(x) for x in cfg.lists[0]]
        assert ["file5"] == [str(x) for x in cfg.lists[1]]

    with subtests.test("paths list nargs='+' empty"):
        cfg = parser.parse_args(["--lists", str(list_file3)])
        assert [[]] == cfg.lists

    with subtests.test("paths list nargs='+' path not exist"):
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--lists", str(list_file4)]))

    with subtests.test("paths list nargs='+' list not exist"):
        pytest.raises(ArgumentError, lambda: parser.parse_args(["--lists", "no-such-file"]))


class Data:
    def __init__(self, path: Optional[os.PathLike] = None):
        pass


def test_enable_path_os_pathlike_subclass_parameter(parser, tmp_cwd):
    data_path = Path("data.json")
    data_path.write_text('{"a": 1}')

    parser.add_argument("--data", type=Data, enable_path=True)

    cfg = parser.parse_args([f"--data={__name__}.Data", f"--data.path={data_path}"])
    assert cfg.data.class_path == f"{__name__}.Data"
    assert cfg.data.init_args == Namespace(path=str(data_path))
