import pytest

from jsonargparse import get_config_read_mode, set_config_read_mode
from jsonargparse.optionals import (
    argcomplete_support,
    docstring_parser_support,
    fsspec_support,
    get_docstring_parse_options,
    import_argcomplete,
    import_docstring_parser,
    import_fsspec,
    import_jsonnet,
    import_jsonschema,
    import_requests,
    import_ruyaml,
    jsonnet_support,
    jsonschema_support,
    ruyaml_support,
    set_docstring_parse_options,
    url_support,
)
from jsonargparse_tests.conftest import skip_if_docstring_parser_unavailable

# jsonschema support


@pytest.mark.skipif(not jsonschema_support, reason="jsonschema package is required")
def test_jsonschema_support_true():
    import_jsonschema("test_jsonschema_support_true")


@pytest.mark.skipif(jsonschema_support, reason="jsonschema package should not be installed")
def test_jsonschema_support_false():
    with pytest.raises(ImportError) as context:
        import_jsonschema("test_jsonschema_support_false")
    assert "test_jsonschema_support_false" in str(context.value)


# jsonnet support


@pytest.mark.skipif(not jsonnet_support, reason="jsonnet package is required")
def test_jsonnet_support_true():
    import_jsonnet("test_jsonnet_support_true")


@pytest.mark.skipif(jsonnet_support, reason="jsonnet package should not be installed")
def test_jsonnet_support_false():
    with pytest.raises(ImportError) as context:
        import_jsonnet("test_jsonnet_support_false")
        assert "test_jsonnet_support_false" in str(context.value)


# requests support


@pytest.mark.skipif(not url_support, reason="requests package is required")
def test_url_support_true():
    import_requests("test_url_support_true")


@pytest.mark.skipif(url_support, reason="requests package should not be installed")
def test_url_support_false():
    with pytest.raises(ImportError) as context:
        import_requests("test_url_support_false")
        assert "test_url_support_false" in str(context.value)


# docstring-parser support


@skip_if_docstring_parser_unavailable
def test_docstring_parser_support_true():
    import_docstring_parser("test_docstring_parser_support_true")


@pytest.mark.skipif(docstring_parser_support, reason="docstring-parser package should not be installed")
def test_docstring_parser_support_false():
    with pytest.raises(ImportError) as context:
        import_docstring_parser("test_docstring_parser_support_false")
        assert "test_docstring_parser_support_false" in str(context.value)


@skip_if_docstring_parser_unavailable
def test_docstring_parse_options():
    from docstring_parser import DocstringStyle

    options = get_docstring_parse_options()
    options["style"] = None
    options = get_docstring_parse_options()

    for style in [DocstringStyle.NUMPYDOC, DocstringStyle.GOOGLE]:
        set_docstring_parse_options(style=style)
        assert options["style"] == style
    with pytest.raises(ValueError):
        set_docstring_parse_options(style="invalid")

    assert options["attribute_docstrings"] is False
    for attribute_docstrings in [True, False]:
        set_docstring_parse_options(attribute_docstrings=attribute_docstrings)
        assert options["attribute_docstrings"] is attribute_docstrings
    with pytest.raises(ValueError):
        set_docstring_parse_options(attribute_docstrings="invalid")


# argcomplete support


@pytest.mark.skipif(not argcomplete_support, reason="argcomplete package is required")
def test_argcomplete_support_true():
    import_argcomplete("test_argcomplete_support_true")


@pytest.mark.skipif(argcomplete_support, reason="argcomplete package should not be installed")
def test_argcomplete_support_false():
    with pytest.raises(ImportError) as context:
        import_argcomplete("test_argcomplete_support_false")
        assert "test_argcomplete_support_false" in str(context.value)


# fsspec support


@pytest.mark.skipif(not fsspec_support, reason="fsspec package is required")
def test_fsspec_support_true():
    import_fsspec("test_fsspec_support_true")


@pytest.mark.skipif(fsspec_support, reason="fsspec package should not be installed")
def test_fsspec_support_false():
    with pytest.raises(ImportError) as context:
        import_fsspec("test_fsspec_support_false")
        assert "test_fsspec_support_false" in str(context.value)


# ruyaml support


@pytest.mark.skipif(not ruyaml_support, reason="ruyaml package is required")
def test_ruyaml_support_true():
    import_ruyaml("test_ruyaml_support_true")


@pytest.mark.skipif(ruyaml_support, reason="ruyaml package should not be installed")
def test_ruyaml_support_false():
    with pytest.raises(ImportError) as context:
        import_ruyaml("test_ruyaml_support_false")
        assert "test_ruyaml_support_false" in str(context.value)


# config read mode tests


@pytest.mark.skipif(not url_support, reason="requests package is required")
def test_config_read_mode_url_support_true():
    assert "fr" == get_config_read_mode()
    set_config_read_mode(urls_enabled=True)
    assert "fur" == get_config_read_mode()
    set_config_read_mode(urls_enabled=False)
    assert "fr" == get_config_read_mode()


@pytest.mark.skipif(url_support, reason="request package should not be installed")
def test_config_read_mode_url_support_false():
    assert "fr" == get_config_read_mode()
    with pytest.raises(ImportError):
        set_config_read_mode(urls_enabled=True)
    assert "fr" == get_config_read_mode()
    set_config_read_mode(urls_enabled=False)
    assert "fr" == get_config_read_mode()


@pytest.mark.skipif(not fsspec_support, reason="fsspec package is required")
def test_config_read_mode_fsspec_support_true():
    assert "fr" == get_config_read_mode()
    set_config_read_mode(fsspec_enabled=True)
    assert "fsr" == get_config_read_mode()
    set_config_read_mode(fsspec_enabled=False)
    assert "fr" == get_config_read_mode()


@pytest.mark.skipif(fsspec_support, reason="fsspec package should not be installed")
def test_config_read_mode_fsspec_support_false():
    assert "fr" == get_config_read_mode()
    with pytest.raises(ImportError):
        set_config_read_mode(fsspec_enabled=True)
    assert "fr" == get_config_read_mode()
    set_config_read_mode(fsspec_enabled=False)
    assert "fr" == get_config_read_mode()
