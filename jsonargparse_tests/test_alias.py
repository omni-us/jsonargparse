def test_alias():
    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--foo', '--bar')
    parsed = parser.parse_string('foo: "aaa"')
    assert parsed.foo == 'aaa'
    parsed = parser.parse_string('bar: "bbb"')
    assert parsed.foo == 'bbb'
    parsed = parser.parse_args(['--bar', 'ccc'])
    assert parsed.foo == 'ccc'
