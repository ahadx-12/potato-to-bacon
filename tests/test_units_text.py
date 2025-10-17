from potatobacon.parser.units_text import parse_units_text


def test_parse_units_text_tolerates_comments_and_commas():
    text = """
    m: kg
    v: m/s   # velocity
    E: J,
    # ignored
    bad line
    : missing
    """

    result = parse_units_text(text)
    assert result.units == {"m": "kg", "v": "m/s", "E": "J"}
    assert len(result.warnings) == 2
    assert "missing ':'" in result.warnings[0]
    assert "empty variable" in result.warnings[1]


def test_parse_units_text_empty_input():
    result = parse_units_text(None)
    assert result.units == {}
    assert result.warnings == []
