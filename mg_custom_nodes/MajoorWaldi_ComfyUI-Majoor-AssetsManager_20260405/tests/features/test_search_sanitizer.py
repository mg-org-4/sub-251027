from mjr_am_backend.features.index.searcher import IndexSearcher


def _mk_searcher():
    return IndexSearcher.__new__(IndexSearcher)


def test_sanitize_fts_query_uppercase_terms_are_lowercased():
    s = _mk_searcher()
    out = s._sanitize_fts_query("HELLO WORLD")
    assert out == "hello* world*"


def test_sanitize_fts_query_reserved_tokens_are_removed():
    s = _mk_searcher()
    out = s._sanitize_fts_query("DOG AND CAT OR NOT")
    assert out == "dog* cat*"


def test_sanitize_fts_query_only_reserved_falls_back_to_browse():
    s = _mk_searcher()
    out = s._sanitize_fts_query("AND OR NOT NEAR")
    assert out == ""


def test_sanitize_fts_query_normalizes_diacritics():
    s = _mk_searcher()
    out = s._sanitize_fts_query("caf\u00e9 fleurs")
    assert out == "cafe* fleurs*"


def test_sanitize_fts_query_keeps_quoted_phrase_clause():
    s = _mk_searcher()
    out = s._sanitize_fts_query('"delicate asymmetrical petals" rose')
    assert out == '"delicate asymmetrical petals" OR rose*'


def test_sanitize_fts_query_long_text_uses_or_clause():
    s = _mk_searcher()
    out = s._sanitize_fts_query(
        "a single organic flower is blooming at the center of the frame with delicate asymmetrical petals"
    )
    assert " OR " in out
    assert "single*" in out
    assert "petals*" in out


def test_sanitize_fts_query_removes_english_stopwords():
    s = _mk_searcher()
    out = s._sanitize_fts_query("the flower in the garden with light")
    assert out == "flower* garden* light*"


def test_sanitize_fts_query_removes_french_stopwords():
    s = _mk_searcher()
    out = s._sanitize_fts_query("une fleur dans le jardin avec lumiere")
    assert out == "fleur* jardin* lumiere*"
