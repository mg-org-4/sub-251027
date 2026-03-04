from mjr_am_backend.features.index.vector_searcher import VectorSearcher


def test_semantic_query_variants_fr_animal_expands_to_english() -> None:
    variants = VectorSearcher._semantic_query_variants("animal")
    assert variants
    assert variants[0] == "animal"
    assert any("dog" in v for v in variants)


def test_semantic_query_variants_keeps_original_only_when_no_mapping() -> None:
    variants = VectorSearcher._semantic_query_variants("landscape")
    assert variants == ["landscape"]
