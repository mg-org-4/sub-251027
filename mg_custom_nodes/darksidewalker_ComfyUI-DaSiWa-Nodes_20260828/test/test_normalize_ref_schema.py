"""Tests for normalize_ref_schema cross-mode compatibility."""
import copy
from nodes.helper_minimax_h3_prompt_builder import (
    default_builder_state,
    normalize_ref_schema,
)


def test_ref2va_defaults_contain_v1_keys_after_normalize():
    """REF2VA defaults use v2 keys; after normalize, v1 keys must exist."""
    state = default_builder_state("REF2VA")
    normalize_ref_schema(state["ref"])
    ref = state["ref"]
    # v2 originals
    assert "subject_definitions" in ref
    assert "summary" in ref
    assert "retention_analysis" in ref
    assert "detailed_description" in ref
    # v1 backfills required by prompt_payload
    assert "subject_defs" in ref
    assert "summary_text" in ref
    assert "retention" in ref
    assert "style_line" in ref
    assert "detail" in ref


def test_non_ref2va_with_old_v1_ref_still_works():
    """Non-REF2VA mode merges old v1-style ref data without losing keys."""
    state = default_builder_state("FL2VA")
    old_v1_ref = {
        "subject_defs": [{"text": "A cyberpunk cityscape"}, {"text": "Neon rain"}],
        "summary_types": ["keyframe completion"],
        "summary_text": "Smooth transition between two frames.",
        "retention": [
            {"label": "city", "context": "Picture 1", "marker": "fully_preserved", "note": "keep layout"}
        ],
        "style_line": "Cinematic lighting, volumetric fog.",
        "detail": "Rain on glass, reflections on wet pavement.",
    }
    state["ref"] = {**default_builder_state("FL2VA")["ref"], **old_v1_ref}
    normalize_ref_schema(state["ref"])
    ref = state["ref"]
    # v1 keys preserved
    assert ref["subject_defs"] == old_v1_ref["subject_defs"]
    assert ref["summary_text"] == old_v1_ref["summary_text"]
    assert len(ref["retention"]) >= 1
    assert ref["style_line"] == old_v1_ref["style_line"]
    assert ref["detail"] == old_v1_ref["detail"]
    # v2 keys derived
    assert "subject_definitions" in ref and "cyberpunk cityscape" in ref["subject_definitions"].lower()
    assert "summary" in ref and "keyframe completion" in ref["summary"]
    assert "retention_analysis" in ref and "city" in ref["retention_analysis"]
    assert "detailed_description" in ref and "Cinematic lighting" in ref["detailed_description"]


def test_ref2va_mode_with_legacy_v1_data():
    """REF2VA mode loaded from workflow saved in non-REF2VA era (v1 keys only)."""
    state = default_builder_state("REF2VA")
    legacy_v1 = {
        "subject_defs": [{"text": "Portrait of a woman"}],
        "summary_types": ["reference generation"],
        "summary_text": "Generate consistent portrait.",
        "retention": [],
        "style_line": "",
        "detail": "",
    }
    state["ref"] = {**default_builder_state("REF2VA")["ref"], **legacy_v1}
    normalize_ref_schema(state["ref"])
    ref = state["ref"]
    assert ref["subject_defs"] == legacy_v1["subject_defs"]
    assert "subject_definitions" in ref and "woman" in ref["subject_definitions"].lower()
    assert "summary" in ref and "reference generation" in ref["summary"]
    assert "summary_text" in ref


def test_ref2va_mode_with_new_v2_data():
    """REF2VA mode with pure v2 keys still produces v1 fallbacks."""
    state = default_builder_state("REF2VA")
    v2_only = {
        "subject_definitions": "Golden retriever puppy\nMountain lake background",
        "summary": "[reference generation] Consistent dog portrait.",
        "retention_analysis": "dog: fully_preserved - keep breed features\nbackground: weak_reference - approximate mood",
        "detailed_description": "Soft natural light, shallow depth of field.\nPuppy looking at camera with curiosity.",
    }
    state["ref"] = {**default_builder_state("REF2VA")["ref"], **v2_only}
    normalize_ref_schema(state["ref"])
    ref = state["ref"]
    # v2 originals untouched
    assert ref["subject_definitions"] == v2_only["subject_definitions"]
    assert ref["summary"] == v2_only["summary"]
    assert ref["retention_analysis"] == v2_only["retention_analysis"]
    assert ref["detailed_description"] == v2_only["detailed_description"]
    # v1 derived
    assert isinstance(ref["subject_defs"], list) and len(ref["subject_defs"]) == 2
    assert ref["summary_text"] == "Consistent dog portrait."
    assert isinstance(ref["retention"], list) and len(ref["retention"]) == 2
    assert ref["style_line"] == "Soft natural light, shallow depth of field."
    assert ref["detail"] == "Puppy looking at camera with curiosity."


def test_empty_ref_does_not_crash():
    """Empty ref dict should normalize without errors."""
    ref = {}
    normalize_ref_schema(ref)
    assert isinstance(ref["subject_defs"], list)
    assert isinstance(ref["retention"], list)
    assert isinstance(ref["summary_text"], str)
    assert isinstance(ref["style_line"], str)
    assert isinstance(ref["detail"], str)


def test_roundtrip_v1_to_v2_to_v1():
    """Converting v1→v2→v1 should preserve core content."""
    original = {
        "subject_defs": [{"text": "Subject A"}, {"text": "Subject B"}],
        "summary_types": ["video editing"],
        "summary_text": "Edit sequence smoothly.",
        "retention": [
            {"label": "face", "context": "Shot 1", "marker": "fully_preserved", "note": "critical"}
        ],
        "style_line": "HDR cinematic.",
        "detail": "Slow motion, color graded.",
    }
    state = default_builder_state("FL2VA")
    state["ref"] = {**default_builder_state("FL2VA")["ref"], **copy.deepcopy(original)}
    normalize_ref_schema(state["ref"])
    ref = state["ref"]

    # v1 keys still correct
    assert ref["subject_defs"] == original["subject_defs"]
    assert ref["summary_text"] == original["summary_text"]
    assert len(ref["retention"]) >= 1
    assert ref["style_line"] == original["style_line"]
    assert ref["detail"] == original["detail"]

    # v2 keys contain equivalent info
    assert "Subject A" in ref["subject_definitions"]
    assert "video editing" in ref["summary"]
    assert "face" in ref["retention_analysis"]
    assert "HDR cinematic" in ref["detailed_description"]
