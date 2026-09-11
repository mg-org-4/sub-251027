# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nodes as dg  # noqa: E402
import scripts.dg_to_ltx_prompt_injector as prompt_injector  # noqa: E402


def _packet(prompt: str, metadata: dict | None = None, segments: list[dict] | None = None) -> dict:
    return {
        "ltx_prompt": prompt,
        "ideogram_prompt": "",
        "negative_prompt": "blurry",
        "scene_segments": segments if segments is not None else [{"index": 0, "duration_seconds": 2, "prompt": prompt}],
        "metadata": metadata or {},
    }


def _split(packet_or_text, context=None, target=None):
    text = packet_or_text if isinstance(packet_or_text, str) else json.dumps(packet_or_text)
    return dg.DiffusionGemmaJSONSplitter().split(
        text,
        gemma_context=context,
        target_profile_config=target,
    )


def _metadata(result):
    return json.loads(result[4])


def _media_context(user_prompt: str, media: dict, visual_description: str = ""):
    metadata = dict(media)
    if visual_description:
        metadata["visual_description"] = visual_description
    return dg.GemmaContext(
        user_prompt=user_prompt,
        images=object(),
        source=str(metadata.get("source", "image")),
        media_metadata=metadata,
        visual_description=visual_description,
        warnings=[],
    )


def _h3_ref_context(user_prompt: str, manifest: str):
    normalized_manifest = dg._normalize_minimax_h3_reference_manifest(manifest)
    definitions = dg._minimax_h3_reference_definitions(normalized_manifest)
    return dg.GemmaContext(
        user_prompt=user_prompt,
        source="minimax_h3_ref2va",
        media_metadata={
            "minimax_h3_mode": "ref2va",
            "minimax_h3_reference_manifest": normalized_manifest,
            "minimax_h3_reference_tags": [tag for tag, _description in definitions],
            "minimax_h3_reference_manifest_reasons": dg._minimax_h3_reference_manifest_validation_reasons(
                normalized_manifest
            ),
        },
    )


def _assert_ready(result):
    assert result[8] is True, _metadata(result)
    assert result[12] is True, _metadata(result)
    assert result[0], "expected executable LTX prompt"


def _assert_blocked(result, expected_reason: str):
    metadata = _metadata(result)
    assert result[8] is False, metadata
    assert result[12] is False, metadata
    assert result[0] == "", metadata
    assert expected_reason in metadata.get("blocked_reasons", []), metadata


def test_readiness_gates():
    ready = _split(
        _packet(
            "In a wide shot, the camera remains static at eye level as a red kite rises over a windy beach at sunset."
        )
    )
    _assert_ready(ready)

    segmented = _split(
        _packet(
            "In a medium shot, the camera remains static at eye level as two simple beats unfold.",
            segments=[
                {"index": 0, "duration_seconds": 2, "prompt": "First beat."},
                {"index": 1, "duration_seconds": 3, "prompt": "Second beat."},
            ],
        )
    )
    assert segmented[7] == "2, 3", segmented[7]

    salvaged = _split("ltx_prompt: A clean plain text prompt that was not valid JSON.")
    _assert_blocked(salvaged, "salvaged_output")

    fallback = _split("", context=dg.GemmaContext(user_prompt="make a cinematic shot", media_metadata={}))
    _assert_blocked(fallback, "template_fallback")

    metadata_only_context = _media_context(
        "Use this image as an LTX prompt.",
        {
            "source": "image",
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": False,
        },
    )
    metadata_only_claim = _split(_packet("A sunlit daylight courtyard with people in the background."), metadata_only_context)
    _assert_blocked(metadata_only_claim, "metadata_only_media_reference_request")

    night_context = _media_context(
        "Use the supplied visual description.",
        {
            "source": "image",
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": False,
        },
        "Nighttime locked-off static camera view under neon streetlights.",
    )
    model_media = {
        "media": {
            "source": "image",
            "pixels_sent_to_backend": True,
            "visual_description": "daylight exterior",
        }
    }
    night_ready = _split(
        _packet(
            "In a wide shot from an eye-level front view, the camera remains locked and static on a nighttime neon street scene.",
            model_media,
        ),
        night_context,
    )
    _assert_ready(night_ready)
    night_metadata = _metadata(night_ready)
    assert "Nighttime" in night_metadata["media"]["visual_description"], night_metadata
    assert "model_reported_media" in night_metadata, night_metadata

    daylight_contradiction = _split(_packet("A bright daylight street with a blue sky.", model_media), night_context)
    _assert_blocked(daylight_contradiction, "visual_description_contradiction_daylight_claim")

    camera_contradiction = _split(_packet("The camera slowly zooms in on the neon street.", model_media), night_context)
    _assert_blocked(camera_contradiction, "visual_description_contradiction_camera_motion")


def test_prompt_packet_schema():
    schema = json.loads((ROOT / "schemas" / "prompt_packet.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    # Packets created before MiniMax H3 support remain valid.
    valid_packet = _packet("A clean prompt.", {"ready_for_generation": True, "blocked_reasons": []})
    validator.validate(valid_packet)

    h3_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    h3_packet["minimax_h3_prompt"] = (
        "integrated_multimodal_description: [Shot 1] A locked-off wide shot holds on a rain-darkened city roof.\n\n"
        "overall_soundscape: Wind moves across the roof while distant traffic hums below.\n\n"
        "non_diegetic_music: N/A"
    )
    validator.validate(h3_packet)

    invalid_h3_packet = dict(h3_packet, minimax_h3_prompt=["not", "a", "string"])
    assert list(validator.iter_errors(invalid_h3_packet)), "schema should reject a non-string H3 prompt"

    invalid_packet = {"ltx_prompt": "missing required fields"}
    errors = sorted(validator.iter_errors(invalid_packet), key=lambda item: item.path)
    assert errors, "schema should reject incomplete prompt packets"


def test_minimax_h3_contract_and_splitter_compatibility():
    profile_choices = dg.DiffusionGemmaTargetProfile.INPUT_TYPES()["required"]["target_profile"][0]
    assert "minimax_h3" in profile_choices, profile_choices

    target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=20.0,
        audio_mode="auto_scene_audio",
    )
    model_prompt = dg._build_model_prompt(
        "A runner crosses rain-darkened rooftops in four hard-cut shots.",
        dg.DEFAULT_MASTER_PROMPT,
        "minimax_h3",
        {},
        target_duration_seconds=20.0,
        audio_mode="auto_scene_audio",
        thinking_mode="on",
    )
    lowered = model_prompt.lower()
    assert dg._target_profile_config_to_dict(target)["target_duration_seconds"] == 20.0
    assert "minimax_h3_prompt" in model_prompt
    assert "integrated_multimodal_description" in model_prompt
    assert "overall_soundscape" in model_prompt
    assert "non_diegetic_music" in model_prompt
    assert "20" in model_prompt, "20-second H3 requests must not be clamped to the published 15-second guidance"
    assert "target profile: minimax_h3" in lowered
    assert "composed for 1:1" not in lowered, "the Ideogram-only aspect control must not silently frame H3"
    for phrase in (
        "binding directing specification",
        "continuity ledger",
        "generic real animals",
        "prior shot's resulting position and state",
        "an incoming cut, shot size, angle, subject framing",
        "event-specific sound immediately beside its visible cause",
        "the final frame holds on",
    ):
        assert phrase in lowered, phrase

    inferred_duration_prompt = dg._build_model_prompt(
        "Make a 20-second rooftop chase.",
        dg.DEFAULT_MASTER_PROMPT,
        "minimax_h3",
        {},
        target_duration_seconds=0.0,
        audio_mode="auto_scene_audio",
        thinking_mode="on",
    )
    assert "planned for exactly 20.000 seconds" in inferred_duration_prompt, inferred_duration_prompt

    h3_prompt = (
        "integrated_multimodal_description: [Shot 1] A high side angle tracks a runner toward a wet rooftop edge.\n\n"
        "[Shot 2] At 00:10.000, the camera cuts to a wide profile as the runner clears the gap.\n\n"
        "overall_soundscape: Strong wind, rapid footsteps, and distant city traffic surround the action.\n\n"
        "non_diegetic_music: A low percussion pulse builds under the chase."
    )
    packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    packet["minimax_h3_prompt"] = h3_prompt
    result = _split(packet, context=None, target=target)

    assert dg.DiffusionGemmaJSONSplitter.RETURN_NAMES[:13] == (
        "ltx_prompt",
        "ideogram_prompt",
        "negative_prompt",
        "aspect_ratio",
        "metadata_json",
        "scene_segments_json",
        "local_prompts",
        "segment_lengths",
        "is_valid",
        "resolution_selector_preset",
        "resolution_width",
        "resolution_height",
        "ready_for_generation",
    )
    assert dg.DiffusionGemmaJSONSplitter.RETURN_NAMES[13:] == (
        "minimax_h3_prompt",
        "candidate_ltx_prompt",
        "candidate_negative_prompt",
        "candidate_minimax_h3_prompt",
    )
    assert len(result) == 17, result
    assert result[13] == h3_prompt, result[13]
    assert "\n\noverall_soundscape:" in result[13], result[13]
    assert "\n\nnon_diegetic_music:" in result[13], result[13]
    assert result[0] == "", result[0]
    assert result[1] == "", result[1]
    assert result[2] == "", result[2]
    assert result[16] == h3_prompt, result[16]
    assert result[8] is True, _metadata(result)
    assert result[12] is True, _metadata(result)
    assert result[14] == "", result[14]
    assert result[15] == "", result[15]

    model_style_h3_prompt = (
        "[Shot 1] A low-angle tracking shot follows a runner across a rain-darkened rooftop. "
        "[Shot 2] At 00:04.500 the camera cuts to a wide profile as the runner clears the gap.\n\n"
        "Strong wind, rapid footsteps, splashing puddles, and distant city traffic surround the action.\n\n"
        "A restrained electronic score uses deep cello pulses and measured percussion, building with each leap."
    )
    model_style_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    model_style_packet["minimax_h3_prompt"] = model_style_h3_prompt
    model_style_result = _split(model_style_packet, context=None, target=target)
    normalized_h3_prompt = model_style_result[13]
    assert model_style_result[12] is True, _metadata(model_style_result)
    assert normalized_h3_prompt.startswith("integrated_multimodal_description: [Shot 1]"), normalized_h3_prompt
    assert "[Shot 2] At 00:04.500, the camera cuts to" in normalized_h3_prompt, normalized_h3_prompt
    assert "\n\noverall_soundscape: Strong wind" in normalized_h3_prompt, normalized_h3_prompt
    assert "\n\nnon_diegetic_music: A restrained electronic score" in normalized_h3_prompt, normalized_h3_prompt
    assert dg._normalize_minimax_h3_prompt(normalized_h3_prompt) == normalized_h3_prompt

    unlabeled_cut_prompt = (
        "integrated_multimodal_description: [Shot 1] A locked-off wide shot holds on a rain-darkened roof. "
        "At 00:04.500, the camera cuts to a low-angle tracking shot beside the runner. "
        "At 00:09.000, the camera cuts to an overhead static hold as the runner lands.\n\n"
        "overall_soundscape: Wind, footsteps, and splashing water follow the action.\n\n"
        "non_diegetic_music: Low percussion builds under the cuts."
    )
    normalized_unlabeled_cut_prompt = dg._normalize_minimax_h3_prompt(unlabeled_cut_prompt)
    assert "[Shot 2] At 00:04.500," in normalized_unlabeled_cut_prompt, normalized_unlabeled_cut_prompt
    assert "[Shot 3] At 00:09.000," in normalized_unlabeled_cut_prompt, normalized_unlabeled_cut_prompt
    assert "minimax_h3_unlabeled_cut" not in dg._minimax_h3_prompt_validation_reasons(
        normalized_unlabeled_cut_prompt,
        20.0,
    )

    ambiguous_unlabeled_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    ambiguous_unlabeled_packet["minimax_h3_prompt"] = (
        "[Shot 1] A locked-off view holds on a rooftop.\n\nA second unlabeled visual paragraph follows."
    )
    ambiguous_unlabeled_result = _split(ambiguous_unlabeled_packet, context=None, target=target)
    assert ambiguous_unlabeled_result[13] == "", ambiguous_unlabeled_result[13]
    assert "minimax_h3_missing_integrated_multimodal_description" in _metadata(ambiguous_unlabeled_result)["blocked_reasons"]

    invalid_cases = (
        (
            h3_prompt.replace("[Shot 2] At 00:10.000,", "[10s-20s] Shot 2:"),
            "minimax_h3_noncanonical_time_range",
        ),
        (
            h3_prompt.replace("At 00:10.000,", "At 00:20.000,"),
            "minimax_h3_cut_timestamp_out_of_range",
        ),
        (
            h3_prompt.replace("[Shot 1]", "[Shot 1] At 00:00.000,"),
            "minimax_h3_shot1_timestamp_invalid",
        ),
        (
            h3_prompt.rsplit("non_diegetic_music:", 1)[0] + "non_diegetic_music:",
            "minimax_h3_empty_non_diegetic_music",
        ),
        (
            h3_prompt.replace(
                "[Shot 1] A high side angle tracks a runner toward a wet rooftop edge.",
                "[Shot 1]",
            ),
            "minimax_h3_empty_visual_timeline",
        ),
        (
            h3_prompt.replace("[Shot 2]", "[shot 2]"),
            "minimax_h3_shot_syntax_invalid",
        ),
        (
            h3_prompt.replace("\n\noverall_soundscape:", "\n\nCamera: handheld.\n\noverall_soundscape:"),
            "minimax_h3_extra_top_level_field",
        ),
        (
            h3_prompt.replace(
                "A high side angle tracks a runner toward a wet rooftop edge.",
                "A high side angle tracks a runner who says <d>[English] Jump now.</d> near a wet rooftop edge.",
            ),
            "minimax_h3_dialogue_speaker_invalid",
        ),
    )
    for invalid_prompt, expected_reason in invalid_cases:
        invalid_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
        invalid_packet["minimax_h3_prompt"] = invalid_prompt
        invalid_result = _split(invalid_packet, context=None, target=target)
        invalid_metadata = _metadata(invalid_result)
        assert invalid_result[13] == "", invalid_result[13]
        assert expected_reason in invalid_metadata["blocked_reasons"], invalid_metadata

    visual_only_target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=20.0,
        audio_mode="visual_only",
    )
    visual_only_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    visual_only_packet["minimax_h3_prompt"] = h3_prompt
    visual_only_result = _split(visual_only_packet, context=None, target=visual_only_target)
    assert visual_only_result[13] == "", visual_only_result[13]
    assert "minimax_h3_visual_only_audio_fields_invalid" in _metadata(visual_only_result)["blocked_reasons"]

    requested_transition_context = dg.GemmaContext(user_prompt="Use a cross-dissolve for the second shot.", media_metadata={})
    requested_transition_prompt = h3_prompt.replace("the camera cuts to", "the camera cross-dissolves to")
    requested_transition_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    requested_transition_packet["minimax_h3_prompt"] = requested_transition_prompt
    requested_transition_result = _split(requested_transition_packet, requested_transition_context, target)
    assert requested_transition_result[13] == requested_transition_prompt, _metadata(requested_transition_result)

    oversized_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    oversized_packet["minimax_h3_prompt"] = h3_prompt.replace(
        "A high side angle tracks a runner toward a wet rooftop edge.",
        "A high side angle tracks a runner toward a wet rooftop edge. " + ("Visible rain streaks across the frame. " * 260),
    )
    oversized_result = _split(oversized_packet, context=None, target=target)
    assert oversized_result[13] == "", oversized_result[13]
    assert "minimax_h3_prompt_truncated" in _metadata(oversized_result)["blocked_reasons"]


def test_minimax_h3_ref2va_contract():
    mode_input = dg.DiffusionGemmaTargetProfile.INPUT_TYPES()["optional"]["minimax_h3_mode"]
    assert mode_input[0] == ["t2va", "ref2va"], mode_input
    assert mode_input[1]["default"] == "t2va", mode_input
    default_target = dg._make_target_profile_config(target_profile="minimax_h3", target_duration_seconds=8.0)
    assert dg._target_profile_config_to_dict(default_target)["minimax_h3_mode"] == "t2va"

    default_t2va_prompt = (
        "integrated_multimodal_description: [Shot 1] A static wide shot holds on a quiet harbor at dawn. "
        "The final frame holds on the anchored boats beneath the pale sky.\n\n"
        "overall_soundscape: Small waves touch the dock while ropes creak in the breeze.\n\n"
        "non_diegetic_music: N/A"
    )
    default_t2va_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    default_t2va_packet["minimax_h3_prompt"] = default_t2va_prompt
    default_t2va_result = _split(default_t2va_packet, target=default_target)
    assert default_t2va_result[13] == default_t2va_prompt, _metadata(default_t2va_result)
    assert default_t2va_result[12] is True, _metadata(default_t2va_result)

    manifest = (
        "<Picture 1>: first-frame composition and primary identity reference for the rooftop courier; fully preserve the face, "
        "short black hair, red waxed-canvas coat, silver satchel, body proportions, and opening screen position.\n"
        "<Audio 1>: synchronized soundtrack reference for <Video 1>; reference its pulse, footstep timing, wind texture, and impact "
        "accents without copying speech.\n"
        "<Video 1>: motion and camera reference; transfer the running cadence, lateral tracking path, hard-cut rhythm, and landing "
        "mechanics, but not the source performer's identity or wardrobe."
    )
    assert dg._minimax_h3_reference_manifest_validation_reasons(manifest) == []
    assert [tag for tag, _description in dg._minimax_h3_reference_definitions(manifest)] == [
        "<Picture 1>",
        "<Audio 1>",
        "<Video 1>",
    ]
    context = _h3_ref_context(
        "Create an eight-second live-action rooftop escape. Use Picture 1 for the courier and opening composition, Video 1 for "
        "motion and camera rhythm, and Audio 1 for timing and sound texture.",
        manifest,
    )
    target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=8.0,
        audio_mode="auto_scene_audio",
        minimax_h3_mode="ref2va",
    )
    model_prompt = dg._build_model_prompt(
        context.user_prompt,
        dg.DEFAULT_MASTER_PROMPT,
        "minimax_h3",
        context.media_metadata,
        target_duration_seconds=8.0,
        audio_mode="auto_scene_audio",
        minimax_h3_mode="ref2va",
        minimax_h3_reference_manifest=manifest,
    )
    assert "Final MiniMax H3 Ref2VA full-reference prompt" in model_prompt
    assert "subject_definitions, summary, retention_analysis, detailed_description" in model_prompt
    assert "a semantic reference, not a guaranteed first-frame or last-frame pixel lock" in model_prompt
    assert manifest in model_prompt
    ref_prompt = (
        "subject_definitions:\n"
        "<Subject 1> is the rooftop courier from <Picture 1>, with an angular face, short black hair, a red waxed-canvas coat, "
        "a silver cross-body satchel, lean body proportions, and black running boots.\n"
        "<Picture 1> is the first frame of [Shot 1], fixing the courier on the left third of a rain-darkened roof with a low "
        "parapet ahead and the blue dusk skyline behind.\n"
        "<Video 1> is the whole-video motion and camera reference for running cadence, lateral tracking, hard-cut rhythm, and the "
        "physical mechanics of the final landing; its performer identity and wardrobe are not transferred.\n"
        "<Audio 1> is the synchronized sound reference for the running pulse, timed footsteps, wind texture, and landing accent; "
        "no source speech is copied.\n\n"
        "summary:\n"
        "[keyframe completion + reference generation + audio reference] The target video begins from <Picture 1> and follows "
        "<Subject 1> through a three-shot rooftop escape, transferring the motion, camera path, and cut rhythm of <Video 1> while "
        "using <Audio 1> only as timing and sound-texture guidance.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the angular face, short black hair, red coat, "
        "silver satchel, lean proportions, and black boots remain unchanged.\n"
        "<Picture 1> ([Shot 1] first frame): fully_preserved - the opening subject placement, wet parapet, and blue dusk skyline "
        "are retained as the exact opening composition.\n"
        "<Video 1> (running cadence, camera path, cut rhythm, and landing mechanics): attribute_transfer - its temporal behavior "
        "is transferred to <Subject 1> without copying the source performer or wardrobe.\n"
        "<Audio 1>: reference - its pulse, footstep timing, wind texture, and landing accent guide newly generated audio without "
        "copying speech or the original signal.\n\n"
        "detailed_description:\n"
        "The target video uses realistic live-action practical film photography with an anamorphic lens, restrained cool-blue dusk "
        "grading, shallow depth of field, fine film grain, and rain-slick textures. The courier identity, wardrobe, weather, and "
        "skyline geography remain continuous through all three hard-cut shots.\n"
        "[Shot 1] The shot begins from <Picture 1> exactly: a medium-wide side view places <Subject 1> on the left third of the "
        "rain-darkened roof, with the angular face in profile, short black hair damp, red coat darkened at the shoulders, and silver "
        "satchel against the right hip. A low parapet crosses the middle distance and the blue skyline recedes through mist. After a "
        "readable hold, <Subject 1> drives forward into the cadence referenced from <Video 1>, pushing off with the left boot. The "
        "camera trucks right smoothly at the runner's speed, keeping the torso centered while foreground vents slide left with clear "
        "parallax. Each newly generated boot impact follows <Audio 1>; wet rubber hits the roof, puddles splash, coat fabric snaps, "
        "and the runner exhales without speaking. The courier reaches the parapet with the right foot planted and eyes on the next roof.\n"
        "[Shot 2] At 00:02.750, the camera hard cuts to a wide profile view that follows the cut rhythm of <Video 1>. <Subject 1> "
        "springs from the planted right foot, clears the parapet, and crosses the narrow alley in one continuous jump. The red coat "
        "opens behind the lean silhouette, the satchel lifts while its strap remains across the chest, and both boots stay visible "
        "against the skyline. The camera tracks right at fast speed with medium amplitude, matching the body without orbiting, zooming, "
        "or changing screen direction. Warm windows and wet reflections streak through shallow focus below. Wind rises around the "
        "airborne body and the pulse referenced from <Audio 1> tightens without copied speech. The courier extends both legs toward "
        "the far roof and braces the left arm, making the landing preparation visible before contact.\n"
        "[Shot 3] At 00:05.500, the camera hard cuts to a low three-quarter tracking view beside the destination roof. <Subject 1> "
        "lands with the mechanics referenced from <Video 1>: both boots contact in sequence, knees compress, the left hand touches "
        "the wet surface, and momentum becomes one controlled shoulder roll. The new landing accent follows <Audio 1>, combining a "
        "heavy impact, satchel-buckle clink, fabric scrape, and water spray. The camera shakes slightly on impact, then steadies and "
        "trucks right as the same courier rises without changing face, hair, coat, satchel, proportions, or screen direction. The "
        "courier takes two strides, stops behind a ventilation housing, and looks back across the completed gap while breathing with "
        "closed lips. The final frame holds on <Subject 1> crouched safely on the right third, the satchel at the right hip, with the "
        "empty gap and blue skyline behind.\n\n"
        "overall_soundscape:\n"
        "Cross-roof wind, distant traffic, wet boot impacts, puddle splashes, coat movement, breath, and the final buckle clink are "
        "newly generated, with their cadence and texture guided by <Audio 1>. No dialogue or copied source signal is audible.\n\n"
        "non_diegetic_music:\n"
        "A restrained low electronic pulse follows the tempo reference of <Audio 1>, tightening during the jump and dropping out "
        "on the landing accent before one sustained bass note closes the final hold."
    )
    sections = dg._minimax_h3_ref_sections(ref_prompt)
    assert sections is not None
    assert 350 <= len(sections["detailed_description"].split()) <= 500
    assert dg._normalize_minimax_h3_prompt(ref_prompt, "ref2va") == ref_prompt

    packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    packet["minimax_h3_prompt"] = ref_prompt
    result = _split(packet, context=context, target=target)
    assert result[13] == ref_prompt, _metadata(result)
    assert result[12] is True, _metadata(result)

    invalid_manifests = (
        ("", "minimax_h3_ref_manifest_missing"),
        (
            "<Picture 1>: identity and wardrobe reference for the courier.\n"
            "<Picture 1>: duplicate composition reference for the same courier.",
            "minimax_h3_ref_manifest_duplicate_tag",
        ),
        (
            "<Picture 2>: identity and wardrobe reference for the courier.",
            "minimax_h3_ref_manifest_nonconsecutive_tags",
        ),
        (
            "<Picture 10>: identity and wardrobe reference for the courier.",
            "minimax_h3_ref_manifest_limit_exceeded",
        ),
        (
            "<Audio 1>: voice timing and soundtrack rhythm reference only.",
            "minimax_h3_ref_manifest_audio_only",
        ),
        (
            "<Picture 1>: identity. Color.",
            "minimax_h3_ref_manifest_role_missing",
        ),
        (
            "<Picture 1>: identity and wardrobe reference taken from <Video 1>.",
            "minimax_h3_ref_manifest_undefined_tag",
        ),
    )
    for invalid_manifest, expected_reason in invalid_manifests:
        reasons = dg._minimax_h3_reference_manifest_validation_reasons(invalid_manifest)
        assert expected_reason in reasons, (expected_reason, reasons)
        invalid_context = _h3_ref_context(context.user_prompt, invalid_manifest)
        invalid_result = _split(packet, context=invalid_context, target=target)
        assert invalid_result[13] == "", _metadata(invalid_result)
        assert expected_reason in _metadata(invalid_result)["blocked_reasons"], _metadata(invalid_result)

    invalid_subject_usage_sections = sections.copy()
    invalid_subject_usage_sections["detailed_description"] = sections["detailed_description"].replace(
        "<Subject 1>",
        "the rooftop courier",
    )
    invalid_subject_usage_prompt = dg._rebuild_minimax_h3_ref_prompt(invalid_subject_usage_sections)
    invalid_prompts = (
        (
            default_t2va_prompt,
            {"minimax_h3_ref_sections_invalid"},
        ),
        (
            ref_prompt.replace("<Video 1>", "<Video 2>"),
            {"minimax_h3_ref_undefined_tag"},
        ),
        (
            ref_prompt.replace(
                "<Subject 1> is the rooftop courier from <Picture 1>, with an angular face, short black hair, a red waxed-canvas "
                "coat, a silver cross-body satchel, lean body proportions, and black running boots.",
                "<Subject 1> is <Picture 1>.",
            ),
            {"minimax_h3_ref_subject_definition_invalid"},
        ),
    )
    for invalid_prompt, expected_reasons in invalid_prompts:
        invalid_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
        invalid_packet["minimax_h3_prompt"] = invalid_prompt
        invalid_result = _split(invalid_packet, context=context, target=target)
        invalid_metadata = _metadata(invalid_result)
        assert invalid_result[13] == "", (expected_reasons, invalid_metadata)
        assert expected_reasons.issubset(set(invalid_metadata["blocked_reasons"])), invalid_metadata

    repairable_prompts = (
        invalid_subject_usage_prompt,
        ref_prompt.replace("fully_preserved", "preserved", 1),
        ref_prompt.replace("<Audio 1>", "the soundtrack"),
    )
    for repairable_prompt in repairable_prompts:
        repairable_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
        repairable_packet["minimax_h3_prompt"] = repairable_prompt
        repairable_result = _split(repairable_packet, context=context, target=target)
        repairable_metadata = _metadata(repairable_result)
        assert repairable_result[13], repairable_metadata
        assert repairable_result[12] is True, repairable_metadata
        assert repairable_metadata["blocked_reasons"] == [], repairable_metadata


def test_minimax_h3_ref2va_structure_repair_and_generation_gate():
    manifest = (
        "<Picture 1>: courier identity and wardrobe reference; preserve the face, short black hair, red coat, silver satchel, "
        "lean proportions, and black boots."
    )
    user_prompt = "Create an eight-second 2D hand-drawn cel-animated rooftop run using Picture 1 as the courier reference."
    full_sequence_manifest = manifest + " This is a full-sequence reference, not a keyframe."
    assert dg._minimax_h3_ref_summary_task_types(user_prompt, full_sequence_manifest) == ["reference generation"]
    colon_subject_prompt = (
        "subject_definitions:\n"
        "<Subject 1>: The rooftop courier from <Picture 1>, with an angular face, short black hair, a red waxed-canvas coat, "
        "a silver cross-body satchel, lean body proportions, and black running boots.\n\n"
        "summary:\n"
        "[reference generation] The target video follows <Subject 1> across a rain-darkened rooftop while preserving the identity "
        "and wardrobe defined by <Picture 1>.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - the angular face, short black hair, red coat, silver satchel, lean "
        "proportions, and black boots remain unchanged.\n\n"
        "detailed_description:\n"
        "The target video uses immutable 2D hand-drawn cel animation with crisp inked outlines, flat painted colors, restrained "
        "two-tone shadows, and no live action, photorealism, or 3D CGI.\n"
        "[Shot 1] A static medium-wide side view frames <Subject 1> on the left third of a rain-darkened rooftop, with the angular "
        "face in profile, short black hair damp against the forehead, red waxed-canvas coat darkened at the shoulders, silver satchel "
        "against the right hip, and black boots planted beside a shallow puddle. The courier leans forward, pushes off with the left "
        "boot, and accelerates toward a low parapet while the camera trucks right smoothly at the same speed, keeping the torso "
        "centered as vents slide left with clear parallax. Each footfall throws a small painted splash and the satchel swings once "
        "behind the hip without changing its strap position. Near the parapet, the courier shortens the final stride, plants the "
        "right boot, and stops under control rather than beginning a jump. The camera eases to a static hold. The final frame holds "
        "on <Subject 1> standing safely on the right third, looking across the gap with the same face, hair, coat, satchel, body "
        "proportions, and boots clearly visible.\n\n"
        "overall_soundscape:\n"
        "Cross-roof wind, wet boot impacts, puddle splashes, coat movement, and one quiet breath accompany the run.\n\n"
        "non_diegetic_music:\n"
        "N/A"
    )
    colon_reasons = dg._minimax_h3_ref2va_validation_reasons(
        colon_subject_prompt,
        8.0,
        user_prompt,
        "auto_scene_audio",
        8000,
        manifest,
    )
    assert colon_reasons == [], colon_reasons

    missing_n_music_heading = colon_subject_prompt.replace(
        "non_diegetic_music:",
        "on_diegetic_music:",
        1,
    )
    missing_n_music_repaired = dg._repair_minimax_h3_ref2va_structure(
        missing_n_music_heading,
        user_prompt,
        manifest,
        8.0,
    )
    assert missing_n_music_repaired == colon_subject_prompt, missing_n_music_repaired
    assert dg._minimax_h3_ref2va_validation_reasons(
        missing_n_music_repaired,
        8.0,
        user_prompt,
        "auto_scene_audio",
        8000,
        manifest,
    ) == []

    no_colon_headings = colon_subject_prompt
    for field_name in dg._MINIMAX_H3_REF_FIELDS:
        no_colon_headings = no_colon_headings.replace(f"{field_name}:\n", f"{field_name}\n", 1)
    no_colon_repaired = dg._repair_minimax_h3_ref2va_structure(no_colon_headings, user_prompt, manifest)
    assert dg._minimax_h3_ref_sections(no_colon_repaired) is not None, no_colon_repaired
    assert dg._minimax_h3_ref2va_validation_reasons(
        no_colon_repaired,
        8.0,
        user_prompt,
        "auto_scene_audio",
        8000,
        manifest,
    ) == []

    mixed_headings = colon_subject_prompt
    for field_name in dg._MINIMAX_H3_REF_FIELDS[::2]:
        mixed_headings = mixed_headings.replace(f"{field_name}:\n", f"{field_name}\n", 1)
    mixed_repaired = dg._repair_minimax_h3_ref2va_structure(mixed_headings, user_prompt, manifest)
    assert dg._minimax_h3_ref_sections(mixed_repaired) is not None, mixed_repaired

    colon_context = _h3_ref_context(user_prompt, manifest)
    ref_target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=8.0,
        audio_mode="auto_scene_audio",
        minimax_h3_mode="ref2va",
    )
    colon_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    colon_packet["minimax_h3_prompt"] = colon_subject_prompt
    colon_result = _split(colon_packet, context=colon_context, target=ref_target)
    assert colon_result[12] is True, _metadata(colon_result)
    assert colon_result[13], _metadata(colon_result)

    canonical_sections = dg._minimax_h3_ref_sections(colon_subject_prompt)
    assert canonical_sections is not None
    malformed_prompt = (
        f"{canonical_sections['subject_definitions']}\n"
        "summary[animation, duel]: The target video follows <Subject 1> across a rain-darkened rooftop while preserving the "
        "identity and wardrobe defined by <Picture 1>.\n"
        "retention_analysis:\n"
        "[<Subject 1>: fully_preserved - preserve the angular face, short black hair, red coat, silver satchel, lean proportions, "
        "and black boots.]\n"
        f"detailed_description:\n{canonical_sections['detailed_description']}\n"
        f"overall_soundscape:\n{canonical_sections['overall_soundscape']}\n"
        f"non_diegetic_music:\n{canonical_sections['non_diegetic_music']}"
    )
    repaired_prompt = dg._repair_minimax_h3_ref2va_structure(
        malformed_prompt,
        user_prompt,
        manifest,
    )
    assert repaired_prompt != malformed_prompt
    assert repaired_prompt.startswith("subject_definitions:\n")
    repaired_sections = dg._minimax_h3_ref_sections(repaired_prompt)
    assert repaired_sections is not None, repaired_prompt
    task_prefix = repaired_sections["summary"].split("]", 1)[0].removeprefix("[")
    task_types = [item.strip() for item in task_prefix.split("+")]
    assert task_types and all(task_type in dg._MINIMAX_H3_REF_TASK_TYPES for task_type in task_types), task_types
    assert "<Subject 1>: fully_preserved - " in repaired_sections["retention_analysis"]
    assert "[<Subject 1>" not in repaired_sections["retention_analysis"]
    repaired_reasons = dg._minimax_h3_ref2va_validation_reasons(
        repaired_prompt,
        8.0,
        user_prompt,
        "auto_scene_audio",
        8000,
        manifest,
    )
    structural_reasons = {
        "minimax_h3_ref_sections_invalid",
        "minimax_h3_ref_field_order_invalid",
        "minimax_h3_ref_field_spacing_invalid",
        *(f"minimax_h3_ref_missing_{field_name}" for field_name in dg._MINIMAX_H3_REF_FIELDS),
    }
    assert structural_reasons.isdisjoint(repaired_reasons), repaired_reasons
    assert "minimax_h3_ref_summary_invalid" not in repaired_reasons, repaired_reasons
    assert "minimax_h3_ref_retention_invalid" not in repaired_reasons, repaired_reasons
    assert dg._repair_minimax_h3_ref2va_structure(repaired_prompt, user_prompt, manifest) == repaired_prompt

    canonical_repairable = colon_subject_prompt.replace(
        "[reference generation] ",
        "[keyframe completion + video editing] ",
        1,
    ).replace(
        "[Shot 1] A static medium-wide",
        "[Shot 1] At 00:00.000 A static medium-wide",
        1,
    ).replace(
        "The camera eases to a static hold. The final frame holds ",
        "The camera eases to a static hold. [Shot 2] At 00:06.000 The camera quick cuts to a locked wide view of the courier "
        "beside the parapet. The final frame holds ",
        1,
    )
    canonical_repaired = dg._repair_minimax_h3_ref2va_structure(canonical_repairable, user_prompt, manifest)
    canonical_repaired_sections = dg._minimax_h3_ref_sections(canonical_repaired)
    assert canonical_repaired_sections is not None, canonical_repaired
    assert canonical_repaired_sections["summary"].startswith("[reference generation] ")
    assert "[Shot 1] At " not in canonical_repaired_sections["detailed_description"]
    assert "[Shot 2] At 00:06.000, the camera cuts to a locked wide view" in canonical_repaired_sections["detailed_description"]
    assert "the shot cuts to: The camera quick cuts" not in canonical_repaired_sections["detailed_description"]

    valid_cut_malformed = malformed_prompt.replace(
        "The camera eases to a static hold. The final frame holds ",
        "The camera eases to a static hold. [Shot 2] At 00:06.000, the camera cuts to a locked wide view as the courier stops "
        "beside the parapet. The final frame holds ",
    )
    valid_cut_repaired = dg._repair_minimax_h3_ref2va_structure(valid_cut_malformed, user_prompt, manifest)
    assert "the shot cuts to: the camera cuts to" not in valid_cut_repaired.lower()
    assert valid_cut_repaired.count("[Shot 2] At 00:06.000, the camera cuts to") == 1, valid_cut_repaired

    unlabeled_manifest = (
        "<Picture 1>: protagonist identity and appearance; preserve exactly across every shot; this is a full-sequence reference, "
        "not a first or last frame.\n"
        "<Picture 2>: western town environment, palette, lighting, and 2D animation style; do not copy pictured subject identity; "
        "this is a full-sequence reference, not a keyframe."
    )
    unlabeled_user_prompt = (
        "Create a 20-second 2D hand-drawn animated monster duel in a late-1800s spaghetti-western town square. Use Picture 1 "
        "for the protagonist and Picture 2 for the environment and style. Show clear action and reaction and a resolved final frame."
    )
    unlabeled_prefix_prompt = (
        "<Subject 1>: A yellow trumpet-shaped monster with two oval eyes and three green leaves at its base, from <Picture 1>.\n"
        "<Subject 2>: A late-1800s western town square with adobe buildings, cacti, and dusty ground, from <Picture 2>.\n\n"
        "[keyframe completion + reference generation + video editing] <Subject 1> engages in a monster duel within <Subject 2> "
        "using a 2D animated style.\n\n"
        "<Subject 1>: fully_preserved - The yellow trumpet body, oval eyes, and green leaf base remain consistent.\n"
        "<Subject 2>: fully_preserved - The adobe town, desert lighting, and western atmosphere remain constant.\n\n"
        "The scene uses 2D hand-drawn cel animation with clean linework, soft cel shading, and stylized proportions; no live action, "
        "photorealism, or 3D CGI. Bright warm light casts soft shadows across the dusty town floor.\n\n"
        "[Shot 1] At 00:00.000 A low-angle wide shot reveals <Subject 1> in the center of <Subject 2>. The creature narrows its "
        "eyes and its leaves twitch in the wind while the camera slowly pushes toward its face. Adobe storefronts and a cactus hold "
        "steady behind it, establishing the opponent across the street and the distance between them.\n\n"
        "[Shot 2] At 00:06.000 A quick cut to a close-up of the shadowy desert opponent. <Subject 1> leans forward, its body "
        "vibrating with tension as dust swirls around its base. The locked camera shakes once with a heavy warning stomp while both "
        "characters maintain their screen positions.\n\n"
        "[Shot 3] At 00:12.000 The action explodes as <Subject 1> lunges forward and emits a blast of air. The shockwave crosses "
        "the street and blows dust backward while the camera tracks beside <Subject 1>, keeping the opponent visible beyond the "
        "advancing wave and carrying both positions into the next view.\n\n"
        "[Shot 4] At 00:17.000 The dust settles and <Subject 1> stands victorious in the center of the frame. The camera pulls back "
        "to a wide establishing view as the warm sunset silhouettes the mountains and holds the resolved final composition.\n\n"
        "overall_soundscape: Wind through cacti, dry footfalls, a resonant air blast, and shifting gravel synchronize with the action.\n\n"
        "non_diegetic_music: A tense western whistle builds into a brief orchestral flourish."
    )
    unlabeled_repaired = dg._repair_minimax_h3_ref2va_structure(
        unlabeled_prefix_prompt,
        unlabeled_user_prompt,
        unlabeled_manifest,
    )
    unlabeled_sections = dg._minimax_h3_ref_sections(unlabeled_repaired)
    assert unlabeled_sections is not None, unlabeled_repaired
    assert unlabeled_sections["summary"].startswith("[reference generation] "), unlabeled_sections["summary"]
    assert "keyframe completion" not in unlabeled_sections["summary"]
    assert "video editing" not in unlabeled_sections["summary"]
    assert "2D hand-drawn cel animation" in unlabeled_sections["detailed_description"]
    assert "[Shot 1] At " not in unlabeled_sections["detailed_description"]
    assert "[Shot 2] At 00:06.000, the camera cuts to a close-up" in unlabeled_sections["detailed_description"]
    assert "[Shot 3] At 00:12.000, the shot cuts to: The action explodes" in unlabeled_sections["detailed_description"]
    unlabeled_reasons = dg._minimax_h3_ref2va_validation_reasons(
        unlabeled_repaired,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    false_structural_reasons = {
        "minimax_h3_ref_missing_detailed_description",
        "minimax_h3_ref_subject_usage_invalid",
        "minimax_h3_ref_summary_invalid",
        "minimax_h3_ref_style_opening_missing",
        "minimax_h3_cut_timestamp_invalid",
    }
    assert false_structural_reasons.isdisjoint(unlabeled_reasons), unlabeled_reasons
    assert "minimax_h3_battle_result_missing" in unlabeled_reasons, unlabeled_reasons

    reported_user_prompt = (
        "Create a 20-second 2D hand-drawn cel-animated monster duel using Picture 1 as the protagonist reference and Picture 2 "
        "as the western town environment and animation-style reference. End in a resolved post-duel state."
    )
    reported_no_colon_prompt = (
        "<Subject 1>: A protagonist with pale skin, large yellow rabbit-like ears with white tufts, bright blue eyes, and a simple "
        "smile, wearing a sleeveless pink dress (derived from <Picture 1>).\n"
        "<Subject 2>: A shadowy, monstrous antagonist in a rugged western-style duster coat, introduced for the duel.\n"
        "<Subject 3>: A late-1800s spaghetti-western town square with wooden storefronts, dusty ground, and a sun-drenched "
        "atmosphere (derived from <Picture 2>).\n\n"
        "[keyframe completion + reference generation] The protagonist <Subject 1> engages in a monster-duel within the "
        "<Subject 3> environment using 2D animation style.\n\n"
        "retention_analysis\n"
        "<Subject 1>: fully_preserved - identity, face, ears, and pink dress are maintained throughout the sequence.\n"
        "<Subject 2>: weak_reference - the antagonist is a new addition following the style of <Picture 2>.\n"
        "<Subject 3>: fully_preserved - the town square environment and color palette remain consistent.\n\n"
        "detail_description\n"
        "2D hand-drawn cel animation, preserve drawn linework, cel shading, stylized proportions, and design continuity across every "
        "cut, no live action, photorealism, or 3D CGI. The scene utilizes the high-contrast lighting of a western with deep shadows "
        "and vibrant highlights.\n\n"
        "[Shot 1] At 00:00.000, a low-angle shot captures <Subject 1> standing in the center of the <Subject 3> town square. The wind "
        "blows through <Subject 1>'s large yellow ears. The camera slowly zooms in on <Subject 1>'s intense blue eyes as a shadow "
        "falls across the face.\n\n"
        "[Shot 2] At 00:05.000, a quick cut to a wide shot shows <Subject 2> standing at the end of the street. <Subject 2> draws a "
        "spectral, glowing energy-weapon. The camera pans rapidly between <Subject 1> and <Subject 2> to build tension.\n\n"
        "[Shot 3] At 00:10.000, the duel begins in a medium tracking shot. <Subject 1> lunges forward with superhuman speed, the pink dress fluttering. "
        "<Subject 2> fires a blast of dark energy. <Subject 1> dodges with a roll, sending up clouds of dust from the <Subject 3> "
        "ground.\n\n"
        "[Shot 4] At 00:15.000, <Subject 1> leaps into the air, delivering a powerful strike to <Subject 2>. The impact creates a "
        "shockwave that shakes the nearby wooden buildings. The camera shakes violently with the force of the hit-stop.\n\n"
        "[Shot 5] At 00:18.000, <Subject 2> dissipates into smoke. <Subject 1> lands gracefully in a defiant pose, breathing heavily, "
        "adjusting the pink dress as the sun sets over the <Subject 3>. The camera holds a medium shot.\n\n"
        "overall_soundscape\n"
        "The whistling wind, the creak of wooden signs, heavy boots sliding on dirt, the hum of energy blasts, and a sharp thud upon "
        "impact.\n\n"
        "non_diegetic_music\n"
        "N/A"
    )
    reported_repaired = dg._repair_minimax_h3_ref2va_structure(
        reported_no_colon_prompt,
        reported_user_prompt,
        unlabeled_manifest,
    )
    reported_sections = dg._minimax_h3_ref_sections(reported_repaired)
    assert reported_sections is not None, reported_repaired
    assert reported_repaired.startswith("subject_definitions:\n")
    assert reported_sections["summary"].startswith("[reference generation] "), reported_sections["summary"]
    assert "keyframe completion" not in reported_sections["summary"]
    assert "2D hand-drawn cel animation" in reported_sections["detailed_description"]
    assert "[Shot 1] At " not in reported_sections["detailed_description"]
    assert "[Shot 2] At 00:05.000, the camera cuts to a wide shot" in reported_sections["detailed_description"]
    assert "[Shot 3] At 00:10.000, the shot cuts to: the duel begins" in reported_sections["detailed_description"]
    reported_reasons = dg._minimax_h3_ref2va_validation_reasons(
        reported_repaired,
        20.0,
        reported_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    assert reported_reasons == [], reported_reasons

    subject_alias_sections = dict(reported_sections)
    subject_alias_sections["detailed_description"] = subject_alias_sections["detailed_description"].replace(
        "<Subject 3>",
        "<Picture 2>",
    )
    subject_alias_prompt = dg._rebuild_minimax_h3_ref_prompt(subject_alias_sections)
    assert "minimax_h3_ref_subject_usage_invalid" in dg._minimax_h3_ref2va_validation_reasons(
        subject_alias_prompt,
        20.0,
        reported_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    subject_alias_repaired = dg._repair_minimax_h3_ref2va_structure(
        subject_alias_prompt,
        reported_user_prompt,
        unlabeled_manifest,
    )
    subject_alias_repaired_sections = dg._minimax_h3_ref_sections(subject_alias_repaired)
    assert subject_alias_repaired_sections is not None, subject_alias_repaired
    assert "<Subject 3> (derived from <Picture 2>)" in subject_alias_repaired_sections["detailed_description"]
    assert dg._minimax_h3_ref2va_validation_reasons(
        subject_alias_repaired,
        20.0,
        reported_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    ) == []
    assert dg._repair_minimax_h3_ref2va_structure(
        subject_alias_repaired,
        reported_user_prompt,
        unlabeled_manifest,
    ) == subject_alias_repaired

    ambiguous_alias_sections = dict(subject_alias_sections)
    ambiguous_alias_sections["subject_definitions"] = ambiguous_alias_sections["subject_definitions"].replace(
        "(derived from <Picture 2>)",
        "(derived from <Picture 1> and <Picture 2>)",
        1,
    )
    ambiguous_alias_prompt = dg._rebuild_minimax_h3_ref_prompt(ambiguous_alias_sections)
    ambiguous_alias_repaired = dg._repair_minimax_h3_ref2va_structure(
        ambiguous_alias_prompt,
        reported_user_prompt,
        unlabeled_manifest,
    )
    ambiguous_alias_repaired_sections = dg._minimax_h3_ref_sections(ambiguous_alias_repaired)
    assert ambiguous_alias_repaired_sections is not None, ambiguous_alias_repaired
    assert "<Subject 3>" not in ambiguous_alias_repaired_sections["detailed_description"]
    assert "minimax_h3_ref_subject_usage_invalid" in dg._minimax_h3_ref2va_validation_reasons(
        ambiguous_alias_repaired,
        20.0,
        reported_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )

    reported_ambiguous = reported_no_colon_prompt.replace(
        "[keyframe completion + reference generation] The protagonist",
        "UNLABELED OUTPUT\n\n[keyframe completion + reference generation] The protagonist",
        1,
    )
    reported_ambiguous_repaired = dg._repair_minimax_h3_ref2va_structure(
        reported_ambiguous,
        reported_user_prompt,
        unlabeled_manifest,
    )
    assert dg._minimax_h3_ref_sections(reported_ambiguous_repaired) is None, reported_ambiguous_repaired

    ambiguous_unlabeled = unlabeled_prefix_prompt.replace("<Subject 1>: fully_preserved - ", "<Subject 1>: preserved - ", 1)
    ambiguous_repaired = dg._repair_minimax_h3_ref2va_structure(ambiguous_unlabeled, unlabeled_user_prompt, unlabeled_manifest)
    assert dg._minimax_h3_ref_sections(ambiguous_repaired) is None, ambiguous_repaired

    duplicate_suffix = unlabeled_prefix_prompt.replace(
        "overall_soundscape:",
        "overall_soundscape: First ambience.\n\noverall_soundscape:",
        1,
    )
    duplicate_suffix_repaired = dg._repair_minimax_h3_ref2va_structure(
        duplicate_suffix,
        unlabeled_user_prompt,
        unlabeled_manifest,
    )
    assert dg._minimax_h3_ref_sections(duplicate_suffix_repaired) is None, duplicate_suffix_repaired
    assert "[Shot 4]" in duplicate_suffix_repaired

    split_retention = unlabeled_prefix_prompt.replace(
        "remain consistent.\n<Subject 2>: fully_preserved",
        "remain consistent.\n\n<Subject 2>: fully_preserved",
        1,
    )
    split_retention_repaired = dg._repair_minimax_h3_ref2va_structure(
        split_retention,
        unlabeled_user_prompt,
        unlabeled_manifest,
    )
    split_retention_sections = dg._minimax_h3_ref_sections(split_retention_repaired)
    assert split_retention_sections is not None, split_retention_repaired
    assert "<Subject 1>: fully_preserved" in split_retention_sections["retention_analysis"]
    assert "<Subject 2>: fully_preserved" in split_retention_sections["retention_analysis"]

    split_invalid_retention = split_retention.replace(
        "<Subject 2>: fully_preserved - ",
        "<Subject 2>: preserved - ",
        1,
    )
    split_invalid_repaired = dg._repair_minimax_h3_ref2va_structure(
        split_invalid_retention,
        unlabeled_user_prompt,
        unlabeled_manifest,
    )
    assert dg._minimax_h3_ref_sections(split_invalid_repaired) is None, split_invalid_repaired

    malformed_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    malformed_packet["minimax_h3_prompt"] = malformed_prompt
    malformed_result = _split(malformed_packet, context=colon_context, target=ref_target)
    assert malformed_result[12] is True, _metadata(malformed_result)
    assert malformed_result[13], _metadata(malformed_result)
    assert malformed_result[13] == repaired_prompt, malformed_result[13]
    splitter_sections = dg._minimax_h3_ref_sections(malformed_result[13])
    assert splitter_sections is not None, malformed_result[13]
    assert splitter_sections["summary"].startswith("[reference generation] "), splitter_sections["summary"]
    assert "<Subject 1>: fully_preserved - " in splitter_sections["retention_analysis"]
    assert "[<Subject 1>" not in splitter_sections["retention_analysis"]

    gate = dg.DiffusionGemmaGenerationGate()
    ready_prompt = "A nonempty generation-ready prompt."
    assert gate.gate(ready_prompt, True) == (ready_prompt,)
    blocked_cases = (
        (ready_prompt, False, json.dumps({"blocked_reasons": ["minimax_h3_ref_sections_invalid"]})),
        ("   ", True, ""),
    )
    for prompt, ready_for_generation, metadata_json in blocked_cases:
        try:
            gate.gate(prompt, ready_for_generation, metadata_json)
        except ValueError as exc:
            if metadata_json:
                assert "minimax_h3_ref_sections_invalid" in str(exc), str(exc)
        else:
            raise AssertionError((prompt, ready_for_generation, metadata_json))

    valid_retry_prompt = unlabeled_repaired.replace(
        "The shockwave crosses the street and blows dust backward",
        "The shockwave strikes the opponent, who is knocked backward two steps as dust blows across the street",
    )
    valid_retry_reasons = dg._minimax_h3_ref2va_validation_reasons(
        valid_retry_prompt,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    assert valid_retry_reasons == [], valid_retry_reasons
    range_retry_prompt = (
        valid_retry_prompt.replace("[Shot 1] ", "[Shot 1] 00:00-00:06: ", 1)
        .replace(
            "[Shot 2] At 00:06.000, the camera cuts to ",
            "[Shot 2] 00:06-00:12: ",
            1,
        )
        .replace(
            "[Shot 3] At 00:12.000, the shot cuts to: ",
            "[Shot 3] 00:12-00:17: ",
            1,
        )
        .replace(
            "[Shot 4] At 00:17.000, the shot cuts to: ",
            "[Shot 4] 00:17-00:20: ",
            1,
        )
    )
    range_retry_repaired = dg._repair_minimax_h3_ref2va_structure(
        range_retry_prompt,
        unlabeled_user_prompt,
        unlabeled_manifest,
        20.0,
    )
    assert "[Shot 1] 00:00-00:06" not in range_retry_repaired
    assert "[Shot 2] At 00:06.000, the camera cuts to" in range_retry_repaired
    assert "[Shot 3] At 00:12.000, the camera cuts to" in range_retry_repaired
    assert dg._minimax_h3_ref2va_validation_reasons(
        range_retry_repaired,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    ) == []

    seconds_range_retry_prompt = (
        valid_retry_prompt.replace("[Shot 1] ", "[Shot 1] (00:00-06:00): ", 1)
        .replace(
            "[Shot 2] At 00:06.000, the camera cuts to ",
            "[Shot 2] (06:00-12:00): ",
            1,
        )
        .replace(
            "[Shot 3] At 00:12.000, the shot cuts to: ",
            "[Shot 3] (12:00-17:00): ",
            1,
        )
        .replace(
            "[Shot 4] At 00:17.000, the shot cuts to: ",
            "[Shot 4] (17:00-20:00): ",
            1,
        )
    )
    seconds_range_retry_repaired = dg._repair_minimax_h3_ref2va_structure(
        seconds_range_retry_prompt,
        unlabeled_user_prompt,
        unlabeled_manifest,
        20.0,
    )
    assert "[Shot 1] (00:00-06:00)" not in seconds_range_retry_repaired
    assert "[Shot 2] At 00:06.000, the camera cuts to" in seconds_range_retry_repaired
    assert "[Shot 4] At 00:17.000, the camera cuts to" in seconds_range_retry_repaired
    assert dg._minimax_h3_ref2va_validation_reasons(
        seconds_range_retry_repaired,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    ) == []
    decimal_seconds_range_prompt = (
        valid_retry_prompt.replace("[Shot 1] ", "[Shot 1] 0.0s-6.0s: ", 1)
        .replace(
            "[Shot 2] At 00:06.000, the camera cuts to ",
            "[Shot 2] 6.0-12.0s: ",
            1,
        )
        .replace(
            "[Shot 3] At 00:12.000, the shot cuts to: ",
            "[Shot 3] 12.0s-17.0s: ",
            1,
        )
        .replace(
            "[Shot 4] At 00:17.000, the shot cuts to: ",
            "[Shot 4] 17-20s: ",
            1,
        )
    )
    decimal_seconds_range_repaired = dg._repair_minimax_h3_ref2va_structure(
        decimal_seconds_range_prompt,
        unlabeled_user_prompt,
        unlabeled_manifest,
        20.0,
    )
    assert "[Shot 1] 0.0s-6.0s" not in decimal_seconds_range_repaired
    assert "[Shot 2] At 00:06.000, the camera cuts to" in decimal_seconds_range_repaired
    assert "[Shot 4] At 00:17.000, the camera cuts to" in decimal_seconds_range_repaired
    assert dg._minimax_h3_ref2va_validation_reasons(
        decimal_seconds_range_repaired,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    ) == []

    unsafe_decimal_range_prompt = decimal_seconds_range_prompt.replace(
        "[Shot 2] 6.0-12.0s:",
        "[Shot 2] 7.0-21.0s:",
        1,
    )
    unsafe_decimal_range_repaired = dg._repair_minimax_h3_ref2va_structure(
        unsafe_decimal_range_prompt,
        unlabeled_user_prompt,
        unlabeled_manifest,
        20.0,
    )
    assert "[Shot 2] 7.0-21.0s:" in unsafe_decimal_range_repaired
    unsafe_decimal_reasons = dg._minimax_h3_ref2va_validation_reasons(
        unsafe_decimal_range_repaired,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    assert "minimax_h3_noncanonical_time_range" in unsafe_decimal_reasons
    assert "minimax_h3_cut_timestamp_invalid" in unsafe_decimal_reasons
    malformed_retry_prompt = (
        f"subject_definitions:\n{unlabeled_sections['subject_definitions']}\n\n"
        f"summary:\n{unlabeled_sections['summary']}\n\n"
        f"retention_analysis:\n{unlabeled_sections['retention_analysis']}\n\n"
        "detail_description:\nThe animation remains 2D hand-drawn.\n\n[Shot 1] A wide shot of\n\n"
        "detailed_description:\n\n"
        f"overall_soundscape:\n{unlabeled_sections['overall_soundscape']}\n\n"
        f"non_diegetic_music:\n{unlabeled_sections['non_diegetic_music']}"
    )

    def refinement_response(prompt: str) -> str:
        return (
            f"{dg.FINAL_JSON_OPEN}"
            + json.dumps(
                {
                    "ltx_prompt": "",
                    "ideogram_prompt": "",
                    "minimax_h3_prompt": prompt,
                    "negative_prompt": "",
                    "scene_segments": [],
                    "metadata": {},
                }
            )
            + f"{dg.FINAL_JSON_CLOSE}"
        )

    runtime = dg.RuntimeConfig(
        model_path="mock",
        backend="transformers_inprocess",
        dtype="auto",
        quantization="none",
        local_files_only=True,
        unload_policy="keep_loaded",
        max_memory_gb=1.0,
        status={"ready": True, "supports_pixels": False},
    )
    retry_context = _h3_ref_context(unlabeled_user_prompt, unlabeled_manifest)
    retry_target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=20.0,
        audio_mode="auto_scene_audio",
        minimax_h3_mode="ref2va",
    )

    def run_refinement(outputs: list, generation_context=retry_context, generation_target=retry_target):
        backend_prompts = []
        original_run_backend = dg._run_backend

        def fake_run_backend(_config, prompt, _media_context, _max_new_tokens, node_id=None):
            backend_prompts.append(prompt)
            output = outputs[len(backend_prompts) - 1]
            if isinstance(output, Exception):
                raise output
            return output

        dg._run_backend = fake_run_backend
        try:
            result = dg._run_generation_packet(
                runtime,
                generation_context,
                generation_target,
                runtime_required=True,
                max_new_tokens=2048,
                max_output_chars=8000,
            )
        finally:
            dg._run_backend = original_run_backend
        return result, backend_prompts

    subject_alias_response = refinement_response(subject_alias_prompt)
    subject_alias_context = _h3_ref_context(reported_user_prompt, unlabeled_manifest)
    subject_alias_generation, subject_alias_backend_prompts = run_refinement(
        [subject_alias_response],
        subject_alias_context,
        retry_target,
    )
    assert len(subject_alias_backend_prompts) == 1, len(subject_alias_backend_prompts)
    assert subject_alias_generation[0]["minimax_h3_prompt"] == subject_alias_repaired
    assert subject_alias_generation[6]["ready_for_generation"] is True, subject_alias_generation[6]
    assert "minimax_h3_refinement" not in subject_alias_generation[6], subject_alias_generation[6]

    subject_usage_refinement_prompt = dg._build_minimax_h3_refinement_prompt(
        reported_user_prompt,
        ambiguous_alias_prompt,
        20.0,
        "auto_scene_audio",
        ["minimax_h3_ref_subject_usage_invalid"],
        "ref2va",
        unlabeled_manifest,
    )
    assert "Every defined <Subject N> must have its own retention_analysis line" in subject_usage_refinement_prompt
    assert "Never substitute its source <Picture N> or <Video N> tag" in subject_usage_refinement_prompt
    assert "use its <Picture N> or <Video N> source only as provenance" in subject_usage_refinement_prompt

    timestamp_refinement_prompt = dg._build_minimax_h3_refinement_prompt(
        unlabeled_user_prompt,
        range_retry_prompt,
        20.0,
        "auto_scene_audio",
        ["minimax_h3_cut_timestamp_invalid"],
        "ref2va",
        unlabeled_manifest,
    )
    assert "Shot 1 must begin exactly with '[Shot 1]' and no timestamp" in timestamp_refinement_prompt
    assert "Never write start-end ranges" in timestamp_refinement_prompt
    assert "'[Shot N] At MM:SS.mmm, the camera cuts to ...'" in timestamp_refinement_prompt

    initial_response = refinement_response(unlabeled_repaired)
    malformed_response = refinement_response(malformed_retry_prompt)
    valid_response = refinement_response(valid_retry_prompt)
    range_response = refinement_response(range_retry_prompt)
    mode_mismatch_prompt = (
        f"integrated_multimodal_description:\n{unlabeled_sections['detailed_description']}\n\n"
        f"overall_soundscape:\n{unlabeled_sections['overall_soundscape']}\n\n"
        f"on_diegetic_music:\n{unlabeled_sections['non_diegetic_music']}"
    )
    improving_retry_prompt = unlabeled_repaired.replace(
        "2D hand-drawn cel animation",
        "2D hand-drawn cel animation with stable ink contours",
        1,
    )
    improving_reasons = dg._minimax_h3_ref2va_validation_reasons(
        improving_retry_prompt,
        20.0,
        unlabeled_user_prompt,
        "auto_scene_audio",
        8000,
        unlabeled_manifest,
    )
    assert improving_reasons, improving_retry_prompt
    assert not set(improving_reasons).intersection(dg._MINIMAX_H3_CATASTROPHIC_STRUCTURE_REASONS), improving_reasons
    range_first_pass, range_first_pass_prompts = run_refinement([initial_response, range_response])
    assert len(range_first_pass_prompts) == 2, len(range_first_pass_prompts)
    assert range_first_pass[0]["minimax_h3_prompt"] == range_retry_repaired
    assert range_first_pass[6]["ready_for_generation"] is True, range_first_pass[6]
    range_first_pass_refinement = range_first_pass[6]["minimax_h3_refinement"]
    assert range_first_pass_refinement["accepted_attempt"] == 1, range_first_pass_refinement

    first_pass_success, first_pass_prompts = run_refinement([initial_response, valid_response])
    assert len(first_pass_prompts) == 2, len(first_pass_prompts)
    first_pass_refinement = first_pass_success[6]["minimax_h3_refinement"]
    assert first_pass_refinement["accepted_attempt"] == 1, first_pass_refinement
    assert "retry_attempted" not in first_pass_refinement, first_pass_refinement

    improving_success, improving_success_prompts = run_refinement(
        [
            refinement_response(mode_mismatch_prompt),
            refinement_response(improving_retry_prompt),
            valid_response,
        ]
    )
    assert len(improving_success_prompts) == 3, len(improving_success_prompts)
    assert "stable ink contours" in improving_success_prompts[2]
    assert "\\non_diegetic_music:" not in improving_success_prompts[2]
    assert "minimax_h3_ref_missing_subject_definitions" not in improving_success_prompts[2]
    assert improving_success[6]["ready_for_generation"] is True, improving_success[6]
    improving_success_refinement = improving_success[6]["minimax_h3_refinement"]
    assert improving_success_refinement["retry_base"] == "attempt_1", improving_success_refinement
    assert improving_success_refinement["attempts"][0]["selected_as_retry_base"] is True

    retry_success, retry_success_prompts = run_refinement([initial_response, malformed_response, valid_response])
    assert len(retry_success_prompts) == 3, len(retry_success_prompts)
    assert "This is the one bounded retry" in retry_success_prompts[2]
    assert "The shockwave crosses the street and blows dust backward" in retry_success_prompts[2]
    assert "minimax_h3_battle_result_missing" in retry_success_prompts[2]
    assert "minimax_h3_ref_detailed_description_short" not in retry_success_prompts[2]
    assert retry_success[0]["minimax_h3_prompt"] == valid_retry_prompt
    assert retry_success[6]["ready_for_generation"] is True, retry_success[6]
    success_refinement = retry_success[6]["minimax_h3_refinement"]
    assert success_refinement["accepted"] is True, success_refinement
    assert success_refinement["accepted_attempt"] == 2, success_refinement
    assert success_refinement["retry_base"] == "initial", success_refinement

    retry_failure, retry_failure_prompts = run_refinement([initial_response, malformed_response, malformed_response])
    assert len(retry_failure_prompts) == 3, len(retry_failure_prompts)
    assert retry_failure[0]["minimax_h3_prompt"] == unlabeled_repaired
    assert retry_failure[6]["ready_for_generation"] is False, retry_failure[6]
    assert "minimax_h3_battle_result_missing" in retry_failure[6]["blocked_reasons"], retry_failure[6]
    failure_refinement = retry_failure[6]["minimax_h3_refinement"]
    assert failure_refinement["accepted"] is False, failure_refinement
    assert len(failure_refinement["attempts"]) == 2, failure_refinement
    assert failure_refinement["best_candidate_attempt"] == 0, failure_refinement

    improving_failure, _improving_failure_prompts = run_refinement(
        [
            refinement_response(mode_mismatch_prompt),
            refinement_response(improving_retry_prompt),
            malformed_response,
        ]
    )
    assert improving_failure[6]["ready_for_generation"] is False, improving_failure[6]
    improving_failure_refinement = improving_failure[6]["minimax_h3_refinement"]
    assert improving_failure_refinement["best_candidate_attempt"] == 1, improving_failure_refinement
    assert improving_failure_refinement["candidate_reasons"] == improving_reasons, improving_failure_refinement
    assert improving_failure_refinement["last_candidate_reasons"] != improving_reasons, improving_failure_refinement
    assert improving_failure_refinement["attempts"][0]["selected_as_retry_base"] is True

    semantic_rank = dg._minimax_h3_refinement_candidate_rank(
        colon_subject_prompt,
        [
            "minimax_h3_shot_camera_unspecified",
            "minimax_h3_requested_final_hold_missing",
            "minimax_h3_unresolved_final_state",
            "minimax_h3_battle_result_missing",
            "minimax_h3_orphaned_sound_event",
        ],
        True,
    )
    incomplete_rank = dg._minimax_h3_refinement_candidate_rank(
        "subject_definitions:\npartial",
        [
            "minimax_h3_ref_missing_summary",
            "minimax_h3_ref_missing_retention_analysis",
            "minimax_h3_ref_missing_detailed_description",
            "minimax_h3_ref_missing_overall_soundscape",
        ],
        True,
    )
    assert semantic_rank < incomplete_rank, (semantic_rank, incomplete_rank)
    assert dg._minimax_h3_refinement_candidate_rank(colon_subject_prompt, [], False) > semantic_rank
    assert dg._minimax_h3_refinement_candidate_rank(colon_subject_prompt, [], True, "salvaged") > semantic_rank

    exception_failure, exception_prompts = run_refinement([initial_response, RuntimeError("mock refinement failure")])
    assert len(exception_prompts) == 2, len(exception_prompts)
    assert exception_failure[0]["minimax_h3_prompt"] == unlabeled_repaired
    assert exception_failure[6]["ready_for_generation"] is False, exception_failure[6]
    exception_refinement = exception_failure[6]["minimax_h3_refinement"]
    assert "mock refinement failure" in exception_refinement["error"], exception_refinement
    assert "backend_seconds" in exception_refinement["attempts"][0], exception_refinement


def test_minimax_h3_reference_context_preserves_picture_batch_with_video():
    import torch

    class FakeVideo:
        def __init__(self, images, frame_rate=3.0):
            self.images = images
            self.frame_rate = frame_rate

        def get_duration(self):
            return 1.0

        def get_frame_count(self):
            return int(self.images.shape[0])

        def get_dimensions(self):
            return (int(self.images.shape[2]), int(self.images.shape[1]))

        def get_components(self):
            return SimpleNamespace(images=self.images, frame_rate=self.frame_rate)

    manifest = (
        "<Picture 1>: courier identity, face, hair, wardrobe, and distinguishing visual traits.\n"
        "<Picture 2>: rainy rooftop environment, dusk palette, wet materials, and lighting reference.\n"
        "<Video 1>: running motion, lateral camera path, cut rhythm, and landing mechanics reference."
    )
    reference_images = torch.ones((2, 6, 8, 3))
    reference_video = FakeVideo(torch.zeros((3, 2, 4, 3)))
    context, _context_json, _preview = dg.DiffusionGemmaH3ReferenceContext().build(
        "Create an eight-second rooftop chase.",
        manifest,
        sample_fps=3.0,
        max_duration_seconds=10.0,
        max_frames=10,
        overlong_policy="trim",
        reference_images=reference_images,
        reference_video=reference_video,
    )
    assert tuple(context.images.shape) == (5, 2, 4, 3), context.images.shape
    assert context.media_metadata["minimax_h3_reference_image_batch_count"] == 2
    assert context.media_metadata["reference_image_count"] == 2
    assert context.media_metadata["video_sampled_frame_count"] == 3
    messages, _processor_kwargs = dg._transformers_messages_and_processor_kwargs(
        "compile the Ref2VA prompt",
        dg._media_context_from_gemma_context(context),
    )
    content = messages[0]["content"]
    labels = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
    assert "<Picture 1> pixel evidence" in labels, labels
    assert "<Picture 2> pixel evidence" in labels, labels
    assert "<Video 1>" in labels, labels
    assert sum(item.get("type") == "image" for item in content) == 5, content


def test_minimax_h3_visual_medium_fidelity():
    pokemon_brief = (
        "Create a 10-second pokemon-inspired animated short with animal-like creatures having a battle in a town square. "
        "The style should be reimagined into a late-1800's spaghetti-western"
    )
    assert dg._minimax_h3_requested_visual_medium(pokemon_brief) == "2d_animation"
    assert dg._minimax_h3_requested_visual_medium("Pokémon-inspired animated short, no 3D CGI") == "2d_animation"
    assert dg._minimax_h3_requested_visual_medium("Pokémon-inspired stylized 3D computer-animated short") == "3d_animation"
    assert dg._minimax_h3_requested_visual_medium("Not a cartoon, use live action") == "live_action"
    assert (
        dg._minimax_h3_requested_visual_medium(
            "Realistic live-action film photography with no animation, cartoon rendering, or CGI."
        )
        == "live_action"
    )

    model_prompt = dg._build_model_prompt(
        pokemon_brief,
        dg.DEFAULT_MASTER_PROMPT,
        "minimax_h3",
        {},
        target_duration_seconds=10.0,
        audio_mode="auto_scene_audio",
        creativity_mode="faithful",
        creative_strength=0.7,
        thinking_mode="on",
    )
    lowered = model_prompt.lower()
    assert '"minimax_h3_requested_visual_medium"' in lowered and '"2d_animation"' in lowered, model_prompt
    assert "requested rendering medium is immutable 2d animation" in lowered
    assert "2d hand-drawn cel animation" in lowered
    assert "no live action, photorealism, or 3d cgi" in lowered
    assert "minimax_h3_requested_style_locks" in lowered
    assert "pokemon-inspired" in lowered
    assert "late-1800's spaghetti-western" in lowered

    target = dg._make_target_profile_config(
        target_profile="minimax_h3",
        target_duration_seconds=10.0,
        audio_mode="auto_scene_audio",
    )
    context = dg.GemmaContext(user_prompt=pokemon_brief, media_metadata={})
    failed_prompt = (
        "integrated_multimodal_description: [Shot 1] A low-angle, extreme close-up of a reptilian-like creature with weathered "
        "scales and a miniature leather duster, standing in a dusty sun-drenched 1800s town square. The creature narrows its eyes. "
        "[Shot 2] At 00:03.500, the camera cuts to a wide shot of a large, avian-like creature with tattered feathers across the street. "
        "The avian creature sends a shockwave across the ground. [Shot 3] At 00:07.200, the camera cuts to a medium shot as the "
        "reptilian creature lunges forward and strikes the ground with a whip-pan.\n\n"
        "overall_soundscape: Wind, gravel, a piercing screech, and electrical crackle surround the action.\n\n"
        "non_diegetic_music: A whistle melody, Spanish guitar, and orchestral swells build through the battle."
    )
    failed_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    failed_packet["minimax_h3_prompt"] = failed_prompt
    raw_failed_reasons = dg._minimax_h3_prompt_validation_reasons(failed_prompt, 10.0, pokemon_brief)
    assert "minimax_h3_requested_visual_medium_missing" in raw_failed_reasons, raw_failed_reasons
    assert "minimax_h3_requested_visual_exclusion_missing" in raw_failed_reasons, raw_failed_reasons
    assert "minimax_h3_requested_style_missing" in raw_failed_reasons, raw_failed_reasons
    failed_result = _split(failed_packet, context=context, target=target)
    failed_metadata = _metadata(failed_result)
    assert failed_result[13] == "", failed_result[13]
    assert failed_result[12] is False, failed_metadata
    assert "minimax_h3_battle_result_missing" in failed_metadata["blocked_reasons"], failed_metadata
    assert "minimax_h3_requested_visual_medium_missing" not in failed_metadata["blocked_reasons"], failed_metadata
    assert "minimax_h3_requested_visual_exclusion_missing" not in failed_metadata["blocked_reasons"], failed_metadata
    assert failed_metadata["minimax_h3_requested_visual_medium"] == "2d_animation", failed_metadata
    assert failed_metadata["minimax_h3_requested_style_locks"] == [
        "pokemon-inspired",
        "late-1800's spaghetti-western",
    ], failed_metadata
    assert failed_metadata["minimax_h3_style_locks_injected"] is True, failed_metadata
    assert failed_metadata["minimax_h3_visual_medium_lock_injected"] is True, failed_metadata
    assert failed_metadata["claim_verification_passed"] is False, failed_metadata

    corrected_prompt = (
        "integrated_multimodal_description: [Shot 1] 2D hand-drawn cel animation, Pokémon-inspired creature designs reimagined "
        "through a late-1800s spaghetti-western aesthetic, with crisp inked outlines, flat cel shading, and no live action, "
        "photorealism, or 3D CGI. A locked-off wide two-shot establishes Embercrest, a small red-orange lizard creature in a tan "
        "duster, facing Skyspur, a tall cobalt bird creature in a black neckerchief, across the dusty town square. "
        "[Shot 2] At 00:03.500, the camera cuts to a low-angle tracking shot beside Skyspur as it beats its wings and fires a blue "
        "wind arc at Embercrest; the arc whistles, hits Embercrest's raised tail, and throws the same lizard two steps backward. "
        "[Shot 3] At 00:07.200, the camera cuts to an overhead static hold as Embercrest plants its boots, redirects the fading arc into "
        "the dirt, and Skyspur lowers its wings across the square. The final frame holds on both unchanged creature designs facing "
        "each other through the settling dust in the overhead composition.\n\n"
        "overall_soundscape: Dry wind crosses the square, leather creaks, boots scrape gravel, and dust settles after the impact.\n\n"
        "non_diegetic_music: A steady Spanish-guitar rhythm and lone whistle pause on the wind-arc impact, then resolve on a low chord."
    )
    corrected_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    corrected_packet["minimax_h3_prompt"] = corrected_prompt
    corrected_result = _split(corrected_packet, context=context, target=target)
    assert corrected_result[13] == corrected_prompt, _metadata(corrected_result)
    assert corrected_result[12] is True, _metadata(corrected_result)

    missing_style_prompt = corrected_prompt.replace(
        "Pokémon-inspired creature designs reimagined through a late-1800s spaghetti-western aesthetic, ",
        "",
    )
    assert "minimax_h3_requested_style_missing" in dg._minimax_h3_prompt_validation_reasons(
        missing_style_prompt,
        10.0,
        pokemon_brief,
    )
    repaired_style_prompt = dg._apply_minimax_h3_style_locks(missing_style_prompt, pokemon_brief)
    assert "pokemon-inspired" in repaired_style_prompt.lower(), repaired_style_prompt
    assert "late-1800's spaghetti-western" in repaired_style_prompt.lower(), repaired_style_prompt
    assert "minimax_h3_requested_style_missing" not in dg._minimax_h3_prompt_validation_reasons(
        repaired_style_prompt,
        10.0,
        pokemon_brief,
    )
    missing_style_packet = _packet("", {"ready_for_generation": True, "blocked_reasons": []}, segments=[])
    missing_style_packet["minimax_h3_prompt"] = missing_style_prompt
    repaired_style_result = _split(missing_style_packet, context=context, target=target)
    assert repaired_style_result[13] == repaired_style_prompt, _metadata(repaired_style_result)
    assert repaired_style_result[12] is True, _metadata(repaired_style_result)
    assert _metadata(repaired_style_result)["minimax_h3_style_locks_injected"] is True

    unresolved_battle_prompt = (
        "integrated_multimodal_description: [Shot 1] 2D hand-drawn cel animation, pokemon-inspired and late-1800's "
        "spaghetti-western visual styling, no live action, photorealism, or 3D CGI. A slow zoom-in frames a fox-like creature "
        "opposite a reptilian creature in the town square. [Shot 2] At 00:03.500, the camera cuts to an extreme close-up of the "
        "reptilian creature lunging forward with glowing claws. [Shot 3] At 00:07.000, the camera cuts to a wide profile-view shot "
        "as the fox creature leaps and summons a golden shockwave. The final frame holds on the fox creature mid-air.\n\n"
        "overall_soundscape: Wind, gravel, growls, and energy crackle surround the action.\n\n"
        "non_diegetic_music: A whistle and rhythmic acoustic guitar build through the final shot."
    )
    unresolved_reasons = dg._minimax_h3_prompt_validation_reasons(
        unresolved_battle_prompt,
        10.0,
        pokemon_brief,
    )
    assert "minimax_h3_battle_result_missing" in unresolved_reasons, unresolved_reasons
    assert "minimax_h3_unresolved_final_state" in unresolved_reasons, unresolved_reasons
    refinement_prompt = dg._build_minimax_h3_refinement_prompt(
        pokemon_brief,
        unresolved_battle_prompt,
        10.0,
        "auto_scene_audio",
        unresolved_reasons,
    ).lower()
    assert "every attack must name attacker, target" in refinement_prompt
    assert "do not end mid-air" in refinement_prompt
    assert "the final frame holds on" in refinement_prompt

    refined_artifact_prompt = (
        "integrated_multimodal_description: [Shot 1] 2D hand-drawn cel animation, [pokemon-inspired\", \"late-1800's "
        "spaghetti-western], no live action, photorealism, or 3D CGI. A slow low-angle zoom frames a fox-creature on the left "
        "and a lizard-creature on the right. [Shot 2] At 00:03.500, the camera cuts to an extreme close-up as the lizard-creature "
        "grips the dirt. [Shot 3] At 00:06.000, the camera cuts to a wide tracking shot as the fox-creature strikes the "
        "lizard-creature; the lizard-creature is knocked backward into a hitching post. The final frame holds on the fox-creature "
        "standing in the center and the dazed lizard-creature against the post on the right.\n\n"
        "overall_soundscape: Wind, gravel, a heavy impact thud, and the crackle of a fire blast.\n\n"
        "non_diegetic_music: Harmonica and acoustic guitar resolve on the final impact."
    )
    assert "minimax_h3_orphaned_sound_event" in dg._minimax_h3_prompt_validation_reasons(
        refined_artifact_prompt,
        10.0,
        pokemon_brief,
    )
    canonical_style_prompt = dg._apply_minimax_h3_style_locks(refined_artifact_prompt, pokemon_brief)
    assert "faithfully using pokemon-inspired and late-1800's spaghetti-western visual styling" in canonical_style_prompt
    assert '[pokemon-inspired"' not in canonical_style_prompt
    sound_repaired_prompt = dg._apply_minimax_h3_sound_fidelity(canonical_style_prompt, pokemon_brief)
    assert "fire blast" not in sound_repaired_prompt.lower(), sound_repaired_prompt
    assert dg._minimax_h3_prompt_validation_reasons(sound_repaired_prompt, 10.0, pokemon_brief) == []


def test_minimax_h3_explicit_storyboard_fidelity():
    brief = (
        "Realistic live-action action trailer in four hard-cut shots. Shot 1 establishes the runner. Shot 2 is the leap. "
        "Shot 3 is the landing. Shot 4 is a final freeze-frame holding on the next launch."
    )
    five_shot_prompt = (
        "integrated_multimodal_description: [Shot 1] Realistic live-action practical film photography. A high-angle tracking shot "
        "follows the runner. [Shot 2] At 00:02.000, the camera cuts to a wide profile as he leaps. "
        "[Shot 3] At 00:04.000, the camera cuts to a low-angle handheld view as he lands. "
        "[Shot 4] At 00:06.000, the camera cuts to an overhead static hold as he runs. "
        "[Shot 5] At 00:08.000, the camera cuts to a long tracking shot as he reaches the next edge.\n\n"
        "overall_soundscape: Wind and footsteps cross the roofs.\n\n"
        "non_diegetic_music: Low percussion accelerates beneath the hard cuts."
    )
    reasons = dg._minimax_h3_prompt_validation_reasons(five_shot_prompt, 10.0, brief)
    assert "minimax_h3_requested_shot_count_mismatch" in reasons, reasons
    assert "minimax_h3_requested_final_hold_missing" in reasons, reasons

    no_camera_prompt = five_shot_prompt.replace("A high-angle tracking shot follows the runner.", "The runner sprints toward the edge.")
    assert "minimax_h3_shot_camera_unspecified" in dg._minimax_h3_prompt_validation_reasons(
        no_camera_prompt,
        10.0,
        brief,
    )


def test_ltx_camera_choreography_contract():
    prompt = dg._build_model_prompt(
        "Recreate the source video as an LTX prompt.",
        dg.DEFAULT_MASTER_PROMPT,
        "ltx",
        {
            "source": "video",
            "duration_seconds": 10,
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": True,
            "visual_grounding_mode": "pixels",
        },
        audio_mode="visual_only",
        creativity_mode="faithful",
        creative_strength=0.2,
        thinking_mode="on",
    )
    lowered = prompt.lower()
    for phrase in (
        "camera choreography",
        "camera-choreography extraction",
        "first-frame framing",
        "camera-subject relationship",
        "zoom/dolly/truck/boom/pan/tilt/roll/orbit/rotation",
        "parallax",
        "180 degrees",
        "do not invent one",
    ):
        assert phrase in lowered, phrase


def test_ltx_speech_boundary_contract():
    audio_guidance = (
        'Deep off-screen narrator, exact words only: "The truth never stays buried forever." No other speech.'
    )
    prompt = dg._build_model_prompt(
        'A deep off-screen voice whispers, "The truth never stays buried forever." The camera widens after the line.',
        dg.DEFAULT_MASTER_PROMPT,
        "ltx",
        {
            "source": "text",
            "duration_seconds": 15,
            "visual_grounding_mode": "none",
        },
        audio_mode="explicit_sound_design",
        audio_guidance=audio_guidance,
        creativity_mode="cinematic",
        creative_strength=0.2,
        thinking_mode="off",
    )
    lowered = prompt.lower()
    for phrase in (
        "identify the speaker or off-screen narrator and delivery",
        "preserve any user-supplied wording verbatim",
        "only the exact intelligible words intended to be heard",
        "balanced straight double quotation marks",
        "no intervening whitespace",
        "visible subjects do not speak",
        "describe every requested closing visual beat before the final voiceover attribution",
        "exact quoted words the final non-whitespace characters of the entire prompt",
        "never append a speech-stop sentence",
        "any other prose after that final closing quotation mark",
    ):
        assert phrase in lowered, phrase
    assert f"Audio guidance: {dg._audio_guidance_text(audio_guidance)}" in prompt

    correctly_closed = 'A narrator whispers, "The truth never stays buried forever." The camera widens.'
    assert dg._sanitize_prompt_text(correctly_closed) == correctly_closed
    single_quoted = "A narrator whispers, 'The truth never stays buried forever.' The camera widens."
    assert dg._sanitize_prompt_text(single_quoted) == single_quoted
    curly_quoted = "A narrator whispers, “The truth never stays buried forever.” The camera widens."
    assert dg._sanitize_prompt_text(curly_quoted) == curly_quoted
    missing_opening_space = 'A narrator says:"Stay back."'
    assert dg._sanitize_prompt_text(missing_opening_space) == 'A narrator says: "Stay back."'

    unsafe = (
        'A medium shot at eye level opens on a dark corridor, with the camera holding static. '
        'A score swells while a deep, ominous off-screen voiceover intones, '
        '"The truth never stays buried forever. " The shot ends on a medium shot at eye level, '
        "the camera holding static against the darkness as the light suddenly cuts to black."
    )
    repaired, report = dg._repair_ltx_exact_voiceover_boundary(
        unsafe,
        audio_guidance,
    )
    terminal_quote = '"The truth never stays buried forever."'
    assert report["applied"] is True, report
    assert report["status"] == "repaired", report
    assert repaired.endswith(terminal_quote), repaired
    assert repaired.count(terminal_quote) == 1, repaired
    assert "The shot ends on a medium shot" in repaired, repaired
    assert repaired.index("The shot ends on a medium shot") < repaired.index(terminal_quote), repaired
    assert dg._ltx_exact_voiceover_validation_reasons(
        unsafe,
        "explicit_sound_design",
        audio_guidance,
    ) == ["ltx_exact_voiceover_not_terminal"]
    assert not dg._ltx_exact_voiceover_validation_reasons(
        repaired,
        "explicit_sound_design",
        audio_guidance,
    )
    already_safe, safe_report = dg._repair_ltx_exact_voiceover_boundary(
        repaired,
        audio_guidance,
    )
    assert already_safe == repaired
    assert safe_report["status"] == "already_terminal", safe_report

    target = dg.TargetProfileConfig(
        target_profile="ltx",
        audio_mode="explicit_sound_design",
        audio_guidance=audio_guidance,
        target_duration_seconds=15.0,
    )
    split_result = _split(
        _packet(
            unsafe,
            segments=[
                {
                    "index": 0,
                    "duration_seconds": 15.0,
                    "prompt": unsafe,
                }
            ],
        ),
        context=dg.GemmaContext(
            user_prompt="A cinematic thriller teaser with one exact closing voiceover.",
            source="text",
            media_metadata={"source": "text", "duration_seconds": 15.0},
        ),
        target=target,
    )
    assert split_result[0] == repaired, split_result[0]
    assert split_result[8] is True, _metadata(split_result)
    assert split_result[12] is True, _metadata(split_result)
    split_metadata = _metadata(split_result)
    assert split_metadata["ltx_exact_voiceover_boundary"]["applied"] is True, split_metadata
    rebuilt_segments = json.loads(split_result[5])
    assert rebuilt_segments, rebuilt_segments
    assert repaired != unsafe, repaired
    assert unsafe not in json.dumps(rebuilt_segments), rebuilt_segments
    assert rebuilt_segments[-1]["prompt"].endswith(terminal_quote), rebuilt_segments

    # The main request is also an authoritative speech source when Audio
    # guidance is blank, which is common in simpler LTX workflows.
    user_request = (
        "A deep, ominous off-screen voiceover whispers, "
        "'The truth never stays buried forever.' The shot cuts to black."
    )
    fallback_repaired, fallback_report = dg._repair_ltx_exact_voiceover_boundary(
        unsafe,
        "",
        request_text=user_request,
    )
    assert fallback_report["applied"] is True, fallback_report
    assert fallback_report["source"] == "context.user_prompt", fallback_report
    assert fallback_repaired.endswith(terminal_quote), fallback_repaired

    visual_only = dg.TargetProfileConfig(
        target_profile="ltx",
        audio_mode="visual_only",
        audio_guidance=audio_guidance,
    )
    visual_only_result = _split(
        _packet(unsafe),
        context=dg.GemmaContext(user_prompt=user_request, source="text"),
        target=visual_only,
    )
    _assert_blocked(visual_only_result, "ltx_visual_only_contains_speech")

    extra_dialogue = (
        'The woman says, "Run now." A deep off-screen narrator intones, '
        '"The truth never stays buried forever." The camera cuts to black.'
    )
    extra_result = _split(
        _packet(extra_dialogue),
        context=dg.GemmaContext(user_prompt=user_request, source="text"),
        target=target,
    )
    _assert_blocked(extra_result, "ltx_exact_voiceover_additional_or_ambiguous_speech")

    opening_timing = (
        'At the beginning, a deep off-screen narrator intones, '
        '"The truth never stays buried forever." The camera later cuts to black.'
    )
    opening_repaired, opening_report = dg._repair_ltx_exact_voiceover_boundary(
        opening_timing,
        audio_guidance,
    )
    assert opening_repaired == opening_timing, opening_repaired
    assert opening_report["status"] == "explicit_nonterminal_timing", opening_report
    opening_result = _split(
        _packet(opening_timing),
        context=dg.GemmaContext(user_prompt=user_request, source="text"),
        target=target,
    )
    _assert_blocked(opening_result, "ltx_exact_voiceover_not_terminal")

    visual_quote = (
        'A sign reads "The truth never stays buried forever." A deep off-screen narrator intones, '
        '"The truth never stays buried forever." The camera cuts to black.'
    )
    visual_quote_repaired, visual_quote_report = dg._repair_ltx_exact_voiceover_boundary(
        visual_quote,
        audio_guidance,
    )
    assert visual_quote_report["applied"] is True, visual_quote_report
    assert visual_quote_repaired.count(terminal_quote) == 2, visual_quote_repaired
    assert visual_quote_repaired.endswith(terminal_quote), visual_quote_repaired

    contraction_guidance = "One off-screen narrator line, exact words only: 'Don't look back.'"
    contraction_prompt = "An off-screen narrator whispers, 'Don't look back.' The door slams shut."
    contraction_repaired, contraction_report = dg._repair_ltx_exact_voiceover_boundary(
        contraction_prompt,
        contraction_guidance,
    )
    assert contraction_report["applied"] is True, contraction_report
    assert contraction_repaired.endswith('"Don\'t look back."'), contraction_repaired

    upstream_packet = _packet(repaired, metadata={"ltx_exact_voiceover_boundary": report})
    upstream_result = _split(
        upstream_packet,
        context=dg.GemmaContext(user_prompt=user_request, source="text"),
        target=target,
    )
    upstream_metadata = _metadata(upstream_result)["ltx_exact_voiceover_boundary"]
    assert upstream_metadata["applied"] is True, upstream_metadata
    assert upstream_metadata["verified_at_splitter"] is True, upstream_metadata
    assert dg._sanitize_prompt_text("Hello.World") == "Hello. World"


def test_image_identity_video_control_contract():
    media = {
        "source": "image+video",
        "duration_seconds": 3,
        "pixel_tensor_present": True,
        "pixels_sent_to_backend": True,
        "visual_grounding_mode": "pixels",
        "media_synthesis_mode": "image_identity_video_control",
        "image_role": "identity_reference",
        "video_role": "control_structure_pose_depth_canny_composition_motion_camera",
        "reference_image_count": 1,
        "video_sampled_frame_count": 3,
        "sampled_frame_count": 4,
    }
    prompt = dg._build_model_prompt(
        "Replace the woman in the video with the girl in the image.",
        dg.DEFAULT_MASTER_PROMPT,
        "ltx",
        media,
        audio_mode="visual_only",
        creativity_mode="faithful",
        creative_strength=0.2,
        thinking_mode="on",
    )
    lowered = prompt.lower()
    for phrase in (
        "media_synthesis_mode",
        "image_identity_video_control",
        "reference-image subject performing the video action/control structure",
        "image controls identity and appearance",
        "wardrobe, accessories, styling",
        "video controls pose, depth, canny/edge layout",
        "do not copy the control-video subject's clothing",
        "replace the video subject appearance with the reference-image appearance",
        "do not let the person in the control video override",
    ):
        assert phrase in lowered, phrase

    intro = dg._sampled_video_frame_sequence_intro(
        {
            **media,
            "sampled_indices": [0, 12, 24],
            "source_fps": 24,
        },
        4,
    )
    intro_lower = intro.lower()
    assert "optional still reference" not in intro_lower, intro
    assert "first attached image is the identity reference" in intro_lower, intro
    assert "remaining attached images are ordered sampled video control frames" in intro_lower, intro
    assert "video frame times: 1=0.00s, 2=0.50s, 3=1.00s" in intro_lower, intro
    assert "wardrobe, accessories, styling" in intro_lower, intro
    assert "do not copy the control-video subject's clothing" in intro_lower, intro

    synthesis_context = _media_context(
        "Replace the woman in the video with the girl in the image.",
        media,
    )
    ready = _split(
        _packet("The girl from the reference image follows the video pose, room blocking, and camera motion."),
        synthesis_context,
    )
    _assert_ready(ready)
    ready_metadata = _metadata(ready)
    assert ready_metadata["media_synthesis_mode"] == "image_identity_video_control", ready_metadata
    assert ready_metadata["image_role"] == "identity_reference", ready_metadata
    assert ready_metadata["video_sampled_frame_count"] == 3, ready_metadata

    missing_image = _media_context(
        "Replace the woman in the video with the girl in the image.",
        {
            "source": "video",
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": True,
            "media_synthesis_mode": "image_identity_video_control",
            "reference_image_count": 0,
            "video_sampled_frame_count": 3,
            "sampled_frame_count": 3,
        },
    )
    _assert_blocked(
        _split(_packet("The reference-image subject follows the control-video pose and camera motion."), missing_image),
        "synthesis_missing_identity_image",
    )

    unattached_image = _media_context(
        "Replace the woman in the video with the girl in the image.",
        {
            "source": "video",
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": True,
            "media_synthesis_mode": "image_identity_video_control",
            "reference_image_count": 1,
            "reference_image_backend_attached": False,
            "video_sampled_frame_count": 3,
            "sampled_frame_count": 3,
        },
    )
    _assert_blocked(
        _split(_packet("The reference-image subject follows the control-video pose and camera motion."), unattached_image),
        "synthesis_identity_image_not_attached",
    )

    missing_video = _media_context(
        "Replace the woman in the video with the girl in the image.",
        {
            "source": "image",
            "pixel_tensor_present": True,
            "pixels_sent_to_backend": True,
            "media_synthesis_mode": "image_identity_video_control",
            "reference_image_count": 1,
            "video_sampled_frame_count": 0,
            "sampled_frame_count": 1,
        },
    )
    _assert_blocked(
        _split(_packet("The reference-image subject follows the control-video pose and camera motion."), missing_video),
        "synthesis_missing_control_video",
    )


def test_media_sampler_forces_identity_image_in_synthesis_mode():
    import torch

    class FakeVideo:
        def __init__(self, images, frame_rate=3.0):
            self.images = images
            self.frame_rate = frame_rate

        def get_duration(self):
            return 1.0

        def get_frame_count(self):
            return int(self.images.shape[0])

        def get_dimensions(self):
            return (int(self.images.shape[2]), int(self.images.shape[1]))

        def get_components(self):
            return SimpleNamespace(images=self.images, frame_rate=self.frame_rate)

    image = torch.ones((1, 2, 2, 3))
    media_context, metadata_json = dg.DiffusionGemmaMediaSampler().sample(
        3.0,
        10.0,
        10,
        "trim",
        False,
        image=image,
        video=FakeVideo(torch.zeros((3, 2, 2, 3))),
        media_synthesis_mode="image_identity_video_control",
    )
    metadata = json.loads(metadata_json)
    assert media_context.source == "image+video", metadata
    assert metadata["media_synthesis_mode"] == "image_identity_video_control", metadata
    assert metadata["reference_image_count"] == 1, metadata
    assert metadata["reference_image_backend_attached"] is True, metadata
    assert metadata["reference_image_resized_for_video_control"] is False, metadata
    assert metadata["video_sampled_frame_count"] == 3, metadata
    assert metadata["sampled_frame_count"] == 4, metadata
    assert tuple(media_context.images.shape) == (4, 2, 2, 3)

    mismatched_image = torch.ones((1, 6, 8, 3))
    mismatched_context, mismatched_metadata_json = dg.DiffusionGemmaMediaSampler().sample(
        3.0,
        10.0,
        10,
        "trim",
        False,
        image=mismatched_image,
        video=FakeVideo(torch.zeros((3, 2, 4, 3))),
        media_synthesis_mode="image_identity_video_control",
    )
    mismatched_metadata = json.loads(mismatched_metadata_json)
    assert mismatched_context.source == "image+video", mismatched_metadata
    assert mismatched_metadata["reference_image_backend_attached"] is True, mismatched_metadata
    assert mismatched_metadata["reference_image_resized_for_video_control"] is True, mismatched_metadata
    assert mismatched_metadata["reference_image_original_width"] == 8, mismatched_metadata
    assert mismatched_metadata["reference_image_original_height"] == 6, mismatched_metadata
    assert mismatched_metadata["reference_image_backend_width"] == 4, mismatched_metadata
    assert mismatched_metadata["reference_image_backend_height"] == 2, mismatched_metadata
    assert mismatched_metadata["sampled_frame_count"] == 4, mismatched_metadata
    assert tuple(mismatched_context.images.shape) == (4, 2, 4, 3)
    assert bool(torch.all(mismatched_context.images[0] > 0.99)), mismatched_context.images[0]


def test_transformers_media_message_labels_identity_and_control_frames():
    import torch

    media_context = dg.MediaContext(
        images=torch.zeros((4, 2, 2, 3)),
        source="image+video",
        metadata={
            "source": "image+video",
            "media_synthesis_mode": "image_identity_video_control",
            "reference_image_count": 1,
            "video_sampled_frame_count": 3,
            "sampled_frame_count": 4,
            "sampled_indices": [0, 12, 24],
            "source_fps": 24,
            "transformers_video_transport": "sampled_frame_images",
        },
    )
    messages, processor_kwargs = dg._transformers_messages_and_processor_kwargs("Final prompt contract.", media_context)
    assert processor_kwargs == {}, processor_kwargs
    content = messages[0]["content"]
    texts = "\n".join(str(item["text"]) for item in content if item.get("type") == "text").lower()
    assert "identity reference image" in texts, texts
    assert "video control frames" in texts, texts
    assert "do not copy the control-video subject's clothing" in texts, texts
    assert "reference wardrobe and accessories" in texts, texts
    assert "not part of the video timeline" in texts, texts
    assert sum(1 for item in content if item.get("type") == "image") == 4, content


def _minimal_ltx_workflow() -> dict:
    def subgraph(subgraph_id: str, name: str, nodes: list[dict]) -> dict:
        return {"id": subgraph_id, "name": name, "nodes": nodes, "links": []}

    return {
        "id": "smoke",
        "version": 0.4,
        "nodes": [
            {
                "id": 5496,
                "type": "VHS_LoadVideo",
                "title": "VHS_LoadVideo",
                "widgets_values": {"video": "old_video.mp4", "force_rate": 24},
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                subgraph(
                    "fe45e814-d005-4dd9-97dc-e31c4c6e9773",
                    "I2V Image",
                    [
                        {
                            "id": 5038,
                            "type": "LoadImage",
                            "title": "Reference Object",
                            "widgets_values": ["old_reference.png", "image"],
                        }
                    ],
                ),
                subgraph(
                    "ba181d97-b80f-49a8-ae30-e552bd7c8f49",
                    "Mask",
                    [
                        {
                            "id": 5433,
                            "type": "SAM3Segment",
                            "title": "SAM3 Segmentation (RMBG) - 1 Pass",
                            "widgets_values": ["old target", "Merged", 0.5],
                        }
                    ],
                ),
                subgraph(
                    "f935d44a-3a0d-41cf-b9bc-b2261f8ba60f",
                    "New Subgraph",
                    [
                        {
                            "id": 5676,
                            "type": "SAM3Segment",
                            "title": "SAM3 Segmentation (RMBG) - 2 Pass",
                            "widgets_values": ["", "Merged", 0.5],
                        }
                    ],
                ),
                subgraph(
                    "a0619931-10e4-48af-9d29-114a59e7c821",
                    "New Subgraph",
                    [
                        {
                            "id": 5745,
                            "type": "SAM3Segment",
                            "title": "SAM3 Segmentation (RMBG) - 2 Pass",
                            "widgets_values": ["", "Merged", 0.5],
                        }
                    ],
                ),
                subgraph(
                    "483656b1-20e1-4b50-a7e2-3b6126ad4bb7",
                    "Prompt",
                    [
                        {
                            "id": 5630,
                            "type": "CLIPTextEncode",
                            "title": "Manual Prompt",
                            "widgets_values": ["old positive"],
                        },
                        {
                            "id": 5626,
                            "type": "CLIPTextEncode",
                            "title": "CLIPTextEncode",
                            "widgets_values": [""],
                        },
                    ],
                ),
            ]
        },
    }


def test_dg_to_ltx_prompt_injector():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        workflow_path = tmp / "workflow.json"
        dg_path = tmp / "dg_prompt.json"
        output_path = tmp / "patched.json"
        manifest_path = tmp / "manifest.json"
        workflow_path.write_text(json.dumps(_minimal_ltx_workflow()), encoding="utf-8")
        dg_path.write_text(
            json.dumps(
                {
                    "sam_target": "scratched panel",
                    "positive_prompt": "Repair only the small scratch on the panel.",
                    "negative_prompt": "full scene reroll, new object, identity change",
                    "preservation_constraints": ["preserve camera", "preserve lighting"],
                    "do_not_change": ["background", "panel shape"],
                    "repair_scope": "localized_masked_inpaint",
                }
            ),
            encoding="utf-8",
        )

        manifest = prompt_injector.build_patched_workflow(
            workflow_path,
            dg_path,
            output_path,
            manifest_path,
            video_path="input/source.mp4",
            reference_image_path="input/reference.png",
        )
        patched = json.loads(output_path.read_text(encoding="utf-8"))
        assert manifest["only_prompt_path_widgets_changed"] is True
        assert manifest["ltx_graph_topology_unchanged"] is True
        assert manifest["sam3_remains_in_workflow"] is True
        assert manifest["sam3_node_count"] == 3
        assert manifest["changed_widget_count"] == 7
        assert manifest["generation_run"] is False
        assert manifest["semantic_repair_success_claimed"] is False
        assert not manifest["forbidden_changes"]
        changed_ids = {item["field_id"] for item in manifest["changes"]}
        assert changed_ids == {
            "video_input",
            "reference_image_input",
            "sam3_target_primary",
            "sam3_target_pass2_a",
            "sam3_target_pass2_b",
            "ltx_positive_prompt",
            "ltx_negative_prompt",
        }
        assert patched["nodes"][0]["widgets_values"]["video"] == "input/source.mp4"
        prompt_sg = next(item for item in patched["definitions"]["subgraphs"] if item["id"] == "483656b1-20e1-4b50-a7e2-3b6126ad4bb7")
        positive = next(item for item in prompt_sg["nodes"] if item["id"] == 5630)["widgets_values"][0]
        assert "Repair only the small scratch" in positive
        assert "Preservation constraints" in positive
        assert "Do not change" in positive
        assert next(item for item in prompt_sg["nodes"] if item["id"] == 5626)["widgets_values"][0] == "full scene reroll, new object, identity change"
        inventory = prompt_injector.inspect_workflow(output_path)["field_inventory"]
        assert len(inventory) == 7


def test_minimax_h3_ref2va_example_workflow_serialization():
    workflow_path = ROOT / "examples" / "10_minimax_h3_ref2va_director.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    links = {link[0]: link for link in workflow["links"]}

    assert len(workflow["nodes"]) == 31
    assert len(links) == 43
    guard_node = next(node for node in workflow["nodes"] if node["id"] == 152)
    assert guard_node["type"] == "DiffusionGemmaGroundingGuardSettings"
    cot_node = next(node for node in workflow["nodes"] if node["id"] == 145)
    assert cot_node["inputs"][3]["name"] == "grounding_guard_config"
    assert cot_node["inputs"][3]["link"] == 43
    assert links[43][1:5] == [152, 0, 145, 3]
    gate_node = next(node for node in workflow["nodes"] if node["id"] == 151)
    assert gate_node["type"] == "DiffusionGemmaGenerationGate"
    assert [input_slot["link"] for input_slot in gate_node["inputs"]] == [12, 40, 41]
    assert gate_node["outputs"][0]["links"] == [42]
    assert links[12][1:5] == [146, 13, 151, 0]
    assert links[40][1:5] == [146, 12, 151, 1]
    assert links[41][1:5] == [146, 4, 151, 2]
    assert links[42][1:5] == [151, 0, 136, 9]
    for node in workflow["nodes"]:
        for output_index, output in enumerate(node.get("outputs", [])):
            output_links = output.get("links")
            assert output_links is None or isinstance(output_links, list), (node["id"], output["name"], output_links)
            for link_id in output_links or []:
                link = links[link_id]
                assert link[1:3] == [node["id"], output_index], (node["id"], output["name"], link)
        for input_index, input_slot in enumerate(node.get("inputs", [])):
            link_id = input_slot.get("link")
            if link_id is None:
                continue
            assert isinstance(link_id, int), (node["id"], input_slot["name"], link_id)
            link = links[link_id]
            assert link[3:5] == [node["id"], input_index], (node["id"], input_slot["name"], link)


def main():
    test_readiness_gates()
    test_prompt_packet_schema()
    test_minimax_h3_contract_and_splitter_compatibility()
    test_minimax_h3_ref2va_contract()
    test_minimax_h3_ref2va_structure_repair_and_generation_gate()
    test_minimax_h3_reference_context_preserves_picture_batch_with_video()
    test_minimax_h3_visual_medium_fidelity()
    test_minimax_h3_explicit_storyboard_fidelity()
    test_ltx_camera_choreography_contract()
    test_ltx_speech_boundary_contract()
    test_image_identity_video_control_contract()
    test_media_sampler_forces_identity_image_in_synthesis_mode()
    test_transformers_media_message_labels_identity_and_control_frames()
    test_dg_to_ltx_prompt_injector()
    test_minimax_h3_ref2va_example_workflow_serialization()
    print("Prompt Builder readiness/schema smoke passed")


if __name__ == "__main__":
    main()
