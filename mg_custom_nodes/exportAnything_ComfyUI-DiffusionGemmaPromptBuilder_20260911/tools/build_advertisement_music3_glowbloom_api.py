#!/usr/bin/env python3
"""Build the isolated GlowBloom Advertisement + MiniMax Music 3 API graph.

This is a validation fixture builder, not a migration of the user's known-good
V6 workflow.  It starts from the exact runtime-proven GlowBloom API capture,
keeps the proven Director/H3 model path, replaces the temporary uploaded-song
branch with native MiniMax Music 3, and routes the result through the additive
Advertisement contracts, four-reference relay plan, deterministic finishing,
real cutdowns, and honest QA status.

The source capture and the checked-in V6 UI workflow are hash guarded and are
never written by this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
KNOWN_GOOD_UI = REPOSITORY / "examples" / "15_minimax_h3_ref2va_music_video_v6.json"
KNOWN_GOOD_UI_SHA256 = "594f6aef94531f87b74659e1e8004cccdff00716d0cfe263cee6bf2149545c83"
SOURCE_API = (
    REPOSITORY
    / "examples"
    / "api"
    / "16_minimax_h3_ref2va_advertisement_glowbloom_v1_api.json"
)
SOURCE_API_CANONICAL_SHA256 = "6296d86a7640ce1babe307f68c3ddab845341e5e1185cb2216924097e9b4f743"
OUTPUT_API = (
    REPOSITORY
    / "examples"
    / "api"
    / "17_minimax_h3_ref2va_advertisement_music3_glowbloom_v1_api.json"
)
OUTPUT_MANIFEST = OUTPUT_API.with_suffix(".manifest.json")

SCHEMA = "diffusiongemma.advertisement_music3_runtime_fixture"
VERSION = 1

ASSET_PATHS = {
    "performer_hero": "diffusiongemma_campaigns/glowbloom_20260823/glowbloom_hero_identity.png",
    "performer_sheet": "diffusiongemma_campaigns/glowbloom_20260823/glowbloom_performer_contact_sheet_v1.png",
    "product_sheet": "diffusiongemma_campaigns/glowbloom_20260823/glowbloom_product_contact_sheet_v1.png",
}
ASSET_SHA256 = {
    "performer_hero": "98ced6b029fe36888b4ada491f5cbd7cb9efd4724001da064f3713b3da4798e4",
    "performer_sheet": "cf13e146624290c1932727fb1a54e38bbe275a866b60ed649d1ca0ee4fd1939b",
    "product_sheet": "ed534b55bc417c1610051f7d91fa807360bf67757e7aa41a96bcc655d4e5db8e",
}
DIRECTOR_MODEL_PATH = "models/LLM/diffusiongemma-26B-A4B-it-NVFP4"
MUSIC3_MODELS = {
    "diffusion_model": "minimax_music3_dit_fp16.safetensors",
    "text_encoder": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    "vae": "minimax_music3_dav.safetensors",
}
MODEL_INVENTORY = {
    "director": {
        "checkpoint_directory": DIRECTOR_MODEL_PATH,
    },
    "minimax_h3": {
        "diffusion_model": "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "text_encoder": "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": "models/vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "models/vae/minimax_h3_audio_vae_fp32.safetensors",
        "turbo_lora": "models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    },
    "minimax_music3": {
        "diffusion_model": "models/diffusion_models/minimax_music3_dit_fp16.safetensors",
        "text_encoder": "models/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "vae": "models/vae/minimax_music3_dav.safetensors",
    },
}


class FixtureError(RuntimeError):
    """Raised when an immutable fixture or graph invariant is violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _load_guarded_json(path: Path, *, byte_hash: str = "", canonical_hash: str = "") -> Any:
    data = path.read_bytes()
    if byte_hash and _sha256_bytes(data) != byte_hash:
        raise FixtureError(f"Immutable source changed: {path}")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"Invalid UTF-8 JSON source: {path}") from exc
    if canonical_hash and _canonical_sha256(parsed) != canonical_hash:
        raise FixtureError(f"Immutable canonical graph changed: {path}")
    return parsed


def _node(class_type: str, title: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _copy_node(source: dict[str, Any], node_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(source[node_id])
    except KeyError as exc:
        raise FixtureError(f"Runtime source is missing node {node_id}.") from exc


def _required_source_nodes(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    top_level = (
        "179", "182", "183", "185", "186", "187", "188", "418",
        "651", "653", "654", "661", "662", "673", "674", "675",
        "676", "678", "681", "683", "691", "699", "713", "716",
    )
    removed_h3 = {
        "710:665", "710:682", "710:689", "710:690", "710:697", "710:698",
        "710:725", "710:726", "710:727", "710:728", "710:729", "710:730",
    }
    selected: dict[str, dict[str, Any]] = {
        node_id: _copy_node(source, node_id) for node_id in top_level
    }
    for node_id in source:
        if node_id.startswith("710:") and node_id not in removed_h3:
            selected[node_id] = _copy_node(source, node_id)
    return selected


def _patch_director_and_h3(graph: dict[str, dict[str, Any]]) -> None:
    graph["182"]["inputs"]["model_path"] = DIRECTOR_MODEL_PATH
    graph["179"]["inputs"] = {"image": ASSET_PATHS["performer_hero"]}
    graph["179"]["_meta"]["title"] = "Picture 1 — performer hero identity"
    # Keep the runtime-proven off baseline explicit to the operator.  The
    # Advertisement reference contract, packet repair governance checks, JSON
    # splitter, and branch gate remain independently fail-closed downstream.
    graph["185"]["_meta"]["title"] = (
        "5b. H3 Grounding Guard — OFF baseline (contract validation stays active)"
    )
    graph["721"] = _node(
        "LoadImage",
        "Picture 2 — same-performer contact sheet",
        image=ASSET_PATHS["performer_sheet"],
    )
    graph["800"] = _node(
        "LoadImage",
        "Picture 3 — independent product/package contact sheet",
        image=ASSET_PATHS["product_sheet"],
    )

    graph["183"]["inputs"].update(
        {
            "user_prompt": ["801", 1],
            "image": ["803", 6],
            "target_duration_seconds": ["849", 0],
            "max_duration_seconds": ["849", 0],
        }
    )
    graph["186"]["inputs"].update(
        {
            "gemma_context": ["674", 0],
            "target_profile_config": ["673", 0],
            "measured_audio_report_json": ["819", 0],
            "temperature": 0.35,
            "creativity_mode": "cinematic",
            "creative_strength": 0.7,
            "thinking_mode": "off",
            "cache_mode": "reuse",
        }
    )
    graph["820"] = _node(
        "DiffusionGemmaAdvertisementDirectorPacketRepair",
        "6b. Recover transport-only JSON inside host-governed Director packet",
        director_final_json=["186", 0],
        director_raw_response=["186", 2],
        director_metadata_json=["186", 3],
        grounding_status=["186", 4],
        grounding_report_json=["186", 5],
        reference_contract_json=["802", 0],
    )
    graph["187"]["inputs"].update(
        {
            "final_json": ["820", 0],
            "gemma_context": ["674", 0],
            "target_profile_config": ["673", 0],
            "resolution_selector_megapixels": 0.4,
            "resolution_selector_multiple": 32,
            "resolution_aspect_ratio": "9:16 (Portrait Widescreen)",
            "resolution_aspect_ratio_override": ["849", 15],
        }
    )
    graph["188"]["inputs"] = {
        "prompt": ["187", 13],
        "ready_for_generation": ["187", 12],
        "metadata_json": ["187", 4],
    }
    graph["673"]["inputs"].update(
        {
            "generation_mode": ["849", 7],
            "target_duration_seconds": ["849", 0],
            "audio_mode": ["849", 10],
            "audio_guidance": ["675", 1],
            "shot_count": ["849", 8],
            "custom_shot_count": ["849", 1],
            "dialogue_mode": ["849", 11],
            "dialogue_line_count": 2,
            "dialogue_guidance": "",
            "negative_prompt_mode": "auto",
            "negative_prompt_guidance": (
                "No extra people, duplicate performer, identity drift, masculinization, alternate face, "
                "alternate hair or wardrobe, product drift, alternate package, malformed can, alcohol cues, "
                "nightclub, concert, unintended singing, speaking, or lip sync outside the selected performance "
                "mode, floating text, invented logos, unreadable "
                "packaging, incoherent camera path, flicker, or discontinuous product placement."
            ),
            "shot_count_override": ["849", 9],
        }
    )
    graph["674"]["inputs"].update(
        {
            "reference_manifest": ["802", 1],
            "sample_fps": 1.0,
            "max_duration_seconds": 60.0,
            "max_frames": 60,
            "overlong_policy": "trim",
            "visual_description": (
                "Picture 1 and Picture 2 show the same sole adult performer and wardrobe. Picture 3 is a "
                "product-only multi-view contact sheet for the same orange GLOWBLOOM YUZU PEAR can. The "
                "product is independent Subject 2; contact-sheet grids, seams, repeated panels, backgrounds, "
                "and viewpoint sequences never transfer into the video."
            ),
            "reference_manifest_preset": "custom",
            "expected_subject_count": ["802", 2],
            "user_prompt": ["801", 1],
            "reference_images": ["803", 6],
        }
    )
    graph["675"]["inputs"] = {
        "performance_mode": ["849", 2],
        "base_audio_guidance": (
            "Use the exact selected soundtrack as the timing authority. Synchronize edits, hand actions, "
            "product reveals, practical transitions, and physically coherent expressive H3 camera accents to "
            "measured rhythm and recovery. Follow the single selected performance mode: Natural avoids forced "
            "singing or choreography, Dance emphasizes whole-body music sync without lyric mouthing, and Lyrics "
            "permits vocal articulation only when real lyrics and timing evidence are present. Allow controlled "
            "arcs, parallax, sweeps, and compound camera paths when performer and package remain legible."
        ),
    }
    graph["676"]["inputs"] = {
        "production_duration_seconds": ["849", 0],
        "aspect_ratio": ["849", 6],
        "excerpt_start_seconds": ["849", 12],
        "excerpt_duration_seconds": ["849", 0],
        "generation_model": ["849", 13],
        "deliverables_json": ["849", 5],
        "max_h3_shot_seconds": ["849", 14],
        "creative_brief": ["801", 1],
        "master_audio_sha256": ["818", 2],
        "lyrics": ["804", 2],
    }
    graph["678"]["inputs"] = {
        "root_seed": ["418", 0],
        "project_manifest_json": ["676", 0],
    }

    # Disable measured 1.00x acceleration experiments and keep the baseline
    # self-contained. Native ComfyUI attention is used after the proven LoRA;
    # sigma shifts, scheduler, and sampler remain unchanged.
    graph["716"]["inputs"]["model"] = ["713", 0]
    graph["713"]["inputs"]["strength_model"] = 0.75
    graph["713"]["_meta"]["title"] = "Official Ref2VA Turbo LoRA — Advertisement baseline strength 0.75"
    graph["710:658"]["_meta"]["title"] = "Shared H3 schedule — simple / 7 sampling steps"

    for node_id in ("710:717", "710:718", "710:719", "710:720"):
        graph[node_id] = _node(
            "DiffusionGemmaAdvertisementMemoryBarrier",
            graph[node_id]["_meta"]["title"],
            value=graph[node_id]["inputs"]["value"],
            policy="Unload models + clear CUDA cache",
        )

    planner = "823"
    trim_ids = ("681", "683", "691", "699")
    h3_ids = ("710:667", "710:684", "710:692", "710:700")
    decoder_ids = ("710:656", "710:688", "710:696", "710:704")
    relay_gates = (None, "826", "828", "830")
    for index, (trim_id, h3_id, decoder_id, relay_gate) in enumerate(
        zip(trim_ids, h3_ids, decoder_ids, relay_gates), start=1
    ):
        base_output = 4 + (index - 1) * 5
        graph[trim_id]["inputs"] = {
            "start_index": [planner, base_output + 1],
            "duration": [planner, base_output + 2],
            "audio": ["817", 0],
        }
        inputs = graph[h3_id]["inputs"]
        # MiniMaxH3ReferenceToVideo builds conditioning only.  The diffusion
        # MODEL belongs on the downstream scheduler/guiders, and current live
        # ComfyUI rejects a model keyword on this node.  Remove a stale source
        # key before applying the Advertisement-owned conditioning inputs.
        inputs.pop("model", None)
        inputs.update(
            {
                "clip": ["662", 0],
                "vae": ["653", 0],
                "audio_vae": ["654", 0],
                "ref_images.ref_image_0": ["803", 0],
                "ref_images.ref_image_1": ["803", 1],
                "ref_images.ref_image_2": ["803", 2],
                "ref_audios.ref_audio_0": [trim_id, 0],
                "prompt": [planner, base_output],
                "width": ["187", 10],
                "height": ["187", 11],
                "length": [planner, base_output + 3],
                "ref_image_size": "match",
            }
        )
        if relay_gate is None:
            inputs.pop("ref_images.ref_image_3", None)
            graph[h3_id]["_meta"]["title"] = (
                "H3 GENERATION LANE 1 — P1+P2 performer + P3 product + Audio 1"
            )
        else:
            inputs["ref_images.ref_image_3"] = [relay_gate, 0]
            graph[h3_id]["_meta"]["title"] = (
                f"H3 GENERATION LANE {index} — P1+P2 performer + P3 product + P4 relay + Audio 1"
            )
        graph[decoder_id]["_meta"]["title"] = f"Advertisement H3 lane {index} decoded frames"

    graph["710:663"]["inputs"]["noise_seed"] = ["678", 0]
    graph["710:685"]["inputs"]["noise_seed"] = ["678", 1]
    graph["710:693"]["inputs"]["noise_seed"] = ["678", 2]
    graph["710:701"]["inputs"]["noise_seed"] = ["678", 3]


def build_graph(source: dict[str, Any]) -> dict[str, Any]:
    graph = _required_source_nodes(source)
    graph["849"] = _node(
        "DiffusionGemmaAdvertisementWorkflowControls",
        "ADVERTISEMENT CONTROLS — single duration / shot / performance authority",
        master_duration="30 seconds",
        native_shot_count=8,
        performance_mode="Natural / audio-led sync",
    )
    _patch_director_and_h3(graph)

    graph["801"] = _node(
        "DiffusionGemmaAdvertisementCampaignContract",
        "ADVERTISEMENT CAMPAIGN — exact copy, claims, timing and delivery",
        brand_name="GLOWBLOOM",
        product_name="Sparkling Botanical Beverage",
        variant_name="Yuzu Pear",
        audience="Style-conscious adults in their twenties and thirties who want a bright, non-alcoholic social ritual.",
        campaign_objective="Build distinct product recognition and make GLOWBLOOM feel tactile, modern, optimistic, and naturally social.",
        headline="FIND YOUR BRIGHT SIDE",
        call_to_action="SIP THE SHIFT",
        claims_json="[]",
        production_duration_seconds=["849", 0],
        aspect_ratio=["849", 6],
        end_card_duration_seconds=3.0,
        deliverables_json=["849", 5],
        campaign_name="GlowBloom Bright Side",
        legal_line="",
        offer_text="",
        price_text="",
    )
    graph["802"] = _node(
        "DiffusionGemmaAdvertisementReferenceContract",
        "ADVERTISEMENT REFERENCES — performer P1/P2, product P3, relay P4",
        slot_policy="2 performer + 1 product + relay",
        performer_description=(
            "The same adult woman in every performer reference: warm medium-brown skin, oval face, dark "
            "textured shoulder-length hair, slim athletic build, confident friendly presence, and the same "
            "modern coral-and-cream wardrobe."
        ),
        product_description=(
            "The same slim orange GLOWBLOOM YUZU PEAR sparkling botanical beverage can with a silver top, "
            "cream vertical brand panel, warm yellow-orange gradient, and precise cylindrical proportions."
        ),
        product_retention_attributes=(
            "can silhouette, height-to-width ratio, silver lid, orange-to-yellow palette, cream brand panel, "
            "GLOWBLOOM wordmark placement, YUZU PEAR variant hierarchy, material highlights"
        ),
        product_reference_kind="product contact sheet",
        performer_retention_attributes="face, skin tone, hair, body proportions, wardrobe, age, gender presentation",
        package_copy_json='["GLOWBLOOM","YUZU PEAR"]',
    )
    graph["803"] = _node(
        "DiffusionGemmaAdvertisementReferenceAssetPrep",
        "ADVERTISEMENT ASSET PREP — three independent hash-locked roles",
        performer_hero_image=["179", 0],
        performer_sheet_image=["721", 0],
        product_sheet_image=["800", 0],
        # This is the exact 0.4 MP, 9:16, multiple-of-32 result used by
        # JSON Splitter below. Keeping it explicit breaks the otherwise hidden
        # reference-prep -> Director -> reference-context dependency cycle.
        generation_width=480,
        generation_height=864,
        performer_area_ratio=1.0,
        performer_hero_share=0.6,
        product_area_ratio=0.5,
    )
    graph["804"] = _node(
        "DiffusionGemmaAdvertisementSoundtrackContract",
        "ADVERTISEMENT SOUNDTRACK — governed MiniMax Music 3 instrumental",
        content_mode="Instrumental",
        genre_style="luminous future-disco and modern indie-pop instrumental for a premium beverage commercial",
        mood="energetic, stylish, optimistic, tactile, polished, cool and human rather than aggressive",
        instrumentation="tight kick, dry claps, rubbery disco bass, muted funk guitar, fizzy glass percussion, bright arpeggio and shimmering analog pads",
        target_duration_seconds=["849", 0],
        bpm=122.0,
        time_signature="4",
        language="unknown",
        lyrics="",
        voice_over_policy="None",
        arrangement_notes=(
            "Open immediately with a crisp pickup and two-note motif; establish the groove by four seconds; "
            "lift near twelve seconds; thin briefly near twenty seconds for a liquid-splash accent; return to "
            "the fullest hook for the hero reveal; simplify from twenty-eight seconds into a confident final "
            "cadence and short sparkling tail for the end card."
        ),
        do_not_sound_like="nightclub EDM, festival drops, aggressive trap, novelty jingle, vocals, chants or spoken words",
    )
    graph["805"] = _node(
        "DiffusionGemmaAdvertisementMusic3PromptAdapter",
        "MiniMax Music 3 governed prompt adapter",
        soundtrack_contract_json=["804", 0],
    )
    graph["806"] = _node(
        "CLIPLoader",
        "MiniMax Music 3 pruned INT8 text encoder",
        clip_name=MUSIC3_MODELS["text_encoder"],
        type="minimax",
        device="default",
    )
    graph["807"] = _node(
        "UNETLoader",
        "MiniMax Music 3 DiT FP16",
        unet_name=MUSIC3_MODELS["diffusion_model"],
        weight_dtype="default",
    )
    graph["808"] = _node(
        "VAELoader",
        "MiniMax Music 3 DAV",
        vae_name=MUSIC3_MODELS["vae"],
    )
    graph["809"] = _node(
        "MiniMaxMusic3TextEncode",
        "MiniMax Music 3 — 35s commercial candidate conditioning",
        clip=["806", 0],
        caption=["805", 0],
        lyrics=["805", 1],
        seed=2026082302,
        max_duration=["805", 4],
        cfg_scale=1.7,
        top_k=50,
    )
    graph["810"] = _node(
        "ConditioningZeroOut",
        "MiniMax Music 3 negative conditioning",
        conditioning=["809", 0],
    )
    graph["811"] = _node(
        "EmptyMiniMaxMusic3LatentAudio",
        "MiniMax Music 3 candidate latent",
        seconds=["809", 1],
        batch_size=1,
    )
    graph["812"] = _node(
        "KSampler",
        "MiniMax Music 3 — official 30-step path",
        model=["807", 0],
        seed=2026082302,
        steps=30,
        cfg=1.7,
        sampler_name="euler",
        scheduler="simple",
        positive=["809", 0],
        negative=["810", 0],
        latent_image=["811", 0],
        denoise=1.0,
    )
    graph["813"] = _node(
        "VAEDecodeAudioTiled",
        "MiniMax Music 3 tiled decode",
        samples=["812", 0],
        vae=["808", 0],
        tile_size=512,
        overlap=64,
    )
    graph["814"] = _node(
        "DiffusionGemmaAdvertisementSoundtrackSourceRouter",
        "SOUNDTRACK SOURCE — MiniMax Music 3 default / upload / ACE fallback",
        source_mode="MiniMax Music 3",
        music3_candidate_count=1,
        music3_expected_bpm=["805", 3],
        music3_lyrics=["805", 1],
        music3_duration_seconds=["809", 1],
        music3_candidate_1=["813", 0],
    )
    graph["815"] = _node(
        "DiffusionGemmaAdvertisementAudioCandidateSelector",
        "MiniMax Music 3 audition — select a locked 30s excerpt",
        soundtrack_contract_json=["804", 0],
        candidate_count=["814", 4],
        selection_mode="auto_select",
        expected_bpm=["814", 5],
        excerpt_duration_seconds=["849", 0],
        minimum_score=0.52,
        locked_waveform_sha256="",
        locked_start_seconds=-1.0,
        candidate_1=["814", 0],
        candidate_2=["814", 1],
        candidate_3=["814", 2],
        candidate_4=["814", 3],
        source_policy=["814", 8],
    )
    graph["816"] = _node(
        "TrimAudioDuration",
        "Selected MiniMax Music 3 30s master window",
        audio=["815", 0],
        start_index=["815", 1],
        duration=["849", 0],
    )
    graph["817"] = _node(
        "DiffusionGemmaAdvertisementAudioMixer",
        "Music-only motion guide and exact final mix",
        music_audio=["816", 0],
        soundtrack_contract_json=["804", 0],
        target_duration_seconds=["849", 0],
        music_gain_db=0.0,
        voice_over_gain_db=0.0,
        ducking_db=-9.0,
        duck_attack_ms=80.0,
        duck_release_ms=250.0,
        peak_ceiling_dbfs=-1.0,
        clipping_policy="Attenuate mix to ceiling",
    )
    graph["818"] = _node(
        "DiffusionGemmaAdvertisementAudioCandidateSelector",
        "Verify exact 30s master audio and rebuild relative timing report",
        soundtrack_contract_json=["804", 0],
        candidate_count=1,
        selection_mode="lock_candidate_1",
        expected_bpm=["814", 5],
        excerpt_duration_seconds=["849", 0],
        minimum_score=0.0,
        locked_waveform_sha256="",
        locked_start_seconds=0.0,
        candidate_1=["817", 1],
        source_policy="minimax_music3",
    )
    graph["819"] = _node(
        "DiffusionGemmaAdvertisementMemoryBarrier",
        "Unload MiniMax Music 3 before Director and H3",
        value=["818", 3],
        policy="Unload models + clear CUDA cache",
    )
    graph["821"] = _node(
        "DiffusionGemmaAdvertisementMasterContract",
        "ADVERTISEMENT MASTER — campaign, references, soundtrack and H3 lanes",
        campaign_contract_json=["801", 0],
        reference_contract_json=["802", 0],
        soundtrack_contract_json=["804", 0],
        project_manifest_json=["676", 0],
        performer_hero_sha256=["803", 3],
        performer_contact_sheet_sha256=["803", 4],
        product_package_sha256=["803", 5],
        master_audio_sha256=["818", 2],
        native_shot_count=["849", 1],
        boundary_mode="relay_continuity",
    )
    graph["822"] = _node(
        "DiffusionGemmaAdvertisementPlanningDefaults",
        "PRODUCT PRESENCE + RELAY — non-technical defaults",
        advertisement_contract_json=["821", 0],
        product_use=(
            "The performer naturally discovers, carries, opens, presents, pours, and enjoys the can while "
            "its orange package, silver top, cream panel, and proportions remain clearly recognizable."
        ),
        seam_style="Campaign default",
    )
    graph["823"] = _node(
        "DiffusionGemmaAdvertisementMultiShotPlanner",
        "ADVERTISEMENT H3 PLAN — 8 native shots across 2x15s lanes",
        advertisement_contract_json=["821", 0],
        base_h3_prompt=["188", 0],
        shot_metadata_json=["822", 0],
        lane_states_json=["822", 1],
        performance_mode=["849", 2],
    )

    for completed, next_lane, artifact_id, gate_id, decoder_id in (
        (1, 2, "825", "826", "710:656"),
        (2, 3, "827", "828", "710:688"),
        (3, 4, "829", "830", "710:696"),
    ):
        graph[artifact_id] = _node(
            "DiffusionGemmaAdvertisementRelayArtifact",
            f"Persist exact retained tail from lane {completed} for Picture 4",
            lane_images=[decoder_id, 0],
            advertisement_plan_json=["823", 0],
            campaign_id=["821", 1],
            completed_lane_index=completed,
            next_lane_index=next_lane,
            max_megapixels=0.064,
        )
        graph[gate_id] = _node(
            "DiffusionGemmaAdvertisementRelayGate",
            f"Verify persistent Picture 4 for lane {next_lane}",
            relay_artifact_json=[artifact_id, 0],
            advertisement_plan_json=["823", 0],
            campaign_id=["821", 1],
            lane_index=next_lane,
            enabled=True,
        )

    graph["831"] = _node(
        "DiffusionGemmaAdvertisementMasterAssembler",
        "Exact 30s Advertisement master assembly",
        advertisement_contract_json=["821", 0],
        advertisement_plan_json=["823", 0],
        master_audio=["817", 1],
        lane_1_images=["710:656", 0],
        lane_2_images=["710:688", 0],
        lane_3_images=["710:696", 0],
        lane_4_images=["710:704", 0],
    )
    graph["832"] = _node(
        "DiffusionGemmaAdvertisementEndCardRenderer",
        "Deterministic 3s GLOWBLOOM end card",
        advertisement_contract_json=["821", 0],
        width=["187", 10],
        height=["187", 11],
        background_color="#ef6f3c",
        brand_color="#fff8e7",
        accent_color="#ffe06b",
    )
    graph["833"] = _node(
        "DiffusionGemmaAdvertisementMasterFinisher",
        "Replace exact master tail with governed end card",
        assembled_master=["831", 0],
        master_audio=["831", 1],
        end_card_images=["832", 0],
        end_card_report_json=["832", 1],
        advertisement_contract_json=["821", 0],
        assembly_report_json=["831", 2],
    )
    graph["834"] = _node(
        "DiffusionGemmaAdvertisementAdaptationStatus",
        "Requested aspect adaptations — honest render status",
        advertisement_contract_json=["821", 0],
    )
    graph["835"] = _node(
        "DiffusionGemmaAdvertisementCutdownRenderer",
        "Render exact 15s and 6s cutdowns",
        finished_master=["833", 0],
        finished_audio=["833", 1],
        advertisement_contract_json=["821", 0],
        cutdown_15_start_seconds=["849", 3],
        cutdown_6_start_seconds=["849", 4],
        adaptation_status_json=["834", 0],
    )
    graph["836"] = _node(
        "DiffusionGemmaAdvertisementMediaQAGate",
        "REVIEW QA — unmeasured identity/copy checks never approve delivery",
        advertisement_contract_json=["821", 0],
        delivery_manifest_json=["835", 4],
        checks_json=(
            '{"technical_integrity":{"status":"pass","evidence":"exact frame/audio assembly contract"},'
            '"product_identity":"not_measured","performer_identity":"not_measured",'
            '"copy_legibility":"not_measured","audio_sync":"not_measured"}'
        ),
        required_checks_json='["product_identity","performer_identity","copy_legibility","audio_sync","technical_integrity"]',
    )
    graph["837"] = _node(
        "CreateVideo", "Mux exact 30s Advertisement master", images=["833", 0], audio=["833", 1], fps=24.0, bit_depth=8
    )
    graph["838"] = _node(
        "SaveVideo", "SAVE REVIEW DRAFT — 30s GLOWBLOOM master", video=["837", 0],
        filename_prefix="diffusiongemma_advertisement/glowbloom_music3/review_drafts/master_30s", format="mp4", codec="h264", **{"codec.encoding": "auto"}
    )
    graph["839"] = _node(
        "CreateVideo", "Mux exact 15s cutdown", images=["835", 0], audio=["835", 1], fps=24.0, bit_depth=8
    )
    graph["840"] = _node(
        "SaveVideo", "SAVE REVIEW DRAFT — 15s GLOWBLOOM cutdown", video=["839", 0],
        filename_prefix="diffusiongemma_advertisement/glowbloom_music3/review_drafts/cutdown_15s", format="mp4", codec="h264", **{"codec.encoding": "auto"}
    )
    graph["841"] = _node(
        "CreateVideo", "Mux exact 6s cutdown", images=["835", 2], audio=["835", 3], fps=24.0, bit_depth=8
    )
    graph["842"] = _node(
        "SaveVideo", "SAVE REVIEW DRAFT — 6s GLOWBLOOM cutdown", video=["841", 0],
        filename_prefix="diffusiongemma_advertisement/glowbloom_music3/review_drafts/cutdown_6s", format="mp4", codec="h264", **{"codec.encoding": "auto"}
    )
    graph["843"] = _node("PreviewAny", "ADVERTISEMENT QA STATUS", source=["836", 2])
    graph["844"] = _node(
        "SaveAudio", "SAVE REVIEW DRAFT — exact 30s MiniMax Music 3 master", audio=["817", 1],
        filename_prefix="diffusiongemma_advertisement/glowbloom_music3/review_drafts/soundtrack_master_30s"
    )
    graph["845"] = _node("PreviewAudio", "AUDITION exact 30s MiniMax Music 3 master", audio=["817", 1])
    graph["900"] = _node(
        "PreviewAny",
        "SOUNDTRACK QC — always visible, including blocked no-media runs",
        source=["815", 4],
    )
    graph["901"] = _node(
        "PreviewAny",
        "DIRECTOR VALIDATION AUDIT — always visible, including blocked no-media runs",
        source=["187", 4],
    )
    graph["902"] = _node(
        "PreviewAny",
        "DIRECTOR RAW RESPONSE AUDIT — visible when structured output is malformed",
        source=["186", 2],
    )

    # The old runtime capture's output node is retained only as a recognizable
    # source id, but it now points at the new deterministic finished master.
    graph.pop("651", None)

    validate_graph(graph)
    return graph


def _connection_origins(graph: dict[str, Any]) -> set[str]:
    origins: set[str] = set()
    for node in graph.values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                origins.add(value[0])
    return origins


def _assert_acyclic(graph: dict[str, Any]) -> None:
    dependencies = {
        node_id: {
            value[0]
            for value in node.get("inputs", {}).values()
            if isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and value[0] in graph
        }
        for node_id, node in graph.items()
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, trail: list[str]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            start = trail.index(node_id) if node_id in trail else 0
            cycle = trail[start:] + [node_id]
            raise FixtureError("Advertisement graph contains a dependency cycle: " + " -> ".join(cycle))
        visiting.add(node_id)
        trail.append(node_id)
        for dependency in sorted(dependencies[node_id]):
            visit(dependency, trail)
        trail.pop()
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(graph):
        visit(node_id, [])


def validate_graph(graph: dict[str, Any]) -> None:
    missing = sorted(_connection_origins(graph) - set(graph))
    if missing:
        raise FixtureError("Advertisement graph has dangling origins: " + ", ".join(missing))
    _assert_acyclic(graph)
    required_classes = {
        "DiffusionGemmaAdvertisementWorkflowControls",
        "MiniMaxMusic3TextEncode",
        "DiffusionGemmaAdvertisementCampaignContract",
        "DiffusionGemmaAdvertisementReferenceAssetPrep",
        "DiffusionGemmaAdvertisementSoundtrackSourceRouter",
        "DiffusionGemmaAdvertisementAudioCandidateSelector",
        "DiffusionGemmaAdvertisementMemoryBarrier",
        "DiffusionGemmaAdvertisementDirectorPacketRepair",
        "DiffusionGemmaAdvertisementMasterContract",
        "DiffusionGemmaAdvertisementMultiShotPlanner",
        "DiffusionGemmaAdvertisementRelayArtifact",
        "DiffusionGemmaAdvertisementRelayGate",
        "DiffusionGemmaAdvertisementMasterAssembler",
        "DiffusionGemmaAdvertisementEndCardRenderer",
        "DiffusionGemmaAdvertisementCutdownRenderer",
        "DiffusionGemmaAdvertisementMediaQAGate",
    }
    classes = {str(node.get("class_type", "")) for node in graph.values()}
    absent = sorted(required_classes - classes)
    if absent:
        raise FixtureError("Advertisement graph is missing required classes: " + ", ".join(absent))
    outputs = [node for node in graph.values() if node.get("class_type") == "SaveVideo"]
    if len(outputs) != 3:
        raise FixtureError("Advertisement graph must save one master and two real cutdowns.")
    if graph.get("900", {}).get("inputs", {}).get("source") != ["815", 4]:
        raise FixtureError("Advertisement soundtrack QC must remain independently visible when media is blocked.")
    if graph.get("901", {}).get("inputs", {}).get("source") != ["187", 4]:
        raise FixtureError("Advertisement Director validation metadata must remain visible when media is blocked.")
    if graph.get("902", {}).get("inputs", {}).get("source") != ["186", 2]:
        raise FixtureError("Advertisement raw Director response must remain inspectable when structured output is malformed.")
    if graph.get("818", {}).get("inputs", {}).get("expected_bpm") != ["814", 5]:
        raise FixtureError("Advertisement verifier must use the selected soundtrack source BPM authority.")
    shared = graph.get("849", {})
    if shared.get("class_type") != "DiffusionGemmaAdvertisementWorkflowControls":
        raise FixtureError("Advertisement graph lost its single shared settings authority.")
    for node_id, input_name, output_slot in (
        ("801", "production_duration_seconds", 0),
        ("804", "target_duration_seconds", 0),
        ("183", "target_duration_seconds", 0),
        ("183", "max_duration_seconds", 0),
        ("673", "target_duration_seconds", 0),
        ("676", "production_duration_seconds", 0),
        ("815", "excerpt_duration_seconds", 0),
        ("816", "duration", 0),
        ("817", "target_duration_seconds", 0),
        ("818", "excerpt_duration_seconds", 0),
        ("673", "custom_shot_count", 1),
        ("821", "native_shot_count", 1),
        ("675", "performance_mode", 2),
        ("823", "performance_mode", 2),
        ("835", "cutdown_15_start_seconds", 3),
        ("835", "cutdown_6_start_seconds", 4),
        ("801", "deliverables_json", 5),
        ("676", "deliverables_json", 5),
        ("801", "aspect_ratio", 6),
        ("676", "aspect_ratio", 6),
        ("673", "generation_mode", 7),
        ("673", "shot_count", 8),
        ("673", "shot_count_override", 9),
        ("673", "audio_mode", 10),
        ("673", "dialogue_mode", 11),
        ("676", "excerpt_start_seconds", 12),
        ("676", "generation_model", 13),
        ("676", "max_h3_shot_seconds", 14),
        ("187", "resolution_aspect_ratio_override", 15),
    ):
        if graph[node_id]["inputs"].get(input_name) != ["849", output_slot]:
            raise FixtureError(f"Advertisement shared control drifted at {node_id}:{input_name}.")
    forbidden_runtime_classes = {
        "EasyCache",
        "TorchCompileModel",
        "PathchSageAttentionKJ",
        "easy cleanGpuUsed",
        "FL_UnloadAllModels",
    }
    if forbidden_runtime_classes & classes:
        raise FixtureError("Unmeasured acceleration experiments must not enter the Advertisement baseline.")
    for lane, node_id in enumerate(("710:667", "710:684", "710:692", "710:700"), start=1):
        inputs = graph[node_id]["inputs"]
        for reference_index in range(3):
            if f"ref_images.ref_image_{reference_index}" not in inputs:
                raise FixtureError(f"H3 lane {lane} lost persistent Picture {reference_index + 1}.")
        if lane == 1 and "ref_images.ref_image_3" in inputs:
            raise FixtureError("H3 lane 1 cannot consume relay Picture 4.")
        if lane > 1 and "ref_images.ref_image_3" not in inputs:
            raise FixtureError(f"H3 lane {lane} lost relay Picture 4.")


def _manifest(graph: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_api": str(SOURCE_API.relative_to(REPOSITORY)).replace("\\", "/"),
        "source_api_canonical_sha256": SOURCE_API_CANONICAL_SHA256,
        "known_good_v6": str(KNOWN_GOOD_UI.relative_to(REPOSITORY)).replace("\\", "/"),
        "known_good_v6_sha256": KNOWN_GOOD_UI_SHA256,
        "known_good_v6_is_never_overwritten": True,
        "api_graph_canonical_sha256": _canonical_sha256(graph),
        "api_graph_file_sha256": _sha256_bytes(_json_bytes(graph)),
        "node_count": len(graph),
        "models": MODEL_INVENTORY,
        "assets": {
            key: {"relative_input_path": path, "sha256": ASSET_SHA256[key]}
            for key, path in ASSET_PATHS.items()
        },
        "audio_source_default": "MiniMax Music 3",
        "api_audio_sources": ["MiniMax Music 3"],
        "companion_ui_only_audio_sources": ["Upload song", "Legacy ACE-Step"],
        "h3_reference_policy": {
            "picture_1": "performer hero",
            "picture_2": "same-performer contact sheet",
            "picture_3": "independent product/package contact sheet",
            "picture_4": "hash-verified, downscaled copy of the exact retained prior-lane final frame; continuity only",
        },
        "deliverables": ["30s 9:16 review draft", "15s 9:16 review draft", "6s 9:16 review draft"],
        "adaptations": "status-only until a semantic reframe renderer is connected; never falsely marked rendered",
        "qa_default": "not_measured for visual identity, copy legibility, and audio sync until a human or measured checker supplies evidence",
        "save_policy": "Saved media are review drafts; QA pass with evidence is required before delivery approval.",
    }
    value["manifest_sha256"] = _canonical_sha256(value)
    return value


def _write_guarded(path: Path, payload: bytes, *, force: bool) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        current = path.read_bytes()
        if current == payload:
            return "verified"
        if not force:
            raise FixtureError(f"Refusing to overwrite a different artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "replaced" if existed else "created"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_API)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    _load_guarded_json(KNOWN_GOOD_UI, byte_hash=KNOWN_GOOD_UI_SHA256)
    source = _load_guarded_json(SOURCE_API, canonical_hash=SOURCE_API_CANONICAL_SHA256)
    if not isinstance(source, dict):
        raise FixtureError("Source API graph must be a JSON object.")
    graph = build_graph(source)
    manifest = _manifest(graph)
    if args.check:
        print(
            json.dumps(
                {
                    "graph_nodes": len(graph),
                    "graph_canonical_sha256": manifest["api_graph_canonical_sha256"],
                    "known_good_v6_sha256": KNOWN_GOOD_UI_SHA256,
                },
                sort_keys=True,
            )
        )
        return 0
    graph_action = _write_guarded(args.output, _json_bytes(graph), force=args.force)
    manifest_action = _write_guarded(args.manifest, _json_bytes(manifest), force=args.force)
    print(
        json.dumps(
            {
                "graph": graph_action,
                "graph_path": str(args.output.resolve()),
                "manifest": manifest_action,
                "manifest_path": str(args.manifest.resolve()),
                "graph_canonical_sha256": manifest["api_graph_canonical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
