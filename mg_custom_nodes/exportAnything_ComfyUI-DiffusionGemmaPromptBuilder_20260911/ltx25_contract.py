"""Pure LTX-2.5 prompt contracts and host-side validation.

The functions in this module deliberately avoid ComfyUI and Torch imports so
the compiler policy can be regression-tested without loading the UI runtime.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


LTX25_CONTRACT_SCHEMA = "dg-ltx25-contract/1"
LTX25_GENERATION_MODES = (
    "auto",
    "text_to_video",
    "image_to_video",
    "first_last_frame",
)
LTX25_LONG_HORIZON_SCHEMA = "dg-ltx25-long-horizon/1"
LTX25_LONG_HORIZON_MODES = (
    "off",
    "auto",
    "on",
)
LTX25_CAMERA_CAPABILITIES = (
    "stable",
    "advanced",
)
LTX25_LONG_HORIZON_AUTO_THRESHOLD_SECONDS = 20.0

_MODE_ALIASES = {
    "auto": "auto",
    "automatic": "auto",
    "auto_recommended": "auto",
    "text": "text_to_video",
    "t2v": "text_to_video",
    "text_to_video": "text_to_video",
    "text2video": "text_to_video",
    "image": "image_to_video",
    "i2v": "image_to_video",
    "image_to_video": "image_to_video",
    "image2video": "image_to_video",
    "first_last": "first_last_frame",
    "first_and_last": "first_last_frame",
    "first_last_frame": "first_last_frame",
    "first_last_frames": "first_last_frame",
    "flf": "first_last_frame",
    "flf2v": "first_last_frame",
}

_CUT_RE = re.compile(
    r"\b(?:hard|smash|jump|match|rapid(?:-fire)?|quick)\s+cuts?\b"
    r"(?!\s+to\s+(?:black|white)\b)|"
    r"\bafter\s+(?:a|the)\s+cuts?\b(?!\s+to\s+(?:black|white)\b)|"
    r"\b(?:a|the|another)\s+cuts?\s+(?:reveals?|shows?|switches?|transitions?|"
    r"moves?|jumps?|to(?!\s+(?:black|white)\b))\b|"
    r"\b(?:cuts?|cutting)\s+(?:to(?!\s+(?:black|white)\b)|between|back)\b|"
    r"\bcross[- ]?dissolv(?:e|es|ed|ing)\b|"
    r"\b(?:a|the)\s+(?:dissolve|wipe)\s+(?:reveals?|shows?|transitions?|moves?|to)\b|"
    r"\b(?:image|shot|view|scene|picture|frame|screen|it)\s+"
    r"(?:dissolv(?:e|es|ed|ing)|wipe(?:s|d|ing)?)\b|"
    r"\b(?:image|shot|view|scene|picture|frame|screen|it)\s+fades?\s+(?:into|to)\s+"
    r"(?!(?:black|white|darkness)\b)|"
    r"\b(?:montage|rapid-fire cuts|shot changes?)\b",
    re.IGNORECASE,
)
_NEGATED_CUT_ITEM_PATTERN = (
    r"(?:(?:hard|smash|jump|match|rapid(?:-fire)?|quick)\s+)?cuts?|"
    r"cross[- ]?dissolves?|dissolves?|wipes?|fades?"
)
_NEGATED_EDIT_ACTION_PATTERN = (
    r"(?:us(?:e|es|ed|ing)|mak(?:e|es|ing)|made|perform(?:s|ed|ing)?|"
    r"execut(?:e|es|ed|ing)|introduc(?:e|es|ed|ing))"
)
_NEGATED_CUT_SEQUENCE_RE = re.compile(
    rf"\b(?:no|without|never|not(?!\s+only\b)|avoid(?:s|ed|ing)?|do(?:es)?\s+not|"
    rf"do(?:es)?n['’]?t|must\s+not|should\s+not)\s+(?:any\s+)?(?:a\s+|an\s+|the\s+)?"
    rf"(?:{_NEGATED_EDIT_ACTION_PATTERN}\s+)?(?:any\s+)?(?:a\s+|an\s+|the\s+)?"
    rf"(?P<items>(?:{_NEGATED_CUT_ITEM_PATTERN})"
    rf"(?:\s*(?:,\s*(?:and|or|nor)?|and|or|nor|/)\s*(?:{_NEGATED_CUT_ITEM_PATTERN}))*)",
    re.IGNORECASE,
)
_SHOT_VIEWPOINT_INSERT_PATTERN = (
    r"(?:\s*,?\s*(?:at\s+|from\s+)?(?:eye[- ]level|front[- ]facing|"
    r"back[- ]facing|side(?:[- ]view)?|rear(?:[- ]view)?|low[- ]angle|"
    r"high[- ]angle|slightly\s+(?:low[- ]angle|high[- ]angle)|top[- ]down|"
    r"overhead|three[- ]quarter|over[- ]the[- ]shoulder|pov)"
    r"(?:\s+(?:angle|view))?)?"
)
_SHOT_NOUN_RE = re.compile(
    rf"\b(?:(?:extreme[- ]+wide|wide|medium|medium[- ]+wide|long|establishing|"
    rf"medium[- ]+close){_SHOT_VIEWPOINT_INSERT_PATTERN}|(?:extreme\s+)?(?:tight\s+)?"
    rf"close[- ]?up|full[- ]body|insert|over[- ]the[- ]shoulder|pov)[- ]+shot\b",
    re.IGNORECASE,
)
_SHOT_SCALE_RE = re.compile(
    rf"\b(?:"
    rf"(?:extreme[- ]+wide|wide|medium|medium[- ]+wide|long|establishing)"
    rf"{_SHOT_VIEWPOINT_INSERT_PATTERN}[- ]+(?:shot|view|composition)|"
    rf"medium[- ]+close{_SHOT_VIEWPOINT_INSERT_PATTERN}[- ]+shot|"
    rf"(?:extreme[- ]+close[- ]+up|tight[- ]+close[- ]+up|"
    rf"medium[- ]+close[- ]+up|close[- ]+up)(?:[- ]+shot)?|"
    rf"full[- ]+body(?:[- ]+shot)?"
    rf")\b",
    re.IGNORECASE,
)
_SHOT_SCALE_OBJECT_MODIFIER_RE = re.compile(
    r"^\s+(?:portrait|photo(?:graph)?|image|picture|texture|print|painting|"
    r"illustration|artwork|graphic|logo|pattern|detail)\b",
    re.IGNORECASE,
)
_SHOT_SCALE_EXPLICIT_SUFFIX_RE = re.compile(
    r"^\s+(?:shot|view|composition|of\b|frames?|shows?|captures?|holds?|centers?)\b",
    re.IGNORECASE,
)
_SHOT_SCALE_EXPLICIT_PREFIX_RE = re.compile(
    r"\b(?:in|as|from)\s+(?:an?\s+)?$",
    re.IGNORECASE,
)
_SHOT_SCALE_DESTINATION_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:ends?|ending|finishes?|finishing|settles?|settling|arrives?|arriving|"
    r"resolves?|resolving|culminates?|culminating|concludes?|concluding)\b"
    r"[^.!?;]{0,96}\b(?:in|into|on|at|as|with)\s+(?:an?\s+)?"
    r"(?:(?:tighter|wider|closer|looser)\s+)?|"
    r"\b(?:push(?:es|ed|ing)?|pull(?:s|ed|ing)?|doll(?:y|ies|ied|ying)|"
    r"zoom(?:s|ed|ing)?|mov(?:e|es|ed|ing)|glid(?:e|es|ed|ing)|"
    r"transition(?:s|ed|ing)?|shift(?:s|ed|ing)?|tighten(?:s|ed|ing)?|"
    r"widen(?:s|ed|ing)?|refram(?:e|es|ed|ing)|evolv(?:e|es|ed|ing))\b"
    r"[^.!?;]{0,96}\b(?:to|into|toward(?:s)?|until|as)\s+(?:an?\s+)?"
    r"(?:(?:tighter|wider|closer|looser)\s+)?|"
    r"\b(?:becomes?|becoming|reaches?|reaching)\s+(?:an?\s+)?"
    r"(?:(?:tighter|wider|closer|looser)\s+)?"
    r")$",
    re.IGNORECASE,
)
_SHOT_SCALE_DIRECT_DESTINATION_RE = re.compile(
    r"\b(?:to|into|toward(?:s)?|until)\s+(?:an?\s+)?"
    r"(?:(?:tighter|wider|closer|looser)\s+)?$",
    re.IGNORECASE,
)
_UNAMBIGUOUS_EDIT_MATCH_RE = re.compile(
    r"\b(?:hard|smash|jump|match|rapid(?:-fire)?|quick)\s+cuts?\b|"
    r"\bcross[- ]?dissolv(?:e|es|ed|ing)\b|"
    r"\b(?:a|the)\s+(?:dissolve|wipe)\b|"
    r"\b(?:image|shot|view|scene|picture|frame|screen|it)\s+"
    r"(?:dissolv(?:e|es|ed|ing)|wipe(?:s|d|ing)?|fades?)\b|"
    r"\b(?:montage|rapid-fire cuts|shot changes?)\b",
    re.IGNORECASE,
)
_EDITORIAL_CUT_SUBJECT_RE = re.compile(
    r"\b(?:camera|image|shot|view|scene|picture|frame|screen|edit|editor|film|video|"
    r"sequence|we)\b(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)
_EDITORIAL_IMPERATIVE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:then|next|now|finally|immediately|subsequently|afterward)\s*[,;:]?\s*)?$",
    re.IGNORECASE,
)
_EDITORIAL_DESTINATION_RE = re.compile(
    r"\b(?:(?:new|another|second|next|different)\s+)?"
    r"(?:shot|scene|view|framing|frame|image|picture|composition)\b",
    re.IGNORECASE,
)
_EDITORIAL_SHOT_CATEGORY_RE = re.compile(
    r"\b(?:pov|over[- ]the[- ]shoulder(?:\s+shot)?|insert\s+shot)\b",
    re.IGNORECASE,
)
_CAMERA_MOTION_ITEM_PATTERN = (
    r"(?:tracking\s+(?:shot|arc)|whip[- ]pan(?:s|ned|ning)?|"
    r"push(?:es|ed|ing)?(?:[- ]in)?|pull(?:s|ed|ing)?(?:[- ]back)?|"
    r"mov(?:e|es|ed|ing)|"
    r"doll(?:y|ies|ied|ying)(?:[- ]?(?:in|out|forward|back(?:ward)?))?|"
    r"pan(?:s|ned|ning)?|tilt(?:s|ed|ing)?|"
    r"track(?:s|ed|ing)?|truck(?:s|ed|ing)?|follow(?:s|ed|ing)?|"
    r"glid(?:e|es|ed|ing)|sweep(?:s|ed|ing)?|drift(?:s|ed|ing)?|slid(?:e|es|ing)|"
    r"pedestal(?:s|ed|ing)?|boom(?:s|ed|ing)?|crane(?:s|ed|ing)?|"
    r"ris(?:e|es|ing)|rose|descend(?:s|ed|ing)?|"
    r"arc(?:s|ed|ing)?|orbit(?:s|ed|ing)?|circl(?:e|es|ed|ing)|"
    r"zoom(?:s|ed|ing)?|roll(?:s|ed|ing)?|rotat(?:e|es|ed|ing)|"
    r"spin(?:s|ning)?|spun|swirl(?:s|ed|ing)?|"
    r"shak(?:e|es|ing)|shook)"
)
_CAMERA_MOTION_CANDIDATE_RE = re.compile(
    rf"\b(?P<phrase>{_CAMERA_MOTION_ITEM_PATTERN})\b",
    re.IGNORECASE,
)
_CAMERA_MOTION_MODIFIER_PATTERN = (
    r"(?:very|slow|slowly|fast|quick|quickly|brisk|briskly|rapid|rapidly|"
    r"energetic|energetically|assertive|assertively|dynamic|dynamically|"
    r"gentle|gently|smooth|smoothly|gradual|gradually|"
    r"steady|steadily|subtle|subtly|brief|briefly|seamless|seamlessly|slight|slightly|"
    r"controlled|deliberate|deliberately|continuous|cinematic)"
)
_NEGATED_CAMERA_EXTRA_MODIFIER_PATTERN = (
    r"(?:circular|dramatic|sweeping|swirling|dizzying|layered|pronounced)"
)
_NEGATED_CAMERA_ITEM_PATTERN = (
    rf"(?:(?:a|an|the|any)\s+)?"
    rf"(?:camera\s+)?"
    rf"(?:(?:us(?:e|es|ed|ing)|perform(?:s|ed|ing)?|mak(?:e|es|ing)|made|"
    rf"execut(?:e|es|ed|ing)|allow(?:s|ed|ing)?|permit(?:s|ted|ting)?|"
    rf"let(?:s|ting)?)\s+)?"
    rf"(?:(?:a|an|the|any)\s+)?"
    rf"(?:(?:{_CAMERA_MOTION_MODIFIER_PATTERN}|{_NEGATED_CAMERA_EXTRA_MODIFIER_PATTERN})\s+)*"
    rf"(?:camera\s+)?"
    rf"(?:(?:to|from)\s+)?"
    rf"(?:(?:{_CAMERA_MOTION_MODIFIER_PATTERN}|{_NEGATED_CAMERA_EXTRA_MODIFIER_PATTERN})\s+)*"
    rf"{_CAMERA_MOTION_ITEM_PATTERN}(?:\s+(?:shot|move|movement))?"
)
_NEGATED_CAMERA_SEQUENCE_RE = re.compile(
    rf"\b(?:no|without|never|not(?!\s+only\b)|avoid(?:s|ed|ing)?|do(?:es)?\s+not|"
    rf"do(?:es)?n['’]?t|(?:is|are|was|were)\s+not|(?:is|are|was|were)n['’]?t|"
    rf"cannot|can['’]?t|must\s+not|should\s+not|"
    rf"forbid(?:s|den)?|prohibit(?:s|ed)?)\s+"
    rf"(?P<items>{_NEGATED_CAMERA_ITEM_PATTERN}"
    rf"(?:\s*(?:,\s*(?:and|or|nor)?|and|or|nor|/)\s*{_NEGATED_CAMERA_ITEM_PATTERN})*)",
    re.IGNORECASE,
)
_CAMERA_SUPPORT_RE = re.compile(
    r"\b(?:steadicam|gimbal|handheld|tripod|shoulder[- ]mounted|jib|drone|stabili[sz]ed)\b",
    re.IGNORECASE,
)
_CAMERA_BINDING_TAIL_RE = re.compile(
    rf"\b(?:camera|view|framing|lens|shot)\b"
    rf"(?:\s+(?:is|as\s+it|it|begins?|starts?|continues?|"
    rf"then|now|still|never|not|only|do(?:es)?\s+not|do(?:es)?n['’]?t|to|by|a|an|the|"
    rf"{_CAMERA_MOTION_MODIFIER_PATTERN}))*\s*$",
    re.IGNORECASE,
)
_CAMERA_NOMINAL_PREFIX_RE = re.compile(
    rf"\b(?:a|an|the)\s+(?:{_CAMERA_MOTION_MODIFIER_PATTERN}\s+)+(?:camera\s+)?$",
    re.IGNORECASE,
)
_CAMERA_NOMINAL_SUFFIX_RE = re.compile(
    r"^\s+(?:left|right|up(?:ward)?|down(?:ward)?|forward|back(?:ward)?|"
    r"toward|away|across|along|around|past|through|into|out\b)",
    re.IGNORECASE,
)
_CAMERA_EQUIPMENT_SUFFIX_RE = re.compile(
    r"^[- ]+(?:shutter|artifact|marker|lens|arm|track|rail|rig|support|system|device|"
    r"hardware|head|handle|cage)\b",
    re.IGNORECASE,
)
_CAMERA_POSTPOSED_RE = re.compile(
    r"^\s+(?:camera|shot|view|framing)\b",
    re.IGNORECASE,
)
_CAMERA_VIEWPOINT_BRIDGE_PATTERN = (
    r"(?:\s+(?:at|from)\s+(?:an?\s+)?(?:eye[- ]level|low[- ]angle|high[- ]angle|"
    r"front(?:al)?|side|rear|overhead|top[- ]down|three[- ]quarter)(?:\s+(?:angle|view))?)?"
)
_CAMERA_CONTINUATION_RE = re.compile(
    r"^\s*(?:(?:and|or|then|next|afterward|subsequently|finally|while|as|simultaneously|"
    r"but|yet|instead|only|"
    r"smoothly|slowly|quickly|briskly|rapidly|energetically|assertively|dynamically|"
    r"gently|gradually|steadily|subtly|briefly|seamlessly|slightly|"
    r"further|also|still|now|left|right|up(?:ward)?|down(?:ward)?|forward|back(?:ward)?|"
    r"in|out|away|it|camera|the\s+camera|continues?|continuing|begins?|starts?|"
    r"keeps?|moves?|transitions?|shifts?)\b[\s,]*)*$",
    re.IGNORECASE,
)
_CAMERA_SEQUENTIAL_RE = re.compile(
    r"\b(?:then|next|before|after|later|afterward|subsequently|finally|followed\s+by|"
    r"but\s+instead|yet|transitions?\s+to|shifts?\s+to|ends?\s+with|ending\s+with|"
    r"settles?\s+into)\b",
    re.IGNORECASE,
)
_CAMERA_STATE_PATTERNS = (
    re.compile(
        r"\b(?:no|without)\s+(?:any\s+)?camera\s+(?:motion|movement)\b|"
        r"\bcamera\s+(?:does\s+not|doesn['’]?t|never)\s+move\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:camera|shot|view|framing){_CAMERA_VIEWPOINT_BRIDGE_PATTERN}\s+"
        r"(?:is|remains?|stays?)\s+"
        r"(?:perfectly\s+|completely\s+)?(?:static|locked(?:[- ]off)?|fixed|stationary|steady|still)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:camera|shot|view|framing){_CAMERA_VIEWPOINT_BRIDGE_PATTERN}\s+"
        r"(?:locks?\s+off|(?:holds?|holding|held)\s+"
        r"(?:perfectly\s+|completely\s+)?"
        r"(?:static|still|steady|position|on\b(?!\s+(?:focus|exposure))))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:static|locked(?:[- ]off)?|fixed|stationary|steady)\s+"
        r"(?:(?:extreme|tight|medium|wide|long|full[- ]body|establishing|insert|close[- ]?up|"
        r"eye[- ]level|low[- ]angle|high[- ]angle)\s+){0,4}"
        r"(?:camera|shot|view|framing|close[- ]?up|composition)\b",
        re.IGNORECASE,
    ),
)
_VIEWPOINT_RE = re.compile(
    r"\b(?:eye[- ]level|low[- ]angle|high[- ]angle|overhead|bird['\u2019]?s[- ]eye|ground[- ]level|"
    r"waist[- ]height|shoulder[- ]height|profile|side\s+view|front(?:al)?\s+view|rear\s+view|"
    r"from\s+(?:behind|above|below|the\s+left|the\s+right)|three[- ]quarter\s+view|pov)\b",
    re.IGNORECASE,
)
_CONDITIONED_VIEWPOINT_CONTINUITY_RE = re.compile(
    r"\b(?:"
    r"(?:from|at|preserv(?:e|es|ing))\s+(?:the\s+)?(?:exact\s+|same\s+)?(?:supplied\s+)?"
    r"(?:first|last)(?:[- ]frame)?\s+(?:camera\s+)?(?:viewpoint|angle|framing)"
    r"|camera\s+starts?\s+(?:from|with)\s+(?:the\s+)?(?:exact\s+|same\s+)?(?:supplied\s+)?"
    r"(?:first[- ]frame\s+)?(?:viewpoint|angle|framing|this\s+(?:viewpoint|angle|framing))"
    r"|camera\s+starts?\s+with\s+this\s+(?:static\s+)?framing"
    r")\b",
    re.IGNORECASE,
)
_MAJOR_ACTION_RE = re.compile(
    r"\b(?:begins?\s+to|starts?\s+to|then|suddenly|before|after|while|as\s+she|as\s+he|"
    r"turns?|walks?|runs?|jumps?|falls?|strikes?|hits?|kicks?|punches?|throws?|catches?|"
    r"opens?|closes?|enters?|exits?|disappears?|appears?|transforms?|explodes?|collides?)\b",
    re.IGNORECASE,
)
_AUDIO_LAYER_RE = re.compile(
    r"\b(?:dialogue|voiceover|narration|narrator|whispers?|says?|speaks?|sings?|music|score|"
    r"song|ambience|ambient|room tone|footsteps?|heartbeat|foley|wind|rain|traffic|sirens?|"
    r"machinery|engine|gunshot|impact|echo(?:es|ing)?|silence)\b",
    re.IGNORECASE,
)
_SPEECH_CUE_RE = re.compile(
    r"\b(?:voiceover|narrat(?:or|es?|ion)|says?|speaks?|whispers?|shouts?|calls?|sings?|dialogue|words?)\b",
    re.IGNORECASE,
)


def normalize_ltx25_generation_mode(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text.startswith("auto"):
        return "auto"
    if text.startswith("text_to_video"):
        return "text_to_video"
    if text.startswith("image_to_video"):
        return "image_to_video"
    if text.startswith("first_last") or text.startswith("first_and_last"):
        return "first_last_frame"
    if text == "legacy_video":
        return "legacy_video"
    return _MODE_ALIASES.get(text, "auto")


def normalize_ltx25_long_horizon_mode(value: Any) -> str:
    """Normalize UI labels and serialized values without enabling by accident."""

    text = str(value or "off").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text.startswith("auto"):
        return "auto"
    if text in {"on", "enabled", "enable", "true", "force", "forced"}:
        return "on"
    return "off"


def normalize_ltx25_camera_capability(value: Any) -> str:
    """Normalize the user-facing camera capability without opting in silently."""

    text = str(value or "stable").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text.startswith("advanced") or text in {
        "controlled",
        "camera_control",
        "lora",
        "motion_control",
    }:
        return "advanced"
    return "stable"


def _positive_duration_seconds(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 0.0
    return duration if math.isfinite(duration) and duration > 0 else 5.0


def ltx25_long_horizon_plan(mode: Any, duration_seconds: Any) -> dict[str, Any]:
    """Resolve the experimental long-horizon policy into host-owned pacing data.

    The four windows are phases inside one continuous take, never a request for
    four shots or chapters.  Keeping the schedule host-owned avoids expanding
    the model's JSON output with a second plan that can truncate the packet.
    """

    requested_mode = normalize_ltx25_long_horizon_mode(mode)
    duration = _positive_duration_seconds(duration_seconds)
    if requested_mode == "on":
        active = True
        activation_reason = "forced_on"
    elif requested_mode == "auto" and duration > LTX25_LONG_HORIZON_AUTO_THRESHOLD_SECONDS:
        active = True
        activation_reason = "auto_duration_above_20_seconds"
    elif requested_mode == "auto":
        active = False
        activation_reason = "auto_duration_at_or_below_20_seconds"
    else:
        active = False
        activation_reason = "disabled"

    phases: list[dict[str, Any]] = []
    if active:
        boundaries = [
            0.0,
            round(duration * 0.07, 3),
            round(duration * 0.20, 3),
            round(duration * 0.87, 3),
            round(duration, 3),
        ]
        definitions = (
            (
                "establish",
                "Briefly preserve the opening state and introduce only continuous micro-motion.",
            ),
            (
                "commit",
                "Commit smoothly to the single dominant subject-and-camera trajectory.",
            ),
            (
                "sustain_reveal",
                "Sustain that trajectory through most of the clip, revealing rather than replacing scene content.",
            ),
            (
                "settle_hold",
                "Decelerate into the observable terminal composition and hold it long enough to read.",
            ),
        )
        phases = [
            {
                "name": name,
                "start_seconds": boundaries[index],
                "end_seconds": boundaries[index + 1],
                "purpose": purpose,
            }
            for index, (name, purpose) in enumerate(definitions)
        ]

    return {
        "schema": LTX25_LONG_HORIZON_SCHEMA,
        "requested_mode": requested_mode,
        "active": active,
        "activation_reason": activation_reason,
        "duration_seconds": round(duration, 3),
        "experimental_beyond_seconds": LTX25_LONG_HORIZON_AUTO_THRESHOLD_SECONDS,
        "experimental": bool(active and duration > LTX25_LONG_HORIZON_AUTO_THRESHOLD_SECONDS),
        "phase_count": len(phases),
        "phases": phases,
        "recommended_word_range": [150, 195] if active else [150, 220],
        "hard_max_words": 200 if active else 240,
        "recommended_sentence_range": [4, 8] if active else [],
    }


def _truthy(metadata: Mapping[str, Any], key: str) -> bool:
    return metadata.get(key) is True


def resolve_ltx25_generation_mode(
    configured_mode: Any,
    media_metadata: Mapping[str, Any] | None,
    image_count: int = 0,
) -> dict[str, Any]:
    metadata = dict(media_metadata or {})
    configured = normalize_ltx25_generation_mode(configured_mode)
    hint = normalize_ltx25_generation_mode(metadata.get("ltx_generation_mode_hint", "auto"))
    first_attached = _truthy(metadata, "ltx_first_frame_attached")
    last_attached = _truthy(metadata, "ltx_last_frame_attached")
    ltx_context = bool(metadata.get("ltx25_context_schema"))
    source = str(metadata.get("source", "none") or "none").strip().lower()
    h3_context = bool(metadata.get("minimax_h3_mode")) or source.startswith("minimax_h3")
    warnings: list[str] = []

    if configured != "auto":
        resolved = configured
        resolution_source = "target_profile"
    elif hint != "auto":
        resolved = hint
        resolution_source = "ltx_context"
    elif first_attached and last_attached:
        resolved = "first_last_frame"
        resolution_source = "attached_frames"
    elif first_attached:
        resolved = "image_to_video"
        resolution_source = "attached_frames"
    elif last_attached:
        # A last frame alone is not a valid conditioning contract. Resolve to
        # FLF so readiness reports the missing first anchor explicitly.
        resolved = "first_last_frame"
        resolution_source = "attached_frames"
    elif source in {"video", "image+video"} or str(metadata.get("media_synthesis_mode", "")).lower() == "image_identity_video_control":
        resolved = "legacy_video"
        resolution_source = "legacy_video_context"
    elif image_count > 0 or source == "image":
        resolved = "image_to_video"
        resolution_source = "legacy_context_images"
        warnings.append("ltx_mode_inferred_from_generic_image_context")
    else:
        resolved = "text_to_video"
        resolution_source = "no_conditioning_frames"

    if h3_context:
        warnings.append("ltx_h3_reference_context_incompatible")
    if configured != "auto" and hint != "auto" and configured != hint:
        warnings.append("ltx_explicit_mode_overrides_context_hint")

    return {
        "schema": LTX25_CONTRACT_SCHEMA,
        "configured_mode": configured,
        "resolved_mode": resolved,
        "resolution_source": resolution_source,
        "ltx_context": ltx_context,
        "first_frame_attached": first_attached,
        "last_frame_attached": last_attached,
        "analysis_image_count": max(0, int(image_count or 0)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def ltx25_complexity_budget(
    mode: Any,
    duration_seconds: Any,
    long_horizon_mode: Any = "off",
) -> dict[str, Any]:
    normalized_mode = normalize_ltx25_generation_mode(mode)
    if normalized_mode == "auto":
        normalized_mode = "text_to_video"
    duration = _positive_duration_seconds(duration_seconds)
    long_horizon = ltx25_long_horizon_plan(long_horizon_mode, duration)

    if normalized_mode in {"image_to_video", "first_last_frame"}:
        max_shots = 1
        max_cuts = 0
    else:
        max_shots = 1 if duration < 6.0 else 2 if duration < 10.0 else 3 if duration < 14.0 else 4
        max_cuts = max(0, max_shots - 1)
    if long_horizon["active"]:
        # The experimental policy is explicitly a long continuous take in all
        # LTX generation modes. Its four pacing phases must never be compiled
        # as a four-shot sequence merely because T2V normally allows cuts.
        max_shots = 1
        max_cuts = 0
    # Camera motion is a path-continuity concern, not an official hard count.
    # Keep a duration-scaled advisory so diagnostics can flag unusually dense
    # choreography without preventing generation.
    recommended_camera_motion_phases = max(1, min(4, int(math.ceil(duration / 5.0))))
    if normalized_mode not in {"image_to_video", "first_last_frame"}:
        recommended_camera_motion_phases = max(max_shots, recommended_camera_motion_phases)

    max_major_actions = max(1, min(8, int(math.floor(duration / 2.5)) + 1))
    if long_horizon["active"]:
        # Long coherent motion benefits from fewer causal obligations, not an
        # ever-growing event list.  Continuous camera travel and micro-motion
        # carry the unused time instead.
        max_major_actions = min(max_major_actions, 4)

    return {
        "duration_seconds": round(duration, 3),
        "max_shots": max_shots,
        "max_cuts": max_cuts,
        "recommended_camera_motion_phases": recommended_camera_motion_phases,
        "max_major_actions": max_major_actions,
        "max_audio_layers": max(1, min(5, int(math.ceil(duration / 4.0)) + 1)),
        "max_spoken_words": max(4, int(math.floor(duration * 2.4))),
        "recommended_word_range": list(long_horizon["recommended_word_range"]),
        "hard_max_words": int(long_horizon["hard_max_words"]),
        "long_horizon_active": bool(long_horizon["active"]),
    }


def _unique_match_count(pattern: re.Pattern[str], text: str) -> int:
    return len({match.group(0).casefold() for match in pattern.finditer(text)})


def _camera_motion_family(phrase: str) -> str:
    value = phrase.casefold().replace("-", " ")
    if re.search(r"\b(?:push|pull|doll)", value):
        return "dolly"
    if re.search(r"\b(?:track|truck|follow)", value):
        return "track"
    if re.search(r"\b(?:mov|glid|sweep|drift|slid)", value):
        return "move"
    if re.search(r"\b(?:whip\s+pan|pan)", value):
        return "pan"
    if re.search(r"\btilt", value):
        return "tilt"
    if re.search(r"\b(?:pedestal|boom|crane|ris|rose|descend)", value):
        return "vertical"
    if re.search(r"\b(?:arc|orbit|circl)", value):
        return "orbit"
    if re.search(r"\bzoom", value):
        return "zoom"
    if re.search(r"\bsh(?:ak|ook)", value):
        return "shake"
    return "roll"


def _camera_motion_tail(text: str, end: int) -> str:
    tail_end = min(len(text), end + 72)
    next_motion = _CAMERA_MOTION_CANDIDATE_RE.search(text, end, tail_end)
    if next_motion is not None:
        tail_end = min(tail_end, next_motion.start())
    tail = text[end:tail_end]
    boundary = re.search(
        r"[,.!?;:]|\b(?:then|next|before|after|later|afterward|subsequently|finally)\b",
        tail,
        re.IGNORECASE,
    )
    return tail[: boundary.start()] if boundary is not None else tail


def _first_direction(window: str, choices: tuple[tuple[str, str], ...]) -> str:
    matches: list[tuple[int, str]] = []
    for direction, pattern in choices:
        match = re.search(pattern, window, re.IGNORECASE)
        if match is not None:
            matches.append((match.start(), direction))
    return min(matches, default=(0, "unspecified"), key=lambda item: item[0])[1]


def _camera_motion_direction(text: str, start: int, end: int, phrase: str, family: str) -> str:
    del start
    window = (phrase + " " + _camera_motion_tail(text, end)).casefold().replace("-", " ")
    if family == "dolly":
        return _first_direction(
            window,
            (
                ("forward", r"\b(?:push\w*\s+in|doll\w*\s+(?:in|forward)|in|forward|toward)\b"),
                ("backward", r"\b(?:pull\w*\s+back|doll\w*\s+(?:out|back(?:ward)?)|out|back(?:ward)?|away)\b"),
            ),
        )
    if family == "zoom":
        return _first_direction(window, (("in", r"\bin\b"), ("out", r"\bout\b")))
    if family in {"pan", "track"}:
        return _first_direction(
            window,
            (
                ("left", r"\bleft\b"),
                ("right", r"\bright\b"),
                ("forward", r"\b(?:forward|toward)\b"),
                ("backward", r"\b(?:back(?:ward)?|away)\b"),
            ),
        )
    if family == "move":
        return _first_direction(
            window,
            (
                ("left", r"\bleft\b"),
                ("right", r"\bright\b"),
                ("up", r"\b(?:up|upward)\b"),
                ("down", r"\b(?:down|downward)\b"),
                ("forward", r"\b(?:forward|toward)\b"),
                ("backward", r"\b(?:back(?:ward)?|away)\b"),
            ),
        )
    if family in {"tilt", "vertical"}:
        return _first_direction(
            window,
            (("up", r"\b(?:up|upward|rise|rises|rising)\b"), ("down", r"\b(?:down|downward|descend\w*)\b")),
        )
    if family in {"orbit", "roll"}:
        return _first_direction(
            window,
            (
                ("counterclockwise", r"\b(?:counter[- ]?clockwise|anticlockwise)\b"),
                ("clockwise", r"\bclockwise\b"),
                ("left", r"\bleft\b"),
                ("right", r"\bright\b"),
            ),
        )
    return "unspecified"


def _camera_sentence_start(text: str, position: int) -> int:
    return max(text.rfind(".", 0, position), text.rfind("!", 0, position), text.rfind("?", 0, position)) + 1


def _camera_motion_is_equipment(text: str, start: int, end: int, family: str) -> bool:
    prefix = text[max(0, start - 28) : start]
    suffix = text[end : min(len(text), end + 32)]
    if _CAMERA_EQUIPMENT_SUFFIX_RE.search(suffix):
        return True
    if family == "track" and re.search(r"\bdolly\s*$", prefix, re.IGNORECASE):
        return True
    return False


def _camera_motion_is_intrinsic(text: str, start: int, end: int, family: str) -> bool:
    phrase = text[start:end].casefold()
    if re.fullmatch(
        r"(?:whip[- ]pan\w*|push[- ]in|pull[- ]back|dolly[- ](?:in|out)|"
        r"tracking\s+(?:shot|arc))",
        phrase,
    ):
        return True
    prefix = text[max(0, start - 72) : start]
    suffix = text[end : min(len(text), end + 40)]
    nominal_phrase = bool(
        re.fullmatch(
            r"(?:pan|tilt|zoom|doll(?:y|ies)|orbit|arc|push|pull|pedestal)",
            phrase,
            re.IGNORECASE,
        )
    )
    if nominal_phrase and _CAMERA_NOMINAL_PREFIX_RE.search(prefix):
        return True
    if nominal_phrase and re.search(r"\b(?:a|an|the)\s+$", prefix, re.IGNORECASE) and _CAMERA_NOMINAL_SUFFIX_RE.search(suffix):
        return True
    if phrase.endswith("ing") and _CAMERA_POSTPOSED_RE.search(suffix):
        return True
    return False


def _camera_motion_has_explicit_context(text: str, start: int) -> bool:
    sentence_start = _camera_sentence_start(text, start)
    prefix = text[max(sentence_start, start - 180) : start]
    normalized = re.sub(r",[^,]{0,80},", " ", prefix)
    normalized = re.sub(r"[,;:]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return bool(_CAMERA_BINDING_TAIL_RE.search(normalized))


def _camera_motion_is_continuation(text: str, previous_end: int, start: int) -> bool:
    between = text[previous_end:start]
    # Camera ellipsis normally resumes after the latest comma/semicolon/colon:
    # "the camera pans right, smoothly dollies forward, and tilts up".
    tail = re.split(r"[,;:]", between)[-1]
    return bool(_CAMERA_CONTINUATION_RE.fullmatch(tail))


def _camera_negated_ranges(text: str) -> list[tuple[int, int]]:
    ranges = [(match.start("items"), match.end("items")) for match in _NEGATED_CAMERA_SEQUENCE_RE.finditer(text)]
    postfix = re.compile(
        rf"\b{_CAMERA_MOTION_ITEM_PATTERN}\b(?:(?![.!?;]).){{0,32}}?"
        r"\b(?:is|are)\s+(?:forbidden|prohibited|excluded|not\s+allowed)\b",
        re.IGNORECASE,
    )
    ranges.extend((match.start(), match.end()) for match in postfix.finditer(text))
    return ranges


def _span_within_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start >= range_start and end <= range_end for range_start, range_end in ranges)


def _span_overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _cut_negated_ranges(text: str) -> list[tuple[int, int]]:
    return [
        (match.start("items"), match.end("items"))
        for match in _NEGATED_CUT_SEQUENCE_RE.finditer(text)
    ]


def _cut_match_is_editorial(text: str, match: re.Match[str]) -> bool:
    """Disambiguate edit language from physical cuts and cutting actions.

    Qualified cuts, dissolves, wipes, and image fades are intrinsically
    editorial. Bare ``a cut`` / ``cuts to`` phrases need a shot destination or
    a nearby editorial subject; otherwise ordinary wounds, material cuts, and
    lateral cutting actions must not invalidate a conditioned single take.
    """

    matched = match.group(0)
    if _UNAMBIGUOUS_EDIT_MATCH_RE.search(matched):
        return True

    sentence_start = _camera_sentence_start(text, match.start())
    prefix = text[max(sentence_start, match.start() - 96) : match.start()]
    tail = text[match.end() : min(len(text), match.end() + 160)]
    tail = re.split(r"[.!?;]", tail, maxsplit=1)[0]
    destination_named = bool(
        _SHOT_SCALE_RE.search(tail)
        or _EDITORIAL_DESTINATION_RE.search(tail)
        or _EDITORIAL_SHOT_CATEGORY_RE.search(tail)
    )

    # Article-led and "after a cut" forms are grammatically indistinguishable
    # from physical cuts without a destination cue. The LTX contract already
    # requires the new shot's scale, so that cue is both safe and expected.
    if re.match(r"\s*(?:after\s+)?(?:a|the|another)\s+cuts?\b", matched, re.IGNORECASE):
        return destination_named

    if (
        re.match(r"\s*(?:cut|cutting)\s+(?:to|between|back)\b", matched, re.IGNORECASE)
        and _EDITORIAL_IMPERATIVE_PREFIX_RE.fullmatch(prefix)
    ):
        return True

    # Verb-led forms such as "the camera cuts to" are editorial when their
    # subject is explicitly cinematic, or when the destination names a shot.
    return destination_named or bool(_EDITORIAL_CUT_SUBJECT_RE.search(prefix))


def _active_cut_matches(text: str) -> list[re.Match[str]]:
    negated_ranges = _cut_negated_ranges(text)
    return [
        match
        for match in _CUT_RE.finditer(text)
        if not _span_overlaps_any(match.start(), match.end(), negated_ranges)
        and _cut_match_is_editorial(text, match)
    ]


def _camera_state_is_negated(text: str, start: int) -> bool:
    sentence_start = _camera_sentence_start(text, start)
    prefix = text[max(sentence_start, start - 64) : start]
    return bool(
        re.search(
            r"\b(?:no|without|never|not(?!\s+only\b)|avoid(?:s|ed|ing)?)\s+"
            r"(?:any\s+)?(?:(?:a|an|the)\s+)?$",
            prefix,
            re.IGNORECASE,
        )
    )


def _camera_shot_index(cut_spans: list[tuple[int, int]], position: int) -> int:
    return 1 + sum(1 for _start, end in cut_spans if end <= position)


def _canonical_shot_scale(phrase: str) -> str:
    value = re.sub(r"[-\s]+", " ", phrase.casefold()).strip()
    if "extreme wide" in value:
        return "extreme wide shot"
    if "medium close" in value:
        return "medium close-up"
    if "extreme close" in value:
        return "extreme close-up"
    if "tight close" in value or "close up" in value:
        return "close-up"
    if "medium wide" in value:
        return "medium-wide shot"
    if "full body" in value:
        return "full-body shot"
    if "establishing" in value:
        return "establishing shot"
    if "long" in value:
        return "long shot"
    if "wide" in value:
        return "wide shot"
    return "medium shot"


def _shot_scale_band(scale: str) -> str:
    value = scale.casefold()
    if "close" in value:
        return "close"
    if any(token in value for token in ("wide", "long", "full-body", "establishing")):
        return "wide"
    return "medium"


def _analyze_shot_scale_language(
    text: str,
    cut_spans: list[tuple[int, int]],
) -> dict[str, Any]:
    """Classify scale mentions by shot and whether they cover its opening.

    LTX's framing triple describes the *opening* geometry of every shot.  A
    destination scale later in a continuous move is useful caption detail but
    cannot substitute for that opening type.  Keeping both kinds of mention in
    diagnostics preserves the historical raw count while making readiness use
    per-shot opening coverage.
    """

    mentions: list[dict[str, Any]] = []
    opening_shot_indices: set[int] = set()
    any_scale_shot_indices: set[int] = set()
    for match in _SHOT_SCALE_RE.finditer(text):
        start, end = match.span()
        shot_index = _camera_shot_index(cut_spans, start)
        segment_start = 0 if shot_index <= 1 else cut_spans[shot_index - 2][1]
        sentence_start = _camera_sentence_start(text, start)
        prefix = text[max(segment_start, sentence_start, start - 140) : start]
        suffix = text[end : min(len(text), end + 100)]
        prior_in_shot = any(
            int(item["shot_index"]) == shot_index
            and item.get("role") in {"opening", "destination"}
            for item in mentions
        )
        negated = bool(
            re.search(
                r"\b(?:no|not|never|without|avoid(?:s|ed|ing)?|rather\s+than|instead\s+of)"
                r"\s+(?:an?\s+)?$",
                prefix,
                re.IGNORECASE,
            )
        )
        destination = not negated and (
            bool(_SHOT_SCALE_DESTINATION_CUE_RE.search(prefix))
            or (prior_in_shot and bool(_SHOT_SCALE_DIRECT_DESTINATION_RE.search(prefix)))
        )
        canonical = _canonical_shot_scale(match.group(0))
        explicit_scale_context = bool(
            re.search(r"\b(?:shot|view|composition)\b", match.group(0), re.IGNORECASE)
            or _SHOT_SCALE_EXPLICIT_PREFIX_RE.search(prefix)
            or _SHOT_SCALE_EXPLICIT_SUFFIX_RE.match(suffix)
            or destination
        )
        object_modifier = bool(
            not negated
            and not destination
            and _shot_scale_band(canonical) == "close"
            and not explicit_scale_context
            and _SHOT_SCALE_OBJECT_MODIFIER_RE.match(suffix)
        )
        role = (
            "negated"
            if negated
            else "object_modifier"
            if object_modifier
            else "destination"
            if destination
            else "opening"
        )
        mentions.append(
            {
                "text": match.group(0),
                "canonical_scale": canonical,
                "scale_band": _shot_scale_band(canonical),
                "span": [start, end],
                "shot_index": shot_index,
                "role": role,
                "covers_opening": role == "opening",
                "explicit_scale_context": explicit_scale_context,
            }
        )
        if role not in {"negated", "object_modifier"}:
            any_scale_shot_indices.add(shot_index)
        if role == "opening":
            opening_shot_indices.add(shot_index)

    estimated_shots = max(1 if text else 0, len(cut_spans) + 1)
    missing = [
        shot_index
        for shot_index in range(1, estimated_shots + 1)
        if shot_index not in opening_shot_indices
    ]
    active_mentions = [
        item for item in mentions if item.get("role") in {"opening", "destination"}
    ]
    ignored_mentions = [
        item for item in mentions if item.get("role") not in {"opening", "destination"}
    ]
    return {
        "mentions": active_mentions,
        "ignored_mentions": ignored_mentions,
        "raw_mentions": mentions,
        "mention_count": len(active_mentions),
        "raw_mention_count": len(mentions),
        "scale_shot_indices": sorted(any_scale_shot_indices),
        "opening_shot_indices": sorted(opening_shot_indices),
        "missing_opening_shot_indices": missing,
        "opening_coverage_count": len(opening_shot_indices),
    }


def _analyze_camera_language(text: str, cut_spans: list[tuple[int, int]]) -> dict[str, Any]:
    negated_ranges = _camera_negated_ranges(text)
    candidates = list(_CAMERA_MOTION_CANDIDATE_RE.finditer(text))
    active: list[dict[str, Any]] = []
    negated: list[dict[str, Any]] = []
    previous_active_end = -1
    previous_camera_end = -1
    previous_camera_sentence = -1
    phase_by_shot: dict[int, int] = {}

    for match in candidates:
        start, end = match.span("phrase")
        phrase = text[start:end]
        family = _camera_motion_family(phrase)
        if _camera_motion_is_equipment(text, start, end, family):
            continue
        direction = _camera_motion_direction(text, start, end, phrase, family)
        shot_index = _camera_shot_index(cut_spans, start)
        sentence_start = _camera_sentence_start(text, start)
        explicit = _camera_motion_has_explicit_context(text, start)
        intrinsic = _camera_motion_is_intrinsic(text, start, end, family)
        continuation = bool(
            previous_camera_end >= 0
            and previous_camera_sentence == sentence_start
            and _camera_motion_is_continuation(text, previous_camera_end, start)
        )
        record = {
            "family": family,
            "direction": direction,
            "text": phrase,
            "span": [start, end],
            "shot_index": shot_index,
        }
        if _span_within_any(start, end, negated_ranges):
            negated.append(record)
            if explicit or intrinsic or continuation:
                previous_camera_end = end
                previous_camera_sentence = sentence_start
            continue

        if not (explicit or intrinsic or continuation):
            continue

        if shot_index not in phase_by_shot:
            phase_by_shot[shot_index] = 1
        elif previous_active_end >= 0 and _CAMERA_SEQUENTIAL_RE.search(text[previous_active_end:start]):
            phase_by_shot[shot_index] += 1
        record["phase_index"] = phase_by_shot[shot_index]
        active.append(record)
        previous_active_end = end
        previous_camera_end = end
        previous_camera_sentence = sentence_start

    support_matches = [
        {
            "text": match.group(0),
            "span": [match.start(), match.end()],
            "shot_index": _camera_shot_index(cut_spans, match.start()),
        }
        for match in _CAMERA_SUPPORT_RE.finditer(text)
    ]
    state_matches: list[dict[str, Any]] = []
    seen_state_spans: set[tuple[int, int]] = set()
    for pattern in _CAMERA_STATE_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if span in seen_state_spans or _camera_state_is_negated(text, match.start()):
                continue
            seen_state_spans.add(span)
            state_matches.append(
                {
                    "text": match.group(0),
                    "span": [match.start(), match.end()],
                    "shot_index": _camera_shot_index(cut_spans, match.start()),
                }
            )
    state_matches.sort(key=lambda item: (int(item["span"][0]), int(item["span"][1])))

    return {
        "active_motion_mentions": active,
        "negated_motion_mentions": negated,
        "support_mentions": support_matches,
        "state_mentions": state_matches,
        "distinct_families": sorted({str(item["family"]) for item in active}),
        "motion_phase_count": len({(int(item["shot_index"]), int(item["phase_index"])) for item in active}),
        "motion_shot_indices": sorted({int(item["shot_index"]) for item in active}),
        "state_shot_indices": sorted({int(item["shot_index"]) for item in state_matches}),
    }


_STABLE_CAMERA_PARALLAX_RE = re.compile(
    r"\b(?:pronounced|layered|strong|extreme|dramatic|dizzying|disorienting|"
    r"aggressive|heavy|intense|rapid)\s+(?:depth\s+)?parallax\b|"
    r"\b(?:parallax)\b[^.!?;]{0,48}\b(?:pronounced|layered|dizzying|disorienting)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_ORBIT_ROTATION_RE = re.compile(
    r"\b(?:camera|shot|view|framing)\b\s+(?:(?:then|now|initially|finally)\s+)?"
    r"(?:begins?|starts?|continues?|slows?|eases?|makes?|performs?|executes?|uses?|"
    r"takes?|enters?)\b[^.!?;]{0,56}\b(?:circular\s+)?"
    r"(?:orbit|arc|circle[- ]around|rotation|roll|spin|swirl)\b|"
    r"\b(?:orbiting|circling|rotating|rolling|spinning|swirling)\s+(?:camera|shot|view|framing)\b|"
    r"\b(?:camera|shot|view|framing)[- ](?:orbit|arc|rotation|roll|spin|swirl)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_CIRCULAR_TRACKING_ARC_RE = re.compile(
    r"\bcircular\s+(?:camera\s+)?tracking\s+arc\b|"
    r"\b(?:camera|shot|view|framing|viewpoint|perspective)\b\s+"
    r"(?:rotates?|spins?|swivels?)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_PRONOUN_CARRY_RE = re.compile(
    r"\b(?:camera|shot|view|framing)\b\s+"
    r"(?:begins?|starts?|remains?|stays?|is|holds?)\s+"
    r"(?:(?:in|with)\s+(?:(?:a|the)\s+)?)?"
    r"(?:locked|static|stable|stabilized|fixed|steady)\b"
    r"(?:\s+(?:camera|shot|view|framing|position))?"
    r"(?:(?:[.!?]\s*(?:then\s+|near\s+(?:the\s+)?end\s*,?\s*)?)|"
    r"(?:,\s*(?:but|then|yet)\s+))"
    r"it\s+(?!(?:does\s+not|doesn['’]?t|never|must\s+not|should\s+not)\b)"
    r"(?:(?:then|soon|gradually|finally)\s+)?(?:begins?\s+(?:to\s+)?)?"
    r"(?:orbits?|arcs?|circles?|rolls?|rotates?|spins?|swirls?|"
    r"slows?\s+its\s+(?:rotation|roll|spin|swirl))\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_PERSISTENT_RISK_RE = re.compile(
    r"\bcamera\b\s+(?:"
    r"(?:never\s+stops?|does\s+not\s+stop|doesn['’]?t\s+stop)\s+"
    r"(?:its\s+)?(?:rotat(?:e|es|ing|ion)|orbit(?:s|ing)?|roll(?:s|ing)?|"
    r"spin(?:s|ning)?|swirl(?:s|ing)?|sweeping\s+arc)|"
    r"(?:does\s+not\s+cease|doesn['’]?t\s+cease|never\s+ceases?)\s+to\s+"
    r"(?:orbit|rotate|roll|spin|swirl)|"
    r"is\s+never\s+(?:static|locked|fixed|steady)\s+and\s+instead\s+"
    r"(?:orbits?|rotates?|rolls?|spins?|swirls?)|"
    r"(?:does\s+not|doesn['’]?t|never)\s+slow\s+(?:its\s+)?"
    r"(?:sweeping\s+arc|orbit|rotation|roll|spin|swirl)"
    r")\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_SAME_CLAUSE_PRONOUN_RE = re.compile(
    r"\bcamera\b\s+(?:(?:does\s+not|doesn['’]?t|never)\s+)?"
    r"(?:keeps?\s+moving|stop|pause|settle|hold|remain|continue|move|travel|track)\b"
    r"[^.!?;]{0,28}\b(?:as|while|and|then|but|yet)\s+it\s+"
    r"(?!(?:does\s+not|doesn['’]?t|never|must\s+not|should\s+not)\b)"
    r"(?:(?:then|soon|gradually|finally)\s+)?(?:begins?\s+(?:to\s+)?)?"
    r"(?:orbits?|arcs?|circles?|rolls?|rotates?|spins?|swirls?)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_DOLLY_ZOOM_RE = re.compile(
    r"\b(?:dolly|dollies|dollied|dollying)\s+zoom\b|"
    r"\b(?:vertigo|zolly)\s+(?:effect|shot|move)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_HANDHELD_PURSUIT_RE = re.compile(
    r"\bhandheld\b[^.!?;]{0,80}\b(?:pursuit|chase|chases|chasing|follow|follows|"
    r"following|track|tracks|tracking|run|runs|running)\b|"
    r"\b(?:pursuit|chase|chases|chasing)\b[^.!?;]{0,80}\bhandheld\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_COMPOUND_RE = re.compile(
    r"\b(?:compound|multi[- ]axis|multi[- ]phase|multi[- ]directional|complex)\b"
    r"[^.!?;]{0,56}\b(?:camera|camera[- ]motion|move|movement|path|trajectory|choreography)\b|"
    r"\b(?:camera|camera[- ]motion|move|movement|path|trajectory|choreography)\b"
    r"[^.!?;]{0,56}\b(?:compound|multi[- ]axis|multi[- ]phase|multi[- ]directional)\b|"
    r"\b(?:camera|view|framing)\b[^.!?;]{0,64}\b(?:reverses?\s+(?:direction|course)|"
    r"doubles?\s+back|changes?\s+direction|swings?\s+back)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_SWEEP_RE = re.compile(
    r"\b(?:camera|shot|view|framing)\b\s+(?:(?:then|now|initially|finally)\s+)?"
    r"(?:begins?|starts?|continues?|makes?|performs?|executes?|uses?|sweeps?|whip[- ]?pans?)\b"
    r"[^.!?;]{0,56}\b(?:sweeping\s+(?:arc|orbit|move|movement|rotation)|"
    r"whip[- ]?pan(?:s|ned|ning)?)\b|"
    r"\b(?:sweeping\s+camera\s+(?:arc|orbit|move|movement|rotation)|"
    r"whip[- ]?pan(?:s|ned|ning)?)\b",
    re.IGNORECASE,
)
_STABLE_CAMERA_SUSTAINED_TRAVEL_RE = re.compile(
    r"\b(?:camera|shot|view|framing)\b\s+(?:(?:then|now|initially|finally)\s+)?"
    r"(?:continues?(?:\s+(?:to|its|the))?|sustains?|travels?|tracks?|moves?|orbits?|arcs?|sweeps?)\b"
    r"[^.!?;]{0,100}\b(?:through most|for most|throughout|for the duration|"
    r"for the remainder|all the way through)\b|"
    r"\b(?:through most|for most|throughout|for the duration|for the remainder)\b"
    r"[^.!?;]{0,100}\b(?:camera|shot|view|framing)\b\s+"
    r"(?:continues?(?:\s+(?:to|its|the))?|sustains?|travels?|tracks?|moves?|orbits?|arcs?|sweeps?)\b|"
    r"\bsustained\s+(?:camera\s+)?(?:travel|movement|motion|path|tracking)\b",
    re.IGNORECASE,
)


def _stable_camera_risk_span_is_negated(
    text: str,
    start: int,
    end: int,
    camera_negated_ranges: list[tuple[int, int]],
) -> bool:
    """Treat explicit negative constraints as guards, not positive camera cues."""

    if _span_overlaps_any(start, end, camera_negated_ranges):
        return True
    sentence_start = _camera_sentence_start(text, start)
    prefix = text[sentence_start:start]
    contrast_matches = list(
        re.finditer(r"\b(?:but|however|instead|yet)\b", prefix, re.IGNORECASE)
    )
    if contrast_matches:
        prefix = prefix[contrast_matches[-1].end() :]
    negation_matches = list(
        re.finditer(
            r"\b(?:no(?!\s+longer\b)|without|never|avoid(?:s|ed|ing)?|"
            r"do(?:es)?\s+not|do(?:es)?n['’]?t|must\s+not|should\s+not|"
            r"forbid(?:s|den)?|prohibit(?:s|ed)?)\b",
            prefix,
            re.IGNORECASE,
        )
    )
    if negation_matches:
        scope = prefix[negation_matches[-1].end() :]
        # "without a locked camera" and "no longer locked" negate a safe
        # camera state, not the risky move that follows after the comma.
        state_only = bool(
            re.match(
                r"\s*(?:(?:a|an|the)\s+)?(?:locked|static|fixed|stabilized|steady)\s+"
                r"(?:camera|shot|view|framing)\b",
                scope,
                re.IGNORECASE,
            )
        )
        risk_scoped = bool(
            re.match(
                r"\s*(?:(?:use|uses|using|author|authors|authoring|perform|performs|"
                r"performing|make|makes|making|allow|allows|allowing)\s+)?"
                r"(?:(?:any|a|an|the)\s+)?(?:(?:slow|dramatic|circular|sweeping|"
                r"swirling|dizzying|layered|pronounced|compound|multi[- ]axis)\s+){0,3}"
                r"(?:camera\s+)?(?:orbit|arc|circle|rotation|roll|spin|swirl|sweep|"
                r"whip[- ]?pan|dolly\s+zoom|shake|handheld\s+pursuit|parallax|"
                r"sustained\s+(?:camera\s+)?(?:travel|motion|movement|path|tracking))\b",
                scope,
                re.IGNORECASE,
            )
        )
        if risk_scoped and not state_only:
            return True
    tail = text[end : min(len(text), end + 64)]
    return bool(
        re.match(
            r"[^.!?;]{0,32}\b(?:is|are)\s+(?:forbidden|prohibited|excluded|"
            r"not\s+allowed)\b",
            tail,
            re.IGNORECASE,
        )
    )


def _stable_camera_regex_matches(
    text: str,
    pattern: re.Pattern[str],
    camera_negated_ranges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        if _stable_camera_risk_span_is_negated(
            text,
            match.start(),
            match.end(),
            camera_negated_ranges,
        ):
            continue
        matches.append(
            {
                "text": match.group(0),
                "span": [match.start(), match.end()],
            }
        )
    return matches


def _stable_camera_motion_has_intervening_subject_role(text: str, start: int) -> bool:
    """Reject camera ownership when a later subject explicitly performs the move."""

    sentence_start = _camera_sentence_start(text, start)
    prefix = text[max(sentence_start, start - 140) : start]
    return bool(
        re.search(
            r"\b(?:while|as)\s+(?:the\s+)?(?:[a-z][\w'-]*\s+){0,4}"
            r"(?:performs?|makes?|begins?|starts?|executes?|traces?)\s+"
            r"(?:an?|the)\s*$",
            prefix,
            re.IGNORECASE,
        )
    )


def _stable_camera_pronoun_has_prior_subject(text: str, start: int) -> bool:
    sentence_start = _camera_sentence_start(text, start)
    prefix = text[sentence_start:start]
    return bool(
        re.search(
            r"\b(?:frames?|shows?|captures?|centers?|holds?\s+on)\b",
            prefix,
            re.IGNORECASE,
        )
    )


def _stable_camera_pronoun_has_explicit_camera_ownership(
    text: str,
    start: int,
    end: int,
) -> bool:
    """Resolve a narrow camera-pronoun carry despite an earlier framed subject."""

    matched = text[start:end]
    tail = text[end : min(len(text), end + 48)]
    return bool(
        re.search(
            r"\b(?:camera|shot|view|framing)\b\s+(?:begins?|starts?)\b",
            matched,
            re.IGNORECASE,
        )
        or re.match(r"\s+around\b", tail, re.IGNORECASE)
    )


def analyze_ltx25_stable_camera_risks(
    prompt: Any,
    camera_diagnostics: Mapping[str, Any] | None = None,
    long_horizon_active: bool = False,
) -> dict[str, Any]:
    """Return deterministic high-risk positive camera cues for stable I2V."""

    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    diagnostics = dict(camera_diagnostics or _analyze_camera_language(text, []))
    negated_ranges = _camera_negated_ranges(text)
    active = [
        dict(item)
        for item in diagnostics.get("active_motion_mentions", [])
        if isinstance(item, Mapping)
        and not _stable_camera_motion_has_intervening_subject_role(
            text,
            int(item.get("span", [0, 0])[0]),
        )
        and not _stable_camera_risk_span_is_negated(
            text,
            int(item.get("span", [0, 0])[0]),
            int(item.get("span", [0, 0])[1]),
            negated_ranges,
        )
    ]
    categories: dict[str, list[dict[str, Any]]] = {
        "orbit_or_rotation": [
            item for item in active if str(item.get("family", "")) in {"orbit", "roll"}
        ],
        "sweeping_or_whip_motion": [
            item
            for item in active
            if re.search(r"\b(?:sweep|whip[- ]?pan)", str(item.get("text", "")), re.IGNORECASE)
        ],
        "dolly_zoom": _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_DOLLY_ZOOM_RE,
            negated_ranges,
        ),
        "shake_or_handheld_pursuit": [
            item for item in active if str(item.get("family", "")) == "shake"
        ],
        "compound_or_reversal": _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_COMPOUND_RE,
            negated_ranges,
        ),
        "pronounced_parallax": _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_PARALLAX_RE,
            negated_ranges,
        ),
        "sustained_travel": (
            _stable_camera_regex_matches(
                text,
                _STABLE_CAMERA_SUSTAINED_TRAVEL_RE,
                negated_ranges,
            )
            if long_horizon_active
            else []
        ),
    }
    categories["orbit_or_rotation"].extend(
        _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_ORBIT_ROTATION_RE,
            negated_ranges,
        )
    )
    categories["orbit_or_rotation"].extend(
        _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_CIRCULAR_TRACKING_ARC_RE,
            negated_ranges,
        )
    )
    categories["orbit_or_rotation"].extend(
        _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_SAME_CLAUSE_PRONOUN_RE,
            negated_ranges,
        )
    )
    pronoun_carry_matches = _stable_camera_regex_matches(
        text,
        _STABLE_CAMERA_PRONOUN_CARRY_RE,
        negated_ranges,
    )
    categories["orbit_or_rotation"].extend(
        item
        for item in pronoun_carry_matches
        if (
            not _stable_camera_pronoun_has_prior_subject(
                text,
                int(item.get("span", [0, 0])[0]),
            )
            or _stable_camera_pronoun_has_explicit_camera_ownership(
                text,
                int(item.get("span", [0, 0])[0]),
                int(item.get("span", [0, 0])[1]),
            )
        )
    )
    categories["orbit_or_rotation"].extend(
        _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_PERSISTENT_RISK_RE,
            negated_ranges,
        )
    )
    categories["sweeping_or_whip_motion"].extend(
        _stable_camera_regex_matches(text, _STABLE_CAMERA_SWEEP_RE, negated_ranges)
    )
    categories["shake_or_handheld_pursuit"].extend(
        _stable_camera_regex_matches(
            text,
            _STABLE_CAMERA_HANDHELD_PURSUIT_RE,
            negated_ranges,
        )
    )

    active_families = {
        str(item.get("family", ""))
        for item in active
        if str(item.get("family", ""))
    }
    if len(active_families) > 1:
        categories["compound_or_reversal"].append(
            {
                "text": ", ".join(sorted(active_families)),
                "span": [],
                "diagnostic": "multiple_camera_motion_families",
            }
        )
    directions_by_family: dict[str, set[str]] = {}
    for item in active:
        family = str(item.get("family", ""))
        direction = str(item.get("direction", "unspecified"))
        if family and direction != "unspecified":
            directions_by_family.setdefault(family, set()).add(direction)
    reversed_families = sorted(
        family for family, directions in directions_by_family.items() if len(directions) > 1
    )
    if reversed_families:
        categories["compound_or_reversal"].append(
            {
                "text": ", ".join(reversed_families),
                "span": [],
                "diagnostic": "camera_direction_reversal",
            }
        )

    reason_map = {
        "orbit_or_rotation": "ltx_stable_camera_orbit_or_rotation",
        "sweeping_or_whip_motion": "ltx_stable_camera_sweeping_or_whip_motion",
        "dolly_zoom": "ltx_stable_camera_dolly_zoom",
        "shake_or_handheld_pursuit": "ltx_stable_camera_shake_or_handheld_pursuit",
        "compound_or_reversal": "ltx_stable_camera_compound_or_reversal",
        "pronounced_parallax": "ltx_stable_camera_pronounced_parallax",
        "sustained_travel": "ltx_stable_camera_sustained_travel",
    }
    reasons = [reason_map[name] for name, matches in categories.items() if matches]
    return {
        "camera_capability": "stable",
        "high_risk_positive_language": bool(reasons),
        "categories": categories,
        "reasons": reasons,
    }


_LONG_HORIZON_PHASE_PATTERNS: dict[str, re.Pattern[str]] = {
    "establish": re.compile(
        r"\b(?:initially|at first|for (?:a|one) brief moment|briefly (?:holds?|remains?|stays?)|"
        r"during the opening (?:(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten) seconds?|portion|moments?)|"
        r"the opening (?:portion|moments?) (?:holds?|establishes?))\b",
        re.IGNORECASE,
    ),
    "commit": re.compile(
        r"\b(?:after (?:that|this|the brief hold|the opening)|from there|then (?:begins?|starts?|eases?|(?:gradually )?commits?)|"
        r"commits?(?: gradually)? to|eases? into (?:the|a) (?:move|path|motion|trajectory))\b",
        re.IGNORECASE,
    ),
    "sustain_reveal": re.compile(
        r"\b(?:through most of (?:the )?(?:shot|clip|take)|for most of (?:the )?(?:shot|clip|take)|"
        r"through(?:out)? the (?:long|sustained|middle) (?:portion|section)|"
        r"continues? (?:steadily|at (?:a )?constant|along the same path)|"
        r"sustains? (?:the|this) (?:move|path|motion|trajectory))\b",
        re.IGNORECASE,
    ),
    "settle_hold": re.compile(
        r"\b(?:only near the end|toward(?:s)? the end|(?:only )?during the final (?:(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten) seconds?|portion|moments?)|"
        r"in the final (?:portion|moments?)|finally (?:slows?|decelerates?|settles?|holds?)|"
        r"eases? (?:smoothly )?to (?:a )?(?:stop|rest)|settles? (?:into|on)|"
        r"holds? (?:on|through) the final|the final frame holds?)\b",
        re.IGNORECASE,
    ),
}

_LONG_HORIZON_TERMINAL_COMPOSITION_RE = re.compile(
    r"\b(?:final frame|ends?|finishes?|settles?|resolves?|holds?)\b[^.!?]{0,180}"
    r"\b(?:composition|framing|view|tableau|arrangement|subject|scene|landscape|horizon|shot)\b|"
    r"\b(?:composition|framing|view|tableau|arrangement)\b[^.!?]{0,100}"
    r"\b(?:at|by|near|toward(?:s)?|in|on) the end\b",
    re.IGNORECASE,
)
_LONG_HORIZON_CONSERVATION_RE = re.compile(
    r"\b(?:retain(?:s|ed|ing)?|remain(?:s|ed|ing)?|preserv(?:e|es|ed|ing)|"
    r"maintain(?:s|ed|ing)?|keep(?:s|ing)?|stay(?:s|ed|ing)?)\b[^.!?]{0,150}"
    r"\b(?:same|unchanged|fixed|identity|count|proportion|color|position|relationship|geometry|appearance|wardrobe|prop|anchor)\b|"
    r"\b(?:same|unchanged|fixed)\b[^.!?]{0,100}"
    r"\b(?:identity|count|proportion|color|position|relationship|geometry|appearance|wardrobe|prop|anchor)\b",
    re.IGNORECASE,
)
_LONG_HORIZON_FORBIDDEN_DISCONTINUITY_RE = re.compile(
    r"\b(?:no|without|never)\b[^.!?]{0,140}"
    r"\b(?:cuts?|duplication|replacement|teleport(?:s|ing|ation)?|spontaneous growth|"
    r"popping objects?|identity changes?|lens changes?|lighting drift|abrupt refram(?:e|ing))\b",
    re.IGNORECASE,
)


def _ltx_sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    sentences = re.findall(r"[^.!?]+(?:[.!?]+(?:[\"\u201d])?|$)", text)
    return len([item for item in sentences if re.search(r"[A-Za-z0-9]", item)])


def analyze_ltx25_long_horizon_prompt(
    prompt: Any,
    long_horizon_mode: Any,
    duration_seconds: Any,
    generation_mode: Any = "text_to_video",
) -> dict[str, Any]:
    """Return nonblocking diagnostics for the experimental pacing contract."""

    raw_text = str(prompt or "")
    text = re.sub(r"\s+", " ", raw_text).strip()
    plan = ltx25_long_horizon_plan(long_horizon_mode, duration_seconds)
    phase_hits = {
        name: bool(pattern.search(text))
        for name, pattern in _LONG_HORIZON_PHASE_PATTERNS.items()
    }
    covered_phases = [name for name, matched in phase_hits.items() if matched]
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    sentence_count = _ltx_sentence_count(text)
    paragraph_count = len(
        [item for item in re.split(r"(?:\r?\n){2,}|\r?\n", raw_text) if item.strip()]
    )
    terminal_composition = bool(_LONG_HORIZON_TERMINAL_COMPOSITION_RE.search(text))
    conservation = bool(_LONG_HORIZON_CONSERVATION_RE.search(text))
    forbidden_discontinuities = bool(_LONG_HORIZON_FORBIDDEN_DISCONTINUITY_RE.search(text))
    warnings: list[str] = []
    normalized_generation_mode = normalize_ltx25_generation_mode(generation_mode)

    if plan["active"]:
        if len(covered_phases) < 4:
            warnings.append("ltx_long_horizon_phase_coverage_incomplete")
        if not terminal_composition:
            warnings.append("ltx_long_horizon_terminal_composition_missing")
        if normalized_generation_mode in {"image_to_video", "first_last_frame"} and not conservation:
            warnings.append("ltx_long_horizon_conservation_cue_missing")
        if paragraph_count != 1:
            warnings.append("ltx_long_horizon_not_single_paragraph")
        if sentence_count < 4 or sentence_count > 8:
            warnings.append("ltx_long_horizon_sentence_count_outside_4_8")
        if word_count < 150:
            warnings.append("ltx_long_horizon_prompt_below_150_words")
        elif word_count > 200:
            warnings.append("ltx_long_horizon_prompt_over_200_words")

    return {
        "schema": LTX25_LONG_HORIZON_SCHEMA,
        "requested_mode": plan["requested_mode"],
        "active": bool(plan["active"]),
        "activation_reason": plan["activation_reason"],
        "duration_seconds": plan["duration_seconds"],
        "experimental": bool(plan["experimental"]),
        "phase_plan": list(plan["phases"]),
        "targets": {
            "phase_count": int(plan["phase_count"]),
            "word_range": list(plan["recommended_word_range"]),
            "hard_max_words": int(plan["hard_max_words"]),
            "sentence_range": list(plan["recommended_sentence_range"]),
            "single_paragraph": bool(plan["active"]),
        },
        "observed": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "phase_coverage_count": len(covered_phases),
            "covered_phases": covered_phases,
            "phase_cues": phase_hits,
            "terminal_composition_cue": terminal_composition,
            "conservation_cue": conservation,
            "forbidden_discontinuity_cue": forbidden_discontinuities,
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def analyze_ltx25_prompt_complexity(
    prompt: Any,
    mode: Any,
    duration_seconds: Any,
    long_horizon_mode: Any = "off",
) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    normalized_mode = normalize_ltx25_generation_mode(mode)
    if normalized_mode == "auto":
        normalized_mode = "text_to_video"
    budget = ltx25_complexity_budget(normalized_mode, duration_seconds, long_horizon_mode)
    cut_matches = _active_cut_matches(text)
    cut_count = len(cut_matches)
    cut_spans = [match.span() for match in cut_matches]
    shot_nouns = len(list(_SHOT_NOUN_RE.finditer(text)))
    estimated_shots = max(1 if text else 0, cut_count + 1)
    camera_diagnostics = _analyze_camera_language(text, cut_spans)
    camera_motion_matches = list(camera_diagnostics["active_motion_mentions"])
    camera_moves = len(camera_diagnostics["distinct_families"])
    camera_states = len(camera_diagnostics["state_mentions"])
    scale_diagnostics = _analyze_shot_scale_language(text, cut_spans)
    shot_scales = int(scale_diagnostics["mention_count"])
    viewpoints = len(list(_VIEWPOINT_RE.finditer(text)))
    conditioned_viewpoint_anchors = 0
    if normalized_mode in {"image_to_video", "first_last_frame"}:
        conditioned_viewpoint_anchors = len(
            list(_CONDITIONED_VIEWPOINT_CONTINUITY_RE.finditer(text))
        )
        # A conditioning frame already supplies the actual camera geometry.
        # A source-relative continuity statement is safer than guessing a
        # low/high/three-quarter angle that the evidence ledger may not name.
        viewpoints = max(viewpoints, conditioned_viewpoint_anchors)
    major_actions = _unique_match_count(_MAJOR_ACTION_RE, text)
    audio_layers = _unique_match_count(_AUDIO_LAYER_RE, text)
    spoken_words = 0
    for quote_match in re.finditer(r'["\u201c]([^"\u201d]+)["\u201d]', text):
        attribution = text[max(0, quote_match.start() - 180) : quote_match.start()]
        if _SPEECH_CUE_RE.search(attribution):
            spoken_words += len(re.findall(r"\b[\w'-]+\b", quote_match.group(1)))
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    reasons: list[str] = []
    warnings: list[str] = []

    if normalized_mode in {"image_to_video", "first_last_frame"} and cut_count:
        reasons.append("ltx_conditioned_mode_requires_single_continuous_take")
    if re.search(r"\b(?:rapid(?:-fire)?|quick)\s+cuts?\b", text, flags=re.IGNORECASE):
        reasons.append("ltx_unspecified_rapid_cuts")
    if estimated_shots > int(budget["max_shots"]):
        reasons.append("ltx_shot_count_exceeds_duration_budget")
    if int(camera_diagnostics["motion_phase_count"]) > int(budget["recommended_camera_motion_phases"]):
        warnings.append("ltx_camera_complexity_at_duration_limit")
    if major_actions > int(budget["max_major_actions"]) + 2:
        reasons.append("ltx_action_density_exceeds_duration_budget")
    elif major_actions > int(budget["max_major_actions"]):
        warnings.append("ltx_action_density_at_duration_limit")
    if audio_layers > int(budget["max_audio_layers"]) + 2:
        reasons.append("ltx_audio_density_exceeds_duration_budget")
    elif audio_layers > int(budget["max_audio_layers"]):
        warnings.append("ltx_audio_density_at_duration_limit")
    if spoken_words > int(budget["max_spoken_words"]):
        reasons.append("ltx_speech_exceeds_duration_budget")
    if text and normalized_mode != "legacy_video":
        if scale_diagnostics["missing_opening_shot_indices"]:
            reasons.append("ltx_shot_scale_missing")
        camera_covered_shots = set(camera_diagnostics["motion_shot_indices"]) | set(
            camera_diagnostics["state_shot_indices"]
        )
        if len(camera_covered_shots) < estimated_shots:
            reasons.append("ltx_camera_state_missing")
        if normalized_mode == "text_to_video" and viewpoints < estimated_shots:
            reasons.append("ltx_viewpoint_missing")
    if text and word_count < 140:
        warnings.append("ltx_prompt_below_recommended_detail")
    elif word_count > 240:
        warnings.append("ltx_prompt_above_recommended_detail")
    long_horizon = analyze_ltx25_long_horizon_prompt(
        prompt,
        long_horizon_mode,
        duration_seconds,
        normalized_mode,
    )
    warnings.extend(long_horizon["warnings"])
    if long_horizon["active"] and major_actions > 4:
        warnings.append("ltx_long_horizon_action_density_high")

    return {
        "schema": LTX25_CONTRACT_SCHEMA,
        "mode": normalized_mode,
        "budget": budget,
        "observed": {
            "word_count": word_count,
            "estimated_shots": estimated_shots,
            "explicit_shot_phrases": shot_nouns,
            "shot_scale_count": shot_scales,
            "shot_scale_raw_mention_count": scale_diagnostics["raw_mention_count"],
            "shot_scale_mentions": scale_diagnostics["mentions"],
            "ignored_shot_scale_mentions": scale_diagnostics["ignored_mentions"],
            "shot_scale_shot_indices": scale_diagnostics["scale_shot_indices"],
            "opening_shot_scale_shot_indices": scale_diagnostics["opening_shot_indices"],
            "missing_opening_shot_scale_shot_indices": scale_diagnostics[
                "missing_opening_shot_indices"
            ],
            "opening_shot_scale_coverage_count": scale_diagnostics[
                "opening_coverage_count"
            ],
            "cut_count": cut_count,
            "distinct_camera_moves": camera_moves,
            "distinct_camera_states": camera_states,
            "camera_motion_matches": camera_motion_matches,
            "camera_motion_mention_count": len(camera_motion_matches),
            "camera_motion_phase_count": int(camera_diagnostics["motion_phase_count"]),
            "negated_camera_move_count": len(camera_diagnostics["negated_motion_mentions"]),
            "camera_support_count": len(camera_diagnostics["support_mentions"]),
            "camera_diagnostics": camera_diagnostics,
            "viewpoint_count": viewpoints,
            "conditioned_viewpoint_anchor_count": conditioned_viewpoint_anchors,
            "distinct_major_actions": major_actions,
            "distinct_audio_layers": audio_layers,
            "quoted_spoken_words": spoken_words,
        },
        "long_horizon": long_horizon,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _frame_claims(
    media_metadata: Mapping[str, Any],
    asset_ids: set[str],
) -> list[dict[str, Any]]:
    ledger = media_metadata.get("verified_grounding_ledger")
    if not isinstance(ledger, Mapping):
        return []
    claims: list[dict[str, Any]] = []
    for fact in ledger.get("observed_facts", []):
        if not isinstance(fact, Mapping) or str(fact.get("confidence", "")).lower() not in {"high", "medium"}:
            continue
        evidence = fact.get("evidence")
        if not isinstance(evidence, list) or not any(
            isinstance(item, Mapping) and str(item.get("asset_id", "")).lower() in asset_ids
            for item in evidence
        ):
            continue
        claims.append(dict(fact))
    return claims


def _first_frame_claims(media_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _frame_claims(media_metadata, {"image:1", "picture:1"})


def _last_frame_claims(media_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _frame_claims(media_metadata, {"image:2", "picture:2"})


def _grounding_coverage(claims: list[dict[str, Any]], has_verified_ledger: bool) -> dict[str, Any]:
    categories = {
        str(category)
        for fact in claims
        for category in fact.get("categories", [])
        if isinstance(category, str)
    }
    families: list[str] = []
    if categories & {"identity", "appearance", "object", "count"}:
        families.append("subject")
    if categories & {"environment", "lighting", "color"}:
        families.append("setting_light")
    if categories & {"camera", "composition", "spatial"}:
        families.append("composition")
    return {
        "available": bool(claims),
        "status": "verified" if len(families) >= 2 else "incomplete" if has_verified_ledger else "not_verifiable",
        "families": families,
        "fact_count": len(claims),
    }


def _opening_text(prompt: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", prompt.strip())
    return " ".join(sentences[:2])[:700]


def _source_opening_conflicts(prompt: str, media_metadata: Mapping[str, Any]) -> list[str]:
    claims = _first_frame_claims(media_metadata)
    source = " ".join(str(item.get("claim", "")) for item in claims).lower()
    source_scale_text = " ".join(
        str(item.get("claim", ""))
        for item in claims
        if {
            str(category).casefold()
            for category in item.get("categories", [])
            if isinstance(category, str)
        }
        & {"camera", "composition", "spatial"}
    ).lower()
    opening = _opening_text(prompt).lower()
    conflicts: list[str] = []
    axes = (
        ("daylight_to_night", r"\b(?:daylight|daytime|sunlit|bright day|blue sky)\b", r"\b(?:night|nighttime|moonlight|after dark|midnight)\b"),
        ("night_to_daylight", r"\b(?:night|nighttime|moonlight|after dark|midnight)\b", r"\b(?:daylight|daytime|sunlit|bright day|blue sky)\b"),
        ("indoor_to_outdoor", r"\b(?:indoors?|interior|inside a|inside the)\b", r"\b(?:outdoors?|exterior|outside|open sky)\b"),
        ("outdoor_to_indoor", r"\b(?:outdoors?|exterior|outside|open sky|street)\b", r"\b(?:indoors?|interior|inside a|inside the)\b"),
    )
    for name, source_pattern, opening_pattern in axes:
        if re.search(source_pattern, source) and re.search(opening_pattern, opening):
            conflicts.append(name)

    # Scale evolution later in the same continuous take is not an opening
    # contradiction. Compare the source geometry only with the first scale
    # classified as opening shot coverage, never with a destination close-up
    # or wide shot mentioned in the following sentence.
    source_scale = _analyze_shot_scale_language(source_scale_text, [])["mentions"]
    prompt_cuts = [match.span() for match in _active_cut_matches(prompt)]
    prompt_scale = _analyze_shot_scale_language(prompt, prompt_cuts)["mentions"]
    source_opening = next(
        (item for item in source_scale if item.get("role") == "opening"),
        None,
    )
    prompt_opening = next(
        (
            item
            for item in prompt_scale
            if item.get("role") == "opening" and int(item.get("shot_index", 0)) == 1
        ),
        None,
    )
    if source_opening and prompt_opening:
        source_band = str(source_opening.get("scale_band", ""))
        prompt_band = str(prompt_opening.get("scale_band", ""))
        if source_band == "close" and prompt_band == "wide":
            conflicts.append("closeup_to_wide")
        elif source_band == "wide" and prompt_band == "close":
            conflicts.append("wide_to_closeup")
    return conflicts


def validate_ltx25_prompt(
    prompt: Any,
    mode_report: Mapping[str, Any],
    media_metadata: Mapping[str, Any] | None,
    duration_seconds: Any,
    long_horizon_mode: Any = "off",
    camera_capability: Any = "advanced",
) -> dict[str, Any]:
    metadata = dict(media_metadata or {})
    mode = normalize_ltx25_generation_mode(mode_report.get("resolved_mode", "text_to_video"))
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    reasons: list[str] = []
    warnings = list(mode_report.get("warnings", [])) if isinstance(mode_report.get("warnings"), list) else []

    first_attached = bool(mode_report.get("first_frame_attached"))
    last_attached = bool(mode_report.get("last_frame_attached"))
    if "ltx_h3_reference_context_incompatible" in warnings:
        reasons.append("ltx_h3_reference_context_incompatible")
    if mode == "text_to_video" and (first_attached or last_attached):
        reasons.append("ltx_t2v_has_conditioning_frame")
    elif mode == "image_to_video" and not first_attached:
        if mode_report.get("resolution_source") == "legacy_context_images":
            warnings.append("ltx_first_frame_role_unverified_in_generic_context")
        else:
            reasons.append("ltx_i2v_first_frame_missing")
    elif mode == "first_last_frame":
        if not first_attached:
            reasons.append("ltx_flf_first_frame_missing")
        if not last_attached:
            reasons.append("ltx_flf_last_frame_missing")

    coverage: dict[str, Any] = {"available": False, "status": "not_applicable", "families": []}
    last_coverage: dict[str, Any] = {"available": False, "status": "not_applicable", "families": []}
    if mode in {"image_to_video", "first_last_frame"} and first_attached:
        claims = _first_frame_claims(metadata)
        has_verified_ledger = isinstance(metadata.get("verified_grounding_ledger"), Mapping)
        coverage = _grounding_coverage(claims, has_verified_ledger)
        if has_verified_ledger and coverage["status"] != "verified":
            reasons.append("ltx_first_frame_grounding_incomplete")
        elif not has_verified_ledger:
            warnings.append("ltx_first_frame_feasibility_not_verifiable")
    if mode == "first_last_frame" and last_attached:
        has_verified_ledger = isinstance(metadata.get("verified_grounding_ledger"), Mapping)
        last_coverage = _grounding_coverage(_last_frame_claims(metadata), has_verified_ledger)
        if has_verified_ledger and last_coverage["status"] != "verified":
            reasons.append("ltx_last_frame_grounding_incomplete")
        elif not has_verified_ledger:
            warnings.append("ltx_last_frame_feasibility_not_verifiable")

    contradictions = (
        _source_opening_conflicts(text, metadata)
        if mode in {"image_to_video", "first_last_frame"} and first_attached
        else []
    )
    if contradictions:
        reasons.append("ltx_first_frame_prompt_conflict")

    complexity = analyze_ltx25_prompt_complexity(
        prompt,
        mode,
        duration_seconds,
        long_horizon_mode,
    )
    normalized_camera_capability = normalize_ltx25_camera_capability(
        camera_capability
    )
    stable_camera_risks = analyze_ltx25_stable_camera_risks(
        text,
        complexity.get("observed", {}).get("camera_diagnostics", {}),
        bool(complexity.get("long_horizon", {}).get("active")),
    )
    complexity["camera_capability"] = normalized_camera_capability
    complexity["stable_camera_risks"] = stable_camera_risks
    if normalized_camera_capability == "stable":
        complexity["reasons"] = list(
            dict.fromkeys(
                [*complexity["reasons"], *stable_camera_risks["reasons"]]
            )
        )
    if mode in {"image_to_video", "first_last_frame"} and first_attached:
        # The conditioning frame supplies the real camera geometry.  Keep the
        # compiler instruction that asks for the full framing triple, but do
        # not force the text model to guess an angle or scale that the pixels
        # already determine. Missing scale/viewpoint remains a hard error for
        # unconditioned T2V, and every post-cut shot still needs its own scale.
        observed = complexity["observed"]
        missing_scale_shots = [
            int(item)
            for item in observed.get("missing_opening_shot_scale_shot_indices", [])
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ]
        inherited_opening_scale = 1 in missing_scale_shots
        effective_missing_scale_shots = [
            shot_index for shot_index in missing_scale_shots if shot_index != 1
        ]
        observed["conditioned_shot_scale_inherited_from_frame"] = inherited_opening_scale
        observed["effective_missing_opening_shot_scale_shot_indices"] = (
            effective_missing_scale_shots
        )
        if inherited_opening_scale:
            complexity["warnings"] = list(
                dict.fromkeys(
                    [
                        *complexity["warnings"],
                        "ltx_conditioned_shot_scale_inherited_from_frame",
                    ]
                )
            )
            if not effective_missing_scale_shots:
                complexity["reasons"] = [
                    reason
                    for reason in complexity["reasons"]
                    if reason != "ltx_shot_scale_missing"
                ]
        if int(observed.get("viewpoint_count", 0)) < int(observed.get("estimated_shots", 0)):
            complexity["warnings"] = list(
                dict.fromkeys(
                    [
                        *complexity["warnings"],
                        "ltx_conditioned_viewpoint_inherited_from_frame",
                    ]
                )
            )
            observed["conditioned_viewpoint_inherited_from_frame"] = True
    reasons.extend(complexity["reasons"])
    warnings.extend(complexity["warnings"])
    return {
        "schema": LTX25_CONTRACT_SCHEMA,
        "mode": mode,
        "camera_capability": normalized_camera_capability,
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "first_frame_grounding": coverage,
        "last_frame_grounding": last_coverage,
        "first_frame_conflicts": contradictions,
        "complexity": complexity,
    }


def ltx25_compiler_contract(
    mode: Any,
    duration_seconds: Any,
    budget: Mapping[str, Any] | None = None,
    long_horizon_mode: Any = "off",
    audio_enabled: bool = True,
    camera_capability: Any = "advanced",
) -> str:
    normalized_mode = normalize_ltx25_generation_mode(mode)
    if normalized_mode == "auto":
        normalized_mode = "text_to_video"
    normalized_camera_capability = normalize_ltx25_camera_capability(
        camera_capability
    )
    stable_camera = normalized_camera_capability == "stable"
    long_horizon = ltx25_long_horizon_plan(long_horizon_mode, duration_seconds)
    limits = dict(
        budget
        or ltx25_complexity_budget(
            normalized_mode,
            duration_seconds,
            long_horizon["requested_mode"],
        )
    )
    duration = float(limits.get("duration_seconds", 5.0))
    prompt_length_rule = (
        "Write the LTX-2.5 positive prompt as one continuous chronological natural-English training caption of 150-195 words, never more than 200 words, and generally 4-8 sentences. "
        if long_horizon["active"]
        else "Write the LTX-2.5 positive prompt as one continuous chronological natural-English training caption, normally 150-220 words. "
    )
    audio_contract = (
        "Integrate sound beside the event that causes it. Preserve exact requested speech in ordinary straight double quotes with speaker and delivery attribution. "
        if audio_enabled
        else ""
    )
    fact_scope = "visible and audible facts" if audio_enabled else "visible facts"
    camera_path_contract = (
        "Camera capability is Stable / base model, and this limit overrides creativity intensity and any otherwise-unspecified camera ambition. "
        "Prefer an explicitly locked, static, or stabilized camera. When movement materially improves legibility, use only one restrained single-axis move: a short gentle push or pull, a small pan or tilt, or a short lateral track; keep the subject framing readable and settle without changing perspective aggressively. "
        "Do not author an orbit, circular arc, circle-around, roll, rotation, spin, camera swirl, sweeping or whip movement, dolly zoom, camera shake, handheld pursuit, compound or multi-axis path, direction reversal, pronounced/layered/dizzying parallax, or sustained camera travel. "
        "Creativity may still shape art direction, lighting, subject performance, choreography, sound, and environmental motion. "
        if stable_camera
        else (
            "A continuous path may evolve shot scale and viewpoint without creating a new shot. Compatible simultaneous or sequential motion phases may occur within that path; describe their direction, speed, support, parallax, timing, and start-to-end framing without turning them into cuts. "
        )
    )
    common = prompt_length_rule + (
        f"Describe only objective {fact_scope} in active present tense; do not explain intent or emotion that cannot be observed. "
        "For every shot, weave in exactly one official opening shot type—extreme wide shot, wide shot, medium shot, medium close-up, close-up, or extreme close-up—followed by its viewpoint or angle, then say whether the camera remains explicitly locked/static or how it moves through a physically continuous subject-relative path. Use flowing grammar such as 'a medium shot frames the subject, captured from a front-facing angle'; never emit framing tags or labels. "
    ) + camera_path_contract + audio_contract
    limits_text = (
        f"The clip lasts {duration:.3f} seconds. Use at most {int(limits['max_shots'])} shot(s), "
        f"{int(limits['max_cuts'])} cut(s), and about {int(limits['max_major_actions'])} major action beat(s); "
        "keep camera motion physically achievable in the available time and simplify the brief rather than cramming more events into the duration. "
    )
    if normalized_mode == "image_to_video" and long_horizon["active"]:
        mode_text = (
            "This is long-horizon image-to-video. Silently inspect the supplied first frame and build a minimal critical anchor set containing only the identity, subject count, key props, colors, and spatial relationships that must remain stable. Do not emit that inventory. "
            "In the final caption, open with only the few critical first-frame anchors needed for continuity, then concentrate on what changes next; do not redundantly recaption every static visible detail. "
            "Motion begins from the exact supplied frame without changing its time of day, location, subject identity, subject count, key-prop relationships, or opening camera geometry. "
            "Use one continuous take with no hard cuts, montage, teleports, discontinuous reframing, replacement, duplication, unrequested growth, or newly invented off-frame participant required to make the action work. "
            + (
                "Keep the camera locked or stabilized through the long take; if one restrained move is necessary, complete that single-axis move early and hold the resulting readable composition rather than sustaining camera travel. "
                if stable_camera
                else "Use one dominant physically continuous camera intention. Compatible acceleration, deceleration, reframing, tilt, or parallax phases may evolve inside that same path, but never become separate shots or reversals unless the brief explicitly requests them. "
            )
            + "Conservation applies only to facts the brief does not ask to change: preserve all requested transformations instead of freezing, omitting, or weakening them. "
            "Reveal existing space and detail through motion rather than making objects pop into existence. End in a concrete observable composition and hold it. "
        )
    elif normalized_mode == "image_to_video":
        mode_text = (
            "This is image-to-video. Begin by describing the supplied first frame exactly: existing subjects and count, appearance and wardrobe, environment, lighting, composition, exactly one opening shot type from the official list, viewpoint, and opening camera geometry. The opening shot type must describe the supplied frame; an ending or destination scale later in the camera move does not replace it. "
            "Name the observed angle when it is clear; otherwise say that the camera starts from the exact supplied first-frame camera viewpoint instead of guessing an angle. "
            "The generated motion starts from that exact frame without changing its time of day, location, subject identity, subject count, or opening framing. "
            "A single still frame does not determine the future camera path after the opening instant. Explicit verified camera evidence remains authoritative. "
            + (
                "When future camera behavior is unspecified, default to a locked or stabilized hold and add at most one restrained single-axis move only when it improves legibility. A risky camera request requires Advanced / controlled camera rather than silent execution in Stable mode. "
                if stable_camera
                else "Explicit user camera behavior always wins. When the future path is unspecified, choose a physically suitable camera state or path according to the selected creativity mode and depart smoothly from the exact opening geometry. "
            )
            + "Use one continuous take with no hard cuts, montage, teleports, discontinuous reframing, or newly invented off-frame participant required to make the main action work. "
            + (
                "One continuous take does not require camera travel; carry the duration with readable subject, prop, lighting, atmospheric, and environmental motion while the camera stays stable. "
                if stable_camera
                else "One continuous take constrains editing and spatial continuity, not camera speed, spatial travel, or the number of compatible camera-motion phases. Do not default to a slow zoom, generic pan, or static hold merely because the input is a still. "
            )
            + "Introduce only motion and details that can plausibly develop from the visible first-frame geometry during the available duration. "
        )
    elif normalized_mode == "first_last_frame" and long_horizon["active"]:
        mode_text = (
            "This is long-horizon first-and-last-frame video. Silently identify only the stable correspondences needed to connect the supplied endpoints, then emit a compact caption focused on the changes between them rather than exhaustively restating either image. "
            "Use one physically continuous take from the exact first-frame camera geometry to the exact last-frame subjects, environment, lighting, composition, destination scale, and viewpoint. "
            "Preserve identity, count, key props, and causal state; do not use cuts, montage, teleports, duplication, replacement, discontinuous lighting, or a path that cannot connect both anchors. "
            + (
                "Use the least-invasive locked or single-axis stabilized bridge compatible with both endpoints, then hold the exact terminal composition; require Advanced / controlled camera if the endpoints demand compound travel or a viewpoint reversal. "
                if stable_camera
                else "Use one dominant camera intention with compatible internal motion phases, then decelerate into the exact terminal composition and hold it. "
            )
        )
    elif normalized_mode == "first_last_frame":
        mode_text = (
            "This is first-and-last-frame video. Describe the supplied first frame exactly, including exactly one opening shot type from the official list, then one physically continuous take that bridges to the supplied last frame, and finish on the last frame's exact subjects, environment, lighting, composition, destination shot scale, and viewpoint. The destination scale never substitutes for the opening shot type. "
            "Name observed endpoint angles when they are clear; otherwise describe the move from the exact supplied first-frame camera viewpoint to the exact supplied last-frame camera viewpoint instead of guessing. "
            "Do not use hard cuts, montage, teleports, identity swaps, discontinuous lighting changes, or a camera path that cannot connect both anchors. "
            + (
                "The still endpoints do not determine the bridge between them. Use the least-invasive locked or single-axis stabilized camera state compatible with both endpoints; do not invent assertive, fast, compound, or reversing camera travel, and require Advanced / controlled camera if the endpoint geometries cannot be connected safely. "
                "The continuous bridge may change subject action and environmental state while camera movement remains restrained. "
                if stable_camera
                else (
                    "The still endpoints do not determine the camera path between them or require an inert bridge. Explicit user or verified-evidence camera behavior always wins. When the bridge is unspecified, choose physically suitable choreography according to the selected creativity mode; non-faithful modes may make it assertive, fast, or compound. "
                    "The continuous bridge may use compatible simultaneous or sequential motion phases when they form one feasible path between the anchored viewpoints; endpoint constraints limit discontinuity, not motion ambition. "
                )
            )
            + "Every intermediate action must causally move the first-frame state toward the last-frame state within the available duration. "
        )
    elif normalized_mode == "legacy_video":
        mode_text = (
            "This is a legacy source-video recreation or image-identity/video-control workflow. Preserve the supplied clip chronologically: its opening framing, subject-camera relationship, action and blocking, camera path or static hold, focus and lens changes, and closing framing. "
            "When an identity image is also supplied, transfer only that image's subject identity and appearance while the source video continues to control pose, motion, timing, composition, camera choreography, and scene geometry. "
        )
    elif long_horizon["active"]:
        mode_text = (
            "This is long-horizon text-to-video with no conditioning frame. Establish the opening subject, setting, lighting, composition, shot scale, viewpoint, and camera state directly from the brief. "
            "Develop the entire duration as one continuous take with no cuts, transitions, montage, teleports, or unrelated location changes. "
            + (
                "Use subject, prop, lighting, and environmental evolution—not sustained camera travel—to arrive at the terminal composition in the same stable shot. "
                if stable_camera
                else "Use the terminal composition as the destination of the single camera-and-subject trajectory, not as a new shot. "
            )
        )
    else:
        mode_text = (
            "This is text-to-video with no conditioning frame. Establish the opening subject, setting, lighting, composition, shot scale, viewpoint, and camera state directly from the brief. "
            "A cut is allowed only when it reveals a genuinely new subject, space, state, viewpoint, or time and remains inside the shot budget. "
        )
    long_horizon_text = ""
    if long_horizon["active"]:
        phases = {item["name"]: item for item in long_horizon["phases"]}
        if stable_camera:
            continuity_ledger = (
                "Silently check a conservation ledger, one stable camera state, the observable terminal composition, forbidden unrequested discontinuities, and one continuous audio identity. "
                if audio_enabled
                else "Silently check a conservation ledger, one stable camera state, the observable terminal composition, and forbidden unrequested discontinuities. "
            )
            sustained_detail = (
                "Use at most four independent causal actions regardless of clip length; fill time with readable sustained subject, prop, lighting, atmospheric, and environmental motion, restrained micro-motion, ambience, and synchronized transient sounds rather than camera travel or extra plot beats. "
                if audio_enabled
                else "Use at most four independent causal actions regardless of clip length; fill time with readable sustained subject, prop, lighting, atmospheric, and environmental motion and restrained micro-motion rather than camera travel or extra plot beats. "
            )
            commit_object = "one dominant subject-and-environment motion"
            sustain_object = "that motion while the camera remains stable"
        else:
            continuity_ledger = (
                "Silently check a conservation ledger, one dominant camera path, the observable terminal composition, forbidden unrequested discontinuities, and one continuous audio identity. "
                if audio_enabled
                else "Silently check a conservation ledger, one dominant camera path, the observable terminal composition, and forbidden unrequested discontinuities. "
            )
            sustained_detail = (
                "Use at most four independent causal actions regardless of clip length; fill time with sustained movement, parallax, restrained subject micro-motion, ambience, and synchronized transient sounds rather than extra plot beats. "
                if audio_enabled
                else "Use at most four independent causal actions regardless of clip length; fill time with sustained movement, parallax, and restrained subject micro-motion rather than extra plot beats. "
            )
            commit_object = "the one dominant path"
            sustain_object = "that path"
        long_horizon_text = (
            "Experimental long-horizon planning is active. Before drafting, silently preplan a four-phase pacing arc inside the same continuous take; do not emit the plan, phase names, headings, timestamps, timecodes, shot labels, or planning metadata. "
            "The silent phase labels are establish, commit, sustain/reveal, and settle/hold. "
            f"Use about {float(phases['establish']['end_seconds']) - float(phases['establish']['start_seconds']):.3f} seconds to establish the anchored state with micro-motion, "
            f"about {float(phases['commit']['end_seconds']) - float(phases['commit']['start_seconds']):.3f} seconds to commit smoothly to {commit_object}, "
            f"about {float(phases['sustain_reveal']['end_seconds']) - float(phases['sustain_reveal']['start_seconds']):.3f} seconds to sustain {sustain_object} and reveal existing space, and "
            f"about {float(phases['settle_hold']['end_seconds']) - float(phases['settle_hold']['start_seconds']):.3f} seconds to decelerate into a concrete terminal composition and hold. Treat these durations as elastic pacing guidance, not cuts or chapters. "
            "Express that pacing with natural relative language such as initially, then, through most of the shot, and only near the end. "
            f"{continuity_ledger}"
            f"{sustained_detail}"
            "If the world is already surreal, preserve that premise instead of escalating it through unrelated transformations unless the brief requests transformation. "
        )
    return common + mode_text + limits_text + long_horizon_text


__all__ = [
    "LTX25_CAMERA_CAPABILITIES",
    "LTX25_CONTRACT_SCHEMA",
    "LTX25_GENERATION_MODES",
    "LTX25_LONG_HORIZON_AUTO_THRESHOLD_SECONDS",
    "LTX25_LONG_HORIZON_MODES",
    "LTX25_LONG_HORIZON_SCHEMA",
    "analyze_ltx25_long_horizon_prompt",
    "analyze_ltx25_prompt_complexity",
    "analyze_ltx25_stable_camera_risks",
    "ltx25_compiler_contract",
    "ltx25_complexity_budget",
    "ltx25_long_horizon_plan",
    "normalize_ltx25_generation_mode",
    "normalize_ltx25_camera_capability",
    "normalize_ltx25_long_horizon_mode",
    "resolve_ltx25_generation_mode",
    "validate_ltx25_prompt",
]
