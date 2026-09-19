"""Pure, dependency-free contracts for DiffusionGemma visual grounding.

This module deliberately has no ComfyUI, Transformers, Torch, or network imports.
It owns the host-verifiable parts of the Grounding Guard: settings, asset IDs,
strict ledger validation, refusal-clause detection, guard decisions, compact
reports, and JSON-only trace persistence.
"""

from __future__ import annotations

import copy
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal, DecimalException
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence


GROUNDING_CONFIG_TYPE = "DG_GROUNDING_GUARD_CONFIG"
GROUNDING_LEDGER_SCHEMA_ID = "dg-grounding-ledger/1"
ASSET_REGISTRY_SCHEMA_ID = "dg-asset-registry/1"
GROUNDING_REPORT_SCHEMA_ID = "dg-grounding-report/1"
EXTERNAL_EVIDENCE_SCHEMA_ID = "dg-external-evidence/1"
GROUNDING_EVIDENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "grounding_evidence.schema.json"
)
EXTERNAL_EVIDENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "external_evidence.schema.json"
)

GROUNDING_MODES = ("off", "audit", "strict")
SAMPLING_PROFILES = ("checkpoint_defaults", "full_48_diagnostic")
EVIDENCE_TOKEN_BUDGETS = ("auto", "768", "1024", "1280")
ANALYSIS_STATUSES = (
    "not_run",
    "not_applicable",
    "grounded",
    "uncertain",
    "refused",
    "transport_error",
)
LEDGER_ANALYSIS_STATUSES = ("grounded", "uncertain", "refused", "transport_error")
GUARD_DECISIONS = ("disabled", "not_applicable", "pass", "warn", "block")
CONFIDENCE_LEVELS = ("high", "medium", "low")
FACT_CATEGORIES = (
    "identity",
    "appearance",
    "object",
    "count",
    "color",
    "text",
    "spatial",
    "environment",
    "lighting",
    "action",
    "motion",
    "camera",
    "composition",
    "temporal",
    "other",
)
_FACT_CATEGORY_ALIASES = {
    # The model sometimes describes visible clothing with this intuitive label.
    # Keep the public ledger taxonomy canonical and normalize only this exact,
    # unambiguous alias at the model-output validation boundary.
    "wardrobe": "appearance",
}
KNOWN_EXTERNAL_PROVIDERS = frozenset(
    {"manual", "ocr", "detector", "pose", "tracking", "motion", "siglip"}
)

DEFAULT_TRACE_SUBFOLDER = "diffusiongemma_grounding"
MAX_COMPACT_REPORT_BYTES = 256 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1


def _canonicalize_fact_categories(
    categories_value: Any,
) -> tuple[list[str] | None, tuple[tuple[str, str], ...]]:
    """Return canonical model-authored categories without accepting unknown values."""

    if not isinstance(categories_value, list) or not categories_value:
        return None, ()
    if any(not isinstance(category, str) for category in categories_value):
        return None, ()
    # Preserve the existing strict rule for duplicate model output. A collision
    # introduced only by an approved alias (for example appearance + wardrobe)
    # is safely collapsed below.
    if len(set(categories_value)) != len(categories_value):
        return None, ()

    canonical: list[str] = []
    canonicalizations: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category in categories_value:
        normalized = _FACT_CATEGORY_ALIASES.get(category, category)
        if normalized not in FACT_CATEGORIES:
            return None, ()
        if normalized != category:
            canonicalizations.append((category, normalized))
        if normalized not in seen:
            canonical.append(normalized)
            seen.add(normalized)
    return canonical, tuple(canonicalizations)

SAMPLING_PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "checkpoint_defaults": {
        "max_denoising_steps": 48,
        "temperature_start": 0.8,
        "temperature_end": 0.4,
        "entropy_bound": 0.1,
        "token_stability_threshold": 1,
        "confidence_threshold": 0.005,
        "adaptive_stopping": True,
    },
    "full_48_diagnostic": {
        "max_denoising_steps": 48,
        "temperature_start": 0.8,
        "temperature_end": 0.4,
        "entropy_bound": 0.1,
        "token_stability_threshold": 1,
        "confidence_threshold": 0.005,
        "adaptive_stopping": False,
    },
}

_FACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_CLAIM_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_ASSET_ID_RE = re.compile(r"^(?:image|picture|video):[1-9][0-9]*$")
_MANIFEST_LINE_RE = re.compile(
    r"<(?P<kind>Picture|Video)\s+(?P<number>[1-9][0-9]*)>\s*:\s*(?P<role>[^\r\n]+)",
    re.IGNORECASE,
)

_ROLE_SHARED_FACT_CATEGORIES = {
    "object",
    "count",
    "color",
    "text",
    "spatial",
    "other",
}


def _expand_role_category_family(categories: set[str]) -> set[str]:
    """Allow tightly coupled wording without weakening cross-role boundaries."""

    expanded = set(categories)
    for family in (
        {"identity", "appearance"},
        {"environment", "lighting", "composition"},
        {"action", "motion", "temporal"},
        {"camera", "composition", "motion", "temporal"},
    ):
        if categories & family:
            expanded.update(family)
    return expanded


_ROLE_CATEGORY_GROUPS: tuple[tuple[re.Pattern[str], set[str]], ...] = (
    (
        re.compile(
            r"\b(?:identity|appearance|face|body|silhouette|wardrobe|accessor(?:y|ies)|"
            r"distinguishing|subject|character|protagonist|likeness|outfit|costume|attire|"
            r"garment)\b",
            re.IGNORECASE,
        ),
        {"identity", "appearance"},
    ),
    (
        re.compile(
            r"\b(?:environment|setting|background|location|production\s+design|style|palette|"
            r"lighting|scenery|architecture|landscape)\b",
            re.IGNORECASE,
        ),
        {"environment", "lighting", "composition"},
    ),
    (
        re.compile(
            r"\b(?:action|motion|pose|movement|timing|choreograph\w*|performance|gesture|"
            r"blocking|temporal|whole[-\s]?video\s+edit\w*|edit\w*|continuation|"
            r"continue\w*|cut|cuts|rhythm)\b",
            re.IGNORECASE,
        ),
        {"action", "motion", "composition", "temporal"},
    ),
    (
        re.compile(r"\b(?:camera|framing|shot|composition|lens|angle)\b", re.IGNORECASE),
        {"camera", "composition", "motion", "temporal"},
    ),
    (
        re.compile(r"\b(?:object|product|item|prop|logo|packaging)\b", re.IGNORECASE),
        {"object"},
    ),
    (
        re.compile(r"\b(?:count|number|quantity|how\s+many)\b", re.IGNORECASE),
        {"count"},
    ),
    (
        re.compile(r"\b(?:color|colour|hue|palette)\b", re.IGNORECASE),
        {"color"},
    ),
    (
        re.compile(
            r"\b(?:text|title|copy|lettering|logo|typography|label|caption|wording)\b",
            re.IGNORECASE,
        ),
        {"text"},
    ),
    (
        re.compile(
            r"\b(?:spatial|layout|position|placement|arrangement|left[-\s]?right|"
            r"top[-\s]?bottom|geometry)\b",
            re.IGNORECASE,
        ),
        {"spatial"},
    ),
)
_ROLE_FORBIDS_IDENTITY_RE = re.compile(
    r"(?:"
    r"\b(?:exclude|ignore|omit)\b[^.;\r\n]{0,48}\bidentity\b"
    r"|"
    r"\b(?:do\s+not|don't|must\s+not|should\s+not|never)\b"
    r"[^.;\r\n]{0,32}\b(?:copy|use|derive|take|transfer|source)\w*\b"
    r"[^.;\r\n]{0,48}\bidentity\b"
    r"|"
    r"\b(?:without|no)\b[^.;\r\n]{0,32}\bidentity\b"
    r"[^.;\r\n]{0,32}\b(?:transfer|source|copy|override|replacement)\b"
    r"|"
    r"\bidentity\b[^.;\r\n]{0,96}"
    r"(?:\b(?:must\s+not|should\s+not|does\s+not|is\s+not|cannot|can't|won't|"
    r"mustn't|shouldn't)\b[^.;\r\n]{0,32}\b(?:transfer|override|replace|copy|"
    r"source)\w*\b|\bnot\s+a\s+source\b)"
    r")",
    re.IGNORECASE,
)
_EXPLICIT_ROLE_CATEGORIES_RE = re.compile(
    r"\[\s*dg\s*:\s*([a-z0-9_,\s-]+)\]",
    re.IGNORECASE,
)
_CLAIM_CATEGORY_GROUPS: tuple[tuple[re.Pattern[str], set[str]], ...] = (
    (
        re.compile(
            r"\b(?:identity|appearance|face|facial|body\s+type|hair|wardrobe|clothing|"
            r"beard(?:ed)?|distinguishing\s+trait|same\s+person|likeness)\b",
            re.IGNORECASE,
        ),
        {"identity", "appearance"},
    ),
    (
        re.compile(
            r"\b(?:environment|setting|background|location|town[-\s]?square|production\s+design|"
            r"style|palette|lighting|scenery|architecture|landscape|plaza|cobblestone|"
            r"arches?|room|street|forest|building|interior|exterior)\b",
            re.IGNORECASE,
        ),
        {"environment", "lighting", "composition"},
    ),
    (
        re.compile(
            r"\b(?:action|motion|moves?|moving|walk\w*|run\w*|movement|timing|"
            r"choreograph\w*|performance|gesture|blocking|temporal|turn\w*|jump\w*|"
            r"raise\w*|lower\w*|enter\w*|exit\w*|fall\w*|dance\w*|"
            r"reach\w*|grab\w*|edit\w*|continuation|continue\w*|cuts?|rhythm)\b",
            re.IGNORECASE,
        ),
        {"action", "motion", "temporal"},
    ),
    (
        re.compile(
            r"\b(?:camera|framing|shot|lens|pan\w*|zoom\w*|dolly|tilt\w*|orbit\w*|composition)\b",
            re.IGNORECASE,
        ),
        {"camera", "composition", "motion"},
    ),
    (
        re.compile(r"\b(?:object|product|item|prop|logo|packaging)\b", re.IGNORECASE),
        {"object"},
    ),
    (
        re.compile(
            r"\b(?:count|number|quantity|one|two|three|four|five|six|seven|eight|nine|ten)\b|\b\d+\b",
            re.IGNORECASE,
        ),
        {"count"},
    ),
    (
        re.compile(
            r"\b(?:color|colour|hue|red|orange|yellow|green|blue|purple|violet|pink|"
            r"brown|black|white|gr[ae]y|gold|silver|scarlet|crimson|teal|cyan|magenta|ochre)\b",
            re.IGNORECASE,
        ),
        {"color"},
    ),
    (
        re.compile(
            r"\b(?:text|title|word|words|lettering|logo|typography|reads?|says?|spells?|"
            r"label|caption|wording)\b",
            re.IGNORECASE,
        ),
        {"text"},
    ),
    (
        re.compile(
            r"\b(?:spatial|layout|left|right|above|below|behind|front|center(?:ed)?|centred|"
            r"adjacent|next\s+to|position|placement|arrangement)\b",
            re.IGNORECASE,
        ),
        {"spatial"},
    ),
)

_CONCRETE_APPAREL_PATTERN = (
    r"(?:coats?|jackets?|shirts?|dress(?:es)?(?!\s+(?:rehearsal|code)\b)|"
    r"outfits?|costumes?|attire|garments?|"
    r"accessor(?:y|ies))(?!-)"
)
_PERSON_ROLE_PATTERN = (
    r"(?:persons?|subjects?|characters?|protagonists?|actors?|performers?|couriers?|"
    r"models?|individuals?|m(?:a|e)n|wom(?:a|e)n|girls?|boys?|hero(?:es)?|villains?|"
    r"people)"
)
_PROPER_NAME_PATTERN = r"(?-i:[A-Z][a-z]{1,31})"
_PERSON_REFERENCE_PATTERN = (
    rf"(?:{_PERSON_ROLE_PATTERN}|he|she|they|{_PROPER_NAME_PATTERN})"
)
_CONCRETE_APPAREL_NOUN_RE = re.compile(
    rf"\b{_CONCRETE_APPAREL_PATTERN}\b",
    re.IGNORECASE,
)
_APPAREL_NON_MODIFIER_WORDS = (
    r"a|an|the|this|that|these|those|his|her|their|its|in|on|at|by|for|from|of|"
    r"to|with|without|within|into|onto|over|under|around|near|beside|behind|before|"
    r"after|across|through|past|as|while|and|or|but|which|who|whose|is|are|was|"
    r"were|be|been|being|has|have|had|do|does|did|will|would|can|could|should|"
    r"must|may|might|see|sees|show|shows|showing|hang|hangs|hanging|display|"
    r"displays|displaying|provide|provides|provided|supply|supplies|supplied|use|"
    r"uses|used|take|takes|took|define|defines|control|controls|determine|determines|"
    r"stand|stands|standing|turn|turns|walk|walks|flutter|flutters|remain|remains|"
    r"appear|appears|include|includes|contain|contains"
)
_APPAREL_MODIFIER_TOKEN_PATTERN = (
    rf"(?!(?:{_APPAREL_NON_MODIFIER_WORDS})\b)[a-z0-9]+(?:-[a-z0-9]+)*"
)
_APPAREL_MODIFIER_SEQUENCE_PATTERN = (
    rf"(?:{_APPAREL_MODIFIER_TOKEN_PATTERN}(?:\s+|,\s*)){{0,8}}"
)
_APPAREL_PHRASE_PATTERN = (
    rf"(?:(?:the|an?|this|that|these|those|his|her|their|its)\s+)?"
    rf"{_APPAREL_MODIFIER_SEQUENCE_PATTERN}{_CONCRETE_APPAREL_PATTERN}\b"
)
_APPAREL_WEARER_CUE_RE = re.compile(
    rf"(?:\b{_PERSON_REFERENCE_PATTERN}\b\s*,?\s+"
    rf"(?:(?:is|are|was|were|has\s+been|had\s+been|will\s+be)\s+)?"
    rf"(?:[a-z]+ly\s+){{0,2}}"
    rf"(?:wear|wears|wearing|wore|dressed\s+in|clad\s+in|sporting|dons|donning)\s+"
    rf"{_APPAREL_PHRASE_PATTERN}"
    rf"|\b{_PERSON_REFERENCE_PATTERN}\b\s*,?\s+in\s+{_APPAREL_PHRASE_PATTERN}"
    rf"|\b(?:wearing|dressed\s+in|clad\s+in|donning)\s+{_APPAREL_PHRASE_PATTERN}"
    rf"|\b(?:wear|wears|wore|dons)\s+{_APPAREL_PHRASE_PATTERN}"
    rf"|\bsporting\s+(?:the|an?)\s+"
    rf"{_APPAREL_MODIFIER_SEQUENCE_PATTERN}{_CONCRETE_APPAREL_PATTERN}\b"
    rf"|{_APPAREL_PHRASE_PATTERN}\s+"
    rf"(?:(?:is|are|was|were)\s+)?worn\s+by\s+"
    rf"(?:(?:the|an?)\s+)?{_PERSON_REFERENCE_PATTERN}\b"
    rf"|{_APPAREL_PHRASE_PATTERN}\s+on\s+"
    rf"(?:(?:the|an?)\s+)?{_PERSON_REFERENCE_PATTERN}\b)",
    re.IGNORECASE,
)
_PERSON_APPAREL_CUE_RE = re.compile(
    rf"(?:\b(?:his|her|their)\s+{_APPAREL_MODIFIER_SEQUENCE_PATTERN}"
    rf"{_CONCRETE_APPAREL_PATTERN}\b"
    rf"|\b{_PERSON_ROLE_PATTERN}['\u2019](?:s)?\s+"
    rf"{_APPAREL_MODIFIER_SEQUENCE_PATTERN}{_CONCRETE_APPAREL_PATTERN}\b)",
    re.IGNORECASE,
)
_PROPER_NAME_POSSESSIVE_APPAREL_RE = re.compile(
    rf"\b{_PROPER_NAME_PATTERN}['\u2019]s\s+"
    rf"{_APPAREL_MODIFIER_SEQUENCE_PATTERN}{_CONCRETE_APPAREL_PATTERN}\b"
)
_APPAREL_COMPOUND_WEARER_RE = re.compile(
    rf"(?:\b(?:[a-z0-9]+-)+(?:coated|jacketed|shirted|dressed|costumed)\s+"
    rf"{_PERSON_ROLE_PATTERN}\b"
    rf"|\b(?:coat|jacket|shirt|dress|outfit|costume|garment)-wearing\s+"
    rf"{_PERSON_ROLE_PATTERN}\b)",
    re.IGNORECASE,
)
_APPAREL_TRANSFER_CUE_RE = re.compile(
    r"(?:\b(?:cop(?:y|ies|ied|ying)|deriv(?:e|es|ed|ing)|"
    r"transfer(?:s|red|ring)?|sourc(?:e|es|ed|ing)|preserv(?:e|es|ed|ing)|"
    r"suppl(?:y|ies|ied|ying)|provid(?:e|es|ed|ing))\b\s+"
    rf"(?:directly\s+)?{_APPAREL_PHRASE_PATTERN}"
    rf"|\b(?:uses?|takes?)\s+{_APPAREL_PHRASE_PATTERN}"
    rf"|{_APPAREL_PHRASE_PATTERN}\s*(?:,|\u2014)?\s*"
    r"(?:(?:that\s+)?(?:(?:is|are|was|were)|(?:must|should|will|can)\s+be|"
    r"(?:has|have|had)\s+been)\s+)?(?:directly\s+)?"
    r"(?:copied|derived|transferred|sourced|preserved|supplied|provided)\b)",
    re.IGNORECASE,
)

_REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "apology_refusal",
        re.compile(
            r"\bi(?:\s+am|['\u2019]m)\s+sorry\s*,?\s+but\s+i\s+(?:cannot|can['\u2019]t|won['\u2019]t)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cannot_assist",
        re.compile(
            r"\bi\s+(?:cannot|can['\u2019]t|won['\u2019]t)\s+(?:"
            r"(?:assist|help|comply|provide|analy[sz]e|describe|process)\b|"
            r"(?:review|view)\s+(?:this|that|the)\s+(?:image|video|clip|media|content)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "unable_to_assist",
        re.compile(
            r"\bi(?:\s+am|['\u2019]m)\s+unable\s+to\s+(?:"
            r"(?:assist|help|comply|provide|analy[sz]e|describe|process)\b|"
            r"(?:review|view)\s+(?:this|that|the)\s+(?:image|video|clip|media|content)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "policy_refusal",
        re.compile(
            r"\b(?:this|that|your)\s+(?:request|content)\s+(?:violates|is\s+against|conflicts\s+with)\s+(?:my\s+)?(?:policy|policies|guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "must_decline",
        re.compile(
            r"\bi\s+(?:must|have\s+to)\s+(?:decline|refuse)\s+(?:this|that|the)\s+request\b",
            re.IGNORECASE,
        ),
    ),
    (
        "apology_cannot",
        re.compile(
            r"\bi\s+apologi[sz]e\s*,?\s+but\s+i\s+(?:cannot|can['\u2019]t|won['\u2019]t)\b",
            re.IGNORECASE,
        ),
    ),
)

_FORBIDDEN_TRACE_KEYS = frozenset(
    {
        "logits",
        "raw_logits",
        "logits_tensor",
        "pixel_values",
        "pixels",
        "image_tensor",
        "video_tensor",
        "video_frames",
        "raw_media",
        "image_bytes",
        "video_bytes",
        "media_bytes",
    }
)


def _is_safe_pixel_tensor_descriptor(path: str, key: str, value: Any) -> bool:
    """Allow processor transport facts while continuing to reject pixel payloads."""

    if not path.endswith(".pixel_tensors") or not key.startswith("pixel_values"):
        return False
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype"}:
        return False
    shape = value.get("shape")
    dtype = value.get("dtype")
    return bool(
        isinstance(shape, (list, tuple))
        and shape
        and len(shape) <= 8
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape)
        and all(item <= 2_147_483_647 for item in shape)
        and isinstance(dtype, str)
        and 0 < len(dtype) <= 128
    )


class GroundingGuardError(ValueError):
    """Base error for malformed Grounding Guard data."""


class GroundingLedgerError(GroundingGuardError):
    """Raised when a ledger cannot be extracted as strict JSON."""


@dataclass(frozen=True)
class GroundingGuardConfig:
    mode: str = "audit"
    retry_on_uncertain: bool = True
    sampling_profile: str = "checkpoint_defaults"
    seed: int = 0
    save_detailed_trace: bool = False
    trace_subfolder: str = DEFAULT_TRACE_SUBFOLDER
    external_evidence_json: str = ""
    evidence_token_budget: str = "auto"

    @property
    def sampling_settings(self) -> dict[str, Any]:
        return copy.deepcopy(SAMPLING_PROFILE_SETTINGS[self.sampling_profile])

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sampling_settings"] = self.sampling_settings
        return result


@dataclass(frozen=True)
class GroundingLedgerValidation:
    ledger: dict[str, Any] | None
    schema_valid: bool
    grounded: bool
    analysis_status: str
    model_analysis_status: str = "uncertain"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    covered_asset_ids: tuple[str, ...] = ()
    temporal_regions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    accepted_external_claim_count: int = 0
    external_conflicts: tuple[str, ...] = ()
    refusal_matches: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Compatibility alias for callers that treat verified grounding as valid."""

        return self.grounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "grounded": self.grounded,
            "analysis_status": self.analysis_status,
            "model_analysis_status": self.model_analysis_status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "blocking_reasons": list(self.blocking_reasons),
            "covered_asset_ids": list(self.covered_asset_ids),
            "temporal_regions": {
                key: list(value) for key, value in self.temporal_regions.items()
            },
            "accepted_external_claim_count": self.accepted_external_claim_count,
            "external_conflicts": list(self.external_conflicts),
            "refusal_matches": list(self.refusal_matches),
        }


@dataclass(frozen=True)
class GroundingGuardDecision:
    mode: str
    analysis_status: str
    decision: str
    would_block: bool
    blocked_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "analysis_status": self.analysis_status,
            "decision": self.decision,
            "would_block": self.would_block,
            "blocked_reasons": list(self.blocked_reasons),
            "warnings": list(self.warnings),
        }


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, GroundingGuardConfig):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _safe_trace_subfolder(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return DEFAULT_TRACE_SUBFOLDER
    pure = PurePath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return DEFAULT_TRACE_SUBFOLDER
    if re.match(r"^[A-Za-z]:", text):
        return DEFAULT_TRACE_SUBFOLDER
    return "/".join(pure.parts)


def normalize_grounding_guard_config(
    value: GroundingGuardConfig | Mapping[str, Any] | str | None = None,
    **overrides: Any,
) -> GroundingGuardConfig:
    """Return a safe, canonical settings value.

    Invalid externally supplied modes fail toward ``audit`` rather than silently
    disabling the guard. Seeds are clamped to ComfyUI's unsigned 64-bit range.
    """

    raw = _as_mapping(value)
    raw.update({key: item for key, item in overrides.items() if item is not None})

    mode = str(raw.get("mode", "audit")).strip().lower()
    if mode not in GROUNDING_MODES:
        mode = "audit"
    profile = str(raw.get("sampling_profile", "checkpoint_defaults")).strip().lower()
    if profile not in SAMPLING_PROFILES:
        profile = "checkpoint_defaults"
    evidence_token_budget = str(raw.get("evidence_token_budget", "auto")).strip().lower()
    if evidence_token_budget not in EVIDENCE_TOKEN_BUDGETS:
        evidence_token_budget = "auto"
    try:
        seed = int(raw.get("seed", 0))
    except (TypeError, ValueError, OverflowError):
        seed = 0
    seed = max(0, min(UINT64_MAX, seed))

    external = raw.get("external_evidence_json", "")
    if isinstance(external, (Mapping, list, tuple)):
        external_text = json.dumps(external, separators=(",", ":"), ensure_ascii=False)
    else:
        external_text = str(external or "").strip()

    return GroundingGuardConfig(
        mode=mode,
        retry_on_uncertain=_as_bool(raw.get("retry_on_uncertain"), True),
        sampling_profile=profile,
        seed=seed,
        save_detailed_trace=_as_bool(raw.get("save_detailed_trace"), False),
        trace_subfolder=_safe_trace_subfolder(raw.get("trace_subfolder")),
        external_evidence_json=external_text,
        evidence_token_budget=evidence_token_budget,
    )


def load_grounding_evidence_schema() -> dict[str, Any]:
    """Load the bundled strict ledger schema used in model instructions."""

    value = json.loads(GROUNDING_EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_identity = value.get("properties", {}).get("schema", {}).get("const") if isinstance(value, dict) else None
    if schema_identity != GROUNDING_LEDGER_SCHEMA_ID:
        raise GroundingGuardError("bundled grounding evidence schema has an unexpected identity")
    return value


def load_external_evidence_schema() -> dict[str, Any]:
    """Load the optional typed external-evidence envelope schema."""

    value = json.loads(EXTERNAL_EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_identity = (
        value.get("properties", {}).get("schema", {}).get("const")
        if isinstance(value, dict)
        else None
    )
    if schema_identity != EXTERNAL_EVIDENCE_SCHEMA_ID:
        raise GroundingGuardError("bundled external evidence schema has an unexpected identity")
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _manifest_roles(media_metadata: Mapping[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    manifest = str(media_metadata.get("minimax_h3_reference_manifest", "") or "")
    for match in _MANIFEST_LINE_RE.finditer(manifest):
        prefix = "picture" if match.group("kind").lower() == "picture" else "video"
        roles[f"{prefix}:{int(match.group('number'))}"] = match.group("role").strip()
    return roles


def _explicit_h3_role_categories(role: Any) -> set[str] | None:
    text = str(role or "").strip()
    match = _EXPLICIT_ROLE_CATEGORIES_RE.search(text)
    if not match:
        return None
    values = {
        item.strip().lower().replace("-", "_")
        for item in match.group(1).split(",")
        if item.strip()
    }
    if not values or "other" in values or not values.issubset(FACT_CATEGORIES):
        return set()
    return values


def _strip_h3_role_annotation(role: Any) -> str:
    return re.sub(r"\s*\[\s*dg\s*:[^\]]*\]\s*", " ", str(role or "")).strip()


def _h3_role_category_contract(role: Any) -> tuple[set[str], set[str]]:
    """Translate an H3 manifest role into a host-owned category contract.

    ``[dg:identity,appearance]`` annotations are authoritative. Untagged
    manifests retain conservative vocabulary inference for compatibility.
    """

    text = str(role or "").strip()
    if not text:
        return set(), set()
    explicit = _explicit_h3_role_categories(text)
    if explicit is not None:
        return (
            (_ROLE_SHARED_FACT_CATEGORIES | explicit, explicit)
            if explicit
            else (set(), set())
        )
    allowed = set(_ROLE_SHARED_FACT_CATEGORIES)
    role_specific: set[str] = set()
    matched = False
    forbids_identity = bool(_ROLE_FORBIDS_IDENTITY_RE.search(text))
    for index, (pattern, categories) in enumerate(_ROLE_CATEGORY_GROUPS):
        if index == 0 and forbids_identity:
            continue
        if pattern.search(text):
            allowed.update(categories)
            role_specific.update(categories)
            matched = True
    return (allowed, role_specific) if matched else (set(), set())


def _claim_role_categories(claim: Any) -> set[str]:
    categories: set[str] = set()
    text = str(claim or "")
    for pattern, matched_categories in _CLAIM_CATEGORY_GROUPS:
        if pattern.search(text):
            categories.update(matched_categories)
    if _APPAREL_COMPOUND_WEARER_RE.search(text) or (
        _CONCRETE_APPAREL_NOUN_RE.search(text)
        and (
            _APPAREL_WEARER_CUE_RE.search(text)
            or _PERSON_APPAREL_CUE_RE.search(text)
            or _PROPER_NAME_POSSESSIVE_APPAREL_RE.search(text)
            or _APPAREL_TRANSFER_CUE_RE.search(text)
        )
    ):
        categories.update({"identity", "appearance"})
    return categories


def _temporal_region(ordinal: int, sample_count: int) -> str:
    if sample_count <= 1:
        return "opening"
    if ordinal == 1:
        return "opening"
    if ordinal == sample_count:
        return "closing"
    position = (ordinal - 1) / max(1, sample_count - 1)
    if position < 1.0 / 3.0:
        return "opening"
    if position > 2.0 / 3.0:
        return "closing"
    return "middle"


def _video_samples(
    positions: Sequence[int], media_metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    indices_value = media_metadata.get("sampled_indices")
    indices = list(indices_value) if isinstance(indices_value, (list, tuple)) else []
    explicit_timecodes_value = media_metadata.get("sampled_timecodes_seconds")
    explicit_timecodes = (
        list(explicit_timecodes_value)
        if isinstance(explicit_timecodes_value, (list, tuple))
        else []
    )
    source_fps = _safe_float(media_metadata.get("source_fps"), 0.0)
    sample_fps = max(0.001, _safe_float(media_metadata.get("sample_fps"), 1.0))
    duration = max(0.0, _safe_float(media_metadata.get("duration_seconds"), 0.0))
    sample_count = len(positions)
    samples: list[dict[str, Any]] = []
    for offset, tensor_position in enumerate(positions):
        source_index = max(0, _safe_int(indices[offset], offset)) if offset < len(indices) else offset
        if offset < len(explicit_timecodes):
            timecode = max(0.0, _safe_float(explicit_timecodes[offset], 0.0))
        elif source_fps > 0:
            timecode = source_index / source_fps
        elif duration > 0 and sample_count > 1:
            timecode = duration * offset / (sample_count - 1)
        else:
            timecode = offset / sample_fps
        ordinal = offset + 1
        samples.append(
            {
                "sample_ordinal": ordinal,
                "tensor_batch_position": int(tensor_position),
                "source_frame_index": source_index,
                "timecode_seconds": round(timecode, 6),
                "temporal_region": _temporal_region(ordinal, sample_count),
            }
        )
    return samples


def _still_asset(
    asset_id: str,
    role: str,
    tensor_position: int,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "kind": "image",
        "role": role or "visual_reference",
        "required_for_grounding": True,
        "tensor_batch_positions": [int(tensor_position)],
        "samples": [
            {
                "sample_ordinal": 1,
                "tensor_batch_position": int(tensor_position),
                "source_frame_index": 0,
                "timecode_seconds": 0.0,
                "temporal_region": "opening",
            }
        ],
    }


def build_asset_registry(
    media_metadata: Mapping[str, Any] | None,
    image_count: int,
) -> dict[str, Any]:
    """Build stable host-owned IDs for the batched visual inputs.

    ``image_count`` is the number of image tensors that will be handed to the
    processor. A sampled video therefore contributes one image per sampled
    frame. The function never trusts model-produced IDs.
    """

    metadata = dict(media_metadata or {})
    count = max(0, _safe_int(image_count, 0))
    source = str(metadata.get("source", "none") or "none").strip().lower()
    synthesis_mode = str(
        metadata.get("media_synthesis_mode", "video_recreation") or "video_recreation"
    ).strip().lower()
    h3_ref = str(metadata.get("minimax_h3_mode", "") or "").strip().lower() == "ref2va"
    roles = _manifest_roles(metadata)
    warnings: list[str] = []
    assets: list[dict[str, Any]] = []

    if count == 0:
        declared_visual_asset_ids = sorted(roles)
        return {
            "schema": ASSET_REGISTRY_SCHEMA_ID,
            "source": source,
            "media_synthesis_mode": synthesis_mode,
            "minimax_h3_mode": "ref2va" if h3_ref else "",
            "expected_image_count": 0,
            "assigned_image_count": 0,
            "assets": [],
            "declared_visual_asset_ids": declared_visual_asset_ids,
            "unattached_declared_visual_asset_ids": declared_visual_asset_ids,
            "unexpected_attached_visual_asset_ids": [],
            "warnings": (
                ["declared_visual_assets_are_not_attached"]
                if declared_visual_asset_ids
                else []
            ),
        }

    if h3_ref:
        reference_count = _safe_int(
            metadata.get(
                "minimax_h3_reference_image_batch_count",
                metadata.get("reference_image_count", 0),
            ),
            0,
        )
        attachment_flag_present = "reference_image_backend_attached" in metadata
        reference_attached = (
            bool(metadata.get("reference_image_backend_attached"))
            if attachment_flag_present
            else source in {"image", "image+video"}
        )
        if reference_count and not reference_attached:
            warnings.append("unattached_reference_image_excluded_from_asset_registry")
            reference_count = 0
        reference_count = max(0, min(count, reference_count))
        if reference_count == 0 and source == "image":
            reference_count = count
        for index in range(reference_count):
            asset_id = f"picture:{index + 1}"
            assets.append(
                _still_asset(
                    asset_id,
                    roles.get(asset_id, f"H3 reference picture {index + 1}"),
                    index,
                )
            )
        remaining = list(range(reference_count, count))
        if remaining:
            video_asset = {
                "asset_id": "video:1",
                "kind": "video",
                "role": roles.get(
                    "video:1",
                    str(metadata.get("video_role", "H3 reference video")),
                ),
                "required_for_grounding": True,
                "tensor_batch_positions": remaining,
                "samples": _video_samples(remaining, metadata),
            }
            assets.append(video_asset)
        declared_picture_count = sum(key.startswith("picture:") for key in roles)
        if declared_picture_count and declared_picture_count != reference_count:
            warnings.append(
                "declared_picture_count_does_not_match_attached_reference_count"
            )
    elif source in {"video", "image+video"}:
        attachment_flag_present = "reference_image_backend_attached" in metadata
        reference_attached = (
            bool(metadata.get("reference_image_backend_attached"))
            if attachment_flag_present
            else source == "image+video"
        )
        reference_count = (
            max(0, _safe_int(metadata.get("reference_image_count"), 0))
            if source == "image+video" and reference_attached
            else 0
        )
        if source == "image+video" and reference_attached and reference_count == 0:
            reference_count = 1
        if metadata.get("reference_image_count") and not reference_attached:
            warnings.append("unattached_reference_image_excluded_from_asset_registry")
        reference_count = min(count, reference_count)
        for index in range(reference_count):
            role = str(metadata.get("image_role", "visual_reference"))
            assets.append(_still_asset(f"image:{index + 1}", role, index))
        remaining = list(range(reference_count, count))
        if remaining:
            assets.append(
                {
                    "asset_id": "video:1",
                    "kind": "video",
                    "role": str(metadata.get("video_role", "primary_video_source")),
                    "required_for_grounding": True,
                    "tensor_batch_positions": remaining,
                    "samples": _video_samples(remaining, metadata),
                }
            )
        elif source == "video":
            warnings.append("video_source_has_no_sampled_frames")
    else:
        declared_count = max(0, _safe_int(metadata.get("reference_image_count"), count))
        still_count = count if declared_count == 0 else min(count, declared_count)
        if still_count != count:
            warnings.append("unassigned_tensor_positions_treated_as_still_images")
            still_count = count
        per_image_roles = metadata.get("image_roles")
        if not isinstance(per_image_roles, list):
            per_image_roles = []
        for index in range(still_count):
            role = (
                str(per_image_roles[index])
                if index < len(per_image_roles) and str(per_image_roles[index]).strip()
                else str(metadata.get("image_role", "primary_image_source"))
            )
            assets.append(_still_asset(f"image:{index + 1}", role, index))

    for asset in assets:
        raw_role = str(asset.get("role", ""))
        _allowed_categories, role_categories = _h3_role_category_contract(
            raw_role
        )
        explicit_categories = _explicit_h3_role_categories(raw_role)
        asset["role"] = _strip_h3_role_annotation(raw_role)
        asset["grounding_role_categories"] = sorted(role_categories)
        asset["grounding_role_contract_source"] = (
            "explicit"
            if explicit_categories
            else "invalid_explicit"
            if explicit_categories is not None
            else "inferred"
            if role_categories
            else "unrecognized"
        )

    assigned_positions = sorted(
        position
        for asset in assets
        for position in asset.get("tensor_batch_positions", [])
    )
    if assigned_positions != list(range(count)):
        warnings.append("asset_registry_does_not_cover_tensor_batch")

    declared_visual_asset_ids = sorted(roles) if h3_ref else []
    attached_visual_asset_ids = sorted(
        str(asset.get("asset_id"))
        for asset in assets
        if str(asset.get("asset_id", "")).startswith(("picture:", "video:"))
    )
    unattached_declared = (
        sorted(set(declared_visual_asset_ids) - set(attached_visual_asset_ids))
        if h3_ref
        else []
    )
    unexpected_attached = (
        sorted(set(attached_visual_asset_ids) - set(declared_visual_asset_ids))
        if h3_ref
        else []
    )
    if h3_ref and (unattached_declared or unexpected_attached):
        warnings.append("declared_visual_assets_do_not_match_attached_registry")

    return {
        "schema": ASSET_REGISTRY_SCHEMA_ID,
        "source": source,
        "media_synthesis_mode": synthesis_mode,
        "minimax_h3_mode": "ref2va" if h3_ref else "",
        "expected_image_count": count,
        "assigned_image_count": len(assigned_positions),
        "assets": assets,
        "declared_visual_asset_ids": declared_visual_asset_ids,
        "unattached_declared_visual_asset_ids": unattached_declared,
        "unexpected_attached_visual_asset_ids": unexpected_attached,
        "warnings": list(dict.fromkeys(warnings)),
    }


class _DuplicateJSONKeyError(ValueError):
    pass


class _NonFiniteJSONNumberError(ValueError):
    pass


class _InvalidJSONNumberError(ValueError):
    pass


def _parse_bounded_json_decimal(value: str) -> Decimal:
    """Decode a JSON number without Python's huge-int or Decimal edge crashes."""

    if len(value) > 256:
        raise _InvalidJSONNumberError("JSON number exceeds 256 characters")
    try:
        number = Decimal(value)
        if not number.is_finite():
            raise _InvalidJSONNumberError("JSON number must be finite")
        exponent = number.as_tuple().exponent
        if not isinstance(exponent, int) or abs(exponent) > 4096:
            raise _InvalidJSONNumberError("JSON number exponent is out of range")
        return number
    except _InvalidJSONNumberError:
        raise
    except (DecimalException, ValueError) as exc:
        raise _InvalidJSONNumberError("JSON number is invalid or out of range") from exc


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise _NonFiniteJSONNumberError(f"non-finite JSON number: {value}")


def extract_grounding_ledger(
    value: str | Mapping[str, Any],
    *,
    allow_combined: bool = True,
) -> dict[str, Any]:
    """Extract a ledger from exact JSON or a combined packet.

    Markdown fences, leading commentary, trailing data, and JSON salvage are
    intentionally rejected. Combined Director packets may contain the ledger
    under the exact ``grounding_ledger`` key.
    """

    if isinstance(value, Mapping):
        try:
            parsed: Any = copy.deepcopy(dict(value))
        except RecursionError as exc:
            raise GroundingLedgerError("grounding ledger exceeds the nesting limit") from exc
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise GroundingLedgerError("grounding ledger is empty")
        decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_float=_parse_bounded_json_decimal,
            parse_int=_parse_bounded_json_decimal,
            parse_constant=_reject_nonfinite_json_number,
        )
        try:
            parsed, end = decoder.raw_decode(text)
        except _DuplicateJSONKeyError as exc:
            raise GroundingLedgerError(f"grounding ledger is not strict JSON: {exc}") from exc
        except _NonFiniteJSONNumberError as exc:
            raise GroundingLedgerError(f"grounding ledger is not strict JSON: {exc}") from exc
        except _InvalidJSONNumberError as exc:
            raise GroundingLedgerError(f"grounding ledger has an invalid JSON number: {exc}") from exc
        except RecursionError as exc:
            raise GroundingLedgerError("grounding ledger exceeds the JSON nesting limit") from exc
        except json.JSONDecodeError as exc:
            raise GroundingLedgerError(f"grounding ledger is not strict JSON: {exc.msg}") from exc
        if text[end:].strip():
            raise GroundingLedgerError("grounding ledger contains trailing non-JSON content")
    else:
        raise GroundingLedgerError("grounding ledger must be a JSON object or JSON string")

    if not isinstance(parsed, Mapping):
        raise GroundingLedgerError("grounding ledger root must be an object")
    if parsed.get("schema") == GROUNDING_LEDGER_SCHEMA_ID:
        try:
            return copy.deepcopy(dict(parsed))
        except RecursionError as exc:
            raise GroundingLedgerError("grounding ledger exceeds the nesting limit") from exc
    nested = parsed.get("grounding_ledger")
    if allow_combined and isinstance(nested, Mapping):
        try:
            return copy.deepcopy(dict(nested))
        except RecursionError as exc:
            raise GroundingLedgerError("grounding ledger exceeds the nesting limit") from exc
    if not allow_combined:
        raise GroundingLedgerError("strict grounding evidence root must be the ledger object")
    raise GroundingLedgerError("grounding ledger object is missing")


def detect_refusal_phrases(text: Any) -> list[str]:
    """Return complete refusal-clause labels, never isolated token matches."""

    haystack = str(text or "")
    candidates: list[tuple[int, int, str]] = []
    for label, pattern in _REFUSAL_PATTERNS:
        candidates.extend((match.start(), match.end(), label) for match in pattern.finditer(haystack))
    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        start, end, _label = candidate
        if any(start < other_end and other_start < end for other_start, other_end, _ in selected):
            continue
        selected.append(candidate)
    return [label for _start, _end, label in sorted(selected)]


def _append_error(errors: list[str], blockers: list[str], error: str, blocker: str) -> None:
    errors.append(error)
    blockers.append(blocker)


def _strict_text_list(
    value: Any,
    path: str,
    errors: list[str],
    blockers: list[str],
) -> None:
    if not isinstance(value, list):
        _append_error(errors, blockers, f"{path}:must_be_array", "schema_invalid")
        return
    if len(value) > 256:
        _append_error(errors, blockers, f"{path}:too_many_items", "schema_invalid")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip() or len(item) > 2048:
            _append_error(
                errors,
                blockers,
                f"{path}[{index}]:must_be_nonempty_string_at_most_2048_chars",
                "schema_invalid",
            )


def _decimal_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
        return number if number.is_finite() else None
    except (DecimalException, ValueError):
        return None


def _json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool)) or _decimal_number(value) is not None


def _schema_integer(
    value: Any,
    minimum: int | None = None,
    maximum: int = (1 << 63) - 1,
) -> int | None:
    number = _decimal_number(value)
    try:
        if number is None or number != number.to_integral_value():
            return None
        if number > maximum or (minimum is not None and number < minimum):
            return None
        return int(number)
    except (DecimalException, ValueError, OverflowError):
        return None


def _canonical_decimal(value: Decimal) -> str:
    sign, digits_tuple, exponent = value.as_tuple()
    digits = list(digits_tuple)
    if not digits or all(digit == 0 for digit in digits):
        return "0e0"
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"


def _json_native_number(value: Any) -> int | float | None:
    number = _decimal_number(value)
    if number is None:
        return None
    try:
        if number == number.to_integral_value():
            if number.adjusted() > 1024:
                return None
            return int(number)
        converted = float(number)
    except (DecimalException, ValueError, OverflowError):
        return None
    if not math.isfinite(converted) or Decimal(str(converted)) != number:
        return None
    return converted


def _typed_claim_key(
    claim: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, str, int | None]:
    asset_id = str(claim.get("asset_id", ""))
    ordinal = claim.get("sample_ordinal")
    normalized_ordinal = _schema_integer(ordinal, 1)
    asset = assets.get(asset_id) if assets is not None else None
    if asset is not None and len(_sample_index(asset)) == 1:
        normalized_ordinal = None
    return (
        asset_id,
        str(claim.get("claim_type", "")),
        normalized_ordinal,
    )


def _typed_value(value: Any) -> str:
    if isinstance(value, bool):
        value_type = "boolean"
        canonical_value: Any = value
    elif _decimal_number(value) is not None:
        value_type = "number"
        canonical_value = _canonical_decimal(_decimal_number(value))
    elif value is None:
        value_type = "null"
        canonical_value = None
    else:
        value_type = "string"
        canonical_value = str(value)
    return json.dumps(
        {"type": value_type, "value": canonical_value},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _decimal_to_json_native(value: Any) -> Any:
    if isinstance(value, Decimal):
        converted = _json_native_number(value)
        if converted is None:
            raise GroundingGuardError(
                "decimal value cannot be represented losslessly in the validated ledger"
            )
        return converted
    if isinstance(value, list):
        return [_decimal_to_json_native(item) for item in value]
    if isinstance(value, tuple):
        return [_decimal_to_json_native(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _decimal_to_json_native(item)
            for key, item in value.items()
        }
    return value


def _contains_unrepresentable_decimal(value: Any) -> bool:
    if not isinstance(value, bool) and isinstance(value, (int, float, Decimal)):
        return _json_native_number(value) is None
    if isinstance(value, Mapping):
        return any(_contains_unrepresentable_decimal(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unrepresentable_decimal(item) for item in value)
    return False


def parse_external_evidence(
    value: str | Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse recognized typed claims; unknown/unstructured inputs stay advisory."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return [], []
    if isinstance(value, Mapping):
        parsed: Any = dict(value)
    elif isinstance(value, str):
        try:
            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_float=_parse_bounded_json_decimal,
                parse_int=_parse_bounded_json_decimal,
                parse_constant=_reject_nonfinite_json_number,
            )
        except _DuplicateJSONKeyError:
            return [], ["external_evidence_duplicate_json_key"]
        except _NonFiniteJSONNumberError:
            return [], ["external_evidence_nonfinite_json_number"]
        except _InvalidJSONNumberError:
            return [], ["external_evidence_json_number_invalid_or_out_of_range"]
        except RecursionError:
            return [], ["external_evidence_json_nesting_limit"]
        except json.JSONDecodeError:
            return [], ["external_evidence_unstructured_or_invalid_json"]
    else:
        return [], ["external_evidence_unstructured"]
    if not isinstance(parsed, Mapping):
        return [], ["external_evidence_unstructured"]
    required_root = {"schema", "provider", "claims"}
    if set(parsed) != required_root:
        return [], ["external_evidence_envelope_invalid"]
    if parsed.get("schema") != EXTERNAL_EVIDENCE_SCHEMA_ID:
        return [], ["external_evidence_unknown_schema"]
    provider = parsed.get("provider")
    if not isinstance(provider, str) or provider not in KNOWN_EXTERNAL_PROVIDERS:
        return [], ["external_evidence_unknown_provider"]
    claims = parsed.get("claims")
    if not isinstance(claims, list) or len(claims) > 512:
        return [], ["external_evidence_claims_invalid"]

    valid: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            return [], [f"external_evidence_claim_{index}_invalid"]
        allowed = {
            "asset_id",
            "claim_type",
            "value",
            "sample_ordinal",
            "source_frame_index",
            "timecode_seconds",
            "confidence",
        }
        required = {"asset_id", "claim_type", "value"}
        if set(claim) - allowed:
            return [], [f"external_evidence_claim_{index}_has_unknown_fields"]
        if required - set(claim):
            return [], [f"external_evidence_claim_{index}_missing_required_fields"]
        asset_id = claim.get("asset_id")
        claim_type = claim.get("claim_type")
        if (
            not isinstance(asset_id, str)
            or not isinstance(claim_type, str)
            or not _ASSET_ID_RE.fullmatch(asset_id)
            or not _CLAIM_TYPE_RE.fullmatch(claim_type)
        ):
            return [], [f"external_evidence_claim_{index}_invalid_key"]
        claim_value = claim.get("value")
        if not _json_scalar(claim_value):
            return [], [f"external_evidence_claim_{index}_invalid_value"]
        if _decimal_number(claim_value) is not None and _schema_integer(
            claim_value,
            -(1 << 63),
        ) is None:
            return [], [f"external_evidence_claim_{index}_numeric_value_not_int64"]
        if asset_id.startswith("video:") and "sample_ordinal" not in claim:
            return [], [f"external_evidence_claim_{index}_video_sample_ordinal_required"]
        ordinal = claim.get("sample_ordinal")
        normalized_ordinal = _schema_integer(ordinal, 1)
        if "sample_ordinal" in claim and normalized_ordinal is None:
            return [], [f"external_evidence_claim_{index}_invalid_sample"]
        confidence = claim.get("confidence")
        if "confidence" in claim and confidence not in CONFIDENCE_LEVELS:
            return [], [f"external_evidence_claim_{index}_invalid_confidence"]
        frame = claim.get("source_frame_index")
        if "sample_ordinal" not in claim and (
            "source_frame_index" in claim or "timecode_seconds" in claim
        ):
            return [], [f"external_evidence_claim_{index}_frame_or_time_requires_sample"]
        normalized_frame = _schema_integer(frame, 0)
        if "source_frame_index" in claim and normalized_frame is None:
            return [], [f"external_evidence_claim_{index}_invalid_frame"]
        timecode = claim.get("timecode_seconds")
        if "timecode_seconds" in claim and (
            _decimal_number(timecode) is None
            or _decimal_number(timecode) < 0
        ):
            return [], [f"external_evidence_claim_{index}_invalid_timecode"]
        normalized_claim = copy.deepcopy(dict(claim))
        if "sample_ordinal" in normalized_claim:
            normalized_claim["sample_ordinal"] = normalized_ordinal
        if "source_frame_index" in normalized_claim:
            normalized_claim["source_frame_index"] = normalized_frame
        valid.append(normalized_claim)
    return valid, []


def _asset_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    assets = registry.get("assets")
    if not isinstance(assets, list):
        return {}
    return {
        str(asset.get("asset_id")): dict(asset)
        for asset in assets
        if isinstance(asset, Mapping) and asset.get("asset_id")
    }


def _sample_index(asset: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    samples = asset.get("samples")
    if not isinstance(samples, list):
        return {}
    return {
        int(sample.get("sample_ordinal")): dict(sample)
        for sample in samples
        if isinstance(sample, Mapping)
        and isinstance(sample.get("sample_ordinal"), int)
        and not isinstance(sample.get("sample_ordinal"), bool)
    }


def _validate_evidence_reference(
    reference: Any,
    path: str,
    assets: Mapping[str, dict[str, Any]],
    errors: list[str],
    blockers: list[str],
) -> tuple[str | None, str | None]:
    if not isinstance(reference, Mapping):
        _append_error(errors, blockers, f"{path}:must_be_object", "schema_invalid")
        return None, None
    allowed = {"asset_id", "sample_ordinal", "source_frame_index", "timecode_seconds"}
    if set(reference) - allowed:
        _append_error(errors, blockers, f"{path}:unknown_fields", "schema_invalid")
    asset_id = reference.get("asset_id")
    if not isinstance(asset_id, str) or not _ASSET_ID_RE.fullmatch(asset_id):
        _append_error(errors, blockers, f"{path}.asset_id:invalid", "schema_invalid")
        return None, None
    video_missing_fields: set[str] = set()
    if asset_id.startswith("video:"):
        required = {"sample_ordinal", "source_frame_index", "timecode_seconds"}
        video_missing_fields = required - set(reference)
        if video_missing_fields:
            _append_error(
                errors,
                blockers,
                f"{path}:video_reference_missing:{','.join(sorted(video_missing_fields))}",
                "schema_invalid",
            )
            blockers.append("invalid_frame_reference")
    asset = assets.get(asset_id)
    if asset is None:
        _append_error(errors, blockers, f"{path}.asset_id:unknown_asset:{asset_id}", "unknown_asset")
        return asset_id, None

    samples = _sample_index(asset)
    ordinal = reference.get("sample_ordinal")
    if video_missing_fields:
        return asset_id, None
    if "sample_ordinal" in reference and ordinal is None:
        _append_error(errors, blockers, f"{path}.sample_ordinal:null_not_allowed", "schema_invalid")
        return asset_id, None
    if ordinal is None:
        if "source_frame_index" in reference or "timecode_seconds" in reference:
            _append_error(
                errors,
                blockers,
                f"{path}:frame_or_time_requires_sample_ordinal",
                "schema_invalid",
            )
            blockers.append("invalid_frame_reference")
        return asset_id, None
    normalized_ordinal = _schema_integer(ordinal, 1)
    if normalized_ordinal is None:
        _append_error(errors, blockers, f"{path}.sample_ordinal:invalid", "schema_invalid")
        return asset_id, None
    expected = samples.get(normalized_ordinal)
    if expected is None:
        _append_error(
            errors,
            blockers,
            f"{path}.sample_ordinal:out_of_bounds:{normalized_ordinal}",
            "invalid_frame_reference",
        )
        return asset_id, None
    if "source_frame_index" in reference:
        frame = reference.get("source_frame_index")
        normalized_frame = _schema_integer(frame, 0)
        if normalized_frame is None:
            _append_error(
                errors,
                blockers,
                f"{path}.source_frame_index:invalid",
                "schema_invalid",
            )
        elif normalized_frame != expected.get("source_frame_index"):
            _append_error(
                errors,
                blockers,
                f"{path}.source_frame_index:does_not_match_registry",
                "invalid_frame_reference",
            )
    if "timecode_seconds" in reference:
        timecode = reference.get("timecode_seconds")
        normalized_timecode = _decimal_number(timecode)
        if (
            normalized_timecode is None
            or normalized_timecode < 0
        ):
            _append_error(errors, blockers, f"{path}.timecode_seconds:invalid", "schema_invalid")
        elif abs(
            normalized_timecode
            - (Decimal(str(expected.get("timecode_seconds", 0.0))))
        ) > Decimal("0.075"):
            _append_error(
                errors,
                blockers,
                f"{path}.timecode_seconds:does_not_match_registry",
                "invalid_frame_reference",
            )
    return asset_id, str(expected.get("temporal_region", "")) or None


def _validate_typed_claim(
    claim: Any,
    path: str,
    assets: Mapping[str, dict[str, Any]],
    errors: list[str],
    blockers: list[str],
) -> dict[str, Any] | None:
    initial_error_count = len(errors)
    if not isinstance(claim, Mapping):
        _append_error(errors, blockers, f"{path}:must_be_object", "schema_invalid")
        return None
    allowed = {"asset_id", "claim_type", "value", "sample_ordinal"}
    if set(claim) - allowed:
        _append_error(errors, blockers, f"{path}:unknown_fields", "schema_invalid")
    if not {"asset_id", "claim_type", "value"}.issubset(claim):
        _append_error(errors, blockers, f"{path}:missing_required_fields", "schema_invalid")
        return None
    asset_id = claim.get("asset_id")
    claim_type = claim.get("claim_type")
    if not isinstance(asset_id, str) or not _ASSET_ID_RE.fullmatch(asset_id):
        _append_error(errors, blockers, f"{path}.asset_id:invalid", "schema_invalid")
    elif asset_id not in assets:
        _append_error(errors, blockers, f"{path}.asset_id:unknown_asset", "unknown_asset")
    if not isinstance(claim_type, str) or not _CLAIM_TYPE_RE.fullmatch(claim_type):
        _append_error(errors, blockers, f"{path}.claim_type:invalid", "schema_invalid")
    claim_value = claim.get("value")
    if not _json_scalar(claim_value):
        _append_error(errors, blockers, f"{path}.value:must_be_json_scalar", "schema_invalid")
    elif _decimal_number(claim_value) is not None and _schema_integer(
        claim_value,
        -(1 << 63),
    ) is None:
        _append_error(
            errors,
            blockers,
            f"{path}.value:numeric_value_must_be_signed_64_bit_integer",
            "schema_invalid",
        )
    ordinal = claim.get("sample_ordinal")
    if isinstance(asset_id, str) and asset_id.startswith("video:") and "sample_ordinal" not in claim:
        _append_error(
            errors,
            blockers,
            f"{path}.sample_ordinal:required_for_video_claim",
            "schema_invalid",
        )
    if "sample_ordinal" in claim and ordinal is None:
        _append_error(errors, blockers, f"{path}.sample_ordinal:null_not_allowed", "schema_invalid")
    elif ordinal is not None and _schema_integer(ordinal, 1) is None:
        _append_error(errors, blockers, f"{path}.sample_ordinal:invalid", "schema_invalid")
    elif ordinal is not None and isinstance(asset_id, str) and asset_id in assets:
        normalized_ordinal = _schema_integer(ordinal, 1)
        if normalized_ordinal not in _sample_index(assets[asset_id]):
            _append_error(
                errors,
                blockers,
                f"{path}.sample_ordinal:out_of_bounds:{normalized_ordinal}",
                "invalid_frame_reference",
            )
    if len(errors) != initial_error_count:
        return None
    return copy.deepcopy(dict(claim))


def validate_grounding_ledger(
    ledger: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    external_evidence_json: str | Mapping[str, Any] | None = None,
    *,
    final_text: str = "",
) -> GroundingLedgerValidation:
    """Strictly validate a model ledger against host-created assets."""

    candidate = copy.deepcopy(dict(ledger)) if isinstance(ledger, Mapping) else None
    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    covered_assets: set[str] = set()
    temporal_regions: dict[str, set[str]] = {}
    typed_claims: list[dict[str, Any]] = []

    if candidate is None:
        return GroundingLedgerValidation(
            ledger=None,
            schema_valid=False,
            grounded=False,
            analysis_status="uncertain",
            errors=("ledger:must_be_object",),
            blocking_reasons=("schema_invalid",),
        )

    required_top = {
        "schema",
        "analysis_status",
        "observed_facts",
        "inferred_facts",
        "creative_additions",
        "uncertainties",
        "grounding_failure_reasons",
    }
    allowed_top = required_top
    missing_top = required_top - set(candidate)
    unknown_top = set(candidate) - allowed_top
    if missing_top:
        _append_error(
            errors,
            blockers,
            f"ledger:missing_fields:{','.join(sorted(missing_top))}",
            "schema_invalid",
        )
    if unknown_top:
        _append_error(
            errors,
            blockers,
            f"ledger:unknown_fields:{','.join(sorted(unknown_top))}",
            "schema_invalid",
        )
    if candidate.get("schema") != GROUNDING_LEDGER_SCHEMA_ID:
        _append_error(errors, blockers, "ledger.schema:unsupported", "schema_invalid")
    reported_model_status = candidate.get("analysis_status")
    model_status = reported_model_status
    if model_status not in LEDGER_ANALYSIS_STATUSES:
        _append_error(errors, blockers, "ledger.analysis_status:invalid", "schema_invalid")
        model_status = "uncertain"

    for key in (
        "inferred_facts",
        "creative_additions",
        "uncertainties",
        "grounding_failure_reasons",
    ):
        _strict_text_list(candidate.get(key), f"ledger.{key}", errors, blockers)

    assets = _asset_index(asset_registry)
    facts = candidate.get("observed_facts")
    if not isinstance(facts, list):
        _append_error(errors, blockers, "ledger.observed_facts:must_be_array", "schema_invalid")
        facts = []
    elif len(facts) > 256:
        _append_error(errors, blockers, "ledger.observed_facts:too_many_items", "schema_invalid")

    seen_fact_ids: set[str] = set()
    high_medium_by_asset: dict[str, list[tuple[set[str], set[str]]]] = {}
    categories_by_asset: dict[str, set[str]] = {}
    claims_by_asset: dict[str, list[tuple[str, set[str]]]] = {}
    for fact_index, fact in enumerate(facts):
        path = f"ledger.observed_facts[{fact_index}]"
        if not isinstance(fact, Mapping):
            _append_error(errors, blockers, f"{path}:must_be_object", "schema_invalid")
            continue
        # Validation accepts generic Mapping inputs, but canonicalization below
        # must never assume that a caller-owned/custom mapping is writable.
        fact = dict(fact)
        facts[fact_index] = fact
        required_fact = {"fact_id", "claim", "confidence", "categories", "evidence"}
        allowed_fact = required_fact | {"typed_claims"}
        if set(fact) - allowed_fact:
            _append_error(errors, blockers, f"{path}:unknown_fields", "schema_invalid")
        if required_fact - set(fact):
            _append_error(errors, blockers, f"{path}:missing_required_fields", "schema_invalid")
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not _FACT_ID_RE.fullmatch(fact_id):
            _append_error(errors, blockers, f"{path}.fact_id:invalid", "schema_invalid")
        elif fact_id in seen_fact_ids:
            _append_error(errors, blockers, f"{path}.fact_id:duplicate", "schema_invalid")
        else:
            seen_fact_ids.add(fact_id)
        claim = fact.get("claim")
        if not isinstance(claim, str) or not claim.strip() or len(claim) > 2048:
            _append_error(errors, blockers, f"{path}.claim:invalid", "schema_invalid")
        confidence = fact.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            _append_error(errors, blockers, f"{path}.confidence:invalid", "schema_invalid")
        categories_value = fact.get("categories")
        canonical_categories, category_canonicalizations = _canonicalize_fact_categories(
            categories_value
        )
        if canonical_categories is None:
            _append_error(errors, blockers, f"{path}.categories:invalid", "schema_invalid")
            categories: set[str] = set()
        else:
            categories = set(canonical_categories)
            if category_canonicalizations:
                fact["categories"] = canonical_categories
                for source_category, target_category in category_canonicalizations:
                    warnings.append(
                        f"{path}.categories:canonicalized:"
                        f"{source_category}->{target_category}"
                    )

        evidence_value = fact.get("evidence")
        if not isinstance(evidence_value, list) or not evidence_value or len(evidence_value) > 64:
            _append_error(errors, blockers, f"{path}.evidence:invalid", "schema_invalid")
            evidence_value = []
        fact_assets: set[str] = set()
        fact_regions_by_asset: dict[str, set[str]] = {}
        fact_evidence_ordinals: dict[str, set[int | None]] = {}
        for evidence_index, evidence in enumerate(evidence_value):
            evidence_error_count = len(errors)
            asset_id, region = _validate_evidence_reference(
                evidence,
                f"{path}.evidence[{evidence_index}]",
                assets,
                errors,
                blockers,
            )
            # The host registry is authoritative. Once a sample reference has
            # matched, publish its canonical integer/frame/time values instead
            # of preserving model-authored numeric spellings or precision.
            if (
                len(errors) == evidence_error_count
                and asset_id in assets
                and isinstance(evidence, Mapping)
            ):
                normalized_ordinal = _schema_integer(evidence.get("sample_ordinal"), 1)
                expected_sample = _sample_index(assets[asset_id]).get(normalized_ordinal)
                if expected_sample is not None:
                    normalized_evidence = dict(evidence)
                    normalized_evidence["sample_ordinal"] = normalized_ordinal
                    if "source_frame_index" in normalized_evidence:
                        normalized_evidence["source_frame_index"] = expected_sample.get(
                            "source_frame_index"
                        )
                    if "timecode_seconds" in normalized_evidence:
                        normalized_evidence["timecode_seconds"] = expected_sample.get(
                            "timecode_seconds"
                        )
                    evidence_value[evidence_index] = normalized_evidence
                    evidence = normalized_evidence
            if asset_id in assets:
                fact_assets.add(asset_id)
                if isinstance(evidence, Mapping):
                    ordinal = evidence.get("sample_ordinal")
                    normalized_ordinal = _schema_integer(ordinal, 1)
                    if normalized_ordinal in _sample_index(assets[asset_id]):
                        fact_evidence_ordinals.setdefault(asset_id, set()).add(
                            normalized_ordinal
                        )
                    elif (
                        ordinal is None
                        and assets[asset_id].get("kind") != "video"
                        and "source_frame_index" not in evidence
                        and "timecode_seconds" not in evidence
                    ):
                        fact_evidence_ordinals.setdefault(asset_id, set()).add(None)
                if region:
                    fact_regions_by_asset.setdefault(asset_id, set()).add(region)
        for asset_id in fact_assets:
            categories_by_asset.setdefault(asset_id, set()).update(categories)
            claims_by_asset.setdefault(asset_id, []).append(
                (str(claim or ""), set(categories))
            )
        if isinstance(confidence, str) and confidence in ("high", "medium"):
            covered_assets.update(fact_assets)
            for asset_id in fact_assets:
                regions = fact_regions_by_asset.get(asset_id, set())
                if regions:
                    temporal_regions.setdefault(asset_id, set()).update(regions)
                high_medium_by_asset.setdefault(asset_id, []).append((categories, regions))

        fact_typed = fact.get("typed_claims", [])
        if not isinstance(fact_typed, list) or len(fact_typed) > 64:
            _append_error(errors, blockers, f"{path}.typed_claims:invalid", "schema_invalid")
        else:
            for claim_index, typed_claim in enumerate(fact_typed):
                validated = _validate_typed_claim(
                    typed_claim,
                    f"{path}.typed_claims[{claim_index}]",
                    assets,
                    errors,
                    blockers,
                )
                if validated is not None:
                    typed_asset_id = str(validated.get("asset_id", ""))
                    typed_ordinal = _typed_claim_key(validated, assets)[2]
                    bound_ordinals = fact_evidence_ordinals.get(typed_asset_id, set())
                    if typed_asset_id not in fact_assets or (
                        typed_ordinal is not None and typed_ordinal not in bound_ordinals
                    ):
                        _append_error(
                            errors,
                            blockers,
                            f"{path}.typed_claims[{claim_index}]:not_bound_to_fact_evidence",
                            "schema_invalid",
                        )
                    else:
                        typed_claims.append(validated)

    typed_values_by_key: dict[tuple[str, str, int | None], set[str]] = {}
    for typed_claim in typed_claims:
        typed_values_by_key.setdefault(_typed_claim_key(typed_claim, assets), set()).add(
            _typed_value(typed_claim.get("value"))
        )
    internally_conflicting_typed_keys = sorted(
        (key for key, values in typed_values_by_key.items() if len(values) > 1),
        key=lambda item: (
            item[0],
            item[1],
            -1 if item[2] is None else item[2],
        ),
    )
    if internally_conflicting_typed_keys:
        rendered = [
            f"{asset_id}:{claim_type}:{ordinal if ordinal is not None else 'asset'}"
            for asset_id, claim_type, ordinal in internally_conflicting_typed_keys
        ]
        _append_error(
            errors,
            blockers,
            "ledger.typed_claims:internal_conflict:" + ",".join(rendered),
            "schema_invalid",
        )

    if model_status == "grounded" and not facts:
        _append_error(errors, blockers, "ledger.observed_facts:empty", "insufficient_coverage")

    required_assets = {
        asset_id
        for asset_id, asset in assets.items()
        if asset.get("required_for_grounding", True)
    }
    missing_coverage = sorted(required_assets - covered_assets)
    if model_status == "grounded" and missing_coverage:
        _append_error(
            errors,
            blockers,
            "ledger.observed_facts:missing_high_or_medium_coverage:"
            + ",".join(missing_coverage),
            "insufficient_coverage",
        )

    for asset_id, asset in assets.items():
        if asset.get("kind") != "video" or len(asset.get("samples", [])) <= 1:
            continue
        regions = temporal_regions.get(asset_id, set())
        if model_status == "grounded" and len(regions) < 2:
            _append_error(
                errors,
                blockers,
                f"ledger.observed_facts:insufficient_temporal_regions:{asset_id}",
                "insufficient_coverage",
            )

    if str(asset_registry.get("media_synthesis_mode", "")) == "image_identity_video_control":
        identity_assets = {
            asset_id
            for asset_id, asset in assets.items()
            if asset.get("kind") == "image" and "identity" in str(asset.get("role", "")).lower()
        }
        video_assets = {
            asset_id for asset_id, asset in assets.items() if asset.get("kind") == "video"
        }
        identity_ok = any(
            categories & {"identity", "appearance"}
            for asset_id in identity_assets
            for categories, _regions in high_medium_by_asset.get(asset_id, [])
        )
        action_or_camera_ok = any(
            categories & {"action", "motion", "temporal", "camera", "composition"}
            for asset_id in video_assets
            for categories, _regions in high_medium_by_asset.get(asset_id, [])
        )
        if model_status == "grounded" and not identity_ok:
            _append_error(errors, blockers, "ledger:missing_identity_image_evidence", "insufficient_coverage")
        if model_status == "grounded" and not action_or_camera_ok:
            _append_error(
                errors,
                blockers,
                "ledger:missing_video_action_or_camera_evidence",
                "insufficient_coverage",
            )

        for asset_id in identity_assets:
            if categories_by_asset.get(asset_id, set()) & {
                "action",
                "motion",
                "temporal",
                "camera",
            }:
                _append_error(errors, blockers, f"ledger:role_cross_wired:{asset_id}", "role_cross_wired")
        for asset_id in video_assets:
            if categories_by_asset.get(asset_id, set()) & {"identity", "appearance"}:
                _append_error(errors, blockers, f"ledger:role_cross_wired:{asset_id}", "role_cross_wired")

    # H3 reference manifests assign each Picture/Video an explicit role.  The
    # role comes from the host-created registry, not from model output.  Every
    # role-bound fact must declare a role-specific category, and claim text is
    # independently screened so relabeling a swapped claim as ``other`` cannot
    # bypass the boundary.
    if str(asset_registry.get("minimax_h3_mode", "")) == "ref2va":
        declared_visual_assets = {
            str(asset_id)
            for asset_id in asset_registry.get("declared_visual_asset_ids", [])
            if isinstance(asset_id, str)
        }
        attached_visual_assets = {
            asset_id
            for asset_id in assets
            if asset_id.startswith(("picture:", "video:"))
        }
        if declared_visual_assets != attached_visual_assets:
            _append_error(
                errors,
                blockers,
                "ledger:h3_declared_attached_asset_mismatch:"
                f"missing={','.join(sorted(declared_visual_assets - attached_visual_assets)) or 'none'};"
                f"unexpected={','.join(sorted(attached_visual_assets - declared_visual_assets)) or 'none'}",
                "insufficient_coverage",
            )
        for asset_id, asset in assets.items():
            if not asset_id.startswith(("picture:", "video:")):
                continue
            if asset.get("grounding_role_contract_source") != "explicit":
                _append_error(
                    errors,
                    blockers,
                    f"ledger:h3_role_contract_not_explicit:{asset_id}",
                    "insufficient_coverage",
                )
                continue
            stored_role_categories = asset.get("grounding_role_categories")
            if isinstance(stored_role_categories, list):
                role_specific = {
                    str(category)
                    for category in stored_role_categories
                    if category in FACT_CATEGORIES and category != "other"
                }
                allowed_categories = (
                    _ROLE_SHARED_FACT_CATEGORIES
                    | _expand_role_category_family(role_specific)
                )
            else:
                allowed_categories, role_specific = _h3_role_category_contract(
                    asset.get("role")
                )
            if not role_specific:
                _append_error(
                    errors,
                    blockers,
                    f"ledger:h3_role_contract_unrecognized:{asset_id}",
                    "insufficient_coverage",
                )
                continue
            for claim_text, declared_categories in claims_by_asset.get(asset_id, []):
                semantic_categories = _claim_role_categories(claim_text)
                effective_categories = declared_categories | semantic_categories
                disallowed = effective_categories - allowed_categories
                if disallowed:
                    _append_error(
                        errors,
                        blockers,
                        f"ledger:role_cross_wired:{asset_id}:{','.join(sorted(disallowed))}",
                        "role_cross_wired",
                    )
                if not declared_categories & role_specific:
                    _append_error(
                        errors,
                        blockers,
                        f"ledger:role_specific_category_missing:{asset_id}",
                        "role_cross_wired",
                    )

    external_claims, external_warnings = parse_external_evidence(external_evidence_json)
    warnings.extend(external_warnings)
    accepted_external_claims: list[dict[str, Any]] = []
    external_by_key: dict[tuple[str, str, int | None], set[str]] = {}
    external_registry_valid = True
    for claim_index, claim in enumerate(external_claims):
        asset = assets.get(str(claim.get("asset_id", "")))
        if asset is None:
            warnings.append(f"external_evidence_unknown_asset:{claim.get('asset_id')}")
            external_registry_valid = False
            continue
        ordinal = claim.get("sample_ordinal")
        if str(claim.get("asset_id", "")).startswith("video:") and ordinal is None:
            warnings.append(
                f"external_evidence_claim_{claim_index}_video_sample_ordinal_required"
            )
            external_registry_valid = False
            continue
        if ordinal is not None:
            expected_sample = _sample_index(asset).get(int(ordinal))
            if expected_sample is None:
                warnings.append(f"external_evidence_claim_{claim_index}_sample_out_of_bounds")
                external_registry_valid = False
                continue
            if (
                "source_frame_index" in claim
                and claim.get("source_frame_index") != expected_sample.get("source_frame_index")
            ):
                warnings.append(f"external_evidence_claim_{claim_index}_frame_mismatch")
                external_registry_valid = False
                continue
            if "timecode_seconds" in claim and abs(
                _decimal_number(claim.get("timecode_seconds"))
                - Decimal(str(expected_sample.get("timecode_seconds", 0.0)))
            ) > Decimal("0.075"):
                warnings.append(f"external_evidence_claim_{claim_index}_timecode_mismatch")
                external_registry_valid = False
                continue
        elif "source_frame_index" in claim or "timecode_seconds" in claim:
            warnings.append(f"external_evidence_claim_{claim_index}_missing_sample_ordinal")
            external_registry_valid = False
            continue
        accepted_external_claims.append(claim)
        external_by_key.setdefault(_typed_claim_key(claim, assets), set()).add(
            _typed_value(claim.get("value"))
        )
    if not external_registry_valid:
        accepted_external_claims.clear()
        external_by_key.clear()
    conflicts: list[str] = []
    for key, values in external_by_key.items():
        if len(values) > 1:
            conflicts.append(
                f"{key[0]}:{key[1]}:{key[2] if key[2] is not None else 'asset'}"
            )
    for claim in typed_claims:
        key = _typed_claim_key(claim, assets)
        expected_values = external_by_key.get(key)
        if expected_values and _typed_value(claim.get("value")) not in expected_values:
            rendered_key = f"{key[0]}:{key[1]}:{key[2] if key[2] is not None else 'asset'}"
            conflicts.append(rendered_key)
    if conflicts:
        _append_error(
            errors,
            blockers,
            "ledger.typed_claims:external_conflict:" + ",".join(sorted(set(conflicts))),
            "external_typed_conflict",
        )

    failure_reasons_for_refusal = candidate.get("grounding_failure_reasons", [])
    if not isinstance(failure_reasons_for_refusal, (list, tuple)):
        failure_reasons_for_refusal = ()
    text_for_refusal = "\n".join(
        [
            final_text,
            *(str(fact.get("claim", "")) for fact in facts if isinstance(fact, Mapping)),
            *(
                str(item)
                for item in failure_reasons_for_refusal
                if isinstance(item, str)
            ),
        ]
    )
    refusal_matches = detect_refusal_phrases(text_for_refusal)
    if refusal_matches:
        blockers.append("refusal_detected")

    candidate_has_unrepresentable_decimal = _contains_unrepresentable_decimal(candidate)
    if candidate_has_unrepresentable_decimal:
        _append_error(
            errors,
            blockers,
            "ledger:numeric_value_not_losslessly_representable",
            "schema_invalid",
        )

    unique_errors = tuple(dict.fromkeys(errors))
    unique_warnings = tuple(dict.fromkeys(warnings))
    unique_blockers = tuple(dict.fromkeys(blockers))
    schema_valid = not any(reason == "schema_invalid" for reason in unique_blockers)
    if model_status == "transport_error":
        host_status = "transport_error"
    elif model_status == "refused" or refusal_matches:
        host_status = "refused"
    elif model_status == "grounded" and not unique_blockers:
        host_status = "grounded"
    else:
        host_status = "uncertain"
    candidate["analysis_status"] = host_status
    validated_candidate = (
        None
        if candidate_has_unrepresentable_decimal
        else _decimal_to_json_native(candidate)
    )

    return GroundingLedgerValidation(
        ledger=validated_candidate,
        schema_valid=schema_valid,
        grounded=host_status == "grounded",
        analysis_status=host_status,
        model_analysis_status=str(reported_model_status or "uncertain"),
        errors=unique_errors,
        warnings=unique_warnings,
        blocking_reasons=unique_blockers,
        covered_asset_ids=tuple(sorted(covered_assets)),
        temporal_regions={
            asset_id: tuple(sorted(regions))
            for asset_id, regions in sorted(temporal_regions.items())
        },
        accepted_external_claim_count=len(accepted_external_claims),
        external_conflicts=tuple(sorted(set(conflicts))),
        refusal_matches=tuple(refusal_matches),
    )


def _specific_block_reason(
    validation: GroundingLedgerValidation | None,
    *,
    transport_confirmed: bool,
    telemetry_error: str,
    telemetry_refusal: bool,
) -> tuple[str, str]:
    if telemetry_error:
        return "uncertain", "visual_grounding_telemetry_error"
    if not transport_confirmed:
        return "transport_error", "visual_transport_error"
    if telemetry_refusal or (validation and validation.analysis_status == "refused"):
        return "refused", "visual_grounding_refused"
    if validation is None or not validation.schema_valid:
        return "uncertain", "visual_grounding_schema_invalid"
    if validation.grounded:
        return "grounded", ""
    if "insufficient_coverage" in validation.blocking_reasons:
        return "uncertain", "visual_grounding_insufficient_coverage"
    if validation.analysis_status == "transport_error":
        return "transport_error", "visual_transport_error"
    return "uncertain", "visual_grounding_uncertain"


def decide_grounding_guard(
    config: GroundingGuardConfig | Mapping[str, Any] | str | None,
    validation: GroundingLedgerValidation | None,
    *,
    media_present: bool,
    transport_confirmed: bool,
    telemetry_error: str = "",
    telemetry_refusal: bool = False,
) -> GroundingGuardDecision:
    """Map evidence into the public off/audit/strict decision contract."""

    normalized = normalize_grounding_guard_config(config)
    if normalized.mode == "off":
        return GroundingGuardDecision(
            mode="off",
            analysis_status="not_run",
            decision="disabled",
            would_block=False,
        )
    if not media_present:
        return GroundingGuardDecision(
            mode=normalized.mode,
            analysis_status="not_applicable",
            decision="not_applicable",
            would_block=False,
        )

    status, specific = _specific_block_reason(
        validation,
        transport_confirmed=transport_confirmed,
        telemetry_error=str(telemetry_error or ""),
        telemetry_refusal=bool(telemetry_refusal),
    )
    would_block = bool(specific)
    blocked_reasons = (
        ("visual_grounding_unverified", specific) if specific else ()
    )
    warnings = list(validation.warnings if validation else ())
    if telemetry_error:
        warnings.append(f"grounding telemetry failed: {telemetry_error}")
    if normalized.mode == "audit":
        decision = "warn" if would_block else "pass"
    else:
        decision = "block" if would_block else "pass"
    return GroundingGuardDecision(
        mode=normalized.mode,
        analysis_status=status,
        decision=decision,
        would_block=would_block,
        blocked_reasons=blocked_reasons,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _jsonable(value: Any, *, path: str = "trace") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GroundingGuardError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict(), path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            normalized_key = text_key.strip().lower()
            forbidden = (
                normalized_key in _FORBIDDEN_TRACE_KEYS
                or normalized_key.startswith("pixel_values")
            )
            if forbidden and not _is_safe_pixel_tensor_descriptor(
                path,
                normalized_key,
                item,
            ):
                raise GroundingGuardError(f"{path}.{text_key} may contain raw media or logits")
            result[text_key] = _jsonable(item, path=f"{path}.{text_key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise GroundingGuardError(f"{path} contains unsupported value type {type(value).__name__}")


def _json_size(value: Any) -> int:
    # The graph-facing report is emitted with indentation, sorted keys, and
    # ASCII escaping. Measure that exact representation so the public STRING
    # output cannot exceed the advertised 256 KiB even when it contains
    # non-ASCII text or deeply nested telemetry.
    return len(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _truncate_strings(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        marker = "...[truncated]"
        if max_chars <= len(marker):
            return marker[:max_chars]
        return value[: max_chars - len(marker)] + marker
    if isinstance(value, list):
        return [_truncate_strings(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_strings(item, max_chars) for key, item in value.items()}
    return value


def _bounded_sequence(value: Sequence[Any], max_items: int) -> list[Any]:
    """Keep deterministic endpoints when a report-only sequence must shrink."""

    items = list(value)
    limit = max(0, int(max_items))
    if len(items) <= limit:
        return items
    if limit == 0:
        return []
    if limit == 1:
        return items[:1]
    head_count = (limit + 1) // 2
    tail_count = limit - head_count
    return [*items[:head_count], *items[-tail_count:]]


def _compact_json_tree(value: Any, *, max_chars: int, max_items: int) -> Any:
    """Compact arbitrary JSON report detail without changing scalar types."""

    if isinstance(value, str):
        return _truncate_strings(value, max_chars)
    if isinstance(value, list):
        return [
            _compact_json_tree(item, max_chars=max_chars, max_items=max_items)
            for item in _bounded_sequence(value, max_items)
        ]
    if isinstance(value, dict):
        ordered_items = sorted(value.items(), key=lambda item: str(item[0]))
        selected_items = _bounded_sequence(ordered_items, max_items)
        result: dict[str, Any] = {}
        for key, item in selected_items:
            original_key = str(key)
            compact_key = _truncate_strings(original_key, max_chars)
            if compact_key in result:
                suffix_index = 1
                while True:
                    suffix = f"~{suffix_index}"
                    prefix_chars = max(0, max_chars - len(suffix))
                    collision_key = original_key[:prefix_chars] + suffix
                    if collision_key not in result:
                        compact_key = collision_key
                        break
                    suffix_index += 1
            result[compact_key] = _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_items,
            )
        omitted = len(ordered_items) - len(selected_items)
        if omitted:
            marker = "__dg_omitted_mapping_keys__"
            suffix = 1
            while marker in value or marker in result:
                marker = f"__dg_omitted_mapping_keys_{suffix}__"
                suffix += 1
            result[marker] = omitted
        return result
    return value


def _compact_trajectory_summary(
    trajectory: Any,
    *,
    max_chars: int,
    max_items: int,
) -> tuple[Any, dict[str, int]]:
    """Retain the first/final snapshots and summarize omitted trajectory detail."""

    if not isinstance(trajectory, dict):
        return (
            _compact_json_tree(
                trajectory,
                max_chars=max_chars,
                max_items=max_items,
            ),
            {},
        )

    result = copy.deepcopy(trajectory)
    stats: dict[str, int] = {}
    snapshots = result.get("snapshots")
    if isinstance(snapshots, list):
        selected = _bounded_sequence(snapshots, 2)
        omitted = len(snapshots) - len(selected)
        result["snapshots"] = [
            _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_items,
            )
            for item in selected
        ]
        if omitted:
            result["snapshots_truncated"] = omitted
            result["truncated"] = True
            stats["snapshots_omitted"] = omitted

    calls = result.get("calls")
    if isinstance(calls, list):
        selected_calls = _bounded_sequence(calls, max_items)
        omitted_calls = len(calls) - len(selected_calls)
        result["calls"] = [
            _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_items,
            )
            for item in selected_calls
        ]
        if omitted_calls:
            result["calls_truncated"] = omitted_calls
            result["truncated"] = True
            stats["calls_omitted"] = omitted_calls

    for key, item in list(result.items()):
        if key not in {"snapshots", "calls"}:
            result[key] = _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_items,
            )
    return result, stats


def _compact_validated_ledger(
    ledger: Any,
    *,
    max_chars: int,
    max_facts: int,
    max_text_items: int,
    max_evidence: int,
    max_typed_claims: int,
) -> tuple[Any, dict[str, Any]]:
    """Bound a validated ledger while retaining its typed ledger structure."""

    if not isinstance(ledger, dict):
        return ledger, {}

    result = copy.deepcopy(ledger)
    stats: dict[str, Any] = {}
    original_facts = result.get("observed_facts")
    if isinstance(original_facts, list):
        selected_facts = _bounded_sequence(original_facts, max_facts)
        compact_facts: list[Any] = []
        omitted_evidence = 0
        omitted_typed_claims = 0
        for fact in selected_facts:
            if not isinstance(fact, dict):
                compact_facts.append(
                    _compact_json_tree(
                        fact,
                        max_chars=max_chars,
                        max_items=max_evidence,
                    )
                )
                continue
            compact_fact = {
                key: _compact_json_tree(
                    item,
                    max_chars=max_chars,
                    max_items=max_evidence,
                )
                for key, item in fact.items()
                if key not in {"evidence", "typed_claims"}
            }
            evidence = fact.get("evidence")
            if isinstance(evidence, list):
                selected_evidence = _bounded_sequence(evidence, max_evidence)
                omitted_evidence += len(evidence) - len(selected_evidence)
                compact_fact["evidence"] = [
                    _compact_json_tree(
                        item,
                        max_chars=max_chars,
                        max_items=max_evidence,
                    )
                    for item in selected_evidence
                ]
            elif "evidence" in fact:
                compact_fact["evidence"] = _compact_json_tree(
                    evidence,
                    max_chars=max_chars,
                    max_items=max_evidence,
                )
            typed_claims = fact.get("typed_claims")
            if isinstance(typed_claims, list):
                selected_claims = _bounded_sequence(typed_claims, max_typed_claims)
                omitted_typed_claims += len(typed_claims) - len(selected_claims)
                compact_fact["typed_claims"] = [
                    _compact_json_tree(
                        item,
                        max_chars=max_chars,
                        max_items=max_typed_claims,
                    )
                    for item in selected_claims
                ]
            elif "typed_claims" in fact:
                compact_fact["typed_claims"] = _compact_json_tree(
                    typed_claims,
                    max_chars=max_chars,
                    max_items=max_typed_claims,
                )
            compact_facts.append(compact_fact)
        result["observed_facts"] = compact_facts
        stats["observed_facts_original"] = len(original_facts)
        stats["observed_facts_retained"] = len(compact_facts)
        if omitted_evidence:
            stats["evidence_references_omitted"] = omitted_evidence
        if omitted_typed_claims:
            stats["typed_claims_omitted"] = omitted_typed_claims

    text_list_stats: dict[str, dict[str, int]] = {}
    for key in (
        "inferred_facts",
        "creative_additions",
        "uncertainties",
        "grounding_failure_reasons",
    ):
        values = result.get(key)
        if not isinstance(values, list):
            continue
        selected_values = _bounded_sequence(values, max_text_items)
        result[key] = [
            _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_text_items,
            )
            for item in selected_values
        ]
        text_list_stats[key] = {
            "original": len(values),
            "retained": len(selected_values),
        }
    if text_list_stats:
        stats["text_lists"] = text_list_stats

    for key, item in list(result.items()):
        if key not in {
            "observed_facts",
            "inferred_facts",
            "creative_additions",
            "uncertainties",
            "grounding_failure_reasons",
        }:
            result[key] = _compact_json_tree(
                item,
                max_chars=max_chars,
                max_items=max_text_items,
            )
    return result, stats


_REPORT_COMPACTION_PROFILES: tuple[dict[str, int], ...] = (
    {
        "max_chars": 2048,
        "max_items": 128,
        "max_facts": 256,
        "max_text_items": 128,
        "max_evidence": 32,
        "max_typed_claims": 32,
    },
    {
        "max_chars": 1024,
        "max_items": 64,
        "max_facts": 256,
        "max_text_items": 64,
        "max_evidence": 16,
        "max_typed_claims": 16,
    },
    {
        "max_chars": 512,
        "max_items": 32,
        "max_facts": 192,
        "max_text_items": 32,
        "max_evidence": 8,
        "max_typed_claims": 8,
    },
    {
        "max_chars": 256,
        "max_items": 16,
        "max_facts": 128,
        "max_text_items": 16,
        "max_evidence": 4,
        "max_typed_claims": 4,
    },
    {
        "max_chars": 128,
        "max_items": 8,
        "max_facts": 96,
        "max_text_items": 8,
        "max_evidence": 2,
        "max_typed_claims": 2,
    },
    {
        "max_chars": 64,
        "max_items": 4,
        "max_facts": 48,
        "max_text_items": 4,
        "max_evidence": 1,
        "max_typed_claims": 1,
    },
    {
        "max_chars": 32,
        "max_items": 2,
        "max_facts": 16,
        "max_text_items": 2,
        "max_evidence": 1,
        "max_typed_claims": 1,
    },
)


def _compacted_report_candidate(
    report: dict[str, Any],
    profile: Mapping[str, int],
    *,
    original_bytes: int,
    profile_index: int,
) -> dict[str, Any]:
    max_chars = profile["max_chars"]
    max_items = profile["max_items"]
    candidate: dict[str, Any] = {}
    trajectory_stats: dict[str, int] = {}
    ledger_stats: dict[str, Any] = {}
    for key, value in report.items():
        if key == "trajectory_summary":
            candidate[key], trajectory_stats = _compact_trajectory_summary(
                value,
                max_chars=max_chars,
                max_items=max_items,
            )
        elif key == "validated_ledger":
            candidate[key], ledger_stats = _compact_validated_ledger(
                value,
                max_chars=max_chars,
                max_facts=profile["max_facts"],
                max_text_items=profile["max_text_items"],
                max_evidence=profile["max_evidence"],
                max_typed_claims=profile["max_typed_claims"],
            )
        else:
            candidate[key] = _compact_json_tree(
                value,
                max_chars=max_chars,
                max_items=max_items,
            )
    candidate["report_truncated"] = True
    candidate["compaction"] = {
        "schema": "dg-grounding-report-compaction/1",
        "original_bytes": original_bytes,
        "profile": profile_index,
        "trajectory": trajectory_stats,
        "validated_ledger": ledger_stats,
    }
    return candidate


def _ledger_structure_only(ledger: Any) -> Any:
    """Last-resort ledger shape used only for unusually tiny custom limits."""

    if not isinstance(ledger, dict):
        return ledger
    return {
        "schema": _truncate_strings(
            str(ledger.get("schema", GROUNDING_LEDGER_SCHEMA_ID)), 32
        ),
        "analysis_status": _truncate_strings(
            str(ledger.get("analysis_status", "uncertain")), 16
        ),
        "observed_facts": [],
        "inferred_facts": [],
        "creative_additions": [],
        "uncertainties": [],
        "grounding_failure_reasons": [],
    }


def _structural_compact_report(
    report: dict[str, Any],
    *,
    original_bytes: int,
) -> dict[str, Any]:
    """Retain every public report section even after content is exhausted."""

    result = {
        "schema": report.get("schema", GROUNDING_REPORT_SCHEMA_ID),
        "config": _compact_json_tree(
            report.get("config", {}), max_chars=16, max_items=1
        ),
        "analysis_status": report.get("analysis_status", "uncertain"),
        "decision": report.get("decision", "warn"),
        "would_block": bool(report.get("would_block", True)),
        "grounding_guard_would_block": bool(
            report.get(
                "grounding_guard_would_block",
                report.get("would_block", True),
            )
        ),
        "blocked_reasons": _compact_json_tree(
            report.get("blocked_reasons", []), max_chars=16, max_items=1
        ),
        "asset_registry": _compact_json_tree(
            report.get("asset_registry", {}), max_chars=16, max_items=1
        ),
        "transport": _compact_json_tree(
            report.get("transport", {}), max_chars=16, max_items=1
        ),
        "validated_ledger": _ledger_structure_only(
            report.get("validated_ledger")
        ),
        "validation": _compact_json_tree(
            report.get("validation", {}), max_chars=16, max_items=1
        ),
        "attempt_count": report.get("attempt_count", 0),
        "retry_reasons": _compact_json_tree(
            report.get("retry_reasons", []), max_chars=16, max_items=1
        ),
        "trajectory_summary": {"truncated": True},
        "effective_sampling": _compact_json_tree(
            report.get("effective_sampling", {}), max_chars=16, max_items=1
        ),
        "timings": _compact_json_tree(
            report.get("timings", {}), max_chars=16, max_items=1
        ),
        "forward_call_count": report.get("forward_call_count", 0),
        "warnings": _compact_json_tree(
            report.get("warnings", []), max_chars=16, max_items=1
        ),
        "verification_level": _truncate_strings(
            str(report.get("verification_level", "unknown")), 16
        ),
        "report_truncated": True,
        "compaction": {
            "schema": "dg-grounding-report-compaction/1",
            "original_bytes": original_bytes,
            "profile": "structural",
        },
    }
    if "transport_proof" in report:
        result["transport_proof"] = _compact_json_tree(
            report["transport_proof"], max_chars=16, max_items=1
        )
    return result


def compact_grounding_report(
    report: Mapping[str, Any],
    max_bytes: int = MAX_COMPACT_REPORT_BYTES,
) -> dict[str, Any]:
    """Bound metadata without dropping the public grounding report sections."""

    if max_bytes < 2048:
        raise GroundingGuardError("compact report limit must be at least 2048 bytes")
    compact = _jsonable(report, path="report")
    original_bytes = _json_size(compact)
    if original_bytes <= max_bytes:
        return compact

    # Telemetry is diagnostic detail rather than grounding evidence. Discard
    # intermediate snapshots first, while retaining the first/final views.
    trajectory = compact.get("trajectory_summary")
    if isinstance(trajectory, dict):
        snapshots = trajectory.get("snapshots")
        if isinstance(snapshots, list) and len(snapshots) > 2:
            trajectory["snapshots"] = [snapshots[0], snapshots[-1]]
            trajectory["snapshots_truncated"] = len(snapshots) - 2
            trajectory["truncated"] = True
    compact["report_truncated"] = True
    if _json_size(compact) <= max_bytes:
        return compact

    # Exhaust trajectory-only compaction before touching evidence. This keeps
    # a small validated ledger byte-for-byte intact when telemetry alone is
    # responsible for the overflow.
    trajectory_profiles = (
        (4096, 64),
        (2048, 32),
        (1024, 16),
        (512, 8),
        (256, 4),
        (128, 2),
        (64, 1),
    )
    if "trajectory_summary" in compact:
        original_trajectory = compact.get("trajectory_summary")
        trajectory_probe = dict(compact)
        trajectory_probe["trajectory_summary"] = {}
        trajectory_can_resolve_overflow = _json_size(trajectory_probe) <= max_bytes
        last_trajectory = original_trajectory
        profiles = list(enumerate(trajectory_profiles, start=1))
        if not trajectory_can_resolve_overflow:
            profiles = profiles[-1:]
        for detail_index, (max_chars, max_items) in profiles:
            candidate = dict(compact)
            compacted_trajectory, trajectory_stats = _compact_trajectory_summary(
                original_trajectory,
                max_chars=max_chars,
                max_items=max_items,
            )
            candidate["trajectory_summary"] = compacted_trajectory
            candidate["compaction"] = {
                "schema": "dg-grounding-report-compaction/1",
                "original_bytes": original_bytes,
                "profile": f"trajectory-{detail_index}",
                "trajectory": trajectory_stats,
            }
            if _json_size(candidate) <= max_bytes:
                return candidate
            last_trajectory = compacted_trajectory
        compact["trajectory_summary"] = last_trajectory

    # Validation diagnostics duplicate host conclusions already represented
    # by analysis_status/decision. Bound that detail before reducing the
    # validated evidence ledger itself.
    if "validation" in compact:
        original_validation = compact.get("validation")
        validation_probe = dict(compact)
        validation_probe["validation"] = {}
        validation_can_resolve_overflow = _json_size(validation_probe) <= max_bytes
        last_validation = original_validation
        profiles = list(enumerate(trajectory_profiles, start=1))
        if not validation_can_resolve_overflow:
            profiles = profiles[-1:]
        for detail_index, (max_chars, max_items) in profiles:
            candidate = dict(compact)
            compacted_validation = _compact_json_tree(
                original_validation,
                max_chars=max_chars,
                max_items=max_items,
            )
            candidate["validation"] = compacted_validation
            candidate["compaction"] = {
                "schema": "dg-grounding-report-compaction/1",
                "original_bytes": original_bytes,
                "profile": f"validation-{detail_index}",
            }
            if _json_size(candidate) <= max_bytes:
                return candidate
            last_validation = compacted_validation
        compact["validation"] = last_validation

    # A schema-valid ledger can still be several megabytes (and typed string
    # values are intentionally open-ended). Compact it in deterministic,
    # progressively tighter profiles rather than replacing the report with a
    # small status-only object.
    for profile_index, profile in enumerate(_REPORT_COMPACTION_PROFILES, start=1):
        candidate = _compacted_report_candidate(
            compact,
            profile,
            original_bytes=original_bytes,
            profile_index=profile_index,
        )
        if _json_size(candidate) <= max_bytes:
            return candidate

    structural = _structural_compact_report(
        compact,
        original_bytes=original_bytes,
    )
    if _json_size(structural) > max_bytes:
        raise GroundingGuardError("critical grounding report fields exceed size limit")
    return structural


def build_grounding_report(
    config: GroundingGuardConfig | Mapping[str, Any] | str | None,
    asset_registry: Mapping[str, Any],
    decision: GroundingGuardDecision,
    validation: GroundingLedgerValidation | None = None,
    *,
    transport: Mapping[str, Any] | None = None,
    trajectory_summary: Mapping[str, Any] | None = None,
    attempt_count: int = 0,
    retry_reasons: Sequence[str] = (),
    timings: Mapping[str, Any] | None = None,
    forward_call_count: int = 0,
    warnings: Sequence[str] = (),
    verification_level: str = "transport+structured_self_report",
    effective_sampling: Mapping[str, Any] | None = None,
    max_bytes: int = MAX_COMPACT_REPORT_BYTES,
) -> dict[str, Any]:
    """Build the compact graph-facing report; detailed raw text stays in traces."""

    normalized = normalize_grounding_guard_config(config)
    combined_warnings = [*decision.warnings, *warnings]
    report = {
        "schema": GROUNDING_REPORT_SCHEMA_ID,
        "config": normalized.to_dict(),
        "analysis_status": decision.analysis_status,
        "decision": decision.decision,
        "would_block": decision.would_block,
        "grounding_guard_would_block": decision.would_block,
        "blocked_reasons": list(decision.blocked_reasons),
        "asset_registry": copy.deepcopy(dict(asset_registry)),
        "transport": copy.deepcopy(dict(transport or {})),
        "validated_ledger": copy.deepcopy(validation.ledger) if validation else None,
        "validation": validation.to_dict() if validation else None,
        "attempt_count": max(0, _safe_int(attempt_count, 0)),
        "retry_reasons": [str(reason) for reason in retry_reasons],
        "trajectory_summary": copy.deepcopy(dict(trajectory_summary or {})),
        "effective_sampling": copy.deepcopy(
            dict(effective_sampling)
            if effective_sampling is not None
            else normalized.sampling_settings
        ),
        "timings": copy.deepcopy(dict(timings or {})),
        "forward_call_count": max(0, _safe_int(forward_call_count, 0)),
        "warnings": list(dict.fromkeys(str(item) for item in combined_warnings if str(item))),
        "verification_level": str(verification_level),
    }
    return compact_grounding_report(report, max_bytes=max_bytes)


def grounding_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)


def persist_grounding_trace(
    trace: Mapping[str, Any],
    output_root: str | Path,
    trace_subfolder: str = DEFAULT_TRACE_SUBFOLDER,
    *,
    filename: str | None = None,
    max_bytes: int = MAX_TRACE_BYTES,
) -> Path:
    """Persist a JSON-only trace beneath an explicitly supplied Comfy output root."""

    root = Path(output_root).expanduser().resolve()
    safe_subfolder = _safe_trace_subfolder(trace_subfolder)
    if safe_subfolder != str(trace_subfolder or "").strip().replace("\\", "/"):
        if str(trace_subfolder or "").strip() not in {"", DEFAULT_TRACE_SUBFOLDER}:
            raise GroundingGuardError("trace_subfolder must be a relative path without traversal")
    target_dir = (root / safe_subfolder).resolve()
    try:
        target_dir.relative_to(root)
    except ValueError as exc:
        raise GroundingGuardError("trace path escapes the Comfy output root") from exc

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        chosen_name = f"grounding-{timestamp}-{uuid.uuid4().hex[:8]}.json"
    else:
        chosen_name = str(filename).strip()
        if not chosen_name or Path(chosen_name).name != chosen_name:
            raise GroundingGuardError("trace filename must be a plain filename")
        if not chosen_name.lower().endswith(".json"):
            chosen_name += ".json"

    payload = _jsonable(trace, path="trace")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    if len(encoded) > max_bytes:
        raise GroundingGuardError(
            f"grounding trace is {len(encoded)} bytes, above the {max_bytes}-byte limit"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / chosen_name
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded.decode("utf-8"))
        handle.write("\n")
    return path


__all__ = [
    "ANALYSIS_STATUSES",
    "ASSET_REGISTRY_SCHEMA_ID",
    "EXTERNAL_EVIDENCE_SCHEMA_ID",
    "EXTERNAL_EVIDENCE_SCHEMA_PATH",
    "EVIDENCE_TOKEN_BUDGETS",
    "FACT_CATEGORIES",
    "GROUNDING_CONFIG_TYPE",
    "GROUNDING_EVIDENCE_SCHEMA_PATH",
    "GROUNDING_LEDGER_SCHEMA_ID",
    "GROUNDING_MODES",
    "GROUNDING_REPORT_SCHEMA_ID",
    "GUARD_DECISIONS",
    "GroundingGuardConfig",
    "GroundingGuardDecision",
    "GroundingGuardError",
    "GroundingLedgerError",
    "GroundingLedgerValidation",
    "SAMPLING_PROFILE_SETTINGS",
    "SAMPLING_PROFILES",
    "build_asset_registry",
    "build_grounding_report",
    "compact_grounding_report",
    "decide_grounding_guard",
    "detect_refusal_phrases",
    "extract_grounding_ledger",
    "grounding_report_json",
    "load_grounding_evidence_schema",
    "load_external_evidence_schema",
    "normalize_grounding_guard_config",
    "parse_external_evidence",
    "persist_grounding_trace",
    "validate_grounding_ledger",
]
