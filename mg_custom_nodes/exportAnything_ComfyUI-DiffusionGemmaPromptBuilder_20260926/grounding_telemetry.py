"""Passive, bounded telemetry for DiffusionGemma denoising.

This module deliberately has no import-time dependency on torch or transformers.
Both integration objects use the small duck-typed interfaces consumed by
``DiffusionGemmaGenerationMixin.generate``:

* ``DiffusionGemmaTelemetryLogitsProcessor`` is callable and returns the exact
  logits object it receives.
* ``DiffusionGemmaDraftStreamer`` implements ``put``, ``put_draft``, and
  ``end`` without opting into logits streaming.

Only token IDs, decoded text, and scalar summaries are retained.  Tensor and
logits objects never become collector state.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Callable, Iterable, Sequence


TELEMETRY_SCHEMA_VERSION = "dg-denoising-telemetry/1"
REFERENCE_SELECTED_STEPS = (48, 32, 16, 8, 4, 1)
REFERENCE_LATE_STEPS = (8, 4, 1)
DEFAULT_MAX_SUMMARY_BYTES = 256 * 1024
DEFAULT_CANVAS_SIZE = 256
DEFAULT_POSITION_LIMIT = 16


_REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cannot_help_request",
        re.compile(
            r"\bi (?:cannot|can't) (?:help|assist) "
            r"(?:with )?(?:this|that|the request)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cannot_comply_request",
        re.compile(
            r"\bi (?:cannot|can't) (?:comply with|fulfill|complete) "
            r"(?:this|that|the request)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cannot_analyze_media",
        re.compile(
            r"\bi (?:cannot|can't|am unable to) "
            r"(?:analy[sz]e|describe|review|view|process) "
            r"(?:this|that|the) (?:image|video|clip|media|content)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sorry_but_cannot",
        re.compile(
            r"\bi(?: am|'m) sorry,? but i (?:cannot|can't) "
            r"(?:help|assist|comply|provide|analy[sz]e|describe|review)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unable_to_comply",
        re.compile(
            r"\bi am unable to (?:comply|help|assist|provide) "
            r"(?:with )?(?:this|that|the request)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "must_decline_request",
        re.compile(
            r"\bi (?:must|have to) (?:decline|refuse) "
            r"(?:this|that|the request)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "against_guidelines",
        re.compile(
            r"\b(?:this|that|doing so|the request) (?:is|would be) "
            r"against (?:my|the) (?:guidelines|polic(?:y|ies))\b",
            re.IGNORECASE,
        ),
    ),
)


def scaled_selected_steps(
    max_denoising_steps: int,
    reference_steps: Sequence[int] = REFERENCE_SELECTED_STEPS,
) -> tuple[int, ...]:
    """Scale the canonical 48-step checkpoints to another denoising budget.

    Rounding is deterministic half-up rather than Python's bankers' rounding.
    Collisions in short schedules are removed while preserving descending
    generation order.
    """

    maximum = int(max_denoising_steps)
    if maximum <= 0:
        raise ValueError("max_denoising_steps must be positive")

    scaled: list[int] = []
    for reference in reference_steps:
        raw = (int(reference) * maximum) / 48.0
        value = max(1, min(maximum, int(math.floor(raw + 0.5))))
        if value not in scaled:
            scaled.append(value)
    if maximum not in scaled:
        scaled.insert(0, maximum)
    if 1 not in scaled:
        scaled.append(1)
    return tuple(sorted(scaled, reverse=True))


def deterministic_positions(total_positions: int, limit: int = DEFAULT_POSITION_LIMIT) -> tuple[int, ...]:
    """Return at most ``limit`` evenly distributed flat position indices."""

    total = max(0, int(total_positions))
    count = max(0, min(int(limit), DEFAULT_POSITION_LIMIT))
    if total == 0 or count == 0:
        return ()
    if total <= count:
        return tuple(range(total))
    if count == 1:
        return (0,)

    result: list[int] = []
    for ordinal in range(count):
        raw = (ordinal * (total - 1)) / (count - 1)
        index = int(math.floor(raw + 0.5))
        if not result or result[-1] != index:
            result.append(index)
    return tuple(result)


def find_complete_refusal_clauses(text: str) -> list[dict[str, str]]:
    """Find refusal clauses while ignoring isolated refusal-associated words."""

    normalized = re.sub(r"\s+", " ", str(text or "").replace("’", "'")).strip()
    candidates: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for clause_id, pattern in _REFUSAL_PATTERNS:
        for match in pattern.finditer(normalized):
            matched_text = match.group(0)
            key = (match.start(), match.end(), matched_text.casefold())
            if key not in seen:
                seen.add(key)
                candidates.append(
                    (match.start(), match.end(), clause_id, matched_text)
                )

    # Prefer the most complete clause when a shorter pattern is wholly nested
    # inside it (for example, "I can't describe this image" inside
    # "I'm sorry, but I can't describe this image").
    selected: list[tuple[int, int, str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item[1] - item[0]), item[0], item[2]),
    ):
        start, end, _clause_id, _matched_text = candidate
        if any(
            start < other_end and other_start < end
            for other_start, other_end, *_ in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        {"clause_id": clause_id, "text": matched_text}
        for _start, _end, clause_id, matched_text in selected
    ]


def _plain_int(value: Any) -> int:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("telemetry value does not expose shape")
    return tuple(int(dimension) for dimension in shape)


def _token_batches(value: Any) -> list[list[int]]:
    """Copy tensor-like token IDs into plain Python lists."""

    payload = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if isinstance(payload, tuple):
        payload = list(payload)
    if not isinstance(payload, list):
        raise TypeError("streamed token IDs must be list-like")
    if not payload:
        return [[]]
    if not isinstance(payload[0], (list, tuple)):
        payload = [payload]
    return [[int(token) for token in batch] for batch in payload]


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _pure_row_summary(row: Sequence[float]) -> tuple[int, int | None, float | None, float | None]:
    values = [float(value) for value in row]
    if not values:
        raise ValueError("logits row is empty")
    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = values[top1] - values[top2] if top2 is not None else None

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    entropy = 0.0
    if denominator > 0.0:
        for exponential in exponentials:
            if exponential > 0.0:
                probability = exponential / denominator
                entropy -= probability * math.log(probability)
    return top1, top2, _finite_float(margin), _finite_float(entropy)


def _summarize_scores(scores: Any, position_limit: int) -> dict[str, Any]:
    """Summarize tensor-like ``[batch, canvas, vocab]`` logits.

    Test doubles may expose ``telemetry_rows(indices)`` and return ordinary
    numeric rows.  Real torch tensors follow the vectorized branch, without an
    import of torch and without copying the full logits tensor off device.
    """

    shape = _shape_of(scores)
    if len(shape) != 3:
        raise ValueError(f"expected logits rank 3, received shape {shape}")
    batch_size, canvas_length, vocab_size = shape
    if batch_size <= 0 or canvas_length <= 0 or vocab_size <= 0:
        raise ValueError(f"invalid logits shape {shape}")

    flat_indices = deterministic_positions(batch_size * canvas_length, position_limit)
    fake_rows = getattr(scores, "telemetry_rows", None)
    positions: list[dict[str, Any]] = []

    if callable(fake_rows):
        rows = fake_rows(flat_indices)
        if len(rows) != len(flat_indices):
            raise ValueError("telemetry_rows returned an unexpected row count")
        summarized = [_pure_row_summary(row) for row in rows]
    else:
        flat_scores = scores.reshape(batch_size * canvas_length, vocab_size)
        sampled_scores = flat_scores[list(flat_indices)].float()
        top_values, top_indices = sampled_scores.topk(k=min(2, vocab_size), dim=-1)
        log_normalizer = sampled_scores.logsumexp(dim=-1)
        probabilities = sampled_scores.softmax(dim=-1)
        entropy = log_normalizer - (probabilities * sampled_scores).sum(dim=-1)

        top_value_rows = top_values.detach().cpu().tolist()
        top_index_rows = top_indices.detach().cpu().tolist()
        entropy_values = entropy.detach().cpu().tolist()
        summarized = []
        for values, indices, entropy_value in zip(
            top_value_rows,
            top_index_rows,
            entropy_values,
            strict=True,
        ):
            top1 = int(indices[0])
            top2 = int(indices[1]) if len(indices) > 1 else None
            margin = float(values[0] - values[1]) if len(values) > 1 else None
            summarized.append(
                (top1, top2, _finite_float(margin), _finite_float(entropy_value))
            )

    for flat_index, (top1, top2, margin, entropy_value) in zip(
        flat_indices,
        summarized,
        strict=True,
    ):
        positions.append(
            {
                "flat_index": int(flat_index),
                "batch_index": int(flat_index // canvas_length),
                "canvas_position": int(flat_index % canvas_length),
                "top1_token_id": int(top1),
                "top2_token_id": int(top2) if top2 is not None else None,
                "top1_top2_margin": margin,
                "entropy": entropy_value,
            }
        )

    margins = [item["top1_top2_margin"] for item in positions if item["top1_top2_margin"] is not None]
    entropies = [item["entropy"] for item in positions if item["entropy"] is not None]
    return {
        "sampled_position_count": len(positions),
        "positions": positions,
        "aggregate": {
            "mean_top1_top2_margin": (
                sum(margins) / len(margins) if margins else None
            ),
            "min_top1_top2_margin": min(margins) if margins else None,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
            "max_entropy": max(entropies) if entropies else None,
        },
    }


def _json_size(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _text_preview(text: str, limit: int) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"[:limit]
    return value[: limit - 1] + "…"


def _ids_preview(batch: Sequence[int], limit: int = 64) -> list[int | str]:
    values = [int(token) for token in batch]
    if len(values) <= limit:
        return values
    side = max(1, (limit - 1) // 2)
    return [*values[:side], "…", *values[-side:]]


class GroundingTelemetryCollector:
    """Per-generation collector containing only JSON-safe, compact values."""

    def __init__(
        self,
        *,
        initial_input_length: int,
        max_denoising_steps: int = 48,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        tokenizer: Any | None = None,
        decoder: Callable[..., str] | None = None,
        position_limit: int = DEFAULT_POSITION_LIMIT,
        max_summary_bytes: int = DEFAULT_MAX_SUMMARY_BYTES,
    ) -> None:
        self.initial_input_length = int(initial_input_length)
        self.max_denoising_steps = int(max_denoising_steps)
        self.canvas_size = int(canvas_size)
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.position_limit = max(0, min(int(position_limit), DEFAULT_POSITION_LIMIT))
        self.max_summary_bytes = max(4096, int(max_summary_bytes))
        if self.initial_input_length < 0:
            raise ValueError("initial_input_length cannot be negative")
        if self.canvas_size <= 0:
            raise ValueError("canvas_size must be positive")

        self.selected_steps = scaled_selected_steps(self.max_denoising_steps)
        self.late_steps = frozenset(
            max(
                1,
                min(
                    self.max_denoising_steps,
                    int(
                        math.floor(
                            (reference * self.max_denoising_steps) / 48.0 + 0.5
                        )
                    ),
                ),
            )
            for reference in REFERENCE_LATE_STEPS
        )
        self._selected_step_set = frozenset(self.selected_steps)
        self._last_context: dict[str, Any] | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._stability_comparisons: list[dict[str, Any]] = []
        self._committed_batches: list[list[int]] = []
        self._forward_count = 0
        self._canvas_forward_counts: dict[int, int] = {}
        self._saw_initial_stream_value = False
        self._stream_ended = False
        self._errors: list[dict[str, str]] = []
        self._snapshot_ordinal = 0

    @property
    def has_errors(self) -> bool:
        return bool(self._errors)

    def record_error(self, stage: str, error: BaseException | str) -> None:
        message = str(error)
        entry = {"stage": str(stage), "message": _text_preview(message, 512)}
        if entry not in self._errors:
            self._errors.append(entry)
        if len(self._errors) > 32:
            del self._errors[:-32]

    def _decode_batches(self, batches: Sequence[Sequence[int]]) -> list[str]:
        if self.tokenizer is not None and callable(getattr(self.tokenizer, "batch_decode", None)):
            try:
                decoded = self.tokenizer.batch_decode(
                    [list(batch) for batch in batches],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded = self.tokenizer.batch_decode(
                    [list(batch) for batch in batches],
                    skip_special_tokens=True,
                )
            return [str(text) for text in decoded]

        decode = self.decoder
        if decode is None and self.tokenizer is not None:
            decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return ["" for _ in batches]

        result: list[str] = []
        for batch in batches:
            try:
                text = decode(
                    list(batch),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                try:
                    text = decode(list(batch), skip_special_tokens=True)
                except TypeError:
                    text = decode(list(batch))
            result.append(str(text))
        return result

    def observe_logits(self, input_ids: Any, scores: Any, cur_step: Any) -> None:
        input_shape = _shape_of(input_ids)
        if len(input_shape) != 2:
            raise ValueError(f"expected input_ids rank 2, received shape {input_shape}")
        current_length = input_shape[1]
        delta = current_length - self.initial_input_length
        if delta < 0:
            raise ValueError("input_ids became shorter than the initial prompt")
        if delta % self.canvas_size:
            self.record_error(
                "canvas_tracking",
                f"input length delta {delta} is not divisible by canvas size {self.canvas_size}",
            )
        canvas_number = (delta // self.canvas_size) + 1
        # DiffusionGemma constructs ``cur_step`` on the CUDA device. Calling
        # ``.item()`` here would synchronize the GPU on every denoising
        # forward, so derive the descending step from the per-canvas forward
        # ordinal on CUDA. CPU values and test doubles remain directly
        # inspectable without a device synchronization.
        if bool(getattr(cur_step, "is_cuda", False)):
            step = self.max_denoising_steps - self._canvas_forward_counts.get(
                canvas_number, 0
            )
        else:
            step = _plain_int(cur_step)
        if not 1 <= step <= self.max_denoising_steps:
            raise ValueError(
                f"denoising step {step} is outside 1..{self.max_denoising_steps}"
            )

        self._forward_count += 1
        self._canvas_forward_counts[canvas_number] = (
            self._canvas_forward_counts.get(canvas_number, 0) + 1
        )
        selected = step in self._selected_step_set
        context: dict[str, Any] = {
            "canvas": canvas_number,
            "step": step,
            "selected": selected,
        }
        if selected:
            context["logits_summary"] = _summarize_scores(scores, self.position_limit)
        self._last_context = context

    def _new_snapshot(
        self,
        context: dict[str, Any],
        batches: list[list[int]],
        *,
        committed: bool,
    ) -> dict[str, Any]:
        texts = self._decode_batches(batches)
        refusal_matches = [
            match
            for text in texts
            for match in find_complete_refusal_clauses(text)
        ]
        self._snapshot_ordinal += 1
        snapshot: dict[str, Any] = {
            "ordinal": self._snapshot_ordinal,
            "canvas": int(context["canvas"]),
            "step": int(context["step"]),
            "selected": bool(context.get("selected", False)),
            "committed": bool(committed),
            "draft_token_ids": batches,
            "draft_text": texts,
            "refusal_clauses": refusal_matches,
        }
        if "logits_summary" in context:
            snapshot["logits_summary"] = copy.deepcopy(context["logits_summary"])
        return snapshot

    @staticmethod
    def _token_agreement(
        previous_batches: Sequence[Sequence[int]],
        current_batches: Sequence[Sequence[int]],
    ) -> tuple[int, int, float]:
        """Compare drafts without constructing or retaining tensor objects.

        The denominator is the larger token count for each paired batch, so a
        missing batch or a length mismatch is treated as disagreement rather
        than silently ignored.
        """

        matching_tokens = 0
        compared_tokens = 0
        batch_count = max(len(previous_batches), len(current_batches))
        for batch_index in range(batch_count):
            previous = (
                previous_batches[batch_index]
                if batch_index < len(previous_batches)
                else ()
            )
            current = (
                current_batches[batch_index]
                if batch_index < len(current_batches)
                else ()
            )
            matching_tokens += sum(
                1 for old_token, new_token in zip(previous, current) if old_token == new_token
            )
            compared_tokens += max(len(previous), len(current))
        agreement = (
            matching_tokens / compared_tokens if compared_tokens else 1.0
        )
        return matching_tokens, compared_tokens, agreement

    def _record_draft_stability(
        self,
        context: dict[str, Any],
        batches: list[list[int]],
    ) -> None:
        previous_snapshot = next(
            (
                snapshot
                for snapshot in reversed(self._snapshots)
                if snapshot.get("selected")
                and int(snapshot.get("canvas", -1)) == int(context["canvas"])
            ),
            None,
        )
        if previous_snapshot is None:
            return
        matching, compared, agreement = self._token_agreement(
            previous_snapshot.get("draft_token_ids") or [],
            batches,
        )
        self._stability_comparisons.append(
            {
                "canvas": int(context["canvas"]),
                "from_step": int(previous_snapshot["step"]),
                "to_step": int(context["step"]),
                "matching_tokens": int(matching),
                "compared_tokens": int(compared),
                "agreement": round(float(agreement), 8),
            }
        )

    def capture_draft(self, value: Any) -> None:
        context = self._last_context
        if context is None or not context.get("selected"):
            return
        batches = _token_batches(value)
        self._record_draft_stability(context, batches)
        self._snapshots.append(self._new_snapshot(context, batches, committed=False))

    def capture_stream_value(self, value: Any) -> None:
        if not self._saw_initial_stream_value:
            self._saw_initial_stream_value = True
            return
        context = self._last_context
        if context is None:
            self.record_error("streamer", "committed canvas arrived without denoising context")
            return

        batches = _token_batches(value)
        while len(self._committed_batches) < len(batches):
            self._committed_batches.append([])
        for index, batch in enumerate(batches):
            self._committed_batches[index].extend(batch)

        for snapshot in reversed(self._snapshots):
            if (
                snapshot["canvas"] == context["canvas"]
                and snapshot["step"] == context["step"]
            ):
                snapshot["committed"] = True
                snapshot["committed_matches_draft"] = (
                    snapshot.get("draft_token_ids") == batches
                )
                return
        self._snapshots.append(self._new_snapshot(context, batches, committed=True))

    def end_stream(self) -> None:
        self._stream_ended = True

    def _draft_stability_summary(self, *, include_comparisons: bool = True) -> dict[str, Any]:
        canvases: list[dict[str, Any]] = []
        canvas_numbers = sorted(
            {
                int(snapshot["canvas"])
                for snapshot in self._snapshots
                if snapshot.get("selected")
            }
        )
        for canvas_number in canvas_numbers:
            comparisons = [
                copy.deepcopy(comparison)
                for comparison in self._stability_comparisons
                if int(comparison["canvas"]) == canvas_number
            ]
            matching = sum(int(item["matching_tokens"]) for item in comparisons)
            compared = sum(int(item["compared_tokens"]) for item in comparisons)
            agreements = [float(item["agreement"]) for item in comparisons]
            entry: dict[str, Any] = {
                "canvas": canvas_number,
                "comparison_count": len(comparisons),
                "matching_tokens": matching,
                "compared_tokens": compared,
                "mean_agreement": round(matching / compared, 8) if compared else 1.0,
                "minimum_agreement": round(min(agreements), 8) if agreements else None,
                "final_agreement": round(agreements[-1], 8) if agreements else None,
            }
            if include_comparisons:
                entry["comparisons"] = comparisons
            canvases.append(entry)
        return {
            "comparison_count": len(self._stability_comparisons),
            "canvas_count": len(canvases),
            "per_canvas": canvases,
            "details_truncated": not include_comparisons,
        }

    def _base_summary(self) -> dict[str, Any]:
        final_text = self._decode_batches(self._committed_batches) if self._committed_batches else []
        final_matches = [
            match
            for text in final_text
            for match in find_complete_refusal_clauses(text)
        ]
        late_snapshots = [
            snapshot
            for snapshot in self._snapshots
            if int(snapshot.get("step", -1)) in self.late_steps
        ][-3:]
        late_refusal_count = sum(
            1 for snapshot in late_snapshots if snapshot.get("refusal_clauses")
        )
        persistent_late_refusal = (
            len(late_snapshots) >= 2 and late_refusal_count >= 2
        )
        refusal_detected = bool(final_matches) or persistent_late_refusal

        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "initial_input_length": self.initial_input_length,
            "canvas_size": self.canvas_size,
            "max_denoising_steps": self.max_denoising_steps,
            "selected_steps": list(self.selected_steps),
            "late_steps": sorted(self.late_steps, reverse=True),
            "sampled_position_limit": self.position_limit,
            "forward_count": self._forward_count,
            "canvas_forward_counts": {
                str(canvas): count
                for canvas, count in sorted(self._canvas_forward_counts.items())
            },
            "draft_stability": self._draft_stability_summary(),
            "snapshots": copy.deepcopy(self._snapshots),
            "final_output": {
                "text": final_text,
                "token_count": [len(batch) for batch in self._committed_batches],
                "refusal_clauses": final_matches,
            },
            "refusal": {
                "detected": refusal_detected,
                "final_output_match": bool(final_matches),
                "persistent_late_match": persistent_late_refusal,
                "late_snapshot_match_count": late_refusal_count,
                "late_snapshot_count": len(late_snapshots),
            },
            "stream_ended": self._stream_ended,
            "errors": copy.deepcopy(self._errors),
            "truncated": False,
        }

    @staticmethod
    def _minimal_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        token_batches = snapshot.get("draft_token_ids") or []
        texts = snapshot.get("draft_text") or []
        return {
            "ordinal": snapshot.get("ordinal"),
            "canvas": snapshot.get("canvas"),
            "step": snapshot.get("step"),
            "selected": snapshot.get("selected", False),
            "committed": snapshot.get("committed", False),
            "draft_token_ids_preview": [
                _ids_preview(batch, 32) for batch in token_batches[:2]
            ],
            "draft_token_count": [len(batch) for batch in token_batches],
            "draft_text": [_text_preview(text, 256) for text in texts[:2]],
            "refusal_clauses": snapshot.get("refusal_clauses", []),
            "payload_truncated": True,
        }

    def _cap_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        original_size = _json_size(payload)
        if original_size <= self.max_summary_bytes:
            return payload

        payload["truncated"] = True
        payload["truncation"] = {
            "original_size_bytes": original_size,
            "dropped_intermediate_snapshots": 0,
            "compacted_fields": [],
        }
        snapshots = payload["snapshots"]

        while len(snapshots) > 2 and _json_size(payload) > self.max_summary_bytes:
            snapshots.pop(1)
            payload["truncation"]["dropped_intermediate_snapshots"] += 1

        if _json_size(payload) > self.max_summary_bytes:
            payload["snapshots"] = [
                self._minimal_snapshot(snapshot) for snapshot in snapshots
            ]
            payload["truncation"]["compacted_fields"].append("snapshot_payloads")

        if _json_size(payload) > self.max_summary_bytes:
            payload["draft_stability"] = self._draft_stability_summary(
                include_comparisons=False
            )
            payload["truncation"]["compacted_fields"].append(
                "draft_stability_comparisons"
            )

        if _json_size(payload) > self.max_summary_bytes:
            payload["final_output"]["text"] = [
                _text_preview(text, 512)
                for text in payload["final_output"].get("text", [])[:2]
            ]
            payload["errors"] = [
                {
                    "stage": _text_preview(error.get("stage", ""), 64),
                    "message": _text_preview(error.get("message", ""), 160),
                }
                for error in payload.get("errors", [])[-4:]
            ]
            payload["truncation"]["compacted_fields"].extend(
                ["final_output_text", "errors"]
            )

        if _json_size(payload) > self.max_summary_bytes:
            # A defensive final form.  It still preserves the identity and
            # refusal result of the first/final snapshots, but cannot carry
            # their complete payload under an unusually small configured cap.
            kept = payload.get("snapshots", [])
            payload = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "initial_input_length": self.initial_input_length,
                "canvas_size": self.canvas_size,
                "max_denoising_steps": self.max_denoising_steps,
                "selected_steps": list(self.selected_steps),
                "forward_count": self._forward_count,
                "draft_stability": self._draft_stability_summary(
                    include_comparisons=False
                ),
                "snapshots": [
                    self._minimal_snapshot(snapshot) for snapshot in kept[:1]
                ]
                + (
                    [self._minimal_snapshot(kept[-1])]
                    if len(kept) > 1
                    else []
                ),
                "refusal": payload["refusal"],
                "stream_ended": self._stream_ended,
                "errors": payload.get("errors", [])[-2:],
                "truncated": True,
                "truncation": {
                    "original_size_bytes": original_size,
                    "dropped_intermediate_snapshots": max(
                        0, len(self._snapshots) - min(2, len(self._snapshots))
                    ),
                    "compacted_fields": ["minimal_summary"],
                },
            }

        return payload

    def summary(self) -> dict[str, Any]:
        payload = self._cap_summary(self._base_summary())
        # The 4 KiB minimum and bounded previews make this branch unreachable
        # in normal use.  Keep a hard guarantee for future schema additions.
        if _json_size(payload) > self.max_summary_bytes:
            raise RuntimeError("telemetry summary could not be reduced below its byte cap")
        return payload

    def summary_json(self) -> str:
        return json.dumps(
            self.summary(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )


class DiffusionGemmaTelemetryLogitsProcessor:
    """A mathematically passive DiffusionGemma logits processor."""

    def __init__(self, collector: GroundingTelemetryCollector) -> None:
        self.collector = collector

    def __call__(
        self,
        input_ids: Any,
        scores: Any,
        cur_step: Any,
    ) -> Any:
        try:
            self.collector.observe_logits(input_ids, scores, cur_step)
        except Exception as error:  # Telemetry must never alter generation.
            self.collector.record_error("logits_processor", error)
        return scores


class DiffusionGemmaDraftStreamer:
    """Draft-compatible streamer that never asks DiffusionGemma for logits."""

    def __init__(self, collector: GroundingTelemetryCollector) -> None:
        self.collector = collector

    def put(self, value: Any) -> None:
        try:
            self.collector.capture_stream_value(value)
        except Exception as error:
            self.collector.record_error("streamer.put", error)

    def put_draft(self, value: Any) -> None:
        try:
            self.collector.capture_draft(value)
        except Exception as error:
            self.collector.record_error("streamer.put_draft", error)

    def end(self) -> None:
        try:
            self.collector.end_stream()
        except Exception as error:
            self.collector.record_error("streamer.end", error)


# Compatibility name used by proof-gate capability detection and by callers
# that do not need to distinguish which diffusion model supplies the logits.
PassiveTelemetryLogitsProcessor = DiffusionGemmaTelemetryLogitsProcessor


def build_diffusiongemma_telemetry(
    **collector_kwargs: Any,
) -> tuple[
    GroundingTelemetryCollector,
    DiffusionGemmaTelemetryLogitsProcessor,
    DiffusionGemmaDraftStreamer,
]:
    """Build the three per-call telemetry objects used by ``generate``."""

    collector = GroundingTelemetryCollector(**collector_kwargs)
    return (
        collector,
        DiffusionGemmaTelemetryLogitsProcessor(collector),
        DiffusionGemmaDraftStreamer(collector),
    )


__all__ = [
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_MAX_SUMMARY_BYTES",
    "DEFAULT_POSITION_LIMIT",
    "DiffusionGemmaDraftStreamer",
    "DiffusionGemmaTelemetryLogitsProcessor",
    "GroundingTelemetryCollector",
    "PassiveTelemetryLogitsProcessor",
    "REFERENCE_LATE_STEPS",
    "REFERENCE_SELECTED_STEPS",
    "TELEMETRY_SCHEMA_VERSION",
    "build_diffusiongemma_telemetry",
    "deterministic_positions",
    "find_complete_refusal_clauses",
    "scaled_selected_steps",
]
