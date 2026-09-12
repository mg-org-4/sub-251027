import os
import re
from pathlib import Path
from typing import Any

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io
from kgen import models
from kgen.executor import tipo as tipo_executor
from kgen.executor.tipo import (
    OPERATION_LIST,
    apply_tipo_prompt,
    parse_tipo_request,
    parse_tipo_result,
    tipo_runner,
    tipo_single_request,
)
from kgen.formatter import apply_format, seperate_tags

from ..tipo_installer import ensure_runtime, logger
from ..tipo_installer.hardware import torch_devices

CATEGORY = "utils/promptgen"
LIST = io.Custom("LIST")

models.model_dir = Path(folder_paths.models_dir) / "kgen"
os.makedirs(models.model_dir, exist_ok=True)
logger.info(f"Using model dir: {models.model_dir}")

DEFAULT_FORMAT = """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>"""

LENGTH_OPTIONS = ["very_short", "short", "long", "very_long"]

DEFAULT_OPERATION = "short_to_tag_to_long"

_current_model = None


def managed_gguf_names() -> "set[str]":
    """Filenames download_gguf writes, using its own `{repo stem}_{file}` scheme."""
    return {
        f"{name.split('/')[-1]}_{gguf}"
        for name, ggufs in models.tipo_model_list
        for gguf in ggufs
    }


def model_options():
    names = [
        f"{name} | {gguf}" for name, ggufs in models.tipo_model_list for gguf in ggufs
    ]
    names += [name for name, _ in models.tipo_model_list]

    # Listed above as "repo | file", which the registry resolves; the raw
    # filename would be the same model again under a name it cannot.
    managed = managed_gguf_names()
    names += sorted(
        file
        for file in os.listdir(models.model_dir)
        if file.endswith(".gguf") and file not in managed
    )
    return names


def operation_options():
    return sorted(OPERATION_LIST)


COMMENT_LINE = re.compile(r"^[^\S\n]*#[^\n]*(?:\n|$)", re.MULTILINE)
EXTRA_NETWORK = re.compile(r"<[^<>]+>")
EMBEDDING_PREFIX = re.compile(r"^embedding:", re.IGNORECASE)
DANGLING = re.compile(r"(?:^[\s,]+|[\s,]+$)")
REPEATED_COMMA = re.compile(r",\s*(?=,)")
NL_CATEGORIES = ("extended", "generated")
# Whitelisted, because tag_map also carries control fields such as `tag` and
# `target` whose values are prompt markers like <|tag_to_long|>.
TAG_CATEGORIES = (
    "special",
    "characters",
    "copyrights",
    "artist",
    "general",
    "quality",
    "meta",
    "rating",
)


def strip_comments(text: str) -> str:
    """Drop lines whose first non-space character is '#' (issue #22)."""
    if not text or "#" not in text:
        return text
    cleaned = REPEATED_COMMA.sub("", COMMENT_LINE.sub("", text))
    return DANGLING.sub("", cleaned)


def extract_extra_networks(text: str) -> "tuple[str, list[str]]":
    """Lift `<lora:...>` syntax out of the prompt so TIPO cannot rewrite it."""
    found = []

    def take(match):
        found.append(match.group(0))
        return " "

    remaining = REPEATED_COMMA.sub("", EXTRA_NETWORK.sub(take, text or ""))
    return DANGLING.sub("", remaining), found


def embedding_names() -> "set[str]":
    try:
        files = folder_paths.get_filename_list("embeddings")
    except Exception:
        return set()
    return {os.path.splitext(os.path.basename(str(f)))[0].lower() for f in files}


def embedding_spellings(tags) -> "dict[str, str]":
    """Map kgen's spaced form back to the original, for embeddings only.

    seperate_tags spaces out underscores for any token no tag list claims. That
    is correct for general tags, but an embedding name looks exactly like a tag
    and only the host can tell them apart, so ask ComfyUI which names exist and
    protect just those (issue #119).
    """
    known = embedding_names()
    if not known:
        return {}
    mapping = {}
    for tag in tags:
        bare = EMBEDDING_PREFIX.sub("", tag)
        if len(bare) >= 4 and "_" in bare and bare.lower() in known:
            mapping[bare.replace("_", " ")] = tag
    return mapping


def restore_embeddings(tag_map, mapping):
    if not mapping:
        return tag_map
    for cate, value in tag_map.items():
        if isinstance(value, list):
            tag_map[cate] = [mapping.get(tag, tag) for tag in value]
        elif isinstance(value, str):
            for spaced, original in mapping.items():
                value = value.replace(spaced, original)
            tag_map[cate] = value
    return tag_map


def split_tag_map(tag_map) -> "tuple[str, str]":
    """Split the result into tag text and natural-language text (issue #102)."""
    tags = []
    for cate in TAG_CATEGORIES:
        value = tag_map.get(cate)
        if isinstance(value, list):
            tags.extend(str(tag) for tag in value if str(tag).strip())
        elif isinstance(value, str) and value.strip():
            tags.append(value.strip())

    natural = []
    for cate in NL_CATEGORIES:
        value = tag_map.get(cate)
        if isinstance(value, str) and value.strip():
            natural.append(value.strip())
    return ", ".join(tags), " ".join(natural)


def with_networks(text: str, networks) -> str:
    if not networks:
        return text
    joined = " ".join(networks)
    return f"{text.rstrip().rstrip(',')}, {joined}" if text.strip() else joined


attn_syntax = (
    r"\\\(|"
    r"\\\)|"
    r"\\\[|"
    r"\\]|"
    r"\\\\|"
    r"\\|"
    r"\(|"
    r"\[|"
    r":\s*([+-]?[.\d]+)\s*\)|"
    r"\)|"
    r"]|"
    r"[^\\()\[\]:]+|"
    r":"
)
re_attention = re.compile(attn_syntax, re.VERBOSE)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.DOTALL)


def parse_prompt_attention(text):
    """Split a prompt into (text, weight) pairs, honouring (a:1.2) and [a] syntax."""
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def apply_strength(tag_map, strength_map, strength_map_nl):
    for cate in tag_map:
        if isinstance(tag_map[cate], str):
            if all(part in tag_map[cate] for part, _ in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\\(").replace(")", "\\)")
                    part = part.replace("(", "\\(").replace(")", "\\)")
                    new_prompt += f"({part}:{strength})"
                tag_map[cate] = new_prompt + org_prompt
            continue

        new_list = []
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\\(").replace(")", "\\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
        tag_map[cate] = new_list
    return tag_map


def split_weighted(text):
    """Return (clean text, tag->weight, [(nl part, weight)]) for a weighted prompt."""
    parsed = parse_prompt_attention(text)
    tags = []
    strength_map = {}
    nl_text = ""
    strength_map_nl = []

    for part, strength in parsed:
        nl_text += part
        part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
        tags.extend(part_tags)
        if strength == 1:
            continue
        strength_map_nl.append((part, strength))
        for tag in part_tags:
            strength_map[tag] = strength

    return nl_text, tags, strength_map, strength_map_nl


def device_index():
    index = getattr(args, "cuda_device", None)
    try:
        return int(index)
    except (TypeError, ValueError):
        return 0


def load_model(tipo_model, device):
    global _current_model
    if (tipo_model, device) == _current_model:
        return

    if " | " in tipo_model:
        model_name, gguf_name = tipo_model.split(" | ")
        target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
        if str(models.model_dir / target_file) not in models.list_gguf():
            models.download_gguf(model_name, gguf_name)
        target = os.path.join(str(models.model_dir), target_file)
        gguf = True
    elif tipo_model.endswith(".gguf"):
        target = tipo_model
        gguf = True
    else:
        target = tipo_model
        gguf = False

    extra = {}
    if gguf:
        if device != "cpu":
            extra["main_gpu"] = device_index()
    elif device not in ("cpu", "mps"):
        device = f"{device}:{device_index()}"

    models.load_model(target, gguf, device=device, **extra)
    _current_model = (tipo_model, device)


def collect_addon(tag_map, org_tag_map):
    addon = {"tags": [], "nl": ""}
    for cate in tag_map:
        if cate == "generated" and addon["nl"] == "":
            addon["nl"] = tag_map[cate]
            continue
        if cate == "extended":
            addon["nl"] = tag_map[cate]
            continue
        if cate not in org_tag_map:
            continue
        for tag in tag_map[cate]:
            if tag not in org_tag_map[cate]:
                addon["tags"].append(tag)
    return addon


def common_inputs(fifth):
    """Shared inputs. `fifth` slots in after tipo_model, where format/operation
    have always sat: ComfyUI matches saved widgets_values by position, so moving
    it would shift every stored value in existing workflows."""
    devices = torch_devices()
    models = model_options()
    return [
        io.String.Input("tags", multiline=True, tooltip="Danbooru-style tags."),
        io.String.Input(
            "nl_prompt", multiline=True, tooltip="Natural language prompt."
        ),
        io.String.Input(
            "ban_tags",
            multiline=True,
            tooltip="Comma separated tags to exclude. Regex supported.",
        ),
        io.Combo.Input("tipo_model", options=models, default=models[0]),
        fifth,
        io.Int.Input("width", default=1024, max=16384),
        io.Int.Input("height", default=1024, max=16384),
        io.Float.Input("temperature", default=0.5, step=0.01),
        io.Float.Input("top_p", default=0.95, step=0.01),
        io.Float.Input("min_p", default=0.05, step=0.01),
        io.Int.Input("top_k", default=80),
        io.Combo.Input("tag_length", options=LENGTH_OPTIONS, default="long"),
        io.Combo.Input("nl_length", options=LENGTH_OPTIONS, default="long"),
        # control_after_generate would add a linked widget and shift `device`
        # out of its stored slot in existing workflows.
        io.Int.Input("seed", default=1234),
        io.Combo.Input("device", options=devices, default=devices[0]),
    ]


def prompt_outputs():
    # tag_prompt/nl_prompt are appended, never inserted: output links are stored
    # by index, so reordering would rewire saved workflows.
    return [
        io.String.Output(display_name="prompt"),
        io.String.Output(display_name="user_prompt"),
        io.String.Output(display_name="unformatted_prompt"),
        io.String.Output(display_name="unformatted_user_prompt"),
        io.String.Output(display_name="tag_prompt"),
        io.String.Output(display_name="nl_prompt"),
    ]


class TIPO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TIPO",
            display_name="TIPO",
            category=CATEGORY,
            description="Expand a prompt with TIPO or DanTagGen text-presampling.",
            search_aliases=["tipo", "dantaggen", "prompt gen", "upsample prompt"],
            inputs=common_inputs(
                io.String.Input("format", multiline=True, default=DEFAULT_FORMAT)
            ),
            outputs=prompt_outputs(),
        )

    @classmethod
    def execute(
        cls,
        tags: str,
        nl_prompt: str,
        ban_tags: str,
        tipo_model: str,
        width: int,
        height: int,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        tag_length: str,
        nl_length: str,
        seed: int,
        device: str,
        format: str,
    ) -> io.NodeOutput:
        ensure_runtime()
        load_model(tipo_model, device)

        tags, networks = extract_extra_networks(strip_comments(tags))
        nl_prompt = strip_comments(nl_prompt)

        _, all_tags, strength_map, _ = split_weighted(tags)
        nl_prompt, _, _, strength_map_nl = split_weighted(nl_prompt)
        embeddings = embedding_spellings(all_tags)

        tipo_executor.BAN_TAGS = [
            tag.strip() for tag in ban_tags.split(",") if tag.strip()
        ]

        tag_length = tag_length.replace(" ", "_")
        nl_length = nl_length.replace(" ", "_")
        org_tag_map = restore_embeddings(seperate_tags(all_tags), embeddings)

        meta, operations, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in format)
            or "<|generated|>" in format,
        )
        meta["aspect_ratio"] = f"{width / height:.1f}"

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                tag_length,
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        tag_map = restore_embeddings(tag_map, embeddings)
        addon = apply_strength(
            collect_addon(tag_map, org_tag_map), strength_map, strength_map_nl
        )
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        tag_prompt, nl_output = split_tag_map(tag_map)

        return io.NodeOutput(
            with_networks(formatted_prompt_by_tipo, networks),
            formatted_prompt_by_user,
            with_networks(unformatted_prompt_by_tipo, networks),
            unformatted_prompt_by_user,
            with_networks(tag_prompt, networks),
            nl_output,
        )


class TIPOOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        operations = operation_options()
        default = (
            DEFAULT_OPERATION if DEFAULT_OPERATION in operations else operations[0]
        )
        return io.Schema(
            node_id="TIPOOperation",
            display_name="TIPO Single Operation",
            category=CATEGORY,
            description="Run one TIPO operation and return the raw tag maps.",
            inputs=common_inputs(
                io.Combo.Input("operation", options=operations, default=default)
            ),
            outputs=[
                LIST.Output(display_name="full_output"),
                LIST.Output(display_name="addon_output"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tags: str,
        nl_prompt: str,
        ban_tags: str,
        tipo_model: str,
        width: int,
        height: int,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        tag_length: str,
        nl_length: str,
        seed: int,
        device: str,
        operation: str,
    ) -> io.NodeOutput:
        ensure_runtime()
        load_model(tipo_model, device)

        tags, networks = extract_extra_networks(strip_comments(tags))
        nl_prompt = strip_comments(nl_prompt)

        original_nl_prompt = nl_prompt
        _, all_tags, strength_map, _ = split_weighted(tags)
        nl_prompt, _, _, strength_map_nl = split_weighted(nl_prompt)
        embeddings = embedding_spellings(all_tags)

        tipo_executor.BAN_TAGS = [
            tag.strip() for tag in ban_tags.split(",") if tag.strip()
        ]

        tag_length = tag_length.replace(" ", "_")
        org_tag_map = restore_embeddings(seperate_tags(all_tags), embeddings)

        meta, operations, general, nl_prompt = tipo_single_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            operation=operation,
        )
        meta["aspect_ratio"] = f"{width / height:.1f}"

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        tag_map = restore_embeddings(tag_map, embeddings)
        addon = apply_strength(
            collect_addon(tag_map, org_tag_map), strength_map, strength_map_nl
        )
        addon["user_tags"] = tags
        addon["user_nl"] = original_nl_prompt
        addon["networks"] = networks

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        return io.NodeOutput(tag_map, addon)


class TIPOFormat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TIPOFormat",
            display_name="TIPO Format",
            category=CATEGORY,
            description="Apply a prompt format to the output of TIPO Single Operation.",
            inputs=[
                LIST.Input("full_output"),
                LIST.Input("addon_output"),
                io.String.Input("format", multiline=True, default=DEFAULT_FORMAT),
            ],
            outputs=prompt_outputs(),
        )

    @classmethod
    def execute(
        cls,
        full_output: list,
        addon_output: dict[str, Any],
        format: str,
    ) -> io.NodeOutput:
        ensure_runtime()

        addon = dict(addon_output)
        tags = addon.pop("user_tags", "")
        nl_prompt = addon.pop("user_nl", "")
        networks = addon.pop("networks", [])
        tag_map = full_output

        _, all_tags, strength_map, _ = split_weighted(tags)
        nl_prompt, _, _, strength_map_nl = split_weighted(nl_prompt)
        embeddings = embedding_spellings(all_tags)

        org_tag_map = restore_embeddings(seperate_tags(all_tags), embeddings)
        meta, _, general, nl_prompt = parse_tipo_request(org_tag_map, nl_prompt)

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                "long",
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )

        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )
        tag_prompt, nl_output = split_tag_map(tag_map)

        return io.NodeOutput(
            with_networks(formatted_prompt_by_tipo, networks),
            formatted_prompt_by_user,
            with_networks(unformatted_prompt_by_tipo, networks),
            unformatted_prompt_by_user,
            with_networks(tag_prompt, networks),
            nl_output,
        )


NODES = [TIPO, TIPOOperation, TIPOFormat]
