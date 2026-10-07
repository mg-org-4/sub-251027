---
name: write-h3-v39-reference-prompts
description: Write Sequence Prompts for H3 Continuum V3.9 Reference Images, using fixed image slots and per-chunk assignments. Supports concise hand-written prompts and structured LLM prompts. Does not generate videos or modify workflows.
---

# H3 V3.9 reference prompts

An independent Continuum adaptation, not an official MiniMax skill. Preserve the installed `write-continuum-h3-prompts` skill; this package is separate and self-contained.

## Establish the small input contract

Use known workflow settings and images. Establish only what affects the result:
- Chunks, seconds per chunk, Reference Use, and each connected slot's selected chunks.
- What each image supplies: character identity, clothing, object, environment, or style; what to preserve and what may change.
- Intended action, camera, and start/end state of each chunk.
- Whether First/Last Image, Timeline Video or audio inputs are actually enabled.

Prefer a simple list such as `R1: red-coated woman, chunks 1/2/3; R4: navy-top woman, chunks 1/2; R9: teal-coated woman, chunks 2/3`. Do not require YAML or a large questionnaire. Ask only when missing information materially changes a reference binding or the requested story. Never infer slot numbers from attachment order or invent details of unseen images.

## Reference and routing contracts

- `@R1` to `@R9` identify the fixed sockets on **H3 Continuum Reference Images V3.9**. Current V3.9 All chunks and Per chunk both resolve active tags before encoding. This is not V3.8 syntax compatibility.
- Write `the woman in @R1`, not `@R1 walks`. The tag denotes a source image, not a person.
- `<Subject N>` denotes a continuing entity. Keep its meaning stable throughout the video; it is independent of the R number. Multiple images may describe one Subject; an image may supply an object rather than a person.
- Do not manually renumber reference assets as `<Picture N>`. Their effective Picture numbers vary with selected references and First/Last inputs. If explicitly describing First/Last frame labels, require the actual presentation mapping; do not guess.
- Cite an image only where it is connected and selected. An inactive tag is currently left unchanged with a runtime warning, not repaired automatically. Identify this before handing over text; do not add runtime execution restrictions.
- Reference selection is not a visibility schedule. Turning a reference off does not remove a carried character. Describe exits, entrances and inherited states explicitly. Not every reference needs to appear as a separate object/person.
- Use `Prompt Format = Timeline` for staged chunks. Put every `[Chunk N]` on its own line, covering each requested chunk exactly once. Put text on the next line. Do not put chunk-specific references in a shared preamble before the first header.
- A chunk is not a camera cut. Shot markers belong inside the selected chunk text. Some workflows merge logical chunks into physical groups; check the effective Reference Plan if First/Last or terminal merging is involved. Never promise independent sampling or silently change assignments to bypass a conflict.

## Choose the requested writing style

**Simple / hand-written:** Default to one short paragraph per chunk: image role + observable action + camera + end state. The next paragraph inherits that state. Avoid mandatory Subject numbering or six headings for simple requests. Use explicit source-to-target actions when the reference pose differs. Add short soundscape/music lines only when audio intent matters.

**Structured / LLM:** When requested or useful for multi-entity/attribute-transfer work, use these fields in order inside each chunk:
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
Read [the structured contract and example](references/structured.md) before authoring this form.

Do not equate longer prose with greater control. Preserve requested detail but avoid contradictory actions, redundant appearance intensifiers and too many events for the interval. For continuation, describe the actual intended handoff (position, motion, camera); `Continuation of Chunk N.` is a useful cue, not a sampler command or quality guarantee.

Preserve supplied dialogue, lyrics and visible text in their original language. Do not invent an Audio/Video reference. Driving Audio, reference audio and generated soundscape are different paths; prompt text cannot connect them. For voice references, request the actual audio-label mapping and assign each line to a speaker. An image-only request does not authorize adding dialogue/music.

## Return and check

Return a short settings/assignment note outside a single copy-ready Sequence Prompt block. If prompt-only was requested, omit the note. Keep citations, analysis, audit results and YAML outside the prompt block. English scene descriptions are the default, not a forced translation of supplied speech/text.

Before returning: verify full chunk coverage, active reference tags, stable Subject meanings, attribute ownership, plausible action density, entrance/exit timing intent, and boundary-state consistency. Do not claim observed output states when no video was inspected. Do not guarantee exact timestamps, identity, seamlessness or that this wording is empirically optimal.

Use [the 3 × 5-second simple example](references/simple-3x5.txt) for syntax, not as a fixed duration or mandatory story. The reference assignments for that example are R1=1/2/3, R4=1/2, R9=2/3; all other slots off, First/Last/Video/Audio references disabled.

No skill installation, generation, downloads, workflow edits, node changes or publication are authorized by prompt authoring alone.
