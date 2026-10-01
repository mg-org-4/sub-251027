# Structured reference format

Use only when requested or useful. Each chunk contains the following English fields:

```text
[Chunk 1]
subject_definitions:
<Subject 1> is the red-coated woman in @R1.
<Subject 2> is the navy-top woman in @R4.
summary:
[reference generation] Two women walk together through a covered walkway.
retention_analysis:
<Subject 1>: partially_preserved — preserve face, hairstyle and clothing; adapt pose and gaze to walking.
<Subject 2>: partially_preserved — preserve face, hairstyle and clothing; adapt pose and gaze to walking.
detailed_description:
[Shot 1] An eye-level medium-wide camera tracks alongside both women as they walk from screen left to screen right. Both remain clearly visible. They continue walking as the chunk ends, with open space on the right for a third person to join next.
overall_soundscape:
Footsteps and quiet walkway ambience. No dialogue.
non_diegetic_music:
N/A
```

Keep Subject 1/2 meanings in later chunks. A new person from R9 can be Subject 3; do not rename her Subject 2 because she becomes Picture 2. Re-establish only the needed source binding in each selected prompt. For a continuing person without an active reference, describe inherited identity/state without an inactive @R tag.

Use retention markers with a concrete scope: partially_preserved for preserved identity/clothing with changed motion/background; attribute_transfer for applying clothing or a prop from another source; fully_preserved only when the requested content actually warrants it. These are semantic instructions, not hard masks or guaranteed constraints.

For a bag reference: `<Subject 4> is the tan bag in @R2, worn by <Subject 1>.` In the action text keep this ownership. If the character's source image already contains another bag, explicitly replace that source accessory with the selected bag, while preserving the person's identity and clothes.

In Chunk 2+, start detailed_description with `Continuation of Chunk 1.` (use the actual preceding number) plus the inherited physical state. Keep the six-field order; the continuation sentence is not a parser requirement. Use new shot markers/times only for intentional cuts. Do not invent `<Video 1>` for preceding generated chunks.

Official MiniMax Reference guide uses these six fields and recommends substantial scene detail (normally 350–500 English words in detailed_description for generation). Compact per-chunk wording here is a Continuum adaptation to test, not an officially proven better length. A short simple prompt remains valid input to Continuum.

Sources checked for the design:
- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt
- https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Local V3.9: v3/reference_images_v39.py, v3/driving_nodes.py, v3/reference_runtime.py, v3/reference_effective_plan.py, v2/prompts.py (2026-09-28). V3.9 is a local pre-release implementation, not a promise about every public package named 3.9.
