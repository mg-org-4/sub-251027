# LTX-2.5 Experimental Long Horizon benchmark

This directory defines an **unexecuted benchmark protocol** for the Director's
optional long-horizon policy. It is not a claim that the comparison has already
been rendered or scored. Before running it, map the four asset IDs listed in
the manifest to one first frame and three duration-matched depth guides. The
protocol then uses the same first frame at 5, 20, and 60 seconds so duration is
the principal changing condition.

The four variants are:

- `bare`: the one-line user brief with no Director expansion.
- `current_director`: the existing duration-aware LTX contract with long-horizon
  planning off.
- `long_horizon`: Experimental Long Horizon forced on.
- `long_horizon_structural`: the same long-horizon prompt plus a duration-matched
  depth guide. This fourth arm intentionally measures prompt plus structural
  conditioning and is not part of the prompt-only A/B/C comparison.

Every arm keeps model, resolution, frame rate, sampler, scheduler, step counts,
CFG, negative prompt, and both sampler seeds fixed. The manifest uses 121, 481,
and 1,441 frames at 24 fps, corresponding to 5, 20, and 60 seconds under LTX's
`(frames - 1) / fps` convention.

The locked sampler settings mirror the audited 60-second Director run:
`euler_ancestral`, `bong_tangent`, 12 first-stage steps, and three refinement
evaluations. Results remain empty until all four variants have actually been
run for every listed seed pair and scored against the rubric.

For each seed pair, rate camera monotonicity, anchor persistence, duplication,
spatial drift, abrupt reframing, terminal-composition accuracy, prompt
adherence, and audio continuity. Rate at regular five-second checkpoints for
the 20- and 60-second cases. Treat a single successful seed as anecdotal; the
paired matrix is the acceptance evidence.

The manifest also records the prompt-contract rubric. Long-horizon prompts must
remain one paragraph of at most 200 words and generally four to eight
sentences. The four internal phases are planning concerns, not chapters, shots,
scene changes, emitted headings, or timecodes. Missing phase or terminal cues
are diagnostics and must not block generation.
