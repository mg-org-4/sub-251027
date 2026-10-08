# Gemini media inputs and music captioning

FL Gemini Text API accepts optional `audio` (AUDIO), `image` (IMAGE), and `video` (native VIDEO) connections, together or separately. Its response remains a STRING, and existing text-only workflows still work. Use a Google API key on the node. The selected Gemini model must support the connected media; the custom model field accepts an explicit model ID, subject to Google's availability and capabilities.

Connect Load Audio to `audio`, Load Image to `image`, or Load Video/Create Video to `video`. Every audio/image batch item is sent in order as a separate attachment. An image batch has no frame timing or soundtrack; use VIDEO for timed footage with sound. Native videos are exported through ComfyUI's video interface, including the selected trim and audio.

Media is encoded into temporary files, uploaded through Google's Files API, and deleted remotely after the request, including on failures. Local temporary files and the API client are also cleaned up. File processing has a five-minute deadline. Large files remain subject to the selected model's media and context limits. This node does not silently shorten recordings. The output is text; asking for JSON in the prompt does not enforce a JSON schema.

## What to capture for YuE2

Prioritize audible properties that you would want to request during generation:

| Property | Useful detail |
| --- | --- |
| Rhythm | Half-time or four-on-the-floor feel, swing, syncopation, kick/snare placement, percussion density. Distinguish pulse feel from estimated BPM. |
| Bass | Sub weight, sustained versus rhythmic notes, wobble or modulation, call-and-response with drums. |
| Instruments and motifs | Recurring melodic hook, chord stabs, pads, instrumental roles and how a motif changes. |
| Development | Order of sections, additions and removals, contrast between drops, breakdown length, return of a motif. |
| Energy and transitions | Sparse versus dense passages, tension/release, fills, risers, abrupt cuts, gradual filtering. |
| Space and texture | Dry drums, long dub-delay tails, wide pads, gritty bass, restrained or bright high end. Describe audible effects without inventing hardware or settings. |
| Vocals | Sung/spoken/chopped texture, delivery, language, layering, presence of intelligible words. |
| Harmony | Stable or changing harmony, tonal mood; exact key/chords only when reliable. |

For progression, prefer relationships: “The second drop adds a higher bass response and denser hats” is more informative than “energetic, dark, atmospheric.” Include steady, repetitive arrangements when that is what the recording actually does.

Keep training captions compact and consistent across recordings. A practical starting point is roughly 60–120 words, adjusted to the material; this is a captioning convention, not a YuE2 requirement. Avoid artist identification, praise, invented lyrics, and unsupported technical precision.

## Separate training text from analysis

YuE2's current dataset uses `.caption.txt` for style and `.lyrics.txt` for lyrics. Its AR training request uses planning off. A timestamped arrangement map is not currently read as a training control or converted into a score. Describing an arrangement in the style caption can supply useful conditioning, but does not guarantee exact timed events or train score planning.

Use three outputs from an analysis prompt:

1. **Style:** a concise description of the sound, plus a short chronological account of the arrangement. Copy this field into the reviewed style caption.
2. **Lyrics:** exact audible words, preserving repetitions and justified section labels. Leave empty for instrumental recordings. Keep production notes out of this field.
3. **Timeline and uncertainty:** approximate section boundaries, audible changes, confidence, and passages needing review. Keep these as separate analysis metadata.

If training on extracted excerpts, caption what is audible in each excerpt. A caption describing a full-song buildup and second drop is misleading when paired with a clip containing only the intro. When splitting songs, retain their shared `.song.txt` identity so related excerpts stay together in YuE2's train/validation split.

## Analysis prompt for FL Gemini Text API

Use audio alone when creating factual audio-training labels. If you also attach cover art or video, explicitly separate visual observations from audible evidence.

```text
Listen to the entire attached recording and describe only audible evidence.
Do not identify the artist or use visual content to infer musical properties.
Return JSON with these fields:

style: A compact music-training caption, approximately 60–120 words.
Cover genre, rhythm and pulse feel, bass character, instruments and recurring
motifs, vocals, mood, spatial texture and production. Then describe the actual
progression in order: what enters or leaves, how tension develops, how later
sections differ, and how it ends. Do not invent development if it stays steady.

lyrics: Transcribe all intelligible lyrics verbatim in their original language
and order. Preserve every repetition. Use [Verse], [Chorus], [Bridge], etc. only
when justified. Do not include arrangement descriptions as lyrics. For an
instrumental use an empty string. Do not guess unclear words.

instrumental: Boolean.

sections: Array of {start_seconds, end_seconds, label, audible_description,
change_from_previous, confidence}. Times are approximate seconds from the
beginning of this attachment. Describe audible changes rather than applying
a standard pop structure. Do not invent exact beat or bar boundaries.

uncertainty: List unclear lyric passages and uncertain musical observations.
Omit exact BPM, key, chords or technical settings when uncertain.

complete: Boolean indicating whether you analyzed and transcribed the entire
recording. If incomplete, say what was omitted in uncertainty.
```

Review the JSON before copying fields to the YuE2 sidecars. The existing YuE2 captioner has its own stricter output schema; this richer prompt is for the general Gemini node. To improve that captioner's style output now, add an instruction such as:

> In the style field, describe the arrangement in chronological order. Identify audible entrances, removals, breakdowns, motif changes, and differences between successive drops. Describe steady repetition honestly. Keep timestamps and production notes out of the lyrics field.

Approximate timestamps are review aids, not forced alignment. Check important boundaries against the audio before using them to create training excerpts.

References: [Google audio understanding](https://ai.google.dev/gemini-api/docs/audio), [Google video understanding](https://ai.google.dev/gemini-api/docs/video-understanding).
