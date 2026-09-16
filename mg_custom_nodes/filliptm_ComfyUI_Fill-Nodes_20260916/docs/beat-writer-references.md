# Timeline references and storyboards

All storyboards requested in one Writer response are submitted in one graph of independent asynchronous Nano nodes, without a local concurrency-count cap. Queue items still execute normally; provider throttling, request-size validation and shared batch completion affect actual speed. Large requests may spend credits simultaneously. There are no automatic paid retries.

Timeline and moodboard images use local 256-pixel WebP thumbnails with cache revalidation. Originals load only in the image popup, alongside available generation metadata. Reroll/remove controls sit below each timeline image. Loading indicators distinguish thumbnail loading from queued/generating jobs; reduced-motion preferences are respected. Wheel gestures over reference cards zoom/pan the timeline normally.

All sections requested in a Writer response share the same continuity brief and snapshot of checked moodboards. For stronger consistency, supply a clear character sheet and a style reference with explicit roles. Text alone cannot guarantee identical designs. Rerolls reuse the original brief and moodboard references; ask the Writer for new storyboards to use changed moodboard selections. A first generated master reference can be uploaded as a checked moodboard for subsequent sections; automatic master-reference generation is not implemented.

Ask Beat Writer to generate storyboards for the selected sections. It writes chronological image prompts and queues official Nano Banana 2 Partner Node jobs in ComfyUI. Prefer 2×2 grids for sections up to six seconds, 3×3 for longer sections.

Completed sheets are split into panels locally and assigned automatically. Active sheet thumbnails on timeline clips represent selected video references. If a section changes or attachment fails, its saved sheet remains visible as **not attached**, with a reason and an **Attach** button that uses no generation credits. **×** dismisses an unattached result without changing active references. Generated storyboards take precedence over reference assignments for the same section in one Writer response. Click a thumbnail to preview; **↻** generates a replacement. The old references remain active while a reroll runs. Generation and rerolls use ComfyUI credits without another approval dialog.

Four moodboard slots accept uploads, drops and pasted images. Add an optional role and check **Use** to send that image to the Writer and Google's image-generation endpoint. Unchecked images stay local. Checked moodboards and chat attachments share the Writer's eight-image limit.

The scheduler must connect directly to **FL MiniMax H3 Beat Shot Planner**. Storyboard generation connects **FL Prompt Reference Library** automatically and sets the planner's visual reference mode to **full**. Custom section references replace the planner's default media. The timeline soundtrack is unchanged. Panels are visual references, not exact timed keyframe constraints.

The optional **Section references** inspector supports manual image, video and audio assignments. `<Picture N>` and `<Video N>` each start at 1 within their media type, in selection order. Video audio remains paired; audio numbering also includes the timeline song. Only selected files are loaded and encoded. Grouped render sections must share the same ordered references.

Leave the editor open while the Writer prepares image requests. Submitted image jobs continue in ComfyUI if the editor closes; reopening recovers their results. Edited or deleted sections do not receive stale results automatically. Reconnecting cannot repeat an already submitted paid request. Check ComfyUI history for failed or unknown submissions before generating again; cancellation does not guarantee a refund.

Workflows store metadata, not embedded media. Share the referenced files under `input/fl-prompt-references`, `input/fl-beat-writer` and `output/fl-storyboards` with the workflow, preserving relative paths. Job records stay in the local Beat Writer data directory. Credentials are not stored in these manifests.
