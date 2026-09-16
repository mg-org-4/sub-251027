# H3 Street Scan

A local style study inspired by [Ingi Erlingsson's “please hold”](https://x.com/ingi_erlingsson/status/2097056649559134367), using newly generated footage rather than frames from the reference.

## Audio-reactive Chaos variant

### Native depth camera and audio mapping

Interactive Scan FX includes our own CPU depth-parallax renderer; it does not depend on Depthflow, OpenGL, or another model. It extends the existing depth projection with horizontal/vertical camera offsets and dolly, counter-adjusted around a selected steady-depth plane. `current` preserves the original renderer. `depth_parallax` can affect the `whole_scene` or only the voxel/depth layers inside reveals (`reveals_only`). Both remain 2.5D: newly exposed surfaces are filled from visible samples, not reconstructed.

Reveal weights choose voxel normals, edge scan, or depth **once per gesture** using the seed. Zero disables a layer; the total must remain positive. Depth styles are grayscale, false color, and contours. Equal voxel/edge weights with depth disabled preserve the original selection. Cursor activity thins gestures; reveal size is sampled at gesture start.

The on-node Audio Mapping panel assigns Kick/Snare/Hat envelopes to supported parameters over frame ranges `[start, end)`. Click a mapping to select its range, drag either timeline edge to resize, or drag its interior to move it. Clicking outside the selected range seeks the preview. Numeric range fields are available too. Overlapping enabled mappings for the same parameter are rejected. Mappings replace the parameter's value inside the range; outside it the base value remains. Brightness, saturation and glow retain their original envelope reactions outside mapped ranges, so mappings do not double-apply them.

Smoothing is optional, measured in seconds, and resets at shot boundaries. Speed mappings accumulate animation phase without resetting it. Cube size, FPS and seed are not modulation targets. The timeline shows the last rendered values and authored shot boundaries; settings changes require another execution. A Depth diagnostic tab shows the projected map before camera accents. Old workflows retain the original defaults until the new options are enabled.

The current workflow consolidates the effect into **FL Interactive Scan FX**. Each **FL Scan Analysis** packages a shot's depth, normals and optional masks/detections/pose; connect these in chronological order to the growing shot inputs. Their total frame count and resolution must match the source video, and all three envelopes must match its FPS and frame count. Generation and analysis remain external and cacheable.

The unified node reuses voxel rendering, digital depth projection, audio-locked cursor editing, masked snare brightness, kick saturation and kick edge glow. Its main widgets control the surface, cursor count, camera motion and seed. The Advanced panel contains projection, cut timing, cursor scale/reveal strength and finishing controls. Existing individual nodes remain available.

Its silent diagnostic preview has **Final**, **Surface**, **Mask**, and **Debug** tabs, a frame scrubber and synchronized Kick/Snare/Hat meters. Surface shows projection before camera accents; Debug overlays planned gesture trajectories on the final result. Previews describe the last execution, not live parameter changes. They are temporary MP4 files; rerun if ComfyUI clears its temp directory. Full-resolution final/surface/mask outputs and a timing report remain available. The labeled FL Video Combine output retains audio.

The current Y2K workflow uses `digital_layers` on all four Scan Composite nodes: a full-scene depth projection with filled splat gaps, crisp white borders and offset cobalt/black backing panels. It does not use the eroded fragment mask or temporal edge ghosts. `fragment` remains the default for older workflows; raggedness and echo controls apply only to that mode.

Frame- and second-based prompt schedules can run with no detected beats. Beat-index schedules still require a beat grid. The current workflow uses scheduler envelopes 1/2/3 as kick/snare/hat controls: every beat, every other beat, and short sixteenth-note pulses. These are authored grid events, not detected instrument stems. Its four 48-frame sections produce 192 frames (8 seconds), using seed 65, 10 SA Solver steps, beta57, CFG 1 and denoise 1. Earlier test settings below describe the original Chaos version.

**FL Voxel Normal Relief** now sits between Depth to Normals and each Scan Composite. It draws normal-colored cuboids with shaded sides; relative depth sets their height and a seeded smooth wave adds subtle animation. The four shared controls are cube size (12 px), relief (0.65), animation (0.18), and speed (0.7 cycles/second). This is a 2.5D image effect, not a voxel mesh. Keep its depth and normals aligned, and connect it before projection so the cursor reveals follow the same scene and camera transforms. The raw normal previews remain unchanged. FL Video Combine saves the labeled before/after with audio.

The primary **H3 Street Scan Chaos - Generate and Composite** workflow uses **FL Audio Beat Prompt Schedule** as its audio source, frame-accurate trim and prompt timeline. The test crop is `Inspired by dnb.mp3`, frames 768–1032 at 24 fps (32–43 seconds). Its locally installed Beat This model supplies beat timing.

The schedule feeds **FL MiniMax H3 Beat Shot Planner**, including its matching audio and reactive prompt-envelope outputs. Each render receives a shot-local audio reference and rebased beat-weighted motion prompts. **Beat KSampler** uses the Turbo LoRA, four Euler steps, simple scheduling and the existing 12/3 video/audio sigma shifts. **Shot Assembler** removes H3 padding. Audio conditioning encourages musical movement; it does not guarantee frame-perfect physical accents.

**FL Scan Audio Edit** defaults to `audio_locked` in this workflow. It preserves every source frame chronologically and adds abrupt camera cuts, punch-ins and tilt without retiming generated motion. Its backwards-compatible `remix` mode still permits random seeks and playback-rate changes; those intentionally discard diffusion-time alignment. Keep the four analysis slices and shot-length settings consistent with the authored prompt sections when changing section boundaries.

Cursor gestures start independently on audio accents rather than restarting at every cut. A fixed seed controls their positions, directions, curved paths, sizes, durations and normal/edge layer choices. Silence produces no new gestures. The report includes cursor events and exact source-frame mapping. The reveal mask excludes cursor graphics and can drive downstream masked effects.

The existing Fill reactive brightness, saturation and edge-glow nodes provide additional modulation. The current workflow no longer uses Drum Detector.

The older Chaos FX Lab remains a cached-footage remix workflow. The primary workflow no longer depends on those clips. All Chaos outputs are saved under `output/FL_H3_StreetScan_Chaos`.

## Workflows

- **H3 Street Scan - Generate and Composite**: four editable H3 prompts, native Turbo sampling, analysis, effects and assembly.
- **H3 Street Scan - FX Lab**: the same effects applied to saved H3 clips. Use this for fast look development or replace the four video paths with your own footage.

Both are saved in `user/default/workflows`. Rendered videos and workflow copies are in `output/FL_H3_StreetScan`.

Four 66-frame sections at 24 fps produce an 11-second, 640×640 silent montage. The comparison output places the original on the left and the scan treatment on the right. FX Lab paths refer to local generated clips; include those clips and update paths when sharing. The full generation workflow does not depend on those cached files.

## Pipeline

H3 generates full-frame footage. Depth Anything V2 estimates relative depth; blurred depth feeds Depth to Normals. Impact/Ultralytics supplies person segmentation and face/hand detections. DWPose supplies body and hand joints. `FL Scan Video Detections` matches box IDs between adjacent frames. `FL Street Scan Composite` applies depth reprojection, a chipped scene boundary, normal-map flashes, tracking graphics, pose flashes and short echoes.

This is **2.5D reprojection**, not a multi-view reconstruction or exportable scene mesh. Depth annotations are normalized relative estimates, not metric coordinates. Tracking IDs are nearest-neighbor associations, not persistent identity recognition. The compositor accepts DWPose's pixel-coordinate `POSE_KEYPOINT` batches; pose input is optional.

Each shot is analyzed separately so depth smoothing, feature tracking and echo history reset at cuts. Source, depth, normals, masks, detections and pose must describe the same frames and resolution (pose includes its own canvas dimensions).

## Controls

- **Orbit / depth relief:** parallax. Larger values expose more missing surfaces and reprojection gaps.
- **Scene scale / raggedness:** framing and chipped boundary. The detected subject is retained before reprojection.
- **Normal mix:** surface-color flashes and the rectangular normal-field panel.
- **HUD / pose opacity:** detection graphics and body/hand skeleton flashes.
- **Echo strength:** two-frame edge trails.

Seeds are fixed for repeatable tuning. Depth and normal previews are below each analysis branch. Changing an effect control does not require another H3 generation; Comfy's execution cache can also reuse unchanged analysis while it remains available.

## Live effect preview

The shared workflow's four analysis branches use **FL Scan Video Section**, indices 0–3 with `sections=4`, all connected to the same video as Interactive Scan FX. Boundaries are derived from the input frame count, so the chunks cover the complete video exactly once, including uneven remainders. These are processing chunks, not detected scene cuts. Keep all four branches connected; video/envelope FPS and duration must still agree.

Minimum/maximum cut-frame controls follow each other when crossed. Reversed ranges in older workflows or API requests are sorted before rendering; they do not abort the effect.

The right-hand editor uses Voxels, Camera, Reveals, Overlay and Finish tabs. Numeric controls have bounded sliders/fields; `?` opens parameter help. Reset restores the base value without removing mappings. Tab reset requires confirmation. Solo changes the demo view explicitly; switching tabs does not.

Drag a colored Kick/Snare/Hat chip onto a highlighted parameter, or click the chip then the parameter (keyboard: focus the card and press Enter). Click its colored badge for range, smoothing, invert and enable controls. Existing assignments offer replacement or a new unused time range. Overlapping enabled ranges are rejected in the editor and backend. Reset mapping preserves its assigned time section; Remove mapping disconnects it. The timeline below the preview edits the selected badge's range; rendered-value meters show the last execution, not live demo predictions.

Interactive Scan FX has a two-column editor: visuals on the left and effect controls on the right. **Live demo** is a lightweight synthetic depth scene available before execution; use All, Voxels, Depth, Reveals, Overlay, or Finish to inspect approximate effects. Pause the demo to compare parameter changes at a fixed time. It does not run models, queue work, or predict the exact output. Audio mappings, detection, occlusion and edit timing still require a render. **Last render** retains the actual diagnostic video and envelope timeline. Connected parameter inputs take precedence over stored controls; the demo uses stored values. Changes still require execution to update output videos.

## Installed dependencies used

- ComfyUI native MiniMax H3 and FL MiniMax H3 LoRA Block Loader.
- H3 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, Qwen3-VL H3 NVFP4 text encoder, H3 video/audio VAEs, and the ref2v Turbo 4-step LoRA.
- DepthAnythingV2: `depth_anything_v2_vitl_fp32.safetensors` (bf16 compute).
- Impact Pack/Subpack: `person_yolov8m-seg.pt`, `face_yolov8m.pt`, `hand_yolov8s.pt`.
- ControlNet Aux DWPose: `yolox_l.torchscript.pt`, `dw-ll_ucoco_384_bs5.torchscript.pt`.
- Image Filters (Depth to Normals), KJNodes (side-by-side concatenation), VideoHelperSuite, and Fill Nodes.

No paid generation API or new model download was used for this test. The named third-party loader nodes may download missing weights on another installation.
