# Dynamic Scan FX analysis

Reference-file change detection belongs to the scheduler's serialized reference settings. The reference library no longer fingerprints its connected schedule: ComfyUI does not supply linked values during fingerprinting, which previously raised a warning and invalidated downstream caches on every run. Selected reference-file path, modification time and size still invalidate the scheduler when changed.

Connect a video batch and its FL prompt schedule to **FL Scan Video Shots**. Its IMAGE list runs one shared depth/normals/detection/pose branch for every authored shot. Adjacent sections with the same render group stay together. Without a schedule the complete video is one shot. A supplied schedule must match the video's frame count and cover it without gaps or overlaps.

Package each result with **FL Scan Analysis**, then connect **FL Scan Analysis Collect** to one Interactive Scan FX analysis socket. The collector preserves order and shot boundaries without concatenating analysis tensors. Existing manually connected analysis sockets still work. Processing follows actual shot boundaries, not arbitrary quarters; this changes effect resets compared with the old four-quarter layout.

For VFX-only iteration, use an **FL Switch** with the saved raw video on `on_false` and the H3 assembler on `on_true`. Set it to false, and disable any separate output nodes that still request the H3 branch. The saved video is an explicit input, not a cache that automatically updates when prompts change. Select a new saved take after generating one; retain its matching schedule and FPS.

ComfyUI's RAM-pressure cache can evict analysis results. On builds with active/inactive headroom arguments, `--cache-ram 10 16` retains results when memory permits while keeping headroom. These are free-RAM thresholds in GB, not cache-size limits. Keep the flags in the launcher used for subsequent starts. A saved source prevents H3 reruns even after cache eviction or restart; analysis may still rerun.

The synthetic Scan FX demo renders at 320 pixels and up to 12 FPS, applies color filters once per surface, and stops animation scheduling when paused, offscreen, inactive, or in a hidden tab. These settings do not alter final video rendering.

Voxel polygon coordinates and face colors are prepared in arrays before rasterization, preserving painter order and OpenCV drawing. Saturation and edge glow use ComfyUI's selected compute device one frame at a time, including transfers back to CPU output. Low GPU-memory headroom selects CPU instead. Brightness-mask preparation, projection, optical flow and cursor composition remain CPU-based. GPU arithmetic may differ from CPU by floating-point roundoff; this is not a fully GPU-rendered compositor.
