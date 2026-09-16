# Parallax Studio

FL Layered Parallax keeps width, height and frames as standard widgets. Its camera editor writes the existing motion/framing inputs, so saved workflows and connected inputs remain compatible.

- **Scene:** small first-frame previews of the actual layers. Drag to set camera travel; Play and the scrubber preview a six-second camera cycle. Center shows neutral framing without changing the render settings.
- **Depth order:** near-to-far layer diagram. Edit individual depths in the connected layer nodes or extraction review.
- **Camera:** Gentle, Reveal, Punchy and Still presets, motion curves, XY control, horizontal/vertical travel and push/pull. Presets change motion only.
- **Framing:** background distance, coverage, cutout cover/contain and render device. Contain preserves mismatched cutout aspect ratios. Background edges use clamped source pixels, not generated hidden scenery.
- **Relief:** Off, Background, or Background + Artwork, with strength, anchor depth, inversion and smoothing. Text layers from the poster stack remain flat. Zero strength preserves the previous renderer. The small GPU preview responds to strength/anchor/inversion; smoothing is applied on Run. Without WebGL, the preview remains flat.

Connect the stack to **FL Parallax Depth Sources**, then an image-batch depth estimator, then `depth_maps` on the compositor. Analysis sources use neutral gray behind cutout alpha and preserve stack order: background, then cutouts. A single depth map is accepted for background-only relief. This path supports still-image layers only. The workflow uses the installed Depth Anything V2 FP32 model at a 518-pixel analysis edge; the inference node is separate so camera and relief adjustments can reuse ComfyUI's cached maps.

Relief is a bounded inverse-depth warp inside each plate, not a mesh reconstruction. Color and premultiplied alpha share the same sampling grid. Plate draw order remains unchanged, and the Depth order output still visualizes flat layer ordering, not inferred per-pixel depth. Keep relief subtle near silhouettes, reflective objects, and narrow depth gaps. Depth on isolated cutouts can be ambiguous, and unseen surfaces are not reconstructed.

Before source previews are available, the editor displays labeled demo geometry. Landscape, typography and product-shape demos are preview-only. Actual previews use the same plane projection as the renderer, but reduced-resolution images, browser interpolation and frozen source frames mean they are not final-quality video. Upstream changes require Run to refresh the layers. Connected camera inputs are read-only in the editor and show their last rendered values.

Preview playback is limited to 20 drawn frames per second, starts paused, and stops scheduling animation when the node is offscreen or the browser tab is hidden. Source previews are at most 256 pixels on their longest edge and are saved under `output/Parallax/previews`.

The renderer computes a static scene's locked comparison once and shares that frame across the output batch. Animated input plates retain per-frame locked comparisons. Only the representative depth-preview frame is composited. No persistent GPU tensor cache is added.

## Different media

The Image Layer Planner follows the visible source medium. `artwork_type` guides grouping; it does not restyle the image. Auto mode adds text layers only when lettering is visible, retains actual environments, and can make one correction pass for an invalid or over-budget plan. Manual mode remains available for exact targets.

Use subtle travel for portraits, photography, reflective products and fine hair. Illustration and collage usually tolerate more separation; landscapes work best as coherent foreground/middle-ground/background groups. Large camera moves need reconstructed hidden regions or oversized source plates. Generated extraction can alter texture, lettering, shadows and transparency; inspect the RGBA layers before exporting.
