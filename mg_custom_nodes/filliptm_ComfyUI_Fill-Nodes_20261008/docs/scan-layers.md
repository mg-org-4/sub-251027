# Scan FX layers

The **Layers** tab controls the digital back plates and cursor-window overlap. Existing workflows retain four cobalt plates, normal compositing, full reveal opacity, and no fades.

- **Stack count**: 0–8 plates. Spacing multiplies the original depth-relative offset; X/Y set direction, rotation fans each successive plate, and opacity affects plates and their outlines without fading the video.
- **Window order**: newest on top, a preferred effect on top, or seeded random ordering on Snare (Envelope 2 rising through 0.5). Random ranks hold between hits. They do not shuffle every frame.
- **Window blend**: normal, screen, or add. This is separate from Finish's glow blend mode. Per-effect opacity multiplies Reveal strength and the existing audio accent.
- **Fade in/out**: seconds for revealed content, not cursor graphics or borders. Fade-in starts when dragging begins; fade-out finishes on the gesture's last frame.

Drag Kick, Snare, or Hat onto spacing, X/Y spread, rotation, stack opacity, or any reveal opacity to assign an envelope. Count, palette, and ordering remain discrete settings. A useful starting mapping is Kick → stack spacing, minimum 0.6, maximum 2, smoothing 0.08 seconds.

The compact parameter panel uses two columns when space permits, `?` help, numeric ranges in control tooltips, and individual reset buttons. Mapping editors expand across both columns. Render-value meters appear only for assigned parameters.

**Live demo → Layers** approximates the stack and overlap on a synthetic scene. **All** includes the other effects. Actual envelopes, geometry, timing, and occlusion must still be checked in **Last render → Final**. These effects reuse existing images and maps; they do not run additional inference.
