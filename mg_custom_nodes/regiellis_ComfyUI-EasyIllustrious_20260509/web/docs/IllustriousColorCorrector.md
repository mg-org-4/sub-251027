# IllustriousColorCorrector

A node for advanced color correction and grading in Illustrious workflows.

## What does it do?

- Adjusts color balance, brightness, contrast, and saturation.
- Can apply LUTs (Look-Up Tables) or custom color curves.
- Designed for anime, illustration, and stylized outputs.

## Inputs

- **image**: The image to correct.
- **preset**: (Optional) Predefined color grading style (anime, pastel, etc.).
- **brightness/contrast/saturation**: Manual adjustment sliders.
- **lut**: (Optional) LUT file or tensor for advanced grading.
- **curve**: (Optional) Custom color curve.
- **strength**: How strongly to apply the correction (0 = none, 1 = full).

## Outputs

- Color-corrected image.

## Artist Tips

- Use presets for quick anime or illustration looks.
- Fine-tune with manual sliders for unique styles.
- Combine with LUTs for cinematic or dramatic effects.
- Use low strength for subtle polish, high for bold changes.

## Example Workflow

1. Connect your generated image.
2. Select a preset or adjust sliders.
3. (Optional) Add a LUT or custom curve.
4. Output is ready for sharing or further editing.

---
For more on color workflows, see the Color Suite and grading guides.
