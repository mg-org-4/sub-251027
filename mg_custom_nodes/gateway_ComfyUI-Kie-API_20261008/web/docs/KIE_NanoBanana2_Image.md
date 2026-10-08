# KIE Nano Banana 2 (Image)

Generate images using KIE's Nano Banana 2 model.

This node supports optional reference images, Google web-search grounding,
and the same async task flow as the other KIE image nodes.

Maximum reference image inputs: **14**.

---

## Inputs

- **Prompt**  
  Text description of the image to generate.

- **Images (optional)**  
  Up to 14 reference images used for guidance.

- **Google Search**  
  Enable Google web-search grounding for real-time information.

- **Aspect Ratio**  
  Output image aspect ratio.

- **Resolution**  
  `1K`, `2K`, or `4K`.

- **Output Format**  
  `jpg` or `png`.

- **Poll Interval**  
  How often the node checks task status (seconds).

- **Timeout**  
  Maximum wait time before failing (seconds).

- **Log**  
  Enable progress output in the console.

---

## Outputs

- **IMAGE**  
  ComfyUI image tensor (BHWC, float32, range 0-1).

---

## Notes

- Credits are consumed per request.
- If no images are connected, `image_input` is sent as an empty list.
