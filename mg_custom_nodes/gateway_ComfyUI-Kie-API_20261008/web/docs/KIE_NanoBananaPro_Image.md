# KIE Nano Banana Pro (Image)

Generate high-quality images using KIE’s Nano Banana Pro model.

This node supports optional reference images and exposes major generation
controls directly in ComfyUI.

Maximum reference image inputs: **8**.

---

## Inputs

- **Prompt**  
  Text description of the image to generate.

- **Images (optional)**  
  Up to 8 reference images used for guidance.

- **Aspect Ratio**  
  Output image aspect ratio (e.g. 1:1, 16:9).

- **Resolution**  
  1K, 2K, or 4K output resolution.

- **Output Format**  
  PNG or JPG.

- **Poll Interval**  
  How often the node checks task status (seconds).

- **Timeout**  
  Maximum wait time before failing (seconds).

- **Log**  
  Enable progress output in the console.

---

## Outputs

- **IMAGE**  
  ComfyUI image tensor (BHWC, float32, range 0–1).

---

## Notes

- Credits are consumed per request.
- If the API is under heavy load, generation may take longer.
- If no images are connected, `image_input` is sent as an empty list.

## Debugging
You can always visit https://kie.ai/logs to see the progress, why something may of failed. If their is a timeout you can usually see your final image/video here as well
