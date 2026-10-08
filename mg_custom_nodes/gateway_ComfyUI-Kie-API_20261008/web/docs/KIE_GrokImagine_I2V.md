# KIE Grok Imagine (I2V)

Generate a short video clip from either one uploaded external image or a Grok-generated image reference.

---

## Inputs

- **prompt**
  Optional motion prompt. Maximum length: 5000 characters.

- **images**
  Optional external source image batch. If connected, only the first image is uploaded and used.

- **task_id**
  Optional Grok image-generation task id. Use this instead of `images` when you want to animate one of Grok's generated images.

- **index**
  Which image to use from the Grok `task_id` source. Range: 0 to 5.

- **mode**
  Choose from fun, normal, or spicy. Spicy is only supported for `task_id` sources.

- **duration**
  Output length in seconds: 6, 10, or 15.

- **resolution**
  Choose 480p or 720p.

- **log**
  Enable console logging.

---

## Output

- **VIDEO**
  ComfyUI VIDEO output compatible with the SaveVideo node.

---

## Source Modes

- **External image mode**
  Connect `images` and leave `task_id` empty.

- **Grok image reference mode**
  Set `task_id` and `index`, and leave `images` disconnected.

---

## Notes

- Provide exactly one source method: `images` or `task_id`.
- External image mode supports one image only.
- If multiple ComfyUI images are connected, the first image is used.
- If `mode=spicy` is selected with an external image, the node logs a warning and sends `normal`.
- The KIE callback example for this endpoint appears to show a `.jpg` URL in `resultUrls`; this node treats the endpoint as video and downloads the first result as VIDEO output.
