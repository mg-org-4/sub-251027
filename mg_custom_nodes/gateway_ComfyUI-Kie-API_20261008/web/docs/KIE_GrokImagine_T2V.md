# KIE Grok Imagine (T2V)

Generate a short video clip from a text prompt using the Grok Imagine text-to-video model.

---

## Inputs

- **prompt**
  Text prompt describing the desired video motion. Maximum length: 5000 characters.

- **aspect_ratio**
  Choose from 2:3, 3:2, 1:1, 9:16, or 16:9.

- **mode**
  Choose from fun, normal, or spicy.

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

## Example

Prompt:
"A couple of doors open to the right one by one randomly and stay open, revealing tiny rooms with little people living inside."

Mode: normal
Aspect ratio: 16:9
Duration: 6
Resolution: 480p

---

## Notes

- All enum inputs are validated before the task is submitted.
- This node uses internal defaults for polling, retries, and timeout handling.
- The KIE callback example for this endpoint appears to show a `.jpg` URL in `resultUrls`; this node treats the endpoint as video and downloads the first result as VIDEO output.
