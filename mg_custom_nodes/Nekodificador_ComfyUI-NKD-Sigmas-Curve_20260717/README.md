
# ComfyUI NKD Sigmas Curve

A ComfyUI custom node that replaces trial-and-error sigma tuning with a visual, interactive spline editor. Design your diffusion noise schedule exactly the way you want it, then plug it straight into any sampler.

## Why this exists

It’s all about control. Standard schedulers (Karras, exponential, etc.) give you a fixed curve shape (nothing wrong with that) but once you unlock the power of custom sigmas, you can decide exactly how you want to denoise the image. That gives you fine-grained control over composition, details, and a bunch of nerdy stuff.

https://github.com/user-attachments/assets/281fa043-0900-4e7b-883d-1018953b01e0

This is all with a fixed seed. As you can see, specially in the las 2 generations. Tuning the sigma curve lets you nail the shapes and details at just the right moment during generation. For instance, I use it to swap out a bare chest for a T-shirt on the fly.


## How it works / How to use it

- I **strongly, highly, super recommend using it alongside the [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF.git)** node pack (and joining the bongmath cult), but technically you could plug it into any sigmas input, like in a _CustomSampler_. 
- The node overrides the scheduler and steps, so set the **Ksampler to 1.0 denoise and control these from the Sigmas Curves node instead**.
- If you know nothing about sigmas, treat the _max_sigma_ value as your new "denoise" setting (kind of).
- The curve is your new "scheduler" (you're basically drawing it yourself instead of picking one from a dropdown).
- You can choose between linear curve or b-spline type. Up to you.

## Features

- **Interactive canvas widget** embedded directly in the ComfyUI node, no external tools needed
- **Click** to add control points, **drag** to reposition, **Shift+click** to remove
- **Two interpolation modes:**
  - **Smooth** — B-spline with tension weights
  - **Linear** — Piecewise linear between control points
- Outputs a standard `SIGMAS` tensor compatible with **all ComfyUI samplers**
- No extra Python dependencies beyond what ComfyUI already includes

## Updates

### v1.3.0 — Reference overlay

You can now connect any SIGMAS output (from a scheduler, another curve node, anything) to the new **reference_sigmas** input. Once you run the node, a ghost curve appears on the canvas so you can compare your design against the reference at a glance.

- **Show / Hide** — toggle the reference overlay on and off
- **Match** — copy the reference shape into your curve as editable control points, so you can use it as a starting point and tweak from there
- The overlay appears automatically as soon as the node executes with a reference connected
- Point tooltips now show the exact step and σ value while hovering or dragging
- `max_sigma` widget precision increased to 3 decimal places


https://github.com/user-attachments/assets/b91c680c-544b-4720-a5a1-1b43a5a807f6

### v1.2.0 — Snap to steps & progress dot

Now it shows the progress in real time, so you can make better decisions 🫡

https://github.com/user-attachments/assets/42c3c7af-4d89-43c1-bb6d-d623547c8e5d


## Installation

### Via ComfyUI Manager *(recommended)*

Search for **NKD Sigmas Curve** in the ComfyUI Manager and install with one click.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nekodificador/ComfyUI-NKD-Sigmas-Curve
```

No additional Python dependencies required. Restart ComfyUI after installing.

> **Note:** The JavaScript widget (`web/nkd_sigma_curve.js`) is pre-built and ready to use. If you want to modify the Vue source, see [Development](#development) below.

## Requirements

- ComfyUI (V3 API / Nodes 2.0 compatible)
- Python 3.10 or higher
- PyTorch (included with ComfyUI)

## Development

The widget is written in **Vue 3 + TypeScript** and bundled with Vite.

```bash
cd ComfyUI/custom_nodes/nkd_sigma_curve
npm install
npm run build   # outputs to web/nkd_sigma_curve.js
npm run dev     # watch mode
```

## Inspired by
[Custom Sigma Editor](https://github.com/JoeNavark/comfyui_custom_sigma_editor.git)

## License

MIT, use it, modify it, share it freely.
