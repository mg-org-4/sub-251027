# ComfyUI-Darkroom

Professional color grading and film emulation suite for ComfyUI — 36 nodes, 161 film stocks, 102 lens profiles, zero API costs.

The most complete color toolset in the ComfyUI ecosystem. From physics-based film emulation to DaVinci Resolve-level color grading, Camera Raw processing, optical simulation, LUT export, and ACES color management — everything runs locally with no external dependencies.

## Nodes

### Film Emulation (6 nodes)

| Node | Description |
|------|-------------|
| **Film Stock (Color)** | 111 color film stocks with per-channel H&D characteristic curves. Kodak Portra, Ektar, Fuji Velvia, Cinestill, Polaroid, and more. Capture One curve data integration. |
| **Film Stock (B&W)** | 50 B&W stocks with real spectral sensitivity coefficients. Ilford HP5+, Kodak Tri-X, T-MAX, with pushed variants. |
| **Film Grain** | Multi-octave luminance-dependent grain. ISO-scaled, resolution-aware, blue-channel emphasis like real film. |
| **Halation** | Physics-based light bounce from film base. Screen-blended highlight glow with disk blur. |
| **Print Stock** | Photographic paper simulation — the negative-to-print chain. |
| **Cross Process** | E-6 in C-41 and C-41 in E-6 cross-processing color shifts. |

### Camera Raw Tools (9 nodes)

| Node | Description |
|------|-------------|
| **White Balance** | Color temperature (Kelvin) and tint adjustment via Planckian locus approximation. |
| **Exposure & Tone** | EV-stop exposure, S-curve contrast, parametric shadows/highlights/whites/blacks. |
| **HSL Selective** | Per-hue adjustments to hue, saturation, and luminance across 8 color bands with smooth feathering. |
| **Clarity / Texture / Dehaze** | Local contrast enhancement, surface texture detail, and atmospheric haze removal. |
| **Vibrance** | Intelligent saturation with skin tone protection — boosts muted colors, protects already-saturated areas. |
| **Sharpening Pro** | Advanced unsharp mask with edge-aware masking and radius/amount/threshold control. |
| **Noise Reduction** | Multi-pass guided filter with 7 presets — from light denoise to heavy smoothing. |
| **Skin Tone Uniformity** | Mask-weighted mean pull for even skin tones. 6 skin type presets. Preserves texture. |
| **Color Qualifier** | Color range analysis and isolation with 19 action presets combining selection + correction. |

### Color Grading (9 nodes)

| Node | Description |
|------|-------------|
| **Tone Curve** | 5-point cubic spline per channel (PchipInterpolator — monotonic, no overshoot). 11 presets: S-curves, Faded Blacks, Matte Film, Cross-over pushes. |
| **Lift Gamma Gain** | DaVinci Resolve primary corrector. Per-channel R/G/B + Master for Lift, Gamma, Gain, and Offset — 16 precision sliders. |
| **Log Wheels** | Resolve Log-mode grading. Soft Gaussian zone masks in log2-encoded luminance space. Hue angle + saturation + density per zone. 7 presets. |
| **3-Way Color Balance** | Preset-first creative color tinting. Shadow/midtone/highlight zones with hue + intensity. 15 looks — Orange & Teal, Vintage Warm, Moonlight Blue, Bleach Bypass, and more. |
| **Hue vs Hue** | Remap specific hue ranges to different hues. 8 bands with feathering. 9 presets — skin tone correction, sky shifts, autumn warmth. |
| **Hue vs Sat** | Adjust saturation per hue range. 8 bands. 8 presets — pop blues, mute greens, teal & orange pop. |
| **Lum vs Sat** | Adjust saturation based on luminance. 5 zones. 7 presets — film look (desat highlights), punch midtones, bleach bypass. |
| **Sat vs Sat** | Adjust saturation based on existing saturation level. Compress oversaturated areas, boost muted tones. 4 zones, 6 presets. |
| **Color Warper** | 2D hue + saturation region warping with multi-region presets. 9 presets — Orange & Teal push, skin cleanup, sunset enhance. Manual mode for single-region custom work. |

### Lens & Optics (5 nodes)

| Node | Description |
|------|-------------|
| **Chromatic Aberration** | Lateral CA simulation/correction with per-channel shift. |
| **Vignette** | Optical vignette with shape, midpoint, and falloff control. |
| **Lens Distortion** | Brown-Conrady barrel/pincushion distortion model. |
| **Perspective Correct** | Keystone and trapezoid correction for architectural shots. |
| **Lens Profile** | All-in-one lens correction — distortion + CA + vignette from 102 real lens models (Canon, Nikon, Sony, Zeiss, Leica, vintage). |

### RAW Pipeline (2 nodes)

| Node | Description |
|------|-------------|
| **RAW Load** | Decodes camera RAW files (.cr3, .nef, .arw, .raf, .dng, .rw2, .orf, .pef, .x3f, .iiq, and more) via rawpy/LibRaw. Exposes demosaic algorithm, white balance, highlight mode, output colorspace, linear-scene vs sRGB-display output, and a **Camera Look** profile selector (see below). Outputs an IMAGE and a `RAW_METADATA` sidecar. |
| **RAW Metadata Split** | Splits `RAW_METADATA` into 15 typed primitives: camera make/model, lens make/model, ISO, aperture, shutter, focal length, datetime, sensor type, resolution, Fuji film sim, and more. Wire any primitive directly into a text node or downstream tool. |

#### Camera Look Profiles — `ComfyUI/models/camera_profiles/`

RAW Load can apply Adobe `.dcp` profiles — per-body color calibration (`Adobe Standard`) or creative looks (`Camera Standard`, `Camera Landscape`, Fuji film sims, etc.). The `camera_look` dropdown on the node is a union of every `.dcp` found across these locations, in priority order:

1. **`ComfyUI/models/camera_profiles/<Make Model>/`** — drop user-installed packs here. Respects `extra_model_paths.yaml`. Example: `ComfyUI/models/camera_profiles/Fujifilm X-T5/Fujifilm X-T5 Camera VELVIA.dcp`.
2. **`ComfyUI-Darkroom/data/dcp_looks/`** — reserved for profiles bundled with the pack.
3. **Adobe install paths** (auto-discovered if Camera Raw / Lightroom is installed on the same machine): `C:/ProgramData/Adobe/CameraRaw/CameraProfiles/Camera/` and the Lightroom Classic resources equivalent.

**What ships with Adobe:** Camera Look profiles for Canon, Nikon, Sony, Panasonic, Olympus, Pentax, and others (Camera Standard / Landscape / Portrait / Faithful / Neutral / Monochrome, etc.). If you have Camera Raw or Lightroom installed, these work with zero setup.

**Fujifilm users:** Adobe does not ship Fuji Camera Look profiles as .dcp files (Fuji's in-camera film sims are baked into the Camera Raw engine binary, not exposed on disk). To get Velvia / Provia / Astia / Classic Chrome / Eterna / Acros / Pro Neg / Classic Negative / Nostalgic Neg / Reala ACE as DCPs, install a third-party pack (Stuart Sowerby maintains a widely-used one) into `ComfyUI/models/camera_profiles/Fujifilm <body>/`.

If the selected Camera Look isn't available for the detected body, the node silently falls back to Adobe Standard and logs a console warning.

### Pipeline — LUT & Color Management (7 nodes)

| Node | Description |
|------|-------------|
| **LUT Identity Generator** | Outputs a neutral identity lattice image. Feed into LUT Bake Inject to grade your photo and bake a .cube at the same time. Sizes: 17, 33, 65. |
| **LUT Bake Inject** | Pairs your photo with the identity lattice as a 2-image batch. The grading chain then processes both with identical settings — no node duplication. |
| **LUT Bake Extract** | Splits the batch back out after the grading chain: graded photo to preview, graded lattice to LUT Export. |
| **LUT Export (.cube)** | Bakes any Darkroom processing chain into a standard .cube 3D LUT file. Works in DaVinci Resolve, Premiere Pro, Photoshop, Capture One, FCPX — any tool that supports 3D LUTs. |
| **LUT Apply (.cube)** | Loads and applies any .cube 3D LUT with trilinear interpolation. Import looks from DaVinci Resolve, download creative LUTs, or reuse exported Darkroom grades. Strength slider for blending. |
| **Color Space Transform** | Convert between sRGB, Linear sRGB, ACEScg, ACEScct, Rec.2020, and DCI-P3. The only ACES-aware color management in ComfyUI. Soft gamut compression option. |
| **ACES Tonemap** | Industry-standard tonemapping: ACES Filmic, ACES Fitted (Hill), AgX (Blender), Reinhard, Filmic (Uncharted 2). Exposure bias, ACES gamut conversion, white point control. |

## LUT Bake Workflow — grade your photo and export a .cube in one pass

Building a LUT in Darkroom used to mean running the grading chain twice — once on your photo, once on an identity lattice — with settings duplicated across two parallel chains. That's tedious and error-prone. The **LUT Bake Inject / Extract** pair fixes that: one chain, one set of settings, your photo and the LUT come out the other side together.

**How it works:** Inject pads your photo and the identity lattice to a shared canvas and stacks them as a 2-image batch. Every Darkroom color node iterates the batch dimension and applies the exact same transform to both images. After the chain, Extract splits the batch back into the graded photo and the processed lattice — the lattice goes to LUT Export.

**Wiring:**

```
Load Image ──► photo ─────┐
                          ├─► LUT Bake Inject ─► [grading chain] ─► LUT Bake Extract ─► graded_photo  ──► Preview
LUT Identity ─► lattice ──┤                                                        ├─► graded_lattice ─► LUT Export
                          └─► lut_size ─────────────────────────────────────────── └─► lut_size       ─┘
```

- Connect **LUT Identity Generator → identity_lattice** input of **LUT Bake Inject**.
- Connect your **Load Image → photo** input of **LUT Bake Inject**.
- Run any **color-only** Darkroom nodes between Inject and Extract — Tone Curve, Lift Gamma Gain, HSL Selective, Film Stock, Hue vs X, Color Warper, etc.
- **LUT Bake Extract** gives you three outputs: `graded_photo` (to Preview / Save Image), `graded_lattice` (to LUT Export's `processed_lattice`), and `lut_size` (to LUT Export's `lut_size`).

### Color-only rule — what can go in the chain

A 3D LUT is a per-pixel color lookup. It has no idea about neighboring pixels. So only nodes that transform each pixel independently can be baked:

**Allowed in the bake chain:**
Film Stock (Color), Film Stock (B&W), Print Stock, Cross Process, White Balance, Exposure & Tone, HSL Selective, Vibrance, Tone Curve, Lift Gamma Gain, Log Wheels, 3-Way Color Balance, Hue vs Hue, Hue vs Sat, Lum vs Sat, Sat vs Sat, Color Warper, Color Space Transform, ACES Tonemap, LUT Apply.

**NOT allowed in the bake chain** (they use pixel neighborhoods and will corrupt the lattice):
Film Grain, Halation, Clarity / Texture / Dehaze, Sharpening Pro, Noise Reduction, Skin Tone Uniformity, Color Qualifier (partial — uses local masks), Chromatic Aberration, Vignette, Lens Distortion, Perspective Correct, Lens Profile.

If you want spatial effects on your final image, apply them to `graded_photo` **after** Extract, not inside the bake chain.

### Example workflow

A ready-to-use example is in [`workflows/lut_bake_and_apply.json`](workflows/lut_bake_and_apply.json). Drag it into ComfyUI, load a photo, and press Queue Prompt — you'll get a graded preview and a `.cube` file in `output/luts/`.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jeremieLouvaert/ComfyUI-Darkroom.git
pip install -r ComfyUI-Darkroom/requirements.txt
```

Restart ComfyUI. All 38 nodes appear under **AKURATE/Darkroom/** with subcategories: Film, Raw, Grading, Lens, Pipeline, RAW.

### Dependencies

- **scipy** (>= 1.10.0) — spline interpolation, Gaussian filters, FFT convolution
- **opensimplex** (>= 0.4) — high-quality simplex noise for film grain

No API keys. No GPU required. Pure numpy/scipy computation.

## Architecture

All processing happens in **linear light** (sRGB gamma removed before processing, reapplied after). Every node supports:

- **Strength slider** (0-1) — non-destructive blending with original
- **Batch processing** — handles ComfyUI's multi-image batches
- **Preset + override** — presets provide instant results, sliders fine-tune

Film stock data sourced from Capture One Film Styles (586 .costyle files parsed) and published Kodak/Fuji/Ilford technical data sheets. Lens profiles measured from real optical characteristics.

## License

MIT

## Author

Jeremie Louvaert — [jeremielouvaert.com](https://jeremielouvaert.com)
