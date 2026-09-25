# spectral_film_lut (vendored)

Source: https://github.com/JanLohse/spectral_film_lut
Author: Jan Lohse
License: MIT (see LICENSE in this directory)

This directory is a stripped, vendored copy of `spectral_film_lut` used to
bake pre-generated `.cube` LUTs that encode a full negative->print spectral
simulation for ~35 curated film x paper pairings. The baking pipeline lives
at `tools/bake_spectral_luts.py`; the output .cube files ship in
`data/spectral_luts/` and are consumed by the `DarkroomSpectralFilmStock`
node.

## What was removed from the upstream copy

- PyQt6 GUI (`gui.py`, `gui_objects.py`, `splash_screen.py`,
  `filmstock_selector.py`, `css_theme.py`, `__main__.py`, `resources/`)
- Grain generation module (`grain_generation.py`) -- GUI-dependent; our
  grain pipeline lives in `utils/grain.py` and `nodes/film_grain.py`
- Film loader UI wrapper (`film_loader.py`) -- the baker has its own loader
- Datasheet PDFs (`datasheets/`, ~22 MB reference material)
- Docs site (`docs/`), git history, CI configs, `pyproject.toml`

## What was kept

- Engine core: `film_spectral.py`, `densiometry.py`, `color_space.py`,
  `film_data.py`, `utils.py`, `wratten_filters.py`, `file_formats.py`
- `__init__.py` with the FILM_STOCKS registry
- The per-stock data modules under `negative_film/`, `print_film/`,
  `reversal_film/`, `reversal_print/`, `bw_negative_film/`, `bw_print_film/`

## Notes

The engine uses `numba` for hot loops. ComfyUI's embedded Python ships
NumPy 2.4 which is incompatible with numba's current release, so the baker
installs a no-op numba shim before import -- the code runs in pure Python,
which is slower but fine for an offline batch bake (~0.4s per LUT).

This vendored copy is not imported by the runtime node; it's only touched
by `tools/bake_spectral_luts.py`. Users installing the pack through the
Comfy Registry receive the pre-baked LUTs; they never run the engine
themselves unless they want to regenerate or extend the library.
