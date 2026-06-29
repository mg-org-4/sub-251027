# Custom ICC profiles

Drop any `.icc` or `.icm` CMYK profile into this folder and it will appear
in the CMYK nodes' dropdowns after a ComfyUI restart.

Darkroom also auto-discovers the OS's colour-profile store, so on Windows
you already get FOGRA39, FOGRA27, FOGRA29 (uncoated), GRACoL 2006, SWOP v2,
SNAP newsprint, JapanColor, etc. without needing to copy them here.

## Common free sources

- **ECI** (European Color Initiative) -- FOGRA39L / FOGRA51 / PSO coated v3 /
  PSO uncoated v3 / ISO Newspaper 26v4, all free for commercial use:
  https://www.eci.org/doku.php?id=en:downloads
- **Adobe** ships US coated / uncoated / SWOP in Creative Cloud; they are
  also in Windows at `C:\Windows\System32\spool\drivers\color\`.
- **ICC** (International Color Consortium) -- registered public profiles:
  https://www.color.org/registry/index.xalter

Only drop CMYK output profiles here. RGB display / scanner profiles are
ignored by the discovery pass.
