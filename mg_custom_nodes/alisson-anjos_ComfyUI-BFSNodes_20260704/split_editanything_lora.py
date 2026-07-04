#!/usr/bin/env python3
"""Split an LTXV Edit Anything LoRA into standard LoRA and sidecar module.

Usage:
    python split_editanything_lora.py path/to/edit_anything.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


SIDECAR_PREFIXES = (
    "role_embedding.",
    "_role_embedding.",
    "ref_adaln_proj.",
    "diffusion_model.role_embedding.",
    "diffusion_model.adaln_single.linear.",
    "diffusion_model.audio_adaln_single.linear.",
)


def split_lora(input_path: Path, overwrite: bool = False) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors file, got: {input_path}")

    standard_path = input_path.with_name(f"{input_path.stem}.standard.safetensors")
    module_path = input_path.with_name(
        f"{input_path.stem}.editanything_module.safetensors"
    )

    if not overwrite:
        for path in (standard_path, module_path):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists. Re-run with --overwrite to replace it."
                )

    sd = load_file(str(input_path))
    standard = {}
    module = {}

    for key, value in sd.items():
        if key.startswith(SIDECAR_PREFIXES):
            module[key] = value
        else:
            standard[key] = value

    save_file(standard, str(standard_path))
    save_file(module, str(module_path))

    print(f"Input:    {input_path}")
    print(f"Standard: {standard_path} ({len(standard)} keys)")
    print(f"Module:   {module_path} ({len(module)} keys)")

    if not module:
        print("Warning: module file is empty; no Edit Anything sidecar keys matched.")

    return standard_path, module_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an LTXV Edit Anything LoRA into a clean standard LoRA and an Edit Anything module."
    )
    parser.add_argument("input", type=Path, help="Path to the original .safetensors LoRA.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing .standard/.editanything_module outputs.",
    )
    args = parser.parse_args()
    split_lora(args.input, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
