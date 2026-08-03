"""
File    : convmodel.py
Purpose : Script to convert the original Z-Image model to a format compatible with ComfyUI
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jun 26, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
     ComfyUI nodes designed to power the "Z-Image/Z-Image Turbo" models.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import argparse
#import safetensors.torch
#import torch
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # modules that are not available in the standard library are imported here
    import numpy as np
    import ml_dtypes
    #from safetensors       import safe_open
    #from safetensors.numpy import save as safetensors_bytes
    from safetensors.numpy import save_file as save_safetensors, load_file as load_safetensors


# Nota: Se asume que `fatal_error` está definido en tu módulo de utilidades.
# Si no lo tienes, puedes reemplazarlo por `sys.exit()` con un mensaje formateado.
# from your_project.utils import fatal_error

# get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ANSI escape codes for colored terminal output
RED      = '\033[91m'
DKRED    = '\033[31m'
YELLOW   = '\033[93m'
DKYELLOW = '\033[33m'
GREEN    = '\033[92m'
CYAN     = '\033[96m'
DKGRAY   = '\033[90m'
RESET    = '\033[0m'

#============================= ERROR MESSAGES ==============================#

def disable_colors():
    global RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET
    RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET = "", "", "", "", "", "", "", ""


def info(message: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an informational message to the error stream.
    """
    print(f"{" "*padding}{CYAN}\u24d8 {message}{RESET}", file=file)


def warning(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a warning message to the standard error stream.
    """
    print(f"{" "*padding}{CYAN}[{YELLOW}WARNING{CYAN}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an error message to the standard error stream.
    """
    print(f"{" "*padding}{DKRED}[{RED}ERROR!{DKRED}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def fatal_error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a fatal error message to the standard error stream and exits with status code 1.
    """
    error(message, *info_messages, padding=padding, file=file)
    sys.exit(1)


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args=None, parent_script=None):
    """
    Main entry point for the script.
    Args:
        args          (optional): List of arguments to parse. Default is None, which will use the command line arguments.
        parent_script (optional): The name of the calling script if any. Used for customizing help output.
    """
    prog = None
    if parent_script:
        prog = parent_script + " " + os.path.basename(__file__).split('.')

    parser = argparse.ArgumentParser(
        prog            = prog,
        description     = "Convert and merge original diffusion model weights to ComfyUI format.",
        formatter_class = argparse.RawTextHelpFormatter,
        )
    parser.add_argument('-o', '--output', default='output.safetensors', help="Output safetensors file path.")
    parser.add_argument('input_files', nargs='+', metavar='INPUT', help="One or more input safetensors files to process.")
    parsed_args = parser.parse_args(args=args)

    # Determine target dtype from the output filename
    out_path = parsed_args.output
    cast_to = None
    if "fp16" in out_path:
        cast_to = np.float16
    elif "bf16" in out_path:
        cast_to = ml_dtypes.bfloat16

    # Key replacement mapping for ComfyUI compatibility
    replace_keys = {
        "all_final_layer.2-1.": "final_layer.",
        "all_x_embedder.2-1.": "x_embedder.",
        ".attention.to_out.0.bias": ".attention.out.bias",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.to_out.0.weight": ".attention.out.weight"
    }

    out_sd = {}

    # Process each input safetensors file
    for input_file in parsed_args.input_files:
        if not os.path.exists(input_file):
            fatal_error(f"Input file not found: \"{input_file}\"")

        try:
            sd = load_safetensors(input_file)
        except Exception as e:
            fatal_error(f"Failed to load safetensors file \"{input_file}\": {e}")

        # Buffer for stacking attention projections
        cc = []
        for k, w in sd.items():
            # Apply dtype casting if requested
            if cast_to is not None:
                w = w.astype(cast_to)

            k_out = k

            # Skip keys that are not needed or cause conflicts
            if k_out.endswith(".attention.to_out.0.bias"):
                continue

            # Handle attention projection stacking logic
            if k_out.endswith(".attention.to_k.weight"):
                cc = [w]
                continue
            if k_out.endswith(".attention.to_q.weight"):
                cc = [w] + cc
                continue
            if k_out.endswith(".attention.to_v.weight"):
                cc = cc + [w]
                w = np.concatenate(cc, axis=0)
                k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")
                cc = []  # Reset buffer to prevent cross-contamination between layers

            # Apply key transformations
            for old_key, new_key in replace_keys.items():
                k_out = k_out.replace(old_key, new_key)

            out_sd[k_out] = w

    # Save the merged and converted weights
    try:
        save_safetensors(out_sd, out_path)
        print(f"Successfully saved converted weights to \"{out_path}\" ({len(out_sd)} tensors).")
    except Exception as e:
        fatal_error(f"Failed to save output file \"{out_path}\": {e}")

if __name__ == "__main__":
    main()

