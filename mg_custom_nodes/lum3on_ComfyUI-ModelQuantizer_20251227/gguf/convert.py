# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"  # string describing architecture
    shape_fix = False # whether to reshape tensors
    keys_detect = []  # list of lists to match in state dict
    keys_banned = []  # list of keys that should mark model as invalid for conversion
    keys_hiprec = []  # list of keys that need to be kept in fp32 for some reason

    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
        # Additional FLUX detection patterns for different model variants
        ("double_blocks.0.img_attn.qkv.weight",),
        ("single_blocks.0.linear1.weight",),
        ("transformer_blocks.0.norm1.linear.weight",),
        # More flexible patterns for FLUX models
        ("guidance_in.in_layer.weight", "img_in.in_layer.weight"),
        ("time_in.in_layer.weight", "vector_in.in_layer.weight"),
        # Support for diffusion_model prefix patterns (your FLUX model)
        ("double_blocks.0.img_attn.norm.key_norm.scale",),
        ("double_blocks.0.img_attn.norm.query_norm.scale",),
        ("double_blocks.0.img_mlp.0.weight",),
        ("single_blocks.0.modulation.lin.weight",),
        # Additional patterns based on your model's tensor names
        ("double_blocks.0.img_attn.qkv.bias", "double_blocks.0.img_mlp.0.bias"),
        ("double_blocks.0.img_attn.proj.bias", "double_blocks.0.img_mlp.2.weight"),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        )
    ]
    keys_hiprec = [
        ".ff_i.gate.weight" # nn.parameter, can't load from BF16 ver
    ]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data):
        # Create fix file in the same directory as the convert script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, f"fix_5d_tensors_{self.arch}.safetensors")

        # Load existing fix file or create new one
        existing_fsd = {}
        if os.path.isfile(path):
            try:
                from safetensors.torch import load_file
                existing_fsd = load_file(path)
                tqdm.write(f"5D tensor fix file exists, adding to it: {key} {data.shape}")
            except Exception as e:
                tqdm.write(f"Warning: Could not load existing fix file {path}: {e}")
                # Try to remove the corrupted file and start fresh
                try:
                    os.remove(path)
                    tqdm.write(f"Removed corrupted fix file, starting fresh")
                except:
                    pass
                existing_fsd = {}
        else:
            tqdm.write(f"Creating new 5D tensor fix file: {path}")

        # Add the new tensor to the fix data
        existing_fsd[key] = torch.from_numpy(data)
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")

        # Save the updated fix file with better error handling
        try:
            save_file(existing_fsd, path)
        except Exception as e:
            tqdm.write(f"Error saving fix file: {e}")
            # Try alternative approach: save each tensor individually
            try:
                import tempfile
                import uuid
                temp_path = os.path.join(script_dir, f"fix_5d_tensors_{self.arch}_{uuid.uuid4().hex[:8]}.safetensors")
                save_file({key: torch.from_numpy(data)}, temp_path)
                tqdm.write(f"Saved 5D tensor to alternative file: {temp_path}")
            except Exception as e2:
                tqdm.write(f"Failed to save 5D tensor fix: {e2}")
                # As last resort, just continue without saving the fix
                tqdm.write(f"WARNING: 5D tensor {key} could not be saved for fixing!")

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    keys_hiprec = [
        ".modulation" # nn.parameter, can't load from BF16 ver
    ]

    def handle_nd_tensor(self, key, data):
        # Special handling for WAN patch_embedding tensors
        if "patch_embedding.weight" in key:
            # For patch_embedding tensors, we'll reshape them to 4D and store the original shape
            # This allows them to be included in the main GGUF file instead of the fix file
            original_shape = data.shape
            tqdm.write(f"WAN patch_embedding tensor found: {key} {original_shape}")

            # Reshape to 4D by combining some dimensions
            # Original: [5120, 16, 1, 2, 2] -> [5120, 16, 1, 4] (combine last two dims)
            if len(original_shape) == 5:
                new_shape = (original_shape[0], original_shape[1], original_shape[2],
                            original_shape[3] * original_shape[4])
                reshaped_data = data.reshape(new_shape)
                tqdm.write(f"Reshaped {key} from {original_shape} to {new_shape}")

                # Return the reshaped data so it can be processed normally
                # The caller will need to handle this return value
                return reshaped_data, original_shape

        # For other 5D tensors, use the parent class method
        return super().handle_nd_tensor(key, data)

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = [
        "scale_shift_table" # nn.parameter, can't load from BF16 base quant
    ]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

# The architectures are checked in order and the first successful match terminates the search.
arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1]

def is_model_arch(model, state_dict):
    # check if model is correct
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break

    if model_arch is None:
        # Enhanced debugging for unknown architectures
        print("ERROR: Unknown model architecture! Debugging info:")
        print(f"Total tensors: {len(state_dict)}")

        # Show some sample keys to help identify the model
        sample_keys = list(state_dict.keys())[:10]
        print("Sample tensor names:")
        for i, key in enumerate(sample_keys):
            print(f"   {i+1:2d}. {key}")

        # Check for common patterns
        patterns = ["transformer_blocks", "double_blocks", "single_blocks", "img_attn", "txt_attn"]
        print("Pattern analysis:")
        for pattern in patterns:
            matches = [key for key in state_dict.keys() if pattern in key]
            print(f"   {pattern}: {len(matches)} matches")

        # Try to suggest which architecture might be closest
        print("SUGGESTION: Check the tensor names above and update the detection keys in convert.py")

        assert False, "Unknown model architecture! See debugging info above."

    return model_arch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output unet gguf file.")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def strip_prefix(state_dict):
    # Strip prefixes from diffusion model tensors while preserving other essential tensors
    prefix = None
    # Check for various prefixes in order of specificity
    for pfx in ["model.diffusion_model.", "diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    sd = {}
    for k, v in state_dict.items():
        new_key = k

        # If we found a prefix and this key starts with it, strip the prefix
        if prefix and k.startswith(prefix):
            new_key = k.replace(prefix, "")
        # If we found a prefix but this key doesn't start with it, keep the key as-is
        # This preserves essential tensors like patch_embedding.weight, time_in.*, etc.

        sd[new_key] = v

    return sd

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        state_dict = load_file(path)

    return strip_prefix(state_dict)

def handle_tensors(writer, state_dict, model_arch):
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}")
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        # this is so we don't break torch 2.0.X
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        if old_dtype == torch.bfloat16:
            data_qtype = gguf.GGMLQuantizationType.BF16
        # elif old_dtype == torch.float32:
        #     data_qtype = gguf.GGMLQuantizationType.F32
        else:
            data_qtype = gguf.GGMLQuantizationType.F16

        # The max no. of dimensions that can be handled by the quantization code is 4
        if len(data.shape) > MAX_TENSOR_DIMS:
            try:
                result = model_arch.handle_nd_tensor(key, data)
                # Check if the method returned reshaped data (for WAN patch_embedding)
                if isinstance(result, tuple) and len(result) == 2:
                    reshaped_data, original_shape = result
                    # Store the original shape for reconstruction
                    writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in original_shape))
                    # Continue processing with the reshaped data
                    data = reshaped_data
                    tqdm.write(f"Processing reshaped tensor {key}: {data.shape}")
                else:
                    # Normal 5D tensor handling - skip this tensor
                    continue
            except Exception as e:
                tqdm.write(f"Warning: Failed to handle 5D tensor {key}: {e}")
                tqdm.write(f"Skipping tensor {key} with shape {data.shape}")
                continue

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                # one-dimensional tensors should be kept in F32
                # also speeds up inference due to not dequantizing
                data_qtype = gguf.GGMLQuantizationType.F32

            elif n_params <= QUANTIZATION_THRESHOLD:
                # very small tensors
                data_qtype = gguf.GGMLQuantizationType.F32

            elif any(x in key for x in model_arch.keys_hiprec):
                # tensors that require max precision
                data_qtype = gguf.GGMLQuantizationType.F32

        if (model_arch.shape_fix                        # NEVER reshape for models such as flux
            and n_dims > 1                              # Skip one-dimensional tensors
            and n_params >= REARRANGE_THRESHOLD         # Only rearrange tensors meeting the size requirement
            and (n_params / 256).is_integer()           # Rearranging only makes sense if total elements is divisible by 256
            and not (data.shape[-1] / 256).is_integer() # Only need to rearrange if the last dimension is not divisible by 256
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

def convert_file(path, dst_path=None, interact=True, overwrite=False):
    # load & run model detection logic
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f"* Architecture detected from input: {model_arch.arch}")

    # detect & set dtype for output file
    dtypes = [x.dtype for x in state_dict.values()]
    dtypes = {x:dtypes.count(x) for x in set(dtypes)}
    main_dtype = max(dtypes, key=dtypes.get)

    if main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
    # elif main_dtype == torch.float32:
    #     ftype_name = "F32"
    #     ftype_gguf = None
    else:
        ftype_name = "F16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path: # lcpp logic
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input("Output exists enter to continue or ctrl+c to abort!")
        else:
            raise OSError("Output exists and overwriting is disabled!")

    # handle actual file
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    handle_tensors(writer, state_dict, model_arch)
    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fix = os.path.join(script_dir, f"fix_5d_tensors_{model_arch.arch}.safetensors")
    if os.path.isfile(fix):
        print(f"\n### Warning! Fix file found at '{fix}'")
        print(f" you most likely need to run 'fix_5d_tensors.py' after quantization.")

    return dst_path, model_arch

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst)
