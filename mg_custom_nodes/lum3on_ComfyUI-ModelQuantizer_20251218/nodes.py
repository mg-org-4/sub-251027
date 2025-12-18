# nodes.py
# This file contains the implementation of your custom nodes.

import torch
from safetensors.torch import save_file 
from tqdm import tqdm
import os 

# --- Helper Classes (Using original simpler scaling logic as provided by user) ---

class FP8Quantizer:
    """A class to apply FP8 quantization to a state_dict."""
    
    def __init__(self, quant_dtype: str = "float8_e5m2"):
        if not hasattr(torch, quant_dtype):
            raise ValueError(f"Unsupported quant_dtype: {quant_dtype}. PyTorch does not have this attribute.")
        self.quant_dtype = quant_dtype
        self.scale_factors = {} # Not used in current quantize_weights but kept for potential future use

    def quantize_weights(self, weight: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Quantizes a weight tensor to the specified FP8 format using simple scaling."""
        if not weight.is_floating_point():
            return weight # Only quantize floating point tensors
        
        original_device = weight.device
        # Ensure FP8 conversion happens on CUDA if possible, as FP8 types require it
        can_use_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda") if can_use_cuda else torch.device("cpu")
        
        # Warn if trying FP8 on CPU without CUDA
        if not can_use_cuda and "float8" in self.quant_dtype:
             print(f"[FP8Quantizer] Warning: CUDA not available. True {self.quant_dtype} conversion requires CUDA. Attempting on CPU, but results may be unexpected or errors may occur.")
             target_device = torch.device("cpu") # Try on CPU despite warning

        weight_on_target = weight.to(target_device)

        max_val = torch.max(torch.abs(weight_on_target))
        if max_val == 0:
            # For zero tensor, just cast to target dtype on the target device
            target_torch_dtype = getattr(torch, self.quant_dtype)
            return torch.zeros_like(weight_on_target, dtype=target_torch_dtype)
        else:
            # Using the simple scaling from user's provided script
            scale = max_val / 127.0 
            # Clamp scale to avoid division by zero if max_val is extremely small
            scale = torch.max(scale, torch.tensor(1e-12, device=target_device, dtype=weight_on_target.dtype))

        # Quantize: scale, round, unscale (simulated int8 range mapping)
        quantized_weight_simulated = torch.round(weight_on_target / scale * 127.0) / 127.0 * scale
        
        # Final cast to the target FP8 dtype
        target_torch_dtype = getattr(torch, self.quant_dtype)
        quantized_weight = quantized_weight_simulated.to(dtype=target_torch_dtype)
        
        # Return on the device where conversion happened (target_device, ideally CUDA)
        return quantized_weight

    def apply_quantization(self, state_dict: dict) -> dict:
        """Applies direct FP8 quantization to all applicable weights."""
        quantized_state_dict = {}
        eligible_tensors = {name: param for name, param in state_dict.items() if isinstance(param, torch.Tensor) and param.is_floating_point()}
        progress_bar = tqdm(eligible_tensors.items(), desc=f"Quantizing to {self.quant_dtype}", unit="tensor", leave=False)
        
        for name, param in progress_bar:
            # quantize_weights handles device logic now
            quantized_state_dict[name] = self.quantize_weights(param.clone(), name) 
        
        for name, param in state_dict.items():
            if name not in quantized_state_dict:
                quantized_state_dict[name] = param
        return quantized_state_dict

class FP8ScaledQuantizer: 
    """
    Simulated FP8 quantizer using 8-bit scaled float approximation based on user provided script.
    Operations are performed on the device of the input tensors.
    (Used internally by QuantizeModel node for the value simulation step).
    """
    def __init__(self, scaling_strategy: str = "per_tensor"):
        self.scaling_strategy = scaling_strategy
        self.scale_factors = {} # Stores the calculated scales (Python floats or lists)

    def _quantize_fp8_simulated(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Simulate quantization by scaling, clamping to 8-bit range, and dequantizing."""
        # Ensure scale is a tensor on the correct device and dtype
        scale = scale.to(device=tensor.device, dtype=tensor.dtype)
        # Prevent division by zero
        scale = torch.where(scale == 0, torch.tensor(1e-9, device=tensor.device, dtype=tensor.dtype), scale)
        
        # Perform simulation: scale, round, clamp, unscale
        quantized_intermediate = tensor / scale * 127.0
        quantized = torch.round(quantized_intermediate).clamp_(-127.0, 127.0) 
        dequantized = quantized / 127.0 * scale
        return dequantized

    def quantize_weights(self, weight: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Applies the simulated quantization based on the chosen strategy."""
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            return weight # Skip non-float tensors

        current_device = weight.device
        
        if self.scaling_strategy == "per_tensor":
            scale_val = torch.max(torch.abs(weight))
            # Ensure scale_val is a tensor for consistent handling in _quantize_fp8_simulated
            scale = scale_val if scale_val != 0 else torch.tensor(1.0, device=current_device, dtype=weight.dtype)
            self.scale_factors[layer_name] = scale.item() # Store the scale value
            quantized_weight = self._quantize_fp8_simulated(weight, scale)

        elif self.scaling_strategy == "per_channel":
            if weight.ndim < 2: # Fallback to per-tensor for 1D tensors
                scale_val = torch.max(torch.abs(weight))
                scale = scale_val if scale_val != 0 else torch.tensor(1.0, device=current_device, dtype=weight.dtype)
                self.scale_factors[layer_name] = scale.item()
                quantized_weight = self._quantize_fp8_simulated(weight, scale)
            else:
                # Assume channel dimension is 0 for typical Conv layers
                # For Linear layers (e.g., [out_features, in_features]), dim 0 is also common.
                # If weights are [in, out], dim 1 might be needed. Defaulting to dim 0.
                channel_dim = 0 
                dims_to_reduce = [d for d in range(weight.ndim) if d != channel_dim]
                if not dims_to_reduce: # Handle edge case if channel_dim is the only dim somehow
                     scale_val = torch.max(torch.abs(weight))
                     scale = scale_val if scale_val != 0 else torch.tensor(1.0, device=current_device, dtype=weight.dtype)
                     self.scale_factors[layer_name] = scale.item()
                else:
                    scale = torch.amax(torch.abs(weight), dim=dims_to_reduce, keepdim=True)
                    # Store scales as a list of floats
                    self.scale_factors[layer_name] = scale.squeeze().tolist() 
                
                quantized_weight = self._quantize_fp8_simulated(weight, scale)
        else:
            raise ValueError(f"Unknown scaling strategy: {self.scaling_strategy}")
            
        # The output tensor retains the original dtype but has modified values
        return quantized_weight 

    def apply_quantization(self, state_dict: dict) -> dict:
        """Applies simulated FP8 quantization to all applicable weights."""
        quantized_state_dict = {}
        # Process only floating point tensors
        eligible_tensors = {name: param for name, param in state_dict.items() if isinstance(param, torch.Tensor) and param.is_floating_point()}
        progress_bar = tqdm(eligible_tensors.items(), desc=f"Applying scaled ({self.scaling_strategy}) quantization", unit="tensor", leave=False)
        
        for name, param in progress_bar:
            # Pass the clone to avoid modifying the original dict if errors occur mid-way
            quantized_state_dict[name] = self.quantize_weights(param.clone(), name) 
        
        # Add back non-floating point tensors and non-tensor data
        for name, param in state_dict.items():
            if name not in quantized_state_dict:
                quantized_state_dict[name] = param 
        return quantized_state_dict

# --- ComfyUI Nodes ---

class ModelToStateDict: 
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("MODEL",)}}
    RETURN_TYPES = ("MODEL_STATE_DICT",); RETURN_NAMES = ("model_state_dict",)
    FUNCTION = "get_state_dict"; CATEGORY = "Model Quantization/Utils" 
    def get_state_dict(self, model):
        print("[ModelToStateDict] Attempting to extract state_dict...")
        if not hasattr(model, 'model') or not hasattr(model.model, 'state_dict'):
            print("[ModelToStateDict] Error: Invalid MODEL structure."); return ({},)
        try:
            original_state_dict = model.model.state_dict()
            print(f"[ModelToStateDict] Original keys sample: {list(original_state_dict.keys())[:5]}")
            state_dict_to_return = original_state_dict; prefixes_to_try = ["diffusion_model.", "model."]; prefix_found = False
            for prefix in prefixes_to_try:
                num_keys = len(original_state_dict);
                if num_keys == 0: break
                matches = sum(1 for k in original_state_dict if k.startswith(prefix))
                if matches > 0 and (matches / num_keys > 0.5 or matches == num_keys):
                    print(f"[ModelToStateDict] Stripping prefix '{prefix}'...")
                    state_dict_to_return = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in original_state_dict.items()}
                    print(f"[ModelToStateDict] New keys sample: {list(state_dict_to_return.keys())[:5]}"); prefix_found = True; break
            if not prefix_found: print("[ModelToStateDict] No common prefixes stripped.")
            dtypes = {}; total = 0
            for k, v in state_dict_to_return.items():
                if isinstance(v, torch.Tensor): total += 1; dt = str(v.dtype); dtypes[dt] = dtypes.get(dt, 0) + 1
            print(f"[ModelToStateDict] DEBUG: Output Tensors: {total}, Dtypes: {dtypes}")
            return (state_dict_to_return,)
        except Exception as e: print(f"[ModelToStateDict] Error: {e}"); return ({},)

class QuantizeFP8Format: # Direct FP8 conversion node
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model_state_dict": ("MODEL_STATE_DICT",), "fp8_format": (["float8_e4m3fn", "float8_e5m2"], {"default": "float8_e5m2"}), } }
    RETURN_TYPES = ("MODEL_STATE_DICT",); RETURN_NAMES = ("quantized_model_state_dict",)
    FUNCTION = "quantize_model"; CATEGORY = "Model Quantization/FP8 Direct" 
    def quantize_model(self, model_state_dict: dict, fp8_format: str):
        print(f"[QuantizeFP8Format] To {fp8_format}. Keys(sample): {list(model_state_dict.keys())[:3]}")
        if not isinstance(model_state_dict, dict) or not model_state_dict: print("[QuantizeFP8Format] Invalid input."); return ({},)
        try:
            quantizer = FP8Quantizer(quant_dtype=fp8_format) # Uses helper class with simple scaling
            quantized_state_dict = quantizer.apply_quantization(model_state_dict)
            found = False; 
            for n, p in quantized_state_dict.items():
                if isinstance(p, torch.Tensor) and "float8" in str(p.dtype): print(f"[QuantizeFP8Format] Sample '{n}' dtype: {p.dtype}, dev: {p.device}"); found=True; break
            if not found: print(f"[QuantizeFP8Format] No tensor converted to {fp8_format}.")
            print("[QuantizeFP8Format] Complete."); return (quantized_state_dict,)
        except Exception as e: print(f"[QuantizeFP8Format] Error: {e}"); return (model_state_dict,)

class QuantizeModel: # <<< RENAMED CLASS from QuantizeScaled
    """
    Applies simulated FP8 scaling (per-tensor/per-channel) and casts 
    to a specified output dtype (float16, bfloat16, or Original).
    """
    # Removed "FP8" from the list as requested
    OUTPUT_DTYPES_LIST = ["Original", "float16", "bfloat16"] 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_state_dict": ("MODEL_STATE_DICT",),
                "scaling_strategy": (["per_tensor", "per_channel"], {"default": "per_tensor"}), 
                "processing_device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                # Default to float16 for size reduction, user can choose others
                "output_dtype": (s.OUTPUT_DTYPES_LIST, {"default": "float16"}), 
            }
        }

    RETURN_TYPES = ("MODEL_STATE_DICT",)
    RETURN_NAMES = ("quantized_model_state_dict",)
    FUNCTION = "quantize_model_scaled" # Internal function name can stay
    CATEGORY = "Model Quantization" # More general category

    def quantize_model_scaled(self, model_state_dict: dict, scaling_strategy: str, processing_device: str, output_dtype: str):
        # Log using the new class name
        print(f"[QuantizeModel] Strategy: {scaling_strategy}, Device: {processing_device}, Output Dtype: {output_dtype}")
        
        if not isinstance(model_state_dict, dict) or not model_state_dict:
            print("[QuantizeModel] Error: Input model_state_dict is invalid."); 
            return (model_state_dict if isinstance(model_state_dict, dict) else {},)
        
        # Determine processing device
        current_processing_device_str = "cpu"
        if processing_device == "Auto":
            first_tensor_device = next((p.device for p in model_state_dict.values() if isinstance(p, torch.Tensor)), torch.device("cpu"))
            current_processing_device_str = str(first_tensor_device)
        elif processing_device == "CPU": current_processing_device_str = "cpu"
        elif processing_device == "GPU":
            if torch.cuda.is_available(): current_processing_device_str = "cuda"
            else: print("[QuantizeModel] Warning: GPU selected, CUDA unavailable. Defaulting to CPU."); current_processing_device_str = "cpu"
        current_processing_device = torch.device(current_processing_device_str)
        print(f"[QuantizeModel] Value scaling simulation target device: {current_processing_device}")

        # Move input state_dict to the processing device
        state_dict_on_processing_device = {}
        for name, param in model_state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict_on_processing_device[name] = param.to(current_processing_device)
            else: state_dict_on_processing_device[name] = param
        
        scaled_state_dict = {}; final_state_dict = {}

        try:
            # Perform FP8 value simulation using FP8ScaledQuantizer helper (simple scaling version)
            quantizer = FP8ScaledQuantizer(scaling_strategy=scaling_strategy)
            scaled_state_dict = quantizer.apply_quantization(state_dict_on_processing_device) 
            print(f"[QuantizeModel] FP8 value scaling simulation performed on {current_processing_device}.")

            # Cast to final output_dtype (Original, float16, bfloat16)
            if output_dtype == "Original":
                print("[QuantizeModel] Output Dtype: Original. No further dtype casting.")
                final_state_dict = scaled_state_dict 
            else:
                # output_dtype is guaranteed to be 'float16' or 'bfloat16'
                try:
                    target_torch_dtype = getattr(torch, output_dtype)
                    print(f"[QuantizeModel] Casting output to {output_dtype} ({target_torch_dtype})...")
                    for name, param in scaled_state_dict.items():
                        if isinstance(param, torch.Tensor) and param.is_floating_point():
                            final_state_dict[name] = param.to(dtype=target_torch_dtype)
                        else: 
                            final_state_dict[name] = param # Pass non-float tensors or non-tensors
                    print(f"[QuantizeModel] Casting to {output_dtype} complete.")
                except AttributeError: # Should not happen with the restricted list
                    print(f"[QuantizeModel] Error: Invalid torch dtype '{output_dtype}'. Using scaled tensors without final casting.")
                    final_state_dict = scaled_state_dict 
                except Exception as e_cast: 
                    print(f"[QuantizeModel] Error during casting loop to {output_dtype}: {e_cast}. Using scaled tensors without final casting for affected tensors.")
                    for name_done, param_done in final_state_dict.items(): pass 
                    for name_rem, param_rem in scaled_state_dict.items():
                        if name_rem not in final_state_dict: final_state_dict[name_rem] = param_rem
            
            # Verification log
            for name, param in final_state_dict.items():
                if isinstance(param, torch.Tensor) and param.is_floating_point():
                    print(f"[QuantizeModel] Sample output tensor '{name}' final dtype: {param.dtype}, device: {param.device}")
                    break
            print(f"[QuantizeModel] Processing complete.")
            return (final_state_dict,)
        except Exception as e:
            print(f"[QuantizeModel] Major error during processing: {e}")
            return (model_state_dict,)


class SaveAsSafeTensor: # No changes needed
    @classmethod
    def INPUT_TYPES(s): return { "required": { "quantized_model_state_dict": ("MODEL_STATE_DICT",), "absolute_save_path": ("STRING", {"default": "C:/temp/quantized_model.safetensors", "multiline": False}), } }
    RETURN_TYPES = () ; OUTPUT_NODE = True ; FUNCTION = "save_model"; CATEGORY = "Model Quantization/Save" 
    def save_model(self, quantized_model_state_dict: dict, absolute_save_path: str):
        print(f"[SaveAsSafeTensor] Saving to: {absolute_save_path}")
        if not isinstance(quantized_model_state_dict, dict) or not quantized_model_state_dict: print("[SaveAsSafeTensor] Error: Input invalid."); return {"ui": {"text": ["Error: Input invalid."]}}
        if not absolute_save_path: print("[SaveAsSafeTensor] Error: Path empty."); return {"ui": {"text": ["Error: Path empty."]}}
        if not absolute_save_path.lower().endswith(".safetensors"): absolute_save_path += ".safetensors"; print(f"[SaveAsSafeTensor] Appended .safetensors")
        try:
            output_dir = os.path.dirname(absolute_save_path);
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True); print(f"[SaveAsSafeTensor] Created dir.")
            cpu_state_dict = {}; dtype_counts = {}; total_tensors = 0
            for k, v in quantized_model_state_dict.items():
                if isinstance(v, torch.Tensor):
                    total_tensors += 1; tensor_to_save = v.cpu() if v.device.type != 'cpu' else v; cpu_state_dict[k] = tensor_to_save
                    dt_str = str(tensor_to_save.dtype); dtype_counts[dt_str] = dtype_counts.get(dt_str, 0) + 1
                else: cpu_state_dict[k] = v 
            print(f"[SaveAsSafeTensor] DEBUG: Tensors: {total_tensors}, Dtypes: {dtype_counts}")
            save_file(cpu_state_dict, absolute_save_path)
            print(f"[SaveAsSafeTensor] Saved successfully."); return {"ui": {"text": [f"Saved: {absolute_save_path}"]}}
        except Exception as e: print(f"[SaveAsSafeTensor] Error saving: {e}"); return {"ui": {"text": [f"Error: {e}"]}}

# --- Main (for testing outside ComfyUI, not strictly necessary for the plugin) ---
# (Test block remains the same as previous version, it already tests QuantizeModel)
if __name__ == '__main__':
    print("--- Testing Quantization Nodes (Renamed QuantizeModel) ---")

    class MockCoreModel(torch.nn.Module): 
        def __init__(self): super().__init__(); self.layer1 = torch.nn.Linear(10,10).float(); self.layer2=torch.nn.Linear(10,10).float()
        def forward(self,x): return self.layer2(self.layer1(x))
        def state_dict(self, *args, **kwargs): return {k:v.clone() for k,v in super().state_dict(*args,**kwargs).items()}
    class MockModelWithPrefix(torch.nn.Module):
        def __init__(self): super().__init__(); self.model = MockCoreModel() # Use "model." prefix
        def forward(self,x): return self.model(x)
    class MockModelPatcher:
        def __init__(self): self.model = MockModelWithPrefix(); [p.data.normal_().float() for p in self.model.parameters()]
    
    mock_comfy_model = MockModelPatcher()
    node_to_sd = ModelToStateDict()
    base_sd_tuple = node_to_sd.get_state_dict(mock_comfy_model)
    base_sd = base_sd_tuple[0] if base_sd_tuple else {}
    if not base_sd or 'layer1.weight' not in base_sd: print("ModelToStateDict failed."); exit() # Check unprefixed key
    print(f"Base SD 'layer1.weight' dtype: {base_sd['layer1.weight'].dtype}, device: {base_sd['layer1.weight'].device}")

    # Test the renamed node
    node_quantize = QuantizeModel() 
    print("\n--- Test QuantizeModel ---")
    
    # Test Case 1: Output float16
    print("Testing QuantizeModel: Output Dtype = float16, Device = CPU")
    result_fp16_tuple = node_quantize.quantize_model_scaled(base_sd, "per_tensor", "CPU", "float16")
    result_fp16 = result_fp16_tuple[0] if result_fp16_tuple else {}
    if result_fp16 and 'layer1.weight' in result_fp16:
        tensor = result_fp16['layer1.weight']
        print(f"  Output 'layer1.weight' dtype: {tensor.dtype} (Expected float16), device: {tensor.device}")
        assert tensor.dtype == torch.float16
        assert tensor.device.type == 'cpu'
    else: print("  Test Case 1 Failed.")

    # Test Case 2: Output Original
    print("\nTesting QuantizeModel: Output Dtype = Original, Device = CPU")
    result_orig_tuple = node_quantize.quantize_model_scaled(base_sd, "per_tensor", "CPU", "Original")
    result_orig = result_orig_tuple[0] if result_orig_tuple else {}
    if result_orig and 'layer1.weight' in result_orig:
        tensor = result_orig['layer1.weight']
        print(f"  Output 'layer1.weight' dtype: {tensor.dtype} (Expected {base_sd['layer1.weight'].dtype}), device: {tensor.device}")
        assert tensor.dtype == base_sd['layer1.weight'].dtype # Should match original
        assert tensor.device.type == 'cpu'
    else: print("  Test Case 2 Failed.")
    
    print("\n--- Testing Complete ---")