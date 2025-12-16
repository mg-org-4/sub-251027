# controlnet_fp8_node.py
# ControlNet-specific FP8 quantization node for ComfyUI
# Based on the provided safetensors helper script with ComfyUI integration

import torch
import json
import os
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, Union

# ComfyUI imports for folder management
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
    print("✅ ComfyUI folder_paths imported successfully")
except ImportError:
    COMFYUI_AVAILABLE = False
    print("⚠️ ComfyUI folder_paths not available - using manual paths")


def get_controlnet_models():
    """Get list of available ControlNet models from ComfyUI's models/controlnet folder."""
    if COMFYUI_AVAILABLE:
        try:
            # Get ControlNet models from ComfyUI's folder system
            controlnet_models = folder_paths.get_filename_list("controlnet")
            if controlnet_models:
                print(f"✅ Found {len(controlnet_models)} ControlNet models")
                return controlnet_models
            else:
                print("⚠️ No ControlNet models found in models/controlnet folder")
                return ["No models found"]
        except Exception as e:
            print(f"⚠️ Error accessing ControlNet models: {e}")
            return ["Error accessing models"]
    else:
        # Fallback for when ComfyUI folder_paths is not available
        return ["manual_path_required"]


def get_controlnet_model_path(model_name):
    """Get full path to a ControlNet model."""
    if COMFYUI_AVAILABLE and model_name != "manual_path_required" and model_name != "No models found":
        try:
            return folder_paths.get_full_path("controlnet", model_name)
        except Exception as e:
            print(f"⚠️ Error getting model path for {model_name}: {e}")
            return None
    return None


def get_output_folder():
    """Get the output folder for quantized models."""
    if COMFYUI_AVAILABLE:
        try:
            # Try to get the controlnet folder and create a quantized subfolder
            controlnet_folder = folder_paths.get_folder_paths("controlnet")[0]
            quantized_folder = os.path.join(controlnet_folder, "quantized")
            os.makedirs(quantized_folder, exist_ok=True)
            return quantized_folder
        except Exception as e:
            print(f"⚠️ Error creating quantized folder: {e}")
            return "models/controlnet/quantized"
    return "models/controlnet/quantized"

class ControlNetFP8Quantizer:
    """
    Advanced FP8 quantizer specifically designed for ControlNet models.
    Supports precision-aware quantization with tensor calibration and fallback logic.
    """

    def __init__(self,
                 fp8_format: str = "float8_e4m3fn",
                 quantization_strategy: str = "per_tensor",
                 activation_clipping: bool = True,
                 calibration_samples: int = 100):
        """
        Initialize the ControlNet FP8 quantizer.

        Args:
            fp8_format: FP8 format to use ('float8_e4m3fn' or 'float8_e5m2')
            quantization_strategy: 'per_tensor' or 'per_channel'
            activation_clipping: Whether to apply activation clipping
            calibration_samples: Number of samples for tensor calibration
        """
        if not hasattr(torch, fp8_format):
            raise ValueError(f"Unsupported FP8 format: {fp8_format}")

        self.fp8_format = fp8_format
        self.quantization_strategy = quantization_strategy
        self.activation_clipping = activation_clipping
        self.calibration_samples = calibration_samples
        self.scale_factors = {}
        self.metadata = {}

        # FP8 format specific parameters
        if fp8_format == "float8_e4m3fn":
            self.max_val = 448.0  # Maximum representable value for e4m3fn
            self.min_val = -448.0
        else:  # float8_e5m2
            self.max_val = 57344.0  # Maximum representable value for e5m2
            self.min_val = -57344.0

    def _analyze_tensor_statistics(self, tensor: torch.Tensor, layer_name: str) -> Dict[str, float]:
        """Analyze tensor statistics for calibration."""
        with torch.no_grad():
            # Ensure tensor is float for statistical operations
            if not tensor.is_floating_point():
                working_tensor = tensor.float()
            else:
                working_tensor = tensor

            stats = {
                'mean': working_tensor.mean().item(),
                'std': working_tensor.std().item(),
                'min': working_tensor.min().item(),
                'max': working_tensor.max().item(),
                'abs_max': working_tensor.abs().max().item(),
                'sparsity': (working_tensor == 0).float().mean().item()
            }

            # Calculate percentiles for better calibration
            try:
                flattened = working_tensor.flatten()
                stats['p99'] = torch.quantile(torch.abs(flattened), 0.99).item()
                stats['p95'] = torch.quantile(torch.abs(flattened), 0.95).item()
            except Exception as e:
                print(f"[ControlNetFP8Quantizer] Warning: percentile calculation failed for {layer_name}: {e}")
                stats['p99'] = stats['abs_max']
                stats['p95'] = stats['abs_max']

        return stats

    def _calculate_optimal_scale(self, tensor: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Calculate optimal scaling factor for quantization."""
        device = tensor.device
        dtype = tensor.dtype

        # Ensure tensor is float for quantile operations
        if not tensor.is_floating_point():
            # Convert to float32 for calculations
            working_tensor = tensor.float()
            target_dtype = torch.float32
        else:
            working_tensor = tensor
            target_dtype = dtype

        if self.quantization_strategy == "per_tensor":
            if self.activation_clipping:
                # Use 99th percentile for better outlier handling
                abs_tensor = torch.abs(working_tensor)
                try:
                    scale_val = torch.quantile(abs_tensor.flatten(), 0.99)
                except Exception as e:
                    print(f"[ControlNetFP8Quantizer] Warning: quantile failed for {layer_name}, using max: {e}")
                    scale_val = torch.max(abs_tensor)
            else:
                scale_val = torch.max(torch.abs(working_tensor))

            # Ensure scale is not zero
            scale = torch.max(scale_val, torch.tensor(1e-8, device=device, dtype=target_dtype))

        elif self.quantization_strategy == "per_channel":
            # Assume first dimension is the channel dimension for ControlNet
            if working_tensor.ndim >= 2:
                dims_to_reduce = list(range(1, working_tensor.ndim))
                if self.activation_clipping:
                    # Per-channel percentile-based scaling
                    abs_tensor = torch.abs(working_tensor)
                    # Reshape for percentile calculation per channel
                    reshaped = abs_tensor.view(working_tensor.shape[0], -1)
                    try:
                        scale = torch.quantile(reshaped, 0.99, dim=1, keepdim=False)
                        # Reshape scale to match tensor dimensions for broadcasting
                        for _ in range(len(dims_to_reduce)):
                            scale = scale.unsqueeze(-1)
                    except Exception as e:
                        print(f"[ControlNetFP8Quantizer] Warning: per-channel quantile failed for {layer_name}, using max: {e}")
                        scale = torch.amax(abs_tensor, dim=dims_to_reduce, keepdim=True)
                else:
                    scale = torch.amax(torch.abs(working_tensor), dim=dims_to_reduce, keepdim=True)
            else:
                # Fallback to per-tensor for 1D tensors
                scale_val = torch.max(torch.abs(working_tensor))
                scale = torch.max(scale_val, torch.tensor(1e-8, device=device, dtype=target_dtype))

        # Ensure scale has minimum value to prevent division by zero
        scale = torch.clamp(scale, min=1e-8)

        return scale

    def _quantize_tensor_fp8(self, tensor: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Quantize a single tensor to FP8 format with advanced calibration."""
        if not tensor.is_floating_point():
            return tensor

        original_device = tensor.device
        original_dtype = tensor.dtype

        # Move to CUDA if available for FP8 operations
        target_device = torch.device("cuda") if torch.cuda.is_available() else original_device
        tensor_on_device = tensor.to(target_device)

        # Calculate optimal scale
        scale = self._calculate_optimal_scale(tensor_on_device, layer_name)

        # Store scale factor for debugging/analysis
        if self.quantization_strategy == "per_tensor":
            self.scale_factors[layer_name] = scale.item()
        else:
            self.scale_factors[layer_name] = scale.squeeze().tolist() if scale.numel() > 1 else scale.item()

        # Perform quantization simulation
        # Scale tensor to FP8 range
        scaled_tensor = tensor_on_device / scale

        # Clamp to FP8 representable range
        if self.activation_clipping:
            # Use format-specific ranges
            if self.fp8_format == "float8_e4m3fn":
                clamped_tensor = torch.clamp(scaled_tensor, -448.0, 448.0)
            else:  # float8_e5m2
                clamped_tensor = torch.clamp(scaled_tensor, -57344.0, 57344.0)
        else:
            clamped_tensor = scaled_tensor

        # Convert to target FP8 format
        try:
            target_dtype = getattr(torch, self.fp8_format)
            quantized_tensor = clamped_tensor.to(dtype=target_dtype)

            # Convert back to original dtype for compatibility (if needed)
            # For true FP8 storage, keep the FP8 dtype
            result_tensor = quantized_tensor

        except Exception as e:
            print(f"[ControlNetFP8Quantizer] Warning: FP8 conversion failed for {layer_name}: {e}")
            print(f"[ControlNetFP8Quantizer] Falling back to simulated quantization")

            # Fallback: simulate quantization effects without actual FP8 conversion
            # This maintains compatibility while approximating FP8 behavior
            simulated_quantized = torch.round(clamped_tensor * 127.0) / 127.0 * scale
            result_tensor = simulated_quantized.to(dtype=original_dtype)

        return result_tensor.to(original_device)

    def quantize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Quantize an entire state dictionary."""
        quantized_state_dict = {}

        # Filter tensors that should be quantized
        # Only quantize floating point tensors with sufficient size
        quantizable_tensors = {}
        skipped_tensors = {}

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                skipped_tensors[name] = "Not a tensor"
                continue

            # Skip very small tensors (likely bias terms or scalars)
            if tensor.numel() < 4:
                skipped_tensors[name] = f"Too small ({tensor.numel()} elements)"
                continue

            # Skip non-floating point tensors
            if not tensor.is_floating_point():
                skipped_tensors[name] = f"Non-float dtype ({tensor.dtype})"
                continue

            # Skip tensors that are likely indices or embeddings
            if any(keyword in name.lower() for keyword in ['index', 'embedding', 'position']):
                skipped_tensors[name] = "Likely embedding/index tensor"
                continue

            quantizable_tensors[name] = tensor

        print(f"[ControlNetFP8Quantizer] Quantizing {len(quantizable_tensors)} tensors to {self.fp8_format}")
        print(f"[ControlNetFP8Quantizer] Skipping {len(skipped_tensors)} tensors")
        print(f"[ControlNetFP8Quantizer] Strategy: {self.quantization_strategy}, Clipping: {self.activation_clipping}")

        # Log some skipped tensors for debugging
        if skipped_tensors:
            sample_skipped = list(skipped_tensors.items())[:3]
            for name, reason in sample_skipped:
                print(f"[ControlNetFP8Quantizer] Skipped '{name}': {reason}")

        # Progress bar for quantization
        progress_bar = tqdm(
            quantizable_tensors.items(),
            desc=f"FP8 Quantization ({self.fp8_format})",
            unit="tensor",
            leave=False
        )

        for name, tensor in progress_bar:
            progress_bar.set_postfix({"layer": name[:30] + "..." if len(name) > 30 else name})

            try:
                # Analyze tensor statistics
                stats = self._analyze_tensor_statistics(tensor, name)

                # Quantize tensor
                quantized_tensor = self._quantize_tensor_fp8(tensor.clone(), name)
                quantized_state_dict[name] = quantized_tensor

                # Log statistics for important layers
                if any(keyword in name.lower() for keyword in ['conv', 'linear', 'attention', 'norm']):
                    print(f"[ControlNetFP8Quantizer] {name}: "
                          f"abs_max={stats['abs_max']:.6f}, "
                          f"sparsity={stats['sparsity']:.3f}, "
                          f"scale={self.scale_factors.get(name, 'N/A')}")

            except Exception as e:
                print(f"[ControlNetFP8Quantizer] Error quantizing {name}: {e}")
                # Keep original tensor if quantization fails
                quantized_state_dict[name] = tensor

        # Copy non-quantizable tensors
        for name, tensor in state_dict.items():
            if name not in quantized_state_dict:
                quantized_state_dict[name] = tensor

        return quantized_state_dict

    def load_safetensors_with_metadata(self, file_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load safetensors file and extract metadata."""
        # Read metadata from safetensors header
        with open(file_path, 'rb') as f:
            header_size = int.from_bytes(f.read(8), 'little')
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            metadata = header.get('__metadata__', {})

        # Load the actual tensors
        state_dict = load_file(file_path)

        self.metadata = metadata
        return state_dict, metadata

    def save_quantized_model(self,
                           quantized_state_dict: Dict[str, torch.Tensor],
                           save_path: str,
                           original_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save quantized model with updated metadata."""
        try:
            # Prepare metadata - ensure all values are strings for safetensors compatibility
            updated_metadata = {}
            if original_metadata:
                # Convert all original metadata values to strings
                for key, value in original_metadata.items():
                    updated_metadata[key] = str(value)

            # Add quantization metadata
            updated_metadata.update({
                "quantization_format": self.fp8_format,
                "quantization_strategy": self.quantization_strategy,
                "activation_clipping": str(self.activation_clipping),
                "quantizer_version": "ControlNetFP8Quantizer_v1.0",
                "scale_factors_sample": str(list(self.scale_factors.items())[:3])  # Sample for debugging
            })

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Move tensors to CPU for saving
            cpu_state_dict = {}
            for name, tensor in quantized_state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    cpu_state_dict[name] = tensor.cpu()
                else:
                    cpu_state_dict[name] = tensor

            # Save with metadata
            save_file(cpu_state_dict, save_path, metadata=updated_metadata)

            print(f"[ControlNetFP8Quantizer] Successfully saved quantized model to: {save_path}")
            return True

        except Exception as e:
            print(f"[ControlNetFP8Quantizer] Error saving model: {e}")
            return False


# ComfyUI Node Implementation
class ControlNetFP8QuantizeNode:
    """
    ComfyUI node for ControlNet FP8 quantization with advanced features.
    Supports loading, quantizing, and saving ControlNet models in FP8 format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available ControlNet models
        controlnet_models = get_controlnet_models()

        input_types = {
            "required": {
                "controlnet_model": (controlnet_models, {
                    "default": controlnet_models[0] if controlnet_models else "No models found"
                }),
                "fp8_format": (["float8_e4m3fn", "float8_e5m2"], {
                    "default": "float8_e4m3fn"
                }),
                "quantization_strategy": (["per_tensor", "per_channel"], {
                    "default": "per_tensor"
                }),
                "activation_clipping": ("BOOLEAN", {
                    "default": True
                }),
            },
            "optional": {
                "custom_output_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Custom output filename (optional)"
                }),
                "calibration_samples": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10
                }),
                "preserve_metadata": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

        # Add manual path option if ComfyUI folder system is not available
        if not COMFYUI_AVAILABLE or "manual_path_required" in controlnet_models:
            input_types["optional"]["manual_path"] = ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "Manual path to ControlNet model (if not using dropdown)"
            })

        return input_types

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "metadata_info", "quantization_stats")
    FUNCTION = "quantize_controlnet"
    CATEGORY = "Model Quantization/ControlNet"
    OUTPUT_NODE = True

    def quantize_controlnet(self,
                          controlnet_model: str,
                          fp8_format: str,
                          quantization_strategy: str,
                          activation_clipping: bool,
                          custom_output_name: str = "",
                          calibration_samples: int = 100,
                          preserve_metadata: bool = True,
                          manual_path: str = ""):
        """
        Main function to quantize ControlNet models to FP8 format.
        """
        try:
            # Determine the actual model path
            if manual_path and os.path.exists(manual_path):
                # Use manual path if provided and exists
                safetensors_path = manual_path
                print(f"[ControlNetFP8QuantizeNode] Using manual path: {safetensors_path}")
            else:
                # Use dropdown selection
                safetensors_path = get_controlnet_model_path(controlnet_model)
                if not safetensors_path:
                    error_msg = f"Could not find model: {controlnet_model}"
                    print(f"[ControlNetFP8QuantizeNode] Error: {error_msg}")
                    return (f"ERROR: {error_msg}", "", "")
                print(f"[ControlNetFP8QuantizeNode] Using selected model: {controlnet_model}")

            # Validate that the file exists
            if not os.path.exists(safetensors_path):
                error_msg = f"Model file not found: {safetensors_path}"
                print(f"[ControlNetFP8QuantizeNode] Error: {error_msg}")
                return (f"ERROR: {error_msg}", "", "")

            # Generate output path
            output_folder = get_output_folder()
            if custom_output_name:
                output_filename = custom_output_name
                if not output_filename.endswith('.safetensors'):
                    output_filename += '.safetensors'
            else:
                base_name = os.path.splitext(os.path.basename(safetensors_path))[0]
                output_filename = f"{base_name}_fp8_{fp8_format}.safetensors"

            output_path = os.path.join(output_folder, output_filename)
            print(f"[ControlNetFP8QuantizeNode] Output path: {output_path}")

            # Initialize quantizer
            quantizer = ControlNetFP8Quantizer(
                fp8_format=fp8_format,
                quantization_strategy=quantization_strategy,
                activation_clipping=activation_clipping,
                calibration_samples=calibration_samples
            )

            print(f"[ControlNetFP8QuantizeNode] Loading model from: {safetensors_path}")

            # Load model and metadata
            state_dict, metadata = quantizer.load_safetensors_with_metadata(safetensors_path)

            # Analyze model structure
            total_tensors = len(state_dict)
            quantizable_tensors = sum(1 for v in state_dict.values()
                                    if isinstance(v, torch.Tensor) and v.is_floating_point())

            print(f"[ControlNetFP8QuantizeNode] Model loaded: {total_tensors} total tensors, "
                  f"{quantizable_tensors} quantizable")

            # Perform quantization
            print(f"[ControlNetFP8QuantizeNode] Starting quantization...")
            quantized_state_dict = quantizer.quantize_state_dict(state_dict)

            # Calculate statistics
            original_size = sum(v.numel() * v.element_size() for v in state_dict.values()
                              if isinstance(v, torch.Tensor))
            quantized_size = sum(v.numel() * v.element_size() for v in quantized_state_dict.values()
                               if isinstance(v, torch.Tensor))

            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

            # Save quantized model
            save_metadata = metadata if preserve_metadata else {}
            success = quantizer.save_quantized_model(quantized_state_dict, output_path, save_metadata)

            if success:
                status_msg = f"SUCCESS: Quantized model saved to {output_path}"

                # Prepare metadata info
                metadata_info = json.dumps({
                    "original_metadata": metadata,
                    "quantization_metadata": {
                        "fp8_format": fp8_format,
                        "quantization_strategy": quantization_strategy,
                        "activation_clipping": activation_clipping,
                        "calibration_samples": calibration_samples
                    }
                }, indent=2)

                # Prepare quantization statistics
                stats_info = json.dumps({
                    "total_tensors": total_tensors,
                    "quantizable_tensors": quantizable_tensors,
                    "original_size_mb": round(original_size / (1024 * 1024), 2),
                    "quantized_size_mb": round(quantized_size / (1024 * 1024), 2),
                    "compression_ratio": round(compression_ratio, 2),
                    "scale_factors_sample": dict(list(quantizer.scale_factors.items())[:5])
                }, indent=2)

                print(f"[ControlNetFP8QuantizeNode] Quantization completed successfully!")
                print(f"[ControlNetFP8QuantizeNode] Compression ratio: {compression_ratio:.2f}x")

                return (status_msg, metadata_info, stats_info)
            else:
                error_msg = "Failed to save quantized model"
                return (f"ERROR: {error_msg}", "", "")

        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            print(f"[ControlNetFP8QuantizeNode] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return (f"ERROR: {error_msg}", "", "")


class ControlNetMetadataViewerNode:
    """
    ComfyUI node for viewing ControlNet model metadata and structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available ControlNet models
        controlnet_models = get_controlnet_models()

        input_types = {
            "required": {
                "controlnet_model": (controlnet_models, {
                    "default": controlnet_models[0] if controlnet_models else "No models found"
                }),
            }
        }

        # Add manual path option if ComfyUI folder system is not available
        if not COMFYUI_AVAILABLE or "manual_path_required" in controlnet_models:
            input_types["optional"] = {
                "manual_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Manual path to ControlNet model (if not using dropdown)"
                })
            }

        return input_types

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metadata", "tensor_info", "model_analysis")
    FUNCTION = "analyze_model"
    CATEGORY = "Model Quantization/ControlNet"
    OUTPUT_NODE = True

    def analyze_model(self, controlnet_model: str, manual_path: str = ""):
        """Analyze and display ControlNet model information."""
        try:
            # Determine the actual model path
            if manual_path and os.path.exists(manual_path):
                # Use manual path if provided and exists
                safetensors_path = manual_path
                print(f"[ControlNetMetadataViewerNode] Using manual path: {safetensors_path}")
            else:
                # Use dropdown selection
                safetensors_path = get_controlnet_model_path(controlnet_model)
                if not safetensors_path:
                    error_msg = f"Could not find model: {controlnet_model}"
                    print(f"[ControlNetMetadataViewerNode] Error: {error_msg}")
                    return (f"ERROR: {error_msg}", "", "")
                print(f"[ControlNetMetadataViewerNode] Analyzing model: {controlnet_model}")

            # Validate that the file exists
            if not os.path.exists(safetensors_path):
                error_msg = f"Model file not found: {safetensors_path}"
                print(f"[ControlNetMetadataViewerNode] Error: {error_msg}")
                return (f"ERROR: {error_msg}", "", "")

            # Load metadata
            with open(safetensors_path, 'rb') as f:
                header_size = int.from_bytes(f.read(8), 'little')
                header_json = f.read(header_size).decode('utf-8')
                header = json.loads(header_json)
                metadata = header.get('__metadata__', {})

            # Load tensors for analysis
            state_dict = load_file(safetensors_path)

            # Analyze tensor information
            tensor_analysis = {}
            total_params = 0
            dtype_counts = {}

            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
                    dtype_str = str(tensor.dtype)
                    dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

                    tensor_analysis[name] = {
                        "shape": list(tensor.shape),
                        "dtype": dtype_str,
                        "device": str(tensor.device),
                        "numel": tensor.numel(),
                        "size_mb": round(tensor.numel() * tensor.element_size() / (1024 * 1024), 4)
                    }

            # Model analysis
            model_analysis = {
                "total_tensors": len(state_dict),
                "total_parameters": total_params,
                "total_size_mb": round(sum(t.numel() * t.element_size() for t in state_dict.values()
                                         if isinstance(t, torch.Tensor)) / (1024 * 1024), 2),
                "dtype_distribution": dtype_counts,
                "layer_types": self._analyze_layer_types(list(state_dict.keys()))
            }

            # Format outputs
            metadata_str = json.dumps(metadata, indent=2) if metadata else "No metadata found"
            tensor_info_str = json.dumps(tensor_analysis, indent=2)
            analysis_str = json.dumps(model_analysis, indent=2)

            return (metadata_str, tensor_info_str, analysis_str)

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"[ControlNetMetadataViewerNode] Error: {error_msg}")
            return (f"ERROR: {error_msg}", "", "")

    def _analyze_layer_types(self, layer_names):
        """Analyze the types of layers in the model."""
        layer_types = {}
        for name in layer_names:
            if 'conv' in name.lower():
                layer_types['convolution'] = layer_types.get('convolution', 0) + 1
            elif 'linear' in name.lower() or 'fc' in name.lower():
                layer_types['linear'] = layer_types.get('linear', 0) + 1
            elif 'norm' in name.lower() or 'bn' in name.lower():
                layer_types['normalization'] = layer_types.get('normalization', 0) + 1
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_types['attention'] = layer_types.get('attention', 0) + 1
            elif 'embed' in name.lower():
                layer_types['embedding'] = layer_types.get('embedding', 0) + 1
            else:
                layer_types['other'] = layer_types.get('other', 0) + 1
        return layer_types
