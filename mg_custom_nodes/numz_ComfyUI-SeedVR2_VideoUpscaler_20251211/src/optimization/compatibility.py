"""
Compatibility module for SeedVR2
Contains FP8/FP16 compatibility layers and wrappers for different model architectures

Extracted from: seedvr2.py (lines 1045-1630)
"""

# Compatibility shims - Must run before any torch/diffusers import
import sys
import types


def ensure_triton_compat():
    """Create minimal triton.ops stubs only if missing, to allow bitsandbytes import."""
    if 'triton.ops.matmul_perf_model' in sys.modules:
        return
    
    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    
    if 'triton.ops' not in sys.modules:
        sys.modules['triton.ops'] = types.ModuleType('triton.ops')
    
    matmul_perf = types.ModuleType('triton.ops.matmul_perf_model')
    matmul_perf.early_config_prune = lambda configs, *a, **kw: configs
    matmul_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    
    sys.modules['triton.ops'].matmul_perf_model = matmul_perf
    sys.modules['triton.ops.matmul_perf_model'] = matmul_perf


def ensure_flash_attn_safe():
    """
    Pre-test flash_attn package; stub if DLL is broken.
    Prevents diffusers from crashing when flash_attn has broken DLLs.
    """
    if 'flash_attn' in sys.modules:
        return  # Already loaded
    
    try:
        import flash_attn
    except (ImportError, OSError):
        # DLL broken or not installed - create stub with proper __spec__
        import importlib.machinery
        
        stub = types.ModuleType('flash_attn')
        stub.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
        stub.__file__ = None
        stub.__path__ = []
        stub.__loader__ = None
        # Provide attributes that diffusers/transformers import
        stub.flash_attn_func = None
        stub.flash_attn_varlen_func = None
        sys.modules['flash_attn'] = stub


def ensure_xformers_flash_compat():
    """
    Pre-test xformers._C_flashattention; stub if DLL is broken.
    Prevents xformers.ops.fmha.flash from crashing on import.
    """
    if 'xformers._C_flashattention' in sys.modules:
        return  # Already loaded
    
    try:
        from xformers import _C_flashattention  # noqa: F401
    except (ImportError, OSError):
        # DLL broken or not installed - create stub that fails gracefully
        class _FailingStub(types.ModuleType):
            """Stub that lets xformers gracefully disable its flash backend."""
            def __getattr__(self, name):
                # Dunder attributes: raise AttributeError (normal Python behavior)
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                # xformers functional attributes: raise ImportError so xformers catches it
                raise ImportError("_C_flashattention unavailable")
        sys.modules['xformers._C_flashattention'] = _FailingStub('xformers._C_flashattention')


# Run all shims immediately on import, before torch/diffusers
ensure_triton_compat()
ensure_flash_attn_safe()
ensure_xformers_flash_compat()


import torch
import os


# Flash Attention & Triton Compatibility Layer
# 1. Flash Attention - speedup for attention operations
try:
    from flash_attn import flash_attn_varlen_func
    # Force load the CUDA extension to verify it's not corrupted
    import flash_attn_2_cuda  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False

def validate_flash_attention_availability(requested_mode: str, debug=None) -> str:
    """
    Validate Flash Attention availability and warn if fallback needed.
    
    Args:
        requested_mode: Either 'flash_attn' or 'sdpa'
        debug: Optional debug instance for logging
        
    Returns:
        Validated mode ('flash_attn' or 'sdpa')
    """
    if requested_mode == 'flash_attn' and not FLASH_ATTN_AVAILABLE:
        error_msg = (
            f"Cannot use 'flash_attn' attention mode: Flash Attention is not installed.\n"
            f"\n"
            f"Flash Attention provides speedup on some hardware through optimized CUDA kernels.\n"
            f"Falling back to PyTorch SDPA (scaled dot-product attention).\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install Flash Attention: pip install flash-attn\n"
            f"  2. OR change attention_mode to 'sdpa' (default, always available)\n"
            f"\n"
            f"For more info: https://github.com/Dao-AILab/flash-attention"
        )
        if debug:
            debug.log(error_msg, level="WARNING", category="setup", force=True)
        
        return 'sdpa'
    
    return requested_mode


# 2. Triton - Required for torch.compile with inductor backend
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# 3. GGUF - Required for quantized model loading
try:
    import gguf
    from gguf import GGMLQuantizationType
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None
    GGMLQuantizationType = None


def validate_gguf_availability(operation: str = "load GGUF model", debug=None) -> None:
    """
    Validate GGUF availability and raise error if not installed.
    
    Args:
        operation: Description of the operation requiring GGUF
        debug: Optional debug instance for logging
        
    Raises:
        RuntimeError: If GGUF is not available
    """
    if not GGUF_AVAILABLE:
        error_msg = (
            f"Cannot {operation}: GGUF library is not installed.\n"
            f"\n"
            f"GGUF provides quantized model support for memory-efficient loading.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install GGUF: pip install gguf\n"
            f"  2. OR use a non-quantized model format (.safetensors)\n"
            f"\n"
            f"For more info: https://github.com/ggerganov/ggml"
        )
        if debug:
            debug.log(error_msg, level="ERROR", category="setup", force=True)
        raise RuntimeError(f"GGUF library required to {operation}")


# 4. NVIDIA Conv3d Memory Bug - Workaround for PyTorch 2.9-2.10 + cuDNN >= 91002
def _check_conv3d_memory_bug():
    """
    Check if Conv3d memory bug workaround needed.
    Bug: PyTorch 2.9-2.10 with cuDNN >= 91002 uses 3x memory for Conv3d 
    with fp16/bfloat16 due to buggy dispatch layer.
    """
    try:
        # Exclude AMD ROCm/HIP builds (they use MIOpen, not cuDNN)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return False
        
        # Must have CUDA available
        if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            return False
        
        # Must have cuDNN actually available (not just the attribute)
        if not (hasattr(torch.backends.cudnn, 'is_available') and 
                torch.backends.cudnn.is_available()):
            return False
        
        # Check device capability (NVIDIA GPUs)
        if torch.cuda.get_device_capability()[0] < 3:
            return False
        
        # Parse torch version
        version_str = torch.__version__.split('+')[0]
        parts = version_str.split('.')
        torch_version = tuple(int(p) for p in parts[:2])
        
        if not ((2, 9) <= torch_version <= (2, 10)):
            return False
        
        if not hasattr(torch.backends.cudnn, 'version'):
            return False
            
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None or cudnn_version < 91002:
            return False
        
        return True
    except:
        return False

NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = _check_conv3d_memory_bug()


# Log all optimization status once globally (cross-process) using environment variable
if not os.environ.get("SEEDVR2_OPTIMIZATIONS_LOGGED"):
    os.environ["SEEDVR2_OPTIMIZATIONS_LOGGED"] = "1"
    
    # Flash Attention & Triton status
    has_both = FLASH_ATTN_AVAILABLE and TRITON_AVAILABLE
    has_neither = not FLASH_ATTN_AVAILABLE and not TRITON_AVAILABLE
    
    if has_both:
        print("⚡ SeedVR2 optimizations check: Flash Attention ✅ | Triton ✅")
    elif has_neither:
        print("⚠️  SeedVR2 optimizations check: Flash Attention ❌ | Triton ❌")
        print("💡 For best performance: pip install flash-attn triton")
    elif FLASH_ATTN_AVAILABLE:
        print("⚡ SeedVR2 optimizations check: Flash Attention ✅ | Triton ❌")
        print("💡 Install Triton for torch.compile: pip install triton")
    else:  # TRITON_AVAILABLE only
        print("⚠️  SeedVR2 optimizations check: Flash Attention ❌ | Triton ✅")
        print("💡 Install Flash Attention for faster inference: pip install flash-attn")
    
    # Conv3d workaround status (if applicable)
    if NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND:
        torch_ver = torch.__version__.split('+')[0]
        cudnn_ver = torch.backends.cudnn.version()
        print(f"🔧 Conv3d workaround active: PyTorch {torch_ver}, cuDNN {cudnn_ver} (fixing VAE 3x memory bug)")


# Bfloat16 CUBLAS support
def _probe_bfloat16_support() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda:0')
        _ = torch.matmul(a, a)
        del a
        return True
    except RuntimeError as e:
        if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
            return False
        raise

BFLOAT16_SUPPORTED = _probe_bfloat16_support()
COMPUTE_DTYPE = torch.bfloat16 if BFLOAT16_SUPPORTED else torch.float16


def call_rope_with_stability(method, *args, **kwargs):
    """
    Call RoPE method with stability fixes:
    1. Clear cache if available
    2. Disable autocast to prevent numerical issues
    This prevents artifacts in FP8/mixed precision models.
    """
    if hasattr(method, 'cache_clear'):
        method.cache_clear()
    
    with torch.cuda.amp.autocast(enabled=False):
        return method(*args, **kwargs)
    
    
class FP8CompatibleDiT(torch.nn.Module):
    """
    Wrapper for DiT models with automatic compatibility management + advanced optimizations
    
    Precision Handling:
    - FP8: Keeps native FP8 parameters (memory efficient), converts inputs/outputs to compute_dtype for arithmetic
    - FP16: Uses native FP16 precision throughout
    - BFloat16: Uses native BFloat16 precision throughout
    - Float32: Uses full precision for maximum quality
    - RoPE: Converted from FP8 to compute_dtype for numerical consistency
    
    Optimizations:
    - Flash Attention: Automatic optimization of attention layers
    - RoPE Stabilization: Error handling for numerical stability in mixed precision
    - MPS Compatibility: Unified dtype conversion for Apple Silicon backends
    """
    
    def __init__(self, dit_model, debug: 'Debug', compute_dtype: torch.dtype = torch.bfloat16, skip_conversion: bool = False):
        super().__init__()
        self.dit_model = dit_model
        self.debug = debug
        self.compute_dtype = compute_dtype
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16

        # Only convert if not already done (e.g., when reusing cached weights)
        if not skip_conversion and self.is_fp8_model:
            # FP8 models need RoPE frequency conversion to compute dtype
            model_variant = self._get_model_variant()
            self.debug.log(f"Detected NaDiT {model_variant} FP8 - Converting RoPE freqs for FP8 compatibility", 
                        category="precision")
            self.debug.start_timer("_convert_rope_freqs")
            self._convert_rope_freqs(target_dtype=self.compute_dtype)
            self.debug.end_timer("_convert_rope_freqs", "RoPE freqs conversion")
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.debug.log(f"Also converting NaDiT parameters/buffers for MPS backend", category="setup", force=True)
                self.debug.start_timer("_force_nadit_precision")
                self._force_nadit_precision(target_dtype=self.compute_dtype)
                self.debug.end_timer("_force_nadit_precision", "NaDiT parameters/buffers conversion")
            
        # Apply RoPE stabilization for numerical stability
        self.debug.log(f"Stabilizing RoPE computations for numerical stability", category="setup")
        self.debug.start_timer("_stabilize_rope_computations")
        self._stabilize_rope_computations()
        self.debug.end_timer("_stabilize_rope_computations", "RoPE stabilization")
    
    def _detect_model_dtype(self) -> torch.dtype:
        """Detect main model dtype"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _get_model_variant(self) -> str:
        """Detect model variant from module path"""
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit_7b' in model_module:
            return "7B"
        elif 'dit_3b' in model_module:
            return "3B"
        else:
            return "Unknown"
        
    def _convert_rope_freqs(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Convert RoPE frequency buffers from FP8 to target dtype for compatibility.
        
        Args:
            target_dtype: Target dtype for RoPE freqs (default: bfloat16 for stability)
        """
        converted = 0
        for module in self.dit_model.modules():
            if 'RotaryEmbedding' in type(module).__name__:
                if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
                    if module.rope.freqs.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if module.rope.freqs.device.type == "mps":
                            module.rope.freqs.data = module.rope.freqs.to("cpu").to(target_dtype).to("mps")
                        else:
                            module.rope.freqs.data = module.rope.freqs.to(target_dtype)
                        converted += 1
        self.debug.log(f"Converted {converted} RoPE frequency buffers from FP8 to {target_dtype} for compatibility", category="success")
                        
    def _force_nadit_precision(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Force ALL NaDiT parameters to target dtype to avoid promotion errors (MPS requirement).
        
        Args:
            target_dtype: Target dtype for all parameters (default: bfloat16 for MPS compatibility)
        """
        converted_count = 0
        original_dtype = None
        
        # Convert ALL parameters to target dtype
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != target_dtype:
                if param.device.type == "mps":
                    temp_cpu = param.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    param.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    param.data = param.data.to(target_dtype)
                converted_count += 1
                
        # Also convert buffers
        for name, buffer in self.dit_model.named_buffers():
            if buffer.dtype != target_dtype:
                if buffer.device.type == "mps":
                    temp_cpu = buffer.data.to("cpu")
                    temp_converted = temp_cpu.to(target_dtype)
                    buffer.data = temp_converted.to("mps")
                    del temp_cpu, temp_converted
                else:
                    buffer.data = buffer.data.to(target_dtype)
                converted_count += 1
        
        self.debug.log(f"Converted {converted_count} NaDiT parameters/buffers to {target_dtype} for MPS", category="success")
        
        # Update detected dtype
        self.model_dtype = target_dtype
        self.is_fp8_model = (target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2))

    def _stabilize_rope_computations(self):
        """
        Add error handling to RoPE computations to prevent artifacts.
        
        Wraps the get_axial_freqs method of RoPE modules with a try-except handler.
        During normal operation, uses the original cached method for performance.
        Only on exceptions (e.g., numerical instability, NaN propagation) does it
        intervene by clearing the cache and retrying the computation through
        call_rope_with_stability.
        
        This prevents artifacts in FP8, mixed precision, and edge cases while
        maintaining optimal performance for normal operations.
        """
        if not hasattr(self.dit_model, 'blocks'):
            return
        
        rope_count = 0
        
        # Wrap RoPE modules to handle numerical instability
        for name, module in self.dit_model.named_modules():
            if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
                # Check if already wrapped
                if hasattr(module, '_rope_wrapped'):
                    continue
                    
                original_method = module.get_axial_freqs
                
                # Mark as wrapped and store original
                module._rope_wrapped = 'stability'
                module._original_get_axial_freqs = original_method
                
                # Error handler that prevents NaN propagation
                def stable_rope_computation(self, *args, **kwargs):
                    try:
                        return original_method(*args, **kwargs)
                    except Exception:
                        return call_rope_with_stability(original_method, *args, **kwargs)
                
                module.get_axial_freqs = types.MethodType(stable_rope_computation, module)
                rope_count += 1
        
        if rope_count > 0:
            self.debug.log(f"Stabilized {rope_count} RoPE modules", category="success")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with minimal dtype conversion overhead
        
        Conversion strategy:
            - FP16/BFloat16/Float32 models: Use native precision (no conversion needed)
            - FP8 models: Convert FP8 tensors to compute_dtype for arithmetic operations
            (FP8 parameters stay in FP8 for memory efficiency, only converted for computation)
        """
        
        # Only convert if we have an FP8 model for arithmetic operations 
        if self.is_fp8_model:
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            target_dtype = self.compute_dtype
            
            # Convert args
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in fp8_dtypes:
                    converted_args.append(arg.to(target_dtype))
                else:
                    converted_args.append(arg)
            
            # Convert kwargs
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dtype in fp8_dtypes:
                    converted_kwargs[key] = value.to(target_dtype)
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
        
        # Execute forward pass
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            self.debug.log(f"Forward pass error: {e}", level="ERROR", category="generation", force=True)
            if self.is_fp8_model:
                self.debug.log(f"FP8 model - converted FP8 tensors to {self.compute_dtype}", category="info", force=True)
            else:
                self.debug.log(f"{self.model_dtype} model - no conversion applied", category="info", force=True)
            raise
    
    def __getattr__(self, name):
        """Redirect all other attributes to original model"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Redirect assignments to original model except for our attributes"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)
                