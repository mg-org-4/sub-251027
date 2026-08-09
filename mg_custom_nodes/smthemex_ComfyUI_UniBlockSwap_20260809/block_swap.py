"""
UniBlockSwap - Universal single-block swap for ComfyUI.
Safetensor blocks: freed to meta on swap, restored by vbar automatically.
GGUF blocks: freed to CPU on swap, moved to GPU when accessed.
"""

import gc
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CONTAINER_NAMES = (
    "blocks", "transformer_blocks", "double_blocks", "single_blocks",
    "input_blocks", "output_blocks", "middle_block", "layers",
    "double_stream_layers", "single_stream_layers",
    "block",
)


def find_blocks(model):
    for name in CONTAINER_NAMES:
        c = getattr(model, name, None)
        if isinstance(c, (nn.ModuleList, list)) and len(c) > 0 and hasattr(c[0], "forward"):
            return name, c
    return None, None


def _has_ggml_params(module):
    """Check if module has GGMLTensor parameters (quantized GGUF weights)."""
    for p in module.parameters():
        if hasattr(p, 'tensor_type'):
            return True
    return False


def _backup_ggml_refs(module):
    """Preserve the ORIGINAL mmap-backed GGMLTensor objects for every GGML
    parameter in `module`.

    Why a full-reference backup (not just .data): tensor_type / tensor_shape /
    patches live on the GGMLTensor *object*, and a .to(...) round trip creates
    a fresh tensor that loses the mmap mapping. We must keep the original object
    alive so we can point the parameter back at it later.
    """
    if getattr(module, "_ggml_mmap_backup", None) is not None:
        return
    backup = {}
    for name, param in module.named_parameters(recurse=True):
        t = param.data
        if hasattr(t, "tensor_type"):      # a GGMLTensor
            backup[name] = t               # keep the object alive, mmap intact
    module._ggml_mmap_backup = backup


def _restore_ggml_refs(module):
    """Point params back at the original mmap GGMLTensors and drop any GPU
    copies. This is a *pointer assignment* (p.data = orig), so NO anonymous
    heap allocation happens -- unlike module.to(offload_device), which would
    reallocate the dequantized weights as non-reclaimable RAM.

    If a block was never GPU-loaded (no backup), fall back to .to(cpu) which is
    a no-op for an already-mmap'd CPU tensor.
    """
    backup = getattr(module, "_ggml_mmap_backup", None)
    if not backup:
        module.to(module.offload_device if hasattr(module, "offload_device") else "cpu")
        return
    params = dict(module.named_parameters(recurse=True))
    for name, orig in backup.items():
        p = params.get(name)
        if p is not None:
            p.data = orig
    # free the GPU copy of the now-unreferenced tensor
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _free_to_meta(module):
    """Free param data to meta tensor - NO CPU copy created.
    The module structure is preserved. next load() restores from backup."""
    for param in module.parameters(recurse=False):
        param.data = torch.empty(0, device='meta')


class SwappableModuleList(nn.ModuleList):
    def __init__(self, modules, compute_device, offload_device,
                 non_swap_count=0):
        super().__init__(modules)
        self.compute_device = compute_device
        self.offload_device = offload_device
        self.non_swap_count = non_swap_count
        self.total_count = len(modules)
        self._loaded_swap_idx = -1
        self.container_name = ''

    def _load_swap(self, local_idx):
        idx = local_idx + self.non_swap_count
        if local_idx == self._loaded_swap_idx:
            return
        if self._loaded_swap_idx >= 0:
            prev = self._loaded_swap_idx + self.non_swap_count
            try:
                prev_mod = self._modules[str(prev)]
                # FREE previous block GPU memory
                if _has_ggml_params(prev_mod):
                    # GGUF: restore the original mmap-backed GGMLTensor by
                    # pointer assignment. This drops the GPU copy WITHOUT
                    # reallocating the weights as anonymous CPU RAM (which
                    # .to(offload_device) would do after a .to(cuda) round
                    # trip, blowing RAM from 40G to 60G).
                    _restore_ggml_refs(prev_mod)
                else:
                    # Safetensor: set to meta (vbar restores automatically)
                    _free_to_meta(prev_mod)
                for m in prev_mod.modules():
                    for attr in ('_v', '_prefetch', '_v_signature'):
                        if hasattr(m, attr):
                            try:
                                delattr(m, attr)
                            except Exception:
                                pass
            except Exception:
                pass
        # LOAD current block if GGUF
        cur_mod = self._modules[str(idx)]
        if _has_ggml_params(cur_mod):
            # Snapshot the mmap reference so we can later restore it. We do NOT
            # call cur_mod.to(compute_device) here: GGUF weights are dequantized
            # per-layer on demand inside GGMLLayer.cast_bias_weight() when each
            # op runs (self.weight.to(input.device)). Pre-moving the whole block
            # to GPU would force a full dequantization of every layer at once,
            # spiking VRAM and -- on the next swap -- a GPU->"CPU" round trip,
            # both of which defeat the mmap model's whole point.
            _backup_ggml_refs(cur_mod)
        # else: safetensor - vbar handles restoration
        self._loaded_swap_idx = local_idx

    def offload_swap_blocks(self):
        for i in range(self.non_swap_count, self.total_count):
            try:
                blk = self._modules[str(i)]
                if _has_ggml_params(blk):
                    # Restore the original mmap-backed GGMLTensor (pointer
                    # assignment, no anonymous RAM). If a block was never
                    # GPU-loaded the backup is empty and the helper safely
                    # falls back to a no-op .to(cpu).
                    _restore_ggml_refs(blk)
                else:
                    _free_to_meta(blk)
                for m in blk.modules():
                    for attr in ('_v', '_prefetch', '_v_signature'):
                        if hasattr(m, attr):
                            try:
                                delattr(m, attr)
                            except Exception:
                                pass
            except Exception:
                pass
        self._loaded_swap_idx = -1

    def _apply(self, fn, recurse=True):
        """Apply fn to non-swap blocks only.
        
        CRITICAL: Prevents model.to(device_to) from moving swap block
        GGMLTensors to GPU, which would cause a VRAM spike (12GB).
        Safetensor swap blocks are already meta (no-op), so this only
        affects GGUF paths.
        
        nn.ModuleList._apply(recurse=False) applies fn to all _modules
        entries INCLUDING swap blocks. We skip that and handle only
        non_swap_count blocks manually.
        """
        for i in range(self.non_swap_count):
            try:
                child = self._modules.get(str(i))
                if child is not None:
                    child._apply(fn, recurse)
            except Exception:
                pass
        return self

    def __getattr__(self, name):
        try:
            idx = int(name)
            if 0 <= idx < self.total_count:
                return self.__getitem__(idx)
        except (ValueError, TypeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, idx):
        # Support slicing: blocks[start:end]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_count)
            return [self[i] for i in range(start, stop, step)]

        if idx >= self.non_swap_count:
            self._load_swap(idx - self.non_swap_count)

        return super().__getitem__(idx)

    def __iter__(self):
        for idx in range(self.total_count):
            yield self.__getitem__(idx)


def install_block_swap(diffusion_model, compute_device, offload_device,
                       num_blocks=-1):
    all_containers = []
    for name in CONTAINER_NAMES:
        c = getattr(diffusion_model, name, None)
        if isinstance(c, (nn.ModuleList, list)) and len(c) > 0 and hasattr(c[0], "forward"):
            all_containers.append((name, c))

    if not all_containers:
        return None, lambda: None, set()

    first_swl = None
    all_names = set()

    for name, orig in all_containers:
        total = len(orig)
        n = num_blocks if num_blocks > 0 else total
        n = max(1, min(n, total))

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            non_swap_count=total - n,
        )
        swl.container_name = name
        setattr(diffusion_model, name, swl)
        all_names.add(name)
        if first_swl is None:
            first_swl = swl
        logger.info("UniBlockSwap: '%s' = %d blocks, swapping %d",
                     name, total, n)

        # For GGUF: the swap blocks already live in the mmap file-backed mapping
        # on CPU. No copy is needed now; we just record the original references
        # so a later offload can restore them (pointer assignment, no anon RAM).
        # Safetensor blocks stay on GPU (original behavior).
        for i in range(total - n, total):
            blk = swl._modules[str(i)]
            if _has_ggml_params(blk):
                _backup_ggml_refs(blk)

    orig_fwd = diffusion_model.forward

    def wrapped(*args, **kwargs):
        try:
            return orig_fwd(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize(compute_device)
                gc.collect()
                torch.cuda.empty_cache()

    diffusion_model.forward = wrapped

    def cleanup():
        diffusion_model.forward = orig_fwd
        for name, orig in all_containers:
            setattr(diffusion_model, name, orig)

    all_swls = []
    for name in CONTAINER_NAMES:
        c = getattr(diffusion_model, name, None)
        if hasattr(c, 'offload_swap_blocks'):
            all_swls.append(c)

    return first_swl, cleanup, all_names, all_swls


def find_te_containers(cond_stage_model):
    results = []
    seen_ids = set()

    def _recurse(module, depth=0):
        if depth > 20:
            return
        for name in CONTAINER_NAMES:
            c = getattr(module, name, None)
            if (isinstance(c, (nn.ModuleList, list)) and
                len(c) > 0 and hasattr(c[0], "forward") and
                id(c) not in seen_ids):
                seen_ids.add(id(c))
                results.append((name, c, module))
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ModuleList, list)):
                continue
            _recurse(child, depth + 1)

    _recurse(cond_stage_model)
    return results


def install_te_block_swap(cond_stage_model, compute_device, offload_device,
                          num_blocks=-1):
    containers = find_te_containers(cond_stage_model)

    if not containers:
        return [], lambda: None, set()

    mgr_list = []
    container_names = set()
    parent_to_mgrs = {}

    for name, orig, parent in containers:
        total = len(orig)
        n = num_blocks if num_blocks > 0 else total
        n = max(1, min(n, total))

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            non_swap_count=total - n,
        )
        swl.container_name = name
        setattr(parent, name, swl)
        mgr_list.append(swl)
        container_names.add(name)

        parent_id = id(parent)
        if parent_id not in parent_to_mgrs:
            parent_to_mgrs[parent_id] = (parent, parent.forward, [])
        parent_to_mgrs[parent_id][2].append(swl)

        logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, swapping %d",
                     name, type(parent).__name__, total, n)

        for i in range(total - n, total):
            blk = swl._modules[str(i)]
            if _has_ggml_params(blk):
                # Record mmap references; the block stays file-backed on CPU.
                _backup_ggml_refs(blk)
            else:
                _free_to_meta(blk)

    wrapped_parents = []
    for parent_id, (parent, orig_fwd, parent_mgrs) in parent_to_mgrs.items():
        def make_wrapped(_orig_fwd=orig_fwd, _mgrs=parent_mgrs, _cdevice=compute_device,
                         _root=cond_stage_model):
            def wrapped(*args, **kwargs):
                try:
                    return _orig_fwd(*args, **kwargs)
                finally:
                    for m in _mgrs:
                        m.offload_swap_blocks()
                    backup_cleaner = getattr(_root, '_uniblockswap_backup_cleanup', None)
                    patcher = getattr(_root, '_patcher_ref', None)
                    if backup_cleaner is not None and patcher is not None:
                        backup_cleaner(patcher)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(_cdevice)
                        gc.collect()
                        torch.cuda.empty_cache()
            return wrapped
        parent.forward = make_wrapped()
        wrapped_parents.append((parent, orig_fwd))

    def cleanup():
        for name, orig, parent in containers:
            current = getattr(parent, name, None)
            if hasattr(current, 'offload_swap_blocks'):
                setattr(parent, name, orig)
        for parent, orig_fwd in wrapped_parents:
            parent.forward = orig_fwd

    return mgr_list, cleanup, container_names