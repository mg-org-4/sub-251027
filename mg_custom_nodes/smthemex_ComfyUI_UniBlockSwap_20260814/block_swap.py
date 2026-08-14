"""
UniBlockSwap - Universal block swap for ComfyUI (fixed-resident prefix).

num_blocks is the inference LOOP mechanism, not a loading mechanism:
  - num_blocks blocks form a PERMANENT-RESIDENT PREFIX: blocks 0..num_blocks-1
    are pushed into CUDA once at install time and stay resident for the whole
    inference (no transfers while the loop runs over them). The remaining
    blocks (num_blocks..total-1) stay on the ORIGINAL lazy path: the GGUF
    plugin dequantizes + transfers each layer inside forward() (overlapped
    with compute; ComfyUI vbar manages VRAM), no swap ops run on the tail.
    Everything (prefix included) is released when inference ends (ON_CLEANUP /
    offload_swap_blocks).
  -1 / <=0 -> prefix of 1 block
  1 <= N < total -> prefix of N blocks (N blocks resident up front)
  N >= total  -> no swap (entire model resident)

Load/unload per block type (the GGUF plugin's loading mechanism is UNCHANGED):
  Safetensor: swap blocks are filtered out of patcher._load_list, so they
  NEVER get a vbar _v alloc - ComfyUI's vbar cast path is not involved at
  all. They run on ComfyUI's PLAIN cast path (resolve_cast_module_with_vbar
  falls through when hasattr(s, "_v") is False), which transfers weights from
  CPU/mmap inside forward() every step. Prefix blocks are therefore preloaded
  the same way as GGUF: module.to(compute_device) moves the whole block onto
  CUDA up front, so the plain cast path sees weight.device == device and
  skips the transfer - zero per-layer stalls over the resident prefix. A
  reference to the original (mmap-backed) param data is kept for release.
  On unload the params are pointed back at the original data (pointer
  assignment, no anonymous RAM), same idea as the GGUF branch.
  GGUF: for every resident block (prefix or current tail) the whole block's
  quantized weights are moved onto CUDA up front (module.to(compute_device),
  GGMLTensor metadata kept), so the blocks are GPU-resident when the loop
  reaches them; the GGUF plugin's own forward-time dequantization then runs
  against resident data (no per-layer mmap->CUDA stall). On release the params
  are pointed back at the original mmap GGMLTensors (pointer assignment, no
  anonymous RAM).
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


def _is_ggml_tensor(t):
    """True if `t` carries GGMLTensor metadata (quantized or torch-compatible
    GGUF weight). Check both the object itself and its .data - nn.Parameter()
    on a Tensor subclass drops __init__ attrs, and torch.Tensor.data returns
    self for GGMLTensor (detach() -> self), so both paths are covered."""
    if t is None:
        return False
    if getattr(t, "tensor_type", None) is not None:
        return True
    try:
        d = t.data
        if d is not t and getattr(d, "tensor_type", None) is not None:
            return True
    except Exception:
        pass
    return False


def _has_ggml_params(module):
    """Check if module has GGMLTensor parameters (quantized GGUF weights)."""
    for p in module.parameters():
        if _is_ggml_tensor(p):
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
        t = param if _is_ggml_tensor(param) else param.data
        if _is_ggml_tensor(t):             # a GGMLTensor
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
    # any plain (non-GGML) params that _load_block's module.to() moved to CUDA
    # must come back to the offload device too
    offload = module.offload_device if hasattr(module, "offload_device") else "cpu"
    offload_t = torch.device(offload)
    for p in module.parameters(recurse=True):
        if p.device.type != offload_t.type:
            p.data = p.data.to(offload)
    # free the GPU copy of the now-unreferenced tensor
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _is_gguf_block(module):
    """A block is GGUF if we captured its original mmap GGMLTensor refs at
    install time. This flag - not _has_ggml_params - is the source of truth:
    a block whose params were .to(cuda)'d still counts as GGUF so load/offload
    take the pointer-assignment path instead of the vbar/meta path.
    """
    return bool(getattr(module, "_ggml_mmap_backup", None))


def _has_meta_params(module):
    """True if the block's params are still meta (model weights not staged
    yet - happens at install time, before ComfyUI loads the safetensor).
    Nothing can be transferred in that state, so preload must be skipped."""
    for p in module.parameters(recurse=True):
        if getattr(p, "device", None) is not None and p.device.type == "meta":
            return True
    return False


def _free_to_meta(module):
    """Free param data to meta tensor - NO CPU copy created.
    The module structure is preserved. next load() restores from backup."""
    for param in module.parameters(recurse=False):
        param.data = torch.empty(0, device='meta')


def _backup_param_refs(module):
    """Preserve the ORIGINAL (CPU/staged) param .data objects for a safetensor
    block before it is moved to CUDA, so unload can point params back without
    reallocation. Generic counterpart of _backup_ggml_refs for non-GGML params.
    """
    if getattr(module, "_param_ref_backup", None) is not None:
        return
    backup = {}
    for name, param in module.named_parameters(recurse=True):
        backup[name] = param.data
    module._param_ref_backup = backup


def _restore_param_refs(module):
    """Point params back at the original (CPU/staged) data and drop any GPU
    copies. Pointer assignment (p.data = orig) - NO anonymous heap allocation.
    No-op if the block was never backed up (still CPU)."""
    backup = getattr(module, "_param_ref_backup", None)
    if not backup:
        module.to(module.offload_device if hasattr(module, "offload_device") else "cpu")
        return
    params = dict(module.named_parameters(recurse=True))
    for name, orig in backup.items():
        p = params.get(name)
        if p is not None:
            p.data = orig
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


class SwappableModuleList(nn.ModuleList):
    """ModuleList with a fixed-resident prefix; tail stays on the lazy path.

    prefix_count (= num_blocks) blocks are pushed into CUDA once and stay
    resident for the whole inference (no sliding, no transfers while the loop
    runs over them). The tail blocks (prefix_count..total-1) are deliberately
    left untouched: eager per-step tail swapping re-transfers the whole tail on
    every loop pass (the loop restarts at block 0) with blocking transfers and
    empty_cache churn - slower than the plugin's lazy per-layer path. The
    plugin dequantizes + transfers tail layers inside forward(), overlapped
    with compute. Everything (prefix included) is released when inference ends
    via offload_swap_blocks() / ON_CLEANUP.
    """

    def __init__(self, modules, compute_device, offload_device, window_size=1):
        super().__init__(modules)
        self.compute_device = compute_device
        self.offload_device = offload_device
        self.window_size = max(1, window_size)
        self.total_count = len(modules)
        # num_blocks -> how many leading blocks are kept resident in CUDA for
        # the whole inference.
        self.prefix_count = min(self.window_size, self.total_count)
        # Kept at 0 for compatibility with the node file (cleanup / LoRA
        # traversal cover ALL blocks, prefix included).
        self.non_swap_count = 0
        self._prefix_loaded = False  # prefix blocks resident in CUDA?
        self.container_name = ''

    def _offload_block(self, idx):
        try:
            blk = self._modules[str(idx)]
            if _is_gguf_block(blk):
                # GGUF: restore the original mmap-backed GGMLTensor by pointer
                # assignment (no anon RAM, no GPU round trip). If the block was
                # never loaded the backup is empty and the helper safely falls
                # back to a no-op .to(cpu).
                _restore_ggml_refs(blk)
            else:
                # Safetensor: point params back at the original (CPU/staged)
                # data. Swap blocks never have a vbar _v (they are filtered out
                # of _load_list), so meta would leave them unrestorable - the
                # pointer assignment keeps the source data alive instead.
                _restore_param_refs(blk)
            # NOTE: no vbar state (_v/_prefetch/_v_signature) exists for swap
            # blocks - they were filtered out of _load_list, so they never got
            # a vbar alloc and run on the plain cast path instead.
        except Exception:
            pass

    def _preload_safetensor_block(self, idx, blk):
        """Explicitly preload a safetensor block into CUDA so the block is
        GPU-resident before the loop reaches it - same effect as the GGUF
        branch.

        NOTE: safetensor swap blocks NEVER get a vbar _v - they are filtered
        out of patcher._load_list by the node, so _v = vbar.alloc() inside
        patcher.load() never runs for them. They execute on ComfyUI's PLAIN
        cast path (resolve_cast_module_with_vbar falls through when
        hasattr(s, "_v") is False), which re-transfers weights from CPU/mmap
        inside forward() on every step. A bare blk.to(compute_device) is
        therefore exactly right here: once weight.device == device, the plain
        cast path skips the transfer and the block stays GPU-resident.

        Blocks whose params are still meta (model weights not staged yet, e.g.
        at install time) are skipped - the lazy cast path takes over and the
        next preload pass (after patcher.load() materializes the weights)
        actually transfers.
        """
        try:
            if _has_meta_params(blk):
                logger.info("UniBlockSwap: load block %d (safetensor, weights not staged - lazy)",
                            idx)
                return
            _backup_param_refs(blk)
            blk.to(self.compute_device)
            logger.info("UniBlockSwap: load block %d (safetensor, resident)", idx)
        except Exception as e:
            logger.warning("UniBlockSwap: safetensor preload block %d failed (%s) - "
                           "falling back to lazy cast", idx, e)

    def _load_block(self, idx):
        blk = self._modules[str(idx)]
        if _is_gguf_block(blk):
            # GGUF: transfer the WHOLE block onto CUDA UP FRONT, keeping every
            # GGMLTensor's metadata (tensor_type/tensor_shape/patches) intact.
            # We do NOT dequantize here - the GGUF plugin dequantizes inside
            # forward() (its shape logic must not be bypassed). Moving the
            # quantized weights ahead of the loop means the plugin's lazy
            # dequantization runs against CUDA-resident data: the per-layer
            # mmap->CUDA transfer inside the inference loop disappears.
            _backup_ggml_refs(blk)
            blk.to(self.compute_device)
            logger.info("UniBlockSwap: load block %d (GGUF, resident)", idx)
        else:
            # Safetensor: preload the whole block with a plain .to(cuda) so
            # the prefix is CUDA-resident like the GGUF branch (the plain cast
            # path then sees weight.device == device and skips the transfer).
            self._preload_safetensor_block(idx, blk)

    def load_prefix(self):
        """Push the permanent-resident prefix blocks (0..prefix_count-1) into
        CUDA in one go. Idempotent. Called at install time and again lazily on
        first access (e.g. after an ON_LOAD offload wiped the prefix)."""
        if self.prefix_count <= 0 or self._prefix_loaded:
            return
        logger.info("UniBlockSwap: preload prefix [0,%d) into CUDA (num_blocks=%d)",
                    self.prefix_count, self.window_size)
        for j in range(self.prefix_count):
            self._load_block(j)
        self._prefix_loaded = True

    def _ensure_window(self, idx):
        """Ensure the resident prefix is in CUDA.

        Prefix blocks (idx < prefix_count) are loaded once and stay resident
        for the whole inference - zero transfers during the loop. Tail blocks
        (idx >= prefix_count) are deliberately NOT touched: eagerly swapping
        them re-transfers the whole tail on every step (the loop restarts at
        block 0, so the previous step's last tail block must be released and
        re-loaded), with blocking transfers + empty_cache churn - slower than
        the plugin's lazy per-layer path. The plugin handles them inside
        forward() (dequant + transfer, overlapped with compute).
        """
        if idx < self.prefix_count:
            if not self._prefix_loaded:
                self.load_prefix()

    def offload_swap_blocks(self):
        """Release ALL blocks (prefix included) and reset swap state. Called
        when inference ends (ON_CLEANUP / TE forward finally / model unload).
        Tail blocks lazily dequantized by the plugin are pointed back at mmap
        too, so VRAM returns to baseline."""
        for i in range(self.total_count):
            self._offload_block(i)
        self.reset_swap_state()

    def reset_swap_state(self):
        """Forget residency state so the next inference re-preloads the prefix.
        Blocks are already released by the caller; this only resets flags."""
        self._prefix_loaded = False

    def _apply(self, fn, recurse=True):
        """Do NOT move any block with model.to(...).

        All blocks are swap-managed: safetensor blocks are meta (no-op) and
        GGUF blocks must stay mmap-backed on CPU (a .to(gpu) here would force
        a full dequantization VRAM spike). nn.ModuleList._apply would apply fn
        to every child - we skip that entirely.
        """
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

        if 0 <= idx < self.total_count:
            self._ensure_window(idx)

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
        return None, lambda: None, set(), []

    first_swl = None
    all_names = set()

    for name, orig in all_containers:
        total = len(orig)
        # num_blocks = number of leading blocks kept resident in CUDA for the
        # whole inference (prefix). -1 / <=0 -> 1; 1..total-1 -> N; >= total ->
        # no swap.
        win = num_blocks if num_blocks > 0 else 1
        win = max(1, min(win, total))
        if num_blocks > 0 and win >= total:
            logger.info("UniBlockSwap: '%s' = %d blocks, NO swap (num_blocks=%d >= total)",
                         name, total, num_blocks)
            continue

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            window_size=win,
        )
        swl.container_name = name
        setattr(diffusion_model, name, swl)
        all_names.add(name)
        if first_swl is None:
            first_swl = swl

        # Every block participates: the first `win` blocks form the resident
        # prefix (pushed into CUDA once, kept for the whole inference); the
        # rest stay on the plugin's original lazy path. GGUF blocks keep their
        # mmap refs (dequantized on demand by the GGUF plugin); safetensor
        # blocks are restored by vbar on access.
        n_gguf = 0
        for i in range(total):
            if _has_ggml_params(swl._modules[str(i)]):
                _backup_ggml_refs(swl._modules[str(i)])
                n_gguf += 1
        logger.info("UniBlockSwap: '%s' GGUF blocks: %d/%d", name, n_gguf, total)

        logger.info("UniBlockSwap: '%s' = %d blocks, prefix resident = %d, tail lazy (num_blocks=%d)",
                     name, total, swl.prefix_count, num_blocks)

        # Push the resident prefix into CUDA right away: blocks 0..win-1 are
        # GPU-resident before inference starts and stay there until it ends.
        swl.load_prefix()

    if first_swl is None:
        # num_blocks >= total for every container -> no swap at all.
        return None, lambda: None, set(), []

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
        win = num_blocks if num_blocks > 0 else 1
        win = max(1, min(win, total))
        if num_blocks > 0 and win >= total:
            logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, NO swap (num_blocks=%d >= total)",
                         name, type(parent).__name__, total, num_blocks)
            continue

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            window_size=win,
        )
        swl.container_name = name
        setattr(parent, name, swl)
        mgr_list.append(swl)
        container_names.add(name)

        parent_id = id(parent)
        if parent_id not in parent_to_mgrs:
            parent_to_mgrs[parent_id] = (parent, parent.forward, [])
        parent_to_mgrs[parent_id][2].append(swl)

        # Every block participates: the first `win` blocks form the resident
        # prefix (pushed into CUDA once, kept for the whole inference); the
        # rest stay on the plugin's original lazy path. GGUF blocks keep their
        # mmap refs (dequantized on demand by the GGUF plugin); safetensor
        # blocks are restored by vbar on access.
        n_gguf = 0
        for i in range(total):
            if _has_ggml_params(swl._modules[str(i)]):
                _backup_ggml_refs(swl._modules[str(i)])
                n_gguf += 1
        logger.info("UniBlockSwap: '%s' GGUF blocks: %d/%d", name, n_gguf, total)

        logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, prefix resident = %d, tail lazy (num_blocks=%d)",
                     name, type(parent).__name__, total, swl.prefix_count, num_blocks)

        # Push the resident prefix into CUDA right away. The TE forward wrapper
        # offloads everything (prefix included) when the TE run finishes.
        swl.load_prefix()

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
