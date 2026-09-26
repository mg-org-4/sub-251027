from contextvars import ContextVar
from functools import partial

import torch

import comfy.ops


SCOPE = "krea2t_character_lora_image_only"
ACTIVE = SCOPE + ".active"
CURRENT = ContextVar(SCOPE, default=None)


def loaded_linear(weight):
    layer = comfy.ops.manual_cast.Linear(weight.shape[1], weight.shape[0],
                                         bias=False, device="meta", dtype=weight.dtype)
    layer.weight = torch.nn.Parameter(weight)
    return layer


class Factors(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        up, down = adapter.weights[:2]
        self.down = loaded_linear(down)
        self.up = loaded_linear(up)

    def forward(self, x):
        return self.up(self.down(x))


class FactorBank(torch.nn.Module):
    def __init__(self, adapters, compute_dtype):
        super().__init__()
        self.factors = torch.nn.ModuleList(Factors(adapter) for adapter in adapters.values())
        self.manual_cast_dtype = compute_dtype

    def get_dtype(self):
        return self.manual_cast_dtype


def scope_forward(executor, x, timesteps, context, attention_mask=None,
                  ref_latents=None, transformer_options=None, **kwargs):
    options = transformer_options if transformer_options is not None else {}
    token = CURRENT.set({"active": options.get(ACTIVE, ()), "block": None, "rows": None})
    try:
        return executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
    finally:
        CURRENT.reset(token)


class ImageInjection:
    def __init__(self, owner, targets, strength):
        self.owner = owner
        self.targets = targets
        self.strength = strength
        self.handles = []

    def on_model_patcher_clone(self):
        return ImageInjection(self.owner, self.targets, self.strength)

    def before_block(self, index, module, args, kwargs):
        frame = CURRENT.get()
        if frame is None or self.owner not in frame["active"]:
            return
        options = kwargs.get("transformer_options", {})
        bounds = options.get("img_slice")
        if bounds is None or len(bounds) != 2:
            raise RuntimeError("Krea2 did not provide the text/image token boundary for the character LoRA.")
        start, end = map(int, bounds)
        if not 0 <= start <= end == args[0].shape[1]:
            raise RuntimeError("Krea2's text/image token boundary does not match the current block input.")
        frame["block"] = index
        frame["rows"] = start, end

    def add_image_delta(self, index, factor, module, args, output):
        frame = CURRENT.get()
        if frame is None or self.owner not in frame["active"]:
            return output
        if frame["block"] != index or frame["rows"] is None:
            raise RuntimeError("Character LoRA ran outside its shared DiT block scope.")
        start, end = frame["rows"]
        x = args[0]
        if x.ndim != 3 or output.ndim != 3 or x.shape[1] != end or output.shape[1] != end:
            raise RuntimeError("Character LoRA requires Krea2's [batch, tokens, features] linear inputs.")
        if start != end:
            delta = factor(x[:, start:end])
            output[:, start:end].add_(delta.to(dtype=output.dtype), alpha=self.strength)
        return output

    def inject(self, patcher):
        if self.handles:
            return
        bank = patcher.get_additional_models_with_key(self.owner)[0].model
        blocks = set()
        try:
            for key, factor in zip(self.targets, bank.factors):
                parts = key.split(".")
                index = int(parts[2])
                if index not in blocks:
                    block = patcher.model.get_submodule(".".join(parts[:3]))
                    self.handles.append(block.register_forward_pre_hook(
                        partial(self.before_block, index), with_kwargs=True))
                    blocks.add(index)
                module = patcher.model.get_submodule(key.removesuffix(".weight"))
                self.handles.append(module.register_forward_hook(partial(self.add_image_delta, index, factor)))
        except BaseException:
            self.eject()
            raise

    def eject(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def inject(patcher, owner):
    patcher.get_attachment(owner).inject(patcher)


def eject(patcher, owner):
    patcher.get_attachment(owner).eject()
