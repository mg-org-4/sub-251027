import comfy
import torch
from comfy.ldm.flux.layers import timestep_embedding
from einops import rearrange, repeat
from torch import Tensor


def dreamo_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    # dreamo embedding
    dreamo_embedding = transformer_options.get("dreamo_embedding", None)
    if dreamo_embedding is not None:
        img = img + dreamo_embedding

    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[: ,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def dreamo_forward(self, x, timestep, context, y, guidance=None, control=None, transformer_options={}, **kwargs):
    bs, c, h, w = x.shape
    patch_size = self.patch_size
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    origin_img_len = img.shape[1]

    h_len = ((h + (patch_size // 2)) // patch_size)
    w_len = ((w + (patch_size // 2)) // patch_size)
    img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
    img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
    img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

    ref_conds = transformer_options.get('dreamo_ref_conds', [])
    if len(ref_conds) > 0:
        cum_h_len = h_len
        cum_w_len = w_len
        task_embedding = transformer_options.get('dreamo_task_embedding')
        idx_embedding = transformer_options.get('dreamo_idx_embedding')
        embeddings = repeat(task_embedding[1], "c -> n l c", n=bs, l=origin_img_len)
        for ref_idx, ref_cond in enumerate(ref_conds):
            ref_cond = ref_cond.to(img)
            ref_h_len = ref_cond.shape[2] // patch_size
            ref_w_len = ref_cond.shape[3] // patch_size
            ref_img = rearrange(ref_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            ref_img_ids = torch.zeros((ref_h_len, ref_w_len, 3), device=x.device, dtype=x.dtype)
            ref_img_ids[:, :, 1] = ref_img_ids[:, :, 1] + torch.linspace(cum_h_len, cum_h_len + ref_h_len - 1, steps=ref_h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
            ref_img_ids[:, :, 2] = ref_img_ids[:, :, 2] + torch.linspace(cum_w_len, cum_w_len + ref_w_len - 1, steps=ref_w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
            ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)
            cum_h_len += ref_h_len
            cum_w_len += ref_w_len
            img = torch.cat((img, ref_img), dim=1)
            img_ids = torch.cat((img_ids, ref_img_ids), dim=1)
            cur_task_embedding = repeat(task_embedding[0], "c -> n l c", n=bs, l=ref_img.shape[1])
            cur_idx_embedding = repeat(idx_embedding[ref_idx+1], "c -> n l c", n=bs, l=ref_img.shape[1])
            cur_embeddings = cur_task_embedding + cur_idx_embedding
            embeddings = torch.cat((embeddings, cur_embeddings), dim=1)
        transformer_options['dreamo_embedding'] = embeddings.to(img)


    out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))[:, :origin_img_len]
    return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]