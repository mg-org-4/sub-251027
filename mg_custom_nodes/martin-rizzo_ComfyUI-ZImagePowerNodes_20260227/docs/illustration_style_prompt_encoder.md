# Illustration-Style Prompt Encoder
![text](/docs/illustration_style_prompt_encoder.jpg)

Encodes the prompt using the text encoder (CLIP) model, generating embeddings that can guide the diffusion process. Similar to ComfyUI's native "CLIP Text Encode (Prompt)" node, but this one allows selecting a visual illustration style, ensuring the final generated image matches the chosen style.nado.

## Inputs

### clip

The text encoder (CLIP) model used to encode the text and generate embeddings.

### customization

Optional input, can remain disconnected. Allows users to configure styles according to their preferences by connecting a multi-line string that redefines one or more available styles. A style definition starts with ">>>" followed by the name of the style, then includes lines describing the template for that style. The template must include "{$@}" where the user's prompt will be inserted.

Example:
```
>>>Phone Photo
YOUR CONTEXT:
Your photographs has android phone cam-quality.
Your photographs exhibit surprising compositions, sharp complex backgrounds,
natural lighting, and candid moments that feel immediate and authentic.
Your photographs are actual gritty candid photographic background.
YOUR PHOTO:
{$@}


>>>Casual Photo
YOUR CONTEXT:
You are an amateur documentary photographer taking low quality photos.
Your photographs exhibit sharp backgrounds, unpolished realism with natural lighting,
and candid friendship-level moments that feel immediate and authentic.
YOUR PHOTO:
{$@}
```

### style_to_apply

Selects the desired style to apply or "none" if you don't want any predefined style.

### text

This is where you input your prompt. For better style application, avoid including messages in the prompt that might interfere with the selected style.

## Outputs

### conditioning

The encoded prompt embeddings with applied style, used to guide the model during generation.

### string

The prompt text after applying the selected style. Typically not used, but can be employed for advanced purposes.

