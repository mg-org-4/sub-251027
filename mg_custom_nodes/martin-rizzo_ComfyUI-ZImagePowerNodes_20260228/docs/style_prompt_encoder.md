# Style & Prompt Encoder
![text](/docs/style_prompt_encoder.jpg)

Encodes the result of fusing a prompt with a visual style, generating embeddings that guide the diffusion process. This node is similar to ComfyUI's native "CLIP Text Encode (Prompt)", but it includes the option to select a visual style for the generated image.

## Inputs

### clip
The text encoder model used to encode the prompt and style.

### customization
Optional input that can remain disconnected. Allows users to configure styles according to their preferences by connecting a multi-line string that redefines one or more available styles. Each style definition starts with ">>>" followed by the name of the style, then includes lines describing the template for that style. The template must include "{$@}" where the user's prompt will be inserted.

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

### style
Selects the desired style to apply or "none" if you prefer not to use any predefined style.

### "Select Style..."
....

### text
This is where you input your prompt. For better style application, avoid including messages in the prompt that might interfere with the selected style.

## Outputs

### conditioning
The encoded prompt embeddings with applied style, used to guide the model during generation.

### string
The prompt text after applying the selected style. Typically not used, but can be employed for advanced purposes.

