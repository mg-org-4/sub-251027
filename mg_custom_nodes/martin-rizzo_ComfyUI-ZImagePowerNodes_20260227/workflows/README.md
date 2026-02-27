
# Z-Image Power Nodes: Example Workflows

 * `/safetensors_workflows`: Folder containing these same workflows but pre-configured for use with .safetensors checkpoints.
 * `advanced__develop_and_test_new_styles.json`: Advanced workflow for testing and refining new styles through multiple prompts.
 * `advanced__zsampler_turbo_advanced.json`: Example using the "Z-Sampler Turbo (Advanced)" node to configue extra parameters.
 * `custom_styles.json`: Illustrates how to define personalized styles.
 * `inject_style_into_string.json`: Shows how to dynamically insert style text into a prompt string.
 * `inpainting.json`: Demonstrates inpainting techniques utilizing the "Z-Sampler Turbo" node.
 * `my_top_10_styles.json`: Example of building a custom list of favorite styles (easily expandable beyond 10).
 * `z-image_turbo_text2image_workflow.json`: Main reference workflow for text-to-image generation using the "Z-Image Power Nodes".


## Requirements

To utilize any of these workflows, you need to have ComfyUI configured and the
"Z-Image Power Nodes" installed. Additionally, ensure that the required checkpoints
(in GGUF or Safetensors format) are placed in the appropriate directories within
your ComfyUI setup.

If you choose to use GGUF-format checkpoints, it is necessary to install the
"ComfyUI-GGUF" nodes as well, since ComfyUI does not natively support GGUF files.
You can find information about these nodes at: https://github.com/city96/ComfyUI-GGUF

Below in this readme, you will find detailed instructions on how to install the
necessary nodes and checkpoints.

Please note that while my work with "Z-Image Power Nodes" was developed and
tested using the recommended checkpoints listed below, it may also work
correctly with other "Z-Image Turbo" checkpoints or when applying LoRA.
However, if custom modifications are made to the workflows or alternative
checkpoint combinations are used, I cannot guarantee 100% functionality in all
cases. It becomes your responsibility to determine the suitable configuration
for your customized setups.

The node pack includes an "Advanced" version of "Z-Sampler Turbo" that incorporates
additional parameters which may help it function better in your custom workflows.



## GGUF Checkpoints
> [!NOTE]
> <sub>
GGUF checkpoints tend to run slightly slower in ComfyUI. However, if you are building
a complex workflow that involves other models or using heavy LLMs with ollama, GGUF
files can help prevent system freezes and OOM errors during generation, especially
when VRAM is limited. For simple image generation workflows, a safetensors file
(though heavier) might be preferable. When working with GGUF in Z-Image, from my
experience, using the Q5_K_S quantization typically offers the best balance between
file size and prompt response. </sub>

__Recommended Z-Image Turbo checkpoints in GGUF format:__

- "z_image_turbo-Q5_K_S.gguf" [5.19 GB]
  [ Download ]( https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/z_image_turbo-Q5_K_S.gguf )  
  Local Directory: `ComfyUI/models/diffusion_models/`

- "Qwen3-4B.i1-Q5_K_S.gguf" [2.82 GB]
  [ Download ]( https://huggingface.co/mradermacher/Qwen3-4B-i1-GGUF/blob/main/Qwen3-4B.i1-Q5_K_S.gguf )  
  Local Directory: `ComfyUI/models/text_encoders/`

- "ae.safetensors" [335 MB]
  [ Download ]( https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/vae/ae.safetensors )  
  Local Directory: `ComfyUI/models/vae/`



## Safetensors Checkpoints
> [!NOTE]
> <sub>
Safetensors files are generally larger, but ComfyUI includes several built-in optimizations
to speed up generation even with limited VRAM. It's always a good idea to test the safetensors
checkpoints on your system to see how they perform. However, using safetensors in fp8 format
is strongly discouraged as it can significantly reduce quality. If you have an RTX 50 series
GPU based on Blackwell architecture, NVFP4 quantized safetensors could be a better choice.
</sub>

__Recommended Z-Image Turbo checkpoints in Safetensors format:__

  - "z_image_turbo_bf16.safetensors" [12.3 GB]
    [ Download ]( https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors )  
    Local Directory: `ComfyUI/models/diffusion_models/`

  - "qwen_3_4b.safetensors" [8.04 GB]
    [ Download ]( https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors )  
    Local Directory: `ComfyUI/models/text_encoders/`

  - "ae.safetensors" [335 MB]
    [ Download ]( https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/vae/ae.safetensors )  
    Local Directory: `ComfyUI/models/vae/`



## Z-Image Power Nodes

Z-Image Power Nodes can be installed via the ComfyUI Manager or downloaded from its respective repository.

__Installation via ComfyUI Manager (Recommended):__

 - Open ComfyUI and click on the "Manager" button to launch the "ComfyUI Manager Menu".
 - Within the ComfyUI Manager, locate and click on the "Custom Nodes Manager" button.
 - In the search bar, type "Z-Image Power Nodes".
 - Select the option from the search results and click the "Install" button.
 - Restart ComfyUI to ensure the changes take effect.

__Manual Installation:__

 For manual installation, follow the instructions provided in the GitHub repository of the project:
 - https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes


