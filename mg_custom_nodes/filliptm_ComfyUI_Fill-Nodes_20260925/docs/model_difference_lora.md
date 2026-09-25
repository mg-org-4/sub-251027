# FL Model Difference to LoRA

Connect two **Load Diffusion Model** nodes: the fine-tune to `finetuned_model`, and its original base to `base_model`. Queue **FL Model Difference to LoRA** to extract `fine-tune - base` and save a numbered `.safetensors` file under the first configured LoRA directory. The `filename_prefix` can include a subfolder, such as `Krea2/Dirtyrealism_difference`.

Rank defaults to 128. Higher ranks preserve more of the weight difference and produce larger files. `auto` uses ComfyUI's selected device; choose `cpu` if the device lacks SVD support or available memory. Extraction processes one parameter at a time, honors weight patches and quantization scales, and leaves the input models unchanged. Biases and other non-matrix parameters are stored as full differences in ComfyUI's LoRA format. No CLIP extraction is performed.

Apply the exported file to the same base with **Load LoRA (Model Only)** at strength 1. Low-rank compression is approximate, and quantized source models include quantization differences. Logged retained weight energy measures numerical reconstruction, not image quality. Compare generations before replacing a fine-tune in production workflows.

The included [Krea2 workflow](krea2_difference_lora.json) selects `krea2SATDirtyrealism_v5_fp8` and `krea2_turbo_fp8_scaled`, saving into `loras/Krea2`. Restart ComfyUI after installing the node, then open the workflow.
