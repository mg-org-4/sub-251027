import folder_paths
from optimum.quanto import freeze, qint8, quantize
import os
import torch
from .pipeline_dreamomni2 import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from .utils.vprocess import process_vision_info, resizeinput
from PIL import Image
import time
from diffusers.utils import load_image
import comfy.utils
import uuid

def infer_vlm(vlm_model, processor, input_img_path, input_instruction, prefix):
    tp=[]
    for path in input_img_path:
        tp.append({"type": "image", "image": path})
    tp.append({"type": "text", "text": input_instruction+prefix})
    messages = [
            {
                "role": "user",
                "content": tp,
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    vlm_model.to("cuda")
    generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    vlm_model.to("cpu")
    return output_text[0]

def tensor_2_pil(img_tensor):
    i = 255. * img_tensor.squeeze().cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

def tensor_2_path(img):
    img_path = os.path.join(folder_paths.get_temp_directory(), f'{uuid.uuid4()}.png')
    tensor_2_pil(img).save(img_path)
    return img_path

def extract_gen_content(text):
    text = text[6:-7]
    
    return text

class RH_DreamOmni2_Base_Pipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    FUNCTION = "load"

    CATEGORY = "Runninghub/DreamOmni2"

    def load(self, **kwargs):
        base_path = os.path.join(folder_paths.models_dir, 'flux', 'FLUX.1-Kontext-dev')
        vlm_path = os.path.join(folder_paths.models_dir, 'DreamOmni2', 'vlm-model')
        pipeline = DreamOmni2Pipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
        pipeline.enable_model_cpu_offload()
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path, torch_dtype="bfloat16", device_map="cpu"
        )
        processor = AutoProcessor.from_pretrained(vlm_path)
        pipeline.vlm_model = vlm_model
        pipeline.processor = processor
        return (pipeline, )

class RH_DreamOmni2_Gen_Pipeline(RH_DreamOmni2_Base_Pipeline):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RHDreamOmni2GenPipeline",)
    RETURN_NAMES = ("DreamOmni2 Gen Pipeline",)
    FUNCTION = "load"

    CATEGORY = "Runninghub/DreamOmni2"
    OUTPUT_NODE = True

    def load(self, **kwargs):
        gen_lora_path = os.path.join(folder_paths.models_dir, 'DreamOmni2', 'gen_lora')
        pipeline = super().load(**kwargs)[0]
        pipeline.load_lora_weights(
            gen_lora_path,
            adapter_name="generation"  
        )
        pipeline.set_adapters(["generation"], adapter_weights=[1])   

        quantize(pipeline.transformer, qint8)
        freeze(pipeline.transformer)
        return (pipeline, )
    
class RH_DreamOmni2_Generator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RHDreamOmni2GenPipeline", ),
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True,
                                      'default': ''}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "num_inference_steps": ("INT", {"default": 30}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
            "optional": {
                "ref_image_2": ("IMAGE", ),
                "ref_image_3": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"
    CATEGORY = "Runninghub/DreamOmni2"

    def sample(self, **kwargs):
        pipeline = kwargs.get('pipeline')
        prompt = kwargs.get('prompt')
        ref_image = kwargs.get('ref_image')
        ref_image_2 = kwargs.get('ref_image_2', None)
        ref_image_3 = kwargs.get('ref_image_3', None)
        width = kwargs.get('width')
        height = kwargs.get('height')
        guidance_scale = kwargs.get('guidance_scale')
        num_inference_steps = kwargs.get('num_inference_steps')

        self.pbar = comfy.utils.ProgressBar(num_inference_steps + 2)

        input_img_path = []
        input_img_path.append(tensor_2_path(ref_image))
        if ref_image_2 != None:
            input_img_path.append(tensor_2_path(ref_image_2))
        if ref_image_3 != None:
            input_img_path.append(tensor_2_path(ref_image_3))
        print(input_img_path)
        prefix=" It is generation task."
        source_imgs = []
        for path in input_img_path:
            img = load_image(path)
            source_imgs.append(resizeinput(img))

        self.update()

        prompt = infer_vlm(pipeline.vlm_model, pipeline.processor, input_img_path, prompt, prefix)
        prompt = extract_gen_content(prompt)

        self.update()

        image = pipeline(
            images=source_imgs,
            height=height,
            width=width,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            update_func=self.update,
        ).images[0]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )
    
    def update(self):
        self.pbar.update(1)

class RH_DreamOmni2_Edit_Pipeline(RH_DreamOmni2_Base_Pipeline):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RHDreamOmni2EditPipeline",)
    RETURN_NAMES = ("DreamOmni2 Edit Pipeline",)
    FUNCTION = "load"

    CATEGORY = "Runninghub/DreamOmni2"

    def load(self, **kwargs):
        edit_lora_path = os.path.join(folder_paths.models_dir, 'DreamOmni2', 'edit_lora')
        pipeline = super().load(**kwargs)[0]
        pipeline.load_lora_weights(
            edit_lora_path,
            adapter_name="edit" 
        )
        pipeline.set_adapters(["edit"], adapter_weights=[1])  

        quantize(pipeline.transformer, qint8)
        freeze(pipeline.transformer)
        return (pipeline, )
    
class RH_DreamOmni2_Editor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RHDreamOmni2EditPipeline", ),
                "src_image": ("IMAGE", ),
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True,
                                      'default': ''}),
                "num_inference_steps": ("INT", {"default": 30}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"
    CATEGORY = "Runninghub/DreamOmni2"

    def sample(self, **kwargs):
        pipeline = kwargs.get('pipeline')
        prompt = kwargs.get('prompt')
        src_image = kwargs.get('src_image')
        ref_image = kwargs.get('ref_image')
        guidance_scale = kwargs.get('guidance_scale')
        num_inference_steps = kwargs.get('num_inference_steps')

        self.pbar = comfy.utils.ProgressBar(num_inference_steps + 2)

        input_img_path = []
        input_img_path.append(tensor_2_path(src_image))
        input_img_path.append(tensor_2_path(ref_image))
        print(input_img_path)
        prefix=" It is editing task."
        source_imgs = []
        for path in input_img_path:
            img = load_image(path)
            source_imgs.append(resizeinput(img))

        self.update()

        prompt = infer_vlm(pipeline.vlm_model, pipeline.processor, input_img_path, prompt, prefix)
        prompt = extract_gen_content(prompt)

        self.update()

        image = pipeline(
            images=source_imgs,
            height=source_imgs[0].height,
            width=source_imgs[0].width,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            update_func=self.update,
        ).images[0]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )
    
    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub DreamOmni2 Gen Pipeline": RH_DreamOmni2_Gen_Pipeline,
    "RunningHub DreamOmni2 Edit Pipeline": RH_DreamOmni2_Edit_Pipeline,
    "RunningHub DreamOmni2 Generator": RH_DreamOmni2_Generator,
    "RunningHub DreamOmni2 Editor": RH_DreamOmni2_Editor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub DreamOmni2 Gen Pipeline": "RunningHub DreamOmni2 Gen Pipeline",
    "RunningHub DreamOmni2 Edit Pipeline": "RunningHub DreamOmni2 Edit Pipeline",
    "RunningHub DreamOmni2 Generator": "RunningHub DreamOmni2 Generator",
    "RunningHub DreamOmni2 Editor": "RunningHub DreamOmni2 Editor",
} 