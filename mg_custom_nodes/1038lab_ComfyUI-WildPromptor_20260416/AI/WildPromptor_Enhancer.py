import os
import torch
import folder_paths
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download

MODEL_PATH = os.path.join(folder_paths.models_dir, "LLM", "Prompt-Enhance")
os.makedirs(MODEL_PATH, exist_ok=True)

class WildPromptor_Enhancer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = "1038lab/Prompt-Enhance"
        print(f"Using device: {self.device}")
        
        if not os.listdir(MODEL_PATH):
            print(f"Downloading {self.model_checkpoint} model...")
            try:
                snapshot_download(
                    repo_id=self.model_checkpoint,
                    local_dir=MODEL_PATH,
                    local_dir_use_symlinks=False
                )
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        try:
            print("Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            
            self.pipe = pipeline(
                'text2text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                repetition_penalty=1.2,
                device=self.device
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
        self.max_target_length = 512
        self.prefix = "enhance prompt: "

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "Input prompt to be enhanced"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 20, "tooltip": "Number of enhanced prompts to generate"}),
                "combine_output": ("BOOLEAN", {"default": False, "tooltip": "Combine all outputs into one string or output as separate records"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "enhancer"
    CATEGORY = "🧪AILab/🤖AI"
    class_type = "WildPromptor_Enhancer"

    def enhancer(self, prompt, seed, batch_size, combine_output):
        if not prompt or prompt.isspace():
            return ([],)
            
        enhanced_prompts = []
        input_text = self.prefix + prompt
        
        try:
            for i in range(batch_size):
                output_seed = seed + i if seed != 0 else 0
                torch.manual_seed(output_seed)
                
                do_sample = output_seed != 0
                temperature = 0.7 if do_sample else 0.0

                result = self.pipe(
                    input_text,
                    max_length=self.max_target_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    num_return_sequences=1,
                    top_k=50,
                    top_p=0.95,
                )
                
                enhanced_prompts.append(result[0]['generated_text'])
                
        except Exception as e:
            print(f"Error during prompt enhancement: {str(e)}")
            return ([f"Error: {str(e)}"],)
            
        if combine_output:
            return (["\n---\n".join(enhanced_prompts)],)
        else:
            return (enhanced_prompts,)

NODE_CLASS_MAPPINGS = {
    "WildPromptor_Enhancer": WildPromptor_Enhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WildPromptor_Enhancer": "🤖 Prompt Enhancer"
}