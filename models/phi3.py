# Import libraries
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


'''
Phi3 wrapper to use it as agent in workflow
'''
class Phi3:
    def __init__(self, max_new_tokens: int = 256):
        cache_dir = os.path.join('pretrained_models', 'phi3')
        os.makedirs(cache_dir, exist_ok=True)
        # Declare tokenizer, model and model
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda'
        )
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda',
            torch_dtype='auto'
        )
        self.pipeline = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
        )
        # Define generation configurations
        self.generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

    def __call__(self, prompt: str, image_url: str):
        messages = [ 
            {"role": "system", "content": "You are a helpful assistant who is an expert of inventory management."}, 
            {"role": "user", "content": prompt}, 
        ]
        gen_text = self.pipeline(messages, **self.generation_args)[0]['generated_text'].strip()
        return gen_text