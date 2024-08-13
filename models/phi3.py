# Import libraries
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


'''
Phi3 wrapper to use it as agent in workflow
'''
class Phi3:
    def __init__(self):
        cache_dir = os.path.join('pretrained_models', 'phi3')
        os.makedirs(cache_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda'
        )

    def post_processing(self, text: str):
        idx = text.find('<|assistant|>')
        text = text[idx:]
        text = text.replace('<|assistant|>', '').replace('<|end|>', '').strip()
        return text

    def __call__(self, prompt: str, image_url: str, max_new_tokens : int = 256):
        inputs = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to('cuda')
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        gen_text = self.tokenizer.batch_decode(outputs)[0]
        return self.post_processing(gen_text)