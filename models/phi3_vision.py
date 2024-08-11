# import libraries
import os
from PIL import Image 
from transformers import AutoProcessor, AutoModelForCausalLM



'''
Phi3 model which can support also vision input.
'''
class Phi3Vision:
    def __init__(self):
        cache_dir = os.path.join('hf_models', 'phi3_vision')
        os.makedirs(cache_dir, exist_ok=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'
        ) # use _attn_implementation='eager' to disable flash attention device_map='cuda'
        processor = AutoProcessor.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda'
        ) 
    
    def __call__(self, context: str, prompt: str, image_path: str = None, tax_vals: str = None, max_new_tokens : int = 512):
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)
        formatted_prompt = formatted_prompt + '\n' + '<|image_1|>\n'
        image = Image.open(image_path)
        messages = [{"role": "user", "content": formatted_prompt}]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda")
        generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        return response
 

