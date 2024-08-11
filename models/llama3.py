# Import librraies
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_ollama.llms import OllamaLLM


'''
Class to run llama3 using different client
'''
class Llama3_1:
    def __init__(self, client: str='hf'):
        self.client = client
        if client == 'ollama':
            self.model = OllamaLLM(model="llama3.1")
        elif client == 'hf':
            cache_dir = os.path.join('hf_models', 'llama3')
            os.makedirs(cache_dir, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/llama-3-8b-Instruct-bnb-4bit",
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map='cuda'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "unsloth/llama-3-8b-Instruct-bnb-4bit",
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map='cuda'
            )
        else:
            raise ValueError("Invalid name of client for Llama!")

    def ollama_inference(self, prompt: str):
        return self.invoke(prompt)
    
    def hf_inference(self, prompt: str, max_new_tokens: int = 256):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        gen_text = self.tokenizer.batch_decode(outputs)[0]
        idx = gen_text.find('<|start_header_id|>assistant<|end_header_id|>')
        gen_text = gen_text[idx:]
        gen_text = gen_text.replace('<|start_header_id|>assistant<|end_header_id|>', '').replace('<|eot_id|>', '').strip()
        return gen_text
    
    def __call__(self, context: str, prompt: str, image_path: str = None, tax_vals: str = None, max_new_tokens : int = 256):
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)
        if self.client == 'ollama':
            return self.ollama_inference(prompt=formatted_prompt)
        elif self.client == 'hf':
            return self.hf_inference(prompt=formatted_prompt, max_new_tokens=max_new_tokens)
        else:
            raise ValueError("Invalid name of client for Llama!")
