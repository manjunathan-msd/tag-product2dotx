# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer


'''
Phi3 wrapper to use it as agent in workflow
'''
class Phi3:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            trust_remote_code=True, 
            device_map='cuda'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            trust_remote_code=True, 
            device_map='cuda'
        )

    def post_processing(self, text: str):
        idx = text.find('<|assistant|>')
        text = text[idx:]
        text = text.replace('<|assistant|>', '').replace('<|end|>', '').strip()
        return text

    def __call__(self, context: str, prompt: str, tax_vals: str = None, max_new_tokens : int = 256):
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)
            inputs = self.tokenizer.apply_chat_template(formatted_prompt, add_generation_prompt=True, return_tensors="pt").to('cuda')
            outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
            text = self.tokenizer.batch_decode(outputs)[0]
            return self.post_processing(text)
