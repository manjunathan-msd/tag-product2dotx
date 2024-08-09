# Import librraies
from langchain_ollama.llms import OllamaLLM


'''
Class to run llama3 using different client
'''
class Llama3_1:
    def __init__(self, client: str='ollama'):
        self.client = client
        if client == 'ollama':
            self.model = OllamaLLM(model="llama3.1")

    def ollama_inference(self, prompt):
        return self.invoke(prompt)
    
    def __call__(self, context: str, prompt: str, image_path: str = None, tax_vals: str = None, max_new_tokens : int = 256):
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)
        if self.client == 'ollama':
            return self.ollama_inference(prompt=formatted_prompt)
    
