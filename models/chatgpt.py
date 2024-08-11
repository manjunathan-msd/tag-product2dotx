# Import libraries
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()



'''
The class can call different model of chatgpt using different data types.
    - model_name: Model name of ChatGPT. Cureent supported: ['gpt-4o-mini']
    - data_type: Valid data types are 'text', 'image' and 'multimodal'.
    - prompt_path: Path of the prompt.
'''
class MetaChatGPT:
    def __init__(self, model_name: str, data_type: str, prompt_path: str):
        with open(prompt_path) as fp:
            self.prompt = fp.read()
        if model_name == 'gpt-4o-mini' and data_type == 'text':
            self.client = OpenAI()
            self.inference = self.gpt_4o_mini_text
        else:
            raise ValueError("Invalid Model name or Data Type!")
        
    def gpt_4o_mini_text(self, **kwargs):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": self.prompt.format(**kwargs)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response
    
    def __call__(self, **kwargs):
        return self.inference(**kwargs)