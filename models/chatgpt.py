# Import libraries
import time
from dotenv import load_dotenv
load_dotenv()
import json
from openai import OpenAI
from utils.image import encode_image



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
    
    
'''
The class can do inference using both image and texts.
It suppport gpt-4o-mini, gpt-4o and gpt-3.5
- model_name: Name of the model which will be used for inferencing
'''     
class ChatGPT:
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name
        self.client = OpenAI()
        
    def __call__(self, prompt: str, image_url: str = None):
        start = time.time()
        payload = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if image_url and self.model_name in ['gpt-4o-mini', 'gpt-4o']:
            try:
                image_url = encode_image(image_url)
            except Exception as err:
                print("Image URL is not accessible!")
                return 'Not specified', 'NA', 'NA', 'NA'
            payload.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_url}",
                        "detail": "high"
                    }
                }
            )
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = [
                {
                    "role": "user",
                    "content": payload
                }
            ]
        )
        end = time.time()
        response = response.json()
        response = json.loads(response)
        resp, input_tokens, output_tokens, latency = response['choices'][0]['message']['content'], response['usage']['prompt_tokens'], response['usage']['completion_tokens'], end - start
        return resp, input_tokens, output_tokens, latency