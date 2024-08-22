# Import libraries
import time
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
from openai import OpenAI
from utils.image import encode_image, download



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
    def __init__(self, **configs):
        # Get model name of ChatGPT
        self.model_name = configs['model_name']
        if 'text_cols' in configs:
            self.text_cols = [x.strip() for x in configs['text_cols'].split(',')]
        else:
            self.text_cols = None
        if 'image_cols' in configs:
            self.image_cols = [x.strip() for x in configs['image_cols'].split(',')]
        else:
            self.image_cols = None
        self.client = OpenAI()
    
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def get_images(self, data: dict, cols: list):
        images = []
        for col in cols:
            if not pd.isna(col):
                try:
                    image = encode_image(data[col])
                    images.append(image)
                except Exception as err:
                    pass
        return images
        
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        prompt = taxonomy_dict['prompt']
        note = taxonomy_dict['note']
        context = self.get_context(data_dict, taxonomy_dict['default_text_cols'] if self.text_cols is None else self.text_cols)
        images = self.get_images(data_dict, taxonomy_dict['default_image_cols'] if self.image_cols is None else self.image_cols)
        if taxonomy_dict['inference_mode'] == 'attribute' or taxonomy_dict['inference_mode'] == 'presets':
            if taxonomy_dict['node_type'] == 'NA':
                attribute = taxonomy_dict['breadcrumb'].split('>')[-1].strip()
                labels = taxonomy_dict['labels']
                prompt = prompt.format(metadata=json.dumps(metadata_dict, indent=4), context=context, note=note, attribute=attribute, labels=labels)
            else:
                labels = taxonomy_dict['labels']
                prompt = prompt.format(context=context, labels=labels)
        elif taxonomy_dict['inference_mode'] == 'category':
            pass
        else:
            raise ValueError("Invalid inference mode!")
        start = time.time()
        payload = []
        images = self.get_images(data_dict, taxonomy_dict['default_image_cols'])
        for image in images:
            if self.model_name in ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo']:
                payload.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                            "detail": "high"
                        }
                    }
                )
                payload.append(
                    {
                        "type": "text",
                        "text": "Extract the useful information from the image and keep it in mind to answer the questions defined below."
                    }
                )
        payload.append(
            {
                "type": "text",
                "text": prompt
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



