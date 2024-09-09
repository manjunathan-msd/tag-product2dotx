# Import libraries
import time
import os
import json
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, pipeline
from utils.image import *


'''
Phi3 wrapper to use it as agent in workflow
'''
class Phi3:
    def __init__(self, **configs):
        cache_dir = os.path.join('pretrained_models', 'phi3')
        os.makedirs(cache_dir, exist_ok=True)
        # Declare tokenizer, model and model
        self.tokenizer = AutoTokenizer.from_pretrained(
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
            torch_dtype='auto',
            attn_implementation='eager'
        )
        self.pipeline = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer, 
        )
        # Define generation configurations   
        self.generation_args = configs 
        
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        if taxonomy_dict['inference_mode'] == 'category':
            if taxonomy_dict['node_type'] == 'NA':
                prompt = taxonomy_dict['prompt']
                note = taxonomy_dict['note']
                context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
                labels = {k: taxonomy_dict[k]['labels'] for k in taxonomy_dict.keys() if isinstance(taxonomy_dict[k], dict)}
                prompt = prompt.format(metadata=json.dumps(metadata_dict, indent=4), context=context, note=note, labels=json.dumps(labels, indent=4))
            else:
                prompt = taxonomy_dict['prompt']
                context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
                labels = taxonomy_dict['labels']
                prompt = prompt.format(context=context, labels=labels)
        elif taxonomy_dict['inference_mode'] == 'attribute' or taxonomy_dict['inference_mode'] == 'presets':
            prompt = taxonomy_dict['prompt']
            note = taxonomy_dict['note']
            context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
            if taxonomy_dict['node_type'] == 'NA':
                attribute = taxonomy_dict['breadcrumb'].split('>')[-1].strip()
                labels = taxonomy_dict['labels']
                prompt = prompt.format(metadata=json.dumps(metadata_dict, indent=4), context=context, note=note, attribute=attribute, labels=labels)
            else:
                labels = taxonomy_dict['labels']
                prompt = prompt.format(context=context, labels=labels)
        else:
            raise ValueError("Invalid inference mode!")
        start = time.time()
        messages = [ 
            {"role": "system", "content": "You are a helpful assistant who is an expert of inventory management."}, 
            {"role": "user", "content": prompt}, 
        ]
        gen_text = self.pipeline(messages, **self.generation_args)[0]['generated_text'][-1]['content'].strip()
        input_tokens = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0])
        output_tokens = len(self.tokenizer(gen_text, return_tensors='pt')['input_ids'][0])
        end = time.time()
        latency = end - start
        return gen_text, input_tokens, output_tokens, latency
    

class Phi3Vision:
    def __init__(self, **configs):
        cache_dir = os.path.join('pretrained_models', 'phi3_vision')
        os.makedirs(cache_dir, exist_ok=True)
        # Declare model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'     # use _attn_implementation='eager' to disable flash attention, recommended is flash_attention_2
        )  
        self.processor = AutoProcessor.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        # Define generation configurations
        self.generation_args = configs 
    
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def get_images(self, data: dict, cols: list):
        images = []
        for col in cols:
            if not pd.isna(data[col]):
                try:
                    image = download(data[col])
                    images.append(image)
                except Exception as err:
                    pass
        return images
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        if taxonomy_dict['inference_mode'] == 'category':
            if taxonomy_dict['node_type'] == 'NA':
                prompt = taxonomy_dict['prompt']
                note = taxonomy_dict['note']
                context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
                labels = {k: taxonomy_dict[k]['labels'] for k in taxonomy_dict.keys() if isinstance(taxonomy_dict[k], dict)}
                prompt = prompt.format(metadata=json.dumps(metadata_dict, indent=4), context=context, note=note, labels=json.dumps(labels, indent=4))
                self.image_cols = taxonomy_dict['default_image_cols']
            else:
                prompt = taxonomy_dict['prompt']
                context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
                labels = taxonomy_dict['labels']
                prompt = prompt.format(context=context, labels=labels)
                self.image_cols = taxonomy_dict['default_image_cols']
        elif taxonomy_dict['inference_mode'] == 'attribute' or taxonomy_dict['inference_mode'] == 'presets':
            prompt = taxonomy_dict['prompt']
            note = taxonomy_dict['note']
            context = self.get_context(data_dict, taxonomy_dict['default_text_cols'])
            if taxonomy_dict['node_type'] == 'NA':
                attribute = taxonomy_dict['breadcrumb'].split('>')[-1].strip()
                labels = taxonomy_dict['labels']
                prompt = prompt.format(metadata=json.dumps(metadata_dict, indent=4), context=context, note=note, attribute=attribute, labels=labels)
                self.image_cols = taxonomy_dict['default_image_cols']
            else:
                labels = taxonomy_dict['labels']
                prompt = prompt.format(context=context, labels=labels)
                self.image_cols = taxonomy_dict['default_image_cols']
        else:
            raise ValueError("Invalid inference mode!")
        start = time.time()
        # Message template
        payload = [ 
            {"role": "system", "content": "You are a helpful assistant who is an expert of inventory management."}
        ]
        images = self.get_images(data_dict, self.image_cols)
        for i, _ in enumerate(images):
            payload.append({"role": "user", "content": f"<|image_{i+1}|>\nExtract useful information from the image to perform the task described below."})
        payload.append({"role": "user", "content": prompt})
        # Prepare inputs
        prompt = self.processor.tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda") 
        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **self.generation_args
        ) 
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        end = time.time()
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(generate_ids[0])
        latency = end - start
        return response, input_tokens, output_tokens, latency

        
