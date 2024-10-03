# Import libraries
import time
from dotenv import load_dotenv
load_dotenv()
from cachetools import LRUCache
from cachetools import cached
import re
import json
import pandas as pd
from utils.string_utils import tokenize, stem_words
from utils.image_utils import encode_image
from openai import OpenAI



'''
Class which can do keyword search in a given context using stemming and tokenization
- configs:
    - lookup_table: A dictionary which has label and list of keywords as pair.
    - text_cols: Columns of the data should be considered for lookup.
'''
class KeywordStemLookup:
    def __init__(self, **configs):
        self.lookup_table = configs['lookup_table']
        # If any speciic columns of data needs to be checked, then the column name(s) should be passed as text_cols
        if 'text_cols' in configs:
            self.text_cols = configs['text_cols']
        else:
            self.text_cols = None
        self.cache = LRUCache(maxsize=32)
    
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        # Check inference mode
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Inference Mode - 'category' can't be used!")
        # Check that there is any preset text_cols or not
        if self.text_cols is None:
            self.text_cols = taxonomy_dict['default_text_cols']
        # Get context of the data
        context = self.get_context(data_dict, self.text_cols)
        # List to store result
        res = []
        # Tokenize context
        tokenized_text = tokenize(context)
        tokenized_text = list(set(stem_words(tokenized_text)))
        # Check for match 
        for k, vals in self.lookup_table.items():
            vals = list(set(stem_words(vals)))
            for v in vals:
                if v in tokenized_text:
                    res.append(k)
                    break
        # Filter invalid predictions 
        labels = [x.lower().strip() for x in taxonomy_dict['labels']]
        res = [x for x in res if x.lower().strip() in labels]
        # Return the result depending on the metadata
        end = time.time()
        if 'Single' in metadata_dict['Single Value / Multi Value']:
            return res[0] if len(res) >= 1 else 'Not specified', 0, 0, end - start
        else:
            return ','.join(res) if len(res) >= 1 else 'Not specified', 0, 0, end - start
            

'''
Class which can do keyword search in a given context.
- configs:
    - lookup_table: A dictionary which has label and list of keywords as pair.
    - text_cols: Columns of the data should be considered for lookup.
'''
class KeywordLookup:
    def __init__(self, **configs):
        self.lookup_table = configs['lookup_table']
        # If any speciic columns of data needs to be checked, then the column name(s) should be passed as text_cols
        if 'text_cols' in configs:
            self.text_cols = configs['text_cols']
        else:
            self.text_cols = None
        self.cache = LRUCache(maxsize=32)
    
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        # Check inference mode
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Inference Mode - 'category' can't be used!")
        # Check that there is any preset text_cols or not
        if self.text_cols is None:
            self.text_cols = taxonomy_dict['default_text_cols']
        # Get context of the data
        context = self.get_context(data_dict, self.text_cols)
        context = context.lower()
        # List to store result
        res = []
        # Check for match 
        for k, vals in self.lookup_table.items():
            for v in vals:
                pattern = r'\b' + re.escape(v.lower().strip()) + r'\b'
                if re.search(pattern, context):
                    res.append(k)
        # Filter invalid predictions 
        labels = [x.lower().strip() for x in taxonomy_dict['labels']]
        res = [x for x in res if x.lower().strip() in labels]
        # Return the result depending on the metadata
        end = time.time()
        if 'Single' in metadata_dict['Single Value / Multi Value']:
            return res[0] if len(res) >= 1 else 'Not specified', 0, 0, end - start
        else:
            return ','.join(res) if len(res) >= 1 else 'Not specified', 0, 0, end - start
                    
                    
class Lookup:
    def __init__(self, **configs):
        if 'text_cols' in configs:
            self.text_cols = configs['text_cols']
        else:
            self.text_cols = None
    
    def get_context(self, data: dict, cols: list):
        text = ''
        for col in cols:
            if not pd.isna(data[col]):
                text += f'{col} - {data[col]}\n'
        return text
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        # Check inference mode
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Inference Mode - 'category' can't be used!")
        # Check that there is any preset text_cols or not
        if self.text_cols is None:
            self.text_cols = taxonomy_dict['default_text_cols']
        # Get context of the data
        context = self.get_context(data_dict, self.text_cols)
        context = context.lower()
        # List to store result
        res = []
        # Check for match 
        for x in taxonomy_dict['labels']:
            pattern = r'\b' + re.escape(x.lower().strip()) + r'\b'
            if re.search(pattern, context):
                res.append(x.strip())
        end = time.time()
        # Return the result depending on the metadata
        if 'Single' in metadata_dict['Single Value / Multi Value']:
            return res[0] if len(res) >= 1 else 'Not specified', 0, 0, end - start
        else:
            return ', '.join(res) if len(res) >= 1 else 'Not specified', 0, 0, end - start
        
            
'''
The model can do a Direct Mapping of a input feature and output feature.
'''
class DirectLookup:
    def __init__(self, **configs):
        try:
            self.lookup_col =  configs['lookup_col']
        except Exception as err:
            print("For DiectLoopkup, 'lookup_col' must be passed as parameter!")
        
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Can't use this model in 'category' mode.")
        end = time.time()
        return data_dict[self.lookup_col] if not pd.isna(data_dict[self.lookup_col]) else 'Not specified', 0, 0, end - start
    


'''
The class can extract output feature from input feature. If the input feature is NaN then extract the feature using ChatGPT.
'''
class DirectLookupMeetsChatGPT:
    def __init__(self, **configs):
        try:
            self.client = OpenAI()
            self.lookup_col = configs['lookup_col']
            self.prompt_text_cols = configs['promt_text_cols'] if 'prompt_text_cols' in configs.keys() else None
            self.prompt_image_cols = configs['promt_image_cols'] if 'prompt_image_cols' in configs.keys() else None
        except Exception as err:
            print("For DirectLookupMeetsChatGPT, lookup_col is needed as attribute!")
        
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
        
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict):
        start = time.time()
        self.prompt = taxonomy_dict['prompt']
        self.prompt_text_cols = self.prompt_text_cols if self.prompt_text_cols is not None else taxonomy_dict['default_text_cols']
        self.prompt_image_cols = self.prompt_image_cols if self.prompt_image_cols is not None else taxonomy_dict['default_image_cols']
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Can't use this model in 'category' mode.")
        if not pd.isna(data_dict[self.lookup_col]):
            resp, input_tokens, output_tokens, latency = data_dict[self.lookup_col], 0, 0, time.time() - start
        else:
            context = self.get_context(data_dict, self.prompt_text_cols)
            images = self.get_images(data_dict, self.prompt_image_cols)
            payload = []
            for image in images:
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
            attribute = taxonomy_dict['breadcrumb'].split(' > ')[-1]
            labels = taxonomy_dict['labels']
            metadata = metadata_dict
            self.prompt = self.prompt.format(context=context, attribute=attribute, labels=labels, metadata=metadata)
            payload.append(
                {
                    "type": "text",
                    "text": self.prompt
                }
            )
            response = self.client.chat.completions.create(
                model = 'gpt-4o-mini',
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


'''
This class applied KeywordLookup followed by ChatGPT if the result from KWLookup is NaN
'''
class KeywordLookupMeetsChatGPT:
    def __init__(self, **configs):
        try:
            self.client = OpenAI()
            if 'text_cols' in configs:
                self.text_cols = configs['text_cols'].split(",")
            else:
                self.text_cols = None
            self.prompt_text_cols = configs['promt_text_cols'] if 'prompt_text_cols' in configs.keys() else None
            self.prompt_image_cols = configs['promt_image_cols'] if 'prompt_image_cols' in configs.keys() else None
        except Exception as err:
            print(err)
        
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
        
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict):
        start = time.time()
        self.prompt = taxonomy_dict['prompt']
        self.prompt_text_cols = self.prompt_text_cols if self.prompt_text_cols is not None else taxonomy_dict['default_text_cols']
        self.prompt_image_cols = self.prompt_image_cols if self.prompt_image_cols is not None else taxonomy_dict['default_image_cols']
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Can't use this model in 'category' mode.")

        res = []
        start = time.time()
        # Get context of the data
        if self.text_cols is None:
            self.text_cols = taxonomy_dict['default_text_cols']
        context = self.get_context(data_dict, self.text_cols)
        context = context.lower()
        # List to store result
        res = []
        # Check for match 
        for x in taxonomy_dict['labels']:
            pattern = r'\b' + re.escape(x.lower().strip()) + r'\b'
            if re.search(pattern, context):
                res.append(x.strip())
        end = time.time()

        if res != []:
            # Return the result depending on the metadata
            if 'Single' in metadata_dict['Single Value / Multi Value']:
                return res[0] if len(res) >= 1 else 'Not specified', 0, 0, end - start
            else:
                return ', '.join(res) if len(res) >= 1 else 'Not specified', 0, 0, end - start

        # if KeywordLookup result is empty
        else:
            print("Empty res from KWLookup, using GPT")
            context = self.get_context(data_dict, self.prompt_text_cols)
            images = self.get_images(data_dict, self.prompt_image_cols)
            payload = []
            for image in images:
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
            attribute = taxonomy_dict['breadcrumb'].split(' > ')[-1]
            labels = taxonomy_dict['labels']
            metadata = metadata_dict
            self.prompt = self.prompt.format(context=context, attribute=attribute, labels=labels, metadata=metadata)
            payload.append(
                {
                    "type": "text",
                    "text": self.prompt
                }
            )
            response = self.client.chat.completions.create(
                model = 'gpt-4o-mini',
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

'''
The class can do lookup based on some other attribute values predicted by other models.
'''
class CorrelationLookup:
    def __init__(self, **configs):
        pass
    
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict):
        return 'Not specified', 0, 0, 0