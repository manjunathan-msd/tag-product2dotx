# Import librraies
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from taxonomy_builder.utils import TaxonomyTree
from utils.string import *
from models import *


'''
The class can tag data when a TaxonomyTree and data is given
'''
class Tagger:
    def __init__(self, taxonomy: TaxonomyTree, **configs):
        self.taxonomy = taxonomy
        # Load model
        model_name = configs['model']['name']
        if configs['model']['parameters']:
            parameters = configs['model']['parameters']
        if model_name in globals():
            if len(parameters) == 0:
                self.model = globals()[model_name]
            else:
                self.model = globals()[model_name](**parameters)
        else:
            raise ValueError("Invalid LLM name!")
        self.mode = configs['inference']['level']
        # Read prompts
        self.prompts = {}
        with open(configs['prompts']['leaf_prompt_path']) as fp:
            self.prompts['leaf_prompt'] = fp.read()
        with open(configs['prompts']['non_leaf_prompt_path']) as fp:
            self.prompts['non_leaf_prompt'] = fp.read()
        with open(configs['prompts']['note']) as fp:
            self.prompts['note'] = fp.read()
        self.model_taxonomy = {}
        self.additional_prompts = {}
    
    def set(self, x, val):
        if x == 'model_taxonomy':
            self.model_taxonomy = val
        elif x == 'additional_prompts':
            self.additional_prompts = val
        else:
            raise ValueError("Invalid attribute to set!")
    
    def remove(self, x):
        if x == 'model_taxonomy':
            self.model_taxonomy = None
        elif x == 'additional_prompts':
            self.additional_prompts = None
        else:
            raise ValueError("Invalid attribute to remove!")

    def get_context(self, row: dict, text_cols: list):
        text = ''
        for col in text_cols:
            if not pd.isna(row[col]):         
                text += col + ': ' + row[col]
        return text
    
    def get_image(self, row: dict, image_col: str):
        image_url = row[image_col]
        return image_url 

    def tag(self, context: str, image_url: str = None):
        res = {}
        ptr, depth = self.taxonomy.get('root'), 0
        while ptr:
            labels = [x for x in ptr.get('labels')]
            if len(labels) == 1 and ptr.get('Classification / Extraction') == 'Classification':
                res[f'L{depth}'] = labels[0]
                ptr = ptr.get('children')[0]
            else:
                if ptr.get('Classification / Extraction') == 'Classification' and ptr.get('node_type') == 'category':
                    if self.mode == 'presets':
                        name = ptr.get('name')
                        if name in self.additional_prompts:
                            prompt = self.additional_prompts[name].format(context=context, labels=labels)
                        else:
                            prompt = self.prompts['non_leaf_prompt'].format(context=context, labels=labels)
                        if name in self.model_taxonomy:
                            if self.model_taxonomy[name] != 'LLM':
                                temp_model = globals()[self.model_taxonomy[name]]()
                                resp, input_tokens, output_tokens, latency = temp_model(prompt=prompt, image_url=image_url)
                            else:
                                resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                        else:
                            resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)   
                    else:
                        prompt = self.prompts['non_leaf_prompt'].format(context=context, labels=labels)
                        resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                    if resp == 'Not specified':
                        return res
                    label = get_most_similar(labels, resp)
                    res[f'L{depth}'] = {
                        'Label': label,
                        'Input Tokens': input_tokens,
                        'Output Tokens': output_tokens,
                        'Latency': latency
                    }
                    ptr = ptr.get('children')[labels.index(label)]
                    depth += 1
                elif ptr.get('Classification / Extraction') == 'NA' and ptr.get('node_type') == 'NA':
                    if self.mode == 'attribute':
                        temp = {}
                        for attr in ptr.get('children'):
                            attribute = attr.get('name').split(' > ')[-1].strip()
                            labels = attr.get('labels')
                            taxonomy = json.dumps(attr.get('metadata'), indent=2)
                            prompt = self.prompts['leaf_prompt'].format(taxonomy=taxonomy, context=context, note=self.prompts['note'], labels=labels, attribute=attribute)
                            resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                            if attr.get('Classification / Extraction') == 'Classification':
                                label = get_most_similar(labels, resp)
                            else:
                                label = resp
                            temp[attr.get('name').split(' > ')[-1]] = {
                                'Label': label,
                                'Input Tokens': input_tokens,
                                'Output Tokens': output_tokens,
                                'Latency': latency
                            }
                        res[f'L{depth}'] = temp
                        ptr = None
                    elif self.mode == 'category':
                        taxonomies = {}
                        attribute_label_map = {}
                        for attr in ptr.get('children'):
                            attribute = attr.get('name').split(' > ')[-1].strip()
                            labels = attr.get('labels')
                            taxonomy = attr.get('metadata')
                            taxonomies[attribute] = taxonomy
                            attribute_label_map[attribute] = labels
                        taxonomies = json.dumps(taxonomies, indent=4)
                        attribute_label_map = json.dumps(attribute_label_map, indent=4)
                        prompt = self.prompts['leaf_prompt'].format(taxonomy=taxonomies, context=context, note=self.prompts['note'], labels=attribute_label_map)
                        resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                        resp = postprocessing_llm_response(resp)
                        resp = text_to_dict(resp)
                        temp = {}
                        for x, y in resp.items():
                            temp[x] = {
                                'Label': y,
                                'Input Tokens': input_tokens // len(resp),
                                'Output Tokens': output_tokens // len(resp),
                                'Latency': latency / len(resp)
                            }
                        res[f'L{depth}'] = temp
                        ptr = None
                    elif self.mode == 'presets':
                        temp = {}
                        for attr in ptr.get('children'):
                            name = attr.get('name')
                            attribute = attr.get('name').split(' > ')[-1].strip()
                            labels = attr.get('labels')
                            taxonomy = json.dumps(attr.get('metadata'), indent=2)
                            if name in self.additional_prompts:
                                prompt = self.additional_prompts[name].format(taxonomy=taxonomy, context=context, note=self.prompts['note'], labels=labels, attribute=attribute)
                            else:
                                prompt = self.prompts['leaf_prompt'].format(taxonomy=taxonomy, context=context, note=self.prompts['note'], labels=labels, attribute=attribute)
                            if name in self.model_taxonomy:
                                if self.model_taxonomy[name] != 'LLM':
                                    temp_model = globals()[self.model_taxonomy[name]]()
                                    resp, input_tokens, output_tokens, latency = temp_model(prompt=prompt, image_url=image_url)
                                else:
                                    resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                            else:
                                # print("Attribute isn't mentioned in taxonomy. Using default model!")
                                resp, input_tokens, output_tokens, latency = self.model(prompt=prompt, image_url=image_url)
                            if attr.get('Classification / Extraction') == 'Classification':
                                label = get_most_similar(labels, resp)
                            else:
                                label = resp
                            temp[attr.get('name').split(' > ')[-1]] = {
                                'Label': label,
                                'Input Tokens': input_tokens,
                                'Output Tokens': output_tokens,
                                'Latency': latency
                            }
                        res[f'L{depth}'] = temp
                        ptr = None
                    else:
                        raise ValueError("Wrong value of inference mode!")
        return res          

    def __call__(self, df: pd.DataFrame, text_cols: list, image_col: str = None, note: str = None, 
                 model_taxonomy: dict =  None, additional_prompts: dict = None):
        if self.mode == 'presets':
            if model_taxonomy is None:
                raise AttributeError("Model Taxonomy is needed for 'presets' mode!")
            else:
                self.set('model_taxonomy', model_taxonomy)
            if additional_prompts is None:
                raise AttributeError("Additional prompts are needed for 'presets' mode!")
            else:
                self.set('additional_prompts', additional_prompts)
        if image_col is None:
            df['image_path'] = np.nan
            image_col = 'image_path'
        if note is None:
            with open('prompts/note.txt') as fp:
                note = fp.read()
        res = []
        for row in tqdm(df.to_dict(orient='records'), desc='Inferencing', total=df.shape[0]):
            image_url = self.get_image(row, image_col)
            context = self.get_context(row, text_cols)
            try:
                tags = self.tag(context, image_url)
                tags = json.dumps(tags, indent=4)
                row['results'] = tags
                res.append(row)
            except Exception as err:
                print(err)
        res = pd.DataFrame(res)
        if res[image_col].isnull().sum() == res.shape[0]:
            res.drop(columns=image_col, inplace=True)
        if self.mode == 'presets':
            self.remove('model_taxonomy')
            self.remove('additional_prompts')
        return res