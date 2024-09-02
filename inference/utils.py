# Import librraies
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from taxonomy_builder.utils import TaxonomyTree
from utils.string import *
from utils.misc import *
from models import *


'''
The class can tag data when a TaxonomyTree and data is given
'''
class Tagger:
    def __init__(self, taxonomy: TaxonomyTree, **configs):
        # Get the taxonomy
        self.taxonomy = taxonomy
        # Load deafult models and it's configs
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
        # Load inference configs
        self.mode = configs['inference']['level']
        self.text_cols = [x.strip() for x in configs['inference']['text_cols'].split(',')] if configs['inference']['text_cols'] is not None else []
        self.image_cols = [x.strip() for x in configs['inference']['image_cols'].split(',')] if configs['inference']['image_cols'] is not None else []
        # Read default prompts
        self.prompts = {}
        with open(configs['prompts']['leaf_prompt_path']) as fp:
            self.prompts['leaf_prompt'] = fp.read()
        with open(configs['prompts']['non_leaf_prompt_path']) as fp:
            self.prompts['non_leaf_prompt'] = fp.read()
        with open(configs['prompts']['note']) as fp:
            self.prompts['note'] = fp.read()
        # Dictionary to store preset models and taxonomy
        self.model_taxonomy = {}
        self.additional_prompts = {}
    
    # Function to set parameter of the object
    def set(self, x, val):
        if x == 'model_taxonomy':
            self.model_taxonomy = val
        elif x == 'additional_prompts':
            self.additional_prompts = val
        else:
            raise ValueError("Invalid attribute to set!")
    
    # Function to remove paramter of the object
    def remove(self, x):
        if x == 'model_taxonomy':
            self.model_taxonomy = None
        elif x == 'additional_prompts':
            self.additional_prompts = None
        else:
            raise ValueError("Invalid attribute to remove!")

    # Tag data
    def tag(self, data_dict: dict):
        # Store the tags
        res = {}
        # Initialize starting node and depth
        ptr, depth = self.taxonomy.get('root'), 0
        while ptr:
            # Get labels of current node
            labels = ptr.get('labels')
            # If number of labels is 1 and the task is a Classification task then store the label as the answer
            if len(labels) == 1 and ptr.get('Classification / Extraction') == 'Classification':
                res[f'L{depth}'] = labels[0]
                ptr = ptr.get('children')[0]
            else:
                # If the current node is a non-leaf node
                if ptr.get('Classification / Extraction') == 'Classification' and ptr.get('node_type') == 'category':
                    # Get the breadcrumb of the node
                    name = ptr.get('name')
                    # Get the prompt for the node
                    if self.mode == 'presets' and name in self.additional_prompts:
                        prompt = self.additional_prompts[name]
                    else:
                        prompt = self.prompts['non_leaf_prompt']
                    # Prepare taxonomy_dict and metadata_dict
                    taxonomy_dict = {
                        'breadcrumb': name,
                        'labels': labels,
                        'prompt': prompt,
                        'note': self.prompts['note'],
                        'default_text_cols': self.text_cols,
                        'default_image_cols': self.image_cols,
                        'node_type': ptr.get('node_type'),
                        'inference_mode': self.mode
                    }
                    metadata_dict = ptr.get('metadata')
                    # Get the model for the node and use the model for inference
                    if self.mode == 'presets' and name in self.model_taxonomy:
                        # Get the preset model name and it's parameters
                        additional_model_name = self.model_taxonomy[name]['model_name']
                        additional_model_parameters = self.model_taxonomy[name]['parameters']
                        # Declare the model
                        temp_model = globals()[additional_model_name](**additional_model_parameters)
                        # Inference the model
                        resp, input_tokens, output_tokens, latency = temp_model(taxonomy_dict, data_dict, metadata_dict)
                        # Delete the model
                        del temp_model
                    else:
                        # Call the default model
                        resp, input_tokens, output_tokens, latency = self.model(taxonomy_dict, data_dict, metadata_dict)   
                    # If result is 'Not specified' then there is no point to traverse the tree
                    if resp.lower().strip() == 'not specified':
                        res[f'L{depth}'] = 'Not specified'
                        return res
                    # Get the most similar label of response using fuzzy-matching
                    label = get_most_similar(labels, resp)
                    # Store the result of the depth
                    res[f'L{depth}'] = {
                        'Label': label,
                        'Input Tokens': input_tokens,
                        'Output Tokens': output_tokens,
                        'Latency': latency
                    }
                    # Move to the next children
                    ptr = ptr.get('children')[labels.index(label)]
                    depth += 1
                # If the current node is a leaf node
                elif ptr.get('Classification / Extraction') == 'NA' and ptr.get('node_type') == 'NA':
                    # If current mode is 'attribute'
                    if self.mode == 'attribute':
                        # Store each attribute's result temporarily
                        temp = {}
                        # Traverse all attributes
                        for attr in ptr.get('children'):
                            # Prepare taxonomy_dict and metadata_dict
                            taxonomy_dict = {
                                'breadcrumb': attr.get('name'),
                                'labels': attr.get('labels'),
                                'prompt': self.prompts['leaf_prompt'],
                                'note': self.prompts['note'],
                                'default_text_cols': self.text_cols,
                                'default_image_cols': self.image_cols,
                                'node_type': ptr.get('node_type'),
                                'inference_mode': self.mode
                            }
                            metadata_dict = attr.get('metadata')
                            # Use the model for inference
                            resp, input_tokens, output_tokens, latency = self.model(taxonomy_dict, data_dict, metadata_dict)
                            # For classification task, get the most similar label and for extraction task no postprocessing is needed
                            if attr.get('Classification / Extraction') == 'Classification' and resp.lower().strip() != 'not specified':
                                label = []
                                for l in resp.split(','):
                                    label.append(get_most_similar(attr.get('labels'), l))
                                label = [x for x in label if x != 'Not specified']
                                if len(label) == 0:
                                    label = 'Not specified'
                                else:
                                    label = ', '.join(label)
                            else:
                                if resp.lower().strip() == 'not specified':
                                    resp = 'Not specified'
                                label = resp
                            # Store the attribute's result
                            temp[attr.get('name').split(' > ')[-1]] = {
                                'Label': label,
                                'Input Tokens': input_tokens,
                                'Output Tokens': output_tokens,
                                'Latency': latency
                            }
                        # Store all attributes result
                        res[f'L{depth}'] = temp
                        # End of traversal
                        ptr = None
                    # If current model is 'category'
                    elif self.mode == 'category':
                        # Prepare taxonomy and metadata dict
                        taxonomy_dict, metadata_dict = {}, {}
                        # Traverse all the attributes and prepare it
                        for attr in ptr.get('children'):
                            attribute = attr.get('name').split(' > ')[-1].strip()
                            taxonomy_dict[attribute] = {
                                'breadcrumb': attr.get('name'),
                                'labels': attr.get('labels'),
                            } 
                            metadata_dict[attribute] = attr.get('metadata')
                        taxonomy_dict['prompt'] = self.prompts['leaf_prompt']
                        taxonomy_dict['note'] = self.prompts['note']
                        taxonomy_dict['default_text_cols'] = self.text_cols
                        taxonomy_dict['default_image_cols'] = self.image_cols,
                        taxonomy_dict['node_type'] = ptr.get('node_type')
                        taxonomy_dict['inference_mode'] = self.mode
                        resp, input_tokens, output_tokens, latency = self.model(taxonomy_dict, data_dict, metadata_dict)
                        # Postprocessing the response of LLM
                        resp = postprocessing_llm_response(resp)
                        resp = text_to_dict(resp)
                        # Store the result
                        temp = {}
                        for x, y in resp.items():
                            temp[x] = {
                                'Label': y,         # Check for classification can be added
                                'Input Tokens': input_tokens // len(resp),
                                'Output Tokens': output_tokens // len(resp),
                                'Latency': latency / len(resp)
                            }
                        res[f'L{depth}'] = temp
                        # End of the traversal
                        ptr = None
                    # If current mode is presets
                    elif self.mode == 'presets':
                        # Store each attribute's result temporarily
                        temp = {}
                        # Traverse all the attributes
                        for attr in ptr.get('children'):
                            # Get name
                            name = attr.get('name')
                            # Get the prompt
                            if name in self.additional_prompts:
                                prompt = self.additional_prompts[name]
                            else:
                                prompt = self.prompts['leaf_prompt']
                            # Preapre taxonomy_dict and metadata_dict
                            taxonomy_dict = {
                                'breadcrumb': attr.get('name'),
                                'labels': attr.get('labels'),
                                'prompt': prompt,
                                'note': self.prompts['note'],
                                'default_text_cols': self.text_cols,
                                'default_image_cols': self.image_cols,
                                'node_type': ptr.get('node_type'),
                                'inference_mode': self.mode
                            }
                            metadata_dict = attr.get('metadata')
                            # Get inference from the model
                            if name in self.model_taxonomy:
                                # Get the preset model name and it's parameters
                                additional_model_name = self.model_taxonomy[name]['model_name']
                                additional_model_parameters = self.model_taxonomy[name]['parameters']
                                # Declare the model
                                temp_model = globals()[additional_model_name](**additional_model_parameters)
                                # Inference using the model
                                resp, input_tokens, output_tokens, latency = temp_model(taxonomy_dict, data_dict, metadata_dict)
                                del temp_model
                            else:
                                # Inference using default model
                                resp, input_tokens, output_tokens, latency = self.model(taxonomy_dict, data_dict, metadata_dict)
                            # For classification task get most similar label, no postprocessing is needed for extraction
                            if attr.get('Classification / Extraction') == 'Classification' and resp.lower().strip() != 'not specified':
                                label = []
                                for l in resp.split(','):
                                    label.append(get_most_similar(attr.get('labels'), l))
                                label = [x for x in label if x != 'Not specified']
                                if len(label) == 0:
                                    label = 'Not specified'
                                else:
                                    label = ', '.join(label)
                            else:
                                if resp.lower().strip() == 'not specified':
                                    resp = 'Not specified'
                                label = resp
                            # Store the attribute's result
                            temp[attr.get('name').split(' > ')[-1]] = {
                                'Label': label,
                                'Input Tokens': input_tokens,
                                'Output Tokens': output_tokens,
                                'Latency': latency
                            }
                        # Store all attribute's result
                        res[f'L{depth}'] = temp
                        # End of traversal
                        ptr = None
                    # Invalid inferencd mode
                    else:
                        raise ValueError("Wrong value of inference mode!")
        # Return all tags for a record
        return format_tags(res)       

    def process_row(self, row, tag_function):
        try:
            tags = tag_function(row)
            tags = json.dumps(tags, indent=4)
            row['results'] = tags
            return row
        except Exception as err:
            print(traceback.format_exc())
            return None
    
    def __call__(self, df: pd.DataFrame, model_taxonomy: dict =  None, additional_prompts: dict = None):
        # For present model model_taxonomy is needed
        if self.mode == 'presets':
            if model_taxonomy is None:
                raise AttributeError("Model Taxonomy is needed for 'presets' mode!")
            else:
                self.set('model_taxonomy', model_taxonomy)
            if additional_prompts is not None:
                self.set('additional_prompts', additional_prompts)
        # List to store tags
        res = []
        # Process all records
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_row, row, self.tag) for row in df.to_dict(orient='records')]
            for future in tqdm(as_completed(futures), desc='Inferencing', total=df.shape[0]):
                result = future.result()
                if result is not None:
                    res.append(result)
        res = pd.DataFrame(res)
        # Format results
        res = format_result(res)
        # Delete the artifacts of the class
        if self.mode == 'presets':
            self.remove('model_taxonomy')
            self.remove('additional_prompts')
        # Return tagged data
        return res