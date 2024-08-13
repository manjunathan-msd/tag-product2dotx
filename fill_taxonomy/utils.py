# Import libraries
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.chatgpt import MetaChatGPT


'''
Definition: The class can metdata when a taxonomy is given
'''
class FillMetadata:
    # Constructor
    def __init__(self, metadata: dict = None):
        self.client = None
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = {
                'Classification / Extraction': 'prompts/classification_or_extraction.txt', 
                'Single Value / Multi Value': 'prompts/single_value_or_multi_value.txt', 
                'Input Priority': 'prompts/input_priority.txt',
                'Data Type': 'prompts/data_type.txt', 
                'Ranges': 'prompts/ranges.txt', 
                'Units': 'prompts/units.txt'
            }
    
    @staticmethod
    def get(breadcrumb: str, values: str, metdata_name: str):
        # Path of metdata
        metadata = {
            'Classification / Extraction': 'prompts/classification_or_extraction.txt', 
            'Single Value / Multi Value': 'prompts/single_value_or_multi_value.txt', 
            'Input Priority': 'prompts/input_priority.txt',
            'Data Type': 'prompts/data_type.txt', 
            'Ranges': 'prompts/ranges.txt', 
            'Units': 'prompts/units.txt'
        }
        # Client
        client = MetaChatGPT(model_name='gpt-4o-mini', data_type='text', prompt_path=metadata[metdata_name])
        # Inference
        payload = {
            'attribute': breadcrumb,
            'values': values
        }
        response = client(**payload)
        return response

    # Function to get metadata using LLM
    def inference(self, *args):
        breadcrumb, value, res = args
        if not pd.isna(res):
            return res
        payload = {
            'attribute': breadcrumb,
            'values': value
        }
        response = self.client(**payload)
        return response

    # Fill metadata using LLM
    def fill_by_llm(self, df: pd.DataFrame):
        for col in tqdm(self.metadata.keys(), desc='Generating Metdata', total=len(self.metadata)):
            if col in ['Data Type', 'Ranges', 'Units']:
                if col == 'Data Type':
                    df[col] = df['Classification / Extraction'].apply(lambda x: 'Enum' if x == 'Classification' else np.nan)
                else:
                    df[col] = df['Classification / Extraction'].apply(lambda x: 'NA' if x == 'Classification' else np.nan)
            self.client = MetaChatGPT(model_name='gpt-4o-mini', data_type='text', prompt_path=self.metadata[col])
            with ThreadPoolExecutor(max_workers=8) as executor:
                df[col] = list(executor.map(lambda row: self.inference(row['breadcrumb'], row['V'], row[col]), df.to_dict(orient='records')))
        df.drop(columns='breadcrumb', inplace=True)
        return df

    # Menu driven function
    def __call__(self, df: pd.DataFrame):
        # Assign all metadata column by NaN values
        for col in self.metadata:
            if col not in list(df.columns):
                df[col] = np.nan
        # Get valid cols and create breadcrumb
        valid_cols = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        valid_cols.append('A')
        df['breadcrumb'] = df.apply(lambda row: ' > '.join([row[col] for col in valid_cols if not pd.isna(row[col])]), axis=1)
        # Fill the missing values
        return self.fill_by_llm(df)
        
        

'''
Definition: The class can fill synonyms when taxonomy and attribute values are given.
'''
class FindSynonyms:
    # Constructor
    def __init__(self):
        self.client = None

    # Get synonyms using ChatGPT
    def get_synonyms(self, *args):
        taxonomy, attribute, value, note = args
        if note:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'value': value,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'value': value
            }
        response = self.client(**payload)
        return {
            'Breadcrumb': taxonomy + ' > ' + attribute,
            'V': value,
            'Synonyms': response 
        }
    
    # Get the metadata for a single sample
    @staticmethod
    def get(breadcrumb: str, value: str, note: str):
        if note:
            prompt_path = 'prompts/synonyms_dynamic_policy_2.txt'
        else:
            prompt_path = 'prompts/synonyms_policy_2.txt'
        client = MetaChatGPT(model_name='gpt-4o-mini', data_type='text', prompt_path=prompt_path)
        taxonomy , attribute = ' > '.join(breadcrumb.split(' > ')[:-1]), breadcrumb.split(' > ')[-1]
        if note:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'value': value,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'value': value
            }
        synonyms = client(**payload)
        if synonyms.startswith('result') or synonyms.startswith('Result'):
            synonyms.replace('result', '')
            synonyms = synonyms.replace("```", '')
            synonyms = synonyms.strip()
        results = []
        for syn_pair in synonyms.split('\n'):
            idx = syn_pair.find(':')
            word = syn_pair[:idx]
            syns = syn_pair[idx+1:].strip()
            results.append(
                {
                    'Breadcrumb': breadcrumb,
                    'V': word,
                    'Synonyms': syns
                }
            )
        
    # Recall based policy: Policy 1
    def policy_1(self, df: pd.DataFrame, note: str = None):
        levels = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        taxonomy = df.apply(lambda row: ' > '.join([row[x] for x in levels if not pd.isna(row[x])]), axis=1).to_list()
        attributes = df['A'].to_list()
        values = df['V'].to_list()
        data_types = df['Data Type'].to_list()
        args = []
        for tax, attr, value_list, data_type in zip(taxonomy, attributes, values, data_types):
            if pd.isna(value_list) or data_type == 'Numeric':
                continue
            for value in value_list.split(','):
                if value.lower().strip() == 'others' or value.lower().strip() == 'other':
                    continue
                args.append((tax, attr, value.strip(), note))
        with ThreadPoolExecutor(max_workers=16) as executor:
            res = list(tqdm(executor.map(lambda p: self.get_synonyms(*p), args), desc='Finding Synonyms', total=len(args)))
        return pd.DataFrame(res)
    
    # Relevance based policy: Policy 2
    def policy_2(self, df: pd.DataFrame, note: str = None):
        levels = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        taxonomy = df.apply(lambda row: ' > '.join([row[x] for x in levels if not pd.isna(row[x])]), axis=1).to_list()
        attributes = df['A'].to_list()
        values = df['V'].to_list()
        data_types = df['Data Type'].to_list()
        args = []
        for tax, attr, value, data_type in zip(taxonomy, attributes, values, data_types):
            if pd.isna(value) or data_type == 'Numeric':
                continue
            args.append((tax, attr, [x.strip() for x in value.split(',') if x.lower().strip()!='others' and x.lower().strip()!='other'], note))
        with ThreadPoolExecutor(max_workers=16) as executor:
                res = list(tqdm(executor.map(lambda p: self.get_synonyms(*p), args), desc='Finding Synonyms', total=len(args)))
        results = []
        for rec in res: 
            breadcrumb = rec['Breadcrumb']
            synonyms = rec['Synonyms']
            if synonyms.startswith('result') or synonyms.startswith('Result'):
                synonyms.replace('result', '')
            synonyms = synonyms.replace("```", '')
            synonyms = synonyms.strip()
            for syn_pair in synonyms.split('\n'):
                idx = syn_pair.find(':')
                word = syn_pair[:idx]
                syns = syn_pair[idx+1:].strip()
                results.append(
                    {
                        'Breadcrumb': breadcrumb,
                        'V': word,
                        'Synonyms': syns
                    }
                )
        return pd.DataFrame(results)

    # Main function call
    def __call__(self, df: pd.DataFrame, policy: int = 2, prompt_path: str='prompts/synonyms_policy_2.txt', note: str = None):
        if policy == 1:
            if note:
                prompt_path = 'prompts/synonyms_dynamic_policy_1.txt'
            else:
                prompt_path = 'prompts/synonyms_policy_1.txt'
        elif policy == 2:
            if note:
                prompt_path = 'prompts/synonyms_dynamic_policy_2.txt'
            else:
                prompt_path = 'prompts/synonyms_policy_2.txt'
        else:
            raise ValueError("Invalid synonyms policy!")
        self.client = MetaChatGPT(model_name='gpt-4o-mini', data_type='text', prompt_path=prompt_path)
        if policy == 1:
            return self.policy_1(df, note)
        elif policy == 2:
            return self.policy_2(df, note)
        else:
            raise ValueError("Invalid policy for synonyms!")


            

'''
Definition: The class can create attributes from a given breadcrumb.
'''
class FillAttributes:
    # Constructor
    def __init__(self):
        self.client = None
    
    # Create hirearchy from the levels and attributes of the given dataframe
    def create_hierarchy(self, df: pd.DataFrame, valid_cols: list):
        res = []
        for row in df.to_dict(orient='records'):
            temp = ''
            for col in valid_cols:
                if not pd.isna(col):
                    temp = temp + ' > ' + col
            row['hierarchy'] = temp
        res.append(row)
        df = pd.DataFrame(res)
        return df
    
    # Values using ChatGPT
    def inference(self, hierarchy: str, note: str = None):
        taxonomy, category = ' > '.join(hierarchy.split(' > ')[:-1]), hierarchy.split(' > ')[-1]
        if note:
            payload = {
                'taxonomy': taxonomy,
                'category': category,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'category': category
            }
        response = self.client(**payload)
        return response
    
    # Get the values of each attribute
    def get_attributes(self, row: dict):
        # If values are already filled then ignore it
        if not pd.isna(row['A']):
            return row['A']
        else:
            if 'note' in list(row.keys()):
                return self.inference(row['hierarchy'], row['note'])
            else:
                return self.inference(row['hierarchy'])
    
    # Function to get attributes for a single input    
    @staticmethod
    def get(hierarchy: str, prompt_path: str = 'prompts/fill_attributes.txt', note: str = None):
        if note:
            prompt_path = 'prompts/fill_attributes_dynamic.txt'
        client = MetaChatGPT(
            model_name='gpt-4o-mini',
            data_type='text',
            prompt_path=prompt_path
        )
        taxonomy, category = ' > '.join(hierarchy.split(' > ')[:-1]), hierarchy.split(' > ')[-1]
        if note:
            payload = {
                'taxonomy': taxonomy,
                'category': category,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'category': category
            }
        response = client(**payload)
        return response
    
    # Postprocessing of the final result
    def postprocessing(self, df: pd.DataFrame):
        res = []
        for row in df.to_dict(orient='records'):
            attrs = [a.strip() for a in row['A'].split(',')]
            for a in attrs:
                temp = row.copy()
                temp['A'] = a
                res.append(temp)
        res = pd.DataFrame(res)
        return res
    
    # Main function call 
    def __call__(self, df: pd.DataFrame, prompt_path: str = 'prompts/fill_attributes.txt', note: str = None):
        # Get all the valid cols, valid cols are levels
        valid_cols = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        # Create the prompt path
        if note:
            prompt_path = 'prompts/fill_attributes_dynamic.txt'
        # Create a ChatGPt clinet which will be used for generating repsonse
        self.client = MetaChatGPT(
            model_name='gpt-4o-mini',
            data_type='text',
            prompt_path=prompt_path
        )
        # Create the hierarchy for getting values
        df['hierarchy'] = df.apply(lambda row: ' > '.join([row[col] for col in valid_cols if not pd.isna(row[col])]), axis=1)
        # Add note to dataframe if given
        if note:
            df['note'] = note
        # Get the values of each attribute
        with ThreadPoolExecutor(max_workers=16) as executor:
            res = list(tqdm(executor.map(lambda p: self.get_attributes(p), df.to_dict(orient='records')), desc='Finding Values', total=df.shape[0]))
        # Return the values after postprocessing
        df['A'] = res
        if note:
            df.drop(columns=['hierarchy', 'note'], inplace=True)
        else:
            df.drop(columns='hierarchy', inplace=True)
        df = self.postprocessing(df)
        return df



'''
Definition: The class can create values from a given breadcrumb and it's attribute.
'''
class FillValues:
    # Constructor
    def __init__(self):
        self.client = None
    
    # Create hirearchy from the levels and attributes of the given dataframe
    def create_hierarchy(self, df: pd.DataFrame, valid_cols: list):
        res = []
        for row in df.to_dict(orient='records'):
            temp = ''
            for col in valid_cols:
                if not pd.isna(col):
                    temp = temp + ' > ' + col
            row['hierarchy'] = temp
        res.append(row)
        df = pd.DataFrame(res)
        return df
    
    # Values using ChatGPT
    def inference(self, hierarchy: str, note: str = None):
        taxonomy, attribute = ' > '.join(hierarchy.split(' > ')[:-1]), hierarchy.split(' > ')[-1]
        if note:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
            }
        response = self.client(**payload)
        return response
    
    # Get the values of each attribute
    def get_values(self, row: dict):
        # If values are already filled then ignore it
        if not pd.isna(row['V']):
            return row['V']
        else:
            if 'note' in list(row.keys()):
                return self.inference(row['hierarchy'], row['note'])
            else:
                return self.inference(row['hierarchy'])
    
    # Function to get values for a single input
    @staticmethod
    def get(hierarchy: str, prompt_path: str = 'prompts/fill_values.txt', note: str = None):
        if note:
            prompt_path = 'prompts/fill_values_dynamic.txt'
        client = MetaChatGPT(
            model_name='gpt-4o-mini',
            data_type='text',
            prompt_path=prompt_path
        )
        taxonomy, attribute = ' > '.join(hierarchy.split(' > ')[:-1]), hierarchy.split(' > ')[-1]
        if note:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute,
                'note': note
            }
        else:
            payload = {
                'taxonomy': taxonomy,
                'attribute': attribute
            }
        response = client(**payload)
        return response
    
    # Main function call 
    def __call__(self, df: pd.DataFrame, prompt_path: str = 'prompts/fill_values.txt', note: str = None):
        # Get all the valid cols, valid cols are levels and 'A' which stand for attribute
        valid_cols = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        valid_cols.append('A')
        # Crate the prompt path
        if note:
            prompt_path = 'prompts/fill_values_dynamic.txt'
        # Create a ChatGPt clinet which will be used for generating repsonse
        self.client = MetaChatGPT(
            model_name='gpt-4o-mini',
            data_type='text',
            prompt_path=prompt_path
        )
        # Create the hierarchy for getting values
        df['hierarchy'] = df.apply(lambda row: ' > '.join([row[col] for col in valid_cols if not pd.isna(row[col])]), axis=1)
        # Assign note if given
        if note:
            df['note'] = note
        # Get the values of each attribute
        with ThreadPoolExecutor(max_workers=16) as executor:
            res = list(tqdm(executor.map(lambda p: self.get_values(p), df.to_dict(orient='records')), desc='Finding Values', total=df.shape[0]))
        df['V'] = res
        if note:
            df.drop(columns=['hierarchy', 'note'], inplace=True)
        else:
            df.drop(columns='hierarchy', inplace=True)
        return df
    