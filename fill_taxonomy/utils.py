# Import libraries
import os
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from tqdm import tqdm
from openai import OpenAI




# Class to fill the taxonomy
class FillTaxonomy:
    def __init__(self, df: pd.DataFrame, n_levels: int):
        self.df = df.reset_index(drop=True)
        self.n_levels = n_levels
        self.similarity_matrix = None
        self.client = OpenAI()
        self.syn_prompt = None
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'classification_or_extraction.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'classification_or_extraction.txt')) as fp:
                self.task_prompt = fp.read()
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'single_value_or_multi_value.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'single_value_or_multi_value.txt')) as fp:
                self.return_prompt = fp.read()
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'input_priority.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'input_priority.txt')) as fp:
                self.input_prompt = fp.read()
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'data_type.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'data_type.txt')) as fp:
                self.datatype_prompt = fp.read()
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'ranges.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'ranges.txt')) as fp:
                self.ranges_prompt = fp.read()
        if os.path.exists(os.path.join('fill_taxonomy', 'prompts', 'units.txt')):
            with open(os.path.join('fill_taxonomy', 'prompts', 'units.txt')) as fp:
                self.units_prompt = fp.read()

    def create_hierarchy(self):
        self.df['hierarchy'] = self.df.apply(lambda row: ' > '.join([x for x in row.values if not pd.isna(x)][:self.n_levels]), axis=1)
    
    def get_similarity(self):
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        encoded_vector = model(self.df['hierarchy'].to_list()).numpy()
        self.similarity_matrix = cosine_similarity(encoded_vector)
    
    def fill_by_similarity(self, default_task: str = 'classification', default_return: str = 'single', 
                           default_input: str = 'text'):
        task_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Classification / Extraction'].to_list()])
        return_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Single Value / Multi Value'].to_list()])
        input_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Input Priority'].to_list()])
        data_type_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Data Type'].to_list()])
        ranges_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Ranges'].to_list()])
        units_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Units'].to_list()])
        self.create_hierarchy()
        self.get_similarity()
        res = []
        for i, row in tqdm(enumerate(self.df.to_dict(orient='records')),
                           desc='Processing Records',
                           total=len(enumerate(self.df.to_dict(orient='records')))):
            similarity_vector = self.similarity_matrix[i]
            if pd.isna(row['Classification / Extraction']):
                idx = np.argmax(similarity_vector * task_mask)
                if pd.isna(self.df.at[idx, 'Classification / Extraction']):
                    row['Classification / Extraction'] = default_task
                else:
                    row['Classification / Extraction'] = self.df.at[idx, 'Classification / Extraction']
            if pd.isna(row['Single Value / Multi Value']):
                idx = np.argmax(similarity_vector * return_mask)
                if pd.isna(self.df.at[idx, 'Single Value / Multi Value']):
                    row['Single Value / Multi Value'] = default_return
                else:
                    row['Single Value / Multi Value'] = self.df.at[idx, 'Single Value / Multi Value']
            if pd.isna(row['Input Priority']):
                idx = np.argmax(similarity_vector * input_mask)
                if pd.isna(self.df.at[idx, 'Input Priority']):
                    row['Input Priority'] = default_input
                else:
                    row['Input Priority'] = self.df.at[idx, 'Input Priority']
            if pd.isna(row['Data Type']):
                if row['Classification / Extraction'] == 'Classification':
                    row['Data Type'] = 'Enum'
                else:
                    idx = np.argmax(similarity_vector * data_type_mask)
                    if pd.isna(self.df.at[idx, 'Data Type']):
                        row['Data Type'] = 'Enum'
                    else:
                        row['Data Type'] = self.df.at[idx, 'Data Type']
            if pd.isna(row['Ranges']):
                if row['Data Type'] in ['String', 'Enum']:
                    row['Ranges'] = 'NA'
                else:
                    idx = np.argmax(similarity_vector * ranges_mask)
                    if pd.isna(self.df.at[idx, 'Ranges']):
                        row['Ranges'] = 'NA'
                    else:
                        row['Ranges'] = self.df.at[idx, 'Units']
            if pd.isna(row['Units']):
                if row['Data Type'] in ['String', 'Enum']:
                    row['Units'] = 'NA'
                else:
                    idx = np.argmax(similarity_vector * units_mask)
                    if pd.isna(self.df.at[idx, 'Units']):
                        row['Units'] = 'NA'
                    else:
                        row['Units'] = self.df.at[idx, 'Units']
            res.append(row)
        res = pd.DataFrame(res)
        res.drop(columns='hierarchy', inplace=True)
        return res

    def get_task(self, attribute_name: str, sample_values: list):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert machine learning engineer."},
                {"role": "user", "content": self.task_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response
    
    def get_return_type(self, attribute_name: str, sample_values: list):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert machine learning engineer."},
                {"role": "user", "content": self.return_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response

    def get_input_priority(self, attribute_name: str, sample_values: list):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert machine learning engineer."},
                {"role": "user", "content": self.input_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response
    
    def get_datatype(self, attribute_name: str, sample_values: str):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert of taxonomy management."},
                {"role": "user", "content": self.datatype_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response

    def get_ranges(self, attribute_name: str, sample_values: str):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert of taxonomy management."},
                {"role": "user", "content": self.ranges_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response

    def get_unit(self, attribute_name: str, sample_values: str):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert of taxonomy management."},
                {"role": "user", "content": self.units_prompt.format(attribute_name=attribute_name, sample_values=sample_values)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return response

    def get_synonyms(self, attribute_name: str, value: str):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who is an expert of English language."},
                {"role": "user", "content": self.syn_prompt.format(attribute_name=attribute_name, value=value)}
            ]
        )
        response = response.json()
        response = json.loads(response)
        response = response['choices'][0]['message']['content']
        return {
            'Attribute': attribute_name,
            'Word': value,
            'Synonyms': response
        }
    
    def fill_by_llm(self):
        attributes = self.df[list(self.df.columns)[self.n_levels - 1]].to_list()
        tasks_meta = {k: v for k, v in zip(attributes, self.df['Classification / Extraction'].to_list()) if not pd.isna(v)}
        return_type_meta = {k: v for k, v in zip(attributes, self.df['Single Value / Multi Value'].to_list()) if not pd.isna(v)}
        inputs_meta = {k: v for k, v in zip(attributes, self.df['Input Priority'].to_list()) if not pd.isna(v)}
        data_type_meta = {k: v for k, v in zip(attributes, self.df['Data Type'].to_list()) if not pd.isna(v)}
        ranges_meta = {k: v for k, v in zip(attributes, self.df['Ranges'].to_list()) if not pd.isna(v)}
        units_meta = {k: v for k, v in zip(attributes, self.df['Units'].to_list()) if not pd.isna(v)}
        res = []
        for row in tqdm(self.df.to_dict(orient='records'), desc='Processing Records', total=self.df.shape[0]):
            attribute = list(row.values())[self.n_levels - 1]
            values = list(row.values())[self.n_levels]
            if attribute in tasks_meta.keys():
                row['Classification / Extraction'] = tasks_meta[attribute]
            else:
                row['Classification / Extraction'] = self.get_task(attribute_name=attribute, sample_values=values)
            if attribute in return_type_meta.keys():
                row['Single Value / Multi Value'] = return_type_meta[attribute]
            else:
                row['Single Value / Multi Value'] = self.get_return_type(attribute_name=attribute, sample_values=values)
            if attribute in inputs_meta.keys():
                row['Input Priority'] = inputs_meta[attribute]
            else:
                row['Input Priority'] = self.get_input_priority(attribute_name=attribute, sample_values=values)
            if attribute in data_type_meta.keys():
                row['Data Type'] = data_type_meta[attribute]
            else:
                if row['Classification / Extraction'] == 'Classification':
                    row['Data Type'] = 'Enum'
                else:
                    row['Data Type'] = self.get_datatype(attribute_name=attribute, sample_values=values)
            if attribute in ranges_meta.keys():
                row['Ranges'] = ranges_meta[attribute]
            else:
                if row['Data Type'] in ['String', 'Enum']:
                    row['Ranges'] = 'NA'
                else:
                    row['Ranges'] = self.get_ranges(attribute_name=attribute, sample_values=values)
            if attribute in units_meta.keys():
                row['Units'] = units_meta[attribute]
            else:
                if row['Data Type'] in ['String', 'Enum']:
                    row['Units'] = 'NA'
                else:
                    row['Units'] = self.get_unit(attribute_name=attribute, sample_values=values)
            res.append(row)
        res = pd.DataFrame(res)
        return res

    def __call__(self, strategy: str=None, synonym_policy: int=1):
        print("Metadata filling has been started!")
        if strategy:
            if strategy == 'llm':
                filled_df = self.fill_by_llm()
                print("Metadata is filled by LLM!")
            else:
                filled_df = self.fill_by_similarity()
                print("Metdata is filled by similarity search!")
        else:
            max_non_missing = 0
            for col in ['Classification / Extraction', 'Single Value / Multi Value', 'Input Priority']:
                non_missing_values = self.df.shape[0] - self.df[col].isnull().sum()
                max_non_missing = max(max_non_missing, non_missing_values)
            if max_non_missing / self.df.shape[0] >= 0.6:
                filled_df = self.fill_by_similarity()
                print("Metadata is filled by similarity search!")
            else:
                filled_df = self.fill_by_llm()
                print("Metdata is filled by LLM!")
        print("Synonyms searching using LLM has been started!")
        if synonym_policy == 1:
            with open(os.path.join('fill_taxonomy', 'prompts', 'synonyms_policy_1.txt')) as fp:
                self.syn_prompt = fp.read()
            attributes = self.df[list(self.df.columns)[self.n_levels - 1]].to_list()
            values = self.df[list(self.df.columns)[self.n_levels]].to_list()
            args = []
            for attr, value_list in zip(attributes, values):
                if pd.isna(value_list):
                    continue
                for value in value_list.split(','):
                    args.append((attr, value.strip()))
            with ThreadPoolExecutor(max_workers=16) as executor:
                res = list(tqdm(executor.map(lambda p: self.get_synonyms(*p), args), desc='Finding Synonyms', total=len(args)))
            syn_df = pd.DataFrame(res)
        elif synonym_policy == 2:
            with open(os.path.join('fill_taxonomy', 'prompts', 'synonyms_policy_2.txt')) as fp:
                self.syn_prompt = fp.read()
            attributes = self.df[list(self.df.columns)[self.n_levels - 1]].to_list()
            values = self.df[list(self.df.columns)[self.n_levels]].to_list()
            args = []
            for attr, value in zip(attributes, values):
                if pd.isna(value):
                    continue
                args.append((attr, [x.strip() for x in value.split(',')]))
            with ThreadPoolExecutor(max_workers=16) as executor:
                res = list(tqdm(executor.map(lambda p: self.get_synonyms(*p), args), desc='Finding Synonyms', total=len(args)))
            results = []
            for rec in res: 
                attr = rec['Attribute']
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
                            'Attribute': attr,
                            'Word': word,
                            'Synonyms': syns
                        }
                    )
            syn_df = pd.DataFrame(results)
        else:
            raise ValueError("Invalid synonym policy!")
        print("All tasks are completed!")
        return filled_df, syn_df
