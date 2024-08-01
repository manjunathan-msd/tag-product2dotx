# Import libraries
import os
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
from openai import OpenAI




# Class to fill the taxonomy
class FillTaxonomy:
    def __init__(self, df: pd.DataFrame, n_levels: int):
        self.df = df.reset_index(drop=True)
        self.n_levels = n_levels
        self.similarity_matrix = None
        self.client = OpenAI()
        with open(os.path.join('fill_taxonomy', 'prompts', 'synonyms.txt')) as fp:
            self.syn_prompt = fp.read()
        with open(os.path.join('fill_taxonomy', 'prompts', 'classification_or_extraction.txt')) as fp:
            self.task_prompt = fp.read()
        with open(os.path.join('fill_taxonomy', 'prompts', 'single_value_or_multi_value.txt')) as fp:
            self.return_prompt = fp.read()
        with open(os.path.join('fill_taxonomy', 'prompts', 'input_priority.txt')) as fp:
            self.input_prompt = fp.read()

    def create_hierarchy(self):
        self.df['hierarchy'] = self.df.apply(lambda row: ' > '.join(list(row.values)[:self.n_levels]), axis=1)
    
    def get_similarity(self):
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        encoded_vector = model(self.df['hierarchy'].to_list()).numpy()
        self.similarity_matrix = cosine_similarity(encoded_vector)
    
    def fill_by_similarity(self, default_task: str = 'classification', default_return: str = 'single', 
                           default_input: str = 'text'):
        task_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Classification / Extraction'].to_list()])
        return_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Single Value / Multi Value'].to_list()])
        input_mask = np.array([1 if not pd.isna(x) else 0 for x in self.df['Input Priority'].to_list()])
        self.create_hierarchy()
        self.get_similarity()
        res = []
        for i, row in enumerate(self.df.to_dict(orient='records')):
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
        tasks = self.df['Classification / Extraction'].to_list()
        return_type = self.df['Single Value / Multi Value'].to_list()
        inputs = self.df['Input Priority'].to_list()
        tasks_meta = {k: v for k, v in zip(attributes, tasks) if not pd.isna(v)}
        return_type_meta = {k: v for k, v in zip(attributes, return_type) if not pd.isna(v)}
        inputs_meta = {k: v for k, v in zip(attributes, inputs) if not pd.isna(v)}
        res = []
        for row in self.df.to_dict(orient='records'):
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
            res.append(row)
        res = pd.DataFrame(res)
        return res

    def __call__(self, strategy: str=None):
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
        attributes = self.df[list(self.df.columns)[self.n_levels - 1]].to_list()
        values = self.df[list(self.df.columns)[self.n_levels]].to_list()
        args = []
        for attr, value_list in zip(attributes, values):
            for value in value_list.split(','):
                args.append((attr, value.strip()))
        with ThreadPoolExecutor(max_workers=16) as executor:
            res = list(executor.map(lambda p: self.get_synonyms(*p), args))
        syn_df = pd.DataFrame(res)
        print("All tasks are completed!")
        return filled_df, syn_df
