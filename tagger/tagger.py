import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from models.gpt_model import inference_by_chatgpt
from utils.image_utils import check_cache
from utils.timer_utils import tic, toc
from taxonomy_builder.utils import TaxonomyTree, TaxonomyNode

class Tagger2dotX:
    def __init__(self, config_path='configs.yaml', taxonomy_df=None,taxonomy_depth=5, agent=None, use_meta=True):
        self.configs = self.load_config(config_path)
        self.taxonomy_df = taxonomy_df or self.load_default_taxonomy()
        self.taxonomy_depth=taxonomy_depth
        self.taxonomy = self.create_taxonomy_tree()
        self.agent = agent or inference_by_chatgpt
        self.classification_prompt = self.load_prompt(self.configs['prompt_path_0'])
        self.attribute_prompt = self.load_prompt(self.configs['prompt_path_1'])
        self.custom_prompt = self.load_prompt(self.configs['custom_prompt_path'])
        self.timing_dict = {}
        self.use_meta = use_meta

    def load_config(self, config_path):
        with open(config_path) as fp:
            return yaml.safe_load(fp)

    def load_prompt(self, prompt_path):
        with open(prompt_path) as fp:
            return fp.read()

    def load_default_taxonomy(self):
        categories = ["Fashion", "Accessories", "Beauty", "Electronics", "Home"]
        df = pd.DataFrame()
        for category in categories:
            tax_path = f"dumps/{category} Taxonomy.csv"
            tax_df = pd.read_csv(tax_path)
            tax_df.rename(columns={
                "Metadata": "Classification / Extraction",
                "Single Value/ Multi Value": "Single Value / Multi Value"
            }, inplace=True)
            df = pd.concat([df, tax_df], ignore_index=True)
        return df
    
    def create_taxonomy_tree(self):
        tree = TaxonomyTree(n_levels=self.taxonomy_depth)
        tree(self.taxonomy_df)
        return tree

    def match_response_to_child(self, response, children, fuzzy_threshold=80):
        response_lower = response.lower()
        best_match = None
        best_ratio = 0

        for child in children:
            child_name = child.get_name().split('>')[-1].lower()
            
            if child_name in response_lower:
                return child
            
            if child.labels:
                for label in child.labels:
                    if label.lower() in response_lower:
                        return child
            
            ratio = fuzz.partial_ratio(child_name, response_lower)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = child
            
            if child.labels:
                for label in child.labels:
                    ratio = fuzz.partial_ratio(label.lower(), response_lower)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = child

        return best_match if best_ratio >= fuzzy_threshold else None

    def traverse_tree_and_classify(self, context, image_path, max_depth=100):
        depth = 0
        current_node = self.taxonomy.root
        classification_result = {}

        while depth < max_depth and current_node.get_children() and not (current_node.get_node_type().lower().strip() == "na" and depth != 0):
            tic(self.timing_dict, f"L{depth}")
            
            child_labels = [child.get_name().split('>')[-1] for child in current_node.get_children()]
            res = self.agent(context=context, prompt=self.classification_prompt, image_path=image_path, tax_vals=",".join(child_labels))
            
            chosen_child = self.match_response_to_child(res, current_node.get_children())
            
            if chosen_child:
                classification_result[f"L{depth}"] = chosen_child.get_name().split('>')[-1]
                current_node = chosen_child
            else:
                classification_result[f"L{depth}"] = "Not Specified"
                break
            
            toc(self.timing_dict, f"L{depth}")
            depth += 1

        if current_node.get_node_type() == "NA" and depth != 0:
            taxonomy_schema = {}
            for attribute_node in current_node.get_children():
                attribute_name = attribute_node.get_name().split(">")[-1]
                attribute_values = attribute_node.get_labels()
                attribute_task = attribute_node.get_task()
                attribute_return = attribute_node.get_return_type()

                if self.use_meta:
                    taxonomy_schema[attribute_name] = {
                        'Task Type': attribute_task.capitalize(),
                        'Values' if attribute_task.lower() == "classification" else 'Sample Values': attribute_values,
                        "Return Type": attribute_return
                    }
                else:
                    taxonomy_schema[attribute_name] = attribute_values

            tic(self.timing_dict, f"L{depth}")
            res = self.agent(context=context, prompt=self.attribute_prompt, image_path=image_path, tax_vals=taxonomy_schema)
            classification_result[f"L{depth}"] = res
            toc(self.timing_dict, f"L{depth}")

        return classification_result

    def __call__(self, record):
        # Separate 'image url' and other non-free form text columns
        excluded_columns = ['image url', 'Standard Output', 'Custom Output']  # Add other non-free form text columns here
        
        # Combine all other columns into the context
        context_parts = []
        for col, value in record.items():
            if col not in excluded_columns and pd.notna(value):
                if col == 'title':
                    context_parts.insert(0, f"Title Of the Product: {value}")
                elif col == 'description':
                    context_parts.insert(1, f"Description of the Product: {value}")
                else:
                    context_parts.append(f"{col}: {value}")
        
        context = "\n".join(context_parts)
        
        image_path = check_cache(image_url=record['image url'])
        
        try:
            standard_output = self.traverse_tree_and_classify(context=context, image_path=image_path)
            custom_output = self.agent(context=context, image_path=image_path, prompt=self.custom_prompt)
            
            return {
                'Standard Output': json.dumps(standard_output, indent=4),
                'Custom Output': json.dumps(custom_output, indent=4)
            }
        except Exception as e:
            print(f"Error processing record: {str(e)}")
            return {'Standard_Output': None, 'Custom_Output': None}
