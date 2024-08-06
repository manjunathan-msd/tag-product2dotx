import os
import json
import yaml
import sys
import traceback
import pandas as pd
from pprint import pprint
import time
import random
from tqdm import tqdm
sys.path.append('/home/ubuntu/miniconda3/envs/dev/lib/python3.10/site-packages')
from fuzzywuzzy import fuzz
from models.gpt_model import inference_by_chatgpt
from utils.image_utils import check_cache
from utils.timer_utils import tic, toc
from utils.postprocess_utils import convert_demosite_format

# Import the TaxonomyTree and TaxonomyNode classes
from taxonomy_builder.utils import TaxonomyTree, TaxonomyNode,print_info

# Logging the Timings
timing_dict = {}

# Load configurations
def load_config_and_prompts():
    with open('configs.yaml') as fp:
        configs = yaml.safe_load(fp)
    
    with open(configs['prompt_path_0']) as fp:
        classification_prompt = fp.read()
    with open(configs['prompt_path_1']) as fp:
        attribute_prompt = fp.read()

    with open(configs['custom_prompt_path']) as fp:
        custom_prompt = fp.read()

    print("Configurations and prompts are loaded!")
    return configs, classification_prompt, attribute_prompt, custom_prompt

# Load and create TaxonomyTree
def create_taxonomy_tree():
    
    categories=["Fashion","Accessories","Beauty","Electronics","Home"]
    df=pd.DataFrame()
    for category in categories:
        tax_path=f"dumps/{category} Taxonomy.csv"
        tax_df=pd.read_csv(tax_path)
        tax_df.rename(columns={
            "Metadata":"Classification / Extraction",
            "Single Value/ Multi Value":"Single Value / Multi Value"
        },inplace=True)
        df = pd.concat([df, tax_df], ignore_index=True)
    print(f"Taking a Taxonomy DF of shape: {df.shape}")
    print(df.columns)
    tree = TaxonomyTree(n_levels=5)  # Adjust the number of levels as needed
    tree(df)
    print("Taxonomy tree is created!")
    return tree

#Match Response to the Corresponding Child Node.
def match_response_to_child(response, children, fuzzy_threshold=80):
    """
    Match the agent's response to the correct child node using exact and fuzzy matching.
    
    Args:
    response (str): The response from the agent.
    children (list): List of child TaxonomyNode objects.
    fuzzy_threshold (int): The threshold for fuzzy matching (0-100).
    
    Returns:
    TaxonomyNode or None: The matched child node or None if no match is found.
    """
    # print("Children of the Node:",[child.get_name() for child in children])
    response_lower = response.lower()
    best_match = None
    best_ratio = 0

    for child in children:
        child_name = child.get_name().split('>')[-1].lower()
        
        # Check for exact match
        if child_name in response_lower:
            return child
        
        # Check for exact match in labels
        if child.labels:
            for label in child.labels:
                if label.lower() in response_lower:
                    return child
        
        # Fuzzy matching for child name
        ratio = fuzz.partial_ratio(child_name, response_lower)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = child
        
        # Fuzzy matching for labels
        if child.labels:
            for label in child.labels:
                ratio = fuzz.partial_ratio(label.lower(), response_lower)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = child

    # Return the best fuzzy match if it meets the threshold
    if best_ratio >= fuzzy_threshold:
        return best_match
    
    # If no match is found, return None
    return None


# Traverse Taxonomy Tree and Generate Output
def traverse_tree_and_classify(agent, max_depth, context, image_path, classification_prompt, attribute_prompt, taxonomy_tree):
    if not max_depth:
        max_depth = 100
    
    depth = 0
    current_node = taxonomy_tree.root
    classification_result = {}

    while depth < max_depth and current_node.get_children() and not (current_node.get_node_type().lower().strip()=="na" and depth!=0):
        print("Within a Classification Node")
        # print_info(current_node)
        tic(timing_dict, f"L{depth}")
        
        # Get labels of children nodes
        child_labels = [child.get_name().split('>')[-1] for child in current_node.get_children()]
        
        print("Child Labels of the Node are:")
        print(child_labels)
        # Generate response using the agent
        res = agent(context=context, prompt=classification_prompt, image_path=image_path, tax_vals=",".join(child_labels))
        
        print(f"Raw Agent Response:{res}")
        
        # Use the new function to match the response to a child
        chosen_child = match_response_to_child(res, current_node.get_children())
        
        print(f"Closest Child Is: {chosen_child}")
        # time.sleep(10)
        if chosen_child:
            classification_result[f"L{depth}"] = chosen_child.get_name().split('>')[-1]
            current_node = chosen_child
        else:
            classification_result[f"L{depth}"] = "Not Specified"
            break
        
        toc(timing_dict, f"L{depth}")
        depth += 1

    print("Loop has been broken, No child Present")
    # Handle leaf node
    if current_node.get_node_type()=="NA" and depth!=0:
        print("-------------This is a Leaf Node-----------------")
        taxonomy_schema={}
        for attribute_node in current_node.get_children():
            attribute_name=attribute_node.get_name().split(">")[-1]
            attribute_values=attribute_node.get_labels() #Labels
            attribute_task=attribute_node.get_task() #Task Type
            attribute_return=attribute_node.get_return_type() #Get Return Type - Single/Multi

            taxonomy_schema[attribute_name]={}

            taxonomy_schema[attribute_name]['Task Type']=attribute_task.capitalize()
            if attribute_task.lower()=="classification":
                taxonomy_schema[attribute_name]['Values']=attribute_values
            else:
                taxonomy_schema[attribute_name]['Sample Values']=attribute_values
            
            taxonomy_schema[attribute_name]["Return Type"]=attribute_return

            # print("Taxonomy Schema is ",taxonomy_schema)
            
        
        
        tic(timing_dict, f"L{depth}")
        res = agent(context=context, prompt=attribute_prompt, image_path=image_path, tax_vals=taxonomy_schema)
        print("Raw Agent Response:",res)
        classification_result[f"L{depth}"] = res
        toc(timing_dict, f"L{depth}")

    return classification_result

# Process Input and Run Inference
def process_input_and_infer(csv_path):
    configs, classification_prompt, attribute_prompt, custom_prompt = load_config_and_prompts()
    taxonomy_tree = create_taxonomy_tree()
    df = pd.read_csv(csv_path)
    
    # df=df.tail(1)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing items"):
        context = f"Title Of the Product: {row['title']}\nDescription of the Product: {row['description']}"
        image_url = row['image url']
        image_path = check_cache(image_url=image_url)
        
        try:
            # Generate output with L0, L1, etc.
            standard_output = traverse_tree_and_classify(
                agent=inference_by_chatgpt,
                max_depth=100,
                context=context,
                image_path=image_path,
                classification_prompt=classification_prompt,
                attribute_prompt=attribute_prompt,
                taxonomy_tree=taxonomy_tree
            )
            
            # Generate output with custom prompt
            # custom_output = inference_by_chatgpt(context=context, image_path=image_path, prompt=custom_prompt)
            
            # Concatenate outputs to the DataFrame
            df.at[index, f'Standard Output'] = json.dumps(standard_output,indent=4)
            # df.at[index, 'Custom_Output'] = json.dumps(custom_output,indent=4)
            
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            print(f"Stack Trace: {traceback.format_exc()}")
    
    # Save the updated DataFrame
    output_path = f"output_{os.path.basename(csv_path)}"
    df.to_csv(output_path, index=False)
    print(f"Updated CSV saved to: {output_path}")

    print("Cleaning Up the CSV")
    new_df_lis=convert_demosite_format(df.to_dict(orient="records"))
    new_df=pd.DataFrame(new_df_lis)

    new_df.to_csv(output_path.replace("output","output_cleaned"), index=False)



if __name__ == "__main__":
    csv_path = input("Enter the path to your CSV file: ")
    process_input_and_infer(csv_path)