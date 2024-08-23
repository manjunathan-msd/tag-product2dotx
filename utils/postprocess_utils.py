import pandas as pd
import json
import re
import time
import ast

def process_json(data):
    breadcrumb = []
    attributes = {}
    
    for key in sorted(data.keys()):
        value = data[key]
        cleaned_value = clean_and_extract_dict(value)
        if cleaned_value:
            attributes = cleaned_value
            break
        elif not isinstance(value, (dict, list)):
            breadcrumb.append(str(value))
    return '>'.join(breadcrumb), attributes

def clean_and_extract_dict(value):
    if isinstance(value, dict):
        return value
    
    if not isinstance(value, str):
        return {}

    # Find the leftmost '{' and rightmost '}'
    left_brace = value.find('{')
    right_brace = value.rfind('}')

    if left_brace == -1 or right_brace == -1 or left_brace >= right_brace:
        return {}

    # Extract the potential dictionary string
    dict_str = value[left_brace:right_brace+1]

    # Clean up the string (replace single quotes with double quotes)
    dict_str = re.sub(r"'", '"', dict_str)

    try:
        # Attempt to parse the cleaned string as JSON
        return json.loads(dict_str)
    except json.JSONDecodeError:
        # If parsing fails, try to salvage what we can
        try:
            # Remove the last comma and any incomplete key-value pairs
            dict_str = re.sub(r',\s*"[^"]*"\s*:\s*([^,}]*)$', '', dict_str)
            dict_str += '}'  # Ensure the dictionary is closed
            return json.loads(dict_str)
        except json.JSONDecodeError:
            try:    
                dict_str = dict_str.replace("}}","}")
                dict_str = re.sub(r'"\s*:\s*"([^"]*?)"(,|})', lambda m: '": "{}"{}'.format(m.group(1).replace('"', '\\"'), m.group(2)), dict_str)
                return json.loads(dict_str)
            except:
                try:
                    salvaged_dict = {}
                    potential_items = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', dict_str)
                    for key, value in potential_items:
                        if '"' not in value:
                            salvaged_dict[key] = value
                    return salvaged_dict
                except:
                    return {}
                
def remove_not_specified(json_str):
    try:
        attr_dict = json.loads(json_str)
    except json.JSONDecodeError:
        print("Error", json_str)
        return {}

    print(attr_dict)
    try:
        cleaned_dict = {k: v for k, v in attr_dict.items() if str(v).lower().strip() not in  ["not specified","not applicable","na","not available","not mentioned"]}
    except:
        attr_dict=ast.literal_eval(attr_dict)
        cleaned_dict = {k: v for k, v in attr_dict.items() if str(v).lower().strip() not in  ["not specified","not applicable","na","not available","not mentioned"]}


    return cleaned_dict

def transform_breadcrumb(breadcrumb):
    if pd.isna(breadcrumb):
        return breadcrumb
    
    levels = breadcrumb.split('>')
    transformed_levels = []
    
    for level in levels:
        level = level.strip()
        
        if level.lower() == "not specified" or len(level) > 100:
            continue
        
        words = level.split()
        transformed_words = [word.lower().capitalize() for word in words]
        transformed_level = ' '.join(transformed_words)
        transformed_levels.append(transformed_level)
    
    if not transformed_levels:
        return breadcrumb
    
    return '>'.join(transformed_levels)


def convert_demosite_format(rec):

    op={}

    if pd.notna(rec.get("Standard Output")):
        main_attributes=json.loads(rec.get("Standard Output"))
        
        breadcrumb,attributes=process_json(main_attributes)
        breadcrumb_processed=transform_breadcrumb(breadcrumb)
        attributes_processed=remove_not_specified(json.dumps(attributes))

        op["Breadcrumb"]=breadcrumb_processed
        op["Attributes"]=attributes_processed

    
    if pd.notna(rec.get("Custom Output",None)):
        op["Extras"]=remove_not_specified(rec.get("Custom Output"))

    
    return op