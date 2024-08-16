# Import librraies
import re
from fuzzywuzzy import process


# Get most similar string
def get_most_similar(string_list: list, word: str):
    similar_word, _ = process.extractOne(word, string_list)
    return similar_word

# Postprocessing LLM response
def postprocessing_llm_response(text: str):
    # Remove quotes
    text = text.strip()
    text = text.replace('```', '')
    # Remove result or output string
    for word in ['result', 'Result', 'results', 'Results', 'output', 'outputs', 'Output', 'Outputs']:
        if text.startswith(word):
            text = text.replace(word, '')
    # Remove extra spaces
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text
    
# Create text to dict
def text_to_dict(text: str):
    res = {}
    for line in text.split('\n'):
        key = line.split(':')[0].strip()
        idx = line.find(':')
        val = line[idx+1:].strip() if idx!=-1 else ''
        res[key] = val
    return res
