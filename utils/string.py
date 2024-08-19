# Import librraies
from typing import Union, List
import re
from fuzzywuzzy import process
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer



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
    # Revomre trailing comma
    if text[-1] == ',':
        text = text[:-1]
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

# Tokenize words
def tokenize(text: str, level: str = 'word'):
    if level == 'word':
        # Lowercase the text
        text = text.lower()
        # Remove punctuations by space
        text = re.sub(r'[^\w\s-]', '', text)
        # Remove extra spaces
        text = re.sub(r'[ \t]+', ' ', text).strip()
        # Tokenize text
        tokenized_words = word_tokenize(text)
        return tokenized_words
    else:
        raise ValueError("Invalid parameter of level for tokenization!")
    
    
# Stemming of words
def stem_words(words: Union[str, List[str]]):
    stemmer = PorterStemmer()
    if isinstance(words, str):
        return stemmer.stem(words.lower())
    else:
        return [stemmer.stem(x.lower()) for x in words]
        

    
