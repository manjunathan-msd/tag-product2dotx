# Import librraies
from fuzzywuzzy import process


# get most similar string
def get_most_similar(string_list: list, word: str):
    similar_word, _ = process.extractOne(word, string_list)
    return similar_word