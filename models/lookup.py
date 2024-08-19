# Import libraries
from utils.string import tokenize, stem_words

class TaxonomyLookup:
    def __init__(self, word_dict: dict, return_type: str = 'single'):
        self.word_dict = {k: [x.strip() for x in v.split()] for k, v in word_dict.items()}
        self.return_type = return_type
    
    def __call__(self, prompt: str, image_url: str = None):
        res = []
        tokenized_text = tokenize(prompt)
        tokenized_text = list(set(stem_words(tokenized_text)))
        for k, vals in self.word_dict.items():
            vals = list(set(stem_words(vals)))
            for v in vals:
                if v in tokenized_text:
                    res.append(k)
        if self.return_type == 'single':
            return res[0], 'NA', 'NA', 'NA'
        else:
            return res, 'NA', 'NA', 'NA'
            
                    
        
        