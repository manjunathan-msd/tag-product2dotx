# Import librraies
import numpy as np
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree
from models.phi3 import *

'''
The class can tag data when a TaxonomyTree and data is given
'''
class Tagger:
    def __init__(self, taxonomy: TaxonomyTree, **configs):
        self.taxonomy = taxonomy
        if configs['model']['name'] == 'phi3':
            if len(configs['model']['parameters']):
                self.model = Phi3(**configs['model']['parameters'])
            else:
                self.model = Phi3()
        self.mode = configs['extraction']['mode']
        # Read prompts
        self.prompts = {}
        with open(configs['classification']['non_leaf_classification_prompt_path']) as fp:
            self.prompts['non_leaf_classification_prompt'] = fp.read()
        with open(configs['classification']['leaf_classification_prompt_path']) as fp:
            self.prompts['leaf_classification_prompt_path'] 

    def get_context(self, row: dict, text_cols: list):
        text = ''
        for col in text_cols:
            text += row[col]
        return text
    
    def get_image(self, row: dict, image_col: str):
        image_url = row['image_col']
        return image_url 

    def tag(self, context: str, image_url: str = None):
        res = {}
        ptr, depth = self.taxonomy.get('root'), 0
        while ptr:
            labels = [x for x in ptr.get('labels')]
            if len(labels) == 1 and ptr.get('task') == 'Classification':
                res[f'L{depth}'] = labels[0]
            else:
                

    def __call__(self, df: pd.DataFrame, text_cols: list, image_col: str = None):
        if image_col is None:
            df['image_path'] = np.nan
            image_col = 'image_path'
        res = []
        for row in df.to_dict(orient='records'):
            image_url = self.get_image(row, image_col)
            context = self.get_context(row, text_cols)
            tags = self.tag(context, image_url)
            row['results'] = tags
            res.append(row)
        res = pd.DataFrame(res)
        if res[image_col].isnull().sum() == res.shape[0]:
            res.drop(columns=image_col, inplace=True)
        return res