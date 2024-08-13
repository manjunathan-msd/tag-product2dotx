# Import librraies
import numpy as np
import json
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree
from utils.image import download
from utils.string import get_most_similar
from models.phi3 import *


'''
The class can tag data when a TaxonomyTree and data is given
'''
class Tagger:
    def __init__(self, taxonomy: TaxonomyTree, **configs):
        self.taxonomy = taxonomy
        # Load model
        if configs['model']['name'] == 'phi3':
            if len(configs['model']['parameters']):
                self.model = Phi3(**configs['model']['parameters'])
            else:
                self.model = Phi3()
        self.mode = configs['inference']['level']
        # Read prompts
        self.prompts = {}
        with open(configs['prompts']['leaf_prompt_path']) as fp:
            self.prompts['leaf_prompt'] = fp.read()
        with open(configs['prompts']['non_leaf_prompt_path']) as fp:
            self.prompts['non_leaf_prompt'] = fp.read()
        with open(configs['prompts']['note']) as fp:
            self.prompts['note'] = fp.read()

    def get_context(self, row: dict, text_cols: list):
        text = ''
        for col in text_cols:
            text += col + ': ' + row[col]
        return text
    
    def get_image(self, row: dict, image_col: str):
        image_url = row[image_col]
        return image_url 

    def tag(self, context: str, image_url: str = None):
        res = {}
        ptr, depth = self.taxonomy.get('root'), 0
        while ptr:
            labels = [x for x in ptr.get('labels')]
            if len(labels) == 1 and ptr.get('task') == 'Classification':
                res[f'L{depth}'] = labels[0]
            else:
                if ptr.get('Classification / Extraction') == 'Classification' and ptr.get('node_type') == 'category':
                    prompt = self.prompts['non_leaf_prompt'].format(context=context, labels=labels)
                    resp = self.model(prompt=prompt, image_url=image_url)
                    if resp == 'Not specified':
                        return res
                    label = get_most_similar(labels, resp)
                    res[f'L{depth}'] = label
                    ptr = ptr.get('children')[labels.index(label)]
                    depth += 1
                elif ptr.get('Classification / Extraction') == 'NA' and ptr.get('node_type') == 'NA':
                    temp = {}
                    if self.mode == 'attribute':
                        for attr in ptr.get('children'):
                            attribute = attr.get('name').split(' > ')[-1].strip()
                            labels = attr.get('labels')
                            taxonomy = json.dumps(attr.get('metadata'), indent=2)
                            prompt = self.prompts['leaf_prompt'].format(taxonomy=taxonomy, context=context, note=self.prompts['note'], labels=labels, attribute=attribute)
                            resp = self.model(prompt=prompt, image_url=image_url)
                            temp[attr.get('name').split(' > ')[-1]] = resp
                        res[f'L{depth}'] = temp
                        ptr = None
                    else:
                        pass
                    # Rest of the code is filled here for different policy
        return res          

    def __call__(self, df: pd.DataFrame, text_cols: list, image_col: str = None, note: str = None):
        if image_col is None:
            df['image_path'] = np.nan
            image_col = 'image_path'
        if note is None:
            with open('prompts/note.txt') as fp:
                note = fp.read()
        res = []
        for row in df.to_dict(orient='records'):
            image_url = self.get_image(row, image_col)
            context = self.get_context(row, text_cols)
            tags = self.tag(context, image_url)
            tags = json.dumps(tags, indent=4)
            row['results'] = tags
            res.append(row)
        res = pd.DataFrame(res)
        if res[image_col].isnull().sum() == res.shape[0]:
            res.drop(columns=image_col, inplace=True)
        return res