# Import libraries
import json
import yaml
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger

# Read taxonomy, data and configs
tax_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=2055398816')
with open('configs/tag_configs.yaml') as fp:
    configs = yaml.safe_load(fp)
df = pd.read_csv('dumps/data.csv')
df = df.head(20)
model_dict = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=248209222')
prompt_dict = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=13838047')
model_dict = model_dict.set_index('Breadcrumb').to_dict()['Model Name']
prompt_dict = prompt_dict.set_index('Breadcrumb').to_dict()['Prompts']

# Create Taxonomy Tree
tree = TaxonomyTree()
tree(tax_df, meta_columns=['Classification / Extraction', 'Single Value / Multi Value', 'Input Priority', 
                           'Data Type', 'Ranges', 'Units'])


# Do Tagging
tagger = Tagger(tree, **configs)
res = tagger(df, text_cols=[], image_col='image_url', model_taxonomy=model_dict, additional_prompts=prompt_dict)
res.to_csv('dumps/temp.csv', index=False)

