# Import libraries
import json
import yaml
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger

# Create taxonomy tree
tax_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=2055398816')
tree = TaxonomyTree()
tree(tax_df)
print(tree)


# Create tagger object
with open('configs/tag_configs.yaml') as fp:
    configs = yaml.safe_load(fp)
df = pd.read_csv('dumps/data.csv')
tagger = Tagger(tree, **configs)
tagger(df, text_cols=['title'], image_col='image_url')

