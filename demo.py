# Import libraries
import json
import yaml
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger

# Read taxonomy, data and configs
tax_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=1499665211')
with open('configs/tag_configs.yaml') as fp:
    configs = yaml.safe_load(fp)
df = pd.read_csv('dumps/data.csv')

# Create Taxonomy Tree
tree = TaxonomyTree()
tree(tax_df)

# Do Tagging
tagger = Tagger(tree, **configs)
res = tagger(df, text_cols=['title', 'dimension', 'materials_and_finish', 'color', 'sale_price'], image_col='image_url')
res.to_csv('dumps/phi3_category_text.csv', index=False)

