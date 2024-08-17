# Import libraries
import json
import yaml
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger

# Read taxonomy, data and configs
tax_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=902518976')
with open('configs/tag_configs.yaml') as fp:
    configs = yaml.safe_load(fp)
df = pd.read_csv('/home/ubuntu/efs/users/gk/miravia/training_data/collage/sample_set_to_test_llm.csv')
df = df.head(500)

# Create Taxonomy Tree
tree = TaxonomyTree()
tree(tax_df, meta_columns=['Classification / Extraction', 'Single Value / Multi Value', 'Definition'])

# Do Tagging
tagger = Tagger(tree, **configs)
res = tagger(df, text_cols=[], image_col='Image_URL')
res.to_csv('dumps/miravia_collage_v1.csv', index=False)

