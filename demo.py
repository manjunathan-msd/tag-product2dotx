# Import libraries
import json
import yaml
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree
# from inference.utils import Tagger

# # Read data
url = 'https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=1501522586'
df = pd.read_csv(url)
df = df.head(4)
print(df)

# Declare object
obj = FillAttributes()
df = obj(df)
print(df)
df.to_csv('dumps/temp.csv', index=False)
# print(FillAttributes.get('Fashion > Clothing > Men > Shorts', note='Add -Yahoo after each value'))

# # Read data
# df = pd.read_csv('dumps/ELECTRONICS MASTER TAXONOMY - DEV - Taxonomy class.csv')
# df = df.head(50)
# obj = FillMetadata()
# df = obj(df)
# df.to_csv('dumps/temp.csv', index=False)

# Create taxonomy tree
# tree = TaxonomyTree()
# tree(df)


# # Create tagger object
# with open('configs/tag_configs.yaml') as fp:
#     configs = yaml.safe_load(fp)
# tagger = Tagger(tree, **configs)
# tagger(df, text_cols=[], )

