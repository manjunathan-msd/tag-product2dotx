# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata
from taxonomy_builder.utils import TaxonomyTree

# # Read data
url = 'dumps/ELECTRONICS MASTER TAXONOMY - DEV - Taxonomy class.csv'
df = pd.read_csv(url)
df = df.head(30)
print(df)

# Declare object
# obj = FindSynonyms()
# df = obj(df, policy=2, note='Return -yahooooooo with each synonym')
# print(df)
# df.to_csv('dumps/temp.csv', index=False)
# print(FillAttributes.get('Fashion > Clothing > Men > Shorts', note='Add -Yahoo after each value'))

# # Read data
# df = pd.read_csv('dumps/ELECTRONICS MASTER TAXONOMY - DEV - Taxonomy class.csv')
# df = df.head(50)
# obj = FillMetadata()
# df = obj(df)
# df.to_csv('dumps/temp.csv', index=False)

# Create taxonomy tree
tree = TaxonomyTree()
tree(df)
print(tree)

