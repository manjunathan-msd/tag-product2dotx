# Import libraries
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree,TaxonomyNode,print_info
from utils.general_utils import downloadable_url
import random


# Read data
df = pd.read_csv('dumps/beauty_metadata.csv')
print("Data: \n", df.head())
# df = df.iloc[150:201, :]

# Create a taxonomy tree
taxonomy = TaxonomyTree()

# # Traverse the tree
# def preorder(root):
#     if len(root.get_children()) == 0:
#         print(root)
#         return
#     print(root)
#     for children in root.get_children():
#         preorder(children)

# for row in df.to_dict(orient='records'):
#     row = list(row.values())
#     taxonomy.add(row)

# # Add metadata to taxonomy
# TaxonomyTree.add_metadata(taxonomy.root)


# # Traverse the tree
# preorder(taxonomy.root)

taxonomy(df)
print(taxonomy)