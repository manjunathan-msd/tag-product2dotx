# Import libraries
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree


# Read data
df = pd.read_csv('dumps/Beauty taxonomy Final version - Taxonomy class (1).csv')
print("Data: \n", df.head())
df = df.iloc[150:201, :]

# Create a taxonomy tree
taxonomy = TaxonomyTree(n_levels=5)

# Traverse the tree
def preorder(root):
    if len(root.get_children()) == 0:
        print(root)
        return
    print(root)
    for children in root.get_children():
        preorder(children)

for row in df.to_dict(orient='records'):
    row = list(row.values())
    taxonomy.add(row)

# Add metadata to taxonomy
TaxonomyTree.add_metadata(taxonomy.root)

# Traverse the tree
preorder(taxonomy.root)


