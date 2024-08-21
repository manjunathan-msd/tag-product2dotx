# Import libraries
import time
import yaml
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger


# Read taxonomy
taxonomy = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=887092763')
tree = TaxonomyTree()
tree(taxonomy, meta_columns=['Classification / Extraction', 'Single Value / Multi Value', 'Data Type', 'Ranges', 'Units'])

# Read configs
with open('configs/fairprice_configs.yaml') as fp:
    configs = yaml.safe_load(fp)
    
# Read data
df = pd.read_csv('https://docs.google.com/spreadsheets/d/1o2qh9-WjXQSFZNl83ymJNsuLt_kdDsX8pUre2bJsfGU/export?format=csv&gid=1550743275')
df = df.head(3)

# Start tagging
tagger = Tagger(tree, **configs)
res = tagger(df)

# Save result
res.to_csv(f"result_{str(time.time()).replace('.', '_')}.csv", index=False)
