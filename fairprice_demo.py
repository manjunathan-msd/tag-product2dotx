# Import libraries
import os
import yaml
import pandas as pd
from fill_taxonomy.utils import FillMetadata, FindSynonyms
from taxonomy_builder.utils import TaxonomyTree
from inference.utils import Tagger


# # Read the taxonomy, data and configs
# taxonomy = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=2127293581')


# # Fill metdata
# metadata_dict = {
#     'Classification / Extraction': 'prompts/classification_or_extraction.txt', 
#     'Single Value / Multi Value': 'prompts/single_value_or_multi_value.txt', 
#     'Data Type': 'prompts/data_type.txt', 
#     'Ranges': 'prompts/ranges.txt', 
#     'Units': 'prompts/units.txt'
# }
# fill_metdata = FillMetadata(metadata=metadata_dict)
# meta_taxonomy = fill_metdata(taxonomy)
# meta_taxonomy.to_csv(os.path.join('dumps', 'fairprice_metadata.csv'), index=False)


# # Fill Synonyms
# taxonomy = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=887092763')
# fill_synonyms = FindSynonyms()
# res = fill_synonyms(taxonomy)
# res.to_csv(os.path.join('dumps', 'fairprice_synonyms.csv'), index=False)

# Build taxonomy tree
taxonomy = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=887092763')
synonyms = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=1706505387')
taxonomy_tree = TaxonomyTree()
taxonomy_tree(taxonomy, synonyms, ['Classification / Extraction', 'Single Value / Multi Value', 'Data Type', 
                                   'Ranges', 'Units'])

# Read configs
with open(os.path.join('configs', 'tag_configs.yaml')) as fp:
    configs = yaml.safe_load(fp)

# Inference
df = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=672682465')
tagger = Tagger(taxonomy_tree, **configs)
res = tagger(df, text_cols=['ocr_txt'])
res.to_csv(os.path.join('dumps', 'fairprice_res.csv'), index=False)
