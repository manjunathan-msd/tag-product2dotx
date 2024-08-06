# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillValues, FillMetaData
import tensorflow_hub as hub

# Read data
df = pd.read_csv('dumps/Taxonomy SneakerDrunk - apparel- IGNORE.csv')

# Declare object
obj = FillMetaData(df, n_levels=4)

# # Fill data
# '''
# strtegy is the optional parameter
# To use llm pass strtegy=llm 
# To use similarity search using universal encoder use strategy='similarity'
# Synonym filling is always done by llm.
# If you are not sure about what to use, leave it empty. It will decide by itself.
# '''
filled_df, syn_df = obj(strategy='llm', synonym_policy=1)
print(filled_df.isnull().sum())
print(syn_df.isnull().sum())
filled_df.to_csv('dumps/apparel_metadata.csv', index=False)
syn_df.to_csv('dumps/apparel_synonyms.csv', index=False)