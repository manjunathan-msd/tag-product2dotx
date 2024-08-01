# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillTaxonomy
import tensorflow_hub as hub

# Read data
df = pd.read_csv('dumps/FASHION MASTER TAXONOMY - DEV - Taxonomy class.csv')
df = df.sample(n=40)

# Declare object
obj = FillTaxonomy(df, 5)

# Fill data
'''
strtegy is the optional parameter
To use llm pass strtegy=llm 
To use similarity search using universal encoder use strategy='similarity'
Synonym filling is always done by llm.
If you are not sure about what to use, leave it empty. It will decide by itself.
'''
filled_df, syn_df = obj()
print(filled_df.isnull().sum())
print(syn_df.isnull().sum())
filled_df.to_csv('dumps/fill_temp.csv', index=False)
syn_df.to_csv('dumps/synonyms_temp.csv', index=False)