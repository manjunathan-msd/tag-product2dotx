# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from fill_taxonomy.utils import FillMetaData



# Read data
df = pd.read_csv('dumps/Beauty taxonomy Final version - Taxonomy class (1).csv')
# df = df.sample(n=5, random_state=32)
df = df.head(10)
temp_df = df.copy()
print(temp_df)
for col in ['Classification / Extraction', 'Single Value / Multi Value', 'Input Priority']:
    temp_df[col] = np.nan

# Declare object
obj = FillMetaData(temp_df)

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
# for col in ['Classification / Extraction', 'Single Value / Multi Value', 'Input Priority']:
#     print(col)
#     print(classification_report(
#         df[col].to_list(), filled_df[col].to_list()
#     ))
#     print("=============================================")
filled_df.to_csv('dumps/beauty_metadata.csv', index=False)
syn_df.to_csv('dumps/beauty_synonyms.csv', index=False)