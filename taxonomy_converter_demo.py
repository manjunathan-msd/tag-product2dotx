# Import libraries
import pandas as pd
from taxonomy_converter.st_mapping_to_level_taxonomy import convert



# Read data
df = pd.read_csv('dumps/Copy of 2024 May Deploy_NW3_ST_Mapping_333 - Attributes.csv')
meta_df = pd.read_csv('dumps/Copy of 2024 May Deploy_NW3_ST_Mapping_333 - AttributesMeta.csv')
df = convert(df, meta_df)
df.to_csv('dumps/temp.csv', index=False)