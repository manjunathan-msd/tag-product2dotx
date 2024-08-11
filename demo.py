# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillValues, FillAttributes, FindSynonyms, FillMetadata

# # Read data
# url = 'https://docs.google.com/spreadsheets/d/1z2wvzHUavp76-IXXPBeVJpkU7irMAzsuru8vSzhyxtk/export?format=csv&gid=1499665211'
# df = pd.read_csv(url)
# df = df.head(5)
# print(df)

# # Declare object
# obj = FindSynonyms()
# df = obj(df, policy=2, note='Return -yahooooooo with each synonym')
# print(df)
# df.to_csv('dumps/temp.csv', index=False)
# # print(FillAttributes.get('Fashion > Clothing > Men > Shorts', note='Add -Yahoo after each value'))

# Read data
df = pd.read_csv('dumps/ELECTRONICS MASTER TAXONOMY - DEV - Taxonomy class.csv')
df = df.head(50)
obj = FillMetadata()
df = obj(df)
df.to_csv('dumps/temp.csv', index=False)