# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillMetadata

# Read taxonomy
taxonomy = pd.read_csv('https://docs.google.com/spreadsheets/d/14QL8oscsg_Pdl2hA6dmNcBk6TwwCLFgA5aopxW4bYVc/export?format=csv&gid=412225439')
taxonomy.dropna(inplace=True)
obj = FillMetadata()
taxonomy = obj(taxonomy)
taxonomy.to_csv('Champagne & Sparkling Wine_Taxonomy.csv', index=False)
