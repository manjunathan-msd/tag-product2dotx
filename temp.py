# Import libraries
import pandas as pd
from fill_taxonomy.utils import FillMetadata

# Read taxonomy
df = pd.read_csv('/home/ubuntu/efs/users/ankur/tag-product2dotx/dumps/fairprice_beer_hit_remaining_iamge_count_1.csv')
print(df.shape)
