# Import libraries
import pandas as pd


# Function to convert ST Mapping sheet to Level Taxonomy
def convert(df: pd.DataFrame, meta_df: pd.DataFrame = None):
    df = df[['Client Product Type', 'Client Attribute', 'Client Attribute Value']]
    df = df.groupby(['Client Product Type', 'Client Attribute'])['Client Attribute Value'].apply(lambda x: ','.join(x)).reset_index()
    df['Classification / Extraction'] = ''
    if isinstance(meta_df, pd.DataFrame):
        meta_df = meta_df[['Client Product Type', 'Client Attribute', 'Multi Value']]
        df = pd.merge(df, meta_df, on=['Client Product Type', 'Client Attribute'], how='left')
        df.rename(columns={'Multi Value': 'Single Value / Multi Value'}, inplace=True)
        df['Single Value / Multi Value'] = df['Single Value / Multi Value'].replace({True: 'Multi Value', False: 'Single Value'})
    else:
        df['Single Value / Multi Value'] = ''
    df['Input Priority'] = ''
    df.rename(columns={'Client Product Type': 'L0', 'Client Attribute': 'L1', 'Client Attribute Value': 'L2'}, inplace=True)
    return df