import pandas as pd
from tagger.tagger import Tagger2dotX
from utils.postprocess_utils import convert_demosite_format
from models.msd_api import MiraviaModel
agent = MiraviaModel(
    url="https://use1.vue.ai/api/v1/inference?is_sync=true&is_skip_cache=true",
    headers = {
        'x-api-key': '7432c4722bbc4947a0b2560ce5b48d65',
        'Content-Type': 'application/json'
    },
    catalog_id = "389a5cf510",
    graph_id = "g-da333687-6519-4821-acd8-75818c2aadd1",
    feed_id = "cyborg"
)
taxonomy_df=pd.read_csv("https://docs.google.com/spreadsheets/d/1l5r8AFxSbEtottTZDH7HVdka1Ra4cKh8c0pocT_XUJc/export?gid=0&format=csv")
classifier=Tagger2dotX(agent=agent, use_meta=True, taxonomy_df=taxonomy_df)

def apply_classifier(row):
    result = classifier(row)
    print(result)
    return pd.Series(result)

def process_dataframe(df, use_meta=True):
    df.rename(columns={'Image_URL': 'image url'}, inplace=True)
    subset_df = df[['image url', 'Title']].copy()
    results = subset_df.apply(apply_classifier, axis=1)
    df['Standard Output'] = results['Standard Output']
    # df['Custom Output'] = results['Custom Output']

    return df

def postprocess_df(df):
    df_lis=df.to_dict(orient="records")
    for rec in df_lis:
        res=convert_demosite_format(rec)
        rec.update(res)
    
    return pd.DataFrame(df_lis)

if __name__ == "__main__":
    # csv_path = input("Enter the path to your CSV file: ")
    csv_path="dumps/Miravia - Sample Set 10.csv"
    df=pd.read_csv(csv_path)
    intermediate=process_dataframe(df)
    final=postprocess_df(intermediate)
    final.to_csv("dumps/temp.csv",index=False)
