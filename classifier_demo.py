import pandas as pd
#Models
from models.gpt_model import inference_by_chatgpt
from models.hpt_model import HPT
#Tagger
from tagger.tagger import Tagger2dotX
from utils.postprocess_utils import convert_demosite_format

#Initializing Model:
agent_llm=HPT()

classifier=Tagger2dotX(use_meta=True)

def apply_classifier(row):
    result = classifier(row)
    print(result)
    return pd.Series(result)

def process_dataframe(df, use_meta=True):

    subset_df = df[['title', 'description', 'image url']].copy()

    results = subset_df.apply(apply_classifier, axis=1)

    df['Standard Output'] = results['Standard Output']
    df['Custom Output'] = results['Custom Output']

    return df

def postprocess_df(df):
    df_lis=df.to_dict(orient="records")
    for rec in df_lis:
        res=convert_demosite_format(rec)
        rec.update(res)
    
    return pd.DataFrame(df_lis)

if __name__ == "__main__":
    # csv_path = input("Enter the path to your CSV file: ")
    csv_path="dumps/RewardStyle_VAL_small.csv"
    df=pd.read_csv(csv_path)
    df=df.tail(1)
    intermediate=process_dataframe(df)
    final=postprocess_df(intermediate)

    final.to_csv("temp.csv",index=False)
