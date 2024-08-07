import pandas as pd
from tagger.tagger import Tagger2dotX

classifier=Tagger2dotX(use_meta=True)

def apply_classifier(row):
    result = classifier(row)
    print(result)
    return pd.Series(result)

def process_dataframe(df, use_meta=True):

    subset_df = df[['title', 'description', 'image url']].copy()

    results = subset_df.apply(apply_classifier, axis=1)

    df['Standard_Output'] = results['Standard_Output']
    # df['Custom_Output'] = results['Custom_Output']

    return df

if __name__ == "__main__":
    # csv_path = input("Enter the path to your CSV file: ")
    csv_path="dumps/RewardStyle_VAL_small.csv"
    df=pd.read_csv(csv_path)
    df=df.tail(1)
    print(df.columns)
    process_dataframe(df)
