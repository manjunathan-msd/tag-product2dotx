# Import libraries
import ast
import re
import pandas as pd



# Convert the tags from Ln format to Breadcrumb-Tags format
def format_tags(tags: dict):
    res = {}
    levels = [re.search(r'\bL\d+\b', x).group() for x in tags.keys() if re.search(r'\bL\d+\b', x)]
    res['Breadcrumb'] = ' > '.join(tags[x]['Label'] for x in levels[:-1])
    res['Tags'] = {k: v['Label'] for k, v in tags[levels[-1]].items()}
    return res

# Seperate breadcrumb and tags as different columns from the results column  
def format_result(df: pd.DataFrame):
    res = []
    for row in df.to_dict(orient='records'):
        if isinstance(row['results'], str):
            results = ast.literal_eval(row['results'])
        else:
            results = row['results']
        row['Breadcrumb'] = results['Breadcrumb']
        row['Tags'] = results['Tags']
        res.append(row)
    res = pd.DataFrame(res)
    res.drop(columns='results', inplace=True)
    return res