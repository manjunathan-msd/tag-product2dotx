# Import libraries
import ast
import re
import json
import traceback
import pandas as pd



# Convert the tags from Ln format to Breadcrumb-Tags format
def format_tags(tags: dict):
    res = {}
    levels = [re.search(r'\bL\d+\b', x).group() for x in tags.keys() if re.search(r'\bL\d+\b', x)]
    print(levels)
    if 'Label' in tags[levels[-1]].keys() and tags[levels[-1]]['Label'] == 'Not specified':
        res['Breadcrumb'] = ' > '.join(tags[x]['Label'] for x in levels)
        res['Tags'] = {}
    else:
        res['Breadcrumb'] = ' > '.join(tags[x]['Label'] for x in levels[:-1])
        res['Tags'] = {k: v['Label'] for k, v in tags[levels[-1]].items()}
    return res

# Seperate breadcrumb and tags as different columns from the results column  
def format_result(df: pd.DataFrame):
    res = []
    for row in df.to_dict(orient='records'):
        results = json.loads(row['results'])
        levels = [re.search(r'\bL\d+\b', x).group() for x in results.keys() if re.search(r'\bL\d+\b', x)]
        if 'Label' in results[levels[-1]].keys() and results[levels[-1]]['Label'] == 'Not specified':
            row['Breadcrumb'] = ' > '.join(results[x]['Label'] for x in levels)
            row['Tags'] = {}
        else:
            row['Breadcrumb'] = ' > '.join(results[x]['Label'] for x in levels[:-1])
            row['Tags'] = {k: v['Label'] for k, v in results[levels[-1]].items()}
        res.append(row)
    res = pd.DataFrame(res)
    res.drop(columns='results', inplace=True)
    return res
