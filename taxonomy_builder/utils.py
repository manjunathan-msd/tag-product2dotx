# Import libraries
import re
import numpy as np
import pandas as pd
        
# Node of the taxonomy tree
class TaxonomyNode:
    # Constructor
    def __init__(self, name: str, metadata_names: list = ['Classification / Extraction', 'Single Value / Multi Value', 
                                                          'Data Type', 'Ranges', 'Units', 'Input Priority']):
        self.name = name
        self.children = []
        self.labels = []
        self.node_type = None
        self.synonyms = {}
        self.metadata = {}
        for x in metadata_names:
            self.metadata[x] = None
    
    def get(self, x: str):
        if x == 'root':
            return self.root
        elif x == 'name':
            return self.name
        elif x == 'children':
            return self.children
        elif x == 'labels':
            return self.labels
        elif x == 'node_type':    
            return self.node_type
        else:
            return self.metadata[x]
    
    def add(self, x: str, val: None):
        if x == 'children':
            self.children.append(val)
        elif x == 'labels':
            self.labels = val
        elif x == 'node_type':
            self.node_type = val
        elif x == 'synonyms':
            self.synonyms = val  
        else:
            self.metadata[x] = val  
    
    def __str__(self):
        children_name = [children.get('name') for children in self.children]
        res = f"Name: {self.name}\nChildren: {children_name}\nLabel: {self.labels}\nNode Type: {self.node_type}\nSynonyms: {self.synonyms}\n"
        for x, y in self.metadata.items():
            res += f"{x}: {y}\n"
        return res
    
    def __repr__(self):
        children_name = [children.get('name') for children in self.children]
        res = f"Name: {self.name}\nChildren: {children_name}\nLabel: {self.labels}\nNode Type: {self.node_type}\nSynonyms: {self.synonyms}\n"
        for x, y in self.metadata.items():
            res += f"{x}: {y}\n"
        return res

                   
class TaxonomyTree:
    def __init__(self):
        self.n_levels = None
        self.root = TaxonomyNode(name='root')
        self.res = ''
    
    @staticmethod
    def preorder(root):
        if len(root.get('children')) == 0:
            print(root)
            return
        print(root)
        for children in root.get('children'):
            TaxonomyTree.preorder(children)
    
    def __str__(self):
        self.preorder(self.root)
        return ''

    def add(self, row: list, meta_columns: list, syn_df: pd.DataFrame = None):
        ptr = self.root
        levels, leaf_values = list(row.values())[:self.n_levels], list(row.values())[self.n_levels:-1]
        levels = [x.strip() for x in levels if not pd.isna(x)]
        for node in levels:
            flag = True
            for children in ptr.get('children'):
                if children.get('name').split(' > ')[-1] == node:
                    ptr = children
                    flag = False 
                    break
            if flag:
                if node == levels[-1]:
                    labels=[x.strip() for x in leaf_values[0].split(',')]
                    newnode = TaxonomyNode(
                        name=ptr.get('name') + ' > ' + node if ptr.get('name') != 'root' else node,
                        metadata_names=meta_columns
                    )
                    for x, y in zip(meta_columns, leaf_values[1:]):
                        newnode.add(x, y)
                    newnode.add('labels', labels)
                    if syn_df:
                        for x in labels:
                            syns = syn_df[(syn_df['Breadcrumb']==ptr.get('name')) & (syn_df['A']==x)]['Synonyms'].to_list()
                            if len(syns) > 0:
                                syns = syns[0]
                            else:
                                continue
                            newnode.add('synonyms')(x, syns)
                    ptr.add('node_type', 'NA')
                    for x in meta_columns:
                        ptr.add(x, 'NA')
                else:
                    newnode = TaxonomyNode(
                        name=ptr.get('name') + ' > ' + node if ptr.get('name') != 'root' else node,
                    )
                ptr.add('children', newnode)
                ptr = newnode
    
    @staticmethod
    def add_metadata(root: TaxonomyNode, metadata_names: list):
        if len(root.get('children')) == 0:
            root.add('node_type', 'attribute')
            return
        if root.get('name') == 'root':
            root.add('labels', [children.get('name').split(' > ')[-1] for children in root.get('children')])
            root.add('node_type', 'NA')
            for x in metadata_names:
                root.add(x, 'NA')
        elif root.get('Classification / Extraction') == 'NA':
            pass
        else:
            if 'Classification / Extraction' in metadata_names:
                root.add('Classification / Extraction', 'Classification')
            if 'Single Value / Multi Value' in metadata_names:
                root.add('Single Value / Multi Value', 'Single')
            root.add('node_type', 'category')
            root.add('labels', [children.get('name').split(' > ')[-1] for children in root.get('children')])
            for x in metadata_names:
                if x in ['Classification / Extraction', 'Single Value / Multi Value']:
                    continue
                root.add(x, 'NA')
        for children in root.get('children'):
            TaxonomyTree.add_metadata(children, metadata_names)

    def __call__(self, df: pd.DataFrame, syn_df: pd.DataFrame=None, meta_columns: list = ['Classification / Extraction', 
                                                                                          'Single Value / Multi Value', 
                                                                                          'Data Type', 'Ranges', 'Units', 
                                                                                          'Input Priority']):
        levels = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        levels.extend(['A', 'V'])
        self.n_levels = len(levels) - 1
        for col in meta_columns:
            if col not in list(df.columns):
                df[col] = np.nan
        df = df[levels + meta_columns]
        for row in df.to_dict(orient='records'):
            self.add(row, meta_columns, syn_df)
        TaxonomyTree.add_metadata(self.root, meta_columns)
    
