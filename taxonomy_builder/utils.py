# Import libraries
import re
import numpy as np
import pandas as pd


# Node of the taxonomy tree
class TaxonomyNode:
    def __init__(self, name: str, input_priority: str, labels: str = None, task: str = None, return_type: str = None, 
                 data_type: str = None, ranges: str = None, units: str = None):
        self.name = name
        self.task = task
        self.return_type = return_type
        self.input_priority = input_priority
        self.data_type = data_type
        self.ranges = ranges
        self.units = units
        self.children = []
        self.labels = labels
        self.node_type = None
        self.synonyms = {}
    
    def add_child(self, newnode):
        self.children.append(newnode)
    
    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def add_labels(self, labels: list):
        self.labels = labels
    
    def get_labels(self):
        return self.labels

    def add_task(self, task: str):
        self.task = task
    
    def get_task(self):
        return self.task

    def add_return_type(self, return_type: str):
        self.return_type = return_type
    
    def get_return_type(self):
        return self.return_type
    
    def add_node_type(self, node_type: str):
        self.node_type = node_type
    
    def get_node_type(self):
        return self.node_type
    
    def add_input_priority(self, input_priority: str):
        self.input_priority = input_priority
    
    def get_input_priority(self):
        return self.input_priority
    
    def add_synonyms(self, label: str, synonyms: str):
        self.synonyms[label] = synonyms

    def get_synonyms(self):
        return self.synonyms
    
    def add_data_type(self, data_type: str):
        self.data_type = data_type

    def get_data_type(self):
        return self.data_type
    
    def add_ranges(self, ranges: str):
        self.ranges = ranges
    
    def get_ranges(self):
        return self.ranges

    def add_units(self, units: str):
        self.units = units

    def get_units(self):
        return self.units

    def __str__(self):
        children_name = [children.get_name() for children in self.children]
        return f"Name: {self.name}\nTask Type: {self.task}\nLabels: {self.labels}\nReturn Type: {self.return_type}\nInput Type: {self.input_priority}\n" \
               f"Data Type: {self.data_type}\nRanges: {self.ranges}\nUnits: {self.units}\nNode Type: {self.node_type}\nChildren: {children_name}\n====================\n"
    
    def __repr__(self):
        children_name = [children.get_name() for children in self.children]
        return f"Name: {self.name}\nTask Type: {self.task}\nLabels: {self.labels}\nReturn Type: {self.return_type}\nInput Type: {self.input_priority}\n" \
               f"Data Type: {self.data_type}\nRanges: {self.ranges}\nUnits: {self.units}\nNode Type: {self.node_type}\nChildren: {children_name}\n====================\n"

# Node of the taxonomy tree
class TaxonomyNode2:
    # Constructor
    def __init__(self, name: str, input_priority: str, metdata: dict):
        self.name = name
        self.input_priority = input_priority
        self.metadata = metdata
        self.
    


# Convert the taxonomy to a tree like structure
class TaxonomyTree:
    def __init__(self):
        self.n_levels = None
        self.root = TaxonomyNode(name='root', input_priority='NA')
        self.res = ''
    
    @staticmethod
    def add_metadata(root: TaxonomyNode):
        if len(root.get_children()) == 0:
            root.add_node_type('attribute')
            return
        if root.get_name() == 'root':
            root.add_task('NA')
            root.add_return_type('NA')
            root.add_node_type('NA')
            root.add_data_type('NA')
            root.add_ranges('NA')
            root.add_units('NA')
            root.add_labels([children.get_name().split('>')[-1] for children in root.get_children()])
        elif root.get_task() == 'NA':
            pass
        else:
            root.add_task('classification')
            root.add_return_type('single')
            root.add_node_type('category')
            root.add_data_type('NA')
            root.add_ranges('NA')
            root.add_units('NA')
            root.add_labels([children.get_name().split('>')[-1] for children in root.get_children()])
        for children in root.get_children():
            TaxonomyTree.add_metadata(children)
    
    def add(self, row: list, syn_df: pd.DataFrame):
        ptr = self.root
        levels, leaf_values, input_priority = row[:self.n_levels], row[self.n_levels:-1], row[-1]
        levels = [x for x in levels if not pd.isna(x)]
        if pd.isna(input_priority):
            input_priority = []
        else:
            input_priority = ['ocr' if 'ocr' in x.lower() else 'image' if 'image' in x.lower() else 'text' for x in input_priority.split(',')]
        for node in levels:
            flag = True
            for children in ptr.get_children():
                if children.get_name().split('>')[-1] == node:
                    ptr = children
                    flag = False
                    break
            if flag:
                if node == levels[-1]:
                    labels=[x.strip() for x in leaf_values[0].split(',')]
                    newnode = TaxonomyNode(
                        name=ptr.get_name() + '>' + node if ptr.get_name() != 'root' else node,
                        input_priority=input_priority,
                        labels=labels,
                        task=leaf_values[1].lower() if not pd.isna(leaf_values[1]) else 'NA',
                        return_type='NA' if pd.isna(leaf_values[2]) else 'single' if 'single' in leaf_values[2].lower() else 'multi',
                        data_type='NA' if pd.isna(leaf_values[3]) else 'enum' if 'enum' in leaf_values[3].lower() else 'numeric' if 'numeric' in leaf_values[3].lower() else 'string',
                        ranges='NA' if pd.isna(leaf_values[4]) else pd.isna(leaf_values[4]),
                        units='NA' if pd.isna(leaf_values[5]) else pd.isna(leaf_values[5])
                    )
                    if syn_df:
                        for x in labels:
                            syns = syn_df[(syn_df['Breadcrumb']==ptr.get_name()) & (syn_df['A']==x)]['Synonyms'].to_list()
                            if len(syns) > 0:
                                syns = syns[0]
                            else:
                                continue
                            newnode.add_synonyms(label=x, synonyms=syns)
                    ptr.add_task('NA')
                    ptr.add_return_type('NA')
                    ptr.add_node_type('NA')
                    ptr.add_data_type('NA')
                    ptr.add_ranges('NA')
                    ptr.add_units('NA')
                    ptr.add_input_priority('NA')
                    ptr.add_labels([])
                else:
                    newnode = TaxonomyNode(
                        name=ptr.get_name() + '>' + node if ptr.get_name() != 'root' else node,
                        input_priority=input_priority
                    )
                ptr.add_child(newnode)
                ptr = newnode

    @staticmethod
    def preorder(root):
        if len(root.get_children()) == 0:
            print(root)
            return
        print(root)
        for children in root.get_children():
            TaxonomyTree.preorder(children)

    def __call__(self, df: pd.DataFrame, syn_df: pd.DataFrame=None):
        levels = [re.search(r'\bL\d+\b', x).group() for x in df.columns if re.search(r'\bL\d+\b', x)]
        if 'A' in list(df.columns) and 'V' in list(df.columns):
            self.n_levels = len(levels) + 1
            levels.extend(['A', 'V'])
        else:
            self.n_levels = len(levels) - 1
        meta_columns = ['Classification / Extraction', 'Single Value / Multi Value', 'Data Type', 
                        'Ranges', 'Units', 'Input Priority']
        for col in meta_columns:
            if col not in list(df.columns):
                df[col] = np.nan
        df = df[levels + meta_columns]
        for row in df.to_dict(orient='records'):
            row = list(row.values())
            self.add(row, syn_df)
        TaxonomyTree.add_metadata(self.root)
    
    def __str__(self):
        self.preorder(self.root)
        return ''
        
                    
