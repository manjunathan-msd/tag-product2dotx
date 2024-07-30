# Import libraries
import pandas as pd



# Node of the taxonomy tree
class TaxonomyNode:
    def __init__(self, name: str, input_priority: str, labels: str = None, task: str = None, return_type: str = None):
        self.name = name
        self.task = task
        self.return_type = return_type
        self.input_priority = input_priority
        self.children = []
        self.labels = labels
        self.node_type = None
    
    def add_child(self, newnode):
        self.children.append(newnode)
    
    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def add_labels(self, labels: list):
        self.labels = labels

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
    
    def __str__(self):
        children_name = [children.get_name() for children in self.children]
        return f"Name: {self.name}\nTask Type: {self.task}\nLabels: {self.labels}\nReturn Type: {self.return_type}\nInput Type: {self.input_priority}\n" \
               f"Node Type: {self.node_type}\nChildren: {children_name}\n===================="



# Convert the taxonomy to a tree like structure
class TaxonomyTree:
    def __init__(self, n_levels: int):
        self.n_levels = n_levels
        self.root = TaxonomyNode(name='root', input_priority='NA')
    
    @staticmethod
    def add_metadata(root: TaxonomyNode):
        if len(root.get_children()) == 0:
            root.add_node_type('attribute')
            return
        if root.get_name() == 'root':
            root.add_task('NA')
            root.add_return_type('NA')
            root.add_node_type('NA')
            root.add_labels([children.get_name().split('>')[-1] for children in root.get_children()])
        elif root.get_task() == 'NA':
            pass
        else:
            root.add_task('classification')
            root.add_return_type('single')
            root.add_node_type('category')
            root.add_labels([children.get_name().split('>')[-1] for children in root.get_children()])
        for children in root.get_children():
            TaxonomyTree.add_metadata(children)
    
    def add(self, row: list):
        ptr = self.root
        levels, leaf_values, input_priority = row[:self.n_levels], row[self.n_levels:-1], row[-1]
        levels = [x for x in levels if not pd.isna(x)]
        if pd.isna(input_priority):
            input_priority = []
        else:
            input_priority = ['ocr' if 'ocr' in x.lower() else 'image' if 'image' in x.lower() else 'text' for x in input_priority.split()]
        for node in levels:
            flag = True
            for children in ptr.get_children():
                if children.get_name().split('>')[-1] == node:
                    ptr = children
                    flag = False
                    break
            if flag:
                if node == levels[-1]:
                    newnode = TaxonomyNode(
                        name=ptr.get_name() + '>' + node if ptr.get_name() != 'root' else node,
                        input_priority=input_priority,
                        labels=[x.strip() for x in leaf_values[0].split(',')],
                        task=leaf_values[1].lower() if not pd.isna(leaf_values[1]) else 'NA',
                        return_type='NA' if pd.isna(leaf_values[2]) else 'single' if 'single' in leaf_values[2].lower() else 'multi'
                    )
                    ptr.add_task('NA')
                    ptr.add_return_type('NA')
                    ptr.add_node_type('NA')
                    ptr.add_input_priority('NA')
                    ptr.add_labels([])
                else:
                    newnode = TaxonomyNode(
                        name=ptr.get_name() + '>' + node if ptr.get_name() != 'root' else node,
                        input_priority=input_priority
                    )
                ptr.add_child(newnode)
                ptr = newnode
        
                    
