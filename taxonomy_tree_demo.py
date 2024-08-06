# Import libraries
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree,TaxonomyNode,print_info
from utils.general_utils import downloadable_url
import random



# Create a taxonomy tree
taxonomy = TaxonomyTree(n_levels=5)



def choose_child(labels):
    """
    Placeholder function to make a decision.
    Currently selects a random label from the list.
    """
    return random.choice(labels) if labels else None

def sequential_traverse(node: TaxonomyNode, depth: int = 0):
    
    print_info(node)
    
    #Exit if your Node Type is NA and Go for Getting the Attributes
    if node.get_node_type()=="NA" and depth!=0:
        children=node.get_children()
        print(children)
        print(f"Reached a leaf node. Traversal complete.")
        return

    child_labels = [child.get_name().split('>')[-1] for child in node.get_children()]
    print(f"Available choices: {child_labels}")
    
    decision = choose_child(child_labels)
    print(f"Decision made: {decision}")
    print()

    # Find the chosen child node
    chosen_child = next((child for child in node.get_children() if child.get_name().split('>')[-1] == decision), None)
    
    if chosen_child:
        sequential_traverse(chosen_child, depth + 1)
    else:
        print(f"No matching child node found. Traversal complete.")
    
        
if __name__=="__main__":
    # csv_url=input("Enter your Taxonomy CSV URL:")
    csv_url="https://docs.google.com/spreadsheets/d/1OFkLyevuDzfzrfTW5nBAv72RO2msKzHwyUmVX_j_gd0/edit?gid=1729587615#gid=1729587615"
    print(downloadable_url(csv_url))
    tax_df=pd.read_csv(downloadable_url(csv_url))
    # tax_df=tax_df.iloc[215:,]
    # print(tax_df.head())
    # exit()
    taxonomy(tax_df)
    sequential_traverse(taxonomy.root)
    
    # print(taxonomy)