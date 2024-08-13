# Import librraies
import pandas as pd
from taxonomy_builder.utils import TaxonomyTree



'''
The class can tag data when a TaxonomyTree and data is given
'''
class Tagger:
    def __init__(self, taxonomy: TaxonomyTree, **configs):
        self.taxonomy = taxonomy

    def 