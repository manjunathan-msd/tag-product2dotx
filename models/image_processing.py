# Import libraries
import time
import pandas as pd
from utils.image_utils import download


# Check the image is square or not
class SquareImageSizeChecker:
    def __init__(self, **configs):
        self.min_size = configs['min_size']
        self.max_size = configs['max_size']
        
    def get_images(self, data: dict, cols: list):
        images = []
        for col in cols:
            if not pd.isna(data[col]):
                try:
                    image = download(data[col])
                    images.append(image)
                except Exception as err:
                    pass
        return images
        
    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        self.image_cols = taxonomy_dict['default_image_cols']
        images = self.get_images(data_dict, self.image_cols)
        image = images[0]
        width, height = image.size
        end = time.time()
        if width == height and width >= self.min_size and width <= self.max_size and height >= self.min_size and width <= self.max_size:
            return 'Yes', 0, 0, end - start
        else:
            return 'No', 0, 0, end - start