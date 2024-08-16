# import libraries
from PIL import Image
import requests
from io import BytesIO


# Function to download image
def download(image_url: str):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image