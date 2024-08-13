# import libraries
from PIL import Image
import requests
from io import BytesIO


# Function to download image
def download(image_url: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img