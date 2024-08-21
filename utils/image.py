# import libraries
import requests
from io import BytesIO
import base64
from PIL import Image



# Function to download image
def download(image_url: str):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

# Function to download and encode image
def encode_image(image_url: str):
    # Download the image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("Failed to download image")
    # Open the image with PIL
    image = Image.open(BytesIO(response.content))
    # Resize the image if any dimension is greater than 256
    if image.size[0] > 512 or image.size[1] > 512:
        image = image.resize((512, 512))
    # Convert the image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the base64 string
    return img_base64