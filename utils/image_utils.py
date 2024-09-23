# import libraries
import cachetools
import requests
from io import BytesIO
import base64
from PIL import Image
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
# Declare a cache for the encoding
base64_cache = cachetools.LRUCache(maxsize=10)



# Function to download image
def download(image_url: str):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

# Function to create ranodom user agent
def random_agent():
    # Configure the user agent generator
    software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
    # Generate a random user-agent
    random_user_agent = user_agent_rotator.get_random_user_agent()
    return random_user_agent


# Function to download and encode image
@cachetools.cached(base64_cache)
def encode_image(image_url: str):
    # Download the image
    headers = {
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        'Accept': '*/*',
        'User-Agent': random_agent()
    }
    response = requests.get(image_url, headers=headers)
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
