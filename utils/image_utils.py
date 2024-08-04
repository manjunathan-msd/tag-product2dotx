import pandas as pd
import cv2
import urllib
import numpy as np
import traceback
from PIL import Image
import imagehash

import os


def read_image(image_url):

    """
    I/P: Image URL
    O/P: Image, Height of Image, Width of Image.
    """

    try:
        resp = urllib.request.urlopen(image_url)
    except:
        raise Exception("URL not accessible")
    
    #hash_str = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
    
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    height = image.shape[0]
    width = image.shape[1]
    
    return image

def check_cache(image_url, hash_size=32, base_path= "/efs/users/manjunathan/RewardStyle-POC/data/image_data"):
    
    try:
        np_image=read_image(image_url)
    except:
        print("Image Download Failed")
        print(traceback.format_exc())
        
    PIL_image = Image.fromarray(np.uint8(np_image)).convert('RGB')
    ratio = int((min(PIL_image.size) * 1.0) / hash_size)

    # gen hash
    hash_val = str(imagehash.phash(PIL_image, hash_size=hash_size, highfreq_factor=ratio))
    hash_val_small=hash_val[-10:]

    
    image_path=f"{base_path}/{hash_val_small}.jpg"
    
    if not os.path.exists(image_path):
        PIL_image = Image.fromarray(np.uint8(np_image)).convert('RGB')
        PIL_image.save(image_path)
        
    
    return image_path


# def display_image(image_path,figsize=(400, 400)):
#     try:
#         # Open an image file
#         img = Image.open(image_path)

#         #Resize Image
#         img = img.resize((figsize[0], figsize[1]))
        
#         # Display image
#         display(img)
#     except IOError as e:
#         print(f"Error opening image: {e}")


    

