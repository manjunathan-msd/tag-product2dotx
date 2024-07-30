# Import libraries
import os
import requests
from io import BytesIO
import base64
from PIL import Image
import json
import requests

#Hit Demosite Graph
def hit_api(payload):
    payload = json.dumps({
    
        "catalog_id": "0433f9caa0",
        "input": payload
    })
    url = "https://use1.vue.ai/api/v1/inference?is_sync=true&is_skip_cache=true"
    headers = {
      'x-api-key': '737aaa23439a4e029ec2373da750792f',
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        response = response.json()
        return response
    except Exception as err:
        return response.text
    
def inference_by_demosite(context: str, image_path: str, prompt: str, tax_vals: str = None):
    
    resp={}
    
    payload = {
    "product_id": "107",
    "image_url": [
        image_path
    ],
    "description": context,
    "title": "",
}
    
    flag = False
    try:
        if list(resp.keys())[0] == "message":
            flag = True
    except Exception as err:
        raise Exception(err)
    while flag:
        print("Packet failed!")
        time.sleep(10)
        resp = hit_api(payload)
        try:
            if len(list(resp.keys())) != 1:
                flag = False
        except Exception as err:
            pass
        
    op=parse_attributes(resp)
    
    return op


def parse_attributes(data):
    attributes_dict = {}
    
    msd_tags = data.get("data", {}).get("msd_tags", [])
    
    for tag in msd_tags:
        for attribute in tag.get("attributes", []):
            name = attribute.get("name")
            results = attribute.get("results", [])
            values = [result.get("value") for result in results]
            attributes_dict[name] = ','.join(values)
    
    return attributes_dict
