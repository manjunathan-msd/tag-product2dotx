# Import libraries
import os
import requests
from io import BytesIO
import base64
from PIL import Image
import json
from openai import OpenAI
import requests
import traceback
import time

#Timeout Check Class- to avoid stuck vals
from functools import wraps
import signal
import boto3

secret_name = "tag-taxonomy-gpt-4o"
region_name = "us-east-1"
session = boto3.session.Session()
client = session.client(
    service_name='secretsmanager',
    region_name=region_name
)
get_secret_value_response = client.get_secret_value(
    SecretId=secret_name
)
secret = get_secret_value_response['SecretString']
secret_obj = json.loads(secret)



openai_client = OpenAI(api_key=secret_obj.get("api_key",""))


# Resize and encode image
def encode_image(image_path: str):
    image = Image.open(image_path)
    max_size = 512
    if image.width > max_size or image.height > max_size:
        image = image.resize((max_size, max_size), Image.BILINEAR)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


def inference_by_chatgpt(context: str, image_path: str, prompt: str, tax_vals: str = None):
    try:
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)

        image_str = encode_image(image_path)
        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {secret_obj.get('api_key','')}"
        }
        payload = {
          "model": "gpt-4o",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": formatted_prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/png;base64,{image_str}",
                    "detail": "high"
                  }
                }
              ]
            }
          ]
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)  # 15 seconds timeout for the request
        # print("Response:",response.text)
        response = response.json()
        response = response['choices'][0]['message']['content']
        try:
          return postprocessing_json(response)
        except Exception as err:
            return response
    except (TimeoutError, requests.exceptions.Timeout):
        print("Inference timed out after 10 seconds.")
        return {}
    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        return {}


# Postprocessing of JSON
def postprocessing_json(json_string: str):
    start = json_string.find('{')
    end = json_string.rfind('}')
    json_string = json_string[start:end+1]
    json_obj = json.loads(json_string)
    return json_obj