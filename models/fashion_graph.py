# Import libraries
import time
import requests
import cachetools
from models.chatgpt import ChatGPT
# Declare a cache for the encoding
fashion_cache = cachetools.LRUCache(maxsize=10)

'''
The class can hit the current fashion models of MSD and get tags from that.
'''
class FashionAPI:
    def __init__(self, **configs):
        pass
    
    @cachetools.cached(fashion_cache)
    def inference(self, image_url: str):
        url = "https://use1.vue.ai/api/v1/inference?is_sync=true&is_skip_cache=true"
        headers = {
            "x-api-key": "737aaa23439a4e029ec2373da750792f",
            "Content-Type": "application/json"
        }
        data = {
            "catalog_id": "0433f9caa0",
            "feed_id": "cyborg",
            "input": {
                "image_url": [image_url],
                "product_id": "demosite_cyborg_hit_1"
            }
        }
        response = requests.post(url, headers=headers, json=data).json()
        tags = response['data']['msd_tags']
        res = {}
        for attr in tags[0]['attributes']:
            res[attr['name']] = ','.join([x['value'] for x in attr['results']])
        return res

    def __call__(self, taxonomy_dict: dict, data_dict: dict, metadata_dict: dict):
        start = time.time()
        if taxonomy_dict['inference_mode'] == 'category':
            raise ValueError("Infernce mode can't be catgeory!")
        elif taxonomy_dict['inference_mode'] == 'attribute' or taxonomy_dict['inference_mode'] == 'presets':
            image_col = taxonomy_dict['default_image_cols'][0]
            image_url = data_dict[image_col]
            response = self.inference(image_url)
            attribute = taxonomy_dict['breadcrumb'].split('>')[-1].strip()
            if attribute in response.keys():
                res = response[attribute]
            else:
                res = 'Not specified'
        else:
            raise ValueError("Wrong inference mode!")
        end = time.time()
        return res, 0, 0, end - start



'''
The class can hit the current fashion models of MSD and get tags. It also verify the tags using ChatGPT. 
''' 
class GPTWithFashionAPIFeedback:
    def __init__(self, url: str, headers: dict, catalog_id: str, graph_id: str, feed_id: str):
        self.fashion_model = FashionAPI(url, headers, catalog_id, graph_id, feed_id)
        self.gpt = ChatGPT()
        
    def __call__(self, prompt: str, image_url: str = None):
        resp, _, _, latency1 = self.fashion_model(image_url)
        prompt = f'''
        Depending on the given image and context, please verify that the predictions are correct or not:
        Input:
        ```
        Context:
        {prompt}
        Predictions:
        f{resp}
        ```
        If the predictions are correct then return the prediction as it is. If any of the predictions isn't correct then correct it and return all predictions.
        Return the predictions in the same format as it's given as input. Don't retrun any additional text. Maintain the format strictly.
        '''
        resp, input_tokens, output_tokens, latency2 = self.gpt(prompt, image_url)
        return  resp, input_tokens, output_tokens, latency1 + latency2
        