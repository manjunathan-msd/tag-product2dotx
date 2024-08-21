# Import libraries
import time
import requests
from models.chatgpt import ChatGPT


'''
The class can hit the current fashion models of MSD and get tags from that.
'''
class FashionAPI:
    def __init__(self, url: str, headers: dict, catalog_id: str, graph_id: str, feed_id: str):
        self.url = url
        self.headers = headers
        self.catalog_id = catalog_id
        self.graph_id = graph_id
        self.feed_id = feed_id

    def __call__(self, prompt: str, image_url: str = None):
        start = time.time()
        payload = {
            "catalog_id": self.catalog_id,
            "graph_id": self.graph_id,
            "feed_id": self.feed_id,
            "input": {
                'image_url': image_url,
                'title_english': '',
                'product_id': 'demosite_cyborg_hit'
            }
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response = response.json()
        res = {}
        for attr in response['data']['msd_tags'][0]['attributes']:
            res[attr['name']] = attr['results'][0]['value']
        end = time.time()
        resp = ''
        for x, y in res.items():
            resp += f'{x}: {y}\n'
        return res, 'NA', 'NA', end - start



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
        