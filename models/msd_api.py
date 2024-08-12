# Import libraries
import requests


'''
Class which can predict image guidelines for Miravia
'''
class MiraviaModel:
    def __init__(self, url: str, headers: dict, catalog_id: str, graph_id: str, feed_id: str):
        self.url = url
        self.headers = headers
        self.catalog_id = catalog_id
        self.graph_id = graph_id
        self.feed_id = feed_id

    def __call__(self, image_url: str, context: str):
        payload = {
            "catalog_id": self.catalog_id,
            "graph_id": self.graph_id,
            "feed_id": self.feed_id,
            "input": {
                'image_url': image_url,
                'title_english': context,
                'product_id': 'demosite_cyborg_hit'
            }
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response = response.json()
        res = {}
        for attr in response['data']['msd_tags'][0]['attributes']:
            res[attr['name']] = attr['results'][0]['value']
        return res


