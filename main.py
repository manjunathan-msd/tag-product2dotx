
import logging

from utils import *
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from models.hpt_model import HPT

import os
from azure.storage.blob import BlobServiceClient


def download_model_from_azure(account_name, credentials, container):
    log.info("Downloading weights from azure")
    account_url = f"https://{account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url, credential=credentials)
    container_client = blob_service_client.get_container_client(container=container)
    blobs = container_client.list_blobs(name_starts_with="HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal/")
    local_model_path = './HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal'
    os.makedirs(local_model_path, exist_ok=True)

    for blob in blobs:

        log.info("Downloading blob - %s",blob.name)
        blob_client = container_client.get_blob_client(blob)
        download_file_path = os.path.join(local_model_path, os.path.relpath(blob.name, "HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal/"))
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        log.info("Downloaded!")
    log.info("All weights downloaded successfully")
    

def initialize_node(node_config, **kwargs):
    """
    Initialize node
    """
    download_model_from_azure(os.environ['ACCOUNT_NAME'], os.environ['ACCOUNT_KEY'], os.environ['MAPPING_CONFIG_BUCKET'])
    agent = HPT()
    log.info("LOADED MODEL SUCCESSFULLY!")
    log.info(agent)
    
    


def process(data):
    log.info("HIIIII")
    return "True"

    
    


if __name__ == '__main__':
    pass
