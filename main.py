import boto3 
import json
import logging

from utils import *
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from models.hpt_model import HPT

def initialize_node(node_config, **kwargs):
    """
    Initialize node
    """
    agent = HPT()
    log.info("LOADED MODEL HOPEFULLY")
    log.info(agent)
    
    


def process(data):
    log.info("HIIIII")
    return "True"

    
    


if __name__ == '__main__':
    pass
