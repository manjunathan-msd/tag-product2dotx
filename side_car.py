import os
import json
import logging
from mlaaslib import App, get_config
from main import initialize_node
from main import process as main_process

# from common_utils import setup_logger

# logger = setup_logger(__name__, logging.INFO)

# _ENV_TYPE = os.environ.get('ENV_TYPE')

# App initialize
node = App('test')

# get config
config = get_config()

# model init
model = initialize_node(config)


@node.process
def process(payload):
    
    result = main_process(payload, model)
#     logger.info(f"Final response: {result}")
    return result


if __name__ == "__main__":
    node.start()
