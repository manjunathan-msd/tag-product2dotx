# Import libraries
import sys
sys.path.append('/home/ubuntu/efs/users/manjunathan/RewardStyle-POC/HPT')
sys.path.append('/home/ubuntu/efs/users/manjunathan/RewardStyle-POC/template_utils/')
from PIL import Image
# from template_utils import check_cache
from vlmeval.config import supported_VLM



# Class to use HPT for inference
class HPT:
    def __init__(self):
        self.model = supported_VLM["hpt-air-1-5"]()

    def __call__(self, context: str, image_path: str, prompt: str, tax_vals: str = None):
        if tax_vals:
            formatted_prompt = prompt.format(context=context, taxonomy=tax_vals)
        else:
            formatted_prompt = prompt.format(context=context)
        response = self.model.generate(prompt=formatted_prompt, image_path=image_path, dataset='demo')
        return response

