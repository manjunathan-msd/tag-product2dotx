from .vlm import *
# from .api import GPT4V, GPT4V_Internal, GeminiProVision, QwenVLAPI
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None


models = {
    'hpt-air-mmmu': partial(HPT),
    'hpt-air-mmbench': partial(HPT, vis_scale=392, is_crop=False),
    'hpt-air-seed': partial(HPT, is_crop=False),
    'hpt-air-demo': partial(HPT, vis_scale=392, is_crop=False),
    'hpt-air-demo-local': partial(HPT, vis_scale=392, is_crop=False, global_model_path='../HPT_AIR_HF/'),
    'hpt-air-1-5': partial(HPT1_5, global_model_path='HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimoda', vis_scale=448, prompt_template='llama3_chat'),
    'hpt-edge-1-5': partial(HPT1_5, global_model_path='HyperGAI/HPT1_5-Edge', vis_scale=490, prompt_template='phi3_chat'),
}

supported_VLM = {}
supported_VLM.update(models)