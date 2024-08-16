# Import libraries
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, pipeline
from utils.image import download


'''
Phi3 wrapper to use it as agent in workflow
'''
class Phi3:
    def __init__(self, max_new_tokens: int = 256):
        cache_dir = os.path.join('pretrained_models', 'phi3')
        os.makedirs(cache_dir, exist_ok=True)
        # Declare tokenizer, model and model
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda'
        )
        self.tokenizer  = tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            cache_dir=cache_dir,
            trust_remote_code=True, 
            device_map='cuda',
            torch_dtype='auto'
        )
        self.pipeline = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
        )
        # Define generation configurations
        self.generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

    def __call__(self, prompt: str, image_url: str):
        start = time.time()
        messages = [ 
            {"role": "system", "content": "You are a helpful assistant who is an expert of inventory management."}, 
            {"role": "user", "content": prompt}, 
        ]
        gen_text = self.pipeline(messages, **self.generation_args)[0]['generated_text'].strip()
        end = time.time()
        latency = end - start
        input_tokens = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0])
        output_tokens = len(self.tokenizer(gen_text, return_tensors='pt')['input_ids'][0])
        return gen_text, input_tokens, output_tokens, latency
    

class Phi3Vision:
    def __init__(self, max_new_tokens: int = 256):
        cache_dir = os.path.join('pretrained_models', 'phi3_vision')
        os.makedirs(cache_dir, exist_ok=True)
        # Declare model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'     # use _attn_implementation='eager' to disable flash attention, recommended is flash_attention_2
        )  
        self.processor = AutoProcessor.from_pretrained(
            'microsoft/Phi-3-vision-128k-instruct', 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        # Generation configs
        self.generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "temperature": 0.2, 
            "do_sample": False
        } 
    
    def __call__(self, prompt: str, image_url: str):
        # message template
        messages = [ 
            {"role": "system", "content": "You are a helpful assistant who is an expert of inventory management."},
            {"role": "user", "content": "<|image_1|>\nUse the image to get useful information to perform the task decribed below."},
            {"role": "user", "content": prompt},
        ] 
        # Download image
        image = download(image_url)
        start = time.time()
        # Prepare inputs
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda") 
        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **self.generation_args
        ) 
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        end = time.time()
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(generate_ids[0])
        latency = end - start
        return response, input_tokens, output_tokens, latency

        
