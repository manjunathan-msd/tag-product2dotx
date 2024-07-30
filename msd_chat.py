import os
import json
import yaml
import sys
import traceback
import argparse
import pandas as pd
from pprint import pprint
import time
sys.path.append("/home/ubuntu/efs/users/manjunathan/tag-product2dotx/HPT/")
# sys.path.append('/home/ubuntu/miniconda3/envs/dev/lib/python3.10/site-packages')
from models.hpt_model import HPT
# from template_utils import check_cache
from models.gpt_model import inference_by_chatgpt


#Load Taxonomy
with open('data/Grand Taxonomy.json','r') as file:
    taxonomy=json.load(file)

#Logging the Timings:
import time

# Dictionary to store start times
timing_dict = {}

#Timer Functions
def tic(label="default"):
    timing_dict[label] = time.time()

def toc(label="default"):
    if label in timing_dict:
        elapsed_time = time.time() - timing_dict[label]
        print(f"Elapsed time for {label}: {elapsed_time:.6f} seconds")
    else:
        print(f"No timer found for label: {label}")


# Load configurations
def load_config(override_path=None):
    with open('configs.yaml') as fp:
        configs = yaml.safe_load(fp)
    
    with open(configs['prompt_path_0']) as fp:
        l0_prompt = fp.read()
    with open(configs['prompt_path_1']) as fp:
        l1_prompt = fp.read()

    custom_prompt = None
    if 'custom_prompt_path' in configs:
        print("--Custom Prompt Present - Loading--")
        with open(configs['custom_prompt_path']) as fp:
            custom_prompt = fp.read()


    print("Configurations and prompts are loaded!")
    return configs, l0_prompt, l1_prompt, custom_prompt


#Custom Inference Mode:
def custom_inference(agent, context, image_path, custom_prompt):
    print("Running a Custom Inference")
    try:
        res=agent(context=context, image_path=image_path, prompt=custom_prompt)
        print(res)
    except:
        res={}

    return res


#Inference Mode
def inference(agent, infer_yaml_path=None,use_custom_prompt=False):
    print("------Inferencing On a Subset----------")
    configs, l0_prompt, l1_prompt, custom_prompt = load_config()
    root_folder = infer_yaml_path or configs["data_path"]
    
    processed_count = 0
    total_count = 0
    start = time.time()
    
    for subdir, _, _ in os.walk(root_folder):
        if subdir == root_folder:
            continue
        total_count += 1
        subdir_name = os.path.basename(subdir)
        subdir_num = subdir_name.split('_')[-1]

        output_file = os.path.join(subdir, 'gpt_output_extraction.json' if agent == agent_gpt4 else 'hpt_output_iter0.json')
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                try:
                    if json.load(file):
                        print("Already has the Output. Skipping")
                        continue
                except json.JSONDecodeError:
                    pass

        context, image_path = load_subdir(configs, sub_dir_num=subdir_num, over_ride_path=infer_yaml_path)
        
        try:
            if use_custom_prompt and custom_prompt:
                print("Generating Custom Output")
                output_dict = custom_inference(agent, context, image_path, custom_prompt)
            else:
                output_dict = generate_output(agent=agent, max_depth=100, context=context, image_path=image_path, 
                                          lis_prompt=l0_prompt, tags_prompt=l1_prompt, taxonomy=taxonomy)
        except:
            print(f"--{subdir} Failed to Process--")
            print(f"Stack Trace {traceback.format_exc()}")
            output_dict = {}

        with open(output_file, "w") as file:
            json.dump(output_dict, file, indent=4)

        processed_count += 1
        print(f"--- Total Files Processed: {processed_count}, Total Folders: {total_count} ---")

    finish = time.time()
    print(f"Finished Processing. Total time: {finish-start} seconds")

#ArgParser Definition:
def parse_wilcommen(wilcommen):
    parser = argparse.ArgumentParser(description='Process input commands.')
    parser.add_argument('--subfolder', '--sc', type=int, dest='subfolder', help='Subfolder directory number')
    parser.add_argument('--force', '--f', type=str, choices=['L0', 'L1'], dest='force', help='Force a Specific Prompt level')
    parser.add_argument('--category', '--cat', type=str, dest='category', help='Category to force in L1')
    parser.add_argument('--exit', '--e', action='store_true', dest='exit', help='Exit the program')
    parser.add_argument('--reload', '--r', action='store_true', dest='reload', help='Reload configurations and Prompts')
    parser.add_argument('--infer', '--i', type=str, dest='infer', help='Path to run inference on sub folders')
    parser.add_argument('--agent',type=str, choices=["GPT4", "HPT"], help='Choose the VLM (Vision Language Model) to use:\n'
                             'GPT4: Use GPT-4 model\n'
                             'HPT: Use HPT model')
    parser.add_argument('--custom', action='store_true', help='Use Custom Prompt')
    # parser.add_argument('-h', '--help', action='help',default=argparse.SUPPRESS,help='Show this help message and exit.')

    parser.epilog = """
    Examples:
    - Process subfolder_5 in Custom Data Path:
        --subfolder 5

    - Use Agent GPT4/HPT
        --agent GPT4

    - Force L1 processing for category "Electronics":
        --force L1 --category Electronics
    
    - Run inference on a specific Directory of Sub Folders using GPT4:
        --infer /path/to/data --agent GPT4
    
    - Use custom prompt for inference:
        --infer /path/to/data --custom
    
    - Reload configurations:
        --reload
    
    - Exit the program:
        --exit
        """

    # Split wilcommen into tokens and parse them as if they were command-line arguments
    try:
        args = parser.parse_args(wilcommen.split())
    except argparse.ArgumentError:
        args = None

    return args

#Loading a Sub Directory and Returning Context:
def load_subdir(configs,sub_dir_num,over_ride_path=None):
    context = ""
    image_path = ""
    try:
        sub_dir_name = f"subfolder_{sub_dir_num}"
        print(f"Loading Sub Directory {sub_dir_name}")

        sub_folder_dir = os.path.join(over_ride_path or configs["data_path"], sub_dir_name)

        # Opening data.json
        try:
            data_file = os.path.join(sub_folder_dir, 'data.json')
            print(data_file)
            with open(data_file, "r") as file:
                info_dict = json.load(file)

            # Setting Context
            input_cols = [x.strip() for x in configs['input_cols'].split(',')]
            context = ""
            for col in input_cols:
                context += col + " : " + info_dict.get(col, '') + "\n"
        except:
            context=""

        try:
            image_path = os.path.join(sub_folder_dir, 'image.jpg')
        except:
            image_path=""
    except:
        print(f"Error Loading Sub Folder {sub_dir_num}")
        print(f"Stack Trace:{traceback.format_exc()}")

    # print("Context:",context)
    # time.sleep(20)
    return context,image_path

#Check if In Taxonomy:
def check_taxonomy(depth,val,taxonomy,mode=None,concatenated=None):
    if taxonomy is None:
        raise ValueError("Taxonomy dictionary is required.")
    
    values=taxonomy.get(f"L{depth}",[])
    if concatenated:
        values=values.get(concatenated,[])

    #To check if the Node is final or has more depth:
    if mode=="Search":
        if isinstance(values,dict):
            print("Search Mode Returns Leaf Node")
            return "Leaf Node"
        elif len(values) and isinstance(values,list):
            print("Search Mode Returns Branch")
            return "Branch"
        else:
            print("No More After this, check vals")
            return "No Depth"
    
    if isinstance(values,list):
        if val in values:
            return val
        for key in values:
            if (val.lower() in key.lower()) or (key.lower() in val.lower()):
                return key
    
    return None

#Chainer Prompt:
def generate_output(agent,max_depth:int,context: str, image_path: str,lis_prompt:str,tags_prompt:str,taxonomy:dict):

    if not(max_depth):
        max_depth=100
    
    depth=0
    bread_crumb=""
    output_dict={}
    #List of Values - L0
    tic("L0")
    res=agent(context=context,prompt=lis_prompt,image_path=image_path,tax_vals=",".join(taxonomy["L0"]))
    print(f"Agent Response At Depth {depth} is {res}")
    print("\n\n================\n\n")
    l0=check_taxonomy(depth=0,val=res,taxonomy=taxonomy)
    output_dict.update({
        "L0":l0 if not pd.isna(l0) else "Not Specified"
    })

    toc("L0")

    if pd.isna(l0):
        return output_dict
    
    print("--GOING DEEPER--")
    #Update Depth & BreadCrumb
    depth+=1
    bread_crumb=f"{bread_crumb}{l0}"

    #Values of Anything Beyond Depth 0:
    while check_taxonomy(depth=depth,val=None,taxonomy=taxonomy,mode="Search",concatenated=bread_crumb)=="Branch":
        print("Depth is Presenting Further")
        tic(f"L{depth}")
        res=agent(context=context,prompt=lis_prompt,image_path=image_path,tax_vals=",".join(taxonomy[f"L{depth}"][bread_crumb]))
        print(f"Agent Response At Depth {depth} is {res}")
        print("\n\n================\n\n")
        l_depth=check_taxonomy(depth=depth,val=res,taxonomy=taxonomy,concatenated=bread_crumb)
        toc(f"L{depth}")

        output_dict.update({
        f"L{depth}":l_depth if not pd.isna(l_depth) else "Not Specified"
        })

        #If it is Empty, Our Tagging Stops
        if pd.isna(l_depth):
            return output_dict
        
        #If it is Reached Max Depth: Tagging Stops
        if depth==max_depth:
            return output_dict
        
        depth+=1
        bread_crumb=f"{bread_crumb}>{l_depth}"

    if check_taxonomy(depth=depth,val=None,taxonomy=taxonomy,mode="Search",concatenated=bread_crumb)=="Leaf Node":
        tic(f"L{depth}")
        res=agent(context=context,prompt=tags_prompt,image_path=image_path,tax_vals=taxonomy[f"L{depth}"][bread_crumb])
        print(f"Agent Response At Depth {depth} is {res}")
        print("\n\n================\n\n")
        output_dict.update({
            f"L{depth}":res
        })
        toc(f"L{depth}")
        return output_dict
    
    return output_dict

                
    
# Chat mode
def chat():
    print("Loading Up Configs")
    configs, l0_prompt, l1_prompt, custom_prompt=load_config()
    while True:
        #Welcome Message
        context=None
        image_path=None
        wilcommen = input("Hi. Enter Any Message to Proceed. You can give the Following as CLIs as Well.\n- Enter Exit to exit.\n- Enter Reload to Refresh Prompts.\n- Enter 'Sub Folder/--subfolder' if you want to give a SubFolder Directory\n(type '--help' for instructions, 'exit' to quit)\nEnter Your Message: ")
        args = parse_wilcommen(wilcommen)

        if args is None:
            print("Invalid command format. Please try again.")
            continue

        print("Arguments Given Are: ",args)

        if args.exit:
            print("Exiting The Simulation - Have a Nice Day!")
            break

        if args.reload:
            print("----..Reloading configurations..----")
            configs, l0_prompt, l1_prompt,custom_prompt = load_config()
            continue

        if args.infer:
            inference(agent_gpt4 if args.agent == 'GPT4' else agent, args.infer, use_custom_prompt=args.custom)
            continue
        
        context, image_path = load_subdir(configs, args.subfolder) if args.subfolder is not None else get_custom_input()


        kwargs={}
        if args.force:
            if args.category:
                kwargs[f"force_{args.force}"]=args.category
        print(kwargs)



        try:
            start_time = time.time() 
            if args.custom and custom_prompt:
                res_dict = custom_inference(agent_gpt4 if args.agent == 'GPT4' else agent, context, image_path, custom_prompt)
            else:
                res_dict=generate_output(agent=agent_gpt4 if args.agent == 'GPT4' else agent,max_depth=100,context=context,image_path=image_path,lis_prompt=l0_prompt,tags_prompt=l1_prompt,taxonomy=taxonomy)
            pprint(res_dict,indent=4)
            end_time=time.time()

            print(f"---Total Processing Time Taken : {end_time-start_time}----")
            print("Detailed Processing Times:")
            pprint(timing_dict,indent=4)

        except:
            print("Error Occured While Generating Response")
            print(f"Stack Trace: {traceback.format_exc()}")
        
        print("=" * 50)

def get_custom_input():
    title = input("Enter Product Title: ")
    desc = input("Enter Product Description: ")
    context = f"Title Of the Product:{title}\nDescription of the Product:{desc}"
    image_path = input("Enter the Image Path: ")
    return context, image_path

if __name__ == "__main__":
    loader = int(input("What do you want to load?\n1. GPT\n2. HPT\n3. Both\nEnter your choice: "))
    
    if loader == 1:
        agent_gpt4 = inference_by_chatgpt
        print("Please use --agent GPT4 in CLI.")
    elif loader == 2:
        agent = HPT()
        print("Please use --agent HPT in CLI.")
    else:
        agent = HPT() # In House
        agent_gpt4 = inference_by_chatgpt #GPT API

    choice = int(input("1. Start a Chat with MSD LLMS (HPT/GPT4)\n2. Run An Inference\nEnter your choice: "))
    
    if choice == 1:
        try:
            chat()
        except:
            print(f"Chat failed to load. Stack Trace: {traceback.format_exc()}")
    elif choice == 2:
        try:
            use_custom = input("Use custom prompt? (y/n): ").lower() == 'y'
            agent_choice = int(input("1. HPT\n2. GPT4\nEnter your choice: "))
            inference(agent_gpt4 if agent_choice == 2 else agent, use_custom_prompt=use_custom)
        except:
            print(f"Inference failed to run. Stack Trace: {traceback.format_exc()}")

    



    