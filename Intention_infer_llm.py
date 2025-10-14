import os
from huggingface_hub import hf_hub_download
from openai import OpenAI
from utils import *
import re
import logging

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_URL = 'https://api.openai-proxy.org/v1'
API_KEY = 'sk-YXeIf5Hzq452SluTP77QPGWOeWHq7GFMqH4C4kwr9uFZhbhv'



def infer_intention_llm(code_snippet , model="gpt-3.5-turbo-1106"):
    client = OpenAI(base_url=BASE_URL,
        api_key=API_KEY)
    
#     prompt_template = f"""
# You are a highly skilled software engineer and code analyst.

# I will provide you with a code snippet. Your task is to carefully analyze it and tell me the intention of the program.

# Code:
# {code_snippet}
# """
    prompt_template = f"""
You are an expert competitive programming problem writer.

I will provide you with a solution code. 
Your task is to **reconstruct a plausible problem statement** that this code is likely solving.

If the given code is too incomplete, nonsensical, or incorrect for a reasonable analysis,  
clearly state: "The code is incomplete or invalid; unable to reconstruct a meaningful problem."

Please include in your answer:
1. A clear **problem description** (what the task is about).  
2. The expected **input format**.  
3. The expected **output format**.  

Focus on describing the **task as it would appear in a Codeforces problem statement**, not on explaining the code step by step.

Here is the solution code:

```python
"""

#     prompt_template = f"""
# You are an expert competitive programming problem writer.

# I will provide you with a solution code. 
# Your task is to **reconstruct a plausible problem statement** that this code is likely solving.

# Please include in your answer:
# 1. A clear **problem description** (what the task is about).  
# 2. The expected **input format**.  
# 3. The expected **output format**.  

# Focus on describing the **task as it would appear in a Codeforces problem statement**, not on explaining the code step by step.

# Here is the solution code:

# ```python
# """
    prompt = prompt_template + code_snippet
    messages=[
    {'role': "system" , 'content': "You are an expert software development assistant."},
    {'role': "user" , 'content': prompt}]
    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    print("prompt: \n" , prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )
    return token_info , response.choices[0].message.content

# print(os.getcwd())
# dataset = load_from_disk("./CodeForces2305")
# print(dataset)
# print(dataset[0]['prompt'])

def judge_valid_code(code: str):
    """
        If code contains _RANDOM_GUESS_ , it is a invalid code
    """
    if "_RANDOM_GUESS_" in code:
        return False
    elif 'input' not in code:
        return False
    
    return True


def parse_data(dataset):
    # dataset = load_dataset("MatrixStudio/Codeforces-Python-Submissions")
    # dataset = load_dataset("MatrixStudio/Codeforces-Python-Submissions")
    # print(dataset['train'][0]['code'])

    rating_dict = {'<= 1200' : 0 , '1300 ~ 1500' : 0 , "1600 ~ 1800" : 0 , '1900 ~ 2100' : 0 , "2200+" : 0}
    verdict = {}
    buggy_middle_and_high_level_idx = []

    max_rating = 0
    min_rating = float('inf')

    max_index = 0
    min_index = 0

    mean_rating = 0.0

    for idx , data in enumerate(dataset.select(range(100))):
        # invalid data
        if data['type'] == 'none':
            continue
        elif not judge_valid_code(data['code']):
            continue

        if data['rating'] > max_rating:
            max_rating = data['rating']
            max_index = idx

        if data['rating'] < min_rating:
            min_rating = data['rating']
            min_index = idx


        if data['rating'] <= 1200:
            rating_dict['<= 1200'] += 1
        elif 1300 <= data['rating'] <= 1500:
            rating_dict['1300 ~ 1500'] += 1
        elif 1600 <= data['rating'] <= 1800:
            rating_dict['1600 ~ 1800'] += 1
        elif 1900 <= data['rating'] <= 2100:
            rating_dict['1900 ~ 2100'] += 1
        elif data['rating'] >= 2200:
            rating_dict['2200+'] += 1
        
        if data['verdict'] in verdict.keys():
            verdict[data['verdict']] += 1
        else:
            verdict[data['verdict']] = 1

        if data['rating'] >= 1600 and data['verdict'] != 'OK':
            buggy_middle_and_high_level_idx.append(idx)

    print(f"max_rating: {max_rating:6d}\tmax_index: {max_index}")
    print(f"min_rating: {min_rating:6d}\tmin_index: {min_index}")
    print(f"result: {rating_dict}")
    print(f"verdict: {verdict}")
    print(f"buggy index: {buggy_middle_and_high_level_idx}")

    result_dict = {'max_rating' : max_rating , 'max_index' : max_index , 'min_rating' : min_rating , 'min_index': min_index , 'rating_dict': rating_dict ,
                   'verdict': verdict , 'buggy_middle_and_high_level_idx': buggy_middle_and_high_level_idx}

    return result_dict , buggy_middle_and_high_level_idx


def write_the_parse_result(result_dict: dict):
    with open('./dataset_result.txt' , 'w' , encoding='utf-8') as f:
        f.write(f"max_rating: {result_dict['max_rating']:6d}\tmax_index: {result_dict['max_index']}\n")
        f.write(f"min_rating: {result_dict['min_rating']:6d}\tmin_index: {result_dict['min_index']}\n")
        f.write(f"result: {result_dict['rating_dict']}\n")
        f.write(f"verdict: {result_dict['verdict']}\n")
        f.write(f"buggy index: {result_dict['buggy_middle_and_high_level_idx']}\n")



def generate_intention(dataset , buggy_middle_and_high_level_idx: list , model: str):
        for idx in buggy_middle_and_high_level_idx:
            print(f"---------------start to parse {idx}---------------")
            data = dataset[idx]
            write_file_GT(f"./Infer intention/{idx}.txt" , data)
            code_snippet = data['code']
            write_dataset_code(f'./Code in dataset/{idx}.py' , data)
            token_info , response = infer_intention_llm(code_snippet)
            logger = logging.getLogger(__name__)
            logger.info(f"The length of './Infer intention/{idx}.txt' prompt: {token_info['input_tokens']} / {token_info['max_tokens']}     model: {model}")
            write_llm_response(f"./Infer intention/{idx}.txt" , model , response)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./app.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    dataset = load_data("MatrixStudio/Codeforces-Python-Submissions")
    train_dataset = dataset["train"]
    test_dataset = dataset['test']
    result_dict , buggy_middle_and_high_level_idx = parse_data(train_dataset)
    # write_the_parse_result(result_dict)
    # generate_intention(train_dataset , buggy_middle_and_high_level_idx , "gpt-3.5-turbo-1106")
    generate_intention(train_dataset , buggy_middle_and_high_level_idx , "gpt-4o-mini")