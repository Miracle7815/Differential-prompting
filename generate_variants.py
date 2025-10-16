from utils import *
import os
from openai import OpenAI
import re
import logging
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
BASE_URL = 'https://api.openai-proxy.org/v1'
API_KEY = 'sk-YXeIf5Hzq452SluTP77QPGWOeWHq7GFMqH4C4kwr9uFZhbhv'

def generate_variants_by_llm(problem_info: str , model: str , temperature=0.3):
    client = OpenAI(base_url=BASE_URL , api_key=API_KEY)

    prompt_template = f"""You are a highly skilled competitive programming expert.

Here is the problem statement and specification:

"""
    prompt_template += problem_info
    prompt_task = f"""

---

Your task is:
1. Based on the problem description and specification, generate **two different correct Python solutions**.
2. Each solution must strictly follow the input and output format.
3. Label them clearly as "Solution 1" and "Solution 2".

Output format:

### Solution 1
```python
# code here

### Solution 2
```python
# code here
"""
    prompt = prompt_template + prompt_task
    
    messages=[{'role': "system" , 'content': "You are an expert coding assistant."},
    {'role': "user" , 'content': prompt}]

    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    # print(prompt)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    print(response.choices[0].message.content)

    return token_info , response.choices[0].message.content

def generate_variants_essay_mode(problem_info , model: str , k=2 , temperature=0.3):
    client = OpenAI(base_url=BASE_URL , api_key=API_KEY)
    prompt_template = f"""You are a highly skilled competitive programming expert.

Here is the problem statement and specification:

"""
    prompt_template += problem_info
    prompt_task = f"""

---

Your task is: Based on the problem description and specification, generate a correct Python solution following the input and output format. Only generate the program, do not generate any explain.
"""
    prompt = prompt_template + prompt_task
    messages=[{'role': "system" , 'content': "You are an expert coding assistant."},
    {'role': "user" , 'content': prompt}]

    model_responses_list = []
    token_info_list = []
    
    for i in range(k):
        token_info = check_prompt_fit(messages , model)
        if token_info['fits'] is False:
            token_info_list.append(token_info)
            model_responses_list.append("The length of context is out of bound !!!")
            continue
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        token_info_list.append(token_info)
        model_responses_list.append(response.choices[0].message.content)

    return token_info_list , model_responses_list

def generate_variants_trickbugs(client: OpenAI , problem_description , put_code , model: str , k=2 , temperature=0.3):
    system_prompt = """You are a professional coding competition participant, skilled at identifying bugs and logic flaws in code.
You will receive a description of a coding problem, and a piece of code attempting to solve the problem.
Your task is to find whether there is any bug or logic flaw in the code, if any, please repair code.
Please reply with ONLY the COMPLETE REPAIRED CODE (rather than code fragments) and DO NOT reply any other content.
"""
    user_prompt = """**PROBLEM DESCRIPTION**:
"""
    user_prompt += problem_description + "\n"
    user_prompt += """
**CODE**:
"""
    user_prompt += put_code
    # print(user_prompt)
    
    messages = [
        {'role': "system" , "content": system_prompt},
        {'role': "user" , "content": user_prompt}
    ]
    
    model_responses_list = []
    token_info_list = []

    for idx in range(k):
        token_info = check_prompt_fit(messages , model)
        if token_info['fits'] is False:
            token_info_list.append(token_info)
            model_responses_list.append("The length of context is out of bound !!!")
            continue
            
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature
        )

        token_info_list.append(token_info)
        model_responses_list.append(response.choices[0].message.content)
        time.sleep(1)

    return token_info_list , model_responses_list

def parse_and_generate_variants(model: str , mode , temperature=0.3):
    folder_path = './Infer intention'
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path , file)

        with open(file_path , "r" , encoding='utf-8') as f:
            lines = f.readlines()
        
        content_lines = []
        flag = False

        for line in lines:
            if f'-------------------------------------{model}-------------------------------------' in line and flag is False:
                flag = True
            elif line.startswith("-------------------------------------") and flag is True:
                break
            elif flag is True:
                if 'model response:' in line:
                    continue
                content_lines.append(line)
            else:
                continue
        
        content = "".join(content_lines)
        # print(f"-----------------{file}---------------")  
        # print(content)

        if "The code is incomplete or invalid" and "unable to reconstruct a meaningful problem" in content:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

        if mode == 0:
            token_info , response = generate_variants_by_llm(content , model , temperature=temperature)
            logger = logging.getLogger(__name__)
            logger.info(f"The length of '{file_path}' prompt: {token_info['input_tokens']} / {token_info['max_tokens']}     model: {model}")

            write_llm_response(f'./Variants generation/my/{file.split(".")[0]}.txt' , model , response)  
        elif mode == 1:
            token_info_list , response_list = generate_variants_essay_mode(content , model , temperature=temperature)
            
            token_info = token_info_list[0]
            logger = logging.getLogger(__name__)
            logger.info(f"The length of '{file_path}' prompt: {token_info['input_tokens']} / {token_info['max_tokens']}     model: {model}")
            
            write_variant_genertion_essay(f'./Variants generation/essay/{file.split(".")[0]}.txt' , model , response_list)
            
            for i , response in enumerate(response_list):
                start = response.find("```python") + len("```python\n")
                end = response.rfind("```")
                code = response[start:end].strip()

                if not os.path.exists(f"./Variants/essay/{file.split('.')[0]}"):
                    os.mkdir(f"./Variants/essay/{file.split('.')[0]}")

                with open(f"./Variants/essay/{file.split('.')[0]}/{model}_Solution_{i + 1}.py" , 'w' , encoding='utf-8') as f:
                    f.write(code)
                    f.close()

def parse_and_generate_variants_for_TrickyBugs(client , model: str , k=2 , temperature=0.3):
    dataset_path = "./Datasets/TrickyBugs"
    dataset_code_path = os.path.join(dataset_path , "PUT_python")
    description_path = os.path.join(dataset_path , "problem_descriptions")

    for dir in os.listdir(dataset_code_path):
        problem_description = None
        put_code = None

        with open(os.path.join(description_path , dir , "problem_description.txt") , 'r' , encoding='utf-8') as f:
            problem_description = f.read()
            f.close()
        
        for file in os.listdir(os.path.join(dataset_code_path , dir)):
            print(file)
            with open(os.path.join(dataset_code_path , dir , file) , 'r' , encoding='utf-8') as f:
                put_code = f.read()
                f.close()

            # generate_variants_trickbugs(client , problem_description , put_code , model)
            token_info_list , response_list = generate_variants_trickbugs(client , problem_description , put_code , model , k=k , temperature=temperature)
            for idx , response in enumerate(response_list):
                dir_name = f"./TrickyBugs/{model}/GenProgs/tc_generated_progs_python/{dir}"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(os.path.join(dir_name , file.split(".")[0] + "_num_" + str(idx)) , "w" , encoding='utf-8') as resp_file:
                    resp_file.write(response)

def transform_code(model: str):
    model_generate_folder = './Variants generation/my'
    for file in os.listdir(model_generate_folder):
        file_path = os.path.join(model_generate_folder , file)

        with open(file_path , 'r' , encoding='utf-8') as f:
            lines = f.readlines()
        
        solutions = []
        flag = False
        
        start_line = 0
        end_line = 0

        for line_num , line in enumerate(lines):
            if f"-------------------------------------{model}-------------------------------------" in line and flag is False:
                flag = True
                start_line = line_num
            elif line.startswith("-------------------------------------") and flag is True:
                end_line = line_num
                break
            elif line_num == len(lines) - 1 and flag is True:
                end_line = line_num
            else:
                continue
        
        
        if not (start_line == 0 and end_line == 0):
            solution_1_list = []
            current_line = start_line + 1

            while '### Solution 1' not in lines[current_line] and current_line < end_line:
                current_line += 1

            if '### Solution 1' in lines[current_line]:
                current_line += 1
                start_flag = False
                while start_flag is False and current_line < end_line:
                    if '```python' in lines[current_line]:
                        start_flag = True
                        current_line += 1
                    else:
                        current_line += 1
            
                if current_line < end_line:
                    while '```' not in lines[current_line] and current_line < end_line:
                        solution_1_list.append(lines[current_line])
                        current_line += 1
            
            solution_1_code = ''.join(solution_1_list)
            solutions.append(solution_1_code)

            while '### Solution 2' not in lines[current_line] and current_line < end_line:
                current_line += 1
            
            solution_2_list = []
            if "### Solution 2" in lines[current_line]:
                current_line += 1
                start_flag = False
                while start_flag is False and current_line < end_line:
                    if '```python' in lines[current_line]:
                        start_flag = True
                        current_line += 1
                    else:
                        current_line += 1

                if current_line < end_line:
                    while '```' not in lines[current_line] and current_line < end_line:
                        solution_2_list.append(lines[current_line])
                        current_line += 1
            
            solution_2_code = ''.join(solution_2_list)
            solutions.append(solution_2_code)

            if len(solutions) == 2:
                for idx , code in enumerate(solutions):
                    if not os.path.exists(f'./Variants/my/{file.split(".")[0]}'):
                        os.mkdir(f'./Variants/my/{file.split(".")[0]}')
                    with open(f'./Variants/my/{file.split(".")[0]}/{model}_Solution_{idx + 1}.py' , 'w' , encoding='utf-8') as code_file:
                        code_file.write(code)
        else:
            if not os.path.exists(f'./Variants/my/{file.split(".")[0]}'):
                os.mkdir(f'./Variants/my/{file.split(".")[0]}')
            
            logger = logging.getLogger(__name__)
            logger.info(f"The \'./Variants generation/{file}\' format is false !!!")

def transform_code_for_TrickyBugs(model: str):
    response_dir = f"./TrickyBugs/{model}/GenProgs"
    for dir in os.listdir(os.path.join(response_dir , "tc_generated_progs_python")):
        code_dir = os.path.join(response_dir , "tc_generate_code_python_extracted" , dir)
        for file in os.listdir(os.path.join(response_dir ,"tc_generated_progs_python" , dir)):
            with open(os.path.join(response_dir , "tc_generated_progs_python" , dir , file) , 'r' , encoding='utf-8') as f:
                content = f.read()
            if "```python" in content:
                content_list = content.split('\n')
                flag = False
                code_line = []
                for line in content_list:
                    if "```python" in line and flag is False:
                        flag = True
                    elif flag is True and "```" in line:
                        break
                    else:
                        code_line.append(line)

                code = "\n".join(code_line)
            else:
                code = content
            
            if not os.path.exists(os.path.join(code_dir)):
                os.makedirs(os.path.join(code_dir))
            with open(os.path.join(code_dir , file + ".py") , 'w' , encoding='utf-8') as f:
                f.write(code)


if __name__ == '__main__':
    # logging.basicConfig(
    # filename="./app.log",
    # level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )

    # modes = {'my' : 0 , 'essay': 1}

    # for _ , mode in modes.items():
    #     if mode == 0:
    #         # parse_and_generate_variants(model="gpt-3.5-turbo-1106" , mode=mode)
    #         # transform_code(model="gpt-3.5-turbo-1106")

    #         parse_and_generate_variants(model="gpt-4o-mini" , mode=mode)
    #         transform_code(model="gpt-4o-mini")
    #     elif mode == 1:
    #         # parse_and_generate_variants(model="gpt-3.5-turbo-1106" , mode=mode)

    #         parse_and_generate_variants(model="gpt-4o-mini" , mode=mode)

    client = OpenAI(base_url=BASE_URL , api_key=API_KEY)

    parse_and_generate_variants_for_TrickyBugs(client , model="gpt-4o-mini" , k=6 , temperature=0.8)
    transform_code_for_TrickyBugs("gpt-4o-mini")

    # parse_and_generate_variants_for_TrickyBugs(client , model="gpt-3.5-turbo-1106")
    # transform_code_for_TrickyBugs("gpt-3.5-turbo-1106")


    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model , tokenizer = load_model(model_name)
    # parse_and_generate_variants_for_TrickyBugs_by_model(model , tokenizer , model_name)

    # transform_code_for_TrickyBugs(model_name)