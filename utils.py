import tiktoken
import os
from datasets import load_from_disk , load_dataset

MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo-1106": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "deepseek-chat": 128000,
    "deepseek-reasoner": 128000
}

def load_data(dataset_name):
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print("Fail to load the dataset: " , e)
    else:
        return dataset

def write_file_GT(file_path: str , data):
    """
        Write problem information in Codeforces dataset to the file 
    """
    if not os.path.exists(file_path):
        with open(file_path , 'w' , encoding='utf-8' , newline="") as f:
            f.write(f"Contest id: {data['contestId']} | Group: {data['index']} | Problem name: {data['name']} | Rating: {data['rating']}\n\n")
            f.write(f"Problem description: \n{data['problem-description']}\n\n")
            f.write(f"Input-specification: \n{data['input-specification']}\n\n")
            f.write(f"Output-specification: \n{data['output-specification']}\n\n")
            f.write(f"Result: {data['verdict']}\n\nCode to infer: \n\n{data['code']}\n")
            f.close()

def write_llm_response(file_path: str , model_name , response):
    """
        write model response which contains the intension inference to file
    """
    if os.path.exists(file_path):
        with open(file_path , 'a+' , encoding='utf-8' , newline="") as f:
            f.write(f"-------------------------------------{model_name}-------------------------------------\n")
            f.write(f"model response: \n{response}\n")
            f.close()
    else:
        with open(file_path , 'w' , encoding='utf-8' , newline="") as f:
            f.write(f"-------------------------------------{model_name}-------------------------------------\n")
            f.write(f"model response: \n{response}\n")
            f.close()

def write_llm_test_input(file_path: str , model_name , response):
    if not os.path.exists(file_path):
        with open(file_path , 'w' , encoding='utf-8' , newline="") as f:
            f.write(response)
            f.close()
    else:
        with open(file_path , 'a+' , encoding='utf-8' , newline="") as f:
            f.write(response)
            f.close()

def write_variant_genertion_essay(file_path: str , model_name , responses_list):
    if os.path.exists(file_path):
        with open(file_path , 'a+' , encoding='utf-8' , newline="") as f:
            f.write(f"-------------------------------------{model_name}-------------------------------------\n")
            f.write(f"model response: \n")
            for i , response in enumerate(responses_list):
                f.write(f"### Solution {i + 1}: \n{response}\n")

            f.close()
    else:
        with open(file_path , 'w' , encoding='utf-8' , newline="") as f:
            f.write(f"-------------------------------------{model_name}-------------------------------------\n")
            f.write(f"model response: \n")
            for i , response in enumerate(responses_list):
                f.write(f"### Solution {i + 1}: \n{response}\n")

            f.close()

def write_dataset_code(file_path: str , data):
    """
        Write the code in dataset to file
    """
    if not os.path.exists(file_path):
        with open(file_path , 'w' , encoding='utf-8' , newline="") as f:
            f.write(f"# result:\t{data['verdict']}\n")
            f.write(data['code'])
            f.close()

def count_tokens(text: str , model: str):
    """
        use tiktoken to calculate the token number of str
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as e:
        print(e)
        # use cl100k_base as encoder
        encoding = tiktoken.encoding_for_model("cl100k_base")
    return len(encoding.encode(str))


def count_message_code(messages: list , model: str):
    """
        Calculate a list of messages token number
    """
    tokens_per_message = 3
    tokens_per_name = 1

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as e:
        print(e)
        encoding = tiktoken.encoding_for_model("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # response consume
    return num_tokens

def check_prompt_fit(messages: list , model: str , reserve_output: int = 1000):
    input_tokens = count_message_code(messages , model)
    max_tokens = MODEL_CONTEXT_LIMITS.get(model , 0)
    
    return {"model": model , "max_tokens": max_tokens , "input_tokens": input_tokens,
            "reserved_for_output": reserve_output , "fits": input_tokens <= max_tokens - reserve_output}

def count_test_inputs(file_path: str):
    if os.path.exists(file_path):
        with open(file_path , 'r' , encoding='utf-8' , newline="") as f:
            content = f.read()
        input_list = content.split("Test input:\n")
        return len(input_list) , input_list
    else:
        return 0 , []
    
def count_and_extract_test_inputs(file_path: str):
    if os.path.exists(file_path):
        with open(file_path , 'r' , encoding='utf-8') as f:
            content = f.read()
        input_list = content.split("Test input:\n")
        test_input_list = []
        for input_content in input_list:
            flag = False
            lines = []
            for line in input_content.split("\n"):
                if line == "":
                    continue

                if line.startswith("```") and flag is False:
                    flag = True
                elif line.startswith("```") and flag is True:
                    break
                elif flag is True:
                    lines.append(line)
            if lines != []:
                test_input_list.append("\n".join(lines))
        
        return len(test_input_list) , test_input_list

    else:
        return 0 , []

def extract_test_from_response(response: str):
    input_list = response.split("Test input:\n")
    test_input_list = []
    for input_content in input_list:
        if input_content == "":
            continue

        flag = False
        lines = []
        for line in input_content.split("\n"):
            if line == "":
                continue

            if line.startswith("```") and flag is False:
                flag = True
            elif line.startswith("```") and flag is True:
                break
            elif flag is True:
                lines.append(line)
        if lines != []:
            test_input_list.append("\n".join(lines))
    
    return test_input_list

