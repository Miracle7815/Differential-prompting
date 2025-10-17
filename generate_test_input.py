import os
from openai import OpenAI
import logging
from utils import *
import subprocess

BASE_URL = 'https://api.openai-proxy.org/v1'
API_KEY = 'sk-YXeIf5Hzq452SluTP77QPGWOeWHq7GFMqH4C4kwr9uFZhbhv'

def generate_input_only_use_PUT(client , PUT_code , model: str):
    prompt_template = f"""You are a test data generator for competitive programming problems.
    
I will give you a reference solution code (Program Under Test).

Your task:
- Generate a set of valid **test inputs** that follow the specification.
- The test inputs should be diverse:
  * Small trivial cases
  * Edge cases (minimum and maximum limits)
  * Random non-trivial cases
- Ensure inputs strictly follow the input format.

Important:
- **Do not explain or add comments.**
- **Do not produce outputs.**
- **Return only raw test inputs.**
- **Please put 'Test input': as a separate line in front of each test input**

Program under Test:
```python
"""

    prompt = prompt_template + PUT_code + "\n```" 
    messages = [{'role': "system" , "content": "You are a professional test case generator for competitive programming. You are very strict about following the input specification exactly andproducing only valid raw test inputs with no explanations."},
                {'role': "user" , "content" : prompt}]
    
    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.3
    )

    return token_info , response.choices[0].message.content

def generate_input_use_specification(client , specification , PUT_code , model: str):
    client = OpenAI(base_url=BASE_URL , api_key=API_KEY)
    prompt_template = """You are a test data generator for competitive programming problems.

I will give you:
1. A problem description and input/output specification.
2. A reference solution code (Program Under Test).

Your task:
- Generate a set of valid **test inputs** that follow the specification.
- The test inputs should be diverse:
  * Small trivial cases
  * Edge cases (minimum and maximum limits)
  * Random non-trivial cases
- Ensure inputs strictly follow the input format.

Important:
- **Do not explain or add comments.**
- **Do not produce outputs.**
- **Return only raw test inputs.**
- **Please put 'Test input': as a separate line in front of each test input**

Problem and specification:
"""

    prompt_template += specification + "\n"
    prompt_template += """
Program under test:
```python
"""
    prompt = prompt_template + PUT_code + "\n```"

    print(prompt)
    messages = [{'role': "system" , "content": "You are a professional test case generator for competitive programming. You are very strict about following the input specification exactly andproducing only valid raw test inputs with no explanations."},
                {'role': "user" , "content" : prompt}]
    
    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.3
    )

    return token_info , response.choices[0].message.content

def generate_input_for_TrickyBugs(client , specification , pid , model:str):
    sys_prompt_template = """**INSTRUCTION**:
The following is a description of a coding problem, please write an input generator for this problem (DO NOT generate outputs), use the generator to generate 100 inputs, write the i-th input into the single file './{pid}_chatGenInput_i.in' for every input.
The inputs should meet the given constraints of the problem description.
Please reply with ONLY the code without any other content.

You can use the python library cyaron if necessary, here are some examples of how to use cyaron, which may be helpful:
```python
# Use IO() to write generated inputs into files
# example:
from cyaron import *
for i in range(100):
    io=IO(file_prefix=f'./{pid}_chatGenInput_', data_id=i, disable_output=True) # write the generated contents into ./{pid}_chatGenInput_i.in
    io.input_write(4, 5, 6) # write 4 5 6 into the file chatGenInput_i.in
    io.input_writeln(4, 5, 6) # write 4 5 6 and and add a newline into the file chatGenInput_i.in
    io.input_writeln([1, 2, 3]) # write 1 2 3 and and add a newline into the file chatGenInput_i.in

# Use Vector.random(num=5, position_range=[10], mode=0) to generate vectors
# Parameter 'num': The number of vectors to generate.
# Parameter 'position_range': A list containing elements that determine the dimensions of the output vectors. Each element can be a two-dimensional tuple of integers (or real numbers) (min, max), representing the range of each dimension as [min, max]. Alternatively, it can be a single integer (or real number) 'k', indicating the range of each dimension is [0, k]. If this parameter has only one element, it generates a sequence of numbers rather than a vector.
# Parameter 'mode': Mode selection. 0 for generating non-repeating integer vectors, 1 for allowing repeating integer vectors (each dimension is independently randomized), 2 for real number vectors.
# examples:
from cyaron import *
vector_array = Vector.random(10, [(10,50)]) # Generate 10 sequences of non-repeating numbers within the range [10, 50].
vector_matrix = Vector.random(30, [(10,50), 20]) # Generate 30 non-repeating two-dimensional vectors, where the first dimension ranges from 10 to 50, and the second dimension ranges from 0 to 20.
vector_float = Vector.random(30, [(1,10), (1,10), (1,10)], 2) # Generate 30 three-dimensional real number vectors, with each dimension ranging from 1 to 10.

# Use String.random() to generate random strings
# examples:
from cyaron import *
ALPHABET_SMALL = string.ascii_lowercase
ALPHABET_CAPITAL = string.ascii_uppercase
ALPHABET = ALPHABET_SMALL + ALPHABET_CAPITAL
NUMBERS = string.digits
str = String.random(5) # Generate a random string with 5 letters consisting of lowercase letters
str = String.random(10, charset=ALPHABET_SMALL) # Generate a random string with 10 letters consisting of lowercase letters
str = String.random(10, charset=ALPHABET_CAPITAL) # Generate a random string with 10 letters consisting of uppercase letters
str = String.random(10, charset='#.') # Generate a random string with 10 letters consisting of '#' and '.'

# Use Graph.graph() to generate graphs
# examples:
from cyaron import *
graph = Graph.graph(n, m, self_loop=False, repeated_edges=False,weight_limit=(5, 300)) # Generate an undirected graph with n vertices and m edges, avoiding duplicate edges and self-loops, the edge weights range from 5 to 300.
tree = Graph.tree(n,weight_limit=(1, 10)) # Generate a tree with n vertices, the edge weights range from 1 to 10.
DAG = Graph.DAG(n, m) # Generate a DAG (Directed Acyclic Graph) with n vertices and m edges.
UDAG = Graph.UDAG(n, m) # Generate an undirected connected graph with n vertices and m edges.
io=IO(file_prefix=f'./{pid}_chatGenInput_', data_id=1, disable_output=True) # chatGenInput_1.in
io.input_writeln(graph) # Output the graph in the format of lines 'u v w' for each edge, where u and v are the vertices of the edge, and w is the weight of the edge.
io.input_writeln(tree.to_str(output=Edge.unweighted_edge)) # Output the unweighted graph in the format of lines 'u v' for each edge
```
"""
    sys_prompt = sys_prompt_template.format(model=model , pid=pid)
    user_prompt = """**PROBLEM DESCRIPTION**:
"""
    user_prompt += specification

    messages = [
        {'role': 'system' , 'content': sys_prompt},
        {'role': 'user' , 'content': user_prompt},
    ]

    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )

    return token_info , response.choices[0].message.content

def generate_input_for_EvalPlus(client , put_code , model: str):
    system_prompt = """**INSTRCUTION**:
You are a professional software testing engineer. 
You will get a Program Under Test (PUT), and the comments below the function signature is the docstring that specifies the requirement of the PUT.

Please write an input generator in Python for generating test input for THE PUT (DO NOT generate outputs).
The generator should be a single Python function named 'sample_one()' and return a list of input parameters for PUT.
Each parameter of the generated inputs must adhere to the type and format according to the function signature of PUT. And the format of return value should be list of parameters and should NOT be dict.

You can use the python library random to generate random float and int, for example:
from random import *
# Generate a random integer between 0 and 9
random_number = randint(0, 9)
# Generate a random floating-point number between -1.0 and 1.0
random_float = uniform(-1.0, 1.0)

You can use the python library cyaron to generate random string, for example:
from cyaron import *
ALPHABET_SMALL = string.ascii_lowercase
ALPHABET_CAPITAL = string.ascii_uppercase
ALPHABET = ALPHABET_SMALL + ALPHABET_CAPITAL
NUMBERS = string.digits
str = String.random(5) # Generate a random string with 5 letters consisting of lowercase letters
str = String.random(10, charset=ALPHABET_SMALL) # Generate a random string with 10 letters consisting of lowercase letters
str = String.random(10, charset=ALPHABET_CAPITAL) # Generate a random string with 10 letters consisting of uppercase letters
str = String.random(10, charset=NUMBERS) # Generate a random string with 10 letters consisting of digits letters
str = String.random(10, charset='#.') # Generate a random string with 10 letters consisting of '#' and '.'

Please reply with ONLY the code of the input generator without any other content.

**EXAMPLE**:
Here is an example of how to write input generator:

If the given function and docstring is like:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                return True
    return False
```

Then your reply should be like the following content:
```python
from random import *
from cyaron import *
def sample_one():
    numbers=[]
    length_of_numbers=randint(0,10)
    for i in range(length_of_numbers):
      numbers.append(uniform(-5,5))
    threshold=uniform(-10,10)
    generated_input=[numbers,threshold]
    return generated_input
```
"""
    user_prompt = """**Function and Docstring**:
"""
    user_prompt += put_code

    messages = [
        {'role': 'system' , 'content' : system_prompt} , 
        {'role': 'user' , 'content': user_prompt}
    ]

    token_info = check_prompt_fit(messages , model)
    if token_info['fits'] is False:
        return token_info , "The length of context is out of bound !!!"
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )

    return token_info , response.choices[0].message.content

def parse_and_generate_test(client , model:str , mode):
    PUT_folder_path = "./Code in dataset"
    specification_folder_path = "./Infer intention"
    for file in os.listdir(specification_folder_path):
        specification_path = os.path.join(specification_folder_path , file)

        # print(PUT_file_path)
        # print(specification_path)
        
        with open(specification_path , 'r' , encoding="utf-8") as f:
            content = f.readlines()
            f.close()
        
        specification_list = []
        flag = False

        for line in content:
            if f'-------------------------------------{model}-------------------------------------' in line and flag is False:
                flag = True
            elif line.startswith('-------------------------------------') and flag is True:
                break
            elif flag is True:
                if 'model response:' in line:
                    continue
                specification_list.append(line)
            else:
                continue
                
        specification = ''.join(specification_list)

        if "The code is incomplete or invalid" in specification or "unable to reconstruct a meaningful problem." in specification:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

        print(f"-------------{file}------------")
        # print(content)
        # print(specification)

        problem_id = file.split('.')[0]
        PUT_file_path = os.path.join(PUT_folder_path , problem_id + ".py")
        
        with open(PUT_file_path , 'r' , encoding="utf-8") as f:
            PUT_code = f.read()
            f.close()
        # print(PUT_code)
        
        num_test_inputs = 0
        test_inputs_list = []
        try_num = 0

        if mode == 0:
            if os.path.exists(os.path.join("./Test input/my" , problem_id , problem_id + "_" + model + ".txt")):
                num_test_inputs , test_inputs_list = count_and_extract_test_inputs(os.path.join("./Test input/my" , problem_id , problem_id + "_" + model + ".txt"))
        elif mode == 1:
            if os.path.exists(os.path.join("./Test input/essay" , problem_id , problem_id + "_" + model + ".txt")):
                num_test_inputs , test_inputs_list = count_and_extract_test_inputs(os.path.join("./Test input/my" , problem_id , problem_id + "_" + model + ".txt"))

        invalid_try = 0

        while num_test_inputs < 10 and try_num < 5 and invalid_try < 2:
            if mode == 0: # my
                token_info , response = generate_input_use_specification(client , specification , PUT_code , model)

                temp_input_list = extract_test_from_response(response)
                valid_test_list = []
                for test_input in temp_input_list:
                    if test_input not in test_inputs_list:
                        test_inputs_list.append(test_input)
                        num_test_inputs += 1
                        valid_test_list.append(test_input)
                
                valid_response = ""
                if len(valid_test_list) != 0:
                    for test_input in valid_test_list:
                        valid_response += f"Test input:\n```\n{test_input}\n```\n"

                    file_dir = f"./Test input/my/{problem_id}"
                    file_path = os.path.join(file_dir , problem_id + "_" + model + ".txt")
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir , exist_ok=True)

                    write_llm_test_input(file_path , model , valid_response)

                    logger = logging.getLogger(__name__)
                    logger.info(f"The length of 'Test input/my/{problem_id}/{problem_id}_{model}.txt' prompt: {token_info['input_tokens']} / {token_info['max_tokens']}     model: {model}")
                else:
                    invalid_try += 1

                try_num += 1
            
            elif mode == 1:
                token_info , response = generate_input_only_use_PUT(client , PUT_code , model)

                temp_input_list = extract_test_from_response(response)
                valid_test_list = []
                for test_input in temp_input_list:
                    if test_input not in test_inputs_list:
                        test_inputs_list.append(test_input)
                        num_test_inputs += 1
                        valid_test_list.append(test_input)
                
                valid_response = ""
                if len(valid_test_list) != 0:
                    for test_input in valid_test_list:
                        valid_response += f"Test input:\n```\n{test_input}\n```\n"

                    file_dir = f"./Test input/essay/{problem_id}"
                    file_path = os.path.join(file_dir , problem_id + "_" + model + ".txt")
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir , exist_ok=True)

                    write_llm_test_input(file_path , model , valid_response)

                    logger = logging.getLogger(__name__)
                    logger.info(f"The length of 'Test input/my/{problem_id}/{problem_id}_{model}.txt' prompt: {token_info['input_tokens']} / {token_info['max_tokens']}     model: {model}")
                else:
                    invalid_try += 1

                try_num += 1

def parse_and_generate_test_for_TrickyBugs(client , model: str):
    pid = os.getpid()
    dataset_code_dir = "./Datasets/TrickyBugs/PUT_python"
    specification_dir = "./Datasets/TrickyBugs/problem_descriptions"

    for dir in os.listdir(dataset_code_dir):
        file_name = os.listdir(os.path.join(dataset_code_dir , dir))[0].split(".")[0]
        print(file_name)

        with open(os.path.join(specification_dir , dir , "problem_description.txt") , 'r' , encoding="utf-8") as f:
            specification = f.read()
            f.close()

        assert specification is not None
        token_info , response = generate_input_for_TrickyBugs(client , specification , pid, model)

        test_generator_dir = f"./TrickyBugs/{model}/GenInputs/tc_generator_python"

        result_path = os.path.join(test_generator_dir , dir)
        os.makedirs(result_path , exist_ok=True)

        test_input_dir = os.path.join(f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator" , dir)

        if not os.path.exists(test_input_dir):
            os.makedirs(test_input_dir , exist_ok=True)
        
        with open(os.path.join(result_path ,  file_name + "_test_generator") , 'w' , encoding='utf-8') as f:
            f.write(response)
            f.close()

def parse_and_generate_for_EvalPlus(client , model: str):
    dataset_code_dir = "./Datasets/EvalPlus/PUTs"
    
    for dir in os.listdir(dataset_code_dir):
        print(dir)

        with open(os.path.join(dataset_code_dir , dir , "put0.py") , 'r' , encoding="utf-8") as f:
            put_code = f.read()
            f.close()

        token_info , response = generate_input_for_TrickyBugs(client , put_code, model)

        test_generator_dir = f"./TrickyBugs/{model}/GenInputs/tc_generator_python"

        result_path = os.path.join(test_generator_dir , dir)
        os.makedirs(result_path , exist_ok=True)

        test_input_dir = os.path.join(f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator" , dir)

        # if not os.path.exists(test_input_dir):
        #     os.makedirs(test_input_dir , exist_ok=True)
        
        with open(os.path.join(result_path ,  dir + "_test_generator") , 'w' , encoding='utf-8') as f:
            f.write(response)
            f.close()
        

def extract_test_input(model: str , mode):
    if mode == 0:
        input_folder = "./Test input/my"
        for dir in os.listdir(input_folder):
            problem_id = dir
            for file in os.listdir(os.path.join(input_folder , dir)):
                if model in file:
                    file_path = os.path.join(input_folder , dir , file)
                    print(file_path)
                    input_num , input_list = count_and_extract_test_inputs(file_path)
                    print(f"{problem_id}: {input_num}")
                    print(input_list)
                    for input_test in input_list:
                        print(input_test)
                        print("-------------------------")
    elif mode == 1:
        input_folder = "./Test input/essay"
        for dir in os.listdir(input_folder):
            problem_id = dir
            for file in os.listdir(os.path.join(input_folder , dir)):
                if model in file:
                    file_path = os.path.join(input_folder , dir , file)
                    print(file_path)
                    input_num , input_list = count_and_extract_test_inputs(file_path)
                    print(f"{problem_id}: {input_num}")
                    print(input_list)
                    for input_test in input_list:
                        print(input_test)
                        print("-------------------------")

def extract_test_generator(model: str):
    input_folder = f"./TrickyBugs/{model}/GenInputs/tc_generator_python"
    for dir in os.listdir(input_folder):
        for file in os.listdir(os.path.join(input_folder , dir)):
            with open(os.path.join(input_folder , dir , file) , 'r' , encoding="utf-8") as f:
                content = f.read()
                f.close()
            
            if "```python" in content or "```" in content:
                lines = content.split("\n")
                flag = False
                code_line = []

                for line in lines:
                    if "```python" in line and flag is False:
                        flag = True
                    elif "```" in line and flag is True:
                        break
                    elif flag is True:
                        code_line.append(line)
                
                code = "\n".join(code_line)
            else:
                code = content

            if not "import random" in code:
                code = "import random\n" + code
            
            extract_path = f"./TrickyBugs/{model}/GenInputs/tc_generator_python_extracted"
            os.makedirs(os.path.join(extract_path , dir) , exist_ok=True)

            with open(os.path.join(extract_path , dir , file + ".py") , 'w' , encoding="utf-8") as f:
                f.write(code)
                f.close()

def execute_input_generator(model: str):
    generator_dir = f"./TrickyBugs/{model}/GenInputs/tc_generator_python_extracted"
    for dir in os.listdir(generator_dir)[:1]:
        for python_file in os.listdir(os.path.join(generator_dir , dir)):
            # with open(os.path.join(generator_dir , dir , python_file) , 'r' , encoding='utf-8') as f:
            #     code = f.read()
            #     f.close()
            if python_file.endswith(".py"):
                python_file_path = os.path.join(".." , '..' , 'tc_generator_python_extracted' , dir , python_file)
                try:
                    exe_result = subprocess.run(["python" , python_file_path] , cwd=os.path.join(f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator" , dir) , capture_output=True , text=True , timeout=5)
                    if exe_result.returncode != 0:
                        print(f'Error in {os.path.join(generator_dir , dir , python_file)}: {exe_result.stderr}')
                        continue
                except:
                    print(f'Subprocess called error')
                    continue


if __name__ == "__main__":
    logging.basicConfig(
        filename="./app.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    client = OpenAI(base_url=BASE_URL , api_key=API_KEY)

    # dataset_name = "TrickyBugs"
    dataset_name = 'EvalPlus'

    model_name = "gpt-4o-mini"
    # model_name = "gpt-3.5-turbo-1106"

    # modes = {'my' : 0 , 'essay': 1}

    # for _ , mode in modes.items():
    #     if mode == 0:
    #         continue
    #     parse_and_generate_test(client , "gpt-3.5-turbo-1106" , mode=mode)
    #     extract_test_input("gpt-3.5-turbo-1106" , mode=mode)

    # parse_and_generate_test_for_TrickyBugs(client , "gpt-3.5-turbo-1106")
    # extract_test_generator("gpt-3.5-turbo-1106")

    # parse_and_generate_test_for_TrickyBugs(client , "gpt-3.5-turbo-1106")
    # extract_test_generator("gpt-3.5-turbo-1106")
    # execute_input_generator("gpt-3.5-turbo-1106")
    
    if dataset_name == "TrickyBugs":
        parse_and_generate_test_for_TrickyBugs(client , model_name)
        extract_test_generator(model_name)

        execute_input_generator(model_name)

    elif dataset_name == "EvalPlus":
        parse_and_generate_for_EvalPlus(client , model_name)
        