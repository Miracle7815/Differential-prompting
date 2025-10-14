from utils import *
import os
from openai import OpenAI
import re
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import torch
import coverage
from vllm import LLM, SamplingParams

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,8,9,10,11,12,13,14"
os.environ["CUDA_VISIBLE_DEVICES"] = "0 , 1 , 2 , 3 , 4 , 5 , 6"


def load_model(model_name):
    model_name = f"../{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    return model, tokenizer


def load_model_vllm(model_name):
    model_name = f"../{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=1024)
    model = LLM(model=model_name, tensor_parallel_size=4)

    return model, sampling_params, tokenizer


def generate_variants_by_model_vllm(problem_description, put_code, model, sampling_params, tokenizer, k=2):
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
        {'role': "system", "content": system_prompt},
        {'role': "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_responses_list = []

    for i in range(k):
        output = model.generate(prompt, sampling_params)
        code = output.outputs[0].text

        print(code)

        model_responses_list.append(code)

    return model_responses_list


def generate_variants_by_model(problem_description, put_code, model, tokenizer, k=2, temperature=0.8,
                               max_new_token=1024):
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
        {'role': "system", "content": system_prompt},
        {'role': "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    model_responses_list = []

    for i in range(k):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_token,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(code)

        model_responses_list.append(code)

    return model_responses_list


def parse_and_generate_variants_for_TrickyBugs_by_model_vllm(model, tokenizer, sampling_params, model_name, k=6):
    dataset_path = "./Datasets/TrickyBugs"
    for dir in os.listdir(dataset_path):
        problem_description = None
        put_code = None

        flag = True
        for file in os.listdir(os.path.join(dataset_path, dir)):
            if file == "problem_description.txt":
                with open(os.path.join(dataset_path, dir, file), 'r', encoding='utf-8') as f:
                    problem_description = f.read()
            elif file == "buggy_programs":
                if not os.path.exists(os.path.join(dataset_path, dir, file, "python")):
                    flag = False
                    break
                for sub_dir in os.listdir(os.path.join(dataset_path, dir, file, "python")):
                    assert sub_dir.endswith(".py")
                    with open(os.path.join(dataset_path, dir, file, "python", sub_dir), 'r',
                              encoding='utf-8') as code_file:
                        put_code = code_file.read()
            else:
                continue

        if flag:
            response_list = generate_variants_by_model_vllm(problem_description, put_code, model, sampling_params,
                                                            tokenizer, k)
            for idx, response in enumerate(response_list):
                dir_name = f"./TrickyBugs/{model_name}/GenProgs/tc_generated_progs_python/{dir}"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(os.path.join(dir_name, dir + "_num_" + str(idx)), "w", encoding='utf-8') as resp_file:
                    resp_file.write(response)


def parse_and_generate_variants_for_TrickyBugs_by_model(model, tokenizer, model_name, k=6, temperature=0.8,
                                                        max_new_token=1024):
    dataset_path = "./Datasets/TrickyBugs"
    for dir in os.listdir(dataset_path):
        problem_description = None
        put_code_dict = {}

        flag = True
        for file in os.listdir(os.path.join(dataset_path, dir)):
            if file == "problem_description.txt":
                with open(os.path.join(dataset_path, dir, file), 'r', encoding='utf-8') as f:
                    problem_description = f.read()
            elif file == "buggy_programs":
                if not os.path.exists(os.path.join(dataset_path, dir, file, "python")):
                    flag = False
                    break
                for sub_dir in os.listdir(os.path.join(dataset_path, dir, file, "python")):
                    assert sub_dir.endswith(".py")
                    with open(os.path.join(dataset_path, dir, file, "python", sub_dir), 'r',
                              encoding='utf-8') as code_file:
                        put_code_dict['sub_dir'] = code_file.read()
            else:
                continue

        if flag:
            for code_file_name, put_code in put_code_dict.values():
                response_list = generate_variants_by_model(problem_description, put_code, model, tokenizer, k,
                                                           temperature, max_new_token)
                for idx, response in enumerate(response_list):
                    dir_name = f"./TrickyBugs/{model_name}/GenProgs/tc_generated_progs_python/{dir}/{code_file}"
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    # with open(os.path.join(dir_name , dir + "_num_" + str(idx)) , "w" , encoding='utf-8') as resp_file:
                    with open(os.path.join(dir_name, "_num_" + str(idx)), "w", encoding='utf-8') as resp_file:
                        resp_file.write(response)


def transform_code_for_TrickyBugs(model: str):
    response_dir = f"./TrickyBugs/{model}/GenProgs"
    for dir in os.listdir(os.path.join(response_dir, "tc_generated_progs_python")):
        code_dir = os.path.join(response_dir, "tc_generate_code_python_extracted", dir)
        for file in os.listdir(os.path.join(response_dir, "tc_generated_progs_python", dir)):
            with open(os.path.join(response_dir, "tc_generated_progs_python", dir, file), 'r', encoding='utf-8') as f:
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
            with open(os.path.join(code_dir, file + ".py"), 'w', encoding='utf-8') as f:
                f.write(code)


def generate_input_by_model(model, tokenizer, specification, pid, temperature=0.8, max_new_token=1024):
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
    sys_prompt = sys_prompt_template.format(pid=pid)
    user_prompt = """**PROBLEM DESCRIPTION**:
"""
    user_prompt += specification

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(prompt)
    print("------------------------------------------")

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_token,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(code)

    return code


def parse_and_generate_test_for_TrickyBugs_by_model(model, tokenizer, model_name, temperature=0.8, max_new_token=1024):
    dataset_dir = "./Datasets/TrickyBugs"
    pid = os.getpid()
    for dir in os.listdir(dataset_dir)[:10]:
        specification = None
        flag = True
        for file in os.listdir(os.path.join(dataset_dir, dir)):
            if file == "buggy_programs":
                if not os.path.exists(os.path.join(dataset_dir, dir, file, "python")):
                    flag = False
                    break

            if "problem_description" in file:
                with open(os.path.join(dataset_dir, dir, file), 'r', encoding="utf-8") as f:
                    specification = f.read()
                    f.close()
                break
        if flag is False:
            continue
        assert specification is not None

        response = generate_input_by_model(model, tokenizer, specification, pid, temperature, max_new_token)

        test_generator_dir = f"./TrickyBugs/{model_name}/GenInputs/tc_generator_python"

        result_path = os.path.join(test_generator_dir, dir)
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

        test_input_dir = os.path.join(f"./TrickyBugs/{model_name}/GenInputs/tc_inputs_generator", dir)

        if not os.path.exists(test_input_dir):
            os.makedirs(test_input_dir, exist_ok=True)

        with open(os.path.join(result_path, dir + "_test_generator"), 'w', encoding='utf-8') as f:
            f.write(response)
            f.close()


def extract_test_generator_for_model(model_name):
    input_folder = f"./TrickyBugs/{model_name}/GenInputs/tc_generator_python"
    for dir in os.listdir(input_folder):
        for file in os.listdir(os.path.join(input_folder, dir)):
            with open(os.path.join(input_folder, dir, file), 'r', encoding="utf-8") as f:
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

            extract_path = f"./TrickyBugs/{model_name}/GenInputs/tc_generator_python_extracted"
            os.makedirs(os.path.join(extract_path, dir), exist_ok=True)

            with open(os.path.join(extract_path, dir, file + ".py"), 'w', encoding="utf-8") as f:
                f.write(code)
                f.close()


def execute_input_generator_for_model(model_name):
    generator_dir = f"./TrickyBugs/{model_name}/GenInputs/tc_generator_python_extracted"
    for dir in os.listdir(generator_dir):
        for python_file in os.listdir(os.path.join(generator_dir, dir)):
            # with open(os.path.join(generator_dir , dir , python_file) , 'r' , encoding='utf-8') as f:
            #     code = f.read()
            #     f.close()
            if python_file.endswith(".py"):
                python_file_path = os.path.join("..", '..', 'tc_generator_python_extracted', dir, python_file)
                try:
                    exe_result = subprocess.run(["python", python_file_path], cwd=os.path.join(
                        f"./TrickyBugs/{model_name}/GenInputs/tc_inputs_generator", dir), capture_output=True,
                                                text=True, timeout=5)
                    if exe_result.returncode != 0:
                        print(f'Error in {os.path.join(generator_dir, dir, python_file)}: {exe_result.stderr}')
                        continue
                except:
                    print(f'Subprocess called error')
                    continue


def get_result_for_input(result_list: dict):
    if len(result_list) == 1:
        k = list(result_list.keys())
        return k[0]
    elif len(result_list) > 1:
        max_value = -1
        max_key = None
        for k, v in result_list.items():
            if v > max_value:
                max_value = v
                max_key = k
        return max_key


def execute_test_for_TrickyBugs(model: str):
    test_inputs_dir = f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator"
    code_path = f"./TrickyBugs/{model}/GenProgs/tc_generate_code_python_extracted"
    put_path = f"./Datasets/TrickyBugs"

    total_test_input = 0
    total_valid_input = 0
    total_failure_inducing_input = 0
    total_invalid_input = 0

    for dir in os.listdir(code_path):
        code_file_dir = os.path.join(code_path, dir)
        test_file_dir = os.path.join(test_inputs_dir, dir)
        code_file_path_list = []

        for code_file in os.listdir(code_file_dir):
            code_file_path = os.path.join(code_file_dir, code_file)
            code_file_path_list.append(code_file_path)

        for test_file in os.listdir(test_file_dir):
            test_file_path = os.path.join(test_file_dir, test_file)
            print(test_file_path)
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_input = f.read()

            if test_input.strip("\n") == "":
                continue

            put_dir = os.path.join(put_path, dir, "buggy_programs", "python")

            fixed_code_dir = os.path.join(put_path, dir, "fixed_programs", "python")
            if not os.path.exists(fixed_code_dir):
                fixed_code_dir = None

            if fixed_code_dir is not None:
                fixed_code_file = os.path.join(fixed_code_dir, os.listdir(fixed_code_dir)[0])

            result_list = {}
            for code_file_path in code_file_path_list:
                try:
                    result = subprocess.run(["python", code_file_path], input=test_input, capture_output=True,
                                            text=True, timeout=10)
                    if result.returncode != 0:
                        continue
                except subprocess.TimeoutExpired:
                    continue

                result_output = result.stdout
                result_list[result_output] = result_list.get(result_output, 0) + 1

            for put_code in os.listdir(put_dir):
                flag = True

                put_code_path = os.path.join(put_dir, put_code)
                try:
                    result_put = subprocess.run(["python", put_code_path], input=test_input, capture_output=True,
                                                text=True, timeout=10)
                except subprocess.TimeoutExpired:
                    continue

                if result_put.returncode != 0:
                    flag = False

                # for code_file_path in code_file_path_list:
                #     try:
                #         result = subprocess.run(["python" , code_file_path] , input=test_input , capture_output=True,
                #                                 text=True , timeout=10)
                #         if result.returncode != 0:
                #             continue
                #     except subprocess.TimeoutExpired:
                #         continue

                #     result_output = result.stdout
                #     result_list[result_output] = result_list.get(result_output , 0) + 1

                final_result = None

                if len(result_list) == 0 and flag is False:
                    final_result = "No valid output"
                elif len(result_list) == 0 and flag is True:
                    final_result = result_put.stdout
                elif flag is True and len(result_list) == 1:
                    final_result = get_result_for_input(result_list)
                elif flag is True and len(result_list) > 1:
                    final_result = get_result_for_input(result_list)
                elif flag is False and len(result_list) != 0:
                    final_result = get_result_for_input(result_list)

                if fixed_code_file is not None:
                    if final_result is not None and final_result != "No valid output":
                        total_test_input += 1

                        try:
                            fixed_code_result = subprocess.run(["python", fixed_code_file], input=test_input,
                                                               capture_output=True,
                                                               text=True, timeout=10)
                        except subprocess.TimeoutExpired:
                            total_invalid_input += 1
                            continue

                        if fixed_code_result.returncode != 0:
                            total_invalid_input += 1
                            continue

                        if fixed_code_result.stdout != final_result:
                            total_invalid_input += 1
                        elif fixed_code_result.stdout == final_result:
                            total_valid_input += 1
                            if result_put.returncode != 0 or result_put.stdout != final_result:
                                total_failure_inducing_input += 1
                                failure_inducing_dir_path = os.path.join(
                                    f"./TrickyBugs/{model}/GenInputs/tc_failure_inducing_output", dir)
                                os.makedirs(failure_inducing_dir_path, exist_ok=True)
                                with open(os.path.join(failure_inducing_dir_path,
                                                       put_code.split(".")[0] + "_" + test_file.split(".")[0] + ".out"),
                                          'w', encoding='utf-8') as f:
                                    f.write(str(final_result))

                            result_dir_path = os.path.join(f"./TrickyBugs/{model}/GenInputs/tc_valid_output", dir)
                            os.makedirs(result_dir_path, exist_ok=True)
                            result_file_path = os.path.join(result_dir_path,
                                                            put_code.split(".")[0] + "_" + test_file.split(".")[
                                                                0] + ".out")
                            with open(result_file_path, 'w', encoding='utf-8') as f:
                                f.write(str(final_result))

    if not os.path.exists(f"./TrickyBugs_results/{model}"):
        os.makedirs(f"./TrickyBugs_results/{model}", exist_ok=True)

    with open(f"./TrickyBugs_results/{model}/result_TrickyBugs.txt", 'w', encoding='utf-8') as f:
        f.write(f"Total test input: {total_test_input}\n")
        f.write(f"Total valid input: {total_valid_input}\n")
        f.write(f"Total invalid input: {total_invalid_input}\n")
        f.write(f"Total failure inducing input: {total_failure_inducing_input}\n")
        if total_test_input != 0:
            f.write(f"Ratio of failure-inducing test input: {total_failure_inducing_input / total_test_input : .4f}\n")
            f.write(f"Ratio of valid test input: {total_valid_input / total_test_input : .4f}\n")
            f.write(f"Ratio of invalid test input: {total_invalid_input / total_test_input : .4f}\n")
        else:
            f.write(f"Ratio of failure-inducing test input: 0\n")
            f.write(f"Ratio of valid test input: 0\n")
            f.write(f"Ratio of invalid test input: 0\n")


def calculate_the_coverage(model: str):
    valid_input_dir = f"./TrickyBugs/{model}/GenInputs/tc_valid_output"
    inputs_generator_dir = f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator"
    dataset_path = f"./Datasets/TrickyBugs"

    total_line = 0
    total_missing = 0

    for dir in os.listdir(valid_input_dir):
        report_dir = f"./coverage/TrickyBugs/{dir}/"
        os.makedirs(report_dir, exist_ok=True)
        buggy_code_dir = os.path.join(dataset_path, dir, "buggy_programs", "python")
        for buggy_code_file in os.listdir(buggy_code_dir):
            buggy_code_path = os.path.join(buggy_code_dir, buggy_code_file)

            test_output_dir = os.path.join(valid_input_dir, dir)
            for test_output_file in os.listdir(test_output_dir):
                name_list = test_output_file.split(".")[0].split("_")
                # print(name_list)
                input_file = os.path.join(inputs_generator_dir, dir, "_".join(name_list[2:]) + ".in")
                print(input_file)
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_text = f.read().strip("\n")

                subprocess.run(
                    ["coverage", "run", "--parallel-mode", buggy_code_path],
                    input=input_text,
                    text=True,
                    capture_output=True
                )

            cov = coverage.Coverage()
            cov.combine()
            analysis = cov.analysis(buggy_code_path)
            total_line += len(analysis[1])
            total_missing += len(analysis[2])
            coverage_report_file = os.path.join(report_dir, f"{buggy_code_file.split('.')[0]}_coverage_report.txt")
            with open(coverage_report_file, 'w', encoding='utf-8') as report_file:
                cov.report(file=report_file)
                report_file.write(f"\n{analysis[2]}")

    with open(f"./coverage/TrickyBugs/{model}/total_coverage.txt", 'w', encoding='utf-8') as f:
        f.write(f"Total line: {total_line}\n")
        f.write(f"Total missing: {total_missing}\n")
        f.write(f"Coverage: {(total_line - total_missing) / total_line : .4f}\n")


if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    model, tokenizer = load_model(model_name)
    parse_and_generate_variants_for_TrickyBugs_by_model(model, tokenizer, model_name)
    transform_code_for_TrickyBugs(model_name)

    # parse_and_generate_test_for_TrickyBugs_by_model(model , tokenizer , model_name)
    # extract_test_generator_for_model(model_name)
    # execute_input_generator_for_model(model_name)

    # execute_test_for_TrickyBugs(model_name)
    # calculate_the_coverage(model_name)

    # print(torch.cuda.device_count())

    # model , sampling_paras , tokenizer = load_model_vllm(model_name)
    # parse_and_generate_variants_for_TrickyBugs_by_model_vllm(model , tokenizer , sampling_paras , model_name)
    # transform_code_for_TrickyBugs(model_name)