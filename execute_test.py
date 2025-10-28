import os
from utils import *
import subprocess
import coverage

def run_command_with_pipes(command, input_data):
    # Run the command and return its output as a result
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return process.communicate(input_data.encode("utf-8"))[0]

def execute_test(model):
    code_dir = './Variants/essay'
    
    result_list = {}

    for dir in os.listdir(code_dir):

        for f in os.listdir("."):
            if f.startswith(".coverage"):
                os.remove(f)
        
        report_flag = False

        problem_id = dir
        variant_path_list = []
        for code_file in os.listdir(os.path.join(code_dir , dir)):
            if model in code_file:
                variant_path_list.append(os.path.join(code_dir , problem_id , code_file))
        
        assert len(variant_path_list) == 2
        # print(variant_path_list)

        test_input_path = os.path.join("./Test input/essay" , problem_id , problem_id + "_" + model + ".txt")
        print(test_input_path)

        result_list[problem_id] = []
        test_input_num , test_input_list = count_and_extract_test_inputs(test_input_path)
        # print(test_input_num , test_input_list)

        PUT_path = os.path.join("./Code in dataset" , problem_id + ".py")

        for idx ,test_input in enumerate(test_input_list):
            try:
                result_1 = subprocess.run(["python" , variant_path_list[0]] , input=test_input , capture_output=True,
                                        text=True , timeout=10)
                if result_1.returncode != 0:
                    continue
            except subprocess.TimeoutExpired:
                continue

            result_1_output = result_1.stdout
            # print("Output_1: ")
            # print(result_1_output)

            try:
                result_2 = subprocess.run(["python" , variant_path_list[1]] , input=test_input , capture_output=True,
                                        text=True , timeout=10)
                if result_2.returncode != 0:
                    continue
            except subprocess.TimeoutExpired:
                continue

            result_2_output = result_2.stdout

            # if result_1_output == result_2_output:
            #     print("Test case " + str(idx + 1) + " passed: ")
            #     print(result_2_output)
            #     print("--------------------")

            if result_1_output != result_2_output:
                continue
            
            try:
                result_3 = subprocess.run(["python" , PUT_path] , input=test_input , capture_output=True,
                                        text=True , timeout=10)
                
            except subprocess.TimeoutExpired:
                continue
            
            PUT_output = result_3.stdout
            # if result_1_output == PUT_output:
            #     print("Test case " + str(idx + 1) + " passed: ")
            #     print(PUT_output)
            #     print("--------------------")

            if result_1_output != PUT_output:
                print("Test case " + str(idx + 1) + " maybe a Failure-Inducing test: ")
                print(PUT_output)
                print("--------------------")
                result_list[problem_id].append(f"{idx + 1}:\n{test_input}\noutput: {result_1_output}")

            # if result_1_output == PUT_output:
            #     command = ["coverage" , "run" , PUT_path]
            #     output = run_command_with_pipes(command , test_input)

            #     cov = coverage.Coverage()
            #     cov.load()

            #     coverage_report_file = "./coverage/coverage_report.txt"
            #     with open(coverage_report_file, 'a+' , encoding='utf-8') as report_file:
            #         cov.report(file=report_file)

            if result_1_output == PUT_output:
                subprocess.run(
                    ["coverage" , "run" , "--parallel-mode" , PUT_path],
                    input=test_input,
                    text=True,
                    capture_output=True
                )
                if not report_flag:
                    report_flag = True
        
        if report_flag:
            cov = coverage.Coverage()
            cov.combine()
            analysis = cov.analysis(PUT_path)
            coverage_report_file = "./coverage/coverage_report_essay_essay.txt"
            with open(coverage_report_file, 'a+' , encoding='utf-8') as report_file:
                cov.report(file=report_file)
                report_file.write(f"\n{analysis[2]}")
                report_file.write("\n\n")



    with open("./result/result_file_essay_code_essay_test.txt" , 'w' , encoding='utf-8') as f:
        for problem_id , FI_list in result_list.items():
            f.write(f"Problem {problem_id}: \n")
            for FI_test in FI_list:
                f.write(f"No.{FI_test}\n")
            f.write("\n\n")

def get_result_for_input(result_list: dict):
    if len(result_list) == 1:
        k = list(result_list.keys())
        return k[0]
    elif len(result_list) > 1:
        max_value = -1
        max_key = None
        for k , v in result_list.items():
            if v > max_value:
                max_value = v
                max_key = k
        return max_key

def execute_test_for_TrickyBugs(model: str , k):
    test_inputs_dir = f"./TrickyBugs/{model}/{k}/GenInputs/tc_inputs_generator"
    code_path = f"./TrickyBugs/{model}/{k}/GenProgs/tc_generate_code_python_extracted"
    put_path = f"./Datasets/TrickyBugs/PUT_python"
    fixed_path = f"./Datasets/TrickyBugs/fixed_programs"

    total_generate_input = 0
    total_test_input = 0
    total_valid_input = 0
    total_failure_inducing_input = 0
    total_invalid_input = 0

    for dir in os.listdir(code_path):
        code_file_dir = os.path.join(code_path , dir)
        test_file_dir = os.path.join(test_inputs_dir , dir)
        code_file_path_list = []

        for code_file in os.listdir(code_file_dir):
            code_file_path = os.path.join(code_file_dir , code_file)
            code_file_path_list.append(code_file_path)

        for test_file in os.listdir(test_file_dir):
            total_generate_input += 1
            test_file_path = os.path.join(test_file_dir , test_file)
            print(test_file_path)
            with open(test_file_path , 'r' , encoding='utf-8') as f:
                test_input = f.read()

            if test_input.strip("\n") == "":
                continue

            put_dir = os.path.join(put_path , dir)

            fixed_code_dir = os.path.join(fixed_path , dir)
            fixed_code_file = os.path.join(fixed_code_dir , os.listdir(fixed_code_dir)[0])
            
            result_list = {}

            for code_file_path in code_file_path_list:
                try:
                    result = subprocess.run(["python" , code_file_path] , input=test_input , capture_output=True,
                                            text=True , timeout=10)
                    if result.returncode != 0:
                        continue
                except Exception as e:
                    continue
                    
                result_output = result.stdout
                result_list[result_output] = result_list.get(result_output , 0) + 1
        
            for put_code in os.listdir(put_dir):
                flag = True

                put_code_path = os.path.join(put_dir , put_code)
                try:
                    result_put = subprocess.run(["python" , put_code_path] , input=test_input , capture_output=True,
                                            text=True , timeout=10)
                except subprocess.TimeoutExpired:
                    flag = False
                
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
                
                if flag is True:
                    result_list.pop(result_put.stdout , None)

                if len(result_list) == 0 and flag is False:
                    final_result = "No valid output"
                elif len(result_list) == 0 and flag is True:
                    final_result = "No valid output"
                elif flag is True and len(result_list) == 1:
                    final_result = get_result_for_input(result_list)
                elif flag is True and len(result_list) > 1:
                    final_result = get_result_for_input(result_list)
                elif flag is False and len(result_list) != 0:
                    final_result = get_result_for_input(result_list) 

                if final_result is not None and final_result != "No valid output":
                    total_test_input += 1
                    
                    try:
                        fixed_code_result = subprocess.run(["python" , fixed_code_file] , input=test_input , capture_output=True,
                                                            text=True , timeout=10)
                    except Exception as e:
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
                            failure_inducing_dir_path = os.path.join(f"./TrickyBugs/{model}/{k}/GenInputs/tc_failure_inducing_output" , dir)
                            os.makedirs(failure_inducing_dir_path , exist_ok=True)
                            with open(os.path.join(failure_inducing_dir_path , put_code.split(".")[0] + "_" + test_file.split(".")[0] + ".out") , 'w' , encoding='utf-8') as f:
                                f.write(str(final_result))

                        result_dir_path = os.path.join(f"./TrickyBugs/{model}/{k}/GenInputs/tc_valid_output" , dir)
                        os.makedirs(result_dir_path , exist_ok=True)
                        result_file_path = os.path.join(result_dir_path , put_code.split(".")[0] + "_" + test_file.split(".")[0] + ".out")
                        with open(result_file_path , 'w' , encoding='utf-8') as f:
                            f.write(str(final_result))

    os.makedirs(f"./TrickyBugs_results/{model}/{k}" , exist_ok=True)
    
    with open(f"./TrickyBugs_results/{model}/{k}/result_TrickyBugs.txt" , 'w' , encoding='utf-8') as f:
        f.write(f"Total generate input: {total_generate_input}\n")
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



                # if len(result_list) == 0 and flag is False:
                #     with open(result_file_path , 'w' , encoding='utf-8') as f:
                #         f.write("No valid output")
                # elif len(result_list) == 0 and flag is True:
                #     with open(result_file_path , 'w' , encoding='utf-8') as f:
                #         f.write(str(result_put.stdout))
                # elif flag is True and len(result_list) == 1:
                #     with open(result_file_path , 'w' , encoding='utf-8') as f:
                #         f.write(str(get_result_for_input(result_list)))
                # elif flag is True and len(result_list) > 1:
                #     with open(result_file_path , 'w' , encoding='utf-8') as f:
                #         f.write(str(get_result_for_input(result_list)))
                # elif flag is False and len(result_list) != 0:
                #     with open(result_file_path , 'w' , encoding='utf-8') as f:
                #         f.write(str(get_result_for_input(result_list)))

# def result_for_TrickyBugs(model: str):
#     input_dir_path = f"./TrickyBugs/{model}/GenInputs/tc_inputs_generator"
#     output_dir_path = f"./TrickyBugs/{model}/GenInputs/tc_inputs_output"
#     fixed_code_path = f"./Datasets/TrickyBugs"
#     put_output_path = f"./TrickyBugs/{model}/GenInputs/put_out_put"

#     total_test_input = 0
#     total_valid_input = 0
#     total_failure_inducing_input = 0

#     for dir in os.listdir(input_dir_path):
#         fixed_code_path = os.path.join(fixed_code_path , dir , "fixed_programs" , "python")
#         if not os.path.exists(fixed_code_path):
#             continue

#         fixed_code_path = os.path.join(fixed_code_path , os.listdir(fixed_code_path)[0])

#         for sol_dir in os.listdir(os.path.join(output_dir_path , dir)):
#             for input_file in os.listdir(os.path.join(input_dir_path , dir)):
#                 with open(os.path.join(input_dir_path , dir , input_file) , 'r' , encoding='utf-8') as f:
#                     test_input = f.read().strip("\n")
                
#                 if test_input.strip("\n") == "":
#                     continue
                
#                 with open(os.path.join(output_dir_path , dir , sol_dir , input_file.split(".")[0] + ".out") , 'r' , encoding='utf-8') as f:
#                     test_output = f.read().strip("\n")
                
#                 with open(os.path.join())

def execute_test_for_TrickyBugs_DP(model: str , k):
    test_inputs_dir = f"./TrickyBugs/DP/{model}/{k}/GenInputs/dp_inputs_generator"
    code_path = f"./TrickyBugs/DP/{model}/{k}/GenProgs/dp_generate_code_python_extracted"
    put_path = f"./Datasets/TrickyBugs/PUT_python"
    fixed_path = f"./Datasets/TrickyBugs/fixed_programs"

    total_generate_input = 0
    total_test_input = 0
    total_valid_input = 0
    total_failure_inducing_input = 0
    total_invalid_input = 0

    for dir in os.listdir(code_path):
        code_file_dir = os.path.join(code_path , dir)
        test_file_dir = os.path.join(test_inputs_dir , dir)
        code_file_path_list = []

        for code_file in os.listdir(code_file_dir):
            code_file_path = os.path.join(code_file_dir , code_file)
            code_file_path_list.append(code_file_path)

        for test_file in os.listdir(test_file_dir):
            total_generate_input += 1
            test_file_path = os.path.join(test_file_dir , test_file)
            print(test_file_path)
            with open(test_file_path , 'r' , encoding='utf-8') as f:
                test_input = f.read()

            if test_input.strip("\n") == "":
                continue

            put_dir = os.path.join(put_path , dir)

            fixed_code_dir = os.path.join(fixed_path , dir)
            fixed_code_file = os.path.join(fixed_code_dir , os.listdir(fixed_code_dir)[0])
            
            result_list = {}

            for code_file_path in code_file_path_list:
                try:
                    result = subprocess.run(["python" , code_file_path] , input=test_input , capture_output=True,
                                            text=True , timeout=10)
                    if result.returncode != 0:
                        continue
                except Exception as e:
                    continue
                    
                result_output = result.stdout
                result_list[result_output] = result_list.get(result_output , 0) + 1
        
            for put_code in os.listdir(put_dir):
                flag = True

                put_code_path = os.path.join(put_dir , put_code)
                try:
                    result_put = subprocess.run(["python" , put_code_path] , input=test_input , capture_output=True,
                                            text=True , timeout=10)
                except subprocess.TimeoutExpired:
                    flag = False
                
                if result_put.returncode != 0:
                    flag = False

                if flag is True:
                    result_list.pop(result_put.stdout , None)
            
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
                    final_result = "No valid output"
                elif flag is True and len(result_list) == 1:
                    final_result = get_result_for_input(result_list)
                elif flag is True and len(result_list) > 1:
                    final_result = get_result_for_input(result_list)
                elif flag is False and len(result_list) != 0:
                    final_result = get_result_for_input(result_list) 

                if final_result is not None and final_result != "No valid output":
                    total_test_input += 1
                    
                    try:
                        fixed_code_result = subprocess.run(["python" , fixed_code_file] , input=test_input , capture_output=True,
                                                            text=True , timeout=10)
                    except Exception as e:
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
                            failure_inducing_dir_path = os.path.join(f"./TrickyBugs/DP/{model}/{k}/GenInputs/dp_failure_inducing_output" , dir)
                            os.makedirs(failure_inducing_dir_path , exist_ok=True)
                            with open(os.path.join(failure_inducing_dir_path , test_file.split(".")[0] + ".out") , 'w' , encoding='utf-8') as f:
                                f.write(str(final_result))

                        result_dir_path = os.path.join(f"./TrickyBugs/DP/{model}/{k}/GenInputs/dp_valid_output" , dir)
                        os.makedirs(result_dir_path , exist_ok=True)
                        result_file_path = os.path.join(result_dir_path , test_file.split(".")[0] + ".out")
                        with open(result_file_path , 'w' , encoding='utf-8') as f:
                            f.write(str(final_result))

    os.makedirs(f"./TrickyBugs_results/DP/{model}/{k}" , exist_ok=True)
    
    with open(f"./TrickyBugs_results/DP/{model}/{k}/result_TrickyBugs.txt" , 'w' , encoding='utf-8') as f:
        f.write(f"Total generate input: {total_generate_input}\n")
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


def count_effective_lines(filename):
    count = 0
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith('#'):
                continue
            count += 1
    return count

def calculate_the_coverage(model: str , k):
    valid_input_dir = f"./TrickyBugs/{model}/{k}/GenInputs/tc_valid_output"
    inputs_generator_dir = f"./TrickyBugs/{model}/{k}/GenInputs/tc_inputs_generator"
    dataset_path = f"./Datasets/TrickyBugs"
    put_dir = f"./Datasets/TrickyBugs/PUT_python"

    total_line = 0
    total_missing = 0

    for dir in os.listdir(put_dir):
        report_dir = f"./coverage/TrickyBugs/{model}/{k}/{dir}"
        os.makedirs(report_dir , exist_ok=True)

        valid_output_path = os.path.join(valid_input_dir , dir)
        if not os.path.exists(valid_output_path):
            code_dir = os.path.join(put_dir , dir)
            code_file = os.path.join(code_dir , os.listdir(code_dir)[0])
            line_num = count_effective_lines(code_file)
            total_line += line_num
            total_missing += line_num
            with open(os.path.join(report_dir , os.listdir(code_dir)[0].split('.')[0] + "_coverage_report.txt") , 'w' , encoding='utf-8') as f:
                f.write("no valid input\n")
            continue
        else:
            buggy_code_dir = os.path.join(put_dir , dir)

            for buggy_code_file in os.listdir(buggy_code_dir):
                buggy_code_path = os.path.join(buggy_code_dir , buggy_code_file)

                test_output_dir = os.path.join(valid_input_dir , dir)
                for test_output_file in os.listdir(test_output_dir):
                    name_list = test_output_file.split(".")[0].split("_")
                    # print(name_list)
                    input_file = os.path.join(inputs_generator_dir , dir , "_".join(name_list[1:]) + ".in")
                    print(input_file)
                    with open(input_file , 'r' , encoding='utf-8') as f:
                        input_text = f.read().strip("\n")
                    
                    subprocess.run(
                        ["coverage" , "run" , "--parallel-mode" , buggy_code_path],
                        input=input_text,
                        text=True,
                        capture_output=True
                    )

                cov = coverage.Coverage()
                cov.combine()
                analysis = cov.analysis(buggy_code_path)
                total_line += len(analysis[1])
                total_missing += len(analysis[2])
                coverage_report_file = os.path.join(report_dir , f"{buggy_code_file.split('.')[0]}_coverage_report.txt")
                with open(coverage_report_file, 'w' , encoding='utf-8') as report_file:
                    cov.report(file=report_file)
                    report_file.write(f"\n{analysis[2]}")

    with open(f"./coverage/TrickyBugs/{model}/{k}/total_coverage.txt" , 'w' , encoding='utf-8') as f:
        f.write(f"Total line: {total_line}\n")
        f.write(f"Total missing: {total_missing}\n")
        f.write(f"Coverage: {(total_line -total_missing) / total_line : .4f}\n")

def calculate_the_coverage_for_DP(model: str , k):
    valid_input_dir = f"./TrickyBugs/DP/{model}/{k}/GenInputs/dp_valid_output"
    inputs_generator_dir = f"./TrickyBugs/DP/{model}/{k}/GenInputs/dp_inputs_generator"
    dataset_path = f"./Datasets/TrickyBugs"
    put_dir = f"./Datasets/TrickyBugs/PUT_python"

    total_line = 0
    total_missing = 0

    for dir in os.listdir(put_dir):
        report_dir = f"./coverage/TrickyBugs/DP/{model}/{k}/{dir}"
        os.makedirs(report_dir , exist_ok=True)

        valid_output_path = os.path.join(valid_input_dir , dir)
        if not os.path.exists(valid_output_path):
            code_dir = os.path.join(put_dir , dir)
            code_file = os.path.join(code_dir , os.listdir(code_dir)[0])
            line_num = count_effective_lines(code_file)
            total_line += line_num
            total_missing += line_num
            with open(os.path.join(report_dir , os.listdir(code_dir)[0].split('.')[0] + "_coverage_report.txt") , 'w' , encoding='utf-8') as f:
                f.write("no valid input\n")
            continue
        else:
            buggy_code_dir = os.path.join(put_dir , dir)

            for buggy_code_file in os.listdir(buggy_code_dir):
                buggy_code_path = os.path.join(buggy_code_dir , buggy_code_file)

                test_output_dir = os.path.join(valid_input_dir , dir)
                for test_output_file in os.listdir(test_output_dir):
                    name_list = test_output_file.split(".")[0]
                    # print(name_list)
                    input_file = os.path.join(inputs_generator_dir , dir , name_list + ".in")
                    print(input_file)
                    with open(input_file , 'r' , encoding='utf-8') as f:
                        input_text = f.read().strip("\n")
                    
                    subprocess.run(
                        ["coverage" , "run" , "--parallel-mode" , buggy_code_path],
                        input=input_text,
                        text=True,
                        capture_output=True
                    )

                cov = coverage.Coverage()
                cov.combine()
                analysis = cov.analysis(buggy_code_path)
                total_line += len(analysis[1])
                total_missing += len(analysis[2])
                coverage_report_file = os.path.join(report_dir , f"{buggy_code_file.split('.')[0]}_coverage_report.txt")
                with open(coverage_report_file, 'w' , encoding='utf-8') as report_file:
                    cov.report(file=report_file)
                    report_file.write(f"\n{analysis[2]}")

    with open(f"./coverage/TrickyBugs/DP/{model}/{k}/total_coverage.txt" , 'w' , encoding='utf-8') as f:
        f.write(f"Total line: {total_line}\n")
        f.write(f"Total missing: {total_missing}\n")
        f.write(f"Coverage: {(total_line -total_missing) / total_line : .4f}\n")

def execute_test_for_EvalPlus(model: str):
    pass 

if __name__ == "__main__":
    # mode = "tc"
    mode = "dp"
    
    model_name = "gpt-3.5-turbo-1106"

    k = 6

    if mode == "tc":
        execute_test_for_TrickyBugs(model_name , k)
        calculate_the_coverage(model_name , k)
    elif mode == "dp":
        execute_test_for_TrickyBugs_DP(model_name , k)
        calculate_the_coverage_for_DP(model_name , k)

    # execute_test_for_TrickyBugs("gpt-4o-mini")
    # calculate_the_coverage("gpt-4o-mini")

    # execute_test_for_TrickyBugs("Qwen/Qwen2.5-0.5B-Instruct")
    # calculate_the_coverage("Qwen/Qwen2.5-0.5B-Instruct")