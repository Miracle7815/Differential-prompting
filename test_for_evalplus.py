import os
import json
import inspect
import importlib.util
import multiprocessing
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import ast

# def load_all_functions(py_file):
#     """动态加载一个 .py 文件并返回其中定义的所有函数名与对象。"""
#     spec = importlib.util.spec_from_file_location("target_module", py_file)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)

#     # 从模块里筛出所有可调用函数（排除内置和导入）
#     funcs = {}
#     for name, obj in inspect.getmembers(module, inspect.isfunction):
#         if obj.__module__ == "target_module":  # 避免外部导入
#             funcs[name] = obj
#     return funcs

# def run_tests(py_file, test_file , file_name):
#     with open(test_file, "r", encoding="utf-8") as f:
#         tests = json.load(f)
#     test_cases = tests[file_name]

#     funcs = load_all_functions(py_file)

#     for func_name, fn in funcs.items():
#         print(f"\nTesting function: {func_name}")
#         passed = 0
#         for i, case in enumerate(test_cases, start=1):
#             args = case
#             try:
#                 # result = fn(*args)
#                 result = run_with_timeout(fn, args, timeout=10)
#                 print(i , result)
#             except Exception as e:
#                 print(f"Case {i}: {args} raised {type(e).__name__}: {e}")
#         print(f"Summary: {passed}/{len(test_cases)} passed")

# def worker_function(func, args, result_queue):
#     """在子进程中执行函数"""
#     try:
#         if isinstance(args, list):
#             result = func(*args)
#         elif isinstance(args, dict):
#             result = func(**args)
#         else:
#             result = func(args)
#         result_queue.put({"status": "success", "result": result})
#     except Exception as e:
#         result_queue.put({"status": "error", "error": str(e), "type": type(e).__name__})

# def run_with_timeout(func, args, timeout=10):
#     """使用多进程实现超时控制"""
#     result_queue = Queue()
#     process = Process(target=worker_function, args=(func, args, result_queue))
#     process.start()
#     process.join(timeout=timeout)
    
#     if process.is_alive():
#         # 超时，终止进程
#         process.terminate()
#         process.join()
#         raise TimeoutError(f"Function execution exceeded {timeout} seconds")
    
#     try:
#         result = result_queue.get_nowait()
#         if result["status"] == "success":
#             return result["result"]
#         else:
#             # 重新抛出原始异常
#             raise Exception(f"{result['type']}: {result['error']}")
#     except queue.Empty:
#         raise RuntimeError("Function completed but no result was returned")

# def run_with_timeout(fn, args=(), timeout=2):
#     """在独立进程中运行函数，超时则杀死该进程"""
#     def worker(q):
#         try:
#             result = fn(*args)
#             q.put(("ok", result))
#         except Exception as e:
#             q.put(("error", repr(e)))

#     q = multiprocessing.Queue()
#     p = multiprocessing.Process(target=worker, args=(q,))
#     p.start()
#     p.join(timeout)

#     if p.is_alive():
#         p.terminate()
#         p.join()
#         raise TimeoutError(f"{timeout}s time limit exceeded")

#     if q.empty():
#         raise RuntimeError("No output from process")

#     status, value = q.get()
#     if status == "error":
#         raise RuntimeError(value)
#     return value

def execute_function_in_process(py_file, func_name, args , result_queue , error_queue): 
    """在子进程中重新加载模块并执行函数"""
    # 在子进程中重新加载模块
    try:
        spec = importlib.util.spec_from_file_location("dynamic_module", py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取函数
        func = None
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name == func_name and obj.__module__ == "dynamic_module":
                func = obj
                break
        
        if func is None:
            raise ValueError(f"Function {func_name} not found in {py_file}")
        
        # 执行函数
        if isinstance(args, list):
            result = func(*args)
        elif isinstance(args, dict):
            result = func(**args)
        else:
            result = func(args)
        
        result_queue.put(result)
        
    except Exception as e:
        error_queue.put(e)
    

def run_with_timeout(func, args, timeout=10, py_file=None, func_name=None):
    """修改后的超时执行函数"""
    if py_file and func_name:
        # 使用进程池，传递文件路径和函数名
        # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        #     future = executor.submit(execute_function_in_process, py_file, func_name, args)
        #     try:
        #         result = future.result(timeout=timeout)
        #         return result
        #     except FutureTimeoutError:
        #         future.cancel()
        #         raise TimeoutError(f"Function execution exceeded {timeout} seconds")

        result_queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()
        
        # 创建进程
        process = multiprocessing.Process(
            target=execute_function_in_process,
            args=(py_file, func_name, args, result_queue, error_queue)
        )
        
        # 启动进程
        process.start()
        
        # 等待进程完成或超时
        process.join(timeout)
        
        if process.is_alive():
            # 进程仍在运行，强制终止
            process.terminate()  # 发送SIGTERM信号
            process.join(timeout=1)  # 给进程1秒时间优雅退出
            
            if process.is_alive():
                process.kill()  # 如果还活着，强制杀死(SIGKILL)
                process.join()
            
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")
        
        # 检查是否有错误
        if not error_queue.empty():
            raise error_queue.get()
        
        # 获取结果
        if not result_queue.empty():
            return result_queue.get()
        else:
            # 进程正常结束但没有结果，可能是函数返回None
            return None
    else:
        # 降级到线程执行（无法强制终止）
        return run_with_timeout_thread(func, args, timeout)

def run_with_timeout_thread(func, args, timeout):
    """使用线程的备用方案"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def worker():
        try:
            if isinstance(args, list):
                result = func(*args)
            elif isinstance(args, dict):
                result = func(**args)
            else:
                result = func(args)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Function execution exceeded {timeout} seconds")
    
    if not exception_queue.empty():
        raise exception_queue.get()
    
    if not result_queue.empty():
        return result_queue.get()

def load_all_functions(py_file):
    try:
        """动态加载一个 .py 文件并返回其中定义的所有函数名与对象。"""
        spec = importlib.util.spec_from_file_location("target_module", py_file)
        module = importlib.util.module_from_spec(spec)

        # 在执行模块前，将常用类型注解注入到模块的命名空间
        module.List = List
        module.Dict = Dict
        module.Set = Set
        module.Tuple = Tuple
        module.Optional = Optional
        module.Union = Union
        module.Any = Any
        
        # 也可以直接注入整个typing模块
        import typing
        module.typing = typing
        spec.loader.exec_module(module)

        # 从模块里筛出所有可调用函数（排除内置和导入）
        funcs = {}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == "target_module":  # 避免外部导入
                funcs[name] = obj
        return funcs
    except:
        return {}

def run_tests(py_file, test_file, file_name):
    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)
    test_cases = tests[file_name]

    funcs = load_all_functions(py_file)

    if funcs == {}:
        return []

    result_list = []

    for func_name, fn in funcs.items():
        print(f"\nTesting function: {func_name}")
        passed = 0
        failed = 0
        timeout_count = 0
        
        for i, case in enumerate(test_cases, start=1):
            args = case
            try:
                # 传递文件路径和函数名，而不是函数对象
                result = run_with_timeout(fn, args, timeout=10, 
                                        py_file=py_file, func_name=func_name)
                result_list.append(str(result).strip())
                print(f"Case {i}: {args} -> {result}")
                passed += 1
            except TimeoutError as e:
                print(f"Case {i}: {args} -> TIMEOUT: {e}")
                result_list.append("ERROR_RESULT")
                timeout_count += 1
            except Exception as e:
                print(f"Case {i}: {args} -> ERROR {type(e).__name__}: {e}")
                result_list.append("ERROR_RESULT")
                failed += 1
                
        print(f"Summary: {passed}/{len(test_cases)} passed, "
              f"{failed} failed, {timeout_count} timeout")
    
        return result_list
    
def run_tests_DP(py_file, test_file, file_name):
    test_cases = []

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("```") or not line.startswith("["):
                continue
            elif line:
                try:
                    test_case = ast.literal_eval(line)
                    test_cases.append(test_case)
                except Exception as e:
                    print(f"Error: {e}")

    funcs = load_all_functions(py_file)

    if funcs == {}:
        return []

    result_list = []

    for func_name, fn in funcs.items():
        print(f"\nTesting function: {func_name}")
        passed = 0
        failed = 0
        timeout_count = 0
        
        for i, case in enumerate(test_cases, start=1):
            args = case
            try:
                # 传递文件路径和函数名，而不是函数对象
                result = run_with_timeout(fn, args, timeout=10, 
                                        py_file=py_file, func_name=func_name)
                result_list.append(str(result).strip())
                print(f"Case {i}: {args} -> {result}")
                passed += 1
            except (TimeoutError , FutureTimeoutError) as e:
                print(f"Case {i}: {args} -> TIMEOUT: {e}")
                result_list.append("ERROR_RESULT")
                timeout_count += 1
            except Exception as e:
                print(f"Case {i}: {args} -> ERROR {type(e).__name__}: {e}")
                result_list.append("ERROR_RESULT")
                failed += 1
                
        print(f"Summary: {passed}/{len(test_cases)} passed, "
              f"{failed} failed, {timeout_count} timeout")
    
        return result_list
    
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
        
def execute_test_for_evalplus(model: str):
    put_path = "./Datasets/EvalPlus/PUTs"
    code_path = f"./EvalPlus/{model}/GenProgs/tc_generate_code_python_extracted"
    test_inputs_dir = f"./EvalPlus/{model}/GenInputs/tc_inputs_generator"
    fixed_path = f"./Datasets/EvalPlus/fixed_programs"

    total_generate_input = 0
    total_test_input = 0
    total_valid_input = 0
    total_failure_inducing_input = 0
    total_invalid_input = 0

    for dir in os.listdir(code_path):
        code_file_dir = os.path.join(code_path , dir)
        test_file_dir = os.path.join(test_inputs_dir , dir)
        code_file_path_list = []
        
        if not os.path.exists(test_file_dir):
            continue

        for code_file in os.listdir(code_file_dir):
            if code_file.endswith(".py"):
                code_file_path = os.path.join(code_file_dir , code_file)
                code_file_path_list.append(code_file_path)

        test_file_path = os.path.join(test_file_dir , os.listdir(test_file_dir)[0])
        print(test_file_path)

        result_lists = []    # list of map

        with open(test_file_path, "r", encoding="utf-8") as f:
            tests = json.load(f)
        
        if len(tests[dir]) == 0:
            continue

        for _ in range(len(tests[dir])):
            total_generate_input += 1
            result_lists.append({})

        print(code_file_path_list)
        for code_file_path in code_file_path_list:
            result_list = run_tests(code_file_path , test_file_path , dir)
            # print(result_list)

            for result_num , result in enumerate(result_list):
                if result != "ERROR_RESULT":
                    result_lists[result_num][result] = result_lists[result_num].get(result , 0) + 1
        
        print(len(result_lists))

        put_code_path = os.path.join(put_path , dir , "put0.py")
        put_result_list = run_tests(put_code_path , test_file_path , dir)

        final_result_list = []

        for put_result , result_table in zip(put_result_list , result_lists):
            final_result = None
            # if put_result != "ERROR_RESULT":
            #     result_table.pop(put_result , None)

            if len(result_table) == 0 and put_result == "ERROR_RESULT":
                final_result = "No valid output"
            elif len(result_table) == 0 and put_result != "ERROR_RESULT":
                final_result = "No valid output"
            elif put_result == "ERROR_RESULT" and len(result_table) == 1:
                final_result = get_result_for_input(result_table)
            elif put_result == "ERROR_RESULT" and len(result_table) > 1:
                final_result = get_result_for_input(result_table)
            elif put_result != "ERROR_RESULT" and len(result_table) != 0:
                final_result = get_result_for_input(result_table) 
            
            if final_result is not None and final_result != "No valid output":
                final_result_list.append(final_result)
                total_test_input += 1
            else:
                final_result_list.append("No valid output")

        print(final_result_list)

        fixed_code_path = os.path.join(fixed_path , dir , "put0.py")

        
        gt_result_list = run_tests(fixed_code_path , test_file_path , dir)

        with open(test_file_path , 'r' , encoding='utf-8') as f:
            test_inputs = json.load(f)

        test_input_list = test_inputs[dir]

        fitc_input_list = []
        fitc_result_list = []

        valid_input_list = []
        valid_result_list = []

        for gt_result , put_result , final_result , test_input in zip(gt_result_list , put_result_list , final_result_list , test_input_list):
            if gt_result == "ERROR_RESULT":
                if final_result != "No valid output":
                    total_invalid_input += 1
            elif gt_result != "ERROR_RESULT":
                if gt_result != final_result:
                    total_invalid_input += 1
                elif gt_result == final_result:
                    total_valid_input += 1
                    valid_input_list.append(test_input)
                    valid_result_list.append(final_result)
                    if put_result != final_result:
                        total_failure_inducing_input += 1
                        fitc_input_list.append(test_input)
                        fitc_result_list.append(final_result)
        
        fitc_dir = os.path.join(f"./EvalPlus/{model}/GenInputs/tc_failure_inducing_input" , dir)
        valid_dir = os.path.join(f"./EvalPlus/{model}/GenInputs/tc_valid_input" , dir)
        os.makedirs(valid_dir , exist_ok=True)
        os.makedirs(fitc_dir , exist_ok=True)

        with open(os.path.join(fitc_dir , "fitc_input.json") , 'w' , encoding='utf-8') as f:
            json.dump({"inputs": fitc_input_list , "ouputs": fitc_result_list} , f , indent=4)
            f.close()
        
        with open(os.path.join(valid_dir , "valid_input.json") , 'w' , encoding='utf-8') as f:
            json.dump({"inputs": valid_input_list , "ouputs": valid_result_list} , f , indent=4)
            f.close()

    os.makedirs(f"./EvalPlus_results/TC/{model}" , exist_ok=True)
    
    with open(f"./EvalPlus_results/TC/{model}/result_EvalPlus.txt" , 'w' , encoding='utf-8') as f:
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

def execute_test_for_EvalPlus_DP(model: str):
    put_path = "./Datasets/EvalPlus/PUTs"
    code_path = f"./EvalPlus/DP/{model}/GenProgs/dp_generate_code_python_extracted"
    test_inputs_dir = f"./EvalPlus/DP/{model}/GenInputs/dp_generator_python"
    fixed_path = f"./Datasets/EvalPlus/fixed_programs"

    total_generate_input = 0
    total_test_input = 0
    total_valid_input = 0
    total_failure_inducing_input = 0
    total_invalid_input = 0

    for dir in os.listdir(code_path):
        code_file_dir = os.path.join(code_path , dir)
        test_file_dir = os.path.join(test_inputs_dir , dir)
        code_file_path_list = []
        
        if not os.path.exists(test_file_dir):
            continue

        for code_file in os.listdir(code_file_dir):
            if code_file.endswith(".py"):
                code_file_path = os.path.join(code_file_dir , code_file)
                code_file_path_list.append(code_file_path)

        test_file_path = os.path.join(test_file_dir , os.listdir(test_file_dir)[0])
        print(test_file_path)

        result_lists = []    # list of map

        test_cases = []

        with open(test_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("```") or not line.startswith("["):
                    continue
                elif line:
                    try:
                        test_case = ast.literal_eval(line)
                        test_cases.append(test_case)
                    except Exception as e:
                        print(f"Error: {e}")
        
        if len(test_cases) == 0:
            continue

        for _ in range(len(test_cases)):
            total_generate_input += 1
            result_lists.append({})

        print(code_file_path_list)
        for code_file_path in code_file_path_list:
            result_list = run_tests_DP(code_file_path , test_file_path , dir)
            # print(result_list)

            for result_num , result in enumerate(result_list):
                if result != "ERROR_RESULT":
                    result_lists[result_num][result] = result_lists[result_num].get(result , 0) + 1
        
        print(len(result_lists))

        put_code_path = os.path.join(put_path , dir , "put0.py")
        put_result_list = run_tests_DP(put_code_path , test_file_path , dir)

        final_result_list = []

        for put_result , result_table in zip(put_result_list , result_lists):
            final_result = None
            # if put_result != "ERROR_RESULT":
            #     result_table.pop(put_result , None)

            if len(result_table) == 0 and put_result == "ERROR_RESULT":
                final_result = "No valid output"
            elif len(result_table) == 0 and put_result != "ERROR_RESULT":
                final_result = "No valid output"
            elif put_result == "ERROR_RESULT" and len(result_table) == 1:
                final_result = get_result_for_input(result_table)
            elif put_result == "ERROR_RESULT" and len(result_table) > 1:
                final_result = get_result_for_input(result_table)
            elif put_result != "ERROR_RESULT" and len(result_table) != 0:
                final_result = get_result_for_input(result_table) 
            
            if final_result is not None and final_result != "No valid output":
                final_result_list.append(final_result)
                total_test_input += 1
            else:
                final_result_list.append("No valid output")

        print(final_result_list)

        fixed_code_path = os.path.join(fixed_path , dir , "put0.py")

        
        gt_result_list = run_tests_DP(fixed_code_path , test_file_path , dir)

        fitc_input_list = []
        fitc_result_list = []

        valid_input_list = []
        valid_result_list = []

        for gt_result , put_result , final_result , test_input in zip(gt_result_list , put_result_list , final_result_list , test_cases):
            if gt_result == "ERROR_RESULT":
                if final_result != "No valid output":
                    total_invalid_input += 1
            elif gt_result != "ERROR_RESULT":
                if gt_result != final_result:
                    total_invalid_input += 1
                elif gt_result == final_result:
                    total_valid_input += 1
                    valid_input_list.append(test_input)
                    valid_result_list.append(final_result)
                    if put_result != final_result:
                        total_failure_inducing_input += 1
                        fitc_input_list.append(test_input)
                        fitc_result_list.append(final_result)
        
        fitc_dir = os.path.join(f"./EvalPlus/DP/{model}/GenInputs/dp_failure_inducing_input" , dir)
        valid_dir = os.path.join(f"./EvalPlus/DP/{model}/GenInputs/dp_valid_input" , dir)
        os.makedirs(fitc_dir , exist_ok=True)
        os.makedirs(valid_dir , exist_ok=True)

        with open(os.path.join(fitc_dir , "fitc_input.json") , 'w' , encoding='utf-8') as f:
            json.dump({"inputs": fitc_input_list , "ouputs": fitc_result_list} , f , indent=4)
            f.close()

        with open(os.path.join(valid_dir , "valid_input.json") , 'w' , encoding='utf-8') as f:
            json.dump({"inputs": valid_input_list , "ouputs": valid_result_list} , f , indent=4)
            f.close()

    os.makedirs(f"./EvalPlus_results/DP/{model}" , exist_ok=True)
    
    with open(f"./EvalPlus_results/DP/{model}/result_EvalPlus.txt" , 'w' , encoding='utf-8') as f:
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

if __name__ == "__main__":
    # python_dir = "./EvalPlus/gpt-3.5-turbo-1106/GenProgs/tc_generate_code_python_extracted"
    # python_file = os.path.join(python_dir, "HumanEval_0" , "num_1.py")
    # test_file = "./EvalPlus/gpt-3.5-turbo-1106/GenInputs/tc_inputs_generator/HumanEval_0/HumanEval_0_inputs.json"

    # print(load_all_functions(python_file))

    # run_tests(python_file, test_file , "HumanEval_0")

    model_name = "gpt-3.5-turbo-1106"

    mode = "dp"
    # mode = "tc"

    if mode == "tc":
        execute_test_for_evalplus(model_name)
    elif mode == "dp":
        execute_test_for_EvalPlus_DP(model_name)