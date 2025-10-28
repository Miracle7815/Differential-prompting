# table = {1 : 2 , 10 : 2 , 30 : 3}

table = {}
print(len(table))

# def get_result_for_input(result_list: dict):
#     if len(result_list) == 1:
#         k = list(result_list.keys())
#         return k[0]
#     elif len(result_list) > 1:
#         max_value = -1
#         max_key = None
#         for k , v in result_list.items():
#             if v > max_value:
#                 max_value = v
#                 max_key = k
#         return max_key

# print(get_result_for_input(table))
# for k in table.keys():
#     print(str(k))

# input_folder = f"./EvalPlus/DP/{model}/GenInputs/tc_generator_python"
# for dir in os.listdir(input_folder):
#     for file in os.listdir(os.path.join(input_folder , dir)):
#         with open(os.path.join(input_folder , dir , file) , 'r' , encoding="utf-8") as f:
#             content = f.readlines()
#             f.close()

        
#         extract_path = f"./EvalPlus/{model}/GenInputs/tc_generator_python_extracted"
#         os.makedirs(os.path.join(extract_path , dir) , exist_ok=True)

#         with open(os.path.join(extract_path , dir , file + ".py") , 'w' , encoding="utf-8") as f:
#             f.write(code)
#             f.close()