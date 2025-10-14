table = {1 : 2 , 10 : 3}

def get_result_for_input(result_list: dict):
    if len(result_list) == 1:
        k = result_list.keys()
        return k
    elif len(result_list) > 1:
        max_value = -1
        max_key = None
        for k , v in result_list.items():
            if v > max_value:
                max_value = v
                max_key = k
        return max_key
    
print(type(get_result_for_input(table)))

# for k in table.keys():
#     print(str(k))