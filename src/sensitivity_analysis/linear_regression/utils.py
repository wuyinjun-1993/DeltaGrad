'''
Created on Apr 2, 2019

'''

import random
import torch


def get_relative_change(tensor1, tensor2):
    print('relative magnitude change::', torch.max(torch.pow(tensor1 - tensor2, 2)))
    
    
    

def get_subset_training_data(X, dim, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(dim[0])) - set(delta_data_ids.tolist())))
#     print(selected_rows)
    update_X = torch.index_select(X, 0, selected_rows)
    return update_X, selected_rows

def get_subset_parameter_list(selected_rows, delta_data_ids, para_list, dim, axis):
    para_list_tensor = torch.tensor(para_list, dtype = torch.double)
    update_para_list_tensor = torch.index_select(para_list_tensor, axis, selected_rows)
    return update_para_list_tensor
    

def random_generate_subset_ids(dim, delta_size):
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, dim[0]-1)
        delta_data_ids.add(id)
    
#     for i in range(delta_size):
#     
#         id = random.randint(0, dim[0]-1)
#         
#         if id in delta_data_ids:
#             i = i-1
#             
#             continue
    #     print(id, i)
    #     print(update_X_product)
        
    
    return torch.tensor(list(delta_data_ids))