'''
Created on Sep 11, 2019

'''

from torch import nn, optim
import torch

# try:
#     from ...data_IO import *
# except ImportError:
#     from .data_IO import *

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/logistic_regression')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import *
    from Logistic_regression import *
except ImportError:
    from Load_data import *
    from Logistic_regression import *

import random



def random_generate_subset_ids(num, delta_size):
    
    delta_data_ids = set()
    
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, num-1)
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


def get_subset_training_data(num, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(num)) - set(delta_data_ids.tolist())))
#     print(selected_rows)
#     update_X = torch.index_select(X, 0, selected_rows)
#     
#     update_Y = torch.index_select(Y, 0, selected_rows)
    
    return selected_rows


def get_subset_training_data0(num, delta_data_ids):
    selected_rows = torch.tensor(list(set(range(num)) - set(delta_data_ids.tolist())))
    
    return selected_rows



def compute_model_para_diff(model1_para_list, model2_para_list):
    
    diff = 0
    
    norm1 = 0
    
    norm2 = 0
    
    all_dot = 0
    
    
    for i in range(len(model1_para_list)):
        
        param1 = model1_para_list[i].to('cpu')
        
        param2 = model2_para_list[i].to('cpu')
        
        curr_diff = torch.norm(param1 - param2, p='fro')
        
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        
        
        all_dot += torch.sum(param1*param2)
        
        print("curr_diff:", i, curr_diff)
        
        diff += curr_diff*curr_diff
        
    print(torch.sqrt(diff))
    
    print(all_dot/torch.sqrt(norm1*norm2))
    
def compute_model_para_diff2(para_list1, para_list2):
    
    diff = 0
    
    norm1 = 0
    
    norm2 = 0
    
    all_dot = 0
    
    
    for i in range(len(para_list1)):
        
        param1 = para_list1[i]
        
        print(param1)
        
        param2 = para_list2[i]
        
        curr_diff = torch.norm(param1 - param2, p='fro')
        
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        
        
        print("this_diff0::", curr_diff)
        
        all_dot += torch.sum(param1*param2)
        
        diff += curr_diff*curr_diff
        
    print(torch.sqrt(diff))
    
    print(all_dot/torch.sqrt(norm1*norm2))


def compute_model_para_diff3(model1_para_list, model2_para_list):
    
    diff = 0
    
    norm1 = 0
    
    norm2 = 0
    
    all_dot = 0
    
    
    for i in range(len(model1_para_list)):
        
        param1 = model1_para_list[i]
        
        param2 = model2_para_list[i]
        
        
        param_diff = torch.max(torch.abs(param1 - param2))
        
#         curr_diff = torch.norm(param1 - param2, p='fro')
#         
#         norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
#         
#         norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
#         
#         
#         all_dot += torch.sum(param1*param2)
        
        print("curr_diff:", i, param_diff)
        
#         diff += curr_diff*curr_diff
        
#     print(torch.sqrt(diff))
#     
#     print(all_dot/torch.sqrt(norm1*norm2))



    
